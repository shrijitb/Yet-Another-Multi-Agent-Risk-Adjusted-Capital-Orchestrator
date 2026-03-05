"""
core/hypervisor.py

The Kernel. The brain. The thing that runs everything.

Responsibilities:
  1. Every cycle: fetch market state, run risk checks, allocate capital.
  2. Never let a worker exceed its risk budget.
  3. Call emergency_hedge() if risk limits are breached.
  4. Log everything — you need to audit decisions after the fact.

The Hypervisor does NOT know about specific trading strategies.
It only knows: workers, capital, and risk limits.
Workers are pluggable — you can add a new strategy without touching this file.
"""

import os
import time
import logging
import threading
import signal
import sys
from typing import Dict, List

import config
from core.portfolio_state import PortfolioState, WorkerType
from core.risk_engine import RiskEngine
from core.audit_db import AuditDB
from workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class QuantHypervisor:
    """
    The Kernel.

    Lifecycle:
        __init__  → wire up workers and risk engine
        start()   → enter the main orchestration loop
        _cycle()  → one full evaluation: data → risk → allocate → execute
        stop()    → graceful shutdown (close all positions cleanly)
    """

    def __init__(
        self,
        workers: Dict[WorkerType, BaseWorker],
        initial_capital: float = config.INITIAL_CAPITAL_USD,
        paper_trading: bool    = config.PAPER_TRADING,
    ):
        self.state        = PortfolioState(initial_capital)
        self.risk         = RiskEngine()
        self.db           = AuditDB()
        self.workers      = workers
        self.paper_trading= paper_trading
        self._running     = False
        self._cycle_count = 0
        self._stop_event  = threading.Event()  # Wakes the inter-cycle sleep immediately on shutdown

        mode = "PAPER TRADING" if paper_trading else "LIVE TRADING - REAL MONEY"
        logger.info(f"QuantHypervisor initialized | Mode: {mode} | Capital: ${initial_capital:.2f}")
        if not paper_trading:
            logger.warning("LIVE MODE ACTIVE. Confirm this is intentional.")

    # ── Main Loop ──────────────────────────────────────────────────────────────

    def start(self):
        """Enter the main orchestration loop. Runs until stop() is called."""
        self._running = True
        logger.info("Hypervisor started.")

        while self._running:
            cycle_start = time.time()
            self._cycle_count += 1

            # --- KILLSWITCH: create a file named 'kill' in the hypervisor folder to stop gracefully ---
            if os.path.exists("kill"):
                logger.info("Kill file detected. Initiating graceful shutdown.")
                os.remove("kill")
                self.stop()
                return

            try:
                self._cycle()
            except Exception as e:
                # NEVER let a single cycle crash the whole process.
                # Log it, wait, and try again next cycle.
                logger.error(f"Cycle {self._cycle_count} failed with exception: {e}", exc_info=True)

            # Print state summary with simulated time label
            # Each cycle = 1 funding period = 8 simulated hours
            sim_hours = self._cycle_count * config.VAR_HORIZON_HOURS
            sim_days  = sim_hours / 24
            print(f"  [Sim time: {sim_days:.1f}d / {sim_hours}h elapsed]")
            print(self.state.summary())

            # Interruptible sleep — wakes immediately when stop() sets _stop_event
            elapsed    = time.time() - cycle_start
            sleep_for  = max(0, config.REBALANCE_INTERVAL_SEC - elapsed)
            logger.info(f"Cycle {self._cycle_count} complete in {elapsed:.1f}s. Sleeping {sleep_for:.0f}s.")
            self._stop_event.wait(timeout=sleep_for)

    def stop(self):
        """Graceful shutdown: wake sleep, close all open positions, log final state."""
        logger.info("Hypervisor shutting down -- closing all positions...")
        self._running = False
        self._stop_event.set()   # Wake the inter-cycle sleep immediately
        self._close_all_positions()
        print(self.state.summary())
        logger.info("Shutdown complete.")

    # ── Core Cycle ─────────────────────────────────────────────────────────────

    def _cycle(self):
        """
        One full Hypervisor cycle:

        Step 1: Collect market data from all workers.
        Step 2: Run VaR/CVaR on current exposure.
        Step 3: Gate on risk — if too hot, hedge and skip trading.
        Step 4: Calculate Sharpe for each worker.
        Step 5: Recommend and apply new capital allocation.
        Step 6: Tell eligible workers to execute.
        """
        logger.info(f"--- Cycle {self._cycle_count} ---")

        # Step 1: Collect return history from workers for risk calculation
        worker_returns = self._collect_worker_returns()

        # Step 2: Calculate VaR on total deployed capital
        all_returns = self._flatten_returns(worker_returns)
        deployed_capital = self.state.total_capital - self.state.free_capital
        if deployed_capital > 0 and len(all_returns) >= 10:
            var_usd, cvar_usd = self.risk.calculate_monte_carlo_var(
                position_size_usd  = deployed_capital,
                historical_returns = all_returns,
            )
            self.state.update_var(var_usd, cvar_usd)
        else:
            var_usd  = self.state.var_99
            cvar_usd = self.state.cvar_99

        # Step 3: Risk gate -- hard stop if VaR exceeds threshold
        safe, reason = self.risk.is_safe_to_trade(
            var_usd        = var_usd,
            cvar_usd       = cvar_usd,
            total_capital  = self.state.total_capital,
        )

        # Audit: log every cycle regardless of outcome
        self.db.log_cycle(
            cycle_num      = self._cycle_count,
            total_capital  = self.state.total_capital,
            free_capital   = self.state.free_capital,
            open_positions = len(self.state.positions),
            var_usd        = var_usd,
            cvar_usd       = cvar_usd,
            safe_to_trade  = safe,
            gate_reason    = reason,
            emergency_mode = self.state.emergency_mode,
        )

        if not safe:
            logger.warning(f"Risk gate tripped: {reason}")
            self._emergency_hedge()
            return  # Skip this cycle entirely

        self.state.emergency_mode = False

        # Step 4: Calculate Sharpe for each worker
        sharpe_scores = self.risk.calculate_sharpe_for_all_workers(worker_returns)

        # Audit: log risk metrics for each worker this cycle
        for worker_type, sharpe in sharpe_scores.items():
            metrics = self.risk.get_metrics(worker_type)
            if metrics:
                self.db.log_risk_metrics(
                    cycle_num = self._cycle_count,
                    worker    = worker_type.value,
                    metrics   = metrics,
                )

        # Update state with latest Sharpe scores
        for worker_type, sharpe in sharpe_scores.items():
            self.state.allocations[worker_type].sharpe_ratio = sharpe

        # Step 5: Recommend allocation
        recommended = self.risk.recommend_allocation(
            sharpe_scores      = sharpe_scores,
            total_free_capital = self.state.free_capital,
        )

        # Step 6: Execute workers.
        #
        # TWO cases handled separately:
        #
        # Case A — worker already has an open position (allocated_usd > 0):
        #   Call execute() with capital=0 so it runs its hold/collect/rotate logic.
        #   Do NOT gate this on recommended allocation — the worker must run every
        #   cycle to collect funding payments and monitor for rate reversals.
        #
        # Case B — worker has no position and allocator recommended new capital:
        #   Allocate the recommended amount and call execute() to open a position.

        executed_workers = set()

        # ── Case A ────────────────────────────────────────────────────────────
        # Tick workers that already hold an open position.
        # Gating on _current_pair (not allocated_usd) because allocated_usd
        # transiently drops to 0 mid-rotation while the position is still live.
        # Capital passed = allocated_usd when positive (normal hold/collect path),
        # falling back to _deployed_capital during the rotation window when
        # allocated_usd has already been zeroed by _exit_delta_neutral.
        for worker_type, worker in self.workers.items():
            if getattr(worker, '_current_pair', None) is None:
                continue  # No open position — leave for Case B

            alloc           = self.state.allocations.get(worker_type)
            capital_to_pass = (alloc.allocated_usd
                               if alloc and alloc.allocated_usd > 0
                               else getattr(worker, '_deployed_capital', 0.0))
            try:
                result = worker.execute(
                    capital         = capital_to_pass,
                    portfolio_state = self.state,
                    paper_trading   = self.paper_trading,
                )
                if result:
                    if result.get("action") == "backtest_complete":
                        logger.info("Backtest complete — all historical windows replayed.")
                        self.stop()
                        return
                    logger.info(f"{worker_type.value} executed: {result}")
                executed_workers.add(worker_type)
            except Exception as e:
                logger.error(f"Worker {worker_type.value} tick failed: {e}", exc_info=True)

        # ── Case B ────────────────────────────────────────────────────────────
        # Open new positions for workers with no current exposure.
        # Allocate the recommended amount, call execute(), then immediately
        # reclaim capital if execute() found no eligible opportunity this cycle.
        # This reclaim is what allows the backtest queue to drain: each cycle
        # Case B consumes one historical window and either opens a position
        # (capital stays allocated, Case A takes over) or returns capital so
        # the next cycle can try again with the next window.
        for worker_type, amount in recommended.items():
            if worker_type in executed_workers:
                continue  # Already ticked by Case A above

            worker = self.workers.get(worker_type)
            if worker is None:
                continue

            allocated = self.state.allocate_to_worker(worker_type, amount)
            if not allocated:
                continue

            self.db.log_allocation(
                cycle_num           = self._cycle_count,
                worker              = worker_type.value,
                direction           = "ALLOCATE",
                amount_usd          = amount,
                free_capital_after  = self.state.free_capital,
                total_capital_after = self.state.total_capital,
            )

            result = None
            try:
                result = worker.execute(
                    capital         = amount,
                    portfolio_state = self.state,
                    paper_trading   = self.paper_trading,
                )
            except Exception as e:
                logger.error(f"Worker {worker_type.value} execution failed: {e}", exc_info=True)

            if result is None:
                # No trade placed — return capital immediately so the next cycle
                # has a full free pool and can attempt the subsequent window.
                self.state.deallocate_from_worker(worker_type, amount, pnl=0.0)
                logger.debug(f"{worker_type.value}: no opportunity this window, capital returned")
            elif result.get("action") == "backtest_complete":
                # Queue exhausted inside Case B — reclaim the pre-allocated
                # capital (no position was opened) then trigger clean shutdown.
                self.state.deallocate_from_worker(worker_type, amount, pnl=0.0)
                logger.info("Backtest complete — all historical windows replayed.")
                self.stop()
                return
            else:
                logger.info(f"{worker_type.value} executed: {result}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _collect_worker_returns(self) -> Dict[WorkerType, List[float]]:
        """Ask each worker for its historical return series."""
        result = {}
        for worker_type, worker in self.workers.items():
            try:
                returns = worker.get_return_history()
                result[worker_type] = returns
            except Exception as e:
                logger.error(f"Failed to get return history from {worker_type.value}: {e}")
                result[worker_type] = []
        return result

    def _flatten_returns(self, worker_returns: Dict[WorkerType, List[float]]) -> List[float]:
        """Merge all worker return series into one portfolio-level series."""
        all_returns = []
        for returns in worker_returns.values():
            all_returns.extend(returns)
        return all_returns

    def _emergency_hedge(self):
        """
        Called when VaR/CVaR limits are breached.
        Strategy: close everything, return to cash, set emergency flag.
        In a full implementation this would also open hedge positions.
        """
        logger.warning("EMERGENCY HEDGE TRIGGERED — closing all positions")
        self.state.emergency_mode = True
        self._close_all_positions()

    def _close_all_positions(self):
        """Close every open position. Used on shutdown and emergency hedge."""
        position_keys = list(self.state.positions.keys())
        for key in position_keys:
            pos = self.state.positions.get(key)
            if pos is None:
                continue
            worker = self.workers.get(pos.worker)
            if worker:
                try:
                    worker.close_position(key, self.state, self.paper_trading)
                except Exception as e:
                    logger.error(f"Failed to close position {key}: {e}")
