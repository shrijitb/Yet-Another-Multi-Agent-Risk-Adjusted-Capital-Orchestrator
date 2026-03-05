"""
core/portfolio_state.py

The Hypervisor's memory. Every position, every dollar, every P&L lives here.
This is the single source of truth — workers READ from it, the Hypervisor WRITES to it.
Nothing touches capital directly. Everything goes through here.
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
from enum import Enum

import config

logger = logging.getLogger(__name__)


class WorkerType(str, Enum):
    FUNDING_ARB  = "funding_arb"
    SWING_TREND  = "swing_trend"
    IDLE         = "idle"          # Capital sitting uninvested


class PositionSide(str, Enum):
    LONG  = "long"
    SHORT = "short"


@dataclass
class Position:
    """A single open position (one leg of a trade)."""
    ticker:         str
    side:           PositionSide
    size_usd:       float
    entry_price:    float
    exchange:       str
    worker:         WorkerType
    opened_at:      float = field(default_factory=time.time)  # Unix timestamp
    unrealized_pnl: float = 0.0

    def age_hours(self) -> float:
        return (time.time() - self.opened_at) / 3600


@dataclass
class WorkerAllocation:
    """How much capital is assigned to each worker and its performance stats."""
    worker:         WorkerType
    allocated_usd:  float = 0.0
    realized_pnl:   float = 0.0
    trade_count:    int   = 0
    sharpe_ratio:   float = 0.0
    last_updated:   float = field(default_factory=time.time)

    def total_return_pct(self) -> float:
        if self.allocated_usd == 0:
            return 0.0
        return (self.realized_pnl / self.allocated_usd) * 100


class PortfolioState:
    """
    The kernel's memory bank. Thread-safe record of all capital and positions.

    Design rules:
      1. Workers never modify state directly — they request changes via the Hypervisor.
      2. Every mutation is logged.
      3. State is snapshotted to disk after every change (crash recovery).
    """

    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL_USD):
        self.total_capital:   float = initial_capital
        self.free_capital:    float = initial_capital    # Not deployed anywhere
        self.positions:       Dict[str, Position] = {}  # key = f"{ticker}_{side}"
        self.allocations:     Dict[WorkerType, WorkerAllocation] = {
            w: WorkerAllocation(worker=w) for w in WorkerType
        }
        self.var_99:          float = 0.0   # Last calculated VaR
        self.cvar_99:         float = 0.0   # Last calculated CVaR (Expected Shortfall)
        self.emergency_mode:  bool  = False # True = Hypervisor has halted trading
        self.created_at:      float = time.time()
        self._return_history: list  = []    # Rolling list of hourly returns (for Sharpe)

        logger.info(f"PortfolioState initialized with ${initial_capital:.2f}")

    # ── Capital Management ─────────────────────────────────────────────────────

    def allocate_to_worker(self, worker: WorkerType, amount_usd: float) -> bool:
        """
        Move capital from free pool → worker allocation.
        Returns False (and does nothing) if insufficient free capital.
        """
        if amount_usd > self.free_capital:
            logger.warning(
                f"Allocation rejected: requested ${amount_usd:.2f} "
                f"but only ${self.free_capital:.2f} free"
            )
            return False

        self.free_capital -= amount_usd
        self.allocations[worker].allocated_usd += amount_usd
        self.allocations[worker].last_updated = time.time()
        logger.info(f"Allocated ${amount_usd:.2f} to {worker.value}")
        self._snapshot()
        return True

    def deallocate_from_worker(self, worker: WorkerType, amount_usd: float, pnl: float = 0.0):
        """
        Return capital from worker → free pool, recording any P&L.
        """
        self.free_capital += amount_usd + pnl
        self.total_capital += pnl
        self.allocations[worker].allocated_usd -= amount_usd
        self.allocations[worker].realized_pnl  += pnl
        self.allocations[worker].last_updated  = time.time()

        net_direction = "profit" if pnl >= 0 else "loss"
        logger.info(
            f"Deallocated ${amount_usd:.2f} from {worker.value} "
            f"with {net_direction} of ${abs(pnl):.4f}"
        )
        self._snapshot()

    # ── Position Management ────────────────────────────────────────────────────

    def open_position(self, position: Position) -> str:
        key = f"{position.ticker}_{position.side.value}_{position.exchange}"
        if key in self.positions:
            logger.warning(f"Position {key} already exists - skipping duplicate open")
            return key
        self.positions[key] = position
        self.allocations[position.worker].trade_count += 1
        logger.info(
            f"Opened {position.side.value.upper()} {position.ticker} "
            f"${position.size_usd:.2f} on {position.exchange}"
        )
        self._snapshot()
        return key

    def close_position(self, key: str, exit_price: float) -> Optional[float]:
        """
        Close a position. Returns realized P&L or None if position not found.
        """
        if key not in self.positions:
            logger.error(f"Cannot close unknown position: {key}")
            return None

        pos = self.positions[key]
        if pos.side == PositionSide.LONG:
            pnl = pos.size_usd * ((exit_price - pos.entry_price) / pos.entry_price)
        else:  # SHORT
            pnl = pos.size_usd * ((pos.entry_price - exit_price) / pos.entry_price)

        # Apply fee model
        fee = pos.size_usd * config.FEE_MODEL_PCT * 2  # Entry + exit
        net_pnl = pnl - fee

        del self.positions[key]
        logger.info(
            f"Closed {key} | gross P&L: ${pnl:.4f} | fees: ${fee:.4f} | net: ${net_pnl:.4f}"
        )
        self._snapshot()
        return net_pnl

    def update_var(self, var_99: float, cvar_99: float):
        self.var_99  = var_99
        self.cvar_99 = cvar_99

    def record_hourly_return(self, return_pct: float):
        """Feed hourly returns into rolling history for Sharpe calculation."""
        self._return_history.append(return_pct)
        if len(self._return_history) > 720:  # Keep last 30 days (720 hours)
            self._return_history.pop(0)

    def record_funding_payment(self, worker: WorkerType, amount_usd: float):
        """
        Credit a funding fee payment directly to the worker's P&L and total capital.
        This does NOT move allocated capital — the position stays open.
        It simply adds the earned funding to realized_pnl and total_capital.
        """
        self.total_capital                        += amount_usd
        self.free_capital                         += amount_usd
        self.allocations[worker].realized_pnl     += amount_usd
        self.allocations[worker].last_updated      = time.time()
        logger.info(f"Funding payment credited: +${amount_usd:.4f} to {worker.value}")

    def get_return_history(self) -> list:
        return list(self._return_history)

    # ── Reporting ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"{'='*50}",
            f"  PORTFOLIO STATE",
            f"{'='*50}",
            f"  Total Capital : ${self.total_capital:.2f}",
            f"  Free Capital  : ${self.free_capital:.2f}",
            f"  Open Positions: {len(self.positions)}",
            f"  VaR (99%)     : ${self.var_99:.2f}",
            f"  CVaR (99%)    : ${self.cvar_99:.2f}",
            f"  Emergency Mode: {self.emergency_mode}",
            f"{'-'*50}",
        ]
        for worker, alloc in self.allocations.items():
            if worker == WorkerType.IDLE:
                continue
            lines.append(
                f"  {worker.value:<15} | "
                f"Allocated: ${alloc.allocated_usd:>7.2f} | "
                f"P&L: ${alloc.realized_pnl:>+8.4f} | "
                f"Sharpe: {alloc.sharpe_ratio:.2f}"
            )
        lines.append(f"{'='*50}")
        return "\n".join(lines)

    # ── Persistence ────────────────────────────────────────────────────────────

    def _snapshot(self):
        """Write state to disk after every mutation. Crash recovery."""
        try:
            data = {
                "total_capital":  self.total_capital,
                "free_capital":   self.free_capital,
                "emergency_mode": self.emergency_mode,
                "var_99":         self.var_99,
                "cvar_99":        self.cvar_99,
                "positions": {
                    k: asdict(v) for k, v in self.positions.items()
                },
                "allocations": {
                    k.value: asdict(v) for k, v in self.allocations.items()
                },
                "snapshot_at": time.time(),
            }
            with open(config.STATE_SNAPSHOT_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Never let a snapshot failure crash the trading loop
            logger.error(f"Snapshot failed (non-fatal): {e}")
