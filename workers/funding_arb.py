"""
workers/funding_arb.py

The Delta-Neutral Funding Rate Worker.

Strategy:
  - Scan all perpetual futures pairs for positive funding rates
  - For the best opportunity above minimum threshold:
      BUY  $X of the asset on spot (LONG)
      SELL $X of the same asset on perps (SHORT)
  - Net price exposure = 0 (delta neutral)
  - Collect funding rate every 8 hours from the leveraged longs paying you

Risk profile:
  - No directional market risk (price moves cancel out)
  - Exchange risk (if exchange fails, both legs fail)
  - Liquidation risk on futures leg during violent price spikes
  - Funding rate reversal risk (rate flips negative, you pay instead)

The worker monitors for reversals and exits if rate turns negative.
"""

import time
import random
import logging
from typing import Dict, Optional, Any, List

import config
from workers.base_worker import BaseWorker
from core.portfolio_state import PortfolioState, Position, PositionSide, WorkerType

logger = logging.getLogger(__name__)


class FundingArbWorker(BaseWorker):
    """
    Delta-neutral funding rate arbitrageur.

    In paper mode: uses simulated funding rates and fills.
    In live mode: connects to ccxt exchange APIs (stubbed — you wire credentials in).
    """

    def __init__(self):
        super().__init__(name="FundingArbWorker")
        self._current_pair:       str | None = None
        self._current_exchange:   str | None = None
        self._funding_earned_usd: float      = 0.0
        self._last_funding_check: float      = 0.0
        self._position_opened_at: float      = 0.0
        self._stable_rates:       dict       = {}
        self._rates_generated_at: float      = 0.0
        self._min_hold_seconds:   float      = 80.0
        self._deployed_capital:   float      = 0.0
        self._hold_cycles:        int        = 0

        # Backtest replay queue — list of {pair: rate} snapshots in time order.
        # When set, get_market_data() pops from the front instead of simulating.
        self._backtest_queue:     list       = []
        self._backtest_mode:      bool       = False

    def load_backtest_data(self, snapshots: list) -> None:
        """
        Load historical funding rate snapshots for replay.
        Called once at startup by main.py in --backtest mode.
        Each cycle will consume one snapshot (one 8h funding window).
        """
        self._backtest_queue = list(snapshots)
        self._backtest_mode  = True
        self._min_hold_seconds = 0.0  # Each cycle is already one full 8h window
        logger.info(f"Backtest queue loaded: {len(self._backtest_queue)} funding windows")

    # ── BaseWorker Interface ───────────────────────────────────────────────────

    def execute(
        self,
        capital: float,
        portfolio_state: PortfolioState,
        paper_trading: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Main cycle:
          1. Scan funding rates.
          2. If already in a position, check if rate is still positive.
             If not, exit and look for better opportunity.
          3. If not in a position, enter best available pair.
        """
        # Step 1: Get current rates
        rates = self.get_market_data()
        if not rates:
            if self._backtest_mode:
                return {"action": "backtest_complete"}
            logger.warning("FundingArbWorker: no rates available, skipping cycle")
            return None

        # Filter to positive rates above minimum
        eligible = {k: v for k, v in rates.items() if v >= config.MIN_FUNDING_RATE}
        if not eligible:
            logger.info("FundingArbWorker: no eligible funding rates this cycle")
            return None

        best_pair = max(eligible, key=eligible.get)
        best_rate = eligible[best_pair]

        # Step 2: Check if we should exit current position
        if self._current_pair is not None:
            current_rate     = rates.get(self._current_pair, -1)
            hold_seconds     = time.time() - self._position_opened_at
            # In backtest mode use cycle count — wall-clock is ~0s between cycles
            held_long_enough = (self._hold_cycles >= 1) if self._backtest_mode \
                               else (hold_seconds >= self._min_hold_seconds)

            if current_rate < 0:
                # Rate flipped negative — exit immediately regardless of hold time
                logger.info(
                    f"Funding rate on {self._current_pair} went negative "
                    f"({current_rate:.6f}). Exiting position."
                )
                self._exit_delta_neutral(portfolio_state, paper_trading)

            elif held_long_enough and best_pair != self._current_pair and best_rate > current_rate * 1.5:
                # Only rotate if: held minimum time AND significantly better rate exists
                logger.info(
                    f"Rotating from {self._current_pair} ({current_rate:.5f}) "
                    f"to {best_pair} ({best_rate:.5f}) after {hold_seconds:.0f}s"
                )
                self._exit_delta_neutral(portfolio_state, paper_trading)

            else:
                # Stay in current position — collect funding using actual deployed size
                self._hold_cycles += 1
                collected = self._collect_funding(portfolio_state, current_rate, self._deployed_capital)
                return {"action": "hold", "pair": self._current_pair,
                        "rate": current_rate, "funding_collected_usd": collected,
                        "hold_cycles": self._hold_cycles}

        # Step 3: Open new delta-neutral position
        leg_size = capital / 2  # Split evenly: half spot, half futures short
        result   = self._enter_delta_neutral(best_pair, leg_size, portfolio_state, paper_trading)

        if result:
            return {
                "action":    "opened",
                "pair":      best_pair,
                "rate":      best_rate,
                "leg_size":  leg_size,
                "annual_yield_pct": best_rate * 3 * 365 * 100,  # 3 resets/day * 365
            }
        return None

    def close_position(
        self,
        position_key: str,
        portfolio_state: PortfolioState,
        paper_trading: bool,
    ) -> Optional[float]:
        """Called by Hypervisor on emergency hedge or shutdown."""
        pos = portfolio_state.positions.get(position_key)
        if pos is None:
            return None

        exit_price = self._get_simulated_exit_price(pos.entry_price) if paper_trading else self._get_live_price(pos.ticker)
        pnl = portfolio_state.close_position(position_key, exit_price)
        if pnl is not None:
            portfolio_state.deallocate_from_worker(WorkerType.FUNDING_ARB, pos.size_usd, pnl)
        return pnl

    def get_market_data(self) -> Dict[str, float]:
        """
        Returns the funding rate snapshot for this cycle.

        Backtest mode : pops the next historical snapshot from the queue.
                        When queue is exhausted, returns {} to stop trading.
        Live rates    : fetches from Binance public API (USE_LIVE_RATES=True).
        Simulation    : generates random rates (default).
        """
        if self._backtest_mode:
            if not self._backtest_queue:
                logger.info("Backtest queue exhausted -- all historical windows replayed.")
                return {}
            snapshot = self._backtest_queue.pop(0)
            remaining = len(self._backtest_queue)
            logger.info(
                f"Backtest window: {len(snapshot)} pairs | "
                f"{remaining} windows remaining"
            )
            return snapshot
        if getattr(config, 'USE_LIVE_RATES', False):
            # Use OKX -- works from US IPs and Raspberry Pi (no VPN needed)
            from workers.data_loader import fetch_live_funding_rates_okx
            return fetch_live_funding_rates_okx()
        return self._simulate_funding_rates()

    # ── Strategy Logic ─────────────────────────────────────────────────────────

    def _enter_delta_neutral(
        self,
        ticker: str,
        leg_size_usd: float,
        portfolio_state: PortfolioState,
        paper_trading: bool,
    ) -> bool:
        """Open simultaneous spot LONG + futures SHORT on the same ticker."""
        entry_price = self._get_entry_price(ticker, paper_trading)

        # Apply slippage model
        spot_price    = entry_price * (1 + config.SLIPPAGE_MODEL_PCT)
        futures_price = entry_price * (1 - config.SLIPPAGE_MODEL_PCT)

        # Open spot LONG
        spot_position = Position(
            ticker      = ticker,
            side        = PositionSide.LONG,
            size_usd    = leg_size_usd,
            entry_price = spot_price,
            exchange    = "binance_spot",
            worker      = WorkerType.FUNDING_ARB,
        )
        # Open futures SHORT
        futures_position = Position(
            ticker      = ticker,
            side        = PositionSide.SHORT,
            size_usd    = leg_size_usd,
            entry_price = futures_price,
            exchange    = "binance_futures",
            worker      = WorkerType.FUNDING_ARB,
        )

        portfolio_state.open_position(spot_position)
        portfolio_state.open_position(futures_position)

        # Ensure allocation counter reflects the deployed capital.
        # During a rotation _exit_delta_neutral deallocated both legs, so
        # allocated_usd is now 0. Re-allocate so the hypervisor's books balance.
        alloc = portfolio_state.allocations.get(WorkerType.FUNDING_ARB)
        if alloc is not None and alloc.allocated_usd < leg_size_usd * 2:
            portfolio_state.allocate_to_worker(WorkerType.FUNDING_ARB, leg_size_usd * 2 - alloc.allocated_usd)

        self._current_pair       = ticker
        self._current_exchange   = "binance"
        self._last_funding_check = time.time()
        self._position_opened_at = time.time()
        self._deployed_capital   = leg_size_usd * 2  # Both legs combined
        self._hold_cycles        = 0                  # Cycles held (used instead of wall-clock in backtest)

        logger.info(
            f"Opened delta-neutral on {ticker} | "
            f"Spot LONG ${leg_size_usd:.2f} @ {spot_price:.4f} | "
            f"Futures SHORT ${leg_size_usd:.2f} @ {futures_price:.4f}"
        )
        return True

    def _exit_delta_neutral(self, portfolio_state: PortfolioState, paper_trading: bool):
        """Close both legs of the current delta-neutral position."""
        if self._current_pair is None:
            return

        exit_price = self._get_exit_price(self._current_pair, paper_trading)
        positions_to_close = [
            k for k in portfolio_state.positions
            if self._current_pair in k
        ]
        for key in positions_to_close:
            pos = portfolio_state.positions.get(key)
            if pos:
                pnl = portfolio_state.close_position(key, exit_price)
                if pnl is not None:
                    portfolio_state.deallocate_from_worker(
                        WorkerType.FUNDING_ARB, pos.size_usd, pnl
                    )

        self._current_pair     = None
        self._current_exchange = None
        self._deployed_capital = 0.0
        self._hold_cycles      = 0

    def _collect_funding(
        self,
        portfolio_state: PortfolioState,
        rate: float,
        capital: float,
    ) -> float:
        """
        Simulate collecting the funding fee payment.
        In live mode this happens automatically on exchange — we just track it here.

        Each call to _collect_funding represents one simulated 8-hour funding window.
        We do NOT use real wall-clock time because in fast/test mode cycles are only
        10 seconds apart — waiting for 8 real hours means collecting nothing forever.
        Instead: one hold cycle = one funding period collected.
        """
        collected = capital * rate
        self._funding_earned_usd += collected

        # Credit directly to the worker's P&L in portfolio state
        portfolio_state.record_funding_payment(WorkerType.FUNDING_ARB, collected)

        # Record as a percentage return for Sharpe/Sortino tracking
        return_pct = collected / capital if capital > 0 else 0.0
        self.record_return(return_pct)
        portfolio_state.record_hourly_return(return_pct)

        logger.info(f"Funding collected: ${collected:.4f} (rate={rate*100:.4f}%  capital=${capital:.2f})")
        return collected

    # ── Price Feeds ────────────────────────────────────────────────────────────

    def _get_entry_price(self, ticker: str, paper_trading: bool) -> float:
        if paper_trading:
            return self._simulate_price(ticker)
        return self._get_live_price(ticker)

    def _get_exit_price(self, ticker: str, paper_trading: bool) -> float:
        entry = self._get_entry_price(ticker, paper_trading)
        return self._get_simulated_exit_price(entry) if paper_trading else self._get_live_price(ticker)

    def _get_simulated_exit_price(self, entry_price: float) -> float:
        """Simulate realistic exit price with small random drift."""
        drift = random.gauss(0, 0.002)  # ±0.2% typical short-hold drift
        return entry_price * (1 + drift)

    def _simulate_price(self, ticker: str) -> float:
        """Return a plausible fake price for paper trading."""
        prices = {
            "BTC/USDT": 65000.0,
            "ETH/USDT": 3500.0,
            "SOL/USDT": 150.0,
            "BNB/USDT": 580.0,
            "ARB/USDT": 1.20,
        }
        base = prices.get(ticker, 100.0)
        return base * (1 + random.gauss(0, 0.001))

    def _simulate_funding_rates(self) -> Dict[str, float]:
        """
        Generate simulated funding rates.

        KEY FIX: Rates are cached and only regenerated every 80 seconds
        (simulating the 8-hour funding window). This prevents the system from
        seeing a completely new rate landscape every cycle, which caused
        constant rotation and fee bleed in the original version.

        In real trading, funding rates are announced 1 hour before settlement
        and change at most 3 times per day.
        """
        now = time.time()
        cache_ttl = 80.0  # Refresh every 80s in simulation (= 8hr window in real life)

        if self._stable_rates and (now - self._rates_generated_at) < cache_ttl:
            return self._stable_rates  # Return cached rates — don't re-randomize

        # Generate fresh rates (only happens every 80s)
        pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ARB/USDT",
                 "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "OP/USDT", "SUI/USDT"]
        rates = {}
        for pair in pairs:
            roll = random.random()
            if roll < 0.10:
                rate = random.uniform(-0.0005, 0)       # Negative — you'd pay
            elif roll < 0.30:
                rate = random.uniform(0, 0.0001)        # Too small to bother
            elif roll < 0.85:
                rate = random.uniform(0.0001, 0.0005)   # Normal positive
            else:
                rate = random.uniform(0.0005, 0.002)    # Juicy spike (bull run)
            rates[pair] = round(rate, 6)

        self._stable_rates       = rates
        self._rates_generated_at = now
        logger.info(f"Funding rates refreshed. Best: {max(rates, key=rates.get)} @ {max(rates.values()):.5f}")
        return rates

    # ── Live Exchange Stubs ────────────────────────────────────────────────────
    # Wire these up when you're ready for live trading.
    # Requires: pip install ccxt + your API keys in environment variables.

    def _fetch_live_funding_rates(self) -> Dict[str, float]:
        """
        DEPRECATED: Binance/Bybit block US IPs -- use OKX instead.
        This method is no longer called. Live rates go through data_loader.fetch_live_funding_rates_okx().
        Kept for reference only.
        """
        logger.warning("_fetch_live_funding_rates called directly -- use OKX path via get_market_data()")
        return self._simulate_funding_rates()

    def _get_live_price(self, ticker: str) -> float:
        """
        STUB: Replace with real ccxt price fetch.

        Example implementation:
            import ccxt
            exchange = ccxt.binance()
            ticker_data = exchange.fetch_ticker(ticker)
            return ticker_data['last']
        """
        raise NotImplementedError("Live price fetch not yet implemented.")
