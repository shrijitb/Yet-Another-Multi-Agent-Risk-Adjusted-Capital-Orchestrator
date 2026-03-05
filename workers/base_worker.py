"""
workers/base_worker.py

The contract every strategy worker must satisfy.
The Hypervisor only knows this interface — not the implementation details.
This is what lets you add new strategies without touching core code.

To add a new strategy:
    1. Create a new file in workers/
    2. Subclass BaseWorker
    3. Implement all abstract methods
    4. Register it in main.py
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class BaseWorker(ABC):
    """
    Abstract base class for all strategy workers.

    Workers are stateful — they track their own return history,
    open positions, and execution logic. The Hypervisor only calls
    execute() and get_return_history().
    """

    def __init__(self, name: str):
        self.name           = name
        self._return_history: List[float] = []   # Hourly returns as decimals
        self._is_active     = False

    # ── Required by all workers ────────────────────────────────────────────────

    @abstractmethod
    def execute(
        self,
        capital: float,
        portfolio_state,          # PortfolioState (avoid circular import with string)
        paper_trading: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Main entry point. Called by Hypervisor each cycle with allocated capital.

        Args:
            capital:         Dollar amount allocated to this worker this cycle.
            portfolio_state: Reference to shared PortfolioState for position tracking.
            paper_trading:   If True, simulate fills. If False, use live broker.

        Returns:
            Dict with execution summary (ticker, size, side, entry_price, etc.)
            or None if no trade was placed this cycle.
        """
        pass

    @abstractmethod
    def close_position(
        self,
        position_key: str,
        portfolio_state,
        paper_trading: bool,
    ) -> Optional[float]:
        """
        Close a specific position by its key.
        Returns net P&L of the closed position or None on failure.
        """
        pass

    @abstractmethod
    def get_market_data(self) -> Dict[str, Any]:
        """
        Fetch whatever market data this strategy needs.
        Called independently from execute() so the Hypervisor
        can inspect data without triggering a trade.
        """
        pass

    # ── Shared utility methods ─────────────────────────────────────────────────

    def get_return_history(self) -> List[float]:
        """Return this worker's hourly return series for Sharpe/VaR calculation."""
        return list(self._return_history)

    def record_return(self, return_pct: float):
        """Log a realized hourly return. Called after each funding period closes."""
        self._return_history.append(return_pct)
        if len(self._return_history) > 720:   # 30 days of hourly data
            self._return_history.pop(0)

    def simulate_return_history(self, n_hours: int = 168, seed_sharpe: float = 2.0):
        """
        Generate synthetic return history for testing/paper mode.
        Produces returns consistent with a given Sharpe ratio.
        Used during development before real return data accumulates.
        """
        import numpy as np
        hours_per_year = 8_760
        risk_free_hourly = 0.045 / hours_per_year
        target_mean = (seed_sharpe * 0.02 / np.sqrt(hours_per_year)) + risk_free_hourly
        target_std  = 0.02 / np.sqrt(hours_per_year)
        synthetic = np.random.normal(target_mean, target_std, n_hours).tolist()
        self._return_history = synthetic

    def __repr__(self) -> str:
        return f"<Worker: {self.name} | {len(self._return_history)} return records>"
