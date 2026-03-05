"""
core/risk_engine.py  --  Phase 2 upgrade

The Hedge Fund Layer. This is the gatekeeper.
Nothing gets traded without passing through here first.

Phase 2 additions vs Phase 1:
  - Sortino Ratio: only penalizes DOWNSIDE volatility (better for war-time markets)
  - CVaR at 95% (more conservative trigger -- fires earlier than 99%)
  - Downside Deviation: the denominator for Sortino
  - Beta Sensitivity: measures accidental correlation to systemic crash index
  - Unified RiskMetrics dict returned per worker for SQLite audit logging
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

import config
from core.portfolio_state import WorkerType

logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Container for all risk metrics for one worker in one cycle.
    Passed directly to the SQLite audit logger.
    """
    def __init__(self):
        self.sharpe:            float = 0.0
        self.sortino:           float = 0.0
        self.var_usd:           float = 0.0
        self.cvar_usd:          float = 0.0
        self.downside_dev:      float = 0.0
        self.beta:              float = 0.0
        self.mean_return:       float = 0.0
        self.volatility:        float = 0.0
        self.max_drawdown_pct:  float = 0.0

    def to_dict(self) -> dict:
        return {
            "sharpe":           round(self.sharpe, 6),
            "sortino":          round(self.sortino, 6),
            "var_usd":          round(self.var_usd, 4),
            "cvar_usd":         round(self.cvar_usd, 4),
            "downside_dev":     round(self.downside_dev, 8),
            "beta":             round(self.beta, 6),
            "mean_return":      round(self.mean_return, 8),
            "volatility":       round(self.volatility, 8),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
        }

    def __repr__(self):
        return (
            f"RiskMetrics(sharpe={self.sharpe:.3f}, sortino={self.sortino:.3f}, "
            f"var=${self.var_usd:.2f}, cvar=${self.cvar_usd:.2f}, beta={self.beta:.3f})"
        )


class RiskEngine:
    """
    Quantitative risk layer. Stateless -- all inputs passed in explicitly.
    Same inputs always produce same outputs. Fully testable.

    Key concepts:
      Sharpe:   (Return - RiskFree) / Total Volatility
                Penalizes ALL volatility equally -- upside and downside.

      Sortino:  (Return - RiskFree) / Downside Deviation
                Only penalizes LOSSES. Better for asymmetric strategies like
                funding arb (which has frequent small gains, rare large losses).
                In a manipulated market, upside vol is not your enemy -- downside is.

      VaR:      "With 99% confidence, max loss over N hours is $X"
      CVaR:     "When we DO land in that worst tail, average loss is $Y"
                CVaR is always >= VaR. It's what you actually face.

      Beta:     How correlated is this strategy to the broader market?
                Beta = 1.0 means it moves with S&P 500.
                Beta = 0.0 means it's uncorrelated (ideal for funding arb).
                Beta < 0 means it profits when markets crash (hedge).
    """

    def __init__(self):
        self.last_metrics: Dict[WorkerType, RiskMetrics] = {}
        logger.info("RiskEngine (Phase 2) initialized")

    # -- Full Metrics Suite ----------------------------------------------------

    def calculate_all_metrics(
        self,
        returns: List[float],
        position_size_usd: float,
        benchmark_returns: Optional[List[float]] = None,
    ) -> RiskMetrics:
        """
        Run the full risk suite on a return series. Returns a RiskMetrics object.
        """
        m = RiskMetrics()

        if len(returns) < 10:
            logger.warning("Insufficient return history (<10 points) - metrics unreliable")
            return m

        r = np.array(returns, dtype=np.float64)
        hours_per_year = 8_760
        rf_hourly      = config.SHARPE_RISK_FREE_RATE / hours_per_year

        m.mean_return = float(np.mean(r))
        m.volatility  = float(np.std(r))

        # Sharpe
        if m.volatility > 0:
            m.sharpe = float(
                ((m.mean_return - rf_hourly) / m.volatility) * np.sqrt(hours_per_year)
            )
        else:
            m.sharpe = 5.0 if m.mean_return > rf_hourly else 0.0

        # Sortino
        m.downside_dev = self._downside_deviation(r, rf_hourly)
        if m.downside_dev > 1e-10:
            m.sortino = float(
                ((m.mean_return - rf_hourly) / m.downside_dev) * np.sqrt(hours_per_year)
            )
        else:
            m.sortino = 5.0 if m.mean_return > rf_hourly else 0.0

        # Monte Carlo VaR + CVaR
        m.var_usd, m.cvar_usd = self.calculate_monte_carlo_var(
            position_size_usd  = position_size_usd,
            historical_returns = returns,
        )

        # Max Drawdown
        m.max_drawdown_pct = self._max_drawdown(r)

        # Beta vs benchmark
        if benchmark_returns and len(benchmark_returns) >= len(returns):
            m.beta = self._calculate_beta(r, np.array(benchmark_returns[-len(r):]))

        logger.info(
            f"RiskMetrics: sharpe={m.sharpe:.3f} sortino={m.sortino:.3f} "
            f"var=${m.var_usd:.2f} cvar=${m.cvar_usd:.2f} "
            f"downside_dev={m.downside_dev:.6f} beta={m.beta:.3f}"
        )
        return m

    # -- Monte Carlo VaR + CVaR ------------------------------------------------

    def calculate_monte_carlo_var(
        self,
        position_size_usd:  float,
        historical_returns: List[float],
        horizon_hours:      int   = config.VAR_HORIZON_HOURS,
        n_simulations:      int   = config.VAR_SIMULATIONS,
        confidence:         float = config.VAR_CONFIDENCE,
    ) -> Tuple[float, float]:
        """
        Monte Carlo simulation for VaR and CVaR.
        Returns (var_usd, cvar_usd) as positive dollar loss amounts.
        """
        if len(historical_returns) < 10:
            fallback = position_size_usd * 0.10
            return fallback, fallback * config.CVAR_MULTIPLIER

        r     = np.array(historical_returns, dtype=np.float64)
        mu    = np.mean(r)
        sigma = np.std(r)

        simulated    = np.random.normal(mu, sigma, (n_simulations, horizon_hours))
        path_returns = np.prod(1 + simulated, axis=1) - 1
        path_pnl     = path_returns * position_size_usd
        sorted_pnl   = np.sort(path_pnl)

        var_idx  = int((1 - confidence) * n_simulations)
        var_usd  = float(abs(sorted_pnl[var_idx]))
        cvar_usd = float(abs(np.mean(sorted_pnl[:var_idx])))

        logger.info(
            f"Monte Carlo VaR({confidence*100:.0f}%): ${var_usd:.2f} | "
            f"CVaR: ${cvar_usd:.2f} | "
            f"mu={mu:.5f} sigma={sigma:.5f} | "
            f"{n_simulations:,} paths over {horizon_hours}h"
        )
        return var_usd, cvar_usd

    # -- Per-worker metrics pass -----------------------------------------------

    def calculate_sharpe_for_all_workers(
        self,
        worker_returns: Dict[WorkerType, List[float]],
        position_sizes: Optional[Dict[WorkerType, float]] = None,
    ) -> Dict[WorkerType, float]:
        """
        Backward-compatible shim used by hypervisor._cycle().
        Now computes full RiskMetrics internally and caches them.
        Returns {WorkerType: sharpe_float} to keep hypervisor.py unchanged.
        """
        sharpe_map = {}
        for worker, returns in worker_returns.items():
            size    = (position_sizes or {}).get(worker, 0.0)
            metrics = self.calculate_all_metrics(returns, size)
            self.last_metrics[worker] = metrics
            sharpe_map[worker] = metrics.sharpe
            logger.info(f"  [{worker.value}] {metrics}")
        return sharpe_map

    def get_sortino(self, worker: WorkerType) -> float:
        m = self.last_metrics.get(worker)
        return m.sortino if m else 0.0

    def get_metrics(self, worker: WorkerType) -> Optional[RiskMetrics]:
        return self.last_metrics.get(worker)

    # -- Capital Allocation ----------------------------------------------------

    def recommend_allocation(
        self,
        sharpe_scores: Dict[WorkerType, float],
        total_free_capital: float,
    ) -> Dict[WorkerType, float]:
        """
        Phase 2: uses Sortino from cached metrics as primary allocation signal.
        Falls back to Sharpe if Sortino not yet available.
        Workers with high Beta (>0.8) to market are penalized.
        """
        eligible = {}
        for worker, sharpe in sharpe_scores.items():
            m = self.last_metrics.get(worker)
            score = m.sortino if (m and m.sortino > 0) else sharpe

            if score < config.MIN_SHARPE_TO_TRADE:
                logger.info(
                    f"Excluding {worker.value}: score={score:.3f} "
                    f"below minimum {config.MIN_SHARPE_TO_TRADE}"
                )
                continue

            beta_penalty = max(0.0, (m.beta if m else 0.0) - 0.5) * 0.5
            adjusted     = score * (1 - beta_penalty)
            eligible[worker] = adjusted

        if not eligible:
            logger.warning("No workers passed risk filter. Staying in cash.")
            return {}

        total_score    = sum(eligible.values())
        max_deployable = total_free_capital * config.MAX_POSITION_PCT

        allocation = {}
        for worker, score in eligible.items():
            amount = round(max_deployable * (score / total_score), 2)
            if amount >= config.MIN_TRADE_SIZE_USD:
                allocation[worker] = amount

        logger.info(f"Recommended allocation: {allocation}")
        return allocation

    # -- Risk Gate -------------------------------------------------------------

    def is_safe_to_trade(
        self,
        var_usd:       float,
        cvar_usd:      float,
        total_capital: float,
    ) -> Tuple[bool, str]:
        """
        Phase 2: CVaR is now the PRIMARY gate -- fires before VaR.
        """
        max_var_usd  = total_capital * config.MAX_VAR_PCT
        max_cvar_usd = max_var_usd * config.CVAR_MULTIPLIER

        if cvar_usd > max_cvar_usd:
            return False, (
                f"CVaR ${cvar_usd:.2f} exceeds limit ${max_cvar_usd:.2f}"
            )
        if var_usd > max_var_usd:
            return False, (
                f"VaR ${var_usd:.2f} exceeds limit ${max_var_usd:.2f}"
            )
        return True, "OK"

    def filter_funding_opportunities(
        self,
        rates: Dict[str, float],
    ) -> Dict[str, float]:
        filtered = {
            t: r for t, r in rates.items()
            if r >= config.MIN_FUNDING_RATE
        }
        logger.info(
            f"Funding filter: {len(rates)} pairs -> "
            f"{len(filtered)} above minimum {config.MIN_FUNDING_RATE*100:.4f}%"
        )
        return filtered

    # -- Math Helpers ----------------------------------------------------------

    def _downside_deviation(self, returns: np.ndarray, target: float = 0.0) -> float:
        """
        RMS of returns that fall BELOW the target rate.
        Upside volatility is completely ignored -- that's the point.
        """
        downside = returns[returns < target] - target
        if len(downside) == 0:
            return 1e-9
        return float(np.sqrt(np.mean(downside ** 2)))

    def _max_drawdown(self, returns: np.ndarray) -> float:
        """
        Largest peak-to-trough loss as a positive percentage.
        E.g., 15.0 means the strategy dropped 15% from its peak at worst.
        """
        cumulative  = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns   = (cumulative - running_max) / running_max
        return float(abs(np.min(drawdowns))) * 100

    def _calculate_beta(
        self,
        strategy_returns:  np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """
        Beta = Cov(strategy, benchmark) / Var(benchmark)
        Low/negative beta = strategy is decorrelated from market crashes.
        """
        if np.std(benchmark_returns) == 0:
            return 0.0
        variance = np.var(benchmark_returns)
        if variance == 0:
            return 0.0
        covariance = np.cov(strategy_returns, benchmark_returns)[0][1]
        return float(covariance / variance)
