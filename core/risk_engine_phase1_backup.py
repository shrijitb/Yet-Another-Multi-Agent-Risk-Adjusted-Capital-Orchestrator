"""
core/risk_engine.py

The Hedge Fund Layer. This is the gatekeeper.
Nothing gets traded without passing through here first.

Implements:
  - Monte Carlo Value at Risk (VaR) at 99% confidence
  - Conditional VaR / Expected Shortfall (CVaR) — what VaR misses
  - Dynamic Sharpe Ratio per worker
  - Emergency hedge trigger logic
"""

import logging
import numpy as np
from typing import Dict, List, Tuple

import config
from core.portfolio_state import WorkerType

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Quantitative risk layer. Stateless — all inputs are passed in explicitly.
    This makes it testable: same inputs always produce same outputs.

    Key concepts for beginners:
      VaR:  "With 99% confidence, we won't lose more than $X in the next 8 hours."
      CVaR: "But IF we land in that worst 1%, the average loss will be $Y."
            CVaR is always worse than VaR. It's what you actually care about.
      Sharpe: Risk-adjusted return. High return + low volatility = high Sharpe.
              A Sharpe of 1.0 is acceptable. 2.0+ is excellent. <0.5 = don't trade.
    """

    def __init__(self):
        self.last_var:   float = 0.0
        self.last_cvar:  float = 0.0
        logger.info("RiskEngine initialized")

    # ── Monte Carlo VaR ────────────────────────────────────────────────────────

    def calculate_monte_carlo_var(
        self,
        position_size_usd: float,
        historical_returns: List[float],
        horizon_hours: int = config.VAR_HORIZON_HOURS,
        n_simulations: int = config.VAR_SIMULATIONS,
        confidence: float = config.VAR_CONFIDENCE,
    ) -> Tuple[float, float]:
        """
        Run Monte Carlo simulation to estimate VaR and CVaR.

        Args:
            position_size_usd:  Dollar value of the position to evaluate.
            historical_returns: List of past hourly returns as decimals (e.g. 0.002 = 0.2%).
            horizon_hours:      How many hours forward to simulate.
            n_simulations:      Number of random price paths to generate.
            confidence:         Confidence level (0.99 = 99%).

        Returns:
            (var_usd, cvar_usd): Both expressed as positive dollar loss amounts.

        How it works:
            1. Estimate mean and std deviation of returns from history.
            2. Simulate n_simulations random return paths over horizon_hours.
            3. Find the loss at the (1 - confidence) percentile = VaR.
            4. Average all losses worse than VaR = CVaR.
        """
        if len(historical_returns) < 10:
            # Not enough history — return conservative estimate
            logger.warning("Insufficient return history for VaR — using conservative fallback")
            fallback = position_size_usd * 0.10  # Assume 10% max loss
            self.last_var  = fallback
            self.last_cvar = fallback * config.CVAR_MULTIPLIER
            return self.last_var, self.last_cvar

        returns = np.array(historical_returns, dtype=np.float64)
        mu      = np.mean(returns)       # Average hourly return
        sigma   = np.std(returns)        # Hourly volatility

        # Generate random return paths.
        # Shape: (n_simulations, horizon_hours)
        # Each row is one simulated future: a sequence of hourly returns.
        simulated_returns = np.random.normal(
            loc   = mu,
            scale = sigma,
            size  = (n_simulations, horizon_hours)
        )

        # Compound returns across the horizon to get total path return
        # (1 + r1) * (1 + r2) * ... - 1
        path_returns = np.prod(1 + simulated_returns, axis=1) - 1

        # Convert to dollar P&L
        path_pnl = path_returns * position_size_usd

        # Sort losses (most negative first)
        # VaR = the loss at the confidence threshold (e.g. 1st percentile)
        loss_threshold_idx = int((1 - confidence) * n_simulations)
        sorted_pnl = np.sort(path_pnl)  # Ascending: worst losses first

        var_usd  = abs(sorted_pnl[loss_threshold_idx])  # VaR as positive number
        # CVaR = average of all paths worse than VaR
        cvar_usd = abs(np.mean(sorted_pnl[:loss_threshold_idx]))

        self.last_var  = var_usd
        self.last_cvar = cvar_usd

        logger.info(
            f"Monte Carlo VaR({confidence*100:.0f}%): ${var_usd:.2f} | "
            f"CVaR: ${cvar_usd:.2f} | "
            f"μ={mu:.5f} σ={sigma:.5f} | "
            f"{n_simulations:,} paths over {horizon_hours}h"
        )
        return var_usd, cvar_usd

    # ── Sharpe Ratio ───────────────────────────────────────────────────────────

    def calculate_sharpe(
        self,
        hourly_returns: List[float],
        risk_free_annual: float = config.SHARPE_RISK_FREE_RATE,
    ) -> float:
        """
        Calculate annualized Sharpe Ratio from a list of hourly returns.

        Sharpe = (Strategy Return - Risk Free Rate) / Strategy Volatility

        We annualize by multiplying hourly figures by sqrt(8760) hours/year.
        This lets us compare strategies on a common scale regardless of frequency.

        Returns:
            Sharpe ratio (float). Negative = this strategy is destroying value.
        """
        if len(hourly_returns) < 24:
            logger.warning("Fewer than 24 hours of return data — Sharpe unreliable")
            return 0.0

        returns = np.array(hourly_returns, dtype=np.float64)
        hours_per_year = 8_760

        # Annualize mean and std
        mean_annual = np.mean(returns) * hours_per_year
        std_annual  = np.std(returns)  * np.sqrt(hours_per_year)

        if std_annual == 0:
            # Zero volatility — strategy is perfectly flat (or broken)
            # Funding arb can look like this legitimately; cap at a high value
            return 5.0 if mean_annual > 0 else 0.0

        sharpe = (mean_annual - risk_free_annual) / std_annual
        return round(sharpe, 4)

    def calculate_sharpe_for_all_workers(
        self,
        worker_returns: Dict[WorkerType, List[float]],
    ) -> Dict[WorkerType, float]:
        """
        Calculate Sharpe for every worker in one pass.
        Returns a dict of {WorkerType: sharpe_float}.
        """
        result = {}
        for worker, returns in worker_returns.items():
            sharpe = self.calculate_sharpe(returns)
            result[worker] = sharpe
            logger.info(f"Sharpe[{worker.value}] = {sharpe:.4f}")
        return result

    # ── Capital Allocation Decision ────────────────────────────────────────────

    def recommend_allocation(
        self,
        sharpe_scores: Dict[WorkerType, float],
        total_free_capital: float,
    ) -> Dict[WorkerType, float]:
        """
        Given Sharpe scores for each worker, recommend dollar allocations.

        Algorithm:
          1. Filter out workers below minimum Sharpe threshold (not worth the risk).
          2. Weight remaining workers by their Sharpe score (proportional allocation).
          3. Never allocate more than MAX_POSITION_PCT of total capital.

        Returns:
            Dict of {WorkerType: dollar_amount_to_allocate}
        """
        # Filter: only trade workers that clear the minimum Sharpe bar
        eligible = {
            w: s for w, s in sharpe_scores.items()
            if s >= config.MIN_SHARPE_TO_TRADE
        }

        if not eligible:
            logger.warning(
                f"No workers meet minimum Sharpe of {config.MIN_SHARPE_TO_TRADE}. "
                f"Staying in cash."
            )
            return {}

        total_sharpe  = sum(eligible.values())
        max_deployable = total_free_capital * config.MAX_POSITION_PCT

        allocation = {}
        for worker, sharpe in eligible.items():
            weight         = sharpe / total_sharpe
            dollar_amount  = max_deployable * weight
            if dollar_amount >= config.MIN_TRADE_SIZE_USD:
                allocation[worker] = round(dollar_amount, 2)
            else:
                logger.info(
                    f"Skipping {worker.value}: ${dollar_amount:.2f} below min trade size"
                )

        logger.info(f"Recommended allocation: {allocation}")
        return allocation

    # ── Risk Gate (The Circuit Breaker) ───────────────────────────────────────

    def is_safe_to_trade(
        self,
        var_usd: float,
        cvar_usd: float,
        total_capital: float,
    ) -> Tuple[bool, str]:
        """
        Final binary gate before any trade is placed.
        Returns (True, "OK") or (False, reason_string).

        This is called by the Hypervisor every cycle.
        If it returns False, the Hypervisor calls emergency_hedge() instead.
        """
        max_var_usd  = total_capital * config.MAX_VAR_PCT
        max_cvar_usd = max_var_usd   * config.CVAR_MULTIPLIER

        if var_usd > max_var_usd:
            reason = (
                f"VaR ${var_usd:.2f} exceeds limit ${max_var_usd:.2f} "
                f"({config.MAX_VAR_PCT*100:.1f}% of ${total_capital:.2f})"
            )
            logger.warning(f"RISK GATE BLOCKED: {reason}")
            return False, reason

        if cvar_usd > max_cvar_usd:
            reason = (
                f"CVaR ${cvar_usd:.2f} exceeds limit ${max_cvar_usd:.2f}"
            )
            logger.warning(f"RISK GATE BLOCKED: {reason}")
            return False, reason

        return True, "OK"

    # ── Funding Rate Filter ────────────────────────────────────────────────────

    def filter_funding_opportunities(
        self,
        rates: Dict[str, float],
    ) -> Dict[str, float]:
        """
        From a raw dict of {ticker: funding_rate}, return only tickers
        where the rate is positive and above the minimum threshold.
        Negative rates = YOU pay = never enter.
        """
        filtered = {
            ticker: rate
            for ticker, rate in rates.items()
            if rate >= config.MIN_FUNDING_RATE
        }
        logger.info(
            f"Funding filter: {len(rates)} pairs → "
            f"{len(filtered)} above minimum {config.MIN_FUNDING_RATE*100:.4f}%"
        )
        return filtered
