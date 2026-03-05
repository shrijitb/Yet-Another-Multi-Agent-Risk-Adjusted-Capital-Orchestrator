"""
config.py — Single source of truth for all Hypervisor parameters.
Change values here. Never hardcode them elsewhere.
"""

# ── Capital ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL_USD     = 200.0     # Your starting bankroll
MIN_TRADE_SIZE_USD      = 10.0      # Below this, don't bother (fees eat you)
MAX_POSITION_PCT        = 0.80      # Never deploy more than 80% of capital at once

# ── Risk Engine ────────────────────────────────────────────────────────────────
VAR_CONFIDENCE          = 0.99      # 99% confidence interval for VaR
VAR_SIMULATIONS         = 10_000    # Monte Carlo paths (more = slower but accurate)
VAR_HORIZON_HOURS       = 8         # Match funding rate reset window
MAX_VAR_PCT             = 0.05      # If VaR > 5% of capital, halt and hedge
CVAR_MULTIPLIER         = 1.5       # CVaR threshold = MAX_VAR_PCT * this
LOOKBACK_DAYS           = 30        # Historical window for volatility estimation
MIN_SHARPE_TO_TRADE     = 0.5       # Won't allocate to any worker below this Sharpe
SHARPE_RISK_FREE_RATE   = 0.045     # ~4.5% annual T-bill rate (annualized)

# ── Rebalancing ────────────────────────────────────────────────────────────────
REBALANCE_INTERVAL_SEC  = 3_600     # How often the Hypervisor evaluates allocation (1hr)
FUNDING_RATE_INTERVAL   = 28_800    # Funding resets every 8 hours (28800 seconds)
MIN_FUNDING_RATE        = 0.0001    # Don't enter arb below 0.01% (0.03% was bull-run calibrated; real market is lower)

# ── Execution ─────────────────────────────────────────────────────────────────
PAPER_TRADING           = True      # TRUE = Track 2 (safe). FALSE = live money.
USE_LIVE_RATES          = False     # TRUE = fetch real funding rates via ccxt. FALSE = simulate.
SLIPPAGE_MODEL_PCT      = 0.0005    # Simulate 0.05% slippage on paper fills
FEE_MODEL_PCT           = 0.0004    # Simulate 0.04% taker fee (Binance standard)

# ── Exchanges (ccxt identifiers) ──────────────────────────────────────────────
EXCHANGES               = ["binance", "bybit"]
QUOTE_CURRENCY          = "USDT"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL               = "INFO"
LOG_FILE                = "logs/hypervisor.log"
STATE_SNAPSHOT_FILE     = "logs/portfolio_state.json"
