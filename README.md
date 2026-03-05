# Risk-Adjusted Hypervisor

A production-ready quantitative trading system for delta-neutral funding rate arbitrage with advanced risk management and portfolio optimization.

## 🎯 Overview

The Hypervisor is a modular trading framework designed to capture funding rate arbitrage opportunities in perpetual futures markets while maintaining strict risk controls. It automatically:

- **Scans** all perpetual futures pairs for positive funding rates
- **Executes** delta-neutral positions (spot long + futures short)
- **Manages** portfolio risk through VaR/CVaR limits and position sizing
- **Optimizes** capital allocation based on Sharpe ratio performance
- **Backtests** against historical funding rate data

## 🏗️ Architecture

```
main.py
├── QuantHypervisor (core orchestrator)
│   ├── RiskEngine (VaR/CVaR calculations, allocation)
│   ├── PortfolioState (position tracking, P&L)
│   └── AuditDB (cycle logging, metrics)
└── Workers
    └── FundingArbWorker (delta-neutral arbitrage)
```

### Key Components

- **Hypervisor**: Main orchestration loop that runs every rebalance interval
- **Risk Engine**: Monte Carlo VaR/CVaR calculations and capital allocation
- **Portfolio State**: Real-time position tracking and P&L calculation
- **Audit Database**: SQLite-based logging for performance analysis
- **Workers**: Pluggable trading strategies (currently funding arbitrage)

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Required packages (install via `pip install -r requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/risk-adjusted-hypervisor.git
cd risk-adjusted-hypervisor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your settings in `config.py`

### Running the System

#### Normal Mode (1-hour cycles)
```bash
python main.py
```

#### Fast Mode (10-second cycles with simulated rates)
```bash
python main.py --fast
```

#### Backtesting
```bash
# Backtest on 30 days of historical data
python main.py --backtest 30

# Backtest on 90 days of historical data
python main.py --backtest 90
```

#### Custom Cycle Interval
```bash
# Run with 30-second cycles
python main.py --cycle-seconds 30
```

### Stopping the System

Create a `stop` file in the hypervisor directory:
```bash
touch stop
```

Or double-click `stop.bat` on Windows.

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Capital Management
INITIAL_CAPITAL_USD = 200.0
MIN_TRADE_SIZE_USD = 10.0
MAX_POSITION_PCT = 0.80

# Risk Parameters
VAR_CONFIDENCE = 0.99
VAR_SIMULATIONS = 10_000
MAX_VAR_PCT = 0.05  # 5% VaR limit
MIN_SHARPE_TO_TRADE = 0.5

# Trading Parameters
REBALANCE_INTERVAL_SEC = 3_600  # 1 hour
MIN_FUNDING_RATE = 0.0001  # 0.01%
PAPER_TRADING = True  # Set to False for live trading
```

## 📊 Strategy: Delta-Neutral Funding Arbitrage

### How It Works

1. **Scan**: Monitor all perpetual futures pairs for positive funding rates
2. **Select**: Choose the best opportunity above minimum threshold
3. **Execute**: Open delta-neutral position:
   - Buy spot (long)
   - Sell futures (short)
4. **Collect**: Earn funding payments every 8 hours
5. **Monitor**: Watch for rate reversals and rotate positions

### Risk Management

- **Delta Neutral**: No directional market exposure
- **VaR Limits**: Stop trading if portfolio VaR exceeds 5% of capital
- **Sharpe Filtering**: Only trade opportunities with Sharpe > 0.5
- **Position Sizing**: Dynamic allocation based on risk-adjusted returns
- **Emergency Hedge**: Close all positions if risk limits breached

### Risk Factors

- **Exchange Risk**: Counterparty risk if exchange fails
- **Liquidation Risk**: Futures leg liquidation during extreme moves
- **Funding Reversal**: Rate flips negative, you start paying
- **Slippage**: Execution costs during entry/exit

## 📈 Backtesting

The system includes comprehensive backtesting capabilities:

```bash
# Load 30 days of historical Binance funding data
python main.py --backtest 30
```

Features:
- Historical funding rate replay
- Performance metrics tracking
- Sharpe ratio calculations
- P&L analysis

## 📊 Monitoring & Logging

### Log Files
- `logs/hypervisor.log`: System operation logs
- `logs/portfolio_state.json`: Real-time portfolio snapshot
- `logs/hypervisor.db`: SQLite audit database

### Key Metrics Tracked
- Portfolio VaR/CVaR
- Worker Sharpe ratios
- Position P&L
- Funding payments collected
- Allocation decisions

## 🔧 Development

### Adding New Strategies

1. Create a new worker class inheriting from `BaseWorker`
2. Implement required methods:
   - `execute()`: Main trading logic
   - `get_market_data()`: Market data fetching
   - `close_position()`: Position exit logic
3. Register the worker in `main.py`

### Risk Engine Customization

The `RiskEngine` class handles:
- Monte Carlo VaR/CVaR calculations
- Sharpe ratio computations
- Capital allocation recommendations
- Risk limit enforcement

## 🚨 Important Notes

### Paper Trading vs Live Trading
- **Paper Trading** (default): Simulated execution, safe for testing
- **Live Trading**: Real money execution - use with extreme caution

### Exchange Connectivity
- Currently supports Binance and Bybit via ccxt
- Live trading requires API credentials
- US users should use OKX for funding rate data (Binance/Bybit blocked)

### Risk Warnings
- This is a sophisticated trading system
- Always test thoroughly in paper trading mode first
- Understand all risk parameters before going live
- Never risk more capital than you can afford to lose

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Uses ccxt for exchange connectivity
- SQLite for audit logging
- Python standard library for core functionality

---

**⚠️ Disclaimer**: This software is provided as-is for educational and research purposes. The authors are not responsible for any financial losses or damages resulting from its use. Always conduct thorough testing and understand the risks before deploying any trading system with real capital.