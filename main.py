"""
main.py -- Entry point for the Quant Hypervisor.

Usage:
    venv-run main.py                    # Normal mode (1-hour cycles)
    venv-run main.py --fast             # Fast mode (10-second cycles, simulated rates)
    venv-run main.py --backtest 30      # Backtest on 30 days of real Binance data
    venv-run main.py --backtest 90      # Backtest on 90 days of real Binance data
    venv-run main.py --cycle-seconds 30

How to stop:
    Create a file named 'stop' in the hypervisor folder (or double-click stop.bat).
    In --backtest mode the system stops automatically when historical data runs out.
"""

import os
import sys
import io
import argparse
import logging
import threading

import config

os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

logging.basicConfig(
    level   = getattr(logging, config.LOG_LEVEL),
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")),
    ]
)
logger = logging.getLogger(__name__)

from core.hypervisor import QuantHypervisor
from core.portfolio_state import WorkerType
from workers.funding_arb import FundingArbWorker


def _watch_for_quit(hypervisor: QuantHypervisor, stop_event: threading.Event):
    """
    Background thread that watches for a stop signal.

    Checks every second for a file called 'stop' in the hypervisor folder.
    To stop: double-click stop.bat (Windows) or run: echo > stop  in any terminal.

    We use a file watcher instead of keyboard input because venv-run and
    PowerShell launchers both redirect stdin, making msvcrt and input() unreliable.
    The file approach works regardless of how the process was launched.
    """
    # Clean up any leftover stop file from a previous run
    if os.path.exists("stop"):
        os.remove("stop")

    while not stop_event.is_set():
        if os.path.exists("stop"):
            try:
                os.remove("stop")
            except OSError:
                pass
            print("\n[Hypervisor] Stop file detected -- shutting down...", flush=True)
            hypervisor.stop()
            stop_event.set()
            return
        stop_event.wait(timeout=1.0)


def main():
    parser = argparse.ArgumentParser(description="Quant Hypervisor")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: rebalance every 10 seconds (simulated rates)")
    parser.add_argument("--backtest", type=int, default=None, metavar="DAYS",
                        help="Backtest on N days of real Binance historical funding data")
    parser.add_argument("--cycle-seconds", type=int, default=None,
                        help="Override rebalance interval in seconds")
    args = parser.parse_args()

    backtest_mode = args.backtest is not None

    if backtest_mode:
        config.REBALANCE_INTERVAL_SEC = 1   # Zip through history as fast as possible
    elif args.fast:
        config.REBALANCE_INTERVAL_SEC = 10
    elif args.cycle_seconds:
        config.REBALANCE_INTERVAL_SEC = args.cycle_seconds
    if os.environ.get("FAST_MODE"):
        config.REBALANCE_INTERVAL_SEC = 10

    interval_label = (f"{config.REBALANCE_INTERVAL_SEC}s"
                      if config.REBALANCE_INTERVAL_SEC < 120
                      else f"{config.REBALANCE_INTERVAL_SEC // 60}m")

    mode_label = "BACKTEST" if backtest_mode else ("PAPER" if config.PAPER_TRADING else "LIVE - REAL MONEY")

    logger.info("=" * 60)
    logger.info("  QUANT HYPERVISOR -- STARTING UP")
    logger.info("=" * 60)
    logger.info(f"  Mode           : {mode_label}")
    logger.info(f"  Initial Capital: ${config.INITIAL_CAPITAL_USD:.2f}")
    logger.info(f"  Max VaR        : {config.MAX_VAR_PCT*100:.1f}% of capital")
    if backtest_mode:
        logger.info(f"  Backtest Window: {args.backtest} days of real Binance data")
    else:
        logger.info(f"  Rebalance Every: {interval_label}")
    logger.info("  STOP           : create a file named 'stop'  (or double-click stop.bat)")
    logger.info("=" * 60)

    # -- Register Workers ------------------------------------------------------
    funding_worker = FundingArbWorker()
    funding_worker.simulate_return_history(n_hours=168, seed_sharpe=2.5)

    if backtest_mode:
        from workers.data_loader import load_backtest_rates
        snapshots = load_backtest_rates(days=args.backtest)
        if not snapshots:
            logger.error("Backtest aborted: no data loaded.")
            sys.exit(1)
        funding_worker.load_backtest_data(snapshots)
        total_windows = len(snapshots)
        logger.info(f"Backtest: {total_windows} windows = ~{total_windows // 3} real days to replay")

    workers = {WorkerType.FUNDING_ARB: funding_worker}

    hypervisor = QuantHypervisor(
        workers         = workers,
        initial_capital = config.INITIAL_CAPITAL_USD,
        paper_trading   = config.PAPER_TRADING,
    )

    # -- Start quit-watcher in background thread -------------------------------
    stop_event = threading.Event()
    watcher = threading.Thread(
        target=_watch_for_quit,
        args=(hypervisor, stop_event),
        daemon=True,
        name="QuitWatcher"
    )
    watcher.start()

    # -- Run hypervisor in main thread (blocks until stop() called) ------------
    hypervisor.start()

    # Signal the watcher thread to exit cleanly if hypervisor stopped itself
    stop_event.set()

    if backtest_mode:
        logger.info("=" * 60)
        logger.info("  BACKTEST COMPLETE")
        logger.info("=" * 60)

    logger.info("Hypervisor exited cleanly.")
    sys.exit(0)


if __name__ == "__main__":
    main()
