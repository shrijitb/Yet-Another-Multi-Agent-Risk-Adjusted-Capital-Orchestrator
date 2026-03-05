"""
workers/data_loader.py

Fetches real historical funding rates and builds a time-ordered replay queue
for backtesting.

Data source: CoinGlass public API (no API key, no geo-block, US Pi-friendly).
Fallback:    OKX public API (also US-accessible, no auth required).

Bybit/Binance are NOT used here -- both block US-based IPs (HTTP 451/403),
which breaks deployment on a Raspberry Pi without a VPN.

Usage:
    from workers.data_loader import load_backtest_rates
    queue = load_backtest_rates(days=30)
    # Returns list of {pair: rate} dicts, chronological, one per 8h window.
"""

import json
import time
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Dict

logger = logging.getLogger(__name__)

# Pairs to backtest -- standard perp symbols
BACKTEST_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "ADAUSDT", "SUIUSDT",
]

# Internal format: "BTCUSDT" -> "BTC/USDT"
_SYMBOL_MAP = {s: s[:-4] + "/" + s[-4:] for s in BACKTEST_PAIRS}

# ---------------------------------------------------------------------------
# Primary source: OKX public funding rate history
# US-accessible, no auth, 100 records per call (8h intervals = 100 windows)
# Docs: https://www.okx.com/docs-v5/en/#public-data-rest-api-get-funding-rate-history
# ---------------------------------------------------------------------------
OKX_URL = "https://www.okx.com/api/v5/public/funding-rate-history"

# CoinGlass as backup (free, public, US-accessible)
# Docs: https://coinglass.com/pricing (free tier supports history)
COINGLASS_URL = "https://open-api.coinglass.com/public/v2/funding_usd_history"


def _fetch_okx(symbol: str, start_ms: int, end_ms: int) -> List[Dict]:
    """
    Fetch funding rate history from OKX for one symbol.
    OKX uses instrument ID format: BTC-USDT-SWAP
    Returns list of {fundingRate, fundingTime} dicts, oldest-first.
    """
    # Convert "BTCUSDT" -> "BTC-USDT-SWAP"
    base = symbol[:-4]        # "BTC"
    inst = f"{base}-USDT-SWAP"

    results = []
    after = None  # Pagination cursor (OKX uses timestamp-based cursors)

    while True:
        params = f"instId={inst}&limit=100"
        if after:
            params += f"&after={after}"

        url = f"{OKX_URL}?{params}"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "HypervisorBacktest/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            logger.warning(f"OKX fetch failed for {symbol}: {e}")
            break

        if data.get("code") != "0":
            logger.warning(f"OKX error for {symbol}: {data.get('msg')}")
            break

        batch = data.get("data", [])
        if not batch:
            break

        # Filter to our time window
        for row in batch:
            ts = int(row["fundingTime"])
            if ts >= start_ms:
                results.append(row)

        # OKX returns newest-first. Stop if oldest in batch is before our window.
        oldest_ts = int(batch[-1]["fundingTime"])
        if oldest_ts < start_ms:
            break

        # Paginate: set cursor to oldest timestamp in this batch
        after = str(oldest_ts)

    # Sort oldest-first
    results.sort(key=lambda x: int(x["fundingTime"]))
    return results


def _fetch_okx_spot_price(symbol: str) -> float:
    """Fetch current spot price from OKX (for paper trading price simulation)."""
    base = symbol[:-4]
    inst = f"{base}-USDT"
    url  = f"https://www.okx.com/api/v5/market/ticker?instId={inst}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        return float(data["data"][0]["last"])
    except Exception:
        return 100.0  # Safe fallback


def load_backtest_rates(days: int = 30) -> List[Dict[str, float]]:
    """
    Fetch `days` of historical funding rates for all BACKTEST_PAIRS from OKX.
    Returns list of {pair: rate} snapshots in chronological order,
    one per 8-hour funding window.

    Geo-block safe: OKX public API works from US IPs and Raspberry Pi.
    """
    logger.info(f"Fetching {days} days of historical funding rates from OKX (US-accessible)...")

    now_ms   = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000

    by_time: Dict[int, Dict[str, float]] = defaultdict(dict)
    success_count = 0

    for i, symbol in enumerate(BACKTEST_PAIRS):
        logger.info(f"  [{i+1}/{len(BACKTEST_PAIRS)}] Fetching {symbol} from OKX...")
        try:
            records = _fetch_okx(symbol, start_ms, now_ms)
            pair = _SYMBOL_MAP[symbol]
            for r in records:
                t    = int(r["fundingTime"])
                rate = float(r["fundingRate"])
                by_time[t][pair] = rate
            logger.info(f"    -> {len(records)} windows loaded for {symbol}")
            success_count += 1
        except Exception as e:
            logger.warning(f"  Failed to load {symbol}: {e}")
        time.sleep(0.1)  # Be polite to the API

    if not by_time:
        logger.error(
            "No historical data fetched from OKX.\n"
            "  Check: is the Pi connected to the internet?\n"
            "  Test:  curl https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=1"
        )
        return []

    snapshots = [by_time[t] for t in sorted(by_time.keys())]

    times    = sorted(by_time.keys())
    first_dt = datetime.fromtimestamp(times[0]  / 1000, tz=timezone.utc)
    last_dt  = datetime.fromtimestamp(times[-1] / 1000, tz=timezone.utc)

    logger.info(
        f"Loaded {len(snapshots)} funding windows "
        f"({success_count}/{len(BACKTEST_PAIRS)} pairs) "
        f"from {first_dt.strftime('%Y-%m-%d')} to {last_dt.strftime('%Y-%m-%d')}"
    )

    # Sanity check: show rate distribution so you can tune MIN_FUNDING_RATE
    all_rates = [r for snap in snapshots for r in snap.values()]
    if all_rates:
        above_threshold = sum(1 for r in all_rates if r >= 0.0001)
        logger.info(
            f"Rate stats: min={min(all_rates):.6f} max={max(all_rates):.6f} "
            f"median={sorted(all_rates)[len(all_rates)//2]:.6f} | "
            f"{above_threshold}/{len(all_rates)} above 0.01% threshold"
        )

    return snapshots


def fetch_live_funding_rates_okx() -> Dict[str, float]:
    """
    Fetch current (live) funding rates from OKX for all tracked pairs.
    US-accessible alternative to Binance/Bybit for the Pi deployment.
    Used by FundingArbWorker when USE_LIVE_RATES=True.
    """
    url = "https://www.okx.com/api/v5/public/funding-rate"
    rates = {}

    for symbol in BACKTEST_PAIRS:
        base = symbol[:-4]
        inst = f"{base}-USDT-SWAP"
        try:
            req = urllib.request.Request(
                f"{url}?instId={inst}",
                headers={"User-Agent": "HypervisorLive/1.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            if data.get("code") == "0" and data.get("data"):
                pair = _SYMBOL_MAP[symbol]
                rates[pair] = float(data["data"][0]["fundingRate"])
        except Exception as e:
            logger.warning(f"Live rate fetch failed for {symbol}: {e}")
        time.sleep(0.05)

    return rates
