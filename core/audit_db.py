"""
core/audit_db.py  --  SQLite Audit Trail

Every decision the Hypervisor makes is written here permanently.
This is your portfolio's black box recorder.

Why SQLite vs a flat log file:
  - Queryable: "Show me all cycles where Sortino dropped below 1.0"
  - Replayable: "Reconstruct the portfolio state at any point in time"
  - Portable: single .db file, copies easily to/from Pi, opens in DB Browser
  - Free: zero dependencies beyond Python's standard library

Schema (4 tables):
  cycles      -- one row per Hypervisor cycle (risk metrics, decisions)
  positions   -- one row per position open/close event
  allocations -- one row per capital allocation change
  signals     -- one row per intelligence signal (macro/sentiment, Phase 3+)

Usage:
    from core.audit_db import AuditDB
    db = AuditDB()
    db.log_cycle(cycle_num, var, cvar, safe_to_trade, reason)
    db.log_position_open(position)
    db.log_position_close(key, pnl, exit_price)
    db.log_allocation(worker, amount, direction)
    db.log_risk_metrics(cycle_num, worker, metrics)
"""

import sqlite3
import time
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

import config

logger = logging.getLogger(__name__)

DB_PATH = "logs/hypervisor.db"


class AuditDB:
    """
    Thin wrapper around SQLite for structured audit logging.
    All writes are immediate (no batching) -- correctness over speed.
    Reads are available for the iOS dashboard (Phase 4).
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info(f"AuditDB initialized at {db_path}")

    @contextmanager
    def _conn(self):
        """Context manager: auto-commit on success, rollback on exception."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row   # Rows accessible as dicts
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"AuditDB write failed (rolled back): {e}")
            raise
        finally:
            conn.close()

    # -- Schema ----------------------------------------------------------------

    def _init_schema(self):
        """Create all tables if they don't already exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS cycles (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_num       INTEGER NOT NULL,
                    timestamp       REAL    NOT NULL,
                    timestamp_pt    TEXT    NOT NULL,
                    total_capital   REAL,
                    free_capital    REAL,
                    open_positions  INTEGER,
                    var_usd         REAL,
                    cvar_usd        REAL,
                    safe_to_trade   INTEGER,
                    gate_reason     TEXT,
                    emergency_mode  INTEGER,
                    notes           TEXT
                );

                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_num       INTEGER NOT NULL,
                    timestamp       REAL    NOT NULL,
                    worker          TEXT    NOT NULL,
                    sharpe          REAL,
                    sortino         REAL,
                    var_usd         REAL,
                    cvar_usd        REAL,
                    downside_dev    REAL,
                    beta            REAL,
                    mean_return     REAL,
                    volatility      REAL,
                    max_drawdown_pct REAL
                );

                CREATE TABLE IF NOT EXISTS positions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_key    TEXT    NOT NULL,
                    event           TEXT    NOT NULL,
                    timestamp       REAL    NOT NULL,
                    timestamp_pt    TEXT    NOT NULL,
                    ticker          TEXT,
                    side            TEXT,
                    size_usd        REAL,
                    entry_price     REAL,
                    exit_price      REAL,
                    gross_pnl       REAL,
                    fees_usd        REAL,
                    net_pnl         REAL,
                    exchange        TEXT,
                    worker          TEXT,
                    paper_trade     INTEGER
                );

                CREATE TABLE IF NOT EXISTS allocations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_num       INTEGER,
                    timestamp       REAL    NOT NULL,
                    worker          TEXT    NOT NULL,
                    direction       TEXT    NOT NULL,
                    amount_usd      REAL    NOT NULL,
                    pnl             REAL,
                    free_capital_after REAL,
                    total_capital_after REAL
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       REAL    NOT NULL,
                    source          TEXT    NOT NULL,
                    signal_type     TEXT    NOT NULL,
                    ticker          TEXT,
                    value           REAL,
                    raw_data        TEXT,
                    acted_on        INTEGER DEFAULT 0
                );

                -- Indexes for fast dashboard queries
                CREATE INDEX IF NOT EXISTS idx_cycles_ts      ON cycles(timestamp);
                CREATE INDEX IF NOT EXISTS idx_positions_key  ON positions(position_key);
                CREATE INDEX IF NOT EXISTS idx_positions_ts   ON positions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_risk_worker    ON risk_metrics(worker, timestamp);
                CREATE INDEX IF NOT EXISTS idx_signals_ts     ON signals(timestamp);
            """)

    # -- Cycle Logging ---------------------------------------------------------

    def log_cycle(
        self,
        cycle_num:      int,
        total_capital:  float,
        free_capital:   float,
        open_positions: int,
        var_usd:        float,
        cvar_usd:       float,
        safe_to_trade:  bool,
        gate_reason:    str   = "OK",
        emergency_mode: bool  = False,
        notes:          str   = "",
    ):
        now = time.time()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO cycles
                (cycle_num, timestamp, timestamp_pt, total_capital, free_capital,
                 open_positions, var_usd, cvar_usd, safe_to_trade, gate_reason,
                 emergency_mode, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle_num, now, _pt_time(now),
                total_capital, free_capital, open_positions,
                var_usd, cvar_usd, int(safe_to_trade), gate_reason,
                int(emergency_mode), notes,
            ))

    # -- Risk Metrics Logging --------------------------------------------------

    def log_risk_metrics(self, cycle_num: int, worker: str, metrics):
        """
        Log a RiskMetrics object for one worker.
        'metrics' can be a RiskMetrics instance or a plain dict.
        """
        d = metrics.to_dict() if hasattr(metrics, "to_dict") else metrics
        now = time.time()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO risk_metrics
                (cycle_num, timestamp, worker, sharpe, sortino, var_usd, cvar_usd,
                 downside_dev, beta, mean_return, volatility, max_drawdown_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle_num, now, worker,
                d.get("sharpe"), d.get("sortino"),
                d.get("var_usd"), d.get("cvar_usd"),
                d.get("downside_dev"), d.get("beta"),
                d.get("mean_return"), d.get("volatility"),
                d.get("max_drawdown_pct"),
            ))

    # -- Position Logging ------------------------------------------------------

    def log_position_open(self, position, paper_trade: bool = True):
        now = time.time()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO positions
                (position_key, event, timestamp, timestamp_pt, ticker, side,
                 size_usd, entry_price, exchange, worker, paper_trade)
                VALUES (?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{position.ticker}_{position.side.value}_{position.exchange}",
                now, _pt_time(now),
                position.ticker, position.side.value,
                position.size_usd, position.entry_price,
                position.exchange, position.worker.value,
                int(paper_trade),
            ))

    def log_position_close(
        self,
        position_key: str,
        exit_price:   float,
        net_pnl:      float,
        fees_usd:     float = 0.0,
        paper_trade:  bool  = True,
    ):
        gross_pnl = net_pnl + fees_usd
        now = time.time()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO positions
                (position_key, event, timestamp, timestamp_pt,
                 exit_price, gross_pnl, fees_usd, net_pnl, paper_trade)
                VALUES (?, 'CLOSE', ?, ?, ?, ?, ?, ?, ?)
            """, (
                position_key, now, _pt_time(now),
                exit_price, gross_pnl, fees_usd, net_pnl, int(paper_trade),
            ))

    # -- Allocation Logging ----------------------------------------------------

    def log_allocation(
        self,
        cycle_num:           int,
        worker:              str,
        direction:           str,   # "ALLOCATE" or "DEALLOCATE"
        amount_usd:          float,
        pnl:                 float  = 0.0,
        free_capital_after:  float  = 0.0,
        total_capital_after: float  = 0.0,
    ):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO allocations
                (cycle_num, timestamp, worker, direction, amount_usd,
                 pnl, free_capital_after, total_capital_after)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle_num, time.time(), worker, direction,
                amount_usd, pnl, free_capital_after, total_capital_after,
            ))

    # -- Signal Logging (Phase 3+) ---------------------------------------------

    def log_signal(
        self,
        source:      str,
        signal_type: str,
        ticker:      Optional[str]  = None,
        value:       Optional[float]= None,
        raw_data:    Optional[dict] = None,
        acted_on:    bool           = False,
    ):
        """
        Log an intelligence signal (macro data, sentiment, BDI reading, etc.)
        This table feeds the war-room intelligence layer in Phase 3.
        """
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO signals
                (timestamp, source, signal_type, ticker, value, raw_data, acted_on)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(), source, signal_type, ticker, value,
                json.dumps(raw_data) if raw_data else None,
                int(acted_on),
            ))

    # -- Query Helpers (for dashboard / debugging) -----------------------------

    def get_recent_cycles(self, n: int = 10) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cycles ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_pnl_summary(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_closes,
                    SUM(net_pnl) as total_net_pnl,
                    SUM(fees_usd) as total_fees,
                    AVG(net_pnl) as avg_pnl_per_trade,
                    MIN(net_pnl) as worst_trade,
                    MAX(net_pnl) as best_trade
                FROM positions WHERE event = 'CLOSE'
            """).fetchone()
        return dict(row) if row else {}

    def get_worker_performance(self) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    worker,
                    AVG(sharpe)   as avg_sharpe,
                    AVG(sortino)  as avg_sortino,
                    AVG(cvar_usd) as avg_cvar,
                    COUNT(*)      as sample_count
                FROM risk_metrics
                GROUP BY worker
                ORDER BY avg_sortino DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def get_all_positions(self, event: Optional[str] = None) -> list:
        with self._conn() as conn:
            if event:
                rows = conn.execute(
                    "SELECT * FROM positions WHERE event=? ORDER BY timestamp DESC",
                    (event,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM positions ORDER BY timestamp DESC"
                ).fetchall()
        return [dict(r) for r in rows]


# -- Utility -------------------------------------------------------------------

def _pt_time(ts: float) -> str:
    """
    Convert Unix timestamp to Pacific Time string for human-readable logs.
    PT = UTC-8 (PST) or UTC-7 (PDT). We use UTC-8 as fixed offset for simplicity.
    For proper DST handling, add: pip install pytz
    """
    import datetime
    pt_offset = datetime.timezone(datetime.timedelta(hours=-8))
    dt = datetime.datetime.fromtimestamp(ts, tz=pt_offset)
    return dt.strftime("%Y-%m-%d %H:%M:%S PT")
