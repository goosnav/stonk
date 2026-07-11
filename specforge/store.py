"""SQLite persistence + audit log. One file, WAL mode, plain SQL (no ORM).

Design notes (see dev/ARCHITECTURE.md):
- `trades` doubles as the analog-trade store used by forecast.py for error bars.
  Backtests write source='backtest' rows (into their own DB file, same schema);
  live/paper write source='paper'|'live'. Queries take a source filter.
- `audit` is append-only; every pipeline step writes a row. A trade must be fully
  reconstructable from audit rows alone (tested in Phase 1 exit criteria).
- `kv` holds small mutable state: kill-switch flags, daily counters, runtime
  config overrides from the GUI.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from logging.handlers import RotatingFileHandler
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Optional

_AUDIT_LOG = logging.getLogger("specforge.audit")
_AUDIT_LOG.propagate = False


def configure_file_logging(mode: str, log_dir: str | Path | None = None) -> Path:
    """Mirror the SQLite audit trail to a rotating, grep-friendly JSONL file."""
    directory = Path(log_dir or Path(__file__).resolve().parent.parent / "logs")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"audit-{mode}.jsonl"
    if not any(isinstance(h, RotatingFileHandler) and
               Path(h.baseFilename) == path for h in _AUDIT_LOG.handlers):
        handler = RotatingFileHandler(path, maxBytes=10_000_000, backupCount=5)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _AUDIT_LOG.addHandler(handler)
        _AUDIT_LOG.setLevel(logging.INFO)
    return path

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars(
  symbol TEXT NOT NULL, d TEXT NOT NULL,          -- d = ISO date of the bar
  open REAL, high REAL, low REAL, close REAL, volume REAL,
  source TEXT, ingested_at TEXT,
  PRIMARY KEY(symbol, d)
);
CREATE TABLE IF NOT EXISTS signals(
  id TEXT PRIMARY KEY, cycle_id TEXT, ts TEXT, node_id TEXT, symbol TEXT,
  direction TEXT, score REAL, confidence REAL, horizon_days INTEGER,
  expected_return REAL, expected_volatility REAL, downside REAL, evidence TEXT
);
CREATE TABLE IF NOT EXISTS candidates(
  id TEXT PRIMARY KEY, cycle_id TEXT, ts TEXT, symbol TEXT, final_score REAL,
  payload TEXT                                     -- full TradeCandidate json
);
CREATE TABLE IF NOT EXISTS orders(
  id TEXT PRIMARY KEY, candidate_id TEXT, cycle_id TEXT, symbol TEXT,
  asset_type TEXT, side TEXT, qty REAL, limit_price REAL, notional REAL,
  idempotency_key TEXT UNIQUE, status TEXT, broker_order_id TEXT,
  option_symbol TEXT, created_at TEXT, updated_at TEXT,
  mode TEXT DEFAULT 'paper'                           -- paper|live (shared DB)
);
CREATE TABLE IF NOT EXISTS fills(
  order_id TEXT, symbol TEXT, side TEXT, qty REAL, price REAL, fees REAL,
  filled_at TEXT
);
CREATE TABLE IF NOT EXISTS positions(               -- engine-side open position metadata
  id TEXT PRIMARY KEY, symbol TEXT, asset_type TEXT, qty REAL, avg_cost REAL,
  opened_at TEXT, horizon_days INTEGER, stop_price REAL, candidate_id TEXT,
  nodes TEXT, option_symbol TEXT, status TEXT DEFAULT 'open',  -- open|closed
  mode TEXT DEFAULT 'paper'                          -- paper|live (shared DB)
);
CREATE TABLE IF NOT EXISTS trades(                  -- closed round-trips (+ backtest analogs)
  id TEXT PRIMARY KEY, symbol TEXT, asset_type TEXT,
  entry_date TEXT, exit_date TEXT, entry_price REAL, exit_price REAL, qty REAL,
  pnl REAL, ret REAL,                               -- ret = net simple return incl. modeled costs
  horizon_days INTEGER, score REAL, score_bucket TEXT, regime TEXT,
  nodes TEXT, source TEXT, exit_reason TEXT
);
CREATE TABLE IF NOT EXISTS equity_curve(
  d TEXT, ts TEXT, equity REAL, cash REAL, source TEXT,
  PRIMARY KEY(d, source)
);
CREATE TABLE IF NOT EXISTS node_stats(
  node_id TEXT, computed_at TEXT, payload TEXT,
  PRIMARY KEY(node_id, computed_at)
);
CREATE TABLE IF NOT EXISTS weights(                 -- learned multiplier on base config weight
  node_id TEXT PRIMARY KEY, multiplier REAL, updated_at TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, cycle_id TEXT,
  event_type TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS kv(
  key TEXT PRIMARY KEY, value TEXT
);
CREATE TABLE IF NOT EXISTS ai_ledger(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, day TEXT, model TEXT,
  purpose TEXT, node_id TEXT, in_tokens INTEGER, out_tokens INTEGER,
  cost_usd REAL, cache_hit INTEGER, ok INTEGER
);
CREATE TABLE IF NOT EXISTS approvals(               -- human approval queue
  intent_id TEXT PRIMARY KEY, created_at TEXT, expires_at TEXT,
  status TEXT DEFAULT 'pending',                    -- pending|approved|rejected|expired
  decided_at TEXT
);
CREATE TABLE IF NOT EXISTS hypotheses(              -- AI hypothesis layer (V4/D34)
  id TEXT PRIMARY KEY, tier TEXT,                   -- north_star | short_term
  status TEXT,                                      -- proposed | active | retired
  created_at TEXT, activated_at TEXT, retired_at TEXT,
  thesis TEXT,                                      -- markdown, human-readable
  stances TEXT,                                     -- JSON [{symbol,direction,conviction,horizon_days,rationale}]
  watchlist TEXT,                                   -- JSON [symbols beyond config universe]
  invalidation TEXT,                                -- what would falsify this
  regime TEXT,                                      -- regime at generation time
  source TEXT,                                      -- ai | human
  parent_id TEXT                                    -- hypothesis it replaced
);
CREATE TABLE IF NOT EXISTS steering(                -- non-blocking strategic choices (V4/D34)
  id TEXT PRIMARY KEY, kind TEXT, created_at TEXT, expires_at TEXT,
  title TEXT, context TEXT,                         -- markdown shown to the human
  options TEXT,                                     -- JSON [{key,label,detail}]
  recommended TEXT,                                 -- option key the AI recommends
  default_on_expiry TEXT,                           -- adopt | status_quo
  status TEXT DEFAULT 'pending',                    -- pending | decided | expired
  payload TEXT,                                     -- JSON kind-specific apply data
  decided_key TEXT, decided_at TEXT, decided_via TEXT  -- gui | expiry
);
CREATE TABLE IF NOT EXISTS instruments(             -- official listing catalog
  symbol TEXT PRIMARY KEY, name TEXT, exchange TEXT, security_type TEXT,
  is_etf INTEGER DEFAULT 0, is_adr INTEGER DEFAULT 0, active INTEGER DEFAULT 1,
  first_seen TEXT, last_seen TEXT, source TEXT, cik TEXT, raw_hash TEXT
);
CREATE TABLE IF NOT EXISTS universe_membership(     -- point-in-time tiers
  as_of TEXT, symbol TEXT, tier TEXT, rank INTEGER, reason TEXT, metrics TEXT,
  PRIMARY KEY(as_of, symbol, tier)
);
CREATE TABLE IF NOT EXISTS filing_facts(            -- SEC facts keyed by availability
  cik TEXT, tag TEXT, period_end TEXT, filed TEXT, value REAL, unit TEXT,
  form TEXT, accession TEXT,
  PRIMARY KEY(cik, tag, period_end, filed, accession)
);
CREATE TABLE IF NOT EXISTS graph_versions(          -- immutable champion/challengers
  id TEXT PRIMARY KEY, created_at TEXT, data_as_of TEXT, status TEXT,
  parent_id TEXT, topology TEXT, metrics TEXT, checkpoint TEXT
);
CREATE TABLE IF NOT EXISTS model_runs(              -- global + holding TCNs
  id TEXT PRIMARY KEY, kind TEXT, symbol TEXT, created_at TEXT, data_as_of TEXT,
  status TEXT, parent_id TEXT, metrics TEXT, checkpoint TEXT, feature_hash TEXT
);
CREATE TABLE IF NOT EXISTS model_forecasts(         -- shadow/live learning labels
  model_id TEXT, as_of TEXT, symbol TEXT, horizon INTEGER,
  q10 REAL, q50 REAL, q90 REAL, probability_positive REAL,
  resolved_at TEXT, realized_excess REAL, feature_hash TEXT,
  PRIMARY KEY(model_id, as_of, symbol, horizon)
);
CREATE TABLE IF NOT EXISTS equity_intraday(         -- throttled live marks (V4)
  ts TEXT, equity REAL, cash REAL, source TEXT,
  pnl REAL                                          -- realized+unrealized (D36):
);                                                  -- deposit-independent P&L
CREATE INDEX IF NOT EXISTS idx_audit_cycle ON audit(cycle_id);
CREATE INDEX IF NOT EXISTS idx_orders_dup ON orders(symbol, side, created_at);
CREATE INDEX IF NOT EXISTS idx_trades_analog ON trades(source, regime, score_bucket);
CREATE INDEX IF NOT EXISTS idx_signals_cycle ON signals(cycle_id);
CREATE INDEX IF NOT EXISTS idx_universe_tier ON universe_membership(as_of, tier, rank);
CREATE INDEX IF NOT EXISTS idx_forecasts_unresolved ON model_forecasts(resolved_at, horizon);
CREATE INDEX IF NOT EXISTS idx_filing_fact_asof ON filing_facts(cik, tag, filed);
"""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Store:
    """One connection PER THREAD (threading.local): the FastAPI threadpool
    fires many handlers concurrently, and a shared sqlite3 connection
    interleaves cursors under load (manifested as random 500s with
    JSONDecodeError on empty rows). WAL mode makes concurrent readers safe."""

    def __init__(self, path: str | Path):
        import threading
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.db.executescript(SCHEMA)
        # migration for DBs created before positions.mode existed
        try:
            self.db.execute("ALTER TABLE positions ADD COLUMN mode TEXT DEFAULT 'paper'")
            self.db.commit()
        except sqlite3.OperationalError:
            pass    # column already there
        # same migration for orders (D26): the only pre-migration rows that must
        # be 'live' are those that reached a live broker — those carry a
        # broker_order_id (paper fills never do), so backfill from that. Getting
        # this right keeps a resting live order visible to live-mode reconcile.
        try:
            self.db.execute("ALTER TABLE orders ADD COLUMN mode TEXT DEFAULT 'paper'")
            self.db.execute("UPDATE orders SET mode='live' "
                            "WHERE broker_order_id IS NOT NULL AND broker_order_id != ''")
            self.db.commit()
        except sqlite3.OperationalError:
            pass    # column already there
        try:    # D36: deposit-independent P&L on intraday marks
            self.db.execute("ALTER TABLE equity_intraday ADD COLUMN pnl REAL")
            self.db.commit()
        except sqlite3.OperationalError:
            pass    # column already there

    @property
    def db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False,
                                   timeout=15)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=15000")
            self._local.conn = conn
        return conn

    # ---------- audit ----------
    def audit(self, event_type: str, payload: Any = None, cycle_id: str = "") -> None:
        ts = _now()
        body = json.dumps(payload, default=str)
        self.db.execute(
            "INSERT INTO audit(ts, cycle_id, event_type, payload) VALUES(?,?,?,?)",
            (ts, cycle_id, event_type, body))
        self.db.commit()
        if any(isinstance(h, RotatingFileHandler) for h in _AUDIT_LOG.handlers):
            _AUDIT_LOG.info(json.dumps({"ts": ts, "cycle_id": cycle_id,
                                        "event": event_type, "payload": payload},
                                       default=str))

    def audit_rows(self, cycle_id: str | None = None, limit: int = 500) -> list[dict]:
        q, args = "SELECT * FROM audit", []
        if cycle_id:
            q += " WHERE cycle_id=?"; args.append(cycle_id)
        q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
        return [dict(r) for r in self.db.execute(q, args)]

    # ---------- kv ----------
    def kv_get(self, key: str, default=None):
        r = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def kv_set(self, key: str, value) -> None:
        self.db.execute("INSERT INTO kv(key,value) VALUES(?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, json.dumps(value, default=str)))
        self.db.commit()

    # ---------- bars ----------
    def upsert_bars(self, symbol: str, rows: list[dict], source: str) -> int:
        """rows: [{d, open, high, low, close, volume}]. Returns inserted/updated count."""
        now = _now()
        self.db.executemany(
            "INSERT INTO bars(symbol,d,open,high,low,close,volume,source,ingested_at) "
            "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,d) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, "
            "close=excluded.close, volume=excluded.volume, source=excluded.source, "
            "ingested_at=excluded.ingested_at",
            [(symbol, r["d"], r["open"], r["high"], r["low"], r["close"],
              r.get("volume", 0), source, now) for r in rows])
        self.db.commit()
        return len(rows)

    def get_bars(self, symbol: str, as_of: str, lookback: int = 400) -> list[dict]:
        """Bars with d <= as_of, most recent `lookback`, ascending. The ONLY bar
        read path — as_of filtering here is the lookahead guard."""
        rows = self.db.execute(
            "SELECT * FROM (SELECT * FROM bars WHERE symbol=? AND d<=? "
            "ORDER BY d DESC LIMIT ?) ORDER BY d ASC",
            (symbol, as_of, lookback)).fetchall()
        return [dict(r) for r in rows]

    def latest_bar_date(self, symbol: str) -> Optional[str]:
        r = self.db.execute("SELECT MAX(d) m FROM bars WHERE symbol=?", (symbol,)).fetchone()
        return r["m"]

    # ---------- signals / candidates ----------
    def record_signal(self, sig, cycle_id: str) -> None:
        from .models import new_id
        self.db.execute(
            "INSERT INTO signals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (new_id(), cycle_id, _now(), sig.node_id, sig.symbol, sig.direction,
             sig.score, sig.confidence, sig.horizon_days, sig.expected_return,
             sig.expected_volatility, sig.downside_estimate, json.dumps(sig.evidence)))
        self.db.commit()

    def record_candidate(self, cand, cycle_id: str) -> None:
        from .models import to_json_dict
        self.db.execute(
            "INSERT OR REPLACE INTO candidates VALUES(?,?,?,?,?,?)",
            (cand.id, cycle_id, _now(), cand.symbol, cand.final_score,
             json.dumps(to_json_dict(cand))))
        self.db.commit()

    # ---------- orders / fills ----------
    def record_order(self, o, mode: str = "paper") -> bool:
        """False if idempotency key already exists (duplicate)."""
        try:
            self.db.execute(
                "INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (o.id, o.candidate_id, "", o.symbol, o.asset_type, o.side, o.qty,
                 o.limit_price, o.notional, o.idempotency_key, o.status,
                 o.broker_order_id, o.option_symbol, o.created_at, _now(), mode))
            self.db.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def update_order(self, order_id: str, **fields) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE orders SET {sets}, updated_at=? WHERE id=?",
                        (*fields.values(), _now(), order_id))
        self.db.commit()

    def orders_today(self, side: str | None = None, day: str | None = None,
                     mode: str | None = None) -> list[dict]:
        """day: ISO date; defaults to the real today (backtester passes as_of).
        mode: paper|live — paper and live share this DB, so daily caps must
        count only same-mode orders (D26)."""
        day = day or date.today().isoformat()
        # 'localtime': created_at carries a tz offset; bare date() would shift
        # evening orders to the next UTC day and break daily caps after ~5pm PT
        q, args = "SELECT * FROM orders WHERE date(created_at,'localtime')=?", [day]
        if side:
            q += " AND side=?"; args.append(side)
        if mode:
            q += " AND mode=?"; args.append(mode)
        return [dict(r) for r in self.db.execute(q, args)]

    def recent_order_exists(self, symbol: str, side: str, cooldown_min: int,
                            now_iso: str | None = None,
                            mode: str | None = None) -> bool:
        now = now_iso or datetime.now().astimezone().isoformat()
        # coarse indexed prefilter on the raw string (ISO dates compare fine at
        # day granularity), then exact datetime() check on the survivors —
        # datetime() on both sides because created_at is ISO-with-tz while
        # datetime() yields space-separated UTC (raw compare would misorder)
        coarse = (datetime.fromisoformat(now) - timedelta(days=2)).date().isoformat()
        q = ("SELECT COUNT(*) c FROM orders WHERE symbol=? AND side=? "
             "AND created_at >= ? "
             "AND status NOT IN ('rejected','vetoed','expired','cancelled') "
             "AND datetime(created_at) >= datetime(?, ?)")
        args = [symbol, side, coarse, now, f"-{cooldown_min} minutes"]
        if mode:                              # D26: don't let a paper order block a live entry
            q += " AND mode=?"; args.append(mode)
        r = self.db.execute(q, args).fetchone()
        return r["c"] > 0

    def record_fill(self, f) -> None:
        self.db.execute("INSERT INTO fills VALUES(?,?,?,?,?,?,?)",
                        (f.order_id, f.symbol, f.side, f.qty, f.price, f.fees, f.filled_at))
        self.db.commit()

    # ---------- positions ----------
    def open_positions(self, mode: str | None = None) -> list[dict]:
        """mode filter matters: paper and live share this DB — a live scan
        must never mistake paper positions for real holdings."""
        q, args = "SELECT * FROM positions WHERE status='open'", []
        if mode:
            q += " AND mode=?"; args.append(mode)
        return [dict(r) for r in self.db.execute(q, args)]

    def save_position(self, pid: str, p: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO positions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, p["symbol"], p["asset_type"], p["qty"], p["avg_cost"], p["opened_at"],
             p["horizon_days"], p["stop_price"], p.get("candidate_id", ""),
             json.dumps(p.get("nodes", [])), p.get("option_symbol"),
             p.get("status", "open"), p.get("mode", "paper")))
        self.db.commit()

    def close_position(self, pid: str) -> None:
        self.db.execute("UPDATE positions SET status='closed' WHERE id=?", (pid,))
        self.db.commit()

    # ---------- trades (round-trips + backtest analogs) ----------
    def record_trade(self, t: dict) -> None:
        from .models import new_id
        self.db.execute(
            "INSERT INTO trades VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.get("id") or new_id(), t["symbol"], t.get("asset_type", "equity"),
             t["entry_date"], t["exit_date"], t["entry_price"], t["exit_price"],
             t["qty"], t["pnl"], t["ret"], t.get("horizon_days", 20),
             t.get("score", 0.0), t.get("score_bucket", ""), t.get("regime", ""),
             json.dumps(t.get("nodes", [])), t.get("source", "paper"),
             t.get("exit_reason", "")))
        self.db.commit()

    def analog_returns(self, score_bucket: str, regime: str,
                       sources: tuple = ("backtest", "paper", "live")) -> list[float]:
        """Horizon returns of historical trades in the same (score bucket, regime)
        cell — the raw material for bootstrap error bars (dev/DECISIONS.md D10)."""
        ph = ",".join("?" * len(sources))
        rows = self.db.execute(
            f"SELECT ret FROM trades WHERE score_bucket=? AND regime=? AND source IN ({ph})",
            (score_bucket, regime, *sources)).fetchall()
        return [r["ret"] for r in rows]

    def trades(self, source: str | None = None, limit: int = 10000) -> list[dict]:
        q, args = "SELECT * FROM trades", []
        if source:
            q += " WHERE source=?"; args.append(source)
        q += " ORDER BY exit_date DESC LIMIT ?"; args.append(limit)
        return [dict(r) for r in self.db.execute(q, args)]

    # ---------- equity curve / pnl ----------
    def record_equity(self, equity: float, cash: float, source: str, d: str | None = None) -> None:
        self.db.execute("INSERT OR REPLACE INTO equity_curve VALUES(?,?,?,?,?)",
                        (d or date.today().isoformat(), _now(), equity, cash, source))
        self.db.commit()

    def equity_curve(self, source: str, limit: int = 3650) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM equity_curve WHERE source=? ORDER BY d ASC LIMIT ?",
            (source, limit))]

    def peak_equity(self, source: str, since_d: str = "") -> float:
        """High-water mark, optionally only since a reset date (drawdown
        kill-switch baseline resets on clear — see risk.py D17)."""
        r = self.db.execute("SELECT MAX(equity) m FROM equity_curve "
                            "WHERE source=? AND d>=?", (source, since_d)).fetchone()
        return r["m"] or 0.0

    def equity_on(self, source: str, d: str) -> Optional[float]:
        r = self.db.execute(
            "SELECT equity FROM equity_curve WHERE source=? AND d<=? ORDER BY d DESC LIMIT 1",
            (source, d)).fetchone()
        return r["equity"] if r else None

    # ---------- weights ----------
    def get_weight_multiplier(self, node_id: str) -> float:
        r = self.db.execute("SELECT multiplier FROM weights WHERE node_id=?",
                            (node_id,)).fetchone()
        return r["multiplier"] if r else 1.0

    def set_weight_multiplier(self, node_id: str, mult: float, note: str = "") -> None:
        self.db.execute(
            "INSERT INTO weights VALUES(?,?,?,?) ON CONFLICT(node_id) DO UPDATE SET "
            "multiplier=excluded.multiplier, updated_at=excluded.updated_at, note=excluded.note",
            (node_id, mult, _now(), note))
        self.db.commit()

    # ---------- approvals ----------
    def queue_approval(self, intent_id: str, expires_at: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO approvals VALUES(?,?,?, 'pending', NULL)",
                        (intent_id, _now(), expires_at))
        self.db.commit()

    def pending_approvals(self) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT a.*, o.symbol, o.side, o.qty, o.limit_price, o.notional "
            "FROM approvals a JOIN orders o ON o.id=a.intent_id WHERE a.status='pending'")]

    def decide_approval(self, intent_id: str, status: str) -> None:
        # Guard here, not in the callers: both the GUI and the CLI route
        # through this method, and an intent approved after its expires_at
        # would otherwise be placed at a stale price on the next cycle
        # (the cycle-start expiry sweep only looks at 'pending' rows).
        if status == "approved":
            row = self.db.execute("SELECT expires_at FROM approvals WHERE intent_id=?",
                                  (intent_id,)).fetchone()
            if row and row["expires_at"] < _now():
                self.db.execute("UPDATE approvals SET status='expired', decided_at=? "
                                "WHERE intent_id=?", (_now(), intent_id))
                self.db.execute("UPDATE orders SET status='expired', updated_at=? "
                                "WHERE id=?", (_now(), intent_id))
                self.db.commit()
                raise ValueError(f"intent {intent_id} expired at {row['expires_at']}; "
                                 "wait for the next scan to re-propose it")
        self.db.execute("UPDATE approvals SET status=?, decided_at=? WHERE intent_id=?",
                        (status, _now(), intent_id))
        self.db.commit()

    # ---------- hypotheses (V4/D34) ----------
    def save_hypothesis(self, h: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO hypotheses VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (h["id"], h["tier"], h.get("status", "proposed"), h.get("created_at", _now()),
             h.get("activated_at"), h.get("retired_at"), h.get("thesis", ""),
             json.dumps(h.get("stances", [])), json.dumps(h.get("watchlist", [])),
             h.get("invalidation", ""), h.get("regime", ""),
             h.get("source", "ai"), h.get("parent_id", "")))
        self.db.commit()

    def get_hypothesis(self, hid: str) -> Optional[dict]:
        r = self.db.execute("SELECT * FROM hypotheses WHERE id=?", (hid,)).fetchone()
        return dict(r) if r else None

    def active_hypothesis(self, tier: str, as_of: str | None = None) -> Optional[dict]:
        """Point-in-time read (ROADMAP rule 3): only a hypothesis activated on
        or before as_of exists for a scan at as_of — backtests can't see the
        future, and live just passes today."""
        q = "SELECT * FROM hypotheses WHERE tier=? AND status='active'"
        args: list = [tier]
        if as_of:
            # 'localtime' for the same reason as orders_today: activated_at
            # carries a tz offset and bare date() would shift evening
            # activations to the next UTC day
            q += " AND date(activated_at,'localtime') <= ?"; args.append(as_of)
        q += " ORDER BY activated_at DESC LIMIT 1"
        r = self.db.execute(q, args).fetchone()
        return dict(r) if r else None

    def hypotheses(self, tier: str | None = None, limit: int = 50) -> list[dict]:
        q, args = "SELECT * FROM hypotheses", []
        if tier:
            q += " WHERE tier=?"; args.append(tier)
        q += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
        return [dict(r) for r in self.db.execute(q, args)]

    def update_hypothesis(self, hid: str, **fields) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE hypotheses SET {sets} WHERE id=?",
                        (*fields.values(), hid))
        self.db.commit()

    # ---------- steering (V4/D34) ----------
    def save_steering(self, s: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO steering VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (s["id"], s["kind"], s.get("created_at", _now()), s["expires_at"],
             s.get("title", ""), s.get("context", ""),
             json.dumps(s.get("options", [])), s.get("recommended", ""),
             s.get("default_on_expiry", "status_quo"), s.get("status", "pending"),
             json.dumps(s.get("payload", {})),
             s.get("decided_key"), s.get("decided_at"), s.get("decided_via")))
        self.db.commit()

    def get_steering(self, sid: str) -> Optional[dict]:
        r = self.db.execute("SELECT * FROM steering WHERE id=?", (sid,)).fetchone()
        return dict(r) if r else None

    def steering_requests(self, status: str | None = None, limit: int = 50) -> list[dict]:
        q, args = "SELECT * FROM steering", []
        if status:
            q += " WHERE status=?"; args.append(status)
        q += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
        return [dict(r) for r in self.db.execute(q, args)]

    def update_steering(self, sid: str, **fields) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE steering SET {sets} WHERE id=?",
                        (*fields.values(), sid))
        self.db.commit()

    # ---------- intraday equity marks (V4) ----------
    def record_intraday_mark(self, equity: float, cash: float, source: str,
                             pnl: float | None = None,
                             min_gap_min: int = 5) -> bool:
        """Throttled: skip if the last mark for this source is younger than
        min_gap_min. pnl = realized+unrealized trading P&L at mark time —
        deposit-independent (D36). Returns True if a mark was written."""
        r = self.db.execute("SELECT MAX(ts) m FROM equity_intraday WHERE source=?",
                            (source,)).fetchone()
        now = datetime.now().astimezone()
        if r["m"]:
            try:
                if (now - datetime.fromisoformat(r["m"])).total_seconds() < min_gap_min * 60:
                    return False
            except ValueError:
                pass
        self.db.execute("INSERT INTO equity_intraday VALUES(?,?,?,?,?)",
                        (now.isoformat(timespec="seconds"), equity, cash, source, pnl))
        self.db.commit()
        return True

    def intraday_marks(self, source: str, since_ts: str = "") -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM equity_intraday WHERE source=? AND ts>=? ORDER BY ts ASC",
            (source, since_ts))]

    # ---------- ai ledger ----------
    def ai_spend_today(self) -> float:
        r = self.db.execute("SELECT COALESCE(SUM(cost_usd),0) s FROM ai_ledger "
                            "WHERE day=date('now','localtime')").fetchone()
        return r["s"]

    def ai_spend_month(self, purpose: str | None = None) -> float:
        """Calendar-month spend, optionally per purpose (D36 monthly caps)."""
        q = ("SELECT COALESCE(SUM(cost_usd),0) s FROM ai_ledger "
             "WHERE day LIKE strftime('%Y-%m', 'now', 'localtime') || '%'")
        args: list = []
        if purpose:
            q += " AND purpose=?"; args.append(purpose)
        return self.db.execute(q, args).fetchone()["s"]

    def ai_log(self, model: str, purpose: str, node_id: str, in_tok: int, out_tok: int,
               cost: float, cache_hit: bool, ok: bool) -> None:
        self.db.execute(
            "INSERT INTO ai_ledger(ts,day,model,purpose,node_id,in_tokens,out_tokens,"
            "cost_usd,cache_hit,ok) VALUES(?,date('now','localtime'),?,?,?,?,?,?,?,?)",
            (_now(), model, purpose, node_id, in_tok, out_tok, cost,
             int(cache_hit), int(ok)))
        self.db.commit()
