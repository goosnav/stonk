"""Data ingestion + MarketContext.

Sources (dev/DECISIONS.md D9): Stooq CSV (free, full daily history in one request)
primary; yfinance fallback and for index symbols like ^VIX that Stooq lacks.

MarketContext is the ONLY way pipeline code sees market data, and it slices
everything to `as_of` (D8): the same context class powers live scans and the
backtest, which is what makes lookahead impossible by construction.
"""
from __future__ import annotations

import io
import time
from datetime import datetime, date, timedelta
from functools import lru_cache

import httpx
import pandas as pd

from .store import Store

STOOQ_URL = "https://stooq.com/q/d/l/?s={sym}&i=d"


def _stooq_symbol(symbol: str) -> str | None:
    if symbol.startswith("^"):          # indices: not reliable on stooq → yfinance
        return None
    return symbol.lower().replace("-", ".") + ".us"


def fetch_stooq(symbol: str) -> list[dict]:
    ssym = _stooq_symbol(symbol)
    if not ssym:
        return []
    r = httpx.get(STOOQ_URL.format(sym=ssym), timeout=30,
                  headers={"User-Agent": "stonk-terminal/0.1"})
    r.raise_for_status()
    text = r.text
    if not text.startswith("Date,"):     # stooq returns junk/limits page on miss
        return []
    df = pd.read_csv(io.StringIO(text))
    return [{"d": row.Date, "open": row.Open, "high": row.High, "low": row.Low,
             "close": row.Close, "volume": getattr(row, "Volume", 0) or 0}
            for row in df.itertuples() if pd.notna(row.Close)]


def fetch_yfinance(symbol: str, period: str = "max") -> list[dict]:
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
    if df.empty:
        return []
    out = []
    for idx, row in df.iterrows():
        if pd.isna(row["Close"]):
            continue
        out.append({"d": idx.strftime("%Y-%m-%d"), "open": float(row["Open"]),
                    "high": float(row["High"]), "low": float(row["Low"]),
                    "close": float(row["Close"]), "volume": float(row.get("Volume", 0) or 0)})
    return out


def refresh(store: Store, symbols: list[str], full: bool = False,
            log=print) -> dict[str, int]:
    """Bring bars up to date. Incremental (yfinance 1mo) when we already have
    recent history; full-history (stooq→yfinance) on first sight or `full=True`.
    Returns {symbol: rows_written}. Failures are logged, not raised — staleness
    is enforced later by the risk governor, not by crashing the scan."""
    results = {}
    for sym in symbols:
        try:
            latest = store.latest_bar_date(sym)
            if latest and not full and latest >= (date.today() - timedelta(days=30)).isoformat():
                rows = fetch_yfinance(sym, period="1mo")
                source = "yfinance"
            else:
                rows, source = fetch_stooq(sym), "stooq"
                if not rows:
                    rows, source = fetch_yfinance(sym), "yfinance"
            results[sym] = store.upsert_bars(sym, rows, source) if rows else 0
            if not rows:
                log(f"data: no rows for {sym}")
            time.sleep(0.2)  # ponytail: fixed politeness delay; backoff if sources complain
        except Exception as e:                      # noqa: BLE001 — per-symbol isolation
            log(f"data: {sym} failed: {e}")
            results[sym] = 0
    return results


class MarketContext:
    """As-of view over stored bars. All pipeline reads go through here."""

    def __init__(self, store: Store, cfg, as_of: str | None = None,
                 offline: bool = False):
        self.store = store
        self.cfg = cfg
        self.as_of = as_of or date.today().isoformat()
        # offline=True (backtests): nodes must serve external data (earnings,
        # fundamentals) from kv caches only — never fetch mid-simulation
        self.offline = offline
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def universe(self) -> list[str]:
        return list(self.cfg.get("universe", "symbols", default=[]))

    def df(self, symbol: str, lookback: int = 400) -> pd.DataFrame:
        """OHLCV DataFrame indexed by date string, rows <= as_of only."""
        key = f"{symbol}:{lookback}"
        if key not in self._cache:
            rows = self.store.get_bars(symbol, self.as_of, lookback)
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.set_index("d")[["open", "high", "low", "close", "volume"]]
            self._cache[key] = df
        return self._cache[key]

    def close(self, symbol: str) -> float | None:
        df = self.df(symbol)
        return float(df["close"].iloc[-1]) if len(df) else None

    def closes(self, symbol: str, lookback: int = 400) -> pd.Series:
        df = self.df(symbol, lookback)
        return df["close"] if len(df) else pd.Series(dtype=float)

    def prices(self) -> dict[str, float]:
        """Last known close per universe symbol (for marking/sizing)."""
        out = {}
        for s in self.universe:
            c = self.close(s)
            if c:
                out[s] = c
        return out

    def atr_pct(self, symbol: str, n: int = 14) -> float | None:
        """ATR as fraction of price — the vol-sizing input."""
        df = self.df(symbol)
        if len(df) < n + 1:
            return None
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(n).mean().iloc[-1]
        return float(atr / df["close"].iloc[-1])

    def data_age_days(self, symbol: str) -> int | None:
        """Calendar days between as_of and the newest bar (staleness check)."""
        df = self.df(symbol)
        if df.empty:
            return None
        last = datetime.strptime(df.index[-1], "%Y-%m-%d").date()
        cur = datetime.strptime(self.as_of, "%Y-%m-%d").date()
        return (cur - last).days

    def vix(self) -> float | None:
        return self.close(self.cfg.get("universe", "vix_symbol", default="^VIX"))

    def breadth_above_sma(self, n: int = 50) -> float | None:
        """Fraction of universe trading above its n-day SMA (regime input)."""
        above = total = 0
        for s in self.universe:
            c = self.closes(s)
            if len(c) < n:
                continue
            total += 1
            if c.iloc[-1] > c.rolling(n).mean().iloc[-1]:
                above += 1
        return above / total if total else None
