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
CBOE_HISTORY = {
    "^VIX9D": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "^VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "^VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "^VIX6M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv",
    "^VVIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
}


def fetch_cboe_history(symbol: str) -> list[dict]:
    """Official Cboe daily history for volatility indices."""
    url = CBOE_HISTORY.get(symbol)
    if not url:
        return []
    response = httpx.get(url, timeout=30, headers={"User-Agent": "stonk-terminal/0.1"})
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    df.columns = [str(c).strip().upper() for c in df.columns]
    date_col = "DATE" if "DATE" in df else df.columns[0]
    close_col = "CLOSE" if "CLOSE" in df else symbol.lstrip("^")
    if close_col not in df:
        return []
    out = []
    for _, row in df.iterrows():
        close = pd.to_numeric(row.get(close_col), errors="coerce")
        if pd.isna(close):
            continue
        d = pd.to_datetime(row[date_col], errors="coerce")
        if pd.isna(d):
            continue
        out.append({"d": d.strftime("%Y-%m-%d"),
                    "open": float(pd.to_numeric(row.get("OPEN"), errors="coerce"))
                    if pd.notna(pd.to_numeric(row.get("OPEN"), errors="coerce")) else float(close),
                    "high": float(pd.to_numeric(row.get("HIGH"), errors="coerce"))
                    if pd.notna(pd.to_numeric(row.get("HIGH"), errors="coerce")) else float(close),
                    "low": float(pd.to_numeric(row.get("LOW"), errors="coerce"))
                    if pd.notna(pd.to_numeric(row.get("LOW"), errors="coerce")) else float(close),
                    "close": float(close), "volume": 0.0})
    return out


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
    # Yahoo spells listed share classes BRK-B/AGM-A while official catalogs
    # and brokers use BRK.B/AGM.A.
    yahoo_symbol = symbol if symbol.startswith("^") else symbol.replace(".", "-")
    df = yf.Ticker(yahoo_symbol).history(period=period, interval="1d", auto_adjust=True)
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
            if sym in CBOE_HISTORY:
                rows, source = fetch_cboe_history(sym), "cboe"
                if not rows:
                    rows, source = fetch_yfinance(sym, period="1mo"), "yfinance"
            elif latest and not full and latest >= (date.today() - timedelta(days=30)).isoformat():
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
                 offline: bool = False, historical: bool = False,
                 symbols: list[str] | None = None):
        self.store = store
        self.cfg = cfg
        self.as_of = as_of or date.today().isoformat()
        # offline=True (backtests): nodes must serve external data (earnings,
        # fundamentals) from kv caches only — never fetch mid-simulation
        self.offline = offline
        # Historical replay additionally enforces source-availability time.
        # Cache-only live discovery is offline but is not a replay.
        self.historical = historical
        # Explicit cycle-local universe (Sprint E1): callers pass the symbols
        # they mean instead of mutating cfg.data — shared config stays
        # immutable during execution, so state cannot leak between cycles.
        self._symbols = list(symbols) if symbols is not None else None
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def universe(self) -> list[str]:
        if self._symbols is not None:
            return list(self._symbols)
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

    def volatility_context(self) -> dict:
        symbols = self.cfg.get("universe", "volatility_symbols", default={}) or {}
        values = {name: self.close(symbol) for name, symbol in symbols.items()}
        spot, short, m3, m6 = (values.get("vix"), values.get("vix9d"),
                               values.get("vix3m"), values.get("vix6m"))
        values["slope_9d_3m"] = ((short / m3 - 1) if short and m3 else None)
        values["slope_1m_3m"] = ((spot / m3 - 1) if spot and m3 else None)
        values["slope_3m_6m"] = ((m3 / m6 - 1) if m3 and m6 else None)
        vix_closes = self.closes(symbols.get("vix", "^VIX"), 10)
        values["vix_change_5d"] = (float(vix_closes.iloc[-1] / vix_closes.iloc[-6] - 1)
                                    if len(vix_closes) >= 6 else None)
        bench = self.closes(self.cfg.get("universe", "benchmark", default="SPY"), 30)
        realized = float(bench.pct_change().dropna().tail(21).std() * (252 ** .5) * 100) \
            if len(bench) >= 22 else None
        values["realized_vol_21d"] = realized
        values["implied_realized_spread"] = spot - realized \
            if spot is not None and realized is not None else None
        return values

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
