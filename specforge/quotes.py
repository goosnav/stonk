"""QuoteService — live-ish quotes with provenance (PRODUCT.md pillar 4).

Provider chain (configs data_sources.quotes, default [broker, stooq,
yfinance]): each provider fills whatever symbols the previous ones missed.
Every quote carries {price, change_pct, as_of, source} — the GUI must always
be able to say where a number came from and how old it is. 30s in-process
cache (single-process app; no persistence needed).

Providers:
- broker: real-time via the connected broker adapter (RH MCP when live).
- stooq:  delayed batch CSV, one request for many symbols, no key.
- yfinance: fallback + the only one that serves indices like ^VIX.
"""
from __future__ import annotations

import io
import time
from datetime import datetime

import httpx

STOOQ_LIGHT = "https://stooq.com/q/l/?s={syms}&f=sd2t2ohlcv&h&e=csv"
CACHE_TTL = 30.0


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class QuoteService:
    def __init__(self, cfg, broker=None):
        self.cfg = cfg
        self.broker = broker
        self.chain = cfg.get("data_sources", "quotes",
                             default=["broker", "stooq", "yfinance"])
        self._cache: dict[str, dict] = {}
        self._cached_at = 0.0

    def get(self, symbols: list[str]) -> dict[str, dict]:
        symbols = [s.upper() for s in symbols]
        if time.time() - self._cached_at > CACHE_TTL:
            self._cache = {}
        missing = [s for s in symbols if s not in self._cache]
        if missing:
            for provider in self.chain:
                still = [s for s in missing if s not in self._cache]
                if not still:
                    break
                try:
                    self._cache.update(getattr(self, f"_{provider}")(still))
                except Exception:                    # noqa: BLE001 — chain on
                    continue
            self._cached_at = time.time()
        return {s: self._cache[s] for s in symbols if s in self._cache}

    # ---------------- providers ----------------
    def _broker(self, symbols: list[str]) -> dict[str, dict]:
        # paper broker quotes are engine-injected daily closes — skip, they'd
        # masquerade as live; only a real broker feed counts here
        if self.broker is None or self.broker.name == "paper":
            return {}
        out = {}
        for sym, px in (self.broker.get_quotes(symbols) or {}).items():
            out[sym] = {"price": round(px, 4), "change_pct": None,
                        "as_of": _now(), "source": self.broker.name}
        return out

    def _stooq(self, symbols: list[str]) -> dict[str, dict]:
        plain = [s for s in symbols if not s.startswith("^")]
        if not plain:
            return {}
        syms = ",".join(s.lower().replace("-", ".") + ".us" for s in plain)
        r = httpx.get(STOOQ_LIGHT.format(syms=syms), timeout=15,
                      headers={"User-Agent": "stonk-terminal/0.1"})
        r.raise_for_status()
        import csv
        out = {}
        for row in csv.DictReader(io.StringIO(r.text)):
            sym = (row.get("Symbol") or "").replace(".US", "").upper()
            try:
                close = float(row["Close"])
                opn = float(row["Open"])
            except (KeyError, ValueError):
                continue                              # "N/D" rows for misses
            out[sym] = {"price": close,
                        "change_pct": round(close / opn - 1, 4) if opn else None,
                        "as_of": f"{row.get('Date', '')}T{row.get('Time', '')}",
                        "source": "stooq(delayed)"}
        return out

    def _yfinance(self, symbols: list[str]) -> dict[str, dict]:
        import yfinance as yf
        out = {}
        for sym in symbols:
            try:
                fi = yf.Ticker(sym).fast_info
                px = fi.get("lastPrice") or fi.get("last_price")
                prev = fi.get("previousClose") or fi.get("previous_close")
                if px:
                    out[sym] = {"price": round(float(px), 4),
                                "change_pct": round(float(px) / float(prev) - 1, 4)
                                if prev else None,
                                "as_of": _now(), "source": "yfinance"}
            except Exception:                         # noqa: BLE001
                continue
        return out
