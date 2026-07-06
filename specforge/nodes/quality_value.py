"""Quality/value filter node (AGENTS.md §10.5) — guardrail, not alpha. Vetoes
buying fundamentally broken names during hype; never generates signals.

Data: yfinance .info snapshot, cached 7 days. ETFs/indices pass automatically.
On data failure the filter PASSES (fail-open): a missing fundamentals feed
should not halt an otherwise deterministic pipeline — the flag shows in the
GUI via degraded_reason instead.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..data import MarketContext
from .base import SignalNode

ETFISH = {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
          "XLI", "XLU", "XLB", "XLRE", "XLC"}
CACHE_DAYS = 7


class Node(SignalNode):
    version = "1"
    role = "filter"

    def compute(self, ctx: MarketContext):      # filter nodes emit no signals
        return []

    def _fundamentals(self, ctx: MarketContext, sym: str) -> dict | None:
        key = f"fundamentals_{sym}"
        cached = ctx.store.kv_get(key)
        if ctx.offline:                                   # backtest: cache only
            return cached["data"] if cached else None
        if cached and cached.get("fetched_at", "") >= \
                (datetime.now() - timedelta(days=CACHE_DAYS)).isoformat():
            return cached["data"]
        try:
            import yfinance as yf
            info = yf.Ticker(sym).info or {}
            data = {k: info.get(k) for k in
                    ("grossMargins", "returnOnEquity", "debtToEquity",
                     "freeCashflow", "totalRevenue")}
            ctx.store.kv_set(key, {"fetched_at": datetime.now().isoformat(), "data": data})
            return data
        except Exception as e:                  # noqa: BLE001
            self.degraded_reason = f"fundamentals fetch failed: {e}"
            data = cached["data"] if cached else None
            ctx.store.kv_set(key, {"fetched_at": datetime.now().isoformat(),
                                   "data": data, "failed": str(e)})   # cache failure
            return data

    def passes(self, ctx: MarketContext, symbol: str) -> bool:
        if symbol in ETFISH or symbol.startswith("^"):
            return True
        f = self._fundamentals(ctx, symbol)
        if not f:
            return True                          # fail-open (see module docstring)
        checks = [
            f.get("grossMargins") is None or f["grossMargins"] > 0.05,
            f.get("returnOnEquity") is None or f["returnOnEquity"] > -0.10,
            f.get("debtToEquity") is None or f["debtToEquity"] < 400,
        ]
        return all(checks)
