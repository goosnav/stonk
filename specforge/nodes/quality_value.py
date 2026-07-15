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
from ..models import SignalEvent
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
        point_in_time = self.graph_score(ctx, symbol)
        if point_in_time is not None:
            return point_in_time > -0.5
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

    def graph_score(self, ctx: MarketContext, symbol: str) -> float | None:
        """Auditable, point-in-time balance-sheet quality; no snapshot leakage."""
        inst = ctx.store.db.execute(
            "SELECT cik FROM instruments WHERE symbol=?", (symbol,)).fetchone()
        if not inst or not inst["cik"]:
            return None
        rows = ctx.store.db.execute(
            "SELECT tag,value,period_end,filed FROM filing_facts WHERE cik=? AND filed<=? "
            "AND tag IN ('Assets','StockholdersEquity','EarningsPerShareDiluted') "
            "ORDER BY filed DESC,period_end DESC", (str(inst["cik"]), ctx.as_of)).fetchall()
        latest = {}
        for row in rows:
            latest.setdefault(row["tag"], float(row["value"]))
        assets, equity, eps = (latest.get("Assets"), latest.get("StockholdersEquity"),
                               latest.get("EarningsPerShareDiluted"))
        evidence = []
        if assets and equity is not None:
            evidence.append(max(-1.0, min(1.0, (equity / assets - .2) / .3)))
        if eps is not None:
            evidence.append(.5 if eps > 0 else -.5)
        return sum(evidence) / len(evidence) if evidence else None

    def graph_signal(self, ctx: MarketContext, symbol: str) -> SignalEvent | None:
        score = self.graph_score(ctx, symbol)
        if score is None:
            return None
        as_of = datetime.strptime(ctx.as_of, "%Y-%m-%d")
        return SignalEvent(symbol=symbol, direction="long" if score >= 0 else "avoid",
                           score=round(abs(score), 4), confidence=.7, horizon_days=40,
                           expected_return=0, expected_volatility=0,
                           downside_estimate=0,
                           evidence=[f"point-in-time SEC quality {score:+.2f}"],
                           data_as_of=as_of, node_id=self.id, node_version=self.version)
