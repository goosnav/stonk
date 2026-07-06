"""Insider buying node (AGENTS.md §10.9, disabled by default): open-market
cluster BUYS only — insider selling is noise, buying is signal.

Data: openinsider.com latest-cluster-buys page (pandas.read_html; unofficial,
degrades to silence). Cluster buys = multiple insiders purchasing the same
name within days, the strongest variant of the signal. Filing date is the
tradable date (already how openinsider lists them).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..data import MarketContext
from ..models import SignalEvent
from .base import SignalNode

URL = "http://openinsider.com/latest-cluster-buys"
CACHE_HOURS = 12
FRESH_DAYS = 14


class Node(SignalNode):
    version = "1"
    role = "alpha"

    def _cluster_buys(self, ctx: MarketContext) -> list[dict]:
        key = "insider_cluster_cache"
        cached = ctx.store.kv_get(key)
        fresh = cached and cached.get("at", "") >= \
            (datetime.now() - timedelta(hours=CACHE_HOURS)).isoformat()
        if ctx.offline or fresh:
            return (cached or {}).get("rows", [])
        try:
            import pandas as pd
            tables = pd.read_html(URL)
            big = max(tables, key=len)
            big.columns = [str(c).strip().lower().replace("\xa0", " ")
                           for c in big.columns]
            rows = []
            for _, r in big.iterrows():
                tick = str(r.get("ticker", "")).strip().upper()
                filing = str(r.get("filing date", ""))[:10]
                try:
                    value = abs(float(str(r.get("value", "0")).replace("$", "")
                                      .replace(",", "").replace("+", "")))
                except ValueError:
                    value = 0.0
                insiders = str(r.get("insiders", r.get("#insiders", "1")))
                if tick:
                    rows.append({"ticker": tick, "filing_date": filing,
                                 "value": value, "insiders": insiders})
            ctx.store.kv_set(key, {"at": datetime.now().isoformat(), "rows": rows})
            return rows
        except Exception as e:                              # noqa: BLE001
            self.degraded_reason = f"openinsider fetch failed: {e}"
            ctx.store.kv_set(key, {"at": datetime.now().isoformat(),
                                   "rows": (cached or {}).get("rows", []),
                                   "failed": str(e)})
            return (cached or {}).get("rows", [])

    def compute(self, ctx: MarketContext) -> list[SignalEvent]:
        rows = self._cluster_buys(ctx)
        if not rows:
            return []
        as_of = datetime.strptime(ctx.as_of, "%Y-%m-%d")
        cutoff = (as_of - timedelta(days=FRESH_DAYS)).date().isoformat()
        events = []
        for r in rows:
            sym = r["ticker"]
            if sym not in ctx.universe or not (cutoff <= r["filing_date"] <= ctx.as_of):
                continue
            # scale by disclosed dollar value (log-ish buckets)
            v = r["value"]
            score = 0.3 if v < 1e5 else 0.5 if v < 5e5 else 0.8 if v < 5e6 else 1.0
            vol = (ctx.atr_pct(sym) or 0.02) * (self.horizon_days ** 0.5)
            events.append(SignalEvent(
                symbol=sym, direction="long", score=round(score, 4),
                confidence=0.5, horizon_days=self.horizon_days,
                expected_return=round(score * vol * 0.3, 5),
                expected_volatility=round(vol, 5),
                downside_estimate=round(-2 * vol, 5),
                evidence=[f"insider cluster buy ${v:,.0f} filed {r['filing_date']}"],
                data_as_of=as_of, node_id=self.id, node_version=self.version))
        return events
