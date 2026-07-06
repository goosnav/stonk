"""Congressional trading node (AGENTS.md §10.8) — slow thematic tilt, never a
primary trigger (small default weight, disabled by default).

Data: Capitol Trades public BFF endpoint (unofficial, may break — node degrades
to silence). Lookahead discipline: signals key on the PUBLICATION date (when
the disclosure became public), never the transaction date. Disclosures arrive
up to 45 days late; whatever edge remains is thematic, so horizon is long
(60d) and scores are modest.

Scoring: recent (≤14d since publication) BUY disclosures for a universe ticker,
weighted by disclosed size bucket and cluster count (multiple politicians
buying the same name is the actual signal).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import httpx

from ..data import MarketContext
from ..models import SignalEvent
from .base import SignalNode

BFF_URL = "https://bff.capitoltrades.com/trades?pageSize=100&txType=buy"
CACHE_HOURS = 12
FRESH_DAYS = 14
SIZE_SCORE = {  # capitol trades size buckets → weight
    "1K–15K": 0.2, "15K–50K": 0.4, "50K–100K": 0.6, "100K–250K": 0.8,
    "250K–500K": 0.9, "500K–1M": 1.0, "1M–5M": 1.0, "5M–25M": 1.0,
}


class Node(SignalNode):
    version = "1"
    role = "alpha"

    def _disclosures(self, ctx: MarketContext) -> list[dict]:
        key = "congress_trades_cache"
        cached = ctx.store.kv_get(key)
        fresh = cached and cached.get("at", "") >= \
            (datetime.now() - timedelta(hours=CACHE_HOURS)).isoformat()
        if ctx.offline or fresh:
            return (cached or {}).get("rows", [])
        try:
            r = httpx.get(BFF_URL, timeout=20,
                          headers={"User-Agent": "specforge/0.1"})
            r.raise_for_status()
            rows = []
            for t in (r.json().get("data") or []):
                ticker = ((t.get("issuer") or {}).get("issuerTicker") or "")
                rows.append({
                    "ticker": ticker.split(":")[0],       # "NVDA:US" → "NVDA"
                    "pub_date": str(t.get("pubDate", ""))[:10],
                    "size": t.get("size") or t.get("sizeRangeHigh") or "",
                    "politician": (t.get("politician") or {}).get("lastName", ""),
                })
            ctx.store.kv_set(key, {"at": datetime.now().isoformat(), "rows": rows})
            return rows
        except Exception as e:                              # noqa: BLE001
            self.degraded_reason = f"capitol trades fetch failed: {e}"
            ctx.store.kv_set(key, {"at": datetime.now().isoformat(),
                                   "rows": (cached or {}).get("rows", []),
                                   "failed": str(e)})
            return (cached or {}).get("rows", [])

    def compute(self, ctx: MarketContext) -> list[SignalEvent]:
        rows = self._disclosures(ctx)
        if not rows:
            return []
        as_of = datetime.strptime(ctx.as_of, "%Y-%m-%d")
        cutoff = (as_of - timedelta(days=FRESH_DAYS)).date().isoformat()

        by_ticker: dict[str, list[dict]] = {}
        for r in rows:
            # pub_date <= as_of is the lookahead guard (filing-date discipline)
            if r["ticker"] in ctx.universe and cutoff <= r["pub_date"] <= ctx.as_of:
                by_ticker.setdefault(r["ticker"], []).append(r)

        events = []
        for sym, ds in by_ticker.items():
            size_w = max(SIZE_SCORE.get(str(d["size"]), 0.3) for d in ds)
            cluster = len({d["politician"] for d in ds})
            score = min(1.0, size_w * (0.5 + 0.25 * cluster))
            vol = (ctx.atr_pct(sym) or 0.02) * (self.horizon_days ** 0.5)
            events.append(SignalEvent(
                symbol=sym, direction="long", score=round(score, 4),
                confidence=0.4,               # deliberately humble
                horizon_days=self.horizon_days,
                expected_return=round(score * vol * 0.25, 5),
                expected_volatility=round(vol, 5),
                downside_estimate=round(-2 * vol, 5),
                evidence=[f"{cluster} politician buy disclosure(s) ≤{FRESH_DAYS}d, "
                          f"latest {max(d['pub_date'] for d in ds)}"],
                data_as_of=as_of, node_id=self.id, node_version=self.version))
        return events
