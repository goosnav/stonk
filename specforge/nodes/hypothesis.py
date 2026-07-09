"""Hypothesis node (V4/D34) — the ONLY path a hypothesis influences trading.

Deterministic: reads the stored ACTIVE short-term hypothesis (as-of correct)
and emits its stances as SignalEvents into the ensemble, where they are
weighted, conflict-penalized, attribution-measured, and governor-gated exactly
like every other node. NO AI call happens here — generation is a separate
post-close step (specforge/hypothesis.py). Degrades to silence when the
feature is disabled or no hypothesis is active.
"""
from __future__ import annotations

import json
import math
from datetime import datetime

from ..data import MarketContext
from ..models import SignalEvent
from .base import SignalNode


class Node(SignalNode):
    version = "1"
    role = "alpha"

    def compute(self, ctx: MarketContext) -> list[SignalEvent]:
        if not ctx.cfg.get("hypothesis", "enabled", default=False):
            return []
        h = ctx.store.active_hypothesis("short_term", as_of=ctx.as_of)
        if not h:
            return []
        events = []
        for s in json.loads(h.get("stances") or "[]"):
            sym = s["symbol"]
            if ctx.close(sym) is None:
                continue                       # no data yet (fresh watchlist add)
            conviction = float(s["conviction"])
            if conviction <= 0:
                continue
            horizon = int(s.get("horizon_days", self.horizon_days))
            vol = (ctx.atr_pct(sym) or 0.02) * math.sqrt(horizon)
            score = conviction if s["direction"] == "long" else -conviction
            events.append(SignalEvent(
                symbol=sym, direction=s["direction"],
                score=round(score, 4), confidence=round(conviction, 3),
                horizon_days=horizon,
                expected_return=round(score * vol * 0.5, 5),
                expected_volatility=round(vol, 5),
                downside_estimate=round(-2 * vol, 5),
                evidence=[f"hypothesis {h['id'][:8]}: {s.get('rationale', '')[:120]}"],
                data_as_of=datetime.strptime(ctx.as_of, "%Y-%m-%d"),
                node_id=self.id, node_version=self.version))
        return events
