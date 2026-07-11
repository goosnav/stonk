"""Volatility-contraction breakout node (D40): VCP-lite. Tight ranges after
an advance mark supply absorption; a name coiling near its 60d high in an
uptrend tends to resolve upward. Daily-bars only → fully backtestable.

Conditions: 20d realized vol < 70% of its level 60 sessions ago, price
within 3% of the 60d high, above both 50d and 200d SMA. Score blends the
depth of the contraction with proximity to the pivot.
"""
from __future__ import annotations

import math
from datetime import datetime

from ..data import MarketContext
from ..models import SignalEvent
from .base import SignalNode


class Node(SignalNode):
    version = "1"
    role = "alpha"

    def compute(self, ctx: MarketContext) -> list[SignalEvent]:
        events = []
        for sym in ctx.universe:
            c = ctx.closes(sym)
            if len(c) < 260:
                continue
            r = c.pct_change()
            sd_now = float(r.iloc[-20:].std())
            sd_prev = float(r.iloc[-80:-60].std())
            if not sd_prev or sd_now / sd_prev >= 0.70:
                continue
            px = float(c.iloc[-1])
            hi60 = float(c.iloc[-60:].max())
            if px < hi60 * 0.97:
                continue
            if px < c.rolling(50).mean().iloc[-1] or px < c.rolling(200).mean().iloc[-1]:
                continue
            contraction = 1 - sd_now / sd_prev            # 0.3 … 1
            proximity = 1 - (hi60 - px) / (hi60 * 0.03)   # 0 … 1
            score = round(min(0.9, 0.5 * contraction + 0.4 * proximity), 4)
            vol = (ctx.atr_pct(sym) or 0.02) * math.sqrt(self.horizon_days)
            events.append(SignalEvent(
                symbol=sym, direction="long",
                score=score, confidence=0.5,
                horizon_days=self.horizon_days,
                expected_return=round(score * vol * 0.5, 5),
                expected_volatility=round(vol, 5),
                downside_estimate=round(-2 * vol, 5),
                evidence=[f"vol contraction {contraction:.0%} (sd20 {sd_now:.3f} "
                          f"vs {sd_prev:.3f}), {((px / hi60) - 1):+.1%} from 60d high"],
                data_as_of=datetime.strptime(ctx.as_of, "%Y-%m-%d"),
                node_id=self.id, node_version=self.version))
        return events
