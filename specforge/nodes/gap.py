"""Gap-continuation node (D40): the day-trader staple that fits a 10-minute
cycle. An opening gap up of 1.5–6% in an uptrending name tends to continue
over the next sessions; giant gaps (>6%) are news blowoffs we don't chase.

Live-quote driven (ctx.live_px, set by the engine each live/paper cycle) —
daily bars can't see gaps intraday, so this node is silent in backtests and
must earn its keep through measured paper/live trades. Hence: experimental,
small weight, short horizon.
"""
from __future__ import annotations

import math
from datetime import datetime

from ..data import MarketContext
from ..models import SignalEvent
from .base import SignalNode

GAP_MIN, GAP_MAX = 0.015, 0.06


class Node(SignalNode):
    version = "1"
    role = "alpha"

    def compute(self, ctx: MarketContext) -> list[SignalEvent]:
        live = getattr(ctx, "live_px", None) or {}
        if not live:
            return []                      # backtest or no quotes: silent
        events = []
        for sym in ctx.universe:
            px = live.get(sym)
            c = ctx.closes(sym)
            if not px or len(c) < 60:
                continue
            # last SETTLED close: drop today's bar if the feed already has it
            prev = c.iloc[-2] if c.index[-1] == ctx.as_of else c.iloc[-1]
            gap = px / float(prev) - 1
            sma50 = c.rolling(50).mean().iloc[-1]
            if not (GAP_MIN <= gap <= GAP_MAX) or px < sma50:
                continue
            atr = ctx.atr_pct(sym) or 0.02
            gap_atr = gap / atr           # gap strength in daily-range units
            score = math.tanh(gap_atr / 3) * 0.8
            vol = atr * math.sqrt(self.horizon_days)
            events.append(SignalEvent(
                symbol=sym, direction="long",
                score=round(score, 4), confidence=0.4,
                horizon_days=self.horizon_days,
                expected_return=round(score * vol * 0.5, 5),
                expected_volatility=round(vol, 5),
                downside_estimate=round(-2 * vol, 5),
                evidence=[f"gap {gap:+.1%} ({gap_atr:.1f}×ATR) above 50sma, "
                          f"prev close {float(prev):.2f} → {px:.2f}"],
                data_as_of=datetime.strptime(ctx.as_of, "%Y-%m-%d"),
                node_id=self.id, node_version=self.version))
        return events
