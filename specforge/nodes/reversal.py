"""Short-term reversal node (AGENTS.md §10.2): oversold snap-back in liquid
names — 5d return z-score + RSI(14) + Bollinger position, long side only, and
only when the name is above its 200d SMA (dislocation in an uptrend, not a
falling knife). Short horizon (5d), small default weight.
"""
from __future__ import annotations

import math
from datetime import datetime

from ..data import MarketContext
from ..models import SignalEvent
from .base import SignalNode


def _rsi(closes, n=14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(n).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(n).mean().iloc[-1]
    if loss == 0:
        return 100.0
    return 100 - 100 / (1 + gain / loss)


class Node(SignalNode):
    version = "1"
    role = "alpha"

    def compute(self, ctx: MarketContext) -> list[SignalEvent]:
        as_of = datetime.strptime(ctx.as_of, "%Y-%m-%d")
        events = []
        for sym in ctx.universe:
            c = ctx.closes(sym)
            if len(c) < 210:
                continue
            px = c.iloc[-1]
            if px < c.rolling(200).mean().iloc[-1]:
                continue                        # uptrend names only
            r5 = px / c.iloc[-6] - 1
            daily = c.pct_change().dropna()
            sigma5 = daily.rolling(60).std().iloc[-1] * math.sqrt(5)
            if not sigma5 or sigma5 <= 0:
                continue
            z = r5 / sigma5
            rsi = _rsi(c)
            sma20 = c.rolling(20).mean().iloc[-1]
            band = daily.rolling(20).std().iloc[-1] * math.sqrt(20) * px
            bb_pos = (px - sma20) / (2 * band) if band else 0.0

            # oversold composite: want z very negative, RSI low, below lower band
            if z > -1.5 or rsi > 40:
                continue
            score = min(1.0, (-z - 1.5) / 2 + (40 - rsi) / 80 + max(0.0, -bb_pos))
            vol = (ctx.atr_pct(sym) or 0.02) * math.sqrt(self.horizon_days)
            events.append(SignalEvent(
                symbol=sym, direction="long", score=round(score, 4),
                confidence=0.5, horizon_days=self.horizon_days,
                expected_return=round(min(0.5 * -r5, 2 * vol), 5),  # partial retrace
                expected_volatility=round(vol, 5),
                downside_estimate=round(-2.5 * vol, 5),   # knives do fall
                evidence=[f"5d={r5:+.1%} z={z:.1f} RSI={rsi:.0f} bb={bb_pos:+.2f}"],
                data_as_of=as_of, node_id=self.id, node_version=self.version))
        return events
