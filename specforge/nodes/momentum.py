"""Medium-term momentum node (AGENTS.md §10.1).

Components, each squashed to [-1, 1]:
- 12-1 momentum (252d return excluding most recent 21d) — the classic factor
- 3m and 1m returns
- price above 50d / 200d SMA
- relative strength vs benchmark over 3m

Expected return heuristic: score × horizon volatility × 0.5. That is a crude
prior deliberately — forecast.py replaces/shrinks it with measured analog-trade
outcomes as sample accumulates. The node's job is ranking, not calibration.
"""
from __future__ import annotations

import math

from ..data import MarketContext
from ..models import SignalEvent
from .base import SignalNode


def _squash(x: float, scale: float) -> float:
    return math.tanh(x / scale)


class Node(SignalNode):
    version = "1"
    role = "alpha"

    def compute(self, ctx: MarketContext) -> list[SignalEvent]:
        bench = ctx.cfg.get("universe", "benchmark", default="SPY")
        bench_closes = ctx.closes(bench)
        bench_r63 = (bench_closes.iloc[-1] / bench_closes.iloc[-64] - 1) \
            if len(bench_closes) > 64 else 0.0

        events = []
        for sym in ctx.universe:
            c = ctx.closes(sym)
            if len(c) < 260:
                continue
            px = c.iloc[-1]
            r21 = px / c.iloc[-22] - 1
            r63 = px / c.iloc[-64] - 1
            r12_1 = c.iloc[-22] / c.iloc[-253] - 1          # 12-1 momentum
            sma50 = c.rolling(50).mean().iloc[-1]
            sma200 = c.rolling(200).mean().iloc[-1]
            rel = r63 - bench_r63

            comps = {
                "mom_12_1": _squash(r12_1, 0.30),
                "mom_3m": _squash(r63, 0.15),
                "mom_1m": _squash(r21, 0.08),
                "above_sma50": 0.5 if px > sma50 else -0.5,
                "above_sma200": 0.5 if px > sma200 else -0.5,
                "rel_strength": _squash(rel, 0.10),
            }
            score = sum(comps.values()) / len(comps)
            if abs(score) < 0.10:
                continue                                     # no opinion, no noise
            agree = sum(1 for v in comps.values() if v * score > 0) / len(comps)
            vol = (ctx.atr_pct(sym) or 0.02) * math.sqrt(self.horizon_days)
            events.append(SignalEvent(
                symbol=sym, direction="long" if score > 0 else "avoid",
                score=round(abs(score), 4), confidence=round(agree, 3),
                horizon_days=self.horizon_days,
                expected_return=round(score * vol * 0.5, 5),
                expected_volatility=round(vol, 5),
                downside_estimate=round(-2 * vol, 5),
                evidence=[f"12-1={r12_1:+.1%} 3m={r63:+.1%} rel={rel:+.1%} "
                          f"{'>' if px > sma200 else '<'}sma200"],
                data_as_of=_as_of_dt(ctx), node_id=self.id, node_version=self.version))
        return events


def _as_of_dt(ctx):
    from datetime import datetime
    return datetime.strptime(ctx.as_of, "%Y-%m-%d")
