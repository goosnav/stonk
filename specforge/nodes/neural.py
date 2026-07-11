"""Global/holding TCN signal node; one specialist in the analog graph."""
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
        if ctx.offline:
            return []                      # backtests: silent, lookahead-clean
        from .. import neural
        preds, meta = neural.predict_today(ctx.cfg, ctx.store, ctx)
        if not preds:
            self.degraded_reason = meta.get("silent")
            return []
        events = []
        for sym, forecast in preds.items():
            view = forecast.get(str(self.horizon_days)) or forecast.get("21")
            if not view:
                continue
            pred = float(view["q50"])
            if abs(pred) < 0.01:
                continue
            direction = "long" if pred > 0 else "avoid"
            score = min(1.0, abs(pred) / 0.08)
            conf = min(0.9, max(0.2, float(view.get("probability_positive", .5))
                                if pred > 0 else 1 - float(view.get("probability_positive", .5))))
            vol = (ctx.atr_pct(sym) or 0.02) * math.sqrt(self.horizon_days)
            events.append(SignalEvent(
                symbol=sym, direction=direction,
                score=round(score, 4), confidence=round(conf, 3),
                horizon_days=self.horizon_days,
                expected_return=round(float(pred), 5),
                expected_volatility=round(vol, 5),
                downside_estimate=round(float(view["q10"]), 5),
                evidence=[f"TCN {pred:+.2%} median {self.horizon_days}d excess "
                          f"({float(view['q10']):+.2%}…{float(view['q90']):+.2%}) · "
                          f"champion {meta.get('model_id')} · "
                          f"ckpt {meta.get('checkpoint_age_days')}d old"],
                data_as_of=datetime.strptime(ctx.as_of, "%Y-%m-%d"),
                node_id=self.id, node_version=self.version))
        return events
