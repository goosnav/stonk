"""Global/holding TCN signal node; one specialist in the analog graph.

Trade eligibility keys off the ABSOLUTE forecast after modeled cost — a stock
that merely beats a falling benchmark (positive excess, negative absolute) is
not a long. The graph ranks cross-sectionally, so the event's score/confidence
carry the EXCESS component; expected_return carries the absolute economics.
"""
from __future__ import annotations

import math
from datetime import datetime

from ..data import MarketContext
from ..models import SignalEvent
from ..ml import targets as ml_targets
from .base import SignalNode


class Node(SignalNode):
    version = "2"                          # v2: dual-target absolute/excess semantics
    role = "alpha"

    def compute(self, ctx: MarketContext) -> list[SignalEvent]:
        # Fail-closed forecast cache (C2 audit): cleared BEFORE anything can
        # raise or early-return, so a prior cycle's forecasts can never leak
        # into this cycle's blend or probe. The as_of stamp binds the stash to
        # exactly one cycle; consumers must match it.
        self.last_forecasts, self.last_meta = {}, {}
        self.last_forecast_as_of = None
        if ctx.offline:
            return []                      # backtests: silent, lookahead-clean
        from .. import neural
        preds, meta = neural.predict_today(ctx.cfg, ctx.store, ctx)
        # Stash for the engine's direct blend (ml/policy.py) — one inference
        # pass per cycle, consumed twice.
        self.last_forecasts, self.last_meta = preds, meta
        self.last_forecast_as_of = ctx.as_of
        if not preds:
            self.degraded_reason = meta.get("silent")
            for symbol in ctx.universe:
                if not symbol.startswith("^"):
                    self.symbol_states[symbol] = "unavailable"
            return []
        cost = ml_targets.round_trip_cost(ctx.cfg)
        min_edge = float(ctx.cfg.get("nodes", "neural", "min_absolute_edge", default=0.0))
        min_prob = float(ctx.cfg.get("nodes", "neural", "min_probability", default=0.5))
        events = []
        for symbol in ctx.universe:
            if not symbol.startswith("^"):
                self.symbol_states[symbol] = "verified_neutral"
        for sym, forecast in preds.items():
            nf = forecast.get(str(self.horizon_days)) or forecast.get("21")
            if nf is None:
                continue
            absolute_edge = nf.absolute_edge_after_cost(cost)   # abs q50 − cost
            long_ok = (absolute_edge > min_edge and
                       nf.probability_absolute_edge_positive >= min_prob)
            if long_ok:
                direction = "long"
            elif nf.absolute_q50 < -0.005 and nf.excess_q50 < 0:
                direction = "avoid"          # bearish on BOTH — a genuine avoid
            else:
                # No actionable absolute edge. Crucially, +excess/−absolute lands
                # here: it is neither a long nor a misleading bearish graph vote.
                continue
            # score/confidence are the cross-sectional (excess) component the graph
            # ranks on; expected_return is the absolute economics the sizer uses.
            score = min(1.0, abs(nf.excess_q50) / 0.06)
            conf = min(0.9, max(0.2, nf.probability_excess_positive if direction == "long"
                                else 1 - nf.probability_excess_positive))
            vol = (ctx.atr_pct(sym) or 0.02) * math.sqrt(self.horizon_days)
            events.append(SignalEvent(
                symbol=sym, direction=direction,
                score=round(score, 4), confidence=round(conf, 3),
                horizon_days=self.horizon_days,
                expected_return=round(nf.absolute_q50, 5),
                expected_volatility=round(vol, 5),
                downside_estimate=round(nf.absolute_q10, 5),
                evidence=[f"TCN abs {nf.absolute_q50:+.2%} (edge {absolute_edge:+.2%} "
                          f"after {cost:.2%} cost, P={nf.probability_absolute_edge_positive:.0%}) · "
                          f"excess {nf.excess_q50:+.2%} · "
                          f"abs [{nf.absolute_q10:+.2%}…{nf.absolute_q90:+.2%}] · "
                          f"champion {meta.get('model_id')} · "
                          f"ckpt {meta.get('checkpoint_age_days')}d old"],
                data_as_of=datetime.strptime(ctx.as_of, "%Y-%m-%d"),
                node_id=self.id, node_version=self.version))
            self.symbol_states[sym] = "running"
        return events
