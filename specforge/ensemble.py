"""Ensemble scorer — AGENTS.md §11 step 5.

raw_score(sym)   = Σ_i  effective_weight_i × score_i × confidence_i
conflict_penalty = 1 − dispersion_penalty × normalized score dispersion
cost_penalty     = estimated round-trip friction / |expected gross edge|
final_score      = raw_score × conflict_penalty − sign(raw)×cost_penalty

effective_weight_i = config base weight × learned multiplier (store.weights,
bounded by attribution layer) — this is where self-improvement feeds back in.

Filter-role nodes (quality_value) veto candidates; they don't add score.
Only 'long' direction produces buy candidates in MVP (no shorting). long_call/
long_put directions are handled by the options node itself (Phase 5) which
emits candidates with asset_type='option'.
"""
from __future__ import annotations

import statistics

from .data import MarketContext
from .models import SignalEvent, TradeCandidate, new_id
from .store import Store


def score(events: list[SignalEvent], regime: str, cfg, store: Store,
          filters: list, ctx: MarketContext) -> list[TradeCandidate]:
    by_symbol: dict[str, list[SignalEvent]] = {}
    for e in events:
        by_symbol.setdefault(e.symbol, []).append(e)

    min_score = cfg.get("ensemble", "min_final_score", default=0.15)
    disp_pen = cfg.get("ensemble", "conflict_dispersion_penalty", default=0.5)
    friction = (cfg.get("execution", "spread_cost_bps", default=3)
                + cfg.get("execution", "slippage_bps", default=5)) * 2 / 10000.0  # round trip

    candidates = []
    for symbol, sigs in by_symbol.items():
        # weighted directional score: long=+, avoid/hedge=-, short_bias=- (we
        # can't short; negative scores just suppress the long case)
        contribs, weights = [], []
        for s in sigs:
            w = s_node_weight(s.node_id, cfg, store, regime)
            if w <= 0:
                continue
            sign = {"long": 1, "long_call": 1}.get(s.direction, -1)
            contribs.append(sign * s.score * s.confidence * w)
            weights.append(w)
        if not weights:
            continue
        raw = sum(contribs) / sum(weights)

        # conflict penalty: dispersion of per-node directional scores
        per_node = [c / w for c, w in zip(contribs, weights)]
        dispersion = statistics.pstdev(per_node) if len(per_node) > 1 else 0.0
        conflict = max(0.0, 1.0 - disp_pen * dispersion)

        # horizon/expectation: confidence-weighted average of contributing nodes
        longs = [s for s in sigs if s.direction in ("long", "long_call")]
        if raw <= 0 or not longs:
            continue                          # MVP trades the long side only
        horizon = round(sum(s.horizon_days for s in longs) / len(longs))
        exp_gross = sum(s.expected_return * s.confidence for s in longs) / \
            max(1e-9, sum(s.confidence for s in longs))
        cost_penalty = friction / max(abs(exp_gross), 0.005)   # cap blowup on tiny edges
        final = raw * conflict - cost_penalty * 0.01           # cost term in score units

        if final < min_score:
            continue
        if any(not f.passes(ctx, symbol) for f in filters):
            continue

        vol = statistics.median([s.expected_volatility for s in longs]) if longs else 0.05
        candidates.append(TradeCandidate(
            id=new_id(), symbol=symbol, asset_type="equity", side="buy",
            thesis="; ".join(f"{s.node_id}:{s.evidence[0] if s.evidence else s.score:.2f}"
                             if not s.evidence else f"{s.node_id}: {s.evidence[0]}"
                             for s in longs)[:400],
            final_score=round(final, 4), target_notional=0.0,
            expected_return=round(exp_gross - friction, 5),
            ci_low=0.0, ci_high=0.0, probability_positive=0.0,   # forecast.py fills
            expected_apr=0.0, apr_ci_low=0.0, apr_ci_high=0.0,
            horizon_days=horizon, max_loss=0.0,
            contributing_nodes=sorted({s.node_id for s in longs}),
            regime=regime,
        ))
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates


def s_node_weight(node_id: str, cfg, store: Store, regime: str | None = None) -> float:
    base = float(cfg.get("nodes", node_id, "weight", default=0.0) or 0.0)
    # regime-conditioned multiplier replaces the global one when attribution
    # has a meaningful sample for this (node, regime) cell — see attribution.py
    if regime:
        rm = (store.kv_get("regime_multipliers", {}) or {}).get(node_id, {})
        if regime in rm:
            return base * rm[regime]
    return base * store.get_weight_multiplier(node_id)
