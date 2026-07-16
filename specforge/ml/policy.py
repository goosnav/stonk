"""Direct bounded neural influence (Stage C1).

The TCN's right to contribute does NOT depend on the analog graph. A valid,
fresh champion earns a fixed bounded blend of the final candidate score:

    final = (1 − b) · deterministic + b · neural_score

The graph competes against this fixed blend as a separate meta-model: when the
graph blend is active it OWNS the learned pathway and the direct blend stands
down, so neural influence is never double-counted. When the model is
unavailable, stale, or invalid there are no forecasts and b is 0 — the
deterministic ensemble is always the fallback. Scoring only: exits, kill
switches, and the governor are untouched downstream.
"""
from __future__ import annotations

import math

from . import targets as ml_targets
from .schema import NeuralForecast


def neural_score(f: NeuralForecast, cost: float) -> float:
    """Bounded [-1, 1] trade score with economically correct semantics:
    absolute edge after cost leads, direction confidence and cross-sectional
    (excess) confirmation follow, wide intervals shrink reliability."""
    absolute_edge = f.absolute_edge_after_cost(cost)
    direction_confidence = 2.0 * f.probability_absolute_edge_positive - 1.0
    relative_confirmation = math.tanh(f.excess_q50 / 0.04)
    raw = (0.50 * math.tanh(absolute_edge / 0.04)
           + 0.30 * direction_confidence
           + 0.20 * relative_confirmation)
    uncertainty = max(0.01, f.absolute_q90 - f.absolute_q10)
    reliability = min(1.0, 0.08 / uncertainty)
    return max(-1.0, min(1.0, raw * reliability))


def effective_blend(cfg, graph_blend: float, forecasts_available: bool) -> tuple[float, str]:
    """(blend, reason). Bounded by [min_blend, max_blend]; a configured value
    below the floor means OFF (never silently raised), above the cap is clamped
    down (never silently increased past max_blend)."""
    if not forecasts_available:
        return 0.0, "model unavailable/stale/invalid — deterministic fallback"
    if graph_blend > 0:
        return 0.0, "graph meta-model active — owns the learned pathway"
    b = float(cfg.get("neural", "experimental_blend", default=0.15))
    lo = float(cfg.get("neural", "min_blend", default=0.05))
    hi = float(cfg.get("neural", "max_blend", default=0.40))
    if b <= 0:
        return 0.0, "blend disabled by config"
    if b < lo:
        return 0.0, f"configured blend {b} below min_blend {lo} — treated as off"
    return min(b, hi), "active"


def apply_neural_blend(candidates, forecasts, cfg, store, cycle_id,
                       graph_blend: float) -> dict:
    """Blend calibrated neural scores into candidate final_score, in place.

    `forecasts` is predict_today's {symbol: {horizon: NeuralForecast}}. Every
    touched candidate records blend + contribution (persisted via
    record_candidate) so influence is visible, auditable, and attributable.
    """
    cost = ml_targets.round_trip_cost(cfg)
    blend, reason = effective_blend(cfg, graph_blend, bool(forecasts))
    scored = 0
    for c in candidates:
        hs = (forecasts or {}).get(c.symbol) or {}
        nf = hs.get(str(c.horizon_days)) or hs.get("21")
        if nf is None:
            continue
        score = neural_score(nf, cost)
        c.neural_blend = blend
        if blend:
            prior = c.final_score
            c.final_score = round((1 - blend) * prior + blend * score, 4)
            c.neural_contribution = round(c.final_score - prior, 6)
            c.contributing_nodes = sorted(set(c.contributing_nodes + ["neural_direct"]))
            c.thesis = (c.thesis + f"; neural_direct:{score:+.3f}@{blend:.0%}")[:400]
        scored += 1
    summary = {"blend": blend, "reason": reason, "scored": scored,
               "candidates": len(candidates)}
    store.audit("neural_direct_blend", summary, cycle_id)
    return summary
