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


def _valid_forecast(nf, expected_as_of: str, expected_model_id: str) -> bool:
    """Fail-closed forecast admission (C2 audit): a forecast may influence this
    cycle only if it is a real typed NeuralForecast, produced FOR this cycle's
    as_of, by the model identity the node reported, under the current feature
    schema. Anything else — stale cache, foreign model, malformed value — is
    silently inert (deterministic behaviour unchanged)."""
    from .. import neural
    return (isinstance(nf, NeuralForecast)
            and nf.as_of == expected_as_of
            and bool(expected_model_id)
            and nf.model_id == expected_model_id
            and nf.feature_schema_hash == neural.FEATURE_HASH)


def effective_blend(cfg, graph_blend: float, forecasts_available: bool,
                    permitted: float | None = None) -> tuple[float, str]:
    """(blend, reason). Bounded by [min_blend, max_blend]; a configured value
    below the floor means OFF (never silently raised), above the cap is clamped
    down (never silently increased past max_blend). `permitted` is the serving
    model's lifecycle-granted ceiling (None for a full champion): a ramp-state
    model can never exceed the blend recorded at its lifecycle transition."""
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
    b = min(b, hi)
    if permitted is not None:
        if permitted <= 0:
            return 0.0, "serving model has no permitted lifecycle blend"
        if permitted < b:
            return permitted, f"capped at lifecycle permitted blend {permitted}"
    return b, "active"


def apply_neural_blend(candidates, forecasts, cfg, store, cycle_id,
                       graph_blend: float, *, as_of: str, meta: dict) -> dict:
    """Blend calibrated neural scores into candidate final_score, in place.

    `forecasts` is predict_today's {symbol: {horizon: NeuralForecast}}; `as_of`
    is the CURRENT cycle date and `meta` the node's inference metadata. Every
    forecast is re-validated against both (stale/foreign/malformed → inert).
    Every touched candidate records blend + contribution (persisted via
    record_candidate) so influence is visible, auditable, and attributable.
    """
    cost = ml_targets.round_trip_cost(cfg)
    expected_model = str((meta or {}).get("model_id") or "")
    blend, reason = effective_blend(cfg, graph_blend, bool(forecasts),
                                    permitted=(meta or {}).get("permitted_blend"))
    scored = rejected = 0
    for c in candidates:
        hs = (forecasts or {}).get(c.symbol) or {}
        nf = hs.get(str(c.horizon_days)) or hs.get("21")
        if nf is None:
            continue
        if not _valid_forecast(nf, as_of, expected_model):
            rejected += 1
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
               "rejected_invalid": rejected, "model_id": expected_model,
               "as_of": as_of, "candidates": len(candidates)}
    store.audit("neural_direct_blend", summary, cycle_id)
    return summary


def select_exploration_probe(candidates, targets, forecasts, meta, cfg, store,
                             cycle_id, account, ctx, *, as_of: str,
                             allocated: float = 0.0):
    """Pick at most ONE bounded neural exploration probe, or None.

    The probe must already be a vetted candidate (never fabricated from a raw
    forecast), must not be held or normally selected, and must clear the
    absolute-edge, probability, and uncertainty gates on a validated forecast.
    Sizing runs through portfolio.construct() — the same liquidity/scenario
    checks as a normal entry — with size_multiplier applied there exactly once,
    then is capped by the exploration budget fraction and remaining headroom.
    The governor downstream remains the final authority on every limit.
    """
    ex = cfg.get("neural", "exploration", default={}) or {}
    if not ex.get("enabled", False) or not forecasts:
        return None
    cost = ml_targets.round_trip_cost(cfg)
    expected_model = str((meta or {}).get("model_id") or "")
    min_edge = float(ex.get("min_absolute_edge", 0.0075))
    min_prob = float(ex.get("min_probability", 0.57))
    max_unc = float(ex.get("max_uncertainty", 0.15))
    multiplier = max(0.0, min(1.0, float(ex.get("size_multiplier", 0.25))))
    budget_frac = max(0.0, min(1.0, float(ex.get("budget_fraction", 0.20))))
    max_open = int(ex.get("max_open_positions", 1))

    mode = "live" if cfg.mode == "live" else "paper"
    open_probes = store.db.execute(
        "SELECT COUNT(*) n FROM positions WHERE status='open' AND mode=? "
        "AND entry_mode='probe'", (mode,)).fetchone()["n"]
    if open_probes >= max_open:
        store.audit("neural_probe_skipped", {"reason": "probe slot occupied",
                    "open_probes": open_probes}, cycle_id)
        return None
    held = {p.symbol for p in account.positions if p.qty > 0}
    if len(held) >= int(cfg.get("risk", "max_open_positions", default=12)):
        return None                        # global position cap — never exceeded

    selected = {c.symbol for c, _ in targets}
    eligible = []
    for c in candidates:
        if c.side != "buy" or c.symbol in held or c.symbol in selected:
            continue
        hs = (forecasts or {}).get(c.symbol) or {}
        nf = hs.get(str(c.horizon_days)) or hs.get("21")
        if not _valid_forecast(nf, as_of, expected_model):
            continue
        edge = nf.absolute_edge_after_cost(cost)
        if (edge < min_edge
                or nf.probability_absolute_edge_positive < min_prob
                or (nf.absolute_q90 - nf.absolute_q10) > max_unc):
            continue
        eligible.append((neural_score(nf, cost), c, nf, edge))
    if not eligible:
        return None
    _, cand, nf, edge = max(eligible, key=lambda t: t[0])

    cand.entry_mode, cand.size_multiplier = "probe", multiplier
    cand.entry_mode_reason = (f"neural exploration: model {nf.model_id} "
                              f"abs_edge {edge:+.4f} "
                              f"P {nf.probability_absolute_edge_positive:.2f}")
    from .. import portfolio
    sized = portfolio.construct([cand], account, ctx, cfg)
    if not sized:
        cand.entry_mode, cand.size_multiplier = "normal", 1.0
        cand.entry_mode_reason = None
        return None
    spendable = min(max(0.0, account.cash), max(0.0, account.buying_power))
    headroom = max(0.0, spendable - float(allocated))
    notional = round(min(sized[0][1], budget_frac * account.equity, headroom), 2)
    if notional < 5.0:
        cand.entry_mode, cand.size_multiplier = "normal", 1.0
        cand.entry_mode_reason = None
        return None
    cand.target_notional, cand.max_loss = notional, notional
    store.record_candidate(cand, cycle_id)     # persist probe attribution
    store.audit("neural_probe_selected", {
        "symbol": cand.symbol, "notional": notional, "model_id": nf.model_id,
        "absolute_q50": nf.absolute_q50, "absolute_edge": round(edge, 5),
        "probability_absolute_edge_positive": nf.probability_absolute_edge_positive,
        "uncertainty": round(nf.absolute_q90 - nf.absolute_q10, 5),
        "size_multiplier": multiplier, "as_of": as_of}, cycle_id)
    return cand, notional
