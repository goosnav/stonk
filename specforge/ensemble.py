"""Evidence-family ensemble used by live, paper, CLI, and replay.

Direction owns sign. Families own fixed portions of the production vote, so a
missing AI read cannot silently hand its 50% allocation to momentum. Learned
node multipliers operate *inside* a family and cannot change family budgets.
"""
from __future__ import annotations

import statistics

from .data import MarketContext
from .models import (SignalEvent, TradeCandidate, direction_sign, new_id,
                     signed_alpha)
from .store import Store


ETF_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV",
               "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"}

NODE_FAMILY = {
    "business_fundamentals": "business",
    "company_catalyst": "catalyst",
    "quality_value": "quality",
    "sector_rotation": "context",
    "macro_regime": "context",
    "momentum": "price",
    "reversal": "price",
    "gap": "price",
    "vol_contraction": "price",
}

DEFAULT_COMPANY_FAMILIES = {
    "business": .30, "catalyst": .20, "quality": .15,
    "context": .15, "price": .20,
}
DEFAULT_ETF_FAMILIES = {"context": .50, "price": .50}
REGIME_ACTIVATION = {"risk_on": .65, "neutral": .10,
                     "risk_off": -.50, "stress": -1.0}


def _family_weights(cfg, is_etf: bool) -> dict[str, float]:
    key = "etf_families" if is_etf else "company_families"
    defaults = DEFAULT_ETF_FAMILIES if is_etf else DEFAULT_COMPANY_FAMILIES
    raw = cfg.get("evidence", key, default=defaults) or defaults
    weights = {k: max(0.0, float(v)) for k, v in raw.items()}
    total = sum(weights.values())
    return ({k: v / total for k, v in weights.items()} if total else defaults)


def _detail(event: SignalEvent, family: str, node_weight: float) -> dict:
    return {
        "node": event.node_id, "family": family,
        "direction": event.direction, "magnitude": round(event.score, 6),
        "confidence": round(event.confidence, 6),
        "signed_alpha": round(signed_alpha(event), 6),
        "node_weight": round(node_weight, 6),
        "data_as_of": event.data_as_of.isoformat(),
        "evidence": list(event.evidence), "state": "running",
    }


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
    snapshots = {}
    for symbol in sorted(set(by_symbol) | set(ctx.universe)):
        sigs = by_symbol.get(symbol, [])
        is_etf = symbol in ETF_SYMBOLS
        family_weights = _family_weights(cfg, is_etf)
        grouped: dict[str, list[tuple[SignalEvent, float]]] = {}
        details: list[dict] = []
        for s in sigs:
            family = NODE_FAMILY.get(s.node_id)
            if not family or family not in family_weights:
                continue
            w = s_node_weight(s.node_id, cfg, store, regime)
            if w <= 0:
                continue
            grouped.setdefault(family, []).append((s, w))
            details.append(_detail(s, family, w))

        # Deterministic SEC quality is a scored family input as well as the
        # existing guardrail. It is evaluated once here and is point-in-time.
        for f in filters:
            event = f.graph_signal(ctx, symbol) if hasattr(f, "graph_signal") else None
            if event and "quality" in family_weights:
                w = max(.01, float(cfg.get("evidence", "quality_node_weight", default=1.0)))
                grouped.setdefault("quality", []).append((event, w))
                details.append(_detail(event, "quality", w))

        # Regime is always explicit evidence rather than an invisible global
        # switch. It shares the context family with sector-relative evidence.
        if "context" in family_weights:
            macro = REGIME_ACTIVATION.get(regime, 0.0)
            details.append({"node": "macro_regime", "family": "context",
                            "direction": "long" if macro >= 0 else "avoid",
                            "magnitude": abs(macro), "confidence": 1.0,
                            "signed_alpha": macro, "node_weight": 1.0,
                            "data_as_of": ctx.as_of,
                            "evidence": [f"market regime {regime}"], "state": "running"})
            grouped.setdefault("context", []).append((SignalEvent(
                symbol=symbol, direction="long" if macro >= 0 else "avoid",
                score=abs(macro), confidence=1.0, horizon_days=21,
                expected_return=0.0, expected_volatility=0.0,
                downside_estimate=0.0, evidence=[f"market regime {regime}"],
                data_as_of=__import__("datetime").datetime.strptime(ctx.as_of, "%Y-%m-%d"),
                node_id="macro_regime"), 1.0))

        family_scores: dict[str, float] = {}
        expected_parts: list[tuple[float, float]] = []
        for family, family_weight in family_weights.items():
            members = grouped.get(family, [])
            if not members:
                details.append({"node": family, "family": family, "state": "unavailable",
                                "signed_alpha": 0.0, "family_contribution": 0.0,
                                "evidence": ["no current evidence"]})
                continue
            denom = sum(w for _, w in members) or 1.0
            value = sum(signed_alpha(event) * w for event, w in members) / denom
            family_scores[family] = value
            for event, w in members:
                expected_parts.append((direction_sign(event.direction) *
                                       abs(event.expected_return),
                                       family_weight * w * event.confidence / denom))
            for detail in details:
                if detail.get("family") == family and detail.get("state") == "running":
                    detail["family_weight"] = family_weight
            details.append({"node": f"family:{family}", "family": family,
                            "state": "running", "signed_alpha": round(value, 6),
                            "family_weight": family_weight,
                            "family_contribution": round(family_weight * value, 6),
                            "evidence": [f"{len(members)} evidence node(s)"]})

        raw = sum(family_weights[f] * value for f, value in family_scores.items())
        coverage = sum(family_weights[f] for f in family_scores)

        # conflict penalty: dispersion of per-node directional scores
        per_node = [d["signed_alpha"] for d in details
                    if d.get("state") == "running" and not d.get("node", "").startswith("family:")]
        dispersion = statistics.pstdev(per_node) if len(per_node) > 1 else 0.0
        conflict = max(0.0, 1.0 - disp_pen * dispersion)
        production_score = raw * conflict
        # horizon/expectation: confidence-weighted average of contributing nodes
        longs = [s for s in sigs if s.direction in ("long", "long_call")]
        if raw <= 0 or not longs:
            details.append({"node": "strategy_context", "family": "strategy",
                            "state": "blocked", "signed_alpha": 0.0,
                            "family_contribution": 0.0,
                            "evidence": ["strategy cannot create a trade without positive evidence"]})
            snapshots[symbol] = {
                "score_before_conflict": round(raw, 6), "conflict": round(conflict, 6),
                "production_score": round(production_score, 6),
                "strategy_contribution": 0.0, "score": round(production_score, 6),
                "coverage": round(coverage, 4),
                "families": {k: round(v, 6) for k, v in family_scores.items()},
                "evidence": details}
            continue                          # MVP trades the long side only
        horizon = round(sum(s.horizon_days for s in longs) / len(longs))
        exp_denom = sum(w for _, w in expected_parts)
        exp_gross = (sum(value * w for value, w in expected_parts) / exp_denom
                     if exp_denom else 0.0)
        cost_penalty = friction / max(abs(exp_gross), 0.005)   # cap blowup on tiny edges
        from .strategy import contribution as strategy_contribution
        strategy = strategy_contribution(cfg, store, symbol)
        strategy_value = float(strategy.get("value", 0))
        min_underlying = float(cfg.get("strategy", "min_underlying_score", default=.10))
        min_coverage = float(cfg.get("strategy", "min_evidence_coverage", default=.50))
        if production_score < min_underlying or coverage < min_coverage or \
                exp_gross - friction <= 0:
            strategy_value = 0.0
            if strategy.get("state") == "running":
                strategy["reason"] = "underlying evidence/coverage/after-cost gate did not pass"
        details.append({"node": "strategy_context", "family": "strategy",
                        "state": strategy.get("state", "unavailable"),
                        "signed_alpha": round(strategy_value, 6),
                        "family_contribution": round(strategy_value, 6),
                        "evidence": [strategy.get("reason", "no active AI strategy")],
                        "mandate_id": strategy.get("mandate_id")})
        final = production_score + strategy_value - cost_penalty * 0.01
        snapshots[symbol] = {"score_before_conflict": round(raw, 6),
                             "conflict": round(conflict, 6),
                             "production_score": round(production_score, 6),
                             "strategy_contribution": round(strategy_value, 6),
                             "score": round(final, 6),
                             "coverage": round(coverage, 4),
                             "families": {k: round(v, 6)
                                          for k, v in family_scores.items()},
                             "evidence": details}

        if final < min_score:
            continue
        if any(not f.passes(ctx, symbol) for f in filters):
            continue

        vol = statistics.median([s.expected_volatility for s in longs]) if longs else 0.05
        candidates.append(TradeCandidate(
            id=new_id(), symbol=symbol, asset_type="equity", side="buy",
            thesis="; ".join(
                f"{d['node']} {d.get('signed_alpha', 0):+.2f}: "
                f"{(d.get('evidence') or [''])[0]}" for d in details
                if d.get("state") == "running" and not d.get("node", "").startswith("family:"))[:400],
            final_score=round(final, 4), target_notional=0.0,
            expected_return=round(exp_gross - friction, 5),
            ci_low=0.0, ci_high=0.0, probability_positive=0.0,   # forecast.py fills
            expected_apr=0.0, apr_ci_low=0.0, apr_ci_high=0.0,
            horizon_days=horizon, max_loss=0.0,
            contributing_nodes=sorted({d["node"] for d in details
                                       if d.get("state") == "running" and
                                       not d.get("node", "").startswith("family:")}),
            regime=regime,
            evidence_version="evidence.v2",
            evidence_coverage=round(coverage, 4),
            evidence_details=details,
            production_score=round(production_score, 6),
            strategy_contribution=round(strategy_value, 6),
            strategy_mandate_id=strategy.get("mandate_id"),
        ))
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    store.kv_set("evidence_last_scores", {"schema": "evidence.v2", "as_of": ctx.as_of,
                                           "regime": regime, "symbols": snapshots})
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
