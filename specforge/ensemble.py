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
    "news_sentiment": "catalyst",
    "earnings_drift": "catalyst",
    "insider": "catalyst",
    "congress_trades": "catalyst",
    "quality_value": "quality",
    "sector_rotation": "context",
    "macro_regime": "context",
    "hypothesis": "context",
    "momentum": "price",
    "reversal": "price",
    "gap": "price",
    "vol_contraction": "price",
}
DEFAULT_NODE_PRIORS = {
    "business": {"business_fundamentals": 1.0},
    "catalyst": {"company_catalyst": .50, "earnings_drift": .25,
                 "news_sentiment": .10, "insider": .075,
                 "congress_trades": .075},
    "quality": {"quality_value": 1.0},
    "context": {"sector_rotation": .45, "macro_regime": .45,
                "hypothesis": .10},
    "price": {"momentum": .40, "reversal": .20, "gap": .15,
              "vol_contraction": .25},
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


def _node_priors(cfg, family: str, is_etf: bool) -> dict[str, float]:
    raw = (cfg.get("evidence", "node_priors", family, default=None) or
           DEFAULT_NODE_PRIORS.get(family, {}))
    # ETFs have no company evidence families; the caller's family map already
    # filters them. Human-disabled nodes are the only nodes removed entirely.
    enabled = cfg.get("nodes", default={}) or {}
    values = {node: max(0.0, float(weight)) for node, weight in raw.items()
              if enabled.get(node, {}).get("enabled", node == "macro_regime")}
    total = sum(values.values())
    return {node: value / total for node, value in values.items()} if total else {}


def score(events: list[SignalEvent], regime: str, cfg, store: Store,
          filters: list, ctx: MarketContext,
          node_states: dict[str, str] | None = None,
          symbol_states: dict[str, dict[str, str]] | None = None) -> list[TradeCandidate]:
    by_symbol: dict[str, list[SignalEvent]] = {}
    for e in events:
        by_symbol.setdefault(e.symbol, []).append(e)

    min_score = cfg.get("ensemble", "min_final_score", default=0.15)
    disp_pen = cfg.get("ensemble", "conflict_dispersion_penalty", default=0.5)
    friction = (cfg.get("execution", "spread_cost_bps", default=3)
                + cfg.get("execution", "slippage_bps", default=5)) * 2 / 10000.0  # round trip

    candidates = []
    snapshots = {}
    node_states, symbol_states = node_states or {}, symbol_states or {}
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
            priors = _node_priors(cfg, family, is_etf)
            prior = priors.get(s.node_id, 0.0)
            if prior <= 0:
                continue
            multiplier = s_node_multiplier(s.node_id, cfg, store, regime)
            w = prior * multiplier
            grouped.setdefault(family, []).append((s, w))
            detail = _detail(s, family, w)
            detail.update(prior_weight=round(prior, 6),
                          reliability_multiplier=round(multiplier, 6))
            details.append(detail)

        # Deterministic SEC quality is a scored family input as well as the
        # existing guardrail. It is evaluated once here and is point-in-time.
        for f in filters:
            event = f.graph_signal(ctx, symbol) if hasattr(f, "graph_signal") else None
            if event and "quality" in family_weights:
                priors = _node_priors(cfg, "quality", is_etf)
                prior = priors.get("quality_value", 1.0)
                multiplier = s_node_multiplier("quality_value", cfg, store, regime)
                w = prior * multiplier
                grouped.setdefault("quality", []).append((event, w))
                detail = _detail(event, "quality", w)
                detail.update(prior_weight=round(prior, 6),
                              reliability_multiplier=round(multiplier, 6))
                details.append(detail)

        # Regime is always explicit evidence rather than an invisible global
        # switch. It shares the context family with sector-relative evidence.
        if "context" in family_weights:
            macro = REGIME_ACTIVATION.get(regime, 0.0)
            macro_prior = _node_priors(cfg, "context", is_etf).get("macro_regime", .5)
            macro_multiplier = s_node_multiplier("macro_regime", cfg, store, regime)
            details.append({"node": "macro_regime", "family": "context",
                            "direction": "long" if macro >= 0 else "avoid",
                            "magnitude": abs(macro), "confidence": 1.0,
                            "signed_alpha": macro,
                            "node_weight": round(macro_prior * macro_multiplier, 6),
                            "prior_weight": round(macro_prior, 6),
                            "reliability_multiplier": round(macro_multiplier, 6),
                            "data_as_of": ctx.as_of,
                            "evidence": [f"market regime {regime}"], "state": "running"})
            macro_event = SignalEvent(
                symbol=symbol, direction="long" if macro >= 0 else "avoid",
                score=abs(macro), confidence=1.0, horizon_days=21,
                expected_return=0.0, expected_volatility=0.0,
                downside_estimate=0.0, evidence=[f"market regime {regime}"],
                data_as_of=__import__("datetime").datetime.strptime(ctx.as_of, "%Y-%m-%d"),
                node_id="macro_regime")
            grouped.setdefault("context", []).append(
                (macro_event, macro_prior * macro_multiplier))

        family_scores: dict[str, float] = {}
        expected_parts: list[tuple[float, float]] = []
        coverage = 0.0
        for family, family_weight in family_weights.items():
            members = grouped.get(family, [])
            priors = _node_priors(cfg, family, is_etf)
            present_nodes = {event.node_id for event, _ in members}
            available_nodes = set(present_nodes)
            for node, prior in priors.items():
                if node in present_nodes:
                    continue
                state = (symbol_states.get(node, {}).get(symbol) or
                         node_states.get(node) or "unavailable")
                if state == "neutral":
                    state = "verified_neutral"
                if state in ("running", "verified_neutral", "deemphasized"):
                    available_nodes.add(node)
                details.append({"node": node, "family": family, "state": state,
                                "signed_alpha": 0.0, "prior_weight": round(prior, 6),
                                "reliability_multiplier": round(
                                    s_node_multiplier(node, cfg, store, regime), 6),
                                "family_contribution": 0.0,
                                "evidence": ["no directional event" if state ==
                                             "verified_neutral" else "required data unavailable"]})
            available_prior = sum(priors.get(node, 0.0) for node in available_nodes)
            coverage += family_weight * min(1.0, available_prior)
            if not members:
                continue
            # Priors are normalized across the complete enabled family, not
            # across only the evidence that happened to be available. Missing
            # AI/event evidence therefore cannot donate weight to momentum or
            # any surviving singleton.
            value = sum(signed_alpha(event) * w for event, w in members)
            family_scores[family] = value
            for event, w in members:
                expected_parts.append((direction_sign(event.direction) *
                                       abs(event.expected_return),
                                       family_weight * w * event.confidence))
            for detail in details:
                if detail.get("family") == family and detail.get("state") == "running":
                    detail["family_weight"] = family_weight
            details.append({"node": f"family:{family}", "family": family,
                            "state": "running", "signed_alpha": round(value, 6),
                            "family_weight": family_weight,
                            "family_contribution": round(family_weight * value, 6),
                            "evidence": [f"{len(members)} evidence node(s)"]})

        raw = sum(family_weights[f] * value for f, value in family_scores.items())
        # Conflict is opposing weighted evidence, not unequal magnitude among
        # evidence that agrees. +1.0 and +0.1 are confirmation, not conflict.
        node_contributions = [family_weights[family] * signed_alpha(event) * weight
                              for family, members in grouped.items()
                              for event, weight in members]
        positive = sum(v for v in node_contributions if v > 0)
        negative = -sum(v for v in node_contributions if v < 0)
        opposing_mass = (2 * min(positive, negative) / (positive + negative)
                          if positive + negative else 0.0)
        conflict = max(0.0, 1.0 - disp_pen * opposing_mass)
        # Conflicts reduce positive conviction. They must never make negative
        # evidence less negative and accidentally rescue an avoid case.
        production_score = raw * conflict if raw > 0 else raw
        # horizon/expectation: confidence-weighted average of contributing nodes
        contributing_ids = {event.node_id for members in grouped.values()
                            for event, _ in members}
        longs = [s for s in sigs if s.node_id in contributing_ids and
                 s.direction in ("long", "long_call")]
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

        entry_mode, size_multiplier, entry_reason = "normal", 1.0, None
        if not is_etf:
            from .evidence import latest_dossier
            dossier = latest_dossier(store, symbol, ctx.as_of,
                                      ctx.as_of if getattr(ctx, "historical", False) else None)
            if not dossier or dossier.get("status") != "ready":
                probe_min = float(cfg.get("evidence", "dossierless_probe_min_score",
                                          default=.25))
                required = all(name in family_scores for name in
                               ("quality", "context", "price"))
                if production_score < probe_min or not required or exp_gross - friction <= 0:
                    continue
                entry_mode, size_multiplier = "probe", .25
                entry_reason = "PROBE — DOSSIER PENDING"
                if not ctx.offline:
                    try:
                        from .research import enqueue_job
                        enqueue_job(store, "deep_research",
                                    {"reason": "probe dossier", "symbol": symbol},
                                    priority=5, requested_by="autonomous")
                    except Exception:
                        pass                    # scoring never fails over queue telemetry

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
            entry_mode=entry_mode, size_multiplier=size_multiplier,
            entry_mode_reason=entry_reason,
        ))
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    store.kv_set("evidence_last_scores", {"schema": "evidence.v2", "as_of": ctx.as_of,
                                           "regime": regime, "symbols": snapshots})
    return candidates


def s_node_weight(node_id: str, cfg, store: Store, regime: str | None = None) -> float:
    base = float(cfg.get("nodes", node_id, "weight", default=0.0) or 0.0)
    return base * s_node_multiplier(node_id, cfg, store, regime)


def s_node_multiplier(node_id: str, cfg, store: Store,
                      regime: str | None = None) -> float:
    """Learned reliability with a non-zero floor for every enabled analysis.

    Only an operator toggle removes a node in `_node_priors`. Attribution may
    soften a node, but cannot turn a configured analysis into dead code.
    """
    status = str(cfg.get("nodes", node_id, "status", default="production"))
    floor = float(cfg.get("ensemble", "weight_learning",
                          "experimental_floor" if status == "experimental"
                          else "production_floor",
                          default=.25 if status == "experimental" else .50))
    ceiling = float(cfg.get("ensemble", "weight_learning", "max_multiplier",
                            default=1.5))
    # regime-conditioned multiplier replaces the global one when attribution
    # has a meaningful sample for this (node, regime) cell — see attribution.py
    value = None
    if regime:
        rm = (store.kv_get("regime_multipliers", {}) or {}).get(node_id, {})
        if regime in rm:
            value = float(rm[regime])
    if value is None:
        value = float(store.get_weight_multiplier(node_id))
    return max(floor, min(ceiling, value))
