"""Portfolio construction — simple by design (AGENTS.md §11 step 7: no fragile
mean-variance on noisy estimates).

rank by final_score → volatility-target size → respect caps/cash. The risk
governor re-checks everything; this layer just proposes sensible sizes.
Position size: notional = equity × vol_target_pct / atr_pct, i.e. each position
targets the same expected daily $ wiggle. Capped by single-position cap.
"""
from __future__ import annotations

from datetime import date, datetime

from .data import MarketContext
from .models import AccountState, TradeCandidate

MIN_TARGET_NOTIONAL = 5.0


def _completed_session_since(opened: date, as_of: date) -> bool:
    """True after at least one completed XNYS session, holiday-aware."""
    try:
        import exchange_calendars as xcals
        import pandas as pd
        sessions = xcals.get_calendar("XNYS").sessions_in_range(
            pd.Timestamp(opened), pd.Timestamp(as_of))
        return any(s.date() > opened for s in sessions)
    except Exception:
        return (as_of - opened).days >= 1


def rebalance_plan(candidates: list[TradeCandidate], account: AccountState,
                   ctx: MarketContext, cfg) -> dict:
    """Target-weight portfolio plus bounded partial funding sells."""
    max_positions = int(cfg.get("risk", "max_open_positions", default=12))
    deploy = min(float(cfg.get("risk", "max_account_deployment", default=.70)),
                 1 - float(cfg.get("risk", "min_cash_reserve", default=.30)))
    cap = float(cfg.get("risk", "max_single_equity_position", default=.08))
    # Defense in depth: a score cannot fund a trade when its calibrated return
    # estimate is non-positive after modeled costs.
    ranked = [c for c in candidates if c.side == "buy" and c.expected_return > 0][
        :max_positions]
    raw = {}
    for c in ranked:
        atr = ctx.atr_pct(c.symbol) or .03
        scenario = ((ctx.store.kv_get("research_scenarios") or {}).get("candidates") or {}).get(c.symbol, {})
        scale = max(0.0, float(scenario.get("recommended_scale", 1)))
        raw[c.symbol] = max(0.0, c.final_score * c.expected_return / atr) * scale
    total = sum(raw.values()) or 1.0
    weights = {s: min(cap, deploy * v / total) for s, v in raw.items()}
    # Redistribute unused capped weight without ever exceeding the cap.
    for _ in range(max_positions):
        room, missing = sum(cap - w for w in weights.values() if w < cap), deploy - sum(weights.values())
        if missing <= 1e-6 or room <= 1e-6:
            break
        for s in weights:
            if weights[s] < cap:
                weights[s] += min(cap - weights[s], missing * (cap - weights[s]) / room)
    # A dossierless probe is exactly 25% of the target the full model would
    # otherwise assign. Freed capacity remains cash; normalization must not
    # inflate the probe back to a normal position.
    by_candidate = {c.symbol: c for c in ranked}
    weights = {symbol: weight * max(0.0, min(1.0,
               float(by_candidate[symbol].size_multiplier)))
               for symbol, weight in weights.items()}

    by_symbol = {c.symbol: c for c in candidates}
    outside = ctx.store.kv_get("rebalance_outside_counts", {}) or {}
    # Hysteresis is evidence-model specific. Carrying "outside" counts from a
    # former scoring model into a new rollout can cause an immediate sell on
    # the first cycle (observed during evidence.v2 migration). A new decision
    # contract must earn its own two consecutive comparisons.
    counter_version = next((c.evidence_version for c in candidates
                            if c.evidence_version), "unversioned")
    previous_version = ctx.store.kv_get("rebalance_counter_version")
    version_changed = previous_version != counter_version
    if version_changed:
        if outside:
            ctx.store.audit("rebalance_hysteresis_reset", {
                "from": previous_version, "to": counter_version,
                "discarded_symbols": sorted(outside),
            })
        outside = {}
        ctx.store.kv_set("rebalance_counter_version", counter_version)
    held_symbols = {p.symbol for p in account.positions if p.qty > 0}
    # A provider/model outage can yield no candidates. That is not evidence
    # every holding deserves liquidation; only advance displacement when the
    # cycle produced a real ranked comparison set.
    outside = {s: (0 if s in weights else int(outside.get(s, 0)) + 1)
               for s in held_symbols} if ranked else {s: int(outside.get(s, 0))
                                                       for s in held_symbols}
    ctx.store.kv_set("rebalance_outside_counts", outside)
    # Probes may spend existing cash but never justify selling a holding to
    # fund an evidence-incomplete idea.
    top_unheld = max((c.final_score for c in ranked
                      if c.symbol not in held_symbols and c.entry_mode == "normal"),
                     default=0.0)
    max_turnover = account.equity * float(cfg.get(
        "risk", "max_rebalance_turnover_per_cycle", default=.30))
    sells, deferred_sells, turnover = [], [], 0.0
    stored = {p["symbol"]: p for p in ctx.store.open_positions(
        mode="live" if cfg.mode == "live" else "paper")}
    prices = ctx.prices()
    proposals = []
    for p in account.positions:
        if p.qty <= 0 or p.asset_type != "equity" or p.symbol not in stored:
            continue
        # Missing research is not a bearish thesis. During evidence.v2 an
        # individual company must have a verified dossier before an
        # opportunity-cost *discretionary* trim can fire. Protective stops and
        # time exits run earlier in the engine and are deliberately unaffected.
        if counter_version == "evidence.v2":
            from .ensemble import ETF_SYMBOLS
            if p.symbol not in ETF_SYMBOLS:
                from .evidence import latest_dossier
                dossier = latest_dossier(ctx.store, p.symbol, ctx.as_of,
                                          ctx.as_of if getattr(ctx, "historical", False) else None)
                if not dossier or dossier.get("status") != "ready":
                    deferred_sells.append({
                        "symbol": p.symbol,
                        "reason": "awaiting verified company dossier",
                    })
                    continue
        px = prices.get(p.symbol, p.avg_cost)
        current = p.qty * px
        target = account.equity * weights.get(p.symbol, 0.0)
        delta = current - target
        held_score = by_symbol.get(p.symbol).final_score if p.symbol in by_symbol else 0.0
        opened = datetime.fromisoformat(stored[p.symbol]["opened_at"]).date()
        eligible = _completed_session_since(opened, date.fromisoformat(ctx.as_of))
        displaced = bool(ranked) and weights.get(p.symbol, 0) == 0 and \
            outside.get(p.symbol, 0) >= 2
        improved = top_unheld >= held_score + float(cfg.get(
            "sizing", "rebalance_score_hysteresis", default=.05))
        if version_changed or not eligible or delta < MIN_TARGET_NOTIONAL or \
                not (displaced or improved):
            continue
        # Opportunity rotation trims in portions. Stops/time exits remain the
        # only ordinary path to immediate full liquidation.
        amount = min(delta, current * .50, max_turnover - turnover)
        if amount < MIN_TARGET_NOTIONAL:
            continue
        proposals.append({"position": stored[p.symbol], "symbol": p.symbol,
                          # Carry the capped amount forward. Using `delta` here
                          # silently undid the 50% partial-trim limit when the
                          # funding proposals were sequenced (NFLX/UNH,
                          # 2026-07-15).
                          "qty": min(p.qty, amount / px), "notional": amount,
                          "current_weight": current / account.equity,
                          "target_weight": weights.get(p.symbol, 0.0),
                          "held_score": held_score, "best_new_score": top_unheld,
                          "opportunity_cost": top_unheld - held_score,
                          "overweight": delta / account.equity})
    # Fund from the weakest/redundant names first, not broker position order.
    proposals.sort(key=lambda s: (s["opportunity_cost"], s["overweight"]), reverse=True)
    for proposal in proposals[:int(cfg.get(
            "risk", "max_rebalance_sells_per_cycle", default=2))]:
        amount = min(proposal["notional"], max_turnover - turnover)
        if amount < MIN_TARGET_NOTIONAL:
            continue
        proposal["notional"] = amount
        proposal["qty"] = min(proposal["position"]["qty"],
                              amount / prices.get(proposal["symbol"],
                                                  proposal["position"]["avg_cost"]))
        sells.append(proposal)
        turnover += amount
    actual = {p.symbol: p.qty * prices.get(p.symbol, p.avg_cost) for p in account.positions}
    actual_weights = {s: value / account.equity for s, value in actual.items()}
    buys = [(by_symbol[s], max(0.0, account.equity * w - actual.get(s, 0.0)))
            for s, w in weights.items() if s in by_symbol]
    buys = [(c, n) for c, n in buys if n >= MIN_TARGET_NOTIONAL]
    return {"weights": weights, "actual_weights": actual_weights,
            "target_weights": weights, "sells": sells, "buys": buys,
            "deferred_sells": deferred_sells,
            "turnover": turnover, "turnover_cap": max_turnover}


def construct(candidates: list[TradeCandidate], account: AccountState,
              ctx: MarketContext, cfg) -> list[tuple[TradeCandidate, float]]:
    vol_target = cfg.get("sizing", "vol_target_pct", default=0.01)
    pos_cap = account.equity * cfg.get("risk", "max_single_equity_position", default=0.08)
    max_new = cfg.get("risk", "max_daily_new_positions", default=3)
    held = {p.symbol for p in account.positions if p.qty > 0}
    scenarios = ctx.store.kv_get("research_scenarios") or {}
    scenario_fresh = scenarios.get("as_of") == ctx.store.latest_bar_date(
        cfg.get("universe", "benchmark", default="SPY"))

    targets = []
    for c in candidates:
        if len(targets) >= max_new:
            break
        if c.symbol in held:
            continue                     # no averaging up/down in MVP
        scenario = (scenarios.get("candidates") or {}).get(c.symbol) if scenario_fresh else None
        if scenario and scenario.get("recommended_scale", 1) <= 0:
            c.risk_flags.append("bootstrap median did not improve after costs")
            continue
        price = ctx.close(c.symbol)
        atr = ctx.atr_pct(c.symbol)
        if not price or not atr or atr <= 0:
            continue
        notional = min(account.equity * vol_target / atr, pos_cap)
        # conviction scaling: score 0.15→~60% size, 0.5+→full size
        notional *= min(1.0, 0.5 + c.final_score)
        # Missing evidence never hands its vote to another family. The lower
        # score already reduces size; this explicit quality factor makes the
        # degradation visible and bounded without freezing trading.
        notional *= 0.5 + 0.5 * max(0.0, min(1.0, c.evidence_coverage))
        if scenario:
            notional *= float(scenario.get("recommended_scale", 1))
            if scenario.get("recommended_scale", 1) < 1:
                c.risk_flags.append(
                    f"bootstrap size ×{scenario['recommended_scale']:.2f} for drawdown")
        # entry_mode="probe" sizing happens HERE exactly once. The rebalance
        # path scales weights (never a construct() notional) and a candidate
        # flows through one path per cycle, so it is never re-applied.
        notional *= max(0.0, min(1.0, float(c.size_multiplier)))
        if notional < 5.0:
            continue
        c.target_notional = round(notional, 2)
        c.max_loss = c.target_notional   # equity worst case = full notional (D3)
        targets.append((c, c.target_notional))
    return targets


def fit_to_capacity(targets: list[tuple[TradeCandidate, float]],
                    account: AccountState, cfg, budget: float
                    ) -> list[tuple[TradeCandidate, float]]:
    """Scale the ranked batch once so requested buys fit real spendable cash."""
    max_batch = cfg.get("risk", "max_new_positions_per_cycle", default=3)
    selected = list(targets[:max(0, int(max_batch))])
    # Equity minus settled cash is the broker-marked current exposure. Cost
    # basis would overfund losers and underfund winners.
    deployed = max(0.0, account.equity - account.cash) if account.positions else 0.0
    deploy_frac = min(cfg.get("risk", "max_account_deployment", default=0.70),
                      1.0 - cfg.get("risk", "min_cash_reserve", default=0.0))
    spendable = min(max(0.0, account.cash), max(0.0, account.buying_power))
    capacity = max(0.0, min(budget, spendable,
                            account.equity * deploy_frac - deployed))

    # Drop the weakest allocation one at a time, then recompute. Filtering all
    # sub-$5 scaled rows at once can erase the whole batch even when the top
    # candidate is individually affordable (observed with $7.77 confirmed
    # spendable cash after partial funding sells).
    while selected and capacity >= MIN_TARGET_NOTIONAL:
        requested = sum(n for _, n in selected)
        scale = min(1.0, capacity / requested) if requested else 0.0
        fitted = [(c, round(n * scale, 2)) for c, n in selected]
        if all(n >= MIN_TARGET_NOTIONAL for _, n in fitted):
            for cand, notional in fitted:
                cand.target_notional = notional
                cand.max_loss = notional
            return fitted
        selected.pop()                     # candidates are strongest-first
    return []
