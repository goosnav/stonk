"""Portfolio construction — simple by design (AGENTS.md §11 step 7: no fragile
mean-variance on noisy estimates).

rank by final_score → volatility-target size → respect caps/cash. The risk
governor re-checks everything; this layer just proposes sensible sizes.
Position size: notional = equity × vol_target_pct / atr_pct, i.e. each position
targets the same expected daily $ wiggle. Capped by single-position cap.
"""
from __future__ import annotations

from .data import MarketContext
from .models import AccountState, TradeCandidate

MIN_TARGET_NOTIONAL = 5.0


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
        if scenario:
            notional *= float(scenario.get("recommended_scale", 1))
            if scenario.get("recommended_scale", 1) < 1:
                c.risk_flags.append(
                    f"bootstrap size ×{scenario['recommended_scale']:.2f} for drawdown")
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
    deployed = sum(p.cost_basis for p in account.positions)
    deploy_frac = min(cfg.get("risk", "max_account_deployment", default=0.70),
                      1.0 - cfg.get("risk", "min_cash_reserve", default=0.0))
    spendable = min(max(0.0, account.cash), max(0.0, account.buying_power))
    capacity = max(0.0, min(budget, spendable,
                            account.equity * deploy_frac - deployed))

    # Drop allocations that would become sub-$5 noise, then rescale the
    # remaining ranked candidates to use the same bounded capacity.
    while selected and capacity >= MIN_TARGET_NOTIONAL:
        requested = sum(n for _, n in selected)
        scale = min(1.0, capacity / requested) if requested else 0.0
        fitted = [(c, round(n * scale, 2)) for c, n in selected]
        kept = [(c, n) for c, n in fitted if n >= MIN_TARGET_NOTIONAL]
        if len(kept) == len(selected):
            for cand, notional in kept:
                cand.target_notional = notional
                cand.max_loss = notional
            return kept
        selected = [(c, n) for c, n in selected
                    if n * scale >= MIN_TARGET_NOTIONAL]
    return []
