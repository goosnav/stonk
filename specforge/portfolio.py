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


def construct(candidates: list[TradeCandidate], account: AccountState,
              ctx: MarketContext, cfg) -> list[tuple[TradeCandidate, float]]:
    vol_target = cfg.get("sizing", "vol_target_pct", default=0.01)
    pos_cap = account.equity * cfg.get("risk", "max_single_equity_position", default=0.08)
    max_new = cfg.get("risk", "max_daily_new_positions", default=3)
    held = {p.symbol for p in account.positions if p.qty > 0}

    targets = []
    for c in candidates:
        if len(targets) >= max_new:
            break
        if c.symbol in held:
            continue                     # no averaging up/down in MVP
        price = ctx.close(c.symbol)
        atr = ctx.atr_pct(c.symbol)
        if not price or not atr or atr <= 0:
            continue
        notional = min(account.equity * vol_target / atr, pos_cap)
        # conviction scaling: score 0.15→~60% size, 0.5+→full size
        notional *= min(1.0, 0.5 + c.final_score)
        if notional < 5.0:
            continue
        c.target_notional = round(notional, 2)
        c.max_loss = c.target_notional   # equity worst case = full notional (D3)
        targets.append((c, c.target_notional))
    return targets
