"""Macro regime classifier — the gate node (AGENTS.md §10.7).

Inputs: benchmark trend (SPY vs 50/200 SMA), VIX level, universe breadth.
Output: regime label + deployment multiplier that scales the time-step budget
(stress → 0.0: the system stops opening new positions, exits still run).

Deliberately simple and explainable; upgrade path is HMM/vol-term-structure,
but only after this baseline is measured (dev/DECISIONS.md philosophy).
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import MarketContext


@dataclass
class RegimeState:
    regime: str                  # risk_on | neutral | risk_off | stress
    deployment_multiplier: float
    evidence: list[str]


def classify(ctx: MarketContext, cfg) -> RegimeState:
    bench = cfg.get("universe", "benchmark", default="SPY")
    closes = ctx.closes(bench)
    ev: list[str] = []

    trend_score = 0
    if len(closes) >= 200:
        px = closes.iloc[-1]
        sma50 = closes.rolling(50).mean().iloc[-1]
        sma200 = closes.rolling(200).mean().iloc[-1]
        trend_score = (1 if px > sma50 else -1) + (1 if px > sma200 else -1)
        ev.append(f"{bench} px={px:.2f} sma50={sma50:.2f} sma200={sma200:.2f}")
    else:
        ev.append(f"{bench}: insufficient history ({len(closes)} bars)")

    vix = ctx.vix()
    vix_on = cfg.get("regime", "vix_risk_on_max", default=20)
    vix_stress = cfg.get("regime", "vix_stress_min", default=28)
    ev.append(f"VIX={round(vix, 2) if vix is not None else None}")

    breadth = ctx.breadth_above_sma(50)
    ev.append(f"breadth(>50sma)={breadth if breadth is None else round(breadth, 2)}")

    # stress overrides everything
    if vix is not None and vix >= vix_stress:
        regime = "stress"
    elif trend_score == 2 and (vix is None or vix < vix_on) and \
            (breadth is None or breadth >= cfg.get("regime", "breadth_risk_on_min", default=0.55)):
        regime = "risk_on"
    elif trend_score <= -1 or \
            (breadth is not None and breadth <= cfg.get("regime", "breadth_risk_off_max", default=0.35)):
        regime = "risk_off"
    else:
        regime = "neutral"

    mult = cfg.get("regime", "deployment_multiplier", regime, default=0.5)
    return RegimeState(regime=regime, deployment_multiplier=float(mult), evidence=ev)
