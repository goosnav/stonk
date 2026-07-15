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

    vol = ctx.volatility_context()
    vix = vol.get("vix") if vol.get("vix") is not None else ctx.vix()
    vix_on = cfg.get("regime", "vix_risk_on_max", default=20)
    vix_stress = cfg.get("regime", "vix_stress_min", default=28)
    ev.append(f"VIX={round(vix, 2) if vix is not None else None}")
    ev.append("VIX curve 9d/3m={0} 1m/3m={1} 3m/6m={2} VVIX={3} Δ5d={4}".format(
        *(round(vol.get(k), 3) if vol.get(k) is not None else None for k in
          ("slope_9d_3m", "slope_1m_3m", "slope_3m_6m", "vvix", "vix_change_5d"))))
    ev.append(f"VIX-realized spread={round(vol['implied_realized_spread'], 2) if vol.get('implied_realized_spread') is not None else None}")

    hyg, tlt = ctx.closes("HYG", 30), ctx.closes("TLT", 30)
    credit_proxy = ((hyg.iloc[-1] / hyg.iloc[-22] - 1) -
                    (tlt.iloc[-1] / tlt.iloc[-22] - 1)) \
        if len(hyg) >= 22 and len(tlt) >= 22 else None
    ev.append(f"HYG-TLT 21d={credit_proxy:+.2%}" if credit_proxy is not None
              else "HYG-TLT 21d=None")

    breadth = ctx.breadth_above_sma(50)
    ev.append(f"breadth(>50sma)={breadth if breadth is None else round(breadth, 2)}")

    # stress overrides everything
    curve_inverted = (vol.get("slope_9d_3m") or 0) >= .08 or \
        (vol.get("slope_1m_3m") or 0) >= .05
    vol_shock = (vol.get("vvix") or 0) >= 125 or \
        (vol.get("vix_change_5d") or 0) >= .35
    if (vix is not None and vix >= vix_stress) or (curve_inverted and vol_shock):
        regime = "stress"
    elif trend_score == 2 and (vix is None or vix < vix_on) and \
            (breadth is None or breadth >= cfg.get("regime", "breadth_risk_on_min", default=0.55)):
        regime = "risk_on"
    elif trend_score <= -1 or (credit_proxy is not None and credit_proxy <= -.04) or \
            (breadth is not None and breadth <= cfg.get("regime", "breadth_risk_off_max", default=0.35)):
        regime = "risk_off"
    else:
        regime = "neutral"

    mult = cfg.get("regime", "deployment_multiplier", regime, default=0.5)
    return RegimeState(regime=regime, deployment_multiplier=float(mult), evidence=ev)
