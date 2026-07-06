"""Forecast distributions — error bars are load-bearing, not decorative
(AGENTS.md §13, dev/DECISIONS.md D10).

Primary method: bootstrap over historical analog trades in the same
(score_bucket × regime) cell — populated by backtest runs and by live/paper
round-trips as they close. Fallback when analogs are scarce (<MIN_ANALOGS):
vol-scaled prior centered near zero with wide bars, confidence label 'low'.
The ensemble point estimate is blended toward the analog mean as sample grows.
"""
from __future__ import annotations

import math
import random

from .models import TradeCandidate
from .store import Store

MIN_ANALOGS = 25
BOOTSTRAP_ITERS = 500


def _apr(horizon_ret: float, periods: float) -> float:
    """Annualize a horizon return. Base clamped positive: a modeled horizon
    return ≤ -100% (possible from the wide-prior branch on a fat expected
    return) would otherwise produce a complex number under a fractional
    exponent. -100%/yr is the honest floor for a long-only book."""
    return round(max(1e-4, 1 + horizon_ret) ** periods - 1, 4)


def _bootstrap_ci(returns: list[float], iters: int = BOOTSTRAP_ITERS,
                  lo_pct: float = 10, hi_pct: float = 90,
                  rng: random.Random | None = None) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) of the MEAN horizon return, 80% interval."""
    rng = rng or random.Random(1337)   # deterministic: same inputs → same bars
    n = len(returns)
    means = sorted(sum(rng.choices(returns, k=n)) / n for _ in range(iters))
    mean = sum(returns) / n
    return mean, means[int(iters * lo_pct / 100)], means[int(iters * hi_pct / 100)]


def attach_intervals(candidates: list[TradeCandidate], store: Store,
                     prices: dict[str, float]) -> None:
    from .execution import score_bucket   # local import avoids cycle
    for c in candidates:
        analogs = store.analog_returns(score_bucket(c.final_score), c.regime)
        if len(analogs) >= MIN_ANALOGS:
            mean, lo, hi = _bootstrap_ci(analogs)
            # shrink ensemble estimate toward the measured analog mean
            c.expected_return = round(0.5 * c.expected_return + 0.5 * mean, 5)
            # trade-level spread: individual outcomes, not mean — widen by dist
            rets = sorted(analogs)
            c.ci_low = round(rets[int(len(rets) * 0.10)], 5)
            c.ci_high = round(rets[int(len(rets) * 0.90)], 5)
            c.probability_positive = round(
                sum(1 for r in analogs if r > 0) / len(analogs), 3)
            c.confidence_label = "high" if len(analogs) >= 150 else "medium"
        else:
            # wide prior: ±2× horizon volatility around a haircut estimate
            sigma = max(0.02, abs(c.expected_return) * 3)
            c.expected_return = round(c.expected_return * 0.5, 5)  # decay haircut
            # floor: a long position cannot lose more than 100%
            c.ci_low = round(max(-0.95, c.expected_return - 2 * sigma), 5)
            c.ci_high = round(c.expected_return + 2 * sigma, 5)
            # normal approx for P(>0)
            z = c.expected_return / sigma
            c.probability_positive = round(0.5 * (1 + math.erf(z / math.sqrt(2))), 3)
            c.confidence_label = "low"

        # annualized view — secondary by design (AGENTS.md §13)
        periods = max(1.0, 252.0 / max(1, c.horizon_days))
        c.expected_apr = _apr(c.expected_return, periods)
        c.apr_ci_low = max(-0.99, _apr(c.ci_low, periods))
        c.apr_ci_high = _apr(c.ci_high, periods)


def portfolio_projection(store: Store, source: str) -> dict:
    """Strategy-level projected APR ± bars from closed trades (GUI headline).
    Basis counts by source are reported so the user sees what the number rests on."""
    trades = store.trades()
    if not trades:
        return {"apr": None, "note": "no closed trades yet — run a backtest"}
    rets = [t["ret"] for t in trades]
    horizons = [max(1, t["horizon_days"] or 20) for t in trades]
    mean_h = sum(horizons) / len(horizons)
    periods = 252.0 / mean_h
    mean, lo, hi = _bootstrap_ci(rets)
    basis = {}
    for t in trades:
        basis[t["source"]] = basis.get(t["source"], 0) + 1
    n = len(rets)
    return {
        "apr": _apr(mean, periods),
        "apr_ci_low": max(-0.99, _apr(lo, periods)),
        "apr_ci_high": _apr(hi, periods),
        "prob_positive_trade": round(sum(1 for r in rets if r > 0) / n, 3),
        "n_trades": n, "basis": basis,
        "confidence": "high" if basis.get("live", 0) >= 50 else
                      "medium" if basis.get("paper", 0) + basis.get("live", 0) >= 30 else "low",
    }
