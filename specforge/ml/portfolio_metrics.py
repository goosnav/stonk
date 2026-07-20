"""Net out-of-sample POLICY return — the one metric a model has to win on.

Forecast loss (pinball, IC) answers "is the prediction close?". This module
answers "did acting on the prediction make money after costs?", which is the
only question the R6 bakeoff gate accepts. Lower pinball with worse policy
return is not an improvement.

The metric is deliberately hostile to the usual ways this number gets inflated:

* **Non-overlapping cohorts.** Top-decile returns sampled every `horizon`
  sessions, so forward windows never share a price move. Compounding the
  per-*every*-date version double-counts the same move and understates
  volatility — it is not a valid equity curve.
* **All phases, median-aggregated.** Every staggered alignment (offset
  0..horizon-1) is computed, and the median decides. No single lucky phase
  can carry the result.
* **Fails closed.** Too few cohorts or offsets returns a NEGATIVE utility, not
  an optimistic one, so a promotion gate can never pass on thin evidence.

Lived in graph.py until R6; moved here so the graph, the TCN bakeoff, and the
promotion gate all score on one definition instead of three drifting copies.
"""
from __future__ import annotations

import numpy as np

# Fail-closed gate thresholds: below these the metric is not usable evidence.
MIN_COHORTS_PER_OFFSET = 3
MIN_VALID_OFFSETS = 8


def _cohort_cost(cost, mask, chosen) -> float:
    """Scalar cost, or the mean per-sample cost of the names actually picked."""
    if np.isscalar(cost):
        return float(cost)
    return float(np.asarray(cost)[mask][chosen].mean())


def cohort_returns(pred_col, truth_col, fold_dates, horizon: int, cost=0.0016,
                   offset: int = 0, min_names: int = 8) -> list[float]:
    """After-cost top-decile return per NON-OVERLAPPING decision date, one
    staggered alignment starting at `offset`.

    `cost` is a scalar or a per-row array (R5 per-sample costs), charged once
    per cohort against the names that cohort actually bought.
    """
    fd = np.asarray(fold_dates)
    pc, tc = np.asarray(pred_col), np.asarray(truth_col)
    days = sorted(set(fd.tolist()))                   # sorted-unique decision dates
    out: list[float] = []
    for k in range(offset, len(days), horizon):       # stride = horizon → no overlap
        m = fd == days[k]
        if int(m.sum()) < min_names:
            continue
        chosen = pc[m] >= np.quantile(pc[m], .9)
        out.append(float(tc[m][chosen].mean() - _cohort_cost(cost, m, chosen)))
    return out


def offset_metrics(cohort: list[float], horizon: int) -> dict | None:
    """Annualized/Sharpe/drawdown for ONE independent cohort series, or None."""
    if not cohort:
        return None
    arr = np.asarray(cohort, dtype=float)
    curve = np.cumprod(1 + arr)
    drawdown = float(np.max(1 - curve / np.maximum.accumulate(curve)))
    per_year = 252 / horizon
    return {"annualized": float(arr.mean() * per_year),
            "sharpe": float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(per_year)),
            "max_drawdown": drawdown, "n_cohorts": len(cohort)}


def staggered_portfolio_metrics(pred_col, truth_col, fold_dates, horizon: int = 21,
                                cost=0.0016, min_names: int = 8,
                                min_cohorts: int = MIN_COHORTS_PER_OFFSET,
                                min_offsets: int | None = None) -> dict:
    """Policy utility over ALL `horizon` staggered non-overlapping alignments,
    aggregated by median so no single arbitrary phase decides the metric.

    Returns {} only for no data; thin evidence returns a negative utility.
    """
    if not len(np.asarray(fold_dates)):
        return {}
    # A horizon of h has exactly h phases, so demanding 8 of them is
    # unsatisfiable at h=5 — an impossible gate, not a conservative one. Require
    # every phase when fewer than MIN_VALID_OFFSETS exist, which is stricter in
    # proportion, and the usual floor otherwise.
    if min_offsets is None:
        min_offsets = min(MIN_VALID_OFFSETS, horizon)
    offsets = []
    for off in range(horizon):
        m = offset_metrics(
            cohort_returns(pred_col, truth_col, fold_dates, horizon, cost,
                           offset=off, min_names=min_names), horizon)
        if m and m["n_cohorts"] >= min_cohorts:
            offsets.append(m)
    if len(offsets) < min_offsets:
        return {"portfolio_utility": -1.0, "oos_sharpe": -99.0, "max_drawdown": 1.0,
                "n_valid_offsets": len(offsets), "utility_evidence": "insufficient",
                "utility_basis": "staggered_non_overlapping_21s_cohorts"}
    ann = [o["annualized"] for o in offsets]
    shp = [o["sharpe"] for o in offsets]
    dd = [o["max_drawdown"] for o in offsets]
    med_ann, med_dd = float(np.median(ann)), float(np.median(dd))
    return {"portfolio_utility": round(med_ann - .5 * med_dd, 5),
            "oos_sharpe": round(float(np.median(shp)), 4),
            "max_drawdown": round(med_dd, 5),
            "worst_offset_sharpe": round(float(min(shp)), 4),
            "worst_offset_drawdown": round(float(max(dd)), 5),
            "n_valid_offsets": len(offsets),
            "cohorts_per_offset": [o["n_cohorts"] for o in offsets],
            "utility_evidence": "ok",
            "utility_basis": "staggered_non_overlapping_21s_cohorts"}
