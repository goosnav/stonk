"""Runnable checks for the NN repair pass (dev/NN_REPAIR_IMPLEMENTATION_PLAN).

Stage A: the three architecture-independent honest-math fixes, plus the
Stage-A audit hardening (strict forecast validation, fold embargo proofs,
staggered all-offset portfolio metric that fails closed).
"""
import math

import numpy as np
import pytest

from specforge.ml import NeuralForecast
from specforge.ml.schema import SUPPORTED_HORIZONS
from specforge import graph, neural


# ── A1: NeuralForecast contract — fail loud, never clamp ───────────────────────

def _forecast(**over):
    base = dict(symbol="AAA", as_of="2026-07-15", horizon_sessions=21,
                absolute_q10=-0.03, absolute_q50=0.01, absolute_q90=0.06,
                excess_q10=-0.02, excess_q50=0.02, excess_q90=0.05,
                probability_absolute_positive=0.55,
                probability_excess_positive=0.6,
                model_id="m1", dataset_manifest_id="d1", feature_schema_hash="h1")
    base.update(over)
    return NeuralForecast(**base)


def test_forecast_valid_constructs():
    f = _forecast()
    assert f.horizon_sessions in SUPPORTED_HORIZONS


def test_forecast_rejects_nan():
    with pytest.raises(ValueError):
        _forecast(absolute_q50=float("nan"))


def test_forecast_rejects_positive_infinity():
    with pytest.raises(ValueError):
        _forecast(absolute_q90=float("inf"))


def test_forecast_rejects_negative_infinity():
    with pytest.raises(ValueError):
        _forecast(absolute_q10=float("-inf"))


def test_forecast_rejects_unordered_absolute():
    with pytest.raises(ValueError):
        _forecast(absolute_q50=0.10)          # q50 > q90


def test_forecast_rejects_unordered_excess():
    with pytest.raises(ValueError):
        _forecast(excess_q10=0.10)            # q10 > q50


def test_forecast_rejects_probability_below_zero():
    with pytest.raises(ValueError):
        _forecast(probability_absolute_positive=-0.01)


def test_forecast_rejects_probability_above_one():
    with pytest.raises(ValueError):
        _forecast(probability_excess_positive=1.01)


def test_forecast_rejects_unsupported_and_nonpositive_horizon():
    with pytest.raises(ValueError):
        _forecast(horizon_sessions=7)         # not in SUPPORTED_HORIZONS
    with pytest.raises(ValueError):
        _forecast(horizon_sessions=0)


def test_forecast_requires_provenance():
    with pytest.raises(ValueError):
        _forecast(model_id="")


def test_positive_excess_negative_absolute_is_not_a_long():
    """The semantic bug this contract prevents: a valid forecast that beats a
    falling benchmark is still an absolute loss and must not be a long."""
    f = _forecast(absolute_q10=-0.12, absolute_q50=-0.05, absolute_q90=0.01,
                  excess_q10=0.01, excess_q50=0.05, excess_q90=0.09)
    assert f.excess_q50 > 0                    # beats the benchmark
    assert f.absolute_edge_after_cost(0.0016) < 0   # ...but still loses money


# ── A2: purged walk-forward folds are half-open, embargoed, non-empty ─────────

def test_fold_windows_half_open_embargoed_non_empty():
    n, folds, embargo = 400, 5, 21
    sealed = int(n * .85)
    windows = neural._fold_windows(n, folds, embargo)
    assert len(windows) >= 2
    seen = set()
    for train_pos, ts, te in windows:
        assert te <= sealed                      # never reaches the sealed block
        assert ts < te                           # no fold silently empties
        assert ts - train_pos >= embargo + 1     # embargo gap before test (sessions)
        # last training label ends at train_pos+embargo == ts-1 < ts → no leak
        assert train_pos + embargo < ts
        rng = set(range(ts, te))                 # half-open [ts, te)
        assert not (rng & seen), "adjacent folds share a test session"
        seen |= rng


def test_fold_windows_use_session_indices_not_dates():
    # Pure integer-index arithmetic: identical n/folds/embargo → identical windows
    # regardless of any calendar. (Contract check that boundaries are index-based.)
    assert neural._fold_windows(300, 5, 21) == neural._fold_windows(300, 5, 21)
    assert all(isinstance(x, int) for w in neural._fold_windows(300, 5, 21) for x in w)


# ── A3: graph portfolio metric — staggered offsets, no overlap, fail closed ───

def _synthetic(n_sessions, n_names=12, ret=0.01, seed=0):
    rng = np.random.RandomState(seed)
    dates, preds, truths = [], [], []
    for s in range(n_sessions):
        for _ in range(n_names):
            dates.append(s)
            preds.append(float(rng.rand()))
            truths.append(ret)
    return np.array(preds), np.array(truths), np.array(dates)


def test_cohort_returns_are_non_overlapping():
    # 63 consecutive sessions → offset 0 picks days 0,21,42 = 3 cohorts, not 63.
    pred, truth, dates = _synthetic(63)
    cohort = graph._cohort_returns(pred, truth, dates, horizon=21, cost=0.0, offset=0)
    assert len(cohort) == 3


def test_cohort_offsets_select_different_alignments():
    pred, truth, dates = _synthetic(63)
    c0 = graph._cohort_returns(pred, truth, dates, 21, 0.0, offset=0)
    c5 = graph._cohort_returns(pred, truth, dates, 21, 0.0, offset=5)
    assert len(c0) == 3 and len(c5) == 3        # 0,21,42 vs 5,26,47 — disjoint days


def test_cost_deducted_exactly_once():
    pred, truth, dates = _synthetic(1, n_names=10, ret=0.02)
    cohort = graph._cohort_returns(pred, truth, dates, 21, cost=0.0016, offset=0)
    assert len(cohort) == 1
    assert cohort[0] == pytest.approx(0.02 - 0.0016)


def test_staggered_evaluates_all_offsets():
    pred, truth, dates = _synthetic(210)        # 10 cohorts/offset → all 21 valid
    m = graph._staggered_portfolio_metrics(pred, truth, dates)
    assert m["utility_evidence"] == "ok"
    assert m["n_valid_offsets"] == 21
    assert m["utility_basis"] == "staggered_non_overlapping_21s_cohorts"
    assert math.isfinite(m["oos_sharpe"])


def test_staggered_robust_to_initial_offset():
    pred, truth, dates = _synthetic(210, ret=0.01)
    u0 = graph._staggered_portfolio_metrics(pred, truth, dates)["portfolio_utility"]
    # Relabel every session +1: shifts which day is "offset 0" but all offsets
    # are still evaluated and median-aggregated → the aggregate barely moves.
    u1 = graph._staggered_portfolio_metrics(pred, truth, dates + 1)["portfolio_utility"]
    assert abs(u0 - u1) < 0.02


def test_undersized_cohorts_fail_closed():
    pred, truth, dates = _synthetic(30)          # ≤2 cohorts/offset < min_cohorts
    m = graph._staggered_portfolio_metrics(pred, truth, dates)
    assert m["utility_evidence"] == "insufficient"
    assert m["portfolio_utility"] <= 0           # cannot pass a `> 0` gate
    assert m["oos_sharpe"] <= 0


def test_empty_input_returns_empty():
    assert graph._staggered_portfolio_metrics(
        np.array([]), np.array([]), np.array([])) == {}


def test_offset_metrics_do_not_compound_overlapping_labels():
    # 3 independent cohorts compound 3 points, not 63; drawdown finite & bounded.
    m = graph._offset_metrics([0.01, 0.01, 0.01], horizon=21)
    assert m["n_cohorts"] == 3
    assert m["max_drawdown"] == 0.0
    assert math.isfinite(m["sharpe"])


# ── B2: the earliest of 60 sessions must be able to influence the output ──────

def test_tcn_receptive_field_reaches_first_session():
    torch = pytest.importorskip("torch")
    model = neural._make_model(len(neural.FEATURES), 2)
    model.eval()
    torch.manual_seed(0)
    a = torch.randn(1, 60, len(neural.FEATURES))
    b = a.clone()
    b[0, 0, :] += 5.0                          # perturb ONLY the earliest session
    with torch.no_grad():
        qa, _ = model.forward_all(a)
        qb, _ = model.forward_all(b)
    # The context branch reads only the last row, so any difference here is the
    # temporal encoder genuinely seeing session 0 (fails for a 15-session field).
    assert not torch.allclose(qa, qb, atol=1e-6)
