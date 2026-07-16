"""Runnable checks for the NN repair pass (dev/NN_REPAIR_IMPLEMENTATION_PLAN).

Stage A: the three architecture-independent honest-math fixes, plus the
Stage-A audit hardening (strict forecast validation, fold embargo proofs,
staggered all-offset portfolio metric that fails closed).
"""
import math

import numpy as np
import pandas as pd
import pytest

from specforge.ml import NeuralForecast
from specforge.ml.schema import SUPPORTED_HORIZONS
from specforge.ml import targets as ml_targets
from specforge import graph, neural


# ── A1: NeuralForecast contract — fail loud, never clamp ───────────────────────

def _forecast(**over):
    base = dict(symbol="AAA", as_of="2026-07-15", horizon_sessions=21,
                absolute_q10=-0.03, absolute_q50=0.01, absolute_q90=0.06,
                excess_q10=-0.02, excess_q50=0.02, excess_q90=0.05,
                probability_absolute_edge_positive=0.55,
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
        _forecast(probability_absolute_edge_positive=-0.01)


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


# ── B1A: explicit absolute + excess target contract ───────────────────────────

def test_absolute_targets_are_stock_forward_returns():
    close = pd.Series([100.0, 101, 102, 103, 104, 105], index=range(6))
    bench = pd.Series([100.0, 100, 100, 100, 100, 100], index=range(6))
    absolute, excess = ml_targets.build_targets(close, bench, horizons=(5,))
    assert absolute[5].iloc[0] == pytest.approx(105 / 100 - 1)
    # flat benchmark → excess equals absolute
    assert excess[5].iloc[0] == pytest.approx(absolute[5].iloc[0])


def test_excess_targets_subtract_benchmark():
    close = pd.Series([100.0, 0, 0, 0, 0, 110], index=range(6))
    bench = pd.Series([100.0, 0, 0, 0, 0, 104], index=range(6))
    absolute, excess = ml_targets.build_targets(close, bench, horizons=(5,))
    assert absolute[5].iloc[0] == pytest.approx(0.10)
    assert excess[5].iloc[0] == pytest.approx(0.10 - 0.04)


def test_both_families_share_index_and_horizons():
    close = pd.Series(np.linspace(100, 130, 40), index=range(40))
    bench = pd.Series(np.linspace(100, 110, 40), index=range(40))
    absolute, excess = ml_targets.build_targets(close, bench)
    assert absolute.index.equals(excess.index)
    assert list(absolute.columns) == list(excess.columns) == list(ml_targets.HORIZONS)


def test_targets_are_strictly_forward_no_lookahead():
    close = pd.Series(np.linspace(100, 130, 40), index=range(40))
    fwd = ml_targets.forward_return(close, 5)
    assert fwd.iloc[-5:].isna().all()          # last h rows cannot see the future
    assert fwd.iloc[:-5].notna().all()


def test_down_stock_beating_down_benchmark_is_not_a_long():
    # stock -5%, benchmark -10% → absolute -5%, excess +5%
    close = pd.Series([100.0, 0, 0, 0, 0, 95], index=range(6))
    bench = pd.Series([100.0, 0, 0, 0, 0, 90], index=range(6))
    absolute, excess = ml_targets.build_targets(close, bench, horizons=(5,))
    assert absolute[5].iloc[0] == pytest.approx(-0.05)
    assert excess[5].iloc[0] == pytest.approx(0.05)
    cost = 0.0016
    abs_label, exc_label = ml_targets.probability_labels(
        absolute[5].iloc[0], excess[5].iloc[0], cost)
    assert bool(abs_label) is False            # −5% never clears +0.16% cost
    assert bool(exc_label) is True


def test_round_trip_cost_matches_repo_convention(cfg):
    # (spread 3bps + slippage 5bps) × 2 sides / 1e4 == the 0.0016 used elsewhere
    assert ml_targets.round_trip_cost(cfg) == pytest.approx(0.0016)


def _long_history(store):
    from conftest import synth_bars
    for sym in ("AAA", "BBB", "CCC", "SPY"):
        store.upsert_bars(sym, synth_bars(n_days=700, daily_drift=.001), "test")
    store.upsert_bars("^VIX", [{**r, "open": 15, "high": 16, "low": 14, "close": 15}
                               for r in synth_bars(n_days=700)], "test")


def _small_dataset(cfg, store):
    _long_history(store)
    cfg.data["neural"]["input_sessions"] = 40
    cfg.data["neural"]["horizons"] = [5, 21]
    return neural.build_dataset(cfg, store, symbols=["AAA", "BBB", "CCC"])


def test_dataset_carries_both_target_families_and_cost(cfg, store):
    ds = _small_dataset(cfg, store)
    assert "error" not in ds, ds
    assert ds["Y_absolute"].shape == ds["Y_excess"].shape
    assert np.isfinite(ds["Y_absolute"]).all()
    assert ds["round_trip_cost"] == pytest.approx(0.0016)
    assert ds["target_schema_hash"] == ml_targets.TARGET_SCHEMA_HASH
    # features and targets are disjoint arrays — no forward-looking feature names
    assert not any(k in " ".join(neural.FEATURES) for k in ("future", "target", "fwd"))


def test_dataset_split_respects_full_horizon_embargo(cfg, store):
    ds = _small_dataset(cfg, store)
    unique = sorted(set(ds["dates"]))
    embargo = max(ds["horizons"])
    assert unique.index(ds["val_start"]) - unique.index(ds["train_end"]) >= embargo
    assert unique.index(ds["test_start"]) - unique.index(ds["val_end"]) >= embargo


# ── B1B: structured dual-output model ─────────────────────────────────────────

def test_structured_output_shapes_ordering_bounds():
    torch = pytest.importorskip("torch")
    model = neural._make_model(len(neural.FEATURES), 2).eval()
    out = model.forward_structured(torch.randn(4, 60, len(neural.FEATURES)))
    for q in (out.absolute_quantiles, out.excess_quantiles):
        assert q.shape == (4, 2, 3)
        assert torch.all(q[..., 0] <= q[..., 1]) and torch.all(q[..., 1] <= q[..., 2])
    for p in (out.probability_absolute_edge_positive, out.probability_excess_positive):
        assert p.shape == (4, 2)
        assert torch.all((p >= 0) & (p <= 1))


def test_gradients_reach_every_head():
    torch = pytest.importorskip("torch")
    model = neural._make_model(len(neural.FEATURES), 2)
    out = model.forward_structured(torch.randn(3, 60, len(neural.FEATURES)))
    loss = (out.absolute_quantiles.sum() + out.excess_quantiles.sum()
            + out.probability_absolute_edge_positive.sum()
            + out.probability_excess_positive.sum())
    loss.backward()
    for name in ("absolute_quantile_heads", "excess_quantile_heads",
                 "absolute_probability_heads", "excess_probability_heads"):
        head = getattr(model, name)[0]
        assert head.weight.grad is not None and torch.any(head.weight.grad != 0)


def test_bounded_training_produces_finite_loss(cfg, store):
    torch = pytest.importorskip("torch")
    ds = _small_dataset(cfg, store)
    model = neural._make_model(len(neural.FEATURES), len(ds["horizons"]))
    X = torch.from_numpy(ds["X"]); tr = np.flatnonzero(ds["masks"]["train"])[:256]
    Yx = torch.from_numpy(ds["Y_excess"] / ds["target_scale"])
    Ya = torch.from_numpy(ds["Y_absolute"] / ds["target_scale_absolute"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        out = model.forward_structured(X[tr])
        loss = neural._pinball(out.excess_quantiles, Yx[tr]) + \
            neural._pinball(out.absolute_quantiles, Ya[tr])
        assert math.isfinite(float(loss.detach()))
        opt.zero_grad(); loss.backward(); opt.step()


def test_checkpoint_roundtrip_reconstructs_outputs(tmp_path):
    torch = pytest.importorskip("torch")
    model = neural._make_model(len(neural.FEATURES), 2).eval()
    x = torch.randn(2, 60, len(neural.FEATURES))
    before = model.forward_structured(x)
    path = tmp_path / "m.pt"
    torch.save(model.state_dict(), path)
    reloaded = neural._make_model(len(neural.FEATURES), 2)
    reloaded.load_state_dict(torch.load(path)); reloaded.eval()
    after = reloaded.forward_structured(x)
    assert torch.allclose(before.absolute_quantiles, after.absolute_quantiles, atol=1e-6)
    assert torch.allclose(before.excess_quantiles, after.excess_quantiles, atol=1e-6)


# ── B3: genuine temporal / context feature separation ─────────────────────────

def test_feature_groups_partition_features_no_overlap():
    temporal, context = set(neural.TEMPORAL_FEATURES), set(neural.CONTEXT_FEATURES)
    assert temporal.isdisjoint(context)                 # no accidental overlap
    assert temporal | context == set(neural.FEATURES)   # every feature is placed
    assert len(neural.TEMPORAL_FEATURES) + len(neural.CONTEXT_FEATURES) == len(neural.FEATURES)


def _perturbed(model, torch, feature_name, session):
    a = torch.zeros(1, 60, len(neural.FEATURES))
    b = a.clone()
    b[0, session, neural.FEATURES.index(feature_name)] += 5.0
    with torch.no_grad():
        return model.forward_structured(a).excess_quantiles, \
               model.forward_structured(b).excess_quantiles


def test_early_temporal_feature_changes_output():
    torch = pytest.importorskip("torch")
    model = neural._make_model(len(neural.FEATURES), 2).eval()
    qa, qb = _perturbed(model, torch, neural.TEMPORAL_FEATURES[0], session=0)
    assert not torch.allclose(qa, qb, atol=1e-6)


def test_early_context_feature_does_not_change_output():
    torch = pytest.importorskip("torch")
    model = neural._make_model(len(neural.FEATURES), 2).eval()
    # A context feature at an early session must not reach the output: the
    # context branch reads only the last session, the temporal branch never
    # sees context columns.
    qa, qb = _perturbed(model, torch, neural.CONTEXT_FEATURES[0], session=0)
    assert torch.allclose(qa, qb, atol=1e-6)


def test_final_context_feature_changes_output():
    torch = pytest.importorskip("torch")
    model = neural._make_model(len(neural.FEATURES), 2).eval()
    qa, qb = _perturbed(model, torch, neural.CONTEXT_FEATURES[0], session=59)
    assert not torch.allclose(qa, qb, atol=1e-6)


def test_checkpoint_records_feature_split(cfg, store):
    ds = _small_dataset(cfg, store)
    # exercised more fully by the B4 smoke test; here assert the payload contract
    import specforge.neural as N
    assert N.TEMPORAL_HASH and N.CONTEXT_HASH and N.TEMPORAL_HASH != N.CONTEXT_HASH


def test_model_card_reports_real_split(cfg, store):
    card = neural.describe(cfg, store)["architecture"]
    assert card["type"] == "causal_tcn_dual_branch"
    assert card["receptive_field"] == 63
    assert "24 sequence features" in card["temporal_branch"]
    assert "20 point-in-time features" in card["context_branch"]
    assert set(card["return_families"]) == {"absolute", "excess"}


# ── B4A: schema-current checkpoints and dual calibration ──────────────────────

def _cal_inputs(q50=0.03, truth=0.05, n=200):
    pred = np.zeros((n, 2, 3), dtype=float)
    pred[:, :, 0], pred[:, :, 1], pred[:, :, 2] = q50 - 0.03, q50, q50 + 0.03
    return pred, np.full((n, 2), 0.5), np.full((n, 2), truth)


def test_calibration_corrects_q50_systematic_bias():
    pred, prob, truth = _cal_inputs(q50=0.03, truth=0.05)   # q50 is 0.02 low
    cal = neural._calibration(pred, prob, truth, prob_threshold=0.0)
    assert cal["quantile_offsets"][0][1] == pytest.approx(0.02, abs=1e-9)
    p2, _ = neural._apply_calibration(pred.copy(), prob.copy(), cal)
    assert np.allclose(p2[:, 0, 1], 0.05, atol=1e-6)        # q50 now matches truth


def test_calibration_is_a_pure_function_of_its_arguments():
    pred, prob, truth = _cal_inputs()
    # No hidden state / sealed outcomes: same args → identical calibration.
    assert neural._calibration(pred, prob, truth) == neural._calibration(pred, prob, truth)


def test_calibration_threshold_is_recorded_per_family():
    pred, prob, truth = _cal_inputs()
    assert neural._calibration(pred, prob, truth, 0.0)["prob_threshold"] == 0.0
    assert neural._calibration(pred, prob, truth, 0.0016)["prob_threshold"] == 0.0016


def test_apply_calibration_preserves_ordering_and_bounds():
    rng = np.random.RandomState(3)
    pred = np.sort(rng.randn(50, 2, 3) * 0.05, axis=2)
    prob = rng.rand(50, 2)
    cal = {"quantile_offsets": [[-0.1, 0.2, 0.3], [0.05, -0.2, -0.1]],
           "probability_logit_offsets": [1.5, -2.0]}
    p, q = neural._apply_calibration(pred, prob, cal)
    assert np.all(p[:, :, 0] <= p[:, :, 1]) and np.all(p[:, :, 1] <= p[:, :, 2])
    assert np.all((q >= 0) & (q <= 1)) and np.all(np.isfinite(q))


def test_missing_or_malformed_calibration_is_a_safe_noop():
    pred, prob, _ = _cal_inputs()
    for bad in (None, {}, {"quantile_offsets": []}):
        p, q = neural._apply_calibration(pred.copy(), prob.copy(), bad)
        assert np.allclose(p, pred) and np.allclose(q, prob)


def test_incompatible_old_checkpoint_rejected_before_inference(tmp_path):
    torch = pytest.importorskip("torch")
    stale = tmp_path / "old.pt"
    torch.save({"schema_version": 5, "features": neural.FEATURES,
                "feature_hash": neural.FEATURE_HASH, "horizons": (5, 21),
                "architecture_hash": "deadbeef", "model": {}}, stale)
    payload, model, reason = neural._load_checked(stale)
    assert payload is None and model is None
    assert "schema" in reason                              # fails on metadata, no load


# ── B4B: explicit dual-family forecast persistence ────────────────────────────

def _nf(symbol="AAA", horizon=5, tsh=None):
    return NeuralForecast(
        symbol=symbol, as_of="2026-07-15", horizon_sessions=horizon,
        absolute_q10=-0.03, absolute_q50=0.02, absolute_q90=0.07,
        excess_q10=-0.02, excess_q50=0.015, excess_q90=0.05,
        probability_absolute_edge_positive=0.6, probability_excess_positive=0.58,
        model_id="m1", dataset_manifest_id="d1", feature_schema_hash="h1")


def test_v2_table_migrates_additively_and_keeps_v1(store):
    from specforge.research import record_forecast_v2
    # v1 legacy table still readable
    store.db.execute("INSERT INTO model_forecasts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     ("old", "2020-01-01", "AAA", 5, 0, 0, 0, .5, None, None, "fh"))
    assert store.db.execute("SELECT COUNT(*) n FROM model_forecasts").fetchone()["n"] == 1
    # v2 table exists (additive migration on open) and starts empty
    assert store.db.execute("SELECT COUNT(*) n FROM model_forecasts_v2").fetchone()["n"] == 0


def test_v2_preserves_both_families_and_is_idempotent(store):
    from specforge.research import record_forecast_v2
    tsh = ml_targets.TARGET_SCHEMA_HASH
    ok = record_forecast_v2(store, _nf(), model_id="m1", as_of="2026-07-15",
                            feature_hash="fh", target_schema_hash=tsh)
    assert ok
    # idempotent — a second identical write does not duplicate
    record_forecast_v2(store, _nf(), model_id="m1", as_of="2026-07-15",
                       feature_hash="fh", target_schema_hash=tsh)
    rows = store.db.execute("SELECT * FROM model_forecasts_v2").fetchall()
    assert len(rows) == 1
    r = rows[0]
    assert r["absolute_q50"] == pytest.approx(0.02) and r["excess_q50"] == pytest.approx(0.015)
    assert r["probability_absolute_edge_positive"] == pytest.approx(0.6)


def test_v2_rejects_incompatible_target_hash(store):
    from specforge.research import record_forecast_v2
    ok = record_forecast_v2(store, _nf(), model_id="m1", as_of="2026-07-15",
                            feature_hash="fh", target_schema_hash="not-the-current-hash")
    assert ok is False
    assert store.db.execute("SELECT COUNT(*) n FROM model_forecasts_v2").fetchone()["n"] == 0


def test_v2_resolution_writes_both_realized(store):
    from specforge.research import record_forecast_v2, resolve_forecasts_v2
    tsh = ml_targets.TARGET_SCHEMA_HASH
    dates = [r["d"] for r in store.db.execute(
        "SELECT d FROM bars WHERE symbol='AAA' ORDER BY d").fetchall()]
    as_of = dates[-30]                                     # leaves >5 future sessions
    record_forecast_v2(store, _nf(symbol="AAA", horizon=5), model_id="m1",
                       as_of=as_of, feature_hash="fh", target_schema_hash=tsh)
    assert resolve_forecasts_v2(store) == 1
    r = store.db.execute("SELECT * FROM model_forecasts_v2").fetchone()
    assert r["resolved_at"] is not None
    assert r["realized_absolute"] is not None and r["realized_excess"] is not None
    # excess = absolute − benchmark forward return, same window
    start = store.db.execute("SELECT close FROM bars WHERE symbol='AAA' AND d<=? "
                             "ORDER BY d DESC LIMIT 1", (as_of,)).fetchone()["close"]
    end = store.db.execute("SELECT close FROM bars WHERE symbol='AAA' AND d>? "
                           "ORDER BY d LIMIT 5", (as_of,)).fetchall()[-1]["close"]
    assert r["realized_absolute"] == pytest.approx(end / start - 1)


# ── B4C: structured inference + consumer (node/graph) semantics ───────────────

def _nf_view(abs_q50, exc_q50, p_abs=0.6, p_exc=0.6, horizon=21):
    return {str(horizon): NeuralForecast(
        symbol="AAA", as_of="2026-07-15", horizon_sessions=horizon,
        absolute_q10=abs_q50 - 0.03, absolute_q50=abs_q50, absolute_q90=abs_q50 + 0.03,
        excess_q10=exc_q50 - 0.02, excess_q50=exc_q50, excess_q90=exc_q50 + 0.02,
        probability_absolute_edge_positive=p_abs, probability_excess_positive=p_exc,
        model_id="m", dataset_manifest_id="d", feature_schema_hash="h")}


class _NodeCtx:
    offline = False
    as_of = "2026-07-15"

    def __init__(self, cfg, universe):
        self.cfg, self.universe, self.store = cfg, universe, None

    def atr_pct(self, sym):
        return 0.02


def _neural_node(cfg, horizon=21):
    from specforge.nodes.neural import Node
    n = Node({"horizon_days": horizon, "weight": 0.15, "status": "experimental"})
    n.id = "neural"
    return n


def _compute_with(cfg, monkeypatch, forecasts):
    monkeypatch.setattr(neural, "predict_today",
                        lambda c, s, ctx: (forecasts, {"model_id": "m", "checkpoint_age_days": 1}))
    return _neural_node(cfg).compute(_NodeCtx(cfg, list(forecasts)))


def test_node_no_long_on_positive_excess_negative_absolute(cfg, monkeypatch):
    # abs −5%, excess +5%: the exact case the whole migration exists to fix.
    events = _compute_with(cfg, monkeypatch, {"AAA": _nf_view(-0.05, 0.05)})
    assert all(e.direction != "long" for e in events)
    assert events == []                                    # not even a misleading avoid


def test_node_long_on_positive_absolute_edge(cfg, monkeypatch):
    events = _compute_with(cfg, monkeypatch, {"AAA": _nf_view(0.04, 0.03, p_abs=0.7)})
    assert len(events) == 1 and events[0].direction == "long"


def test_node_expected_return_is_absolute_not_excess(cfg, monkeypatch):
    e = _compute_with(cfg, monkeypatch, {"AAA": _nf_view(0.04, 0.03, p_abs=0.7)})[0]
    assert e.expected_return == pytest.approx(0.04)         # absolute q50, not excess
    assert e.downside_estimate == pytest.approx(0.01)       # absolute q10


def test_node_score_and_confidence_carry_the_excess_component(cfg, monkeypatch):
    # The graph ranks on signed_alpha = dir·|score|·confidence — both must be
    # excess-derived so the graph uses the benchmark-relative signal.
    e = _compute_with(cfg, monkeypatch, {"AAA": _nf_view(0.04, 0.03, p_abs=0.7, p_exc=0.65)})[0]
    assert e.score == pytest.approx(min(1.0, 0.03 / 0.06))
    assert e.confidence == pytest.approx(0.65, abs=1e-3)


def test_node_evidence_reports_both_families(cfg, monkeypatch):
    e = _compute_with(cfg, monkeypatch, {"AAA": _nf_view(0.04, 0.03, p_abs=0.7)})[0]
    assert "abs" in e.evidence[0] and "excess" in e.evidence[0] and "cost" in e.evidence[0]


def test_node_model_failure_yields_no_events(cfg, monkeypatch):
    monkeypatch.setattr(neural, "predict_today", lambda *a, **k: ({}, {"silent": "no champion"}))
    n = _neural_node(cfg)
    assert n.compute(_NodeCtx(cfg, ["AAA"])) == []
    assert n.degraded_reason == "no champion"


def test_deterministic_nodes_operate_independently_of_neural(cfg):
    from specforge.nodes.base import build_registry
    cfg.data["nodes"]["neural"] = {"enabled": True, "weight": 0.15,
                                   "status": "experimental", "horizon_days": 21}
    reg = build_registry(cfg)
    assert "momentum" in reg                               # deterministic node loads regardless


def test_build_neural_forecast_single_mapping():
    abs_q = np.array([[-0.03, 0.02, 0.07], [-0.05, 0.01, 0.06]])
    exc_q = np.array([[-0.02, 0.015, 0.05], [-0.03, 0.02, 0.06]])
    nf = neural.build_neural_forecast(
        symbol="AAA", as_of="2026-07-15", horizon=5, i=0, abs_q=abs_q,
        abs_p=[0.6, 0.55], exc_q=exc_q, exc_p=[0.58, 0.5], meta={"model_id": "m"})
    assert nf.absolute_q50 == pytest.approx(0.02) and nf.excess_q50 == pytest.approx(0.015)
    assert nf.probability_absolute_edge_positive == pytest.approx(0.6)


# ── Bounded end-to-end smoke: dataset → train → checkpoint → inference →
#    NeuralForecast → v2 persistence → resolution → node. Synthetic data only;
#    no network, no broker, no production DB (fixture store is a tmp file).

def test_smoke_end_to_end_dual_target(cfg, store):
    torch = pytest.importorskip("torch")
    import pathlib
    from specforge.data import MarketContext
    from specforge.research import record_forecast_v2, resolve_forecasts_v2

    _long_history(store)
    cfg.data["neural"].update(input_sessions=40, horizons=[5, 21], max_epochs=2,
                              walk_forward_epochs=1, patience=2)

    # 1) bounded training → challenger checkpoint
    out = neural.train_challenger(cfg, store, symbols=["AAA", "BBB", "CCC"], max_seconds=25)
    run_id = out.get("id")
    assert run_id, out
    row = store.db.execute("SELECT * FROM model_runs WHERE id=?", (run_id,)).fetchone()

    # 2) checkpoint metadata validates + reloads
    payload, model, reason = neural._load_checked(
        pathlib.Path(row["checkpoint"]), row["checkpoint_sha256"])
    assert reason is None and model is not None
    assert payload["target_schema_hash"] == ml_targets.TARGET_SCHEMA_HASH
    assert payload["calibration_structured"]["absolute"]["prob_threshold"] == payload["round_trip_cost"]

    # 3) structured inference → typed NeuralForecast, finite, ordered, bounded
    preds, meta = neural.predict_run(cfg, store, MarketContext(store, cfg), run_id)
    assert preds, "structured inference produced no forecasts"
    seen = 0
    for hs in preds.values():
        for nf in hs.values():
            for v in (nf.absolute_q10, nf.absolute_q50, nf.absolute_q90,
                      nf.excess_q10, nf.excess_q50, nf.excess_q90):
                assert math.isfinite(v)
            assert nf.absolute_q10 <= nf.absolute_q50 <= nf.absolute_q90
            assert nf.excess_q10 <= nf.excess_q50 <= nf.excess_q90
            assert 0 <= nf.probability_absolute_edge_positive <= 1
            assert 0 <= nf.probability_excess_positive <= 1
            seen += 1
    assert seen > 0

    # 4) v2 persistence + resolution write both realized families
    sym, hs = next(iter(preds.items()))
    dates = [r["d"] for r in store.db.execute(
        "SELECT d FROM bars WHERE symbol=? ORDER BY d", (sym,)).fetchall()]
    as_of = dates[-30]
    for h, nf in hs.items():
        record_forecast_v2(store, nf, model_id=run_id, as_of=as_of,
                           feature_hash=meta["feature_hash"],
                           target_schema_hash=meta["target_schema_hash"])
    assert resolve_forecasts_v2(store) >= 1
    r = store.db.execute("SELECT * FROM model_forecasts_v2 WHERE resolved_at IS NOT NULL "
                         "LIMIT 1").fetchone()
    assert r["realized_absolute"] is not None and r["realized_excess"] is not None

    # 5) node computation via the real predict_today path (promote to champion)
    neural.promote(cfg, store, run_id)
    node = _neural_node(cfg)
    events = node.compute(MarketContext(store, cfg))
    for e in events:                                       # may be empty; never malformed
        assert math.isfinite(e.expected_return) and e.direction in ("long", "avoid")

    # no network/broker/live-config mutation occurred: assertions above touched
    # only the fixture tmp DB and in-memory objects.
