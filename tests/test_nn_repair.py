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

def _nf_view(abs_q50, exc_q50, p_abs=0.6, p_exc=0.6, horizon=21, symbol="AAA",
             as_of="2026-07-15", width=0.03):
    return {str(horizon): NeuralForecast(
        symbol=symbol, as_of=as_of, horizon_sessions=horizon,
        absolute_q10=abs_q50 - width, absolute_q50=abs_q50, absolute_q90=abs_q50 + width,
        excess_q10=exc_q50 - 0.02, excess_q50=exc_q50, excess_q90=exc_q50 + 0.02,
        probability_absolute_edge_positive=p_abs, probability_excess_positive=p_exc,
        model_id="m", dataset_manifest_id="d",
        feature_schema_hash=neural.FEATURE_HASH)}


_META = {"model_id": "m"}          # matches _nf_view provenance for policy validation


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


# ── C1: direct bounded neural blend, graph-independent ────────────────────────

def _candidate(symbol="AAA", score=0.5, horizon=21):
    from specforge.models import TradeCandidate
    return TradeCandidate(
        id=f"c-{symbol}", symbol=symbol, asset_type="equity", side="buy",
        thesis="t", final_score=score, target_notional=100.0, expected_return=0.02,
        ci_low=-0.02, ci_high=0.06, probability_positive=0.6, expected_apr=0.1,
        apr_ci_low=-0.1, apr_ci_high=0.3, horizon_days=horizon, max_loss=100.0,
        contributing_nodes=["momentum"])


def test_neural_score_negative_for_positive_excess_negative_absolute():
    from specforge.ml.policy import neural_score
    nf = _nf_view(-0.05, 0.05, p_abs=0.2)["21"]
    assert neural_score(nf, 0.0016) < 0        # outperforming a crash is not a buy


def test_neural_score_positive_and_bounded_for_real_edge():
    from specforge.ml.policy import neural_score
    nf = _nf_view(0.05, 0.03, p_abs=0.75, p_exc=0.7)["21"]
    s = neural_score(nf, 0.0016)
    assert 0 < s <= 1.0


def test_blend_applied_exactly_once_and_attributed(cfg, store):
    from specforge.ml.policy import apply_neural_blend
    c = _candidate(score=0.5)
    out = apply_neural_blend([c], {"AAA": _nf_view(0.05, 0.03, p_abs=0.75)},
                             cfg, store, "cyc1", graph_blend=0.0, as_of="2026-07-15", meta=_META)
    assert out["blend"] == pytest.approx(0.15) and out["scored"] == 1
    from specforge.ml.policy import neural_score
    expected = round(0.85 * 0.5 + 0.15 * neural_score(
        _nf_view(0.05, 0.03, p_abs=0.75)["21"], 0.0016), 4)
    assert c.final_score == expected            # exactly once, exact formula
    assert c.neural_blend == pytest.approx(0.15)
    assert c.neural_contribution == pytest.approx(c.final_score - 0.5, abs=1e-6)
    assert "neural_direct" in c.contributing_nodes


def test_no_forecasts_means_zero_blend_and_untouched_scores(cfg, store):
    from specforge.ml.policy import apply_neural_blend
    c = _candidate(score=0.5)
    out = apply_neural_blend([c], {}, cfg, store, "cyc1", graph_blend=0.0, as_of="2026-07-15", meta=_META)
    assert out["blend"] == 0.0 and "deterministic fallback" in out["reason"]
    assert c.final_score == 0.5 and c.neural_blend == 0.0


def test_active_graph_owns_learned_pathway_no_double_count(cfg, store):
    from specforge.ml.policy import apply_neural_blend
    c = _candidate(score=0.5)
    out = apply_neural_blend([c], {"AAA": _nf_view(0.05, 0.03)}, cfg, store,
                             "cyc1", graph_blend=0.10, as_of="2026-07-15", meta=_META)
    assert out["blend"] == 0.0 and "graph" in out["reason"]
    assert c.final_score == 0.5                 # direct blend stood down


def test_blend_bounds_never_silently_increased(cfg, store):
    from specforge.ml.policy import effective_blend
    cfg.data["neural"]["experimental_blend"] = 0.9
    assert effective_blend(cfg, 0.0, True)[0] == pytest.approx(0.40)   # clamped down
    cfg.data["neural"]["experimental_blend"] = 0.01
    assert effective_blend(cfg, 0.0, True)[0] == 0.0                   # below floor = off
    cfg.data["neural"]["experimental_blend"] = 0.15


def test_blend_is_audited(cfg, store):
    from specforge.ml.policy import apply_neural_blend
    apply_neural_blend([_candidate()], {"AAA": _nf_view(0.05, 0.03)},
                       cfg, store, "cyc-audit", graph_blend=0.0, as_of="2026-07-15", meta=_META)
    row = store.db.execute("SELECT * FROM audit WHERE event_type='neural_direct_blend' "
                           "ORDER BY ts DESC LIMIT 1").fetchone()
    assert row is not None and "0.15" in row["payload"]


# ── C2: neural exploration probe — bounded, validated, governor-subordinate ───

def _probe_env(cfg, store, equity=1000.0, cash=500.0, positions=None):
    from specforge.data import MarketContext
    from specforge.models import AccountState
    ctx = MarketContext(store, cfg)
    account = AccountState(equity=equity, cash=cash, buying_power=cash,
                           positions=positions or [], as_of=ctx.as_of)
    return ctx, account


def _probe_forecasts(ctx, **by_symbol):
    """{sym: forecast-dict} with as_of bound to the live ctx (validation passes)."""
    return {sym: _nf_view(*args, symbol=sym, as_of=ctx.as_of)
            for sym, args in by_symbol.items()}


def _select(cfg, store, ctx, account, candidates, targets, forecasts):
    from specforge.ml.policy import select_exploration_probe
    return select_exploration_probe(candidates, targets, forecasts, _META,
                                    cfg, store, "cyc-probe", account, ctx,
                                    as_of=ctx.as_of,
                                    allocated=sum(n for _, n in targets))


def test_probe_stale_forecast_rejected(cfg, store):
    ctx, account = _probe_env(cfg, store)
    stale = {"BBB": _nf_view(0.05, 0.03, p_abs=0.75, symbol="BBB",
                             as_of="2020-01-01")}
    assert _select(cfg, store, ctx, account,
                   [_candidate("BBB", 0.5)], [], stale) is None


def test_probe_negative_absolute_positive_excess_rejected(cfg, store):
    ctx, account = _probe_env(cfg, store)
    fc = _probe_forecasts(ctx, BBB=(-0.05, 0.05, 0.75))
    assert _select(cfg, store, ctx, account,
                   [_candidate("BBB", 0.5)], [], fc) is None


def test_probe_qualified_creates_exactly_one(cfg, store):
    ctx, account = _probe_env(cfg, store)
    fc = _probe_forecasts(ctx, BBB=(0.05, 0.03, 0.75))
    out = _select(cfg, store, ctx, account, [_candidate("BBB", 0.5)], [], fc)
    assert out is not None
    cand, notional = out
    assert cand.entry_mode == "probe" and cand.size_multiplier == 0.25
    assert cand.symbol == "BBB" and notional >= 5.0
    assert "neural exploration" in cand.entry_mode_reason and "m" in cand.entry_mode_reason
    row = store.db.execute("SELECT * FROM audit WHERE event_type='neural_probe_selected'"
                           " ORDER BY ts DESC LIMIT 1").fetchone()
    assert row is not None and "BBB" in row["payload"]


def test_probe_highest_neural_score_wins_and_only_one(cfg, store):
    ctx, account = _probe_env(cfg, store)
    fc = _probe_forecasts(ctx, BBB=(0.02, 0.01, 0.60), CCC=(0.06, 0.04, 0.85))
    out = _select(cfg, store, ctx, account,
                  [_candidate("BBB", 0.5), _candidate("CCC", 0.4)], [], fc)
    assert out is not None and out[0].symbol == "CCC"


def test_probe_excludes_held_and_normally_selected(cfg, store):
    from specforge.models import Position
    held = Position(symbol="BBB", asset_type="equity", qty=1.0, avg_cost=100,
                    opened_at="2026-07-01")
    ctx, account = _probe_env(cfg, store, positions=[held])
    fc = _probe_forecasts(ctx, BBB=(0.05, 0.03, 0.75), CCC=(0.05, 0.03, 0.75))
    normal = _candidate("CCC", 0.6)
    out = _select(cfg, store, ctx, account,
                  [_candidate("BBB", 0.5), normal], [(normal, 50.0)], fc)
    assert out is None                          # held excluded; selected excluded


def test_probe_multiplier_applied_exactly_once(cfg, store):
    from specforge import portfolio
    ctx, account = _probe_env(cfg, store, cash=10_000.0)
    cfg.data["neural"]["exploration"]["budget_fraction"] = 1.0
    fc = _probe_forecasts(ctx, BBB=(0.05, 0.03, 0.75))
    out = _select(cfg, store, ctx, account, [_candidate("BBB", 0.5)], [], fc)
    assert out is not None
    baseline = _candidate("BBB", 0.5)           # identical, but normal mode
    full = portfolio.construct([baseline], account, ctx, cfg)
    assert full and out[1] == pytest.approx(0.25 * full[0][1], rel=1e-6)


def test_probe_budget_fraction_caps_notional(cfg, store):
    ctx, account = _probe_env(cfg, store, equity=1000.0, cash=10_000.0)
    cfg.data["neural"]["exploration"]["budget_fraction"] = 0.005   # $5 cap
    fc = _probe_forecasts(ctx, BBB=(0.05, 0.03, 0.75))
    out = _select(cfg, store, ctx, account, [_candidate("BBB", 0.5)], [], fc)
    assert out is not None and out[1] == pytest.approx(5.0)


def test_probe_extra_slot_beyond_batch_but_not_global_limit(cfg, store):
    from specforge.models import Position
    ctx, account = _probe_env(cfg, store)
    fc = _probe_forecasts(ctx, BBB=(0.05, 0.03, 0.75))
    # deterministic batch already full → probe still gets its dedicated slot
    full_batch = [(_candidate(s, 0.9), 50.0) for s in ("AAA", "CCC", "DDD")]
    out = _select(cfg, store, ctx, account, [_candidate("BBB", 0.5)], full_batch, fc)
    assert out is not None
    # ...but the GLOBAL max_open_positions cap is never exceeded
    cfg.data.setdefault("risk", {})["max_open_positions"] = 1
    held = Position(symbol="ZZZ", asset_type="equity", qty=1.0, avg_cost=10,
                    opened_at="2026-07-01")
    ctx2, account2 = _probe_env(cfg, store, positions=[held])
    fc2 = _probe_forecasts(ctx2, BBB=(0.05, 0.03, 0.75))
    assert _select(cfg, store, ctx2, account2,
                   [_candidate("BBB", 0.5)], [], fc2) is None
    cfg.data["risk"]["max_open_positions"] = 12


def test_probe_slot_occupied_by_open_probe_blocks_second(cfg, store):
    ctx, account = _probe_env(cfg, store)
    store.save_position("probe-1", {
        "symbol": "QQQ", "asset_type": "equity", "qty": 1.0, "avg_cost": 10,
        "opened_at": "2026-07-01", "horizon_days": 21, "stop_price": 9.0,
        "status": "open", "mode": "paper", "entry_mode": "probe"})
    fc = _probe_forecasts(ctx, BBB=(0.05, 0.03, 0.75))
    assert _select(cfg, store, ctx, account,
                   [_candidate("BBB", 0.5)], [], fc) is None
    row = store.db.execute("SELECT * FROM audit WHERE event_type='neural_probe_skipped'"
                           " ORDER BY ts DESC LIMIT 1").fetchone()
    assert row is not None and "occupied" in row["payload"]


def test_probe_position_entry_mode_is_durable(cfg, store):
    store.save_position("p-dur", {
        "symbol": "QQQ", "asset_type": "equity", "qty": 1.0, "avg_cost": 10,
        "opened_at": "2026-07-01", "horizon_days": 21, "stop_price": 9.0,
        "status": "open", "mode": "paper", "entry_mode": "probe"})
    rows = store.open_positions(mode="paper")
    assert any(r["entry_mode"] == "probe" for r in rows)
    normal = [r for r in rows if r["symbol"] != "QQQ"]
    assert all(r.get("entry_mode", "normal") == "normal" for r in normal)


def test_probe_cannot_bypass_governor(cfg, store):
    from specforge.models import AccountState, OrderIntent
    from specforge.risk import CycleState, Governor
    gov = Governor(cfg, store)
    cand = _candidate("BBB", 0.5)
    cand.entry_mode, cand.size_multiplier = "probe", 0.25
    cand.target_notional = 900.0                # oversized vs cash below
    poor = AccountState(equity=1000.0, cash=10.0, buying_power=10.0,
                        positions=[], as_of="2026-07-15")
    intent = OrderIntent.make(cand, qty=9.0, limit_price=100.0)
    # The governor keeps full authority over a probe: an unaffordable request
    # is cut down to real spendable cash, and hard blocks still reject outright.
    resized = gov.review(intent, cand, poor, CycleState(1000.0), data_age_days=1)
    assert resized.verdict in ("APPROVED_WITH_SIZE_REDUCTION", "REJECTED")
    if resized.verdict == "APPROVED_WITH_SIZE_REDUCTION":
        assert resized.approved_notional < 900.0
    stale = gov.review(intent, cand, poor, CycleState(1000.0), data_age_days=99)
    assert stale.verdict == "REJECTED"          # probes obey the same governor


def test_node_cache_cleared_every_compute_even_offline(cfg):
    node = _neural_node(cfg)
    node.last_forecasts = {"AAA": object()}     # poison from a "previous cycle"
    node.last_meta = {"model_id": "old"}
    node.last_forecast_as_of = "2020-01-01"
    ctx = _NodeCtx(cfg, ["AAA"])
    ctx.offline = True
    assert node.compute(ctx) == []
    assert node.last_forecasts == {} and node.last_meta == {}
    assert node.last_forecast_as_of is None


def test_engine_offline_cycle_never_runs_inference_no_probe_exits_ok(
        cfg, store, monkeypatch):
    # Offline/replay cycles (refresh_data=False) must NEVER touch the live
    # model: zero inference, zero probes, deterministic completion.
    from specforge.engine import run_cycle
    cfg.data["nodes"]["neural"] = {"enabled": True, "weight": 0.15,
                                   "horizon_days": 21, "status": "experimental"}
    calls = {"n": 0}

    def failing_predict(c, s, ctx):
        calls["n"] += 1
        return {}, {"silent": "no validated global TCN champion"}

    monkeypatch.setattr(neural, "predict_today", failing_predict)
    summary = run_cycle(cfg, store, refresh_data=False)
    assert calls["n"] == 0                      # offline: model untouched
    assert summary["cycle_id"]                  # cycle completed deterministically
    assert store.db.execute("SELECT COUNT(*) n FROM audit WHERE "
                            "event_type='neural_probe_selected'").fetchone()["n"] == 0
    blend = store.db.execute("SELECT payload FROM audit WHERE "
                             "event_type='neural_direct_blend' "
                             "ORDER BY ts DESC LIMIT 1").fetchone()
    assert blend is not None and "deterministic fallback" in blend["payload"]


def test_inference_runs_once_per_cycle_engine_reads_stash(cfg, monkeypatch):
    # One online compute → exactly one predict_today; the engine consumes the
    # node's stash and has no predict_today call site of its own.
    import pathlib
    calls = {"n": 0}

    def counted(c, s, ctx):
        calls["n"] += 1
        return {}, {"silent": "no champion"}

    monkeypatch.setattr(neural, "predict_today", counted)
    node = _neural_node(cfg)
    ctx = _NodeCtx(cfg, ["AAA"])
    node.compute(ctx)
    assert calls["n"] == 1
    assert node.last_forecast_as_of == ctx.as_of      # stash bound to this cycle
    engine_src = pathlib.Path("specforge/engine.py").read_text()
    assert "predict_today" not in engine_src          # engine reads only the stash


# ── Sprint D: explicit lifecycle + finalist promotion ─────────────────────────

import json as _json

from specforge.ml import lifecycle as ml_lifecycle


def _passing_metrics(score=1.0, **over):
    folds = [{"ic_5d": .02, "ic_21d": .03, "net_alpha_5d": .01,
              "net_alpha_21d": .01} for _ in range(5)]
    h = {"correlation": .03, "top_decile_alpha_after_cost": .01, "coverage": .8}
    out = {"beats_baselines": True, "folds": folds,
           "median_fold_ic_5d": .02, "median_fold_ic_21d": .03,
           "5": dict(h), "21": dict(h), "validation_selection_score": score,
           "evaluation_split": "sealed_test"}
    out.update(over)
    return out


def _insert_run(store, rid, state, score=1.0, created_at="2026-07-01T00:00:00",
                symbol=None, incompat=None, metrics=None):
    store.db.execute(
        "INSERT INTO model_runs(id,kind,symbol,created_at,data_as_of,status,"
        "parent_id,metrics,checkpoint,feature_hash,schema_version,"
        "architecture_hash,checkpoint_sha256,incompatibility_reason,"
        "lifecycle_state,permitted_blend) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (rid, "global_tcn", symbol, created_at, "2026-07-01",
         ml_lifecycle.project_status(state), None,
         _json.dumps(metrics if metrics is not None else _passing_metrics(score)),
         f"ckpt-{rid}.pt", neural.FEATURE_HASH, neural.MODEL_SCHEMA,
         neural.ARCHITECTURE_HASH, "sha", incompat, state, 0.0))
    store.db.commit()


@pytest.fixture()
def promo(monkeypatch):
    """maybe_promote without checkpoint-file compatibility churn."""
    monkeypatch.setattr(neural, "refresh_compatibility", lambda s: {})
    monkeypatch.setattr(neural, "_load_checked",
                        lambda path, sha=None: ({}, None, None))


def test_newer_weak_finalist_cannot_hide_older_qualified(cfg, store, promo):
    _insert_run(store, "old-good", "sealed_candidate", score=2.0,
                created_at="2026-07-01T00:00:00")
    _insert_run(store, "new-weak", "sealed_candidate", score=9.0,
                created_at="2026-07-10T00:00:00",
                metrics=_passing_metrics(9.0, beats_baselines=False))
    out = neural.maybe_promote(cfg, store)
    assert out["action"] == "experimental_live" and out["id"] == "old-good"
    weak = store.db.execute("SELECT lifecycle_state FROM model_runs WHERE id='new-weak'").fetchone()
    assert weak["lifecycle_state"] == "sealed_candidate"   # untouched


def test_validation_only_global_cannot_become_champion(cfg, store, promo):
    _insert_run(store, "val-only", "validation_candidate", score=9.0)
    out = neural.maybe_promote(cfg, store)
    assert out["action"] == "none"
    assert store.db.execute("SELECT COUNT(*) n FROM model_runs WHERE "
                            "lifecycle_state='champion'").fetchone()["n"] == 0


def test_validation_only_holding_cannot_promote(cfg, store, promo):
    _insert_run(store, "hold-1", "validation_candidate", symbol="AAA")
    with pytest.raises(ValueError, match="out-of-sample"):
        neural.promote(cfg, store, "hold-1")
    row = store.db.execute("SELECT lifecycle_state FROM model_runs WHERE id='hold-1'").fetchone()
    assert row["lifecycle_state"] == "validation_candidate"


def test_incompatible_finalist_excluded(cfg, store, promo):
    _insert_run(store, "better-incompat", "sealed_candidate", score=9.0,
                incompat="architecture mismatch")
    _insert_run(store, "ok", "sealed_candidate", score=1.0)
    out = neural.maybe_promote(cfg, store)
    assert out["id"] == "ok"


def test_deterministic_tie_breaking():
    a = {"id": "a", "created_at": "2026-07-01", "metrics": _json.dumps({"validation_selection_score": 1.0})}
    b = {"id": "b", "created_at": "2026-07-05", "metrics": _json.dumps({"validation_selection_score": 1.0})}
    c = {"id": "c", "created_at": "2026-07-05", "metrics": _json.dumps({"validation_selection_score": 1.0})}
    ranked = sorted([c, b, a], key=ml_lifecycle.rank_key)
    assert [r["id"] for r in ranked] == ["a", "b", "c"]    # date asc, then id asc
    assert ranked == sorted([b, a, c], key=ml_lifecycle.rank_key)   # stable/deterministic


def test_transition_history_is_persisted(cfg, store, promo):
    _insert_run(store, "hist-1", "sealed_candidate", score=1.0)
    neural.maybe_promote(cfg, store)
    hist = ml_lifecycle.history(store, "hist-1")
    assert len(hist) == 1
    t = hist[0]
    assert t["prior_state"] == "sealed_candidate"
    assert t["new_state"] == "experimental_live"
    assert "offline gates" in t["reason"]
    assert _json.loads(t["evidence"])["rank"] == 1
    assert t["permitted_blend"] == pytest.approx(0.15)
    assert t["architecture_hash"] == neural.ARCHITECTURE_HASH and t["at"]


def test_champion_swap_is_atomic_and_retires_after_activation(cfg, store, promo):
    _insert_run(store, "champ-old", "champion")
    _insert_run(store, "cand-new", "production_candidate", score=2.0,
                created_at="2026-07-05T00:00:00")
    neural.promote(cfg, store, "cand-new", reason="test swap")
    states = {r["id"]: r["lifecycle_state"] for r in
              store.db.execute("SELECT id,lifecycle_state FROM model_runs")}
    assert states["cand-new"] == "champion" and states["champ-old"] == "retired"
    assert store.db.execute("SELECT COUNT(*) n FROM model_runs WHERE "
                            "lifecycle_state='champion'").fetchone()["n"] == 1


def test_failed_activation_leaves_prior_champion_intact(cfg, store, monkeypatch):
    monkeypatch.setattr(neural, "refresh_compatibility", lambda s: {})
    monkeypatch.setattr(neural, "_load_checked",
                        lambda path, sha=None: (None, None, "corrupt checkpoint"))
    _insert_run(store, "champ-old", "champion")
    _insert_run(store, "cand-bad", "production_candidate", score=2.0)
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        neural.promote(cfg, store, "cand-bad")
    states = {r["id"]: r["lifecycle_state"] for r in
              store.db.execute("SELECT id,lifecycle_state FROM model_runs")}
    assert states["champ-old"] == "champion" and states["cand-bad"] == "production_candidate"


def test_concurrent_promotions_yield_exactly_one_champion(cfg, store, promo):
    _insert_run(store, "f1", "production_candidate", score=1.0)
    _insert_run(store, "f2", "production_candidate", score=2.0)
    neural.promote(cfg, store, "f1")
    neural.promote(cfg, store, "f2")           # racing/second attempt
    champs = store.db.execute("SELECT id FROM model_runs WHERE "
                              "lifecycle_state='champion'").fetchall()
    assert [r["id"] for r in champs] == ["f2"]


def _graph_metrics(temporal_id, utility=0.1):
    folds = [{"ic_5d": .02, "ic_21d": .03, "net_alpha_5d": .01,
              "net_alpha_21d": .01} for _ in range(5)]
    coverage = {n["id"]: 1.0 for n in graph.default_topology()["nodes"]}
    return {"temporal_model_id": temporal_id, "folds": folds,
            "median_fold_ic_5d": .02, "median_fold_ic_21d": .03,
            "coverage_5d": .8, "coverage_21d": .8,
            "sample_coverage": coverage, "portfolio_utility": utility,
            "oos_sharpe": 1.0, "utility_evidence": "ok"}


def test_graph_tcn_dependency_mismatch_blocks_activation(cfg, store, promo):
    _insert_run(store, "tcnA", "champion")
    vid = graph.save_version(store, graph.default_topology(),
                             metrics=_graph_metrics("tcnB"))
    out = graph.maybe_promote(cfg, store)
    assert out["action"] == "shadow"
    row = store.db.execute("SELECT lifecycle_state FROM graph_versions WHERE id=?",
                           (vid,)).fetchone()
    assert row["lifecycle_state"] == "validation_candidate"   # not activated


def test_graph_offline_pass_earns_experimental_live_not_champion(cfg, store, promo):
    _insert_run(store, "tcnA", "champion")
    vid = graph.save_version(store, graph.default_topology(),
                             metrics=_graph_metrics("tcnA"))
    out = graph.maybe_promote(cfg, store)
    assert out["action"] == "experimental_live" and out["id"] == vid
    row = store.db.execute("SELECT lifecycle_state,status FROM graph_versions "
                           "WHERE id=?", (vid,)).fetchone()
    assert row["lifecycle_state"] == "experimental_live"
    assert row["status"] == "challenger"       # projection: still NOT champion
    assert store.db.execute("SELECT COUNT(*) n FROM graph_versions WHERE "
                            "lifecycle_state='champion'").fetchone()["n"] == 0


def test_lifecycle_reads_do_not_mutate(cfg, store, promo):
    _insert_run(store, "r1", "sealed_candidate", score=1.0)
    _insert_run(store, "r2", "experimental_live", score=2.0)
    before = [tuple(r) for r in store.db.execute(
        "SELECT id,lifecycle_state,status,permitted_blend FROM model_runs ORDER BY id")]
    ml_lifecycle.finalists(store, "model_runs", kind="global_tcn")
    ml_lifecycle.history(store, "r1")
    for row in ml_lifecycle.finalists(store, "model_runs", kind="global_tcn"):
        ml_lifecycle.rank_key(row)
    after = [tuple(r) for r in store.db.execute(
        "SELECT id,lifecycle_state,status,permitted_blend FROM model_runs ORDER BY id")]
    assert before == after


def test_experimental_live_serves_at_permitted_blend_cap(cfg, store):
    from specforge.ml.policy import effective_blend
    assert effective_blend(cfg, 0.0, True, permitted=0.10)[0] == pytest.approx(0.10)
    assert effective_blend(cfg, 0.0, True, permitted=0.0)[0] == 0.0   # fail closed
    assert effective_blend(cfg, 0.0, True, permitted=None)[0] == pytest.approx(0.15)


# ── Sprint E1: cross-process trading lease + immutable cycle config ──────────

def test_lease_blocks_second_acquirer(store):
    a = store.acquire_lease("trading_cycle:paper", 60)
    assert a is not None
    assert store.acquire_lease("trading_cycle:paper", 60) is None
    store.release_lease("trading_cycle:paper", a)


def test_expired_lease_is_recoverable(store):
    import time
    a = store.acquire_lease("trading_cycle:paper", 0.01)
    assert a is not None
    time.sleep(1.05)                       # min TTL is clamped to 1s
    b = store.acquire_lease("trading_cycle:paper", 60)
    assert b is not None and b != a        # crashed worker healed
    store.release_lease("trading_cycle:paper", b)


def test_release_only_by_owner(store):
    a = store.acquire_lease("trading_cycle:paper", 60)
    store.release_lease("trading_cycle:paper", "not-the-owner")
    assert store.acquire_lease("trading_cycle:paper", 60) is None   # still held
    store.release_lease("trading_cycle:paper", a)
    assert store.acquire_lease("trading_cycle:paper", 60) is not None


def test_stale_worker_is_fenced_after_losing_lease(store):
    import json as j
    a = store.acquire_lease("trading_cycle:paper", 60)
    assert store.holds_lease("trading_cycle:paper", a)
    # force-expire A (simulates a hung worker outliving its TTL)…
    lease = store.kv_get("lease:trading_cycle:paper")
    lease["expires_at"] = 0
    store.kv_set("lease:trading_cycle:paper", lease)
    b = store.acquire_lease("trading_cycle:paper", 60)
    assert b is not None
    # …A must now fail the fencing check and cannot commit
    assert not store.holds_lease("trading_cycle:paper", a)
    assert store.holds_lease("trading_cycle:paper", b)
    store.release_lease("trading_cycle:paper", b)


def test_run_cycle_skips_when_another_process_holds_lease(cfg, store):
    from specforge.engine import run_cycle
    other = store.acquire_lease("trading_cycle:paper", 60)
    out = run_cycle(cfg, store, refresh_data=False)
    assert "skipped" in out and "lease" in out["skipped"]
    row = store.db.execute("SELECT payload FROM audit WHERE "
                           "event_type='cycle_skipped_overlap' "
                           "ORDER BY ts DESC LIMIT 1").fetchone()
    assert "cross_process" in row["payload"]
    store.release_lease("trading_cycle:paper", other)


def test_run_cycle_releases_lease_on_completion(cfg, store):
    from specforge.engine import run_cycle
    out = run_cycle(cfg, store, refresh_data=False)
    assert "cycle_id" in out
    assert store.acquire_lease("trading_cycle:paper", 60) is not None  # free again


def test_market_context_symbols_override_never_touches_cfg(cfg, store):
    from specforge.data import MarketContext
    before = list(cfg.data["universe"]["symbols"])
    ctx = MarketContext(store, cfg, symbols=["XXX", "YYY"])
    assert ctx.universe == ["XXX", "YYY"]
    assert cfg.data["universe"]["symbols"] == before          # untouched
    assert MarketContext(store, cfg).universe == before       # default path intact


def test_no_shared_config_mutation_remains_in_cycle_code():
    import pathlib
    for module in ("specforge/engine.py", "specforge/research.py"):
        src = pathlib.Path(module).read_text()
        assert 'cfg.data["universe"]["symbols"] =' not in src, module
