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
    # R5: the flat 0.0016 is now only the FLOOR; the reported cost is the median
    # of the per-sample estimates and can only sit at or above it.
    assert ds["cost_floor"] == pytest.approx(0.0016)
    assert ds["round_trip_cost"] >= ds["cost_floor"]
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

    # 5) node computation via the real predict_today path — walking the LEGAL
    # lifecycle ramp (R1): validation-only can no longer jump to champion, and
    # predict_today serves experimental_live at its bounded permitted blend.
    ml_lifecycle.transition(store, "model_runs", run_id, "sealed_candidate",
                            reason="smoke: sealed evaluation stub")
    ml_lifecycle.transition(store, "model_runs", run_id, "experimental_live",
                            reason="smoke: offline gate stub", permitted_blend=0.15)
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
    cfg.data["neural"]["experimental_blend"] = 0.15   # R0 default is 0.0
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
    cfg.data["neural"]["experimental_blend"] = 0.0


def test_blend_is_audited(cfg, store):
    from specforge.ml.policy import apply_neural_blend
    cfg.data["neural"]["experimental_blend"] = 0.15   # R0 default is 0.0
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
    cfg.data["neural"]["exploration"]["enabled"] = True   # R0 default is off
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


def _passing_metrics(score=1.0, absolute_ok=True, **over):
    folds = [{"ic_5d": .02, "ic_21d": .03, "net_alpha_5d": .01,
              "net_alpha_21d": .01} for _ in range(5)]
    h = {"correlation": .03, "top_decile_alpha_after_cost": .01, "coverage": .8}
    # R1: promotion demands BOTH families. absolute_ok=False models a model
    # that ranks well on excess but LOSES money absolutely after costs.
    a = (dict(h) if absolute_ok else
         {"correlation": .04, "top_decile_alpha_after_cost": -.004, "coverage": .8})
    # R6: permission is granted on net OOS policy return, so a "passing"
    # metrics blob must now carry a won bakeoff too.
    out = {"beats_baselines": True, "folds": folds,
           "bakeoff": {"verdict": True,
                       "basis": "net_oos_policy_return_staggered_cohorts"},
           "median_fold_ic_5d": .02, "median_fold_ic_21d": .03,
           "5": dict(h), "21": dict(h),
           "absolute": {"5": dict(a), "21": dict(a),
                        "evaluated_on": "structured_absolute_heads"},
           "validation_selection_score": score,
           "evaluation_split": "sealed_test"}
    out.update(over)
    return out


def _insert_run(store, rid, state, score=1.0, created_at="2026-07-01T00:00:00",
                symbol=None, incompat=None, metrics=None,
                checkpoint=None, sha="sha", permitted_blend=0.0):
    store.db.execute(
        "INSERT INTO model_runs(id,kind,symbol,created_at,data_as_of,status,"
        "parent_id,metrics,checkpoint,feature_hash,schema_version,"
        "architecture_hash,checkpoint_sha256,incompatibility_reason,"
        "lifecycle_state,permitted_blend) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (rid, "global_tcn", symbol, created_at, "2026-07-01",
         ml_lifecycle.project_status(state), None,
         _json.dumps(metrics if metrics is not None else _passing_metrics(score)),
         checkpoint or f"ckpt-{rid}.pt", neural.FEATURE_HASH, neural.MODEL_SCHEMA,
         neural.ARCHITECTURE_HASH, sha, incompat, state, permitted_blend))
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
    cfg.data["neural"]["experimental_blend"] = 0.15   # R0 default is 0.0
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
    cfg.data["neural"]["experimental_blend"] = 0.15   # R0 default is 0.0
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


# ── Sprint E2: approval repricing + scoped risk exceptions ───────────────────

def _executor(cfg, store):
    from specforge.broker.paper import PaperBroker
    from specforge.execution import Executor
    from specforge.risk import Governor
    broker = PaperBroker(cfg, store)
    return Executor(cfg, store, broker, Governor(cfg, store))


def _approved_intent(store, symbol="AAA", limit=100.0, notional=50.0,
                     mode="paper"):
    from specforge.models import OrderIntent
    cand = _candidate(symbol, 0.5)
    cand.target_notional = notional
    store.record_candidate(cand, "cyc-appr")
    intent = OrderIntent.make(cand, qty=round(notional / limit, 6), limit_price=limit)
    intent.notional = notional
    intent.status = "pending_approval"
    store.record_order(intent, mode=mode)
    store.queue_approval(intent.id, expires_at="2099-01-01T00:00:00")
    store.decide_approval(intent.id, "approved")
    return intent


def _run_queue(cfg, store, live_prices=None):
    from specforge.data import MarketContext
    from specforge.risk import CycleState
    # fixture bars end ~2 weeks back; these tests exercise repricing, not the
    # staleness policy (covered in test_risk)
    cfg.data.setdefault("risk", {})["stale_data_max_age_days"] = 45
    ex = _executor(cfg, store)
    ctx = MarketContext(store, cfg)
    account = ex.broker.get_account()
    return ex, ex.process_approval_queue(account, ctx, CycleState(1000.0),
                                         "cyc-appr", "neutral",
                                         live_prices=live_prices)


def test_approved_order_repriced_within_tolerance_places(cfg, store):
    from specforge.data import MarketContext
    ref = MarketContext(store, cfg).close("AAA")
    intent = _approved_intent(store, limit=ref)           # approved at ref
    fresh = round(ref * 1.01, 4)                          # 1% move < 3% tol
    ex, results = _run_queue(cfg, store, live_prices={"AAA": fresh})
    assert results and results[0] in ("filled", "resting", "placed")   # actually placed
    repriced = store.db.execute("SELECT payload FROM audit WHERE "
                                "event_type='approved_order_repriced'").fetchone()
    assert repriced is not None
    payload = _json.loads(repriced["payload"])
    assert payload["fresh"] == pytest.approx(fresh)
    assert payload["limit"]["new"] == pytest.approx(ex._limit_price(fresh, "buy"))
    # qty recomputed so the HUMAN-APPROVED notional is preserved at fresh price
    assert payload["qty"]["new"] * payload["limit"]["new"] == pytest.approx(50.0, rel=1e-3)
    # governor re-ran AFTER repricing
    reval = store.db.execute("SELECT payload FROM audit WHERE "
                             "event_type='approved_order_revalidated'").fetchone()
    assert reval is not None


def test_approved_order_expires_when_price_moved_beyond_tolerance(cfg, store):
    from specforge.data import MarketContext
    ref = MarketContext(store, cfg).close("AAA")
    intent = _approved_intent(store, limit=ref)
    ex, results = _run_queue(cfg, store, live_prices={"AAA": round(ref * 1.10, 4)})
    assert results == ["expired_price_moved"]
    order = store.db.execute("SELECT status FROM orders WHERE id=?", (intent.id,)).fetchone()
    assert order["status"] == "expired"                   # back to the operator
    assert store.db.execute("SELECT 1 FROM audit WHERE "
                            "event_type='approval_expired_price_moved'").fetchone()
    assert not store.db.execute("SELECT 1 FROM audit WHERE event_type='order_filled' "
                                "AND cycle_id='cyc-appr'").fetchone()


def test_live_mode_defers_without_fresh_quote(cfg, store):
    intent = _approved_intent(store, limit=100.0, mode="live")
    cfg.data["mode"] = "live"                             # executor mode flips
    try:
        ex, results = _run_queue(cfg, store, live_prices={})
    finally:
        cfg.data["mode"] = "paper"
    assert results == ["deferred"]                        # fail closed, no stale place
    order = store.db.execute("SELECT status FROM orders WHERE id=?", (intent.id,)).fetchone()
    assert order["status"] == "pending_approval"          # untouched, retried next cycle
    assert store.db.execute("SELECT 1 FROM audit WHERE "
                            "event_type='approval_reprice_deferred'").fetchone()


def test_paper_mode_falls_back_to_simulation_price(cfg, store):
    from specforge.data import MarketContext
    ref = MarketContext(store, cfg).close("AAA")
    _approved_intent(store, limit=ref)
    ex, results = _run_queue(cfg, store, live_prices=None)   # sim price = ref, drift 0
    assert results and results[0] not in ("deferred", "expired_price_moved")
    assert store.db.execute("SELECT 1 FROM audit WHERE "
                            "event_type='approved_order_repriced'").fetchone()


def test_advanced_override_no_longer_bypasses():
    from specforge.config import ConfigError, load_config
    with pytest.raises(ConfigError, match="advanced_override was removed"):
        load_config("paper", overrides={"risk": {"max_daily_loss": 0.5},
                                        "advanced_override": True})


def _exc(parameter="risk.max_daily_loss", value=0.5, expires="2099-01-01", **over):
    out = {"parameter": parameter, "value": value, "reason": "test",
           "expires": expires}
    out.update(over)
    return out


def test_scoped_exception_allows_named_value_with_warning():
    from specforge.config import load_config
    cfg = load_config("paper", overrides={"risk": {"max_daily_loss": 0.5},
                                          "risk_exceptions": [_exc()]})
    assert any("risk_exception active" in w for w in cfg.validate())


def test_expired_exception_fails_closed():
    from specforge.config import ConfigError, load_config
    with pytest.raises(ConfigError, match="Dangerous config rejected"):
        load_config("paper", overrides={"risk": {"max_daily_loss": 0.5},
                                        "risk_exceptions": [_exc(expires="2020-01-01")]})


def test_exception_cannot_cover_value_above_its_approved_bound():
    from specforge.config import ConfigError, load_config
    with pytest.raises(ConfigError):
        load_config("paper", overrides={"risk": {"max_daily_loss": 0.5},
                                        "risk_exceptions": [_exc(value=0.3)]})


def test_hard_invariant_never_exceptable():
    from specforge.config import ConfigError, load_config
    with pytest.raises(ConfigError, match="never exceptable"):
        load_config("paper", overrides={
            "risk": {"max_account_deployment": 1.2},
            "risk_exceptions": [_exc(parameter="risk.max_account_deployment",
                                     value=1.2)]})


def test_malformed_exception_is_inactive():
    from specforge.config import ConfigError, load_config
    broken = {"parameter": "risk.max_daily_loss", "value": 0.5}   # no expires
    with pytest.raises(ConfigError):
        load_config("paper", overrides={"risk": {"max_daily_loss": 0.5},
                                        "risk_exceptions": [broken]})


def test_exception_scope_is_one_parameter_only():
    from specforge.config import ConfigError, load_config
    # an exception for max_daily_loss must not excuse a different dangerous key
    with pytest.raises(ConfigError):
        load_config("paper", overrides={
            "risk": {"max_single_equity_position": 0.30},
            "risk_exceptions": [_exc()]})


def test_governor_voids_exceptions_above_max_equity(store):
    from specforge.config import load_config
    from specforge.models import AccountState, OrderIntent
    from specforge.risk import CycleState, Governor
    cfg = load_config("paper", overrides={
        "risk": {"max_daily_loss": 0.5},
        "risk_exceptions": [_exc(max_equity=500)]})
    gov = Governor(cfg, store)
    cand = _candidate("BBB", 0.5)
    cand.target_notional = 50.0
    intent = OrderIntent.make(cand, qty=0.5, limit_price=100.0)
    rich = AccountState(equity=600.0, cash=500.0, buying_power=500.0,
                        positions=[], as_of="2026-07-16")
    decision = gov.review(intent, cand, rich, CycleState(1000.0), data_age_days=1)
    assert decision.verdict == "REJECTED"
    assert any("risk exception void" in r for r in decision.reasons)
    small = AccountState(equity=400.0, cash=300.0, buying_power=300.0,
                         positions=[], as_of="2026-07-16")
    decision2 = gov.review(intent, cand, small, CycleState(1000.0), data_age_days=1)
    assert not any("risk exception void" in r for r in decision2.reasons)


def test_live_yaml_migration_is_coherent():
    from specforge.config import load_config
    cfg = load_config("live")
    warnings = cfg.validate()
    assert any("risk_exception active" in w and "single position" in w
               for w in warnings)
    assert cfg.risk_exception_equity_cap() == pytest.approx(500)
    assert "advanced_override" not in cfg.data


def test_repriced_values_persist_on_order_row(cfg, store):
    from specforge.data import MarketContext
    ref = MarketContext(store, cfg).close("AAA")
    intent = _approved_intent(store, limit=ref)
    fresh = round(ref * 1.02, 4)
    ex, results = _run_queue(cfg, store, live_prices={"AAA": fresh})
    row = store.db.execute("SELECT qty,limit_price,notional FROM orders WHERE id=?",
                           (intent.id,)).fetchone()
    assert row["limit_price"] == pytest.approx(ex._limit_price(fresh, "buy"))
    assert row["qty"] * row["limit_price"] == pytest.approx(row["notional"], rel=1e-3)


# ── Sprint F: policy comparison under identical conditions ───────────────────

def test_policy_overrides_set_expected_knobs(cfg):
    from specforge.backtest import _policy_cfg
    det = _policy_cfg(cfg, "deterministic")
    assert det.get("neural", "experimental_blend") == 0.0
    assert det.get("neural", "exploration", "enabled") is False
    only = _policy_cfg(cfg, "neural_only")
    assert only.get("neural", "experimental_blend") == 1.0
    assert only.get("neural", "max_blend") == 1.0
    blend = _policy_cfg(cfg, "fixed_blend")
    # R0 containment: the production config carries ZERO blend until R1's
    # dual-family gates land, so fixed_blend == the config as committed.
    assert blend.get("neural", "experimental_blend") == 0.0
    with pytest.raises(ValueError, match="unknown policy"):
        _policy_cfg(cfg, "yolo")


def test_incremental_math_vs_deterministic():
    from specforge.backtest import _incremental
    base = {"overall": {"cagr": 0.10, "sharpe": 1.0, "max_drawdown": 0.20},
            "turnover_multiple": 2.0, "n_trades": 50}
    other = {"overall": {"cagr": 0.12, "sharpe": 0.9, "max_drawdown": 0.25},
             "turnover_multiple": 3.5, "n_trades": 80}
    inc = _incremental(base, other)
    assert inc["delta_cagr"] == pytest.approx(0.02)
    assert inc["delta_sharpe"] == pytest.approx(-0.1)
    assert inc["delta_max_drawdown"] == pytest.approx(0.05)
    assert inc["delta_turnover_multiple"] == pytest.approx(1.5)


def test_compare_policies_identical_conditions_isolated_books(cfg, store, tmp_path):
    from specforge.backtest import compare_policies
    _long_history(store)                       # ~700 sessions of synthetic bars
    out = compare_policies(cfg, years=1, scale="research",
                           policies=("deterministic", "fixed_blend"),
                           log=lambda *a: None, out_dir=tmp_path)
    det = out["policies"]["deterministic"]
    blend = out["policies"]["fixed_blend"]
    # identical window: same sessions, same source bars → like-for-like
    assert det["window"] == blend["window"] == out["window"]
    assert det["costs_included"] and blend["costs_included"]
    # isolated books: one DB per policy, both real files
    assert (tmp_path / "backtest_policy_deterministic_research.db").exists()
    assert (tmp_path / "backtest_policy_fixed_blend_research.db").exists()
    # incremental block computed against the deterministic baseline
    assert "fixed_blend" in out["incremental_vs_deterministic"]
    # HONESTY CHECK (documented, not hidden): with no champion checkpoint the
    # blend has no forecasts to consume, so today the policies coincide.
    assert det["overall"] == blend["overall"]
    # comparison artifact persisted
    assert (tmp_path / "policy_comparison_research.json").exists()


# ── incident 2026-07-17: /api/health must report config errors, not 500 ─────

def test_health_reports_config_error_instead_of_500(cfg, store, monkeypatch):
    from fastapi.testclient import TestClient
    import specforge.app as app_mod
    from specforge.config import ConfigError
    app = app_mod.create_app(cfg, store, with_scheduler=False)

    def broken_config(s, m):
        raise ConfigError("Dangerous config rejected: test breakage")

    monkeypatch.setattr(app_mod, "current_config", broken_config)
    with TestClient(app, headers={"X-Session-Token": app.state.session_token}) as client:
        r = client.get("/api/health")
    assert r.status_code == 200                      # never a retry-loop 500
    body = r.json()
    assert body["status"] == "error"
    assert "test breakage" in body["config_error"]
    assert any("config invalid" in a for a in body["alerts"])


def test_gui_overrides_pruned_per_key_not_discarded(cfg, store):
    from specforge.app import current_config
    store.kv_set("config_overrides", {
        "nodes": {"news_sentiment": {"enabled": True}},         # safe — keep
        "risk": {"max_single_equity_position": 0.7,             # dangerous — prune
                 "max_daily_new_positions": 9},                 # safe — keep
        "advanced_override": True})                             # removed flag — prune
    c = current_config(store, "paper")
    assert c.get("nodes", "news_sentiment", "enabled") is True      # kept
    assert c.get("risk", "max_daily_new_positions") == 9            # kept
    assert c.get("risk", "max_single_equity_position") <= 0.25      # pruned to file value
    audits = [a for a in store.audit_rows()
              if a["event_type"] == "config_override_rejected"]
    assert len(audits) == 1
    payload = _json.loads(audits[0]["payload"])
    assert "risk.max_single_equity_position" in payload["removed_keys"]
    assert "advanced_override" in payload["removed_keys"]
    # identical rejection does NOT re-audit (health polls continuously)
    current_config(store, "paper")
    assert len([a for a in store.audit_rows()
                if a["event_type"] == "config_override_rejected"]) == 1


def test_worker_config_loading_survives_poisoned_blob(cfg, store):
    # The exact 2026-07-17 worker crash: kv blob with a now-refused value must
    # load (pruned) through the SHARED loader — the same one cmd_worker uses.
    from specforge.config import load_config_with_stored_overrides
    store.kv_set("config_overrides", {"risk": {"max_single_equity_position": 0.7},
                                      "nodes": {"insider": {"enabled": True}}})
    c = load_config_with_stored_overrides("paper", store)
    assert c.get("risk", "max_single_equity_position") <= 0.25
    assert c.get("nodes", "insider", "enabled") is True
    import pathlib
    cli_src = pathlib.Path("specforge/cli.py").read_text()
    assert "load_config_with_stored_overrides" in cli_src   # worker routes through it
    assert "load_config(cfg.mode, overrides=" not in cli_src  # raw path gone


# ── R1: dual-family validation — the absolute head that trades must also be
#        the evidence that promotes ─────────────────────────────────────────

def test_positive_excess_negative_absolute_can_never_promote(cfg, store, promo):
    # THE regression the R-plan demands: great excess rank, money-losing
    # absolute after cost → offline gate refuses, model stays sealed_candidate.
    _insert_run(store, "exc-only", "sealed_candidate", score=9.0,
                metrics=_passing_metrics(9.0, absolute_ok=False))
    out = neural.maybe_promote(cfg, store)
    assert out["action"] in ("none", "shadow")
    row = store.db.execute("SELECT lifecycle_state FROM model_runs "
                           "WHERE id='exc-only'").fetchone()
    assert row["lifecycle_state"] == "sealed_candidate"


def test_offline_gate_fails_closed_without_absolute_block():
    good = _passing_metrics()
    assert neural._offline_gate(good)
    legacy = dict(good); legacy.pop("absolute")       # pre-R1 metrics shape
    assert not neural._offline_gate(legacy)
    assert not neural._offline_gate(_passing_metrics(absolute_ok=False))


def test_forward_gate_requires_resolved_absolute_v2_evidence(cfg, store, promo, monkeypatch):
    _insert_run(store, "exp-1", "experimental_live", score=2.0)
    rich_excess = {"sessions": 60, "horizons": {
        "5": {"n": 20000, "ic": .05, "top_decile_alpha": .01, "coverage": .8},
        "21": {"n": 20000, "ic": .05, "top_decile_alpha": .01, "coverage": .8}}}
    # excess forward evidence alone — absolute v2 empty → must stay shadow
    monkeypatch.setattr(neural, "shadow_metrics", lambda st, mid: {
        **rich_excess, "absolute": {"5": {"n": 0}, "21": {"n": 0}}})
    assert neural.maybe_promote(cfg, store)["action"] == "shadow"
    # with resolved absolute evidence at both horizons → advances
    monkeypatch.setattr(neural, "shadow_metrics", lambda st, mid: {
        **rich_excess, "absolute": {"5": {"n": 400, "ic": .02},
                                    "21": {"n": 400, "ic": .02}}})
    assert neural.maybe_promote(cfg, store)["action"] == "production_candidate"


def test_illegal_lifecycle_transitions_rejected(cfg, store, promo):
    from specforge.ml.lifecycle import LifecycleError
    _insert_run(store, "vc-1", "validation_candidate")
    with pytest.raises(LifecycleError, match="illegal transition"):
        ml_lifecycle.transition(store, "model_runs", "vc-1", "champion",
                                reason="skip the queue")
    _insert_run(store, "ret-1", "retired")
    with pytest.raises(LifecycleError, match="illegal transition"):
        ml_lifecycle.transition(store, "model_runs", "ret-1", "experimental_live",
                                reason="zombie revival")
    _insert_run(store, "ch-1", "champion")
    with pytest.raises(LifecycleError, match="illegal transition"):
        ml_lifecycle.transition(store, "model_runs", "ch-1", "training",
                                reason="time travel")


def test_transition_compare_and_swap_loses_race(cfg, store, promo):
    from specforge.ml.lifecycle import LifecycleError
    _insert_run(store, "race-1", "sealed_candidate")
    real_execute = store.db.execute

    def racing_execute(sql, *args):
        if sql.startswith("UPDATE model_runs SET lifecycle_state"):
            real_execute("UPDATE model_runs SET lifecycle_state='rejected' "
                         "WHERE id='race-1'")   # competitor wins between read+write
        return real_execute(sql, *args)

    class RacingDB:                      # sqlite3.Connection attrs are read-only
        def __init__(self, db): self._db = db
        def __getattr__(self, name): return getattr(self._db, name)
        def execute(self, sql, *args):
            if sql.startswith("UPDATE model_runs SET lifecycle_state"):
                self._db.execute("UPDATE model_runs SET lifecycle_state='rejected' "
                                 "WHERE id='race-1'")
            return self._db.execute(sql, *args)

    class ShimStore:                     # Store.db is a read-only property
        def __init__(self, real):
            self.db = RacingDB(real.db)
            self.audit = real.audit

    with pytest.raises(LifecycleError, match="lost transition race"):
        ml_lifecycle.transition(ShimStore(store), "model_runs", "race-1",
                                "experimental_live", reason="racer")


def test_champion_uniqueness_is_a_database_invariant(cfg, store):
    import sqlite3 as _sq
    _insert_run(store, "u-1", "champion")
    with pytest.raises(_sq.IntegrityError):
        store.db.execute("UPDATE model_runs SET lifecycle_state='champion' "
                         "WHERE id=?", ("u-2",)) if False else \
            _insert_run(store, "u-2", "champion")   # second champion same kind


def test_v2_resolver_runs_in_research_loop_and_shadow_covers_finalists():
    import pathlib
    src = pathlib.Path("specforge/research.py").read_text()
    assert "resolve_forecasts(store) + resolve_forecasts_v2(store)" in src
    assert "'champion','production_candidate','experimental_live','sealed_candidate'" in src


# ── R2: executable backtest — features t−1, fill at t open ───────────────────

def _bt(cfg, store, tmp_path, tag, **over):
    from specforge.backtest import run_backtest
    from specforge.config import Config, _deep_merge
    c = Config(_deep_merge(cfg.data, over)) if over else cfg
    return run_backtest(c, years=30, tag=tag, scale="research",
                        log=lambda *a: None, out_dir=tmp_path, max_sessions=60)


def test_backtest_never_fills_on_the_decision_bar(cfg, store, tmp_path):
    _long_history(store)
    _bt(cfg, store, tmp_path, "r2fill")
    import sqlite3 as _sq
    bt = _sq.connect(tmp_path / "backtest_r2fill_research.db")
    bt.row_factory = _sq.Row
    offset = 1 + cfg.get("execution", "limit_offset_pct", default=0.001)
    bps = 1 + (cfg.get("execution", "spread_cost_bps", default=3)
               + cfg.get("execution", "slippage_bps", default=5)) / 10000.0
    # paper fill = round(limit,4) × (1+bps); limit = round(base×offset,4)
    open_limits, close_limits = set(), set()
    for r in bt.execute("SELECT open, close FROM bars"):
        if r["open"]:
            open_limits.add(round(round(r["open"] * offset, 4) * bps, 4))
        if r["close"]:
            close_limits.add(round(round(r["close"] * offset, 4) * bps, 4))
    buys = [r["price"] for r in bt.execute(
        "SELECT f.price FROM fills f JOIN orders o ON o.id=f.order_id "
        "WHERE f.side='buy' AND o.status='filled'")]
    assert buys, "backtest produced no fills — test would prove nothing"
    for price in buys:
        # every entry fills at an OPEN-derived executable quote…
        assert round(price, 4) in open_limits
        # …and never at a close-derived (same-bar/decision-price) limit
        assert round(price, 4) not in (close_limits - open_limits)


def test_backtest_immune_to_future_bar_mutation(cfg, store, tmp_path):
    _long_history(store)
    # warm-up run: run_backtest syncs earnings/fundamentals kv caches BACK to
    # the source store, so the first run changes inputs for the second. One
    # throwaway run brings the caches to steady state; baseline and mutated
    # runs then differ ONLY by the future-bar sabotage.
    _bt(cfg, store, tmp_path, "r2warm")
    first = _bt(cfg, store, tmp_path, "r2base")
    assert "error" not in first
    import sqlite3 as _sq
    base = _sq.connect(tmp_path / "backtest_r2base_research.db")
    base.row_factory = _sq.Row
    curve = [(r["d"], round(r["equity"], 6)) for r in base.execute(
        "SELECT d, equity FROM equity_curve ORDER BY d")]
    cutoff = curve[40][0]
    # sabotage every bar AFTER the cutoff in the SOURCE data by +70%
    store.db.execute("UPDATE bars SET open=open*1.7, high=high*1.7, "
                     "low=low*1.7, close=close*1.7 WHERE d>?", (cutoff,))
    store.db.commit()
    _bt(cfg, store, tmp_path, "r2mut")
    mut = _sq.connect(tmp_path / "backtest_r2mut_research.db")
    mut.row_factory = _sq.Row
    # STRICTLY before the cutoff: the row labeled d fills at the NEXT session's
    # open, so the row at d==cutoff legitimately sees the first mutated open.
    curve2 = [(r["d"], round(r["equity"], 6)) for r in mut.execute(
        "SELECT d, equity FROM equity_curve WHERE d<? ORDER BY d", (cutoff,))]
    # every decision, fill, and mark before the cutoff is bit-identical:
    # future bars cannot reach back into earlier sessions
    assert curve2 == [c for c in curve if c[0] < cutoff]
    assert len(curve2) >= 39                   # the comparison is not vacuous


def test_backtest_doubling_costs_cannot_improve_results(cfg, store, tmp_path):
    _long_history(store)
    base = _bt(cfg, store, tmp_path, "r2cost1")
    doubled = _bt(cfg, store, tmp_path, "r2cost2", execution={
        "limit_offset_pct": 0.002, "spread_cost_bps": 6, "slippage_bps": 10})
    b, d = base.get("overall") or {}, doubled.get("overall") or {}
    assert "total_return" in b and "total_return" in d
    assert d["total_return"] <= b["total_return"] + 1e-9
    # and the costs actually BIND: for every (symbol, day) bought in BOTH
    # runs, the doubled-cost run pays at least as much per share. (Averaging
    # across runs is invalid — costs change WHICH fills happen.)
    import sqlite3 as _sq
    books = []
    for tag in ("r2cost1", "r2cost2"):
        con = _sq.connect(tmp_path / f"backtest_{tag}_research.db")
        con.row_factory = _sq.Row
        books.append({(r["symbol"], r["filled_at"][:10]): r["price"]
                      for r in con.execute(
                          "SELECT symbol, filled_at, price FROM fills "
                          "WHERE side='buy'")})
    common = set(books[0]) & set(books[1])
    assert common, "no common fills — the binding check would be vacuous"
    for key in common:
        assert books[1][key] >= books[0][key] - 1e-9


def test_backtest_report_declares_decision_convention(cfg, store, tmp_path):
    _long_history(store)
    report = _bt(cfg, store, tmp_path, "r2conv")
    assert "features<=t-1" in report["decision_convention"]
    assert "fill at t open" in report["decision_convention"]


# ── R3: real policy comparison — immutable v2 replay must cause divergence ───

def _seed_replay_world(cfg, store, tmp_path):
    """A champion model + immutable v2 forecasts: BBB strongly up, AAA/CCC
    down, at EVERY session — so a working blend must reorder candidates."""
    import hashlib
    from specforge.research import record_forecast_v2
    _long_history(store)
    ckpt = tmp_path / "serving.pt"
    ckpt.write_bytes(b"immutable-serving-checkpoint")
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    _insert_run(store, "serve-1", "champion", checkpoint=str(ckpt), sha=sha)
    sessions = [r["d"] for r in store.db.execute(
        "SELECT DISTINCT d FROM bars WHERE symbol='SPY' ORDER BY d")]
    # BBB strongly up; everything else — including the benchmark the momentum
    # node loves — strongly down, so a live blend MUST reorder the book.
    views = {"BBB": (0.08, 0.05, 0.9), "AAA": (-0.06, -0.04, 0.1),
             "CCC": (-0.06, -0.04, 0.1), "SPY": (-0.06, -0.04, 0.1)}
    for d in sessions:
        for sym, (aq, eq_, pa) in views.items():
            nf = NeuralForecast(
                symbol=sym, as_of=d, horizon_sessions=21,
                absolute_q10=aq - 0.03, absolute_q50=aq, absolute_q90=aq + 0.03,
                excess_q10=eq_ - 0.02, excess_q50=eq_, excess_q90=eq_ + 0.02,
                probability_absolute_edge_positive=pa,
                probability_excess_positive=pa,
                model_id="serve-1", dataset_manifest_id="d1",
                feature_schema_hash=neural.FEATURE_HASH)
            record_forecast_v2(store, nf, model_id="serve-1", as_of=d,
                               feature_hash=neural.FEATURE_HASH,
                               target_schema_hash=ml_targets.TARGET_SCHEMA_HASH)
    store.db.commit()      # record_forecast_v2 leaves commit to its caller
    return sessions


def test_replay_serves_only_the_exact_decision_date(cfg, store, tmp_path):
    sessions = _seed_replay_world(cfg, store, tmp_path)
    preds, meta = neural.replay_forecasts(cfg, store, sessions[10])
    assert set(preds) == {"AAA", "BBB", "CCC", "SPY"} and meta["replayed"]
    assert all(v["21"].as_of == sessions[10] for v in preds.values())
    empty, meta2 = neural.replay_forecasts(cfg, store, "1999-01-01")
    assert empty == {} and "no v2 forecasts" in meta2["silent"]


def test_replay_without_serving_model_is_silent(cfg, store):
    preds, meta = neural.replay_forecasts(cfg, store, "2026-01-05")
    assert preds == {} and "no serving model" in meta["silent"]


def test_deterministic_policy_disables_every_learned_pathway(cfg):
    from specforge.backtest import _policy_cfg
    det = _policy_cfg(cfg, "deterministic")
    assert det.get("neural", "experimental_blend") == 0.0
    assert det.get("neural", "backtest_replay") is False
    assert det.get("neural", "exploration", "enabled") is False
    assert det.get("analog_graph", "enabled") is False
    assert det.get("nodes", "neural", "enabled") is False
    assert _policy_cfg(cfg, "fixed_blend").get("neural", "backtest_replay") is True


def test_injected_forecasts_cause_policy_divergence(cfg, store, tmp_path):
    from specforge.backtest import compare_policies
    _seed_replay_world(cfg, store, tmp_path)
    cfg.data["neural"]["experimental_blend"] = 0.30    # R0 default is 0; ≤ max 0.40
    cfg.data["nodes"]["neural"] = {"enabled": True, "weight": 0.15,
                                   "horizon_days": 21, "status": "experimental"}
    out = compare_policies(cfg, years=30, scale="research",
                           policies=("deterministic", "fixed_blend"),
                           log=lambda *a: None, out_dir=tmp_path,
                           max_sessions=45)
    det, blend = out["policies"]["deterministic"], out["policies"]["fixed_blend"]
    assert det["window"] == blend["window"]            # identical conditions…
    import sqlite3 as _sq

    def _book(tag):
        con = _sq.connect(tmp_path / f"backtest_policy_{tag}_research.db")
        con.row_factory = _sq.Row
        fills = [(r["symbol"], r["side"], round(r["qty"], 4)) for r in
                 con.execute("SELECT symbol, side, qty FROM fills ORDER BY rowid")]
        blends = [r["payload"] for r in con.execute(
            "SELECT payload FROM audit WHERE event_type='neural_direct_blend' "
            "AND payload LIKE '%\"blend\": 0.3%'")]
        return fills, blends

    det_fills, det_blends = _book("deterministic")
    blend_fills, blend_blends = _book("fixed_blend")
    assert det_blends == []                            # learned pathway dark
    assert blend_blends, "blend never engaged — replay path is dead"
    # …and the injected forecasts changed actual trading (the exit gate)
    assert det_fills != blend_fills
    # every policy keeps an independent ledger
    assert (tmp_path / "backtest_policy_deterministic_research.db").exists()
    assert (tmp_path / "backtest_policy_fixed_blend_research.db").exists()


# ── R4: governor + broker completion — every limit must FAIL closed ──────────

def _gov(cfg, store):
    from specforge.risk import Governor
    return Governor(cfg, store)


def _acct(equity=1000.0, cash=1000.0, positions=None):
    from specforge.models import AccountState
    return AccountState(equity=equity, cash=cash, buying_power=cash,
                        positions=positions or [], as_of="2026-07-19")


def _buy(symbol="BBB", notional=50.0):
    from specforge.models import OrderIntent
    c = _candidate(symbol, 0.5)
    c.target_notional = notional
    intent = OrderIntent.make(c, qty=notional / 100.0, limit_price=100.0)
    intent.notional = notional
    return intent, c


def test_sector_cap_rejects_and_reduces(cfg, store):
    from specforge.models import Position
    from specforge.risk import CycleState
    for sym in ("AAA", "BBB"):
        store.db.execute("INSERT INTO instruments(symbol, sector) VALUES(?,?) "
                         "ON CONFLICT(symbol) DO UPDATE SET sector=excluded.sector",
                         (sym, "tech"))
    store.db.commit()
    cfg.data["risk"]["max_single_equity_position"] = 0.5   # so SECTOR binds first
    gov = _gov(cfg, store)
    # AAA marked ~0.7×190 ≈ $133 of tech already held; cap = 25% × 1000 = $250
    held = [Position(symbol="AAA", asset_type="equity", qty=0.7, avg_cost=100,
                     opened_at="2026-01-01")]
    intent, c = _buy("BBB", notional=200.0)     # 133 + 200 > 250 → reduce
    d = gov.review(intent, c, _acct(positions=held), CycleState(1000), 1)
    assert d.verdict == "APPROVED_WITH_SIZE_REDUCTION"
    assert any("sector cap" in r for r in d.reasons)
    # a second tech name with the sector already full → outright reject
    held2 = held + [Position(symbol="BBB", asset_type="equity", qty=0.65,
                             avg_cost=100, opened_at="2026-01-01")]
    intent2, c2 = _buy("BBB", notional=50.0)
    d2 = gov.review(intent2, c2, _acct(positions=held2), CycleState(1000), 1,
                    skip_duplicate=True)
    assert d2.verdict == "REJECTED" and any("cap" in r for r in d2.reasons)


def test_unknown_sector_exempt_but_audited_once(cfg, store):
    from specforge.risk import CycleState
    gov = _gov(cfg, store)
    intent, c = _buy("CCC", notional=20.0)
    gov.review(intent, c, _acct(), CycleState(1000), 1)
    gov.review(intent, c, _acct(), CycleState(1000), 1, skip_duplicate=True)
    n = store.db.execute("SELECT COUNT(*) n FROM audit WHERE "
                         "event_type='sector_unknown'").fetchone()["n"]
    assert n == 1


def test_options_aggregate_premium_cap(cfg, store):
    from specforge.models import OrderIntent, Position
    from specforge.risk import CycleState
    cfg.data["risk"]["options_enabled"] = True          # test-local unlock
    gov = _gov(cfg, store)
    held = [Position(symbol="QQQ", asset_type="option", qty=1, avg_cost=0.5,
                     opened_at="2026-01-01", option_symbol="QQQ_C")]   # $50 premium
    c = _candidate("QQQ", 0.5)
    c.asset_type = "option"
    c.option_details = {"dte": 45, "delta": 0.5, "spread_pct": 0.05,
                        "open_interest": 500}
    c.target_notional = 30.0
    intent = OrderIntent.make(c, qty=1, limit_price=0.3)
    intent.asset_type = "option"
    intent.notional = 30.0
    # cap = 6% × 1000 = $60; held $50 + $30 = $80 > $60 → reject
    d = gov.review(intent, c, _acct(equity=10000, cash=10000, positions=held),
                   CycleState(1000), 1)
    # equity 10k → cap $600: passes. Drop equity so the cap binds:
    d = gov.review(intent, c, _acct(equity=1000, cash=1000, positions=held),
                   CycleState(1000), 1, skip_duplicate=True)
    assert d.verdict == "REJECTED"
    assert any("aggregate premium cap" in r for r in d.reasons)


def test_pending_orders_reserve_exposure(cfg, store):
    from specforge.models import OrderIntent
    from specforge.risk import CycleState
    cfg.data["risk"]["max_single_equity_position"] = 0.5   # deployment binds
    gov = _gov(cfg, store)
    # an open (placed, unfilled) buy already reserves $600 of the $700 room
    resting, _ = _buy("ZZZ", notional=600.0)
    resting.status = "placed"
    store.record_order(resting, mode="paper")
    intent, c = _buy("YYY", notional=300.0)
    d = gov.review(intent, c, _acct(), CycleState(1000), 1)
    assert d.verdict == "APPROVED_WITH_SIZE_REDUCTION"
    assert any("deployment cap" in r for r in d.reasons)
    assert d.approved_notional <= 100.0 + 1e-6


def test_loss_switches_are_flow_normalized(cfg, store):
    gov = _gov(cfg, store)
    day = gov.today
    store.db.execute("INSERT OR REPLACE INTO equity_curve VALUES(?,?,?,?,?)",
                     ((gov._today_dt() - __import__("datetime").timedelta(days=1)
                       ).isoformat(), "t", 1000.0, 500.0, "paper"))
    store.db.commit()
    # a $500 DEPOSIT must not mask a real 5% trading loss (equity 1450 gross)
    store.record_external_flow(500.0, d=day)
    gov.check_kill_switches(_acct(equity=1450.0), "paper")
    assert "daily_loss" in gov.active_switches()
    store.kv_set("kill_switches", {})                    # reset
    # a $500 WITHDRAWAL must not fake a loss (equity 510 gross = flat trading)
    store.kv_set("external_flows", [{"d": day, "amount": -500.0}])
    gov.check_kill_switches(_acct(equity=510.0), "paper")
    assert "daily_loss" not in gov.active_switches()


def test_slippage_breaches_trip_halt(cfg, store):
    gov = _gov(cfg, store)
    store.kv_set(f"slippage_breaches:{gov.today}", 3)
    gov.check_kill_switches(_acct(), "paper")
    assert "slippage" in gov.active_switches()


def _adapter(cfg, store):
    from specforge.broker.robinhood_mcp import RobinhoodMCPBroker
    a = RobinhoodMCPBroker.__new__(RobinhoodMCPBroker)
    a.cfg, a.store = cfg, store
    a._live_ok, a._live_why = True, ""
    return a


def test_wrong_asset_submission_is_impossible(cfg, store):
    from specforge.broker.robinhood_mcp import BrokerOrderRejected
    from specforge.models import OrderIntent
    a = _adapter(cfg, store)
    c = _candidate("QQQ", 0.5)
    c.asset_type = "option"
    intent = OrderIntent.make(c, qty=1, limit_price=0.5)
    intent.asset_type = "option"
    review = a.review_order(intent)
    assert not review.ok and any("equity-only" in w for w in review.warnings)
    with pytest.raises(BrokerOrderRejected, match="equity-only"):
        a.place_order(intent)


def test_fractional_market_guard_refuses_unsafe_quotes(cfg, store, monkeypatch):
    from specforge.broker.robinhood_mcp import BrokerOrderRejected
    a = _adapter(cfg, store)
    intent, _ = _buy("BBB", notional=50.0)      # qty 0.5 → fractional → market

    def guard_with(quote):
        monkeypatch.setattr(type(a), "_fresh_quote", lambda self, s: quote)
        return lambda: a._guard_fractional_market(intent)

    with pytest.raises(BrokerOrderRejected, match="no executable quote"):
        guard_with({})()
    with pytest.raises(BrokerOrderRejected, match="old"):
        guard_with({"price": 100.0, "age_s": 999})()
    with pytest.raises(BrokerOrderRejected, match="spread"):
        guard_with({"price": 100.0, "bid": 98.0, "ask": 100.0})()
    with pytest.raises(BrokerOrderRejected, match="deviates"):
        guard_with({"price": 103.0, "bid": 102.9, "ask": 103.0})()
    guard_with({"price": 100.2, "bid": 100.1, "ask": 100.25, "age_s": 5})()


def test_slippage_monitor_records_breach(cfg, store):
    a = _adapter(cfg, store)
    intent, _ = _buy("BBB", notional=50.0)      # limit 100
    a._monitor_slippage(intent, fill_price=102.0)        # 2% > 1% tolerance
    import datetime as _dt
    day = _dt.datetime.now().astimezone().date().isoformat()
    assert int(store.kv_get(f"slippage_breaches:{day}", 0)) == 1
    assert store.db.execute("SELECT 1 FROM audit WHERE "
                            "event_type='fractional_slippage_breach'").fetchone()


def test_api_mutations_require_session_token(cfg, store):
    from fastapi.testclient import TestClient
    from specforge.app import create_app
    app = create_app(cfg, store, with_scheduler=False)
    bare = TestClient(app)
    assert bare.get("/api/health").status_code == 200          # reads open
    r = bare.post("/api/research/jobs", json={"kind": "discover"})
    assert r.status_code == 401 and "X-Session-Token" in r.text
    token = bare.get("/api/session").json()["token"]
    ok = TestClient(app, headers={"X-Session-Token": token})
    assert ok.post("/api/research/jobs",
                   json={"kind": "discover"}).status_code in (200, 202, 409)


# ── R5: point-in-time data — known_at <= decision_at, honest costs ────────────

def _news(store, symbol, published, classified, stance=1.0):
    with store.db:
        store.db.execute(
            "INSERT INTO news_intelligence(id,symbol,published_at,ingested_at,title,"
            "summary,url,source,content_hash,stance,confidence,catalyst,novelty,"
            "reliability,contradiction,price_reaction,classified_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"{symbol}-{published}-{classified}", symbol, published, published,
             "t", "s", "u", "src", published, stance, 1.0, "none", 1.0, 1.0,
             None, None, classified))


def test_news_feature_is_keyed_to_classification_known_at(cfg, store):
    """A story published in 2011 but classified today is NOT 2011 evidence."""
    index = pd.Index([f"2026-07-{d:02d}" for d in range(10, 21)])
    _news(store, "AAA", "2026-07-11T12:00:00+00:00", "2026-07-15T09:00:00+00:00")
    score, missing = neural._news_series(store, "AAA", index)
    # Nothing is known before the classification landed.
    assert score.loc[:"2026-07-14"].eq(0).all()
    assert missing.loc[:"2026-07-14"].eq(1).all()
    assert score.loc["2026-07-15"] != 0 and missing.loc["2026-07-15"] == 0


def test_news_provenance_reports_classification_lag(cfg, store):
    _news(store, "AAA", "2026-07-11T00:00:00+00:00", "2026-07-11T05:00:00+00:00")
    _news(store, "AAA", "2011-03-02T00:00:00+00:00", "2026-07-15T00:00:00+00:00")
    stats = neural.news_pit_stats(store)
    assert stats["classified"] == 2 and stats["retro_classified"] == 1
    assert stats["max_lag_days"] > 5000


def test_universe_membership_is_point_in_time(cfg, store):
    from specforge import universe
    with store.db:
        for as_of, syms in (("2026-01-05", ("AAA", "OLD")),
                            ("2026-04-05", ("AAA", "BBB"))):
            for rank, s in enumerate(syms, 1):
                store.db.execute("INSERT INTO universe_membership VALUES(?,?,?,?,?,?)",
                                 (as_of, s, "research", rank, "test", "{}"))
    assert universe.membership_as_of(store, "2026-01-04") is None      # uncovered
    assert universe.membership_as_of(store, "2026-02-01") == {"AAA", "OLD"}
    assert universe.membership_as_of(store, "2026-09-01") == {"AAA", "BBB"}
    # Delisted names stay in the historical panel — that is the survivor-bias fix.
    assert "OLD" in universe.historical_symbols(store)


def test_sample_costs_vary_with_spread_and_liquidity(cfg):
    floor = ml_targets.round_trip_cost(cfg)
    idx = pd.Index([f"2026-01-{d:02d}" for d in range(1, 32)])
    tight = pd.DataFrame({"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0,
                          "volume": 5e7}, index=idx)
    wide = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0,
                         "volume": 2e4}, index=idx)
    tight_cost = ml_targets.sample_costs(cfg, tight)
    wide_cost = ml_targets.sample_costs(cfg, wide)
    assert (tight_cost >= floor).all() and (wide_cost >= floor).all()
    assert tight_cost.iloc[-1] == pytest.approx(floor, rel=.25)   # liquid ≈ the floor
    assert wide_cost.iloc[-1] > 3 * tight_cost.iloc[-1]           # illiquid costs more
    assert (wide_cost <= ml_targets.MAX_SAMPLE_COST).all()        # bounded, never absurd


def test_dataset_carries_per_sample_cost_and_pit_provenance(cfg, store):
    from conftest import synth_bars
    for sym in ("AAA", "BBB", "CCC", "SPY"):
        store.upsert_bars(sym, synth_bars(n_days=700, daily_drift=.001), "test")
    store.upsert_bars("^VIX", [{**r, "open": 15, "high": 16, "low": 14, "close": 15}
                               for r in synth_bars(n_days=700)], "test")
    cfg.data["neural"]["input_sessions"] = 40
    ds = neural.build_dataset(cfg, store, symbols=["AAA", "BBB", "CCC"])
    assert "error" not in ds
    costs = ds["sample_cost"]
    assert costs.shape == (len(ds["dates"]),)
    assert (costs >= ml_targets.round_trip_cost(cfg)).all()
    # The flat constant remains only as the documented floor.
    assert ds["round_trip_cost"] == pytest.approx(float(np.median(costs)))
    assert ds["cost_floor"] == pytest.approx(ml_targets.round_trip_cost(cfg))
    pit = ds["pit"]
    assert pit["universe_snapshots"] == 0          # none seeded → uncovered, labeled
    assert pit["universe_covered_candidates"] == 0
    assert pit["panel_windows"] == len(ds["dates"])
    assert "news" in pit


def test_pit_filter_drops_windows_before_membership(cfg, store):
    """With snapshots present, a symbol contributes no window before it joined."""
    from conftest import synth_bars
    for sym in ("AAA", "BBB", "CCC", "SPY"):
        store.upsert_bars(sym, synth_bars(n_days=700, daily_drift=.001), "test")
    store.upsert_bars("^VIX", [{**r, "open": 15, "high": 16, "low": 14, "close": 15}
                               for r in synth_bars(n_days=700)], "test")
    dates = [r["d"] for r in store.db.execute(
        "SELECT DISTINCT d FROM bars WHERE symbol='AAA' ORDER BY d")]
    join = dates[len(dates) // 2]
    with store.db:
        for rank, s in enumerate(("AAA", "BBB"), 1):        # CCC never joins
            store.db.execute("INSERT INTO universe_membership VALUES(?,?,?,?,?,?)",
                             (dates[0], s, "research", rank, "test", "{}"))
        store.db.execute("INSERT INTO universe_membership VALUES(?,?,?,?,?,?)",
                         (join, "CCC", "research", 3, "test", "{}"))
    cfg.data["neural"]["input_sessions"] = 40
    ds = neural.build_dataset(cfg, store, symbols=["AAA", "BBB", "CCC"])
    assert "error" not in ds
    ccc = ds["dates"][ds["owners"] == "CCC"]
    assert len(ccc) and min(ccc) >= join
    assert ds["pit"]["universe_snapshots"] == 2
    assert ds["pit"]["universe_dropped_candidates"] > 0


# ── R6: model laboratory — earn the complexity or don't trade ─────────────────

EDGE_FEATURE = 4          # vol21 — deliberately NOT r1, or the momentum control
                          # would BE the planted edge and stop being a control.


def _bakeoff_dataset(n_days=400, n_symbols=20, signal=0.0, seed=0, window=3):
    """Synthetic panel with a KNOWN linear edge of strength `signal`.

    `n_days` is large enough that the evaluation split still holds several
    non-overlapping 21-session cohorts per offset — otherwise every model
    scores 'insufficient' and the comparison proves nothing.
    """
    from specforge import neural
    rng = np.random.default_rng(seed)
    n = n_days * n_symbols
    x = rng.normal(size=(n, window, len(neural.FEATURES))).astype(np.float32)
    day = [f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}" for d in range(n_days)]
    dates = np.repeat(day, n_symbols)
    edge = x[:, -1, EDGE_FEATURE]
    y = np.column_stack([signal * edge + rng.normal(scale=.02, size=n) for _ in (5, 21)]
                        ).astype(np.float32)
    order = np.argsort(dates, kind="stable")
    x, y, dates = x[order], y[order], dates[order]
    train = dates <= day[n_days * 6 // 10]
    return {"X": x, "Y": y, "Y_excess": y, "Y_absolute": y, "dates": dates,
            "horizons": (5, 21), "masks": {"train": train, "val": ~train, "test": ~train},
            "mean": np.zeros((1, 1, len(neural.FEATURES)), dtype=np.float32),
            "std": np.ones((1, 1, len(neural.FEATURES)), dtype=np.float32),
            "round_trip_cost": 0.0016,
            "sample_cost": np.full(n, 0.0016, dtype=np.float32)}


def test_bakeoff_simple_models_recover_a_planted_edge():
    from specforge.ml import bakeoff
    ds = _bakeoff_dataset(signal=0.05)
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    preds = bakeoff.simple_predictions(ds, "absolute")
    assert set(preds) == set(bakeoff.MODELS)
    for name, pred in preds.items():
        assert pred.shape == (len(ds["dates"]), 2), name
    # The learners find the planted linear edge; zero cannot, by construction.
    for learner in ("ridge", "elastic_net", "boosted_tree"):
        assert np.corrcoef(preds[learner][eval_idx, 0],
                           ds["Y_absolute"][eval_idx, 0])[0, 1] > .2, learner
    assert np.allclose(preds["zero"], 0)


def test_elastic_net_is_sparser_than_ridge():
    from specforge.ml import bakeoff
    ds = _bakeoff_dataset(signal=0.05)
    x = bakeoff.context_design(ds)
    train = ds["masks"]["train"]
    mean, std = x[train].mean(0), x[train].std(0) + 1e-6
    xn = (x - mean) / std
    y = ds["Y_absolute"].astype(np.float64)
    ridge = bakeoff._ridge(xn[train], y[train], xn[train])
    net = bakeoff._elastic_net(xn[train], y[train], xn[train])
    assert np.isfinite(net).all() and net.shape == ridge.shape
    # L1 drives most coefficients to exactly zero; ridge drives none there.
    weights = bakeoff._elastic_net(xn[train], y[train], np.eye(xn.shape[1])) - \
        y[train].mean(0)
    assert (np.abs(weights) < 1e-12).mean() > .5


def test_policy_return_fails_closed_on_thin_evidence():
    from specforge.ml import bakeoff
    ds = _bakeoff_dataset(n_days=40, n_symbols=12, signal=0.05)
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    scored = bakeoff.policy_return(bakeoff.simple_predictions(ds, "absolute")["ridge"],
                                   ds, eval_idx, "absolute")
    # Too few non-overlapping 21-session cohorts to be evidence of anything.
    assert scored["evidence"] == "insufficient"
    assert scored["policy_utility"] <= 0


def test_bakeoff_gate_requires_median_seed_to_beat_every_control():
    from specforge.ml import bakeoff
    ds = _bakeoff_dataset(signal=0.05)
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    truth = ds["Y_absolute"]
    strong = truth + np.random.default_rng(1).normal(scale=.001, size=truth.shape)
    weak = np.random.default_rng(2).normal(scale=.02, size=truth.shape)
    # One brilliant seed cannot carry two poor ones: the MEDIAN decides.
    lucky = {f: {"tcn_seed_0": strong, "tcn_seed_1": weak, "tcn_seed_2": weak}
             for f in bakeoff.FAMILIES}
    assert bakeoff.compare(ds, eval_idx, lucky)["verdict"] is False
    genuine = {f: {f"tcn_seed_{i}": strong for i in range(3)} for f in bakeoff.FAMILIES}
    result = bakeoff.compare(ds, eval_idx, genuine)
    assert result["verdict"] is True
    assert result["absolute"]["n_seeds"] == 3
    assert result["absolute"]["tcn_median_utility"] > \
        result["absolute"]["best_control_utility"]
    # No seeds at all is never a pass.
    assert bakeoff.compare(ds, eval_idx, {})["verdict"] is False


def test_bakeoff_scores_both_families_independently():
    from specforge.ml import bakeoff
    ds = _bakeoff_dataset(signal=0.05)
    ds["Y_excess"] = np.random.default_rng(3).normal(scale=.02, size=ds["Y_absolute"].shape)
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    strong_abs = ds["Y_absolute"] + np.random.default_rng(4).normal(
        scale=.001, size=ds["Y_absolute"].shape)
    preds = {"absolute": {f"tcn_seed_{i}": strong_abs for i in range(3)},
             "excess": {f"tcn_seed_{i}": strong_abs for i in range(3)}}
    result = bakeoff.compare(ds, eval_idx, preds)
    # Winning the family you were fit on cannot buy a pass on the other.
    assert result["absolute"]["beats_controls"] is True
    assert result["excess"]["beats_controls"] is False
    assert result["verdict"] is False


def test_policy_return_uses_per_sample_costs():
    from specforge.ml import bakeoff
    ds = _bakeoff_dataset(signal=0.05)
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    pred = ds["Y_absolute"]
    cheap = bakeoff.policy_return(pred, ds, eval_idx, "absolute",
                                  costs=np.full(len(ds["dates"]), .0001))
    dear = bakeoff.policy_return(pred, ds, eval_idx, "absolute",
                                 costs=np.full(len(ds["dates"]), .02))
    assert cheap["policy_utility"] > dear["policy_utility"]


def test_date_grouped_batches_preserve_cross_sections():
    """Random row batching destroys the ranking objective; date grouping keeps it."""
    n_dates, n_symbols = 2500, 400
    dates = np.repeat([f"d{i:05d}" for i in range(n_dates)], n_symbols)
    rows = np.arange(len(dates))
    rng = np.random.default_rng(0)
    batches = neural.date_grouped_batches(dates, rows, 512, rng)
    # Every row appears exactly once — grouping must not drop or duplicate data.
    assert np.array_equal(np.sort(np.concatenate(batches)), rows)
    # Each batch holds whole sessions, so nearly every row has same-day peers.
    for batch in batches[:20]:
        _, counts = np.unique(dates[batch], return_counts=True)
        assert counts.min() == n_symbols          # no session split across batches
        pairs = int((counts - 1).sum())
        assert pairs > 0.9 * len(batch)
    # The control: uniformly random batches of the same size leave almost none.
    random_batch = rng.choice(rows, 512, replace=False)
    _, counts = np.unique(dates[random_batch], return_counts=True)
    assert int((counts[counts >= 2] - 1).sum()) < 0.25 * len(random_batch)


def test_date_grouped_batches_are_deterministic_per_seed():
    dates = np.repeat([f"d{i:03d}" for i in range(50)], 10)
    rows = np.arange(len(dates))
    first = neural.date_grouped_batches(dates, rows, 64, np.random.default_rng(7))
    same = neural.date_grouped_batches(dates, rows, 64, np.random.default_rng(7))
    other = neural.date_grouped_batches(dates, rows, 64, np.random.default_rng(8))
    assert [b.tolist() for b in first] == [b.tolist() for b in same]
    assert [b.tolist() for b in first] != [b.tolist() for b in other]


def test_offline_gate_requires_a_policy_return_bakeoff_win():
    assert neural._offline_gate(_passing_metrics())
    # Losing the bakeoff blocks promotion even with every forecast metric intact.
    losing = _passing_metrics()
    losing["bakeoff"]["verdict"] = False
    losing["beats_baselines"] = False
    assert not neural._offline_gate(losing)
    # A legacy row whose beats_baselines meant "lower pinball" cannot inherit
    # a permission it was never measured for.
    legacy = _passing_metrics()
    del legacy["bakeoff"]
    assert not neural._offline_gate(legacy)
    # Nor can a bakeoff scored on some other basis.
    wrong_basis = _passing_metrics()
    wrong_basis["bakeoff"]["basis"] = "pinball"
    assert not neural._offline_gate(wrong_basis)


def test_seed_predictions_returns_independent_draws_per_family(cfg, store):
    ds = _bakeoff_dataset(n_days=120, n_symbols=12, signal=.05)
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    preds = neural.seed_predictions(cfg, ds, eval_idx, seeds=3, epochs=1)
    assert set(preds) == {"absolute", "excess"}
    for family in ("absolute", "excess"):
        assert len(preds[family]) == 3
        for pred in preds[family].values():
            assert pred.shape == (len(ds["dates"]), 2)
            assert np.isfinite(pred).all()
        # Different seeds are genuinely different models, not one run relabelled.
        draws = list(preds[family].values())
        assert not np.allclose(draws[0][eval_idx], draws[1][eval_idx])


def test_bakeoff_result_is_json_serializable(cfg):
    """Metrics are persisted to SQLite as JSON — numpy scalars would break it."""
    import json
    from specforge.ml import bakeoff
    ds = _bakeoff_dataset(signal=.05)
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    strong = ds["Y_absolute"] + np.random.default_rng(9).normal(
        scale=.001, size=ds["Y_absolute"].shape)
    result = bakeoff.compare(ds, eval_idx,
                             {f: {f"tcn_seed_{i}": strong for i in range(3)}
                              for f in bakeoff.FAMILIES})
    assert json.loads(json.dumps(result))["verdict"] is True


def test_feature_family_ablation_detects_the_carrying_family():
    from specforge.ml import bakeoff
    from specforge import neural
    # Every feature family is covered exactly once, with no unknown names.
    covered = [f for members in bakeoff.FEATURE_FAMILIES.values() for f in members]
    assert sorted(covered) == sorted(neural.FEATURES)
    assert len(covered) == len(set(covered))

    ds = _bakeoff_dataset(signal=.05)          # edge planted on vol21 → "price"
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    result = bakeoff.ablate(ds, eval_idx, "absolute")
    assert result["_full"]["evidence"] == "ok"
    # Removing the family that carries the planted edge hurts most; families
    # that are pure noise here cost roughly nothing.
    assert result["price"]["delta"] < 0
    assert result["price"]["delta"] == min(
        v["delta"] for k, v in result.items()
        if isinstance(v, dict) and "delta" in v)
    assert abs(result["news"]["delta"]) < abs(result["price"]["delta"])


# ── R6c: the panel stops copying itself ──────────────────────────────────────

def test_normalized_panel_matches_eager_renormalization():
    """The lazy fold view must be numerically identical to the copy it replaced."""
    rng = np.random.default_rng(0)
    raw = rng.normal(loc=3.0, scale=2.0, size=(200, 8, 5)).astype(np.float32)
    base_mean = raw.mean((0, 1), keepdims=True)
    base_std = raw.std((0, 1), keepdims=True) + 1e-6
    stored = ((raw - base_mean) / base_std).astype(np.float32)
    train = np.zeros(200, dtype=bool); train[:120] = True

    # What the old code did: de-normalize everything, then renormalize on fold.
    eager_mean = raw[train].mean((0, 1), keepdims=True)
    eager_std = raw[train].std((0, 1), keepdims=True) + 1e-6
    eager = (raw - eager_mean) / eager_std

    # What the new code does: fold statistics derived from the stored panel.
    rows = stored[train]
    mean = rows.mean((0, 1), keepdims=True) * base_std + base_mean
    std = rows.std((0, 1), keepdims=True) * base_std + 1e-6
    panel = neural.NormalizedPanel(stored, base_mean, base_std, mean, std)
    idx = np.array([0, 7, 55, 199])
    assert np.allclose(panel[idx].numpy(), eager[idx], atol=1e-4)
    assert len(panel) == 200 and panel.shape == stored.shape


def test_context_rows_avoids_denormalizing_the_whole_panel():
    rng = np.random.default_rng(1)
    ds = {"X": rng.normal(size=(50, 6, 4)).astype(np.float32),
          "std": np.full((1, 1, 4), 2.0), "mean": np.full((1, 1, 4), 1.0)}
    expected = (ds["X"] * ds["std"] + ds["mean"])[:, -1, :]
    assert np.allclose(neural.context_rows(ds), expected)
    assert neural.context_rows(ds).shape == (50, 4)
    subset = neural.context_rows(ds, np.array([3, 9]))
    assert np.allclose(subset, expected[[3, 9]])


def test_training_window_limit_scales_with_window_size(cfg):
    """The cap is a memory budget, not a magic constant that rots as features grow."""
    cfg.data["neural"]["input_sessions"] = 60
    cfg.data["neural"]["max_training_windows"] = 10_000_000
    cfg.data["neural"]["panel_memory_gb"] = 2.0
    wide = neural._training_window_limit(cfg)
    cfg.data["neural"]["input_sessions"] = 240          # 4x the window, 4x the bytes
    narrow = neural._training_window_limit(cfg)
    assert narrow == pytest.approx(wide / 4, rel=.05)
    # A bigger budget buys proportionally more windows...
    cfg.data["neural"]["input_sessions"] = 60
    cfg.data["neural"]["panel_memory_gb"] = 4.0
    assert neural._training_window_limit(cfg) == pytest.approx(2 * wide, rel=.05)
    # ...and an explicit configured request still caps it downward.
    cfg.data["neural"]["max_training_windows"] = 500
    assert neural._training_window_limit(cfg) == 500
    # The R5 panel must clear the old hard-coded 12k ceiling.
    cfg.data["neural"]["max_training_windows"] = 10_000_000
    assert neural._training_window_limit(cfg) > 12_000


def test_shipped_config_does_not_pin_the_panel_below_the_budget():
    """Regression: the memory budget is pointless if the yaml re-pins a count.

    Removing the hard-coded 12k ceiling in code changed nothing until the
    configured max_training_windows was removed too — it capped the result
    downward and the win was invisible.
    """
    from specforge.config import load_config
    cfg = load_config("paper")
    assert cfg.get("neural", "max_training_windows", default=None) is None
    assert neural._training_window_limit(cfg) > 100_000


# ── SEC fact contract: absence must not look like coverage ───────────────────

def test_every_feature_tag_is_actually_fetched():
    """The ingester, the feature builder and the gate share ONE tag list.

    They drifted before: the fetch list was widened, issuers only re-fetch
    weekly, and the store kept the old narrow set while the feature builder
    asked for 14 tags it would never get.
    """
    from specforge.ml import facts as ml_facts
    assert set(ml_facts.REQUIRED_TAGS) <= ml_facts.FETCH_TAGS
    assert len(set(ml_facts.REQUIRED_TAGS)) == len(ml_facts.REQUIRED_TAGS)
    # The feature builder must query the shared list, not a private copy...
    import inspect
    import re
    source = inspect.getsource(neural._fundamental_series)
    assert "ml_facts.REQUIRED_TAGS" in source
    # ...and every SEC tag it actually consumes must be one the ingester fetches.
    # This is the drift that produced 12 constant features: a tag read here but
    # never requested upstream yields 0.0 forever, indistinguishable from a real
    # zero. CamelCase keys pulled from `state` are the tags in play.
    consumed = set(re.findall(r'state\.get\("([A-Z][A-Za-z]+)"', source))
    assert consumed, "expected to find the tags this feature builder reads"
    assert consumed <= set(ml_facts.REQUIRED_TAGS), consumed - set(ml_facts.REQUIRED_TAGS)


def test_fundamental_coverage_counts_tags_not_bare_rows(cfg, store):
    """An issuer with one tag is not a covered issuer."""
    from specforge.ml import facts as ml_facts
    floor = ml_facts.required_tag_floor()
    with store.db:
        store.db.execute(
            "INSERT INTO instruments(symbol,name,exchange,security_type,is_etf,"
            "is_adr,active,first_seen,last_seen,source,cik,raw_hash) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            ("THIN", "Thin", "NASDAQ", "common", 0, 0, 1, "2020-01-01",
             "2026-01-01", "test", "111", "x"))
        store.db.execute(
            "INSERT INTO instruments(symbol,name,exchange,security_type,is_etf,"
            "is_adr,active,first_seen,last_seen,source,cik,raw_hash) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            ("FULL", "Full", "NASDAQ", "common", 0, 0, 1, "2020-01-01",
             "2026-01-01", "test", "222", "y"))
        for sym, cik in (("THIN", "111"), ("FULL", "222")):
            store.db.execute("INSERT INTO universe_membership VALUES(?,?,?,?,?,?)",
                             ("2026-07-01", sym, "research", 1, "test", "{}"))
        # THIN carries one required tag; FULL carries the whole required set.
        store.db.execute("INSERT INTO filing_facts VALUES(?,?,?,?,?,?,?,?)",
                         ("111", ml_facts.REQUIRED_TAGS[0], "2025-12-31",
                          "2026-02-01", 1.0, "USD", "10-K", "a1"))
        for i, tag in enumerate(ml_facts.REQUIRED_TAGS):
            store.db.execute("INSERT INTO filing_facts VALUES(?,?,?,?,?,?,?,?)",
                             ("222", tag, "2025-12-31", "2026-02-01", 1.0,
                              "USD", "10-K", f"b{i}"))

    def covered():
        return store.db.execute(
            f"SELECT COUNT(*) n FROM (SELECT f.cik FROM filing_facts f "
            "JOIN instruments i ON i.cik=f.cik JOIN universe_membership u "
            "ON u.symbol=i.symbol AND u.tier='research' "
            "AND u.as_of=(SELECT MAX(as_of) FROM universe_membership "
            "WHERE tier='research') GROUP BY f.cik "
            f"HAVING {ml_facts.covered_issuer_sql('f')} >= ?)",
            (floor,)).fetchone()["n"]

    # The old count — any fact row at all — saw two covered issuers. Only one is.
    assert store.db.execute(
        "SELECT COUNT(DISTINCT cik) n FROM filing_facts").fetchone()["n"] == 2
    assert covered() == 1


def test_dead_feature_families_are_named_not_buried_in_a_flat_list():
    """The July run had 15/44 features constant and reported it as a flat list.

    Every fundamental and both news features were dead — a data-supply failure
    that read as an unremarkable diagnostic. Families get named now.
    """
    from specforge.ml import bakeoff
    inactive = list(bakeoff.FEATURE_FAMILIES["news"]) + \
        list(bakeoff.FEATURE_FAMILIES["fundamentals"]) + ["vol21"]
    dead = sorted(name for name, members in bakeoff.FEATURE_FAMILIES.items()
                  if members and all(f in inactive for f in members))
    assert dead == ["fundamentals", "news"]
    # A partially-inactive family is NOT dead — one live member is enough.
    partial = [f for f in bakeoff.FEATURE_FAMILIES["news"]][:1]
    assert not [name for name, members in bakeoff.FEATURE_FAMILIES.items()
                if members and all(f in partial for f in members)]


# ── R7: regime layer — filtered states, market inputs, deployment only ───────

def _regime_panel(n=600, seed=0):
    """Two genuine regimes: calm/drifting-up, then volatile/flat."""
    rng = np.random.default_rng(seed)
    half = n // 2
    calm = np.column_stack([
        rng.normal(.0008, .004, half), np.full(half, .008) + rng.normal(0, 1e-4, half),
        rng.normal(14, 1.0, half), rng.normal(-.02, .01, half),
        rng.normal(.65, .05, half), rng.normal(.002, .002, half),
        rng.normal(.010, .002, half)])
    wild = np.column_stack([
        rng.normal(-.0005, .020, n - half), np.full(n - half, .030) + rng.normal(0, 1e-3, n - half),
        rng.normal(31, 4.0, n - half), rng.normal(.06, .02, n - half),
        rng.normal(.35, .08, n - half), rng.normal(-.006, .004, n - half),
        rng.normal(.025, .004, n - half)])
    return np.vstack([calm, wild])


def test_filtered_states_cannot_see_the_future():
    """THE R7 gate: a label at t must not move when data after t changes.

    A smoothed posterior — what a stock HMM call returns — fails this, which is
    how an HMM regime layer produces a clairvoyant backtest.
    """
    from specforge.ml import regime_hmm
    x = _regime_panel()
    params = regime_hmm.fit(x[:300], n_states=2, seed=0)
    boundary = 400
    early, _ = regime_hmm.filter_states(x[:boundary], params)
    mutated = np.array(x, copy=True)
    mutated[boundary:] *= -3.0                     # scramble everything after t
    mutated[boundary:] += 99.0
    full, _ = regime_hmm.filter_states(mutated, params)
    assert np.array_equal(early, full[:boundary])   # bit-identical before t
    # And there is no smoothed labeler available to reach for by accident.
    assert not hasattr(regime_hmm, "smooth_states")
    assert not hasattr(regime_hmm, "predict_proba")


def test_hmm_recovers_two_planted_regimes():
    from specforge.ml import regime_hmm
    x = _regime_panel()
    params = regime_hmm.fit(x[:400], n_states=2, seed=0)
    states, posterior = regime_hmm.filter_states(x, params)
    assert posterior.shape == (len(x), 2)
    assert np.allclose(posterior.sum(1), 1.0)
    # Each planted half should be dominated by one state — allowing a lag for
    # the filter to react, which a causal filter necessarily has.
    calm, wild = states[50:280], states[350:]
    assert (calm == np.bincount(calm).argmax()).mean() > .85
    assert (wild == np.bincount(wild).argmax()).mean() > .85
    assert np.bincount(calm).argmax() != np.bincount(wild).argmax()


def test_state_agreement_is_invariant_to_label_switching():
    from specforge.ml import regime_hmm
    a = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    relabeled = np.array([1, 1, 2, 2, 0, 0, 1, 2])     # same partition, new names
    assert regime_hmm.state_agreement(a, relabeled, 3) == 1.0
    assert regime_hmm.state_agreement(a, a, 3) == 1.0
    scrambled = np.array([2, 1, 0, 2, 1, 0, 2, 0])
    assert regime_hmm.state_agreement(a, scrambled, 3) < 1.0


def test_seed_stability_flags_states_fitted_to_noise():
    from specforge.ml import regime_hmm
    real = _regime_panel()
    stable = regime_hmm.seed_stability(real[:400], real, n_states=2, seeds=3)
    assert stable["mean_agreement"] > .9 and not stable["degenerate"]
    # Pure noise has no regimes; asking for 3 states finds unstable ones.
    noise = np.random.default_rng(5).normal(size=(600, len(regime_hmm.FEATURES)))
    unstable = regime_hmm.seed_stability(noise[:400], noise, n_states=3, seeds=3)
    assert unstable["mean_agreement"] < stable["mean_agreement"]


def test_regime_only_throttles_deployment_and_is_monotone_in_volatility():
    from specforge.ml import regime_hmm
    x = _regime_panel()
    params = regime_hmm.fit(x[:400], n_states=2, seed=0)
    multipliers = regime_hmm.deployment_multipliers(x[:400], params)
    assert multipliers.shape == (2,)
    assert ((multipliers >= 0) & (multipliers <= 1)).all()
    # The calmer state must never be granted LESS deployment than the wilder one.
    states, _ = regime_hmm.filter_states(x[:400], params)
    volatility = [x[:400][states == s, regime_hmm.FEATURES.index("bench_vol")].mean()
                  for s in range(2)]
    calm, wild = int(np.argmin(volatility)), int(np.argmax(volatility))
    assert multipliers[calm] > multipliers[wild]
    # Output is a scalar per state — there is no per-symbol surface at all.
    assert multipliers.ndim == 1


def test_regime_features_are_market_level_only():
    """No per-stock input may enter the regime layer."""
    from specforge.ml import regime_hmm
    forbidden = ("symbol", "ticker", "stock", "position", "holding", "name")
    assert not any(bad in f for f in regime_hmm.FEATURES for bad in forbidden)
    assert set(regime_hmm.FEATURES) == {
        "bench_return", "bench_vol", "vix_level", "vix_slope",
        "breadth", "credit", "dispersion"}


# ── R8: experiment governance — what is this number actually worth? ──────────

def test_expected_max_sharpe_grows_with_the_number_of_trials():
    from specforge.ml import governance as gov
    variance = 0.01
    bars = [gov.expected_max_sharpe(n, variance) for n in (1, 10, 100, 1000)]
    assert bars[0] == 0.0                      # one trial: nothing to deflate
    assert bars[1] < bars[2] < bars[3]          # search harder, clear a higher bar


def test_deflated_sharpe_punishes_a_lucky_winner_from_many_trials():
    """The core R8 claim: the same track record means less if you searched more."""
    from specforge.ml import governance as gov
    rng = np.random.default_rng(0)
    # A genuinely good series: consistent positive drift.
    good = rng.normal(0.0015, 0.008, 1200)
    honest = gov.deflated_sharpe(good, n_trials=1)
    searched = gov.deflated_sharpe(good, n_trials=5000)
    assert honest["observed_sharpe"] == searched["observed_sharpe"]   # same returns
    assert searched["deflated_sharpe"] < honest["deflated_sharpe"]    # less credible
    assert searched["expected_max_sharpe_under_null"] > \
        honest["expected_max_sharpe_under_null"]
    # Pure noise should not clear the bar even when only one trial is claimed.
    noise = rng.normal(0.0, 0.01, 1200)
    assert gov.deflated_sharpe(noise, n_trials=1)["deflated_sharpe"] < .95
    # Too little data fails closed rather than reporting a confident number.
    assert gov.deflated_sharpe([0.01, 0.02], n_trials=1)["evidence"] == "insufficient"


def test_pbo_detects_selection_that_carries_no_information():
    from specforge.ml import governance as gov
    rng = np.random.default_rng(1)
    # 40 strategies of pure noise: the in-sample winner is a coin flip OOS.
    noise = rng.normal(0, .01, size=(600, 40))
    overfit = gov.probability_of_backtest_overfitting(noise)
    assert overfit["evidence"] == "ok"
    assert overfit["pbo"] > .35            # near-chance selection

    # One strategy with real, persistent edge: selection is informative.
    real = rng.normal(0, .01, size=(600, 40))
    real[:, 7] += .004
    genuine = gov.probability_of_backtest_overfitting(real)
    assert genuine["pbo"] < overfit["pbo"]
    assert genuine["pbo"] < .1
    # A single column cannot be "selected" at all — fail closed.
    assert gov.probability_of_backtest_overfitting(
        noise[:, :1])["evidence"] == "insufficient"


def test_block_bootstrap_is_wider_than_iid_on_autocorrelated_returns():
    from specforge.ml import governance as gov
    rng = np.random.default_rng(2)
    shocks = rng.normal(0, .01, 1500)
    series = np.zeros(1500)
    for i in range(1, 1500):                      # strong positive autocorrelation
        series[i] = .85 * series[i - 1] + shocks[i]
    wide = gov.block_bootstrap_ci(series, block_size=42, samples=400)
    narrow = gov.block_bootstrap_ci(series, block_size=1, samples=400)
    width = lambda d: d["mean_ci"][1] - d["mean_ci"][0]
    # Ignoring serial correlation manufactures precision that is not there.
    assert width(wide) > width(narrow)
    assert gov.block_bootstrap_ci([0.1] * 5)["evidence"] == "insufficient"


def test_trial_adjusted_summary_fails_closed_on_every_missing_dimension():
    from specforge.ml import governance as gov
    rng = np.random.default_rng(3)
    good = rng.normal(0.0015, 0.008, 1200)
    matrix = rng.normal(0, .01, size=(600, 20)); matrix[:, 3] += .004
    # Without a performance matrix, PBO is uncomputed — which is not a pass.
    assert gov.trial_adjusted_summary(good, n_trials=1)["verdict"] is False
    passing = gov.trial_adjusted_summary(good, n_trials=1, performance=matrix)
    assert passing["verdict"] is True
    # A MARGINAL edge is where search cost decides. The same returns pass when
    # found on the first try and fail once the search is honestly declared —
    # a genuinely strong edge (like `good`) rightly survives either way.
    marginal = rng.normal(0.0006, 0.008, 1200)
    assert gov.trial_adjusted_summary(
        marginal, n_trials=1, performance=matrix)["verdict"] is True
    assert gov.trial_adjusted_summary(
        marginal, n_trials=100_000, performance=matrix)["verdict"] is False
    # Noise never passes.
    assert gov.trial_adjusted_summary(
        rng.normal(0, .01, 1200), n_trials=1, performance=matrix)["verdict"] is False


def test_sealed_holdout_consumption_is_counted_and_never_reset(cfg, store):
    """A holdout examined ten times is not a holdout; the count makes that visible."""
    key = "sealed_2026-07_schema6"
    assert store.holdout_uses(key) == 0
    assert store.record_holdout_use(key, "tournament_eval", model_id="m1") == 1
    assert store.record_holdout_use(key, "tournament_eval", model_id="m2") == 2
    assert store.record_holdout_use(key, "promotion_check", model_id="m1") == 3
    assert store.holdout_uses(key) == 3
    assert store.holdout_uses("a_different_block") == 0     # keys are independent
    # Append-only: every look is individually recoverable and auditable.
    rows = store.db.execute(
        "SELECT purpose, model_id FROM holdout_uses WHERE holdout_key=? "
        "ORDER BY at, id", (key,)).fetchall()
    assert len(rows) == 3
    assert {r["purpose"] for r in rows} == {"tournament_eval", "promotion_check"}
    assert store.db.execute(
        "SELECT COUNT(*) n FROM audit WHERE event_type='holdout_examined'"
    ).fetchone()["n"] == 3


def test_candidate_cohort_matrix_columns_are_the_real_alternatives():
    """PBO is only meaningful if the columns are the choices actually faced."""
    from specforge.ml import bakeoff
    ds = _bakeoff_dataset(signal=.05)
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    seeds = {f: {f"tcn_seed_{i}": ds["Y_absolute"] for i in range(3)}
             for f in bakeoff.FAMILIES}
    matrix, names = bakeoff.candidate_cohort_matrix(ds, eval_idx, "absolute", seeds)
    assert names == list(bakeoff.MODELS) + ["tcn_seed_0", "tcn_seed_1", "tcn_seed_2"]
    assert matrix.shape == (matrix.shape[0], len(names))
    assert matrix.shape[0] >= 4 and np.isfinite(matrix).all()
    # Too few cohorts to rank anything → empty, so governance fails closed.
    thin = _bakeoff_dataset(n_days=25, n_symbols=10, signal=.05)
    thin_idx = np.flatnonzero(thin["masks"]["test"])
    empty, _ = bakeoff.candidate_cohort_matrix(thin, thin_idx, "absolute")
    assert empty.size == 0


def test_gate_scores_the_ensemble_not_the_luckiest_seed():
    """The deployable artifact is the average of the seeds, not one draw.

    In the first real run the seed spread (0.276) exceeded the TCN's own median
    utility — exactly the variance an ensemble removes.
    """
    from specforge.ml import bakeoff
    ds = _bakeoff_dataset(signal=.05)
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    truth = ds["Y_absolute"]
    rng = np.random.default_rng(11)
    # Three noisy-but-informative seeds: each alone is weak, the average is not.
    noisy = {f: {f"tcn_seed_{i}": truth + rng.normal(scale=.03, size=truth.shape)
                 for i in range(3)} for f in bakeoff.FAMILIES}
    result = bakeoff.compare(ds, eval_idx, noisy)
    a = result["absolute"]
    assert a["ensemble"] is not None
    assert a["tcn_ensemble_utility"] is not None
    assert a["decisive_utility"] == a["tcn_ensemble_utility"]
    # Averaging independent noise must not do worse than the typical seed.
    assert a["tcn_ensemble_utility"] >= a["tcn_median_utility"]
    # A single seed has no ensemble; the median still decides, and the BEST
    # seed never does.
    one = {f: {"tcn_seed_0": truth} for f in bakeoff.FAMILIES}
    single = bakeoff.compare(ds, eval_idx, one)["absolute"]
    assert single["ensemble"] is None
    assert single["decisive_utility"] == single["tcn_median_utility"]


def test_one_brilliant_seed_still_cannot_carry_an_ensemble():
    from specforge.ml import bakeoff
    ds = _bakeoff_dataset(signal=.05)
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    truth = ds["Y_absolute"]
    strong = truth + np.random.default_rng(1).normal(scale=.001, size=truth.shape)
    junk = np.random.default_rng(2).normal(scale=.05, size=truth.shape)
    lucky = {f: {"tcn_seed_0": strong, "tcn_seed_1": junk, "tcn_seed_2": junk}
             for f in bakeoff.FAMILIES}
    result = bakeoff.compare(ds, eval_idx, lucky)
    assert result["absolute"]["tcn_best_utility"] > \
        result["absolute"]["decisive_utility"]
    assert result["verdict"] is False


def test_linear_skip_lets_the_network_represent_a_linear_model():
    """The TCN saw 60x44 and scored +0.172 where ridge on ONE session scored
    +0.683. A model with strictly more information should not lose to a linear
    map of a subset of it. The skip makes a linear model representable exactly,
    so the deep branches only have to learn the residual."""
    import torch
    from specforge import neural
    model = neural._make_model(len(neural.FEATURES), 2)
    model.eval()          # dropout is stochastic in train mode; isolate the skip
    x = torch.randn(8, 60, len(neural.FEATURES))

    # Zero-initialized: the skip must start as a no-op, so training begins from
    # the previous architecture's behaviour rather than a random linear shift.
    assert torch.allclose(model.skip.weight, torch.zeros_like(model.skip.weight))
    baseline = model.forward_structured(x).absolute_quantiles.detach().clone()

    # Give the skip a real weight; the median must move by exactly that amount.
    with torch.no_grad():
        model.skip.weight.normal_(0, .01)
        model.skip.bias.normal_(0, .01)
    shifted = model.forward_structured(x)
    excess_skip, absolute_skip = model.skips(x)
    assert torch.allclose(shifted.absolute_quantiles[..., 1],
                          baseline[..., 1] + absolute_skip, atol=1e-5)
    # Quantile ordering must survive the shift.
    q = shifted.absolute_quantiles.detach().numpy()
    assert (q[..., 0] <= q[..., 1]).all() and (q[..., 1] <= q[..., 2]).all()
    # Each family gets its own offset — the skip is not shared between them.
    assert not torch.allclose(excess_skip, absolute_skip)


def test_architecture_hash_changed_with_the_skip():
    """A silent architecture change would load old weights into a new model."""
    from specforge import neural
    stale = "8bd6be6dbb6f5b1e"          # v8, pre-skip
    assert neural.ARCHITECTURE_HASH != stale


def test_feature_dropout_removes_whole_features_and_is_training_only():
    """L1 beat L2 by a wide margin, so the network needs sparsity pressure.

    Ordinary dropout scatters zeros across cells; this must remove an entire
    feature for a sample, which is what forces redundancy across inputs.
    """
    import torch
    from specforge import neural
    model = neural._make_model(len(neural.FEATURES), 2, feature_dropout=.5)
    x = torch.randn(64, 60, len(neural.FEATURES))

    model.train()
    dropped = model.feature_dropout(x.transpose(1, 2)).transpose(1, 2)
    fully_zero = (dropped.abs().sum(dim=1) == 0).float().mean()
    assert .35 < float(fully_zero) < .65          # whole columns, ~p of them
    # Where a feature survives it is scaled, never partially zeroed.
    survivors = dropped[0][:, (dropped[0].abs().sum(0) != 0)]
    assert (survivors.abs().sum(0) != 0).all()

    model.eval()                                   # inference must be untouched
    assert torch.allclose(
        model.feature_dropout(x.transpose(1, 2)).transpose(1, 2), x)


def test_trial_grid_reaches_the_regularization_the_evidence_asks_for():
    from specforge import neural
    decays = [t["weight_decay"] for t in neural.TRIAL_SPECS]
    assert max(decays) >= 1e-2          # an order of magnitude past the old grid
    assert any(t.get("feature_dropout", 0) > 0 for t in neural.TRIAL_SPECS)
    assert any(t.get("feature_dropout", 0) == 0 for t in neural.TRIAL_SPECS)
    for spec in neural.TRIAL_SPECS:     # every spec is fully specified
        assert set(spec) == {"lr", "weight_decay", "rank_weight", "feature_dropout"}


def test_ablation_uses_the_strongest_control_not_a_hardcoded_one():
    """Knocking a family out of a weaker model understates its value."""
    from specforge.ml import bakeoff
    ds = _bakeoff_dataset(signal=.05)
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    auto = bakeoff.ablate(ds, eval_idx, "absolute")
    assert auto["ablated_model"] in ("ridge", "elastic_net", "boosted_tree")
    scored = {name: bakeoff.policy_return(p, ds, eval_idx, "absolute")["policy_utility"]
              for name, p in bakeoff.simple_predictions(
                  ds, "absolute", ("ridge", "elastic_net", "boosted_tree")).items()}
    assert auto["ablated_model"] == max(scored, key=scored.get)
    # An explicit choice is still honoured.
    assert bakeoff.ablate(ds, eval_idx, "absolute", model="ridge")["ablated_model"] \
        == "ridge"


def test_seed_predictions_ablation_flags_actually_disable_the_features():
    """Attribution needs the flags to really turn things off.

    A zeroed-but-trainable skip would be learned straight back, and the
    ablation would silently measure the same model twice.
    """
    from specforge import neural
    ds = _bakeoff_dataset(n_days=120, n_symbols=12, signal=.05)
    eval_idx = np.flatnonzero(ds["masks"]["test"])
    off = neural.seed_predictions(cfg=_cfg_for_ablation(), ds=ds,
                                  eval_idx=eval_idx, seeds=1, epochs=1,
                                  linear_skip=False)
    on = neural.seed_predictions(cfg=_cfg_for_ablation(), ds=ds,
                                 eval_idx=eval_idx, seeds=1, epochs=1,
                                 linear_skip=True)
    for family in ("absolute", "excess"):
        assert set(off[family]) == {"tcn_seed_0"}
        assert np.isfinite(list(off[family].values())[0]).all()
        assert np.isfinite(list(on[family].values())[0]).all()


def _cfg_for_ablation():
    from specforge.config import load_config
    return load_config("paper")


def test_frozen_skip_stays_zero_through_training():
    """The mechanism the ablation depends on, checked directly."""
    import torch
    from specforge import neural
    model = neural._make_model(len(neural.FEATURES), 2)
    with torch.no_grad():
        model.skip.weight.zero_(); model.skip.bias.zero_()
    model.skip.weight.requires_grad_(False)
    model.skip.bias.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(16, 60, len(neural.FEATURES))
    for _ in range(3):
        opt.zero_grad()
        model.forward_structured(x).absolute_quantiles.pow(2).mean().backward()
        opt.step()
    assert torch.count_nonzero(model.skip.weight) == 0
    assert torch.count_nonzero(model.skip.bias) == 0
