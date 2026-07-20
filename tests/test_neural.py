from __future__ import annotations

import numpy as np

from specforge import neural


def _long_history(store):
    from conftest import synth_bars
    for sym in ("AAA", "BBB", "CCC", "SPY"):
        store.upsert_bars(sym, synth_bars(n_days=700, daily_drift=.001), "test")
    vix = [{**r, "open": 15, "high": 16, "low": 14, "close": 15}
           for r in synth_bars(n_days=700)]
    store.upsert_bars("^VIX", vix, "test")


def test_tcn_quantiles_are_ordered():
    import torch
    model = neural._make_model(len(neural.FEATURES), 2)
    out = model(torch.randn(4, 60, len(neural.FEATURES))).detach().numpy()
    assert out.shape == (4, 2, 3)
    assert np.all(out[:, :, 0] <= out[:, :, 1])
    assert np.all(out[:, :, 1] <= out[:, :, 2])
    probability = model.probability(torch.randn(4, 60, len(neural.FEATURES))).detach().numpy()
    assert probability.shape == (4, 2) and np.all((probability >= 0) & (probability <= 1))
    assert model.context[0].in_features == len(neural.CONTEXT_FEATURES)
    assert len(model.excess_quantile_heads) == 2 and len(model.excess_probability_heads) == 2
    assert len(model.absolute_quantile_heads) == 2 and len(model.absolute_probability_heads) == 2
    assert "news_sentiment" in neural.FEATURES and "dilution_missing" in neural.FEATURES


def test_daily_rank_ic_is_cross_sectional():
    dates = np.array(["2026-01-01"] * 8 + ["2026-01-02"] * 8)
    truth = np.tile(np.arange(8), 2)
    pred = np.r_[np.arange(8) * 100, np.arange(8) - 50]
    assert neural._rank_ic(pred, truth, dates) == 1.0


def test_dataset_is_chronological_and_train_normalized(cfg, store):
    _long_history(store)
    cfg.data["neural"]["input_sessions"] = 40
    cfg.data["neural"]["horizons"] = [5, 21]
    ds = neural.build_dataset(cfg, store, symbols=["AAA", "BBB", "CCC"])
    assert "error" not in ds
    assert ds["train_end"] < ds["val_start"] < ds["test_start"]
    unique = sorted(set(ds["dates"]))
    assert unique.index(ds["val_start"]) - unique.index(ds["train_end"]) > 21
    assert unique.index(ds["test_start"]) - unique.index(ds["val_end"]) > 21
    train = ds["X"][ds["masks"]["train"]]
    assert abs(float(train.mean())) < .05


def test_training_writes_challenger_not_champion(cfg, store, tmp_path):
    _long_history(store)
    cfg.data["neural"].update({"input_sessions": 40, "horizons": [5, 21],
                                "max_epochs": 2, "patience": 1,
                                "checkpoint": str(tmp_path / "global.pt"),
                                "max_trials_per_snapshot": 1})
    out = neural.train_challenger(cfg, store, symbols=["AAA", "BBB", "CCC"], max_seconds=2)
    assert out["status"] == "challenger"
    assert not (tmp_path / "global.pt").exists()
    row = store.db.execute("SELECT * FROM model_runs WHERE id=?", (out["id"],)).fetchone()
    assert row and __import__("pathlib").Path(row["checkpoint"]).exists()
    assert row["checkpoint_sha256"] and row["schema_version"] == neural.MODEL_SCHEMA
    payload, _, reason = neural._load_checked(
        __import__("pathlib").Path(row["checkpoint"]), row["checkpoint_sha256"])
    assert reason is None and len(payload["target_scale"]) == 2
    assert payload["active_features"]
    assert payload["calibration"]["quantile_offsets"]
    assert payload["trial_spec"] == neural.TRIAL_SPECS[0]
    assert row["schema_version"] == neural.MODEL_SCHEMA
    assert payload["calibration_structured"]["absolute"]["prob_threshold"] == payload["round_trip_cost"]
    assert payload["target_scale_absolute"] is not None
    assert store.db.execute("SELECT COUNT(*) n FROM model_runs WHERE status='champion'").fetchone()["n"] == 0
    assert neural.train_challenger(cfg, store, symbols=["AAA", "BBB", "CCC"])["status"] == "caught_up"
    # Immutable artifact integrity is enforced before inference or promotion.
    with open(row["checkpoint"], "ab") as damaged:
        damaged.write(b"tamper")
    compatibility = neural.refresh_compatibility(store)
    assert compatibility["incompatible"] == 1
    rejected = store.db.execute("SELECT * FROM model_runs WHERE id=?", (out["id"],)).fetchone()
    assert rejected["status"] == "incompatible"
    assert "hash mismatch" in rejected["incompatibility_reason"]


def test_holding_network_requires_full_symbol_history(cfg, store):
    from conftest import synth_bars
    store.upsert_bars("AAA", synth_bars(n_days=700), "test")
    cfg.data["neural"]["holding_min_bars"] = 1250
    result = neural.train_challenger(cfg, store, symbol="AAA")
    assert result["status"] == "waiting"
    assert "1250" in result["reason"]


def test_holding_gate_requires_improvement_at_both_horizons():
    good = {str(h): {"pinball": .009, "correlation": .1,
                     "directional_accuracy": .53, "coverage": .8}
            for h in (5, 21)}
    good["global_baseline"] = {str(h): {"pinball": .01} for h in (5, 21)}
    assert neural.holding_gate_passed(good)
    good["21"]["coverage"] = .9
    assert not neural.holding_gate_passed(good)


def test_shadow_metrics_exclude_historical_walk_forward_rows(store):
    rows = []
    for i in range(10):
        rows.append(("m", "2025-01-02", f"S{i}", 5, -.1, i / 100, .1, .6,
                     "historical_oos", i / 100, neural.FEATURE_HASH))
    with store.db:
        store.db.executemany(
            "INSERT INTO model_forecasts VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
    assert neural.shadow_metrics(store, "m")["total"] == 0
    with store.db:
        store.db.execute(
            "UPDATE model_forecasts SET resolved_at='2026-01-10T00:00:00' "
            "WHERE model_id='m'")
    metrics = neural.shadow_metrics(store, "m")
    assert metrics["sessions"] == 1 and metrics["horizons"]["5"]["n"] == 10


def test_offline_gate_requires_profitable_folds_and_sealed_test():
    metrics = {str(h): {"correlation": .03,
                        "top_decile_alpha_after_cost": .01,
                        "coverage": .8} for h in (5, 21)}
    metrics["folds"] = [{"ic_5d": .03, "ic_21d": .04,
                          "net_alpha_5d": .01, "net_alpha_21d": .02}
                         for _ in range(5)]
    metrics.update(median_fold_ic_5d=.03, median_fold_ic_21d=.04)
    metrics["beats_baselines"] = True
    # R1: the gate demands BOTH families on structured outputs
    metrics["absolute"] = {str(h): {"correlation": .02,
                                    "top_decile_alpha_after_cost": .005}
                           for h in (5, 21)}
    # R6: and the model must have won the policy-return bakeoff.
    metrics["bakeoff"] = {"verdict": True,
                          "basis": "net_oos_policy_return_staggered_cohorts"}
    assert neural._offline_gate(metrics)
    metrics["absolute"]["21"]["top_decile_alpha_after_cost"] = -.001
    assert not neural._offline_gate(metrics)       # absolute loses → no entry
    metrics["absolute"]["21"]["top_decile_alpha_after_cost"] = .005
    metrics["21"]["top_decile_alpha_after_cost"] = -.001
    assert not neural._offline_gate(metrics)       # excess loses → no entry


def test_tournament_selection_rewards_after_cost_rank_alpha():
    weak = {str(h): {"correlation": .02, "top_decile_alpha_after_cost": -.01,
                     "coverage": .8, "pinball": .01} for h in (5, 21)}
    useful = {str(h): {"correlation": .04, "top_decile_alpha_after_cost": .01,
                       "coverage": .8, "pinball": .012} for h in (5, 21)}
    assert neural._selection_score(useful) > neural._selection_score(weak)


def test_validation_calibration_preserves_quantile_order_and_probability_bounds():
    pred = np.array([[[.01, .02, .03], [-.02, 0, .02]],
                     [[.02, .03, .04], [-.01, .01, .03]]], dtype=float)
    probability = np.array([[.3, .6], [.4, .7]])
    truth = np.array([[.05, -.01], [.06, .04]])
    calibrated, p = neural._apply_calibration(
        pred, probability, neural._calibration(pred, probability, truth))
    assert np.all(calibrated[:, :, 0] <= calibrated[:, :, 1])
    assert np.all(calibrated[:, :, 1] <= calibrated[:, :, 2])
    assert np.all((p > 0) & (p < 1))


def test_sec_facts_activate_valuation_and_event_features(cfg, store):
    import pandas as pd
    bars = neural._bars(store, "AAA", "1900-01-01")
    filed = bars.index[10]
    with store.db:
        store.db.execute("INSERT INTO instruments(symbol,name,exchange,security_type,is_etf,is_adr,active,first_seen,last_seen,source,cik,raw_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                         ("AAA", "Alpha", "NASDAQ", "common", 0, 0, 1,
                          "2020-01-01", "2026-01-01", "test", "123", "x"))
        store.db.execute("INSERT INTO filing_facts VALUES(?,?,?,?,?,?,?,?)",
                         ("123", "EarningsPerShareDiluted", "2024-12-31",
                          filed, 5.0, "USD/shares", "10-K", "a"))
    spy = neural._bars(store, "SPY", "1900-01-01")
    with np.errstate(over="raise", invalid="raise"):
        features = neural._features(bars, spy, pd.DataFrame(), store, "AAA")
    assert features.loc[features.index < filed, "event_proximity"].eq(0).all()
    assert features.loc[features.index >= filed, "valuation_missing"].eq(0).all()
    assert features.loc[filed, "event_proximity"] == 1.0
