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


def test_dataset_is_chronological_and_train_normalized(cfg, store):
    _long_history(store)
    cfg.data["neural"]["input_sessions"] = 40
    cfg.data["neural"]["horizons"] = [5, 21]
    ds = neural.build_dataset(cfg, store, symbols=["AAA", "BBB", "CCC"])
    assert "error" not in ds
    assert ds["train_end"] < ds["val_start"] < ds["test_start"]
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
    assert (tmp_path / "global.challenger.pt").exists()
    assert store.db.execute("SELECT COUNT(*) n FROM model_runs WHERE status='champion'").fetchone()["n"] == 0
    assert neural.train_challenger(cfg, store, symbols=["AAA", "BBB", "CCC"])["status"] == "caught_up"


def test_holding_network_requires_full_symbol_history(cfg, store):
    from conftest import synth_bars
    store.upsert_bars("AAA", synth_bars(n_days=700), "test")
    cfg.data["neural"]["holding_min_bars"] = 1250
    result = neural.train_challenger(cfg, store, symbol="AAA")
    assert result["status"] == "waiting"
    assert "1250" in result["reason"]
