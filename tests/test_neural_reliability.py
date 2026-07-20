from __future__ import annotations

import time

import numpy as np

from specforge import neural


def _history(store, symbols=("AAA", "SPY")):
    from conftest import synth_bars

    for symbol in symbols:
        store.upsert_bars(symbol, synth_bars(n_days=700, daily_drift=.001), "test")
    vix = [{**row, "open": 15, "high": 16, "low": 14, "close": 15}
           for row in synth_bars(n_days=700)]
    store.upsert_bars("^VIX", vix, "test")


def test_dataset_enforces_process_safety_cap_despite_large_config(
        cfg, store, monkeypatch):
    _history(store)
    # R6c: the ceiling is a memory budget now, so pin the budget rather than a
    # window count. Config asking for 250k windows still cannot exceed it.
    window, cap = 40, 240
    budget_gb = cap * window * len(neural.FEATURES) * 4 / 1024 ** 3
    monkeypatch.setattr(neural, "SAFE_PANEL_MEMORY_GB", budget_gb)
    cfg.data["neural"].update({"input_sessions": window, "horizons": [5, 21],
                                "panel_memory_gb": 1000.0,
                                "max_training_windows": 250_000,
                                "max_windows_per_symbol": 250_000})

    dataset = neural.build_dataset(cfg, store, symbols=["AAA"])

    assert "error" not in dataset
    assert len(dataset["X"]) <= cap
    assert dataset["X"].dtype == np.float32


def test_dataset_yields_before_work_when_deadline_has_passed(cfg, store):
    _history(store)

    result = neural.build_dataset(
        cfg, store, symbols=["AAA"], deadline=time.monotonic() - 1)

    assert result["status"] == "yielded"
    assert result["windows_built"] == 0


def test_forward_all_uses_one_encoder_pass_and_matches_separate_heads():
    import torch

    torch.manual_seed(7)
    model = neural._make_model(len(neural.FEATURES), 2).eval()
    sample = torch.randn(3, 60, len(neural.FEATURES))
    expected_quantiles = model(sample)
    expected_probability = model.probability(sample)

    calls = 0
    original_encoded = model.encoded

    def counted_encoded(value):
        nonlocal calls
        calls += 1
        return original_encoded(value)

    model.encoded = counted_encoded
    quantiles, probability = model.forward_all(sample)

    assert calls == 1
    torch.testing.assert_close(quantiles, expected_quantiles)
    torch.testing.assert_close(probability, expected_probability)


def test_compatibility_poll_uses_metadata_and_hash_without_loading_torch_payload(
        store, tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"immutable checkpoint fixture")
    sha = neural._sha256_file(checkpoint)
    with store.db:
        store.db.execute(
            "INSERT INTO model_runs(id,kind,status,checkpoint,feature_hash,"
            "schema_version,architecture_hash,checkpoint_sha256) "
            "VALUES(?,?,?,?,?,?,?,?)",
            ("run-1", "global_tcn", "challenger", str(checkpoint),
             neural.FEATURE_HASH, neural.MODEL_SCHEMA,
             neural.ARCHITECTURE_HASH, sha))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dashboard compatibility must not deserialize a model")

    monkeypatch.setattr(neural, "_load_checked", forbidden)
    assert neural.refresh_compatibility(store) == {"compatible": 1, "incompatible": 0}

    checkpoint.write_bytes(b"tampered")
    assert neural.refresh_compatibility(store) == {"compatible": 0, "incompatible": 1}
    row = store.db.execute("SELECT * FROM model_runs WHERE id='run-1'").fetchone()
    assert row["status"] == "incompatible"
    assert row["incompatibility_reason"] == "checkpoint hash mismatch"
