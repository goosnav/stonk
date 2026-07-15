from __future__ import annotations

import copy

import pytest

from specforge.graph import (champion, default_topology, evaluate, fit_weights,
                             mutate, promote, save_version, validate,
                             walk_forward_fit)


def test_graph_is_real_dag_and_preserves_signed_evidence():
    topology = default_topology()
    validate(topology)
    result = evaluate(topology, {"momentum": .8, "gap": -.6,
                                 "sector_rotation": .4, "macro_regime": .5})
    assert result["activations"]["momentum"]["value"] > 0
    assert result["activations"]["gap"]["value"] < 0
    assert set(result["outputs"]) == {5, 21}
    assert all(evaluate(mutate(topology, seed=i), {}) for i in range(8))


def test_graph_snapshot_records_topology_identity(cfg, store):
    from specforge.graph import blend_candidates
    blend_candidates([], [], "neutral", cfg, store, "cycle-1")
    saved = store.kv_get("graph_last_activations")
    assert saved["topology_schema"] == default_topology()["schema"]
    assert saved["graph_version"] == "default"


def test_graph_rejects_cycles_and_governor_nodes():
    t = default_topology()
    t["nodes"].append({"id": "risk_governor", "layer": 3, "role": "gate",
                       "activation": "sigmoid"})
    with pytest.raises(ValueError, match="risk/execution"):
        validate(t)
    t = default_topology()
    t["edges"].append({"source": "output_21d", "target": "momentum", "weight": 1})
    with pytest.raises(ValueError, match="acyclic"):
        validate(t)


def test_graph_backprop_and_atomic_champion(store):
    bases, targets = [], []
    for i in range(140):
        x = (i % 20 - 10) / 10
        bases.append({"momentum": x, "sector_rotation": x * .5,
                      "macro_regime": .5})
        targets.append([x * .03, x * .06])
    learned, metrics = fit_weights(default_topology(), bases, targets,
                                   max_epochs=8, patience=3)
    assert metrics["n_test"] > 0 and "test_ic_21d" in metrics
    vid = save_version(store, learned, metrics=metrics)
    assert champion(store)["status"] == "shadow"
    promote(store, vid)
    assert champion(store)["id"] == vid
    # Saving another challenger cannot mutate the live champion.
    save_version(store, copy.deepcopy(learned), metrics={"bad": True})
    assert champion(store)["id"] == vid


def test_graph_walk_forward_has_real_embargo_and_sealed_coverage():
    bases, targets = [], []
    from datetime import date, timedelta
    for day in range(80):
        for symbol in range(8):
            x = (symbol - 3.5) / 3.5 + (day % 5) * .01
            bases.append({"__date": (date(2025, 1, 1) + timedelta(days=day)).isoformat(),
                          "__symbol": f"S{symbol}", "momentum": x,
                          "sector_rotation": x * .4, "macro_regime": .3})
            targets.append([x * .02 + (day % 3 - 1) * .001,
                            x * .04 + (day % 5 - 2) * .001])
    _, metrics = walk_forward_fit(default_topology(), bases, targets)
    assert metrics["walk_forward_folds"] == 5
    assert all(f["embargo"] == 21 for f in metrics["folds"])
    assert all(f["embargo_unit"] == "sessions" for f in metrics["folds"])
    assert all(f["train_end"] < f["test_start"] for f in metrics["folds"])
    assert 0 <= metrics["coverage_5d"] <= 1
    assert 0 <= metrics["coverage_21d"] <= 1


def test_legacy_backtest_signals_recover_simulated_date(store):
    import json
    from specforge.research import graph_samples
    dates = [r["d"] for r in store.get_bars("AAA", "9999-12-31", 10_000)]
    simulated = dates[-40]
    cycle = "historical-cycle"
    store.db.execute(
        "INSERT INTO signals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("sig", cycle, "2099-01-01T00:00:00", "momentum", "AAA", "long",
         .7, .8, 21, .03, .1, -.1, json.dumps(["test"])))
    store.db.execute(
        "INSERT INTO audit(ts,cycle_id,event_type,payload) VALUES(?,?,?,?)",
        ("2099-01-01T00:00:00", cycle, "cycle_start",
         json.dumps({"as_of": simulated})))
    store.db.commit()
    bases, targets = graph_samples(store)
    assert len(bases) == len(targets) == 1
    assert bases[0]["momentum"] == pytest.approx(.56)
