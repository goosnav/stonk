from __future__ import annotations

import copy

import pytest

from specforge.graph import (champion, default_topology, evaluate, fit_weights,
                             mutate, promote, save_version, validate)


def test_graph_is_real_dag_and_preserves_signed_evidence():
    topology = default_topology()
    validate(topology)
    result = evaluate(topology, {"momentum": .8, "reversal": -.6,
                                 "sector_rotation": .4, "macro_regime": .5})
    assert result["activations"]["momentum"]["value"] > 0
    assert result["activations"]["reversal"]["value"] < 0
    assert set(result["outputs"]) == {5, 21}
    assert all(evaluate(mutate(topology, seed=i), {}) for i in range(8))


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

