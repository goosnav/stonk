"""Learned analog-neural graph over auditable specialist equations.

Specialists still compute their own SignalEvents.  This module learns only
how their signed activations flow through a bounded DAG; the governor remains
outside the graph.  Champion rows are immutable and challengers are separate
graph_versions records, so research can never degrade the live graph in place.
"""
from __future__ import annotations

import copy
import json
import math
import random
from datetime import datetime

from .models import SignalEvent, new_id

FORBIDDEN = {"risk_governor", "execution", "broker", "portfolio"}
MAX_LAYER = 4
MAX_FAN_IN = 8


def default_topology() -> dict:
    nodes = [
        # layer 1: stock-local specialists
        *(dict(id=n, layer=1, role="alpha", activation="tanh", base_scale=1.0,
               bias=0.0) for n in ("momentum", "reversal", "earnings_drift",
                                    "vol_contraction", "fundamentals", "neural")),
        dict(id="quality_value", layer=1, role="gate", activation="sigmoid",
             base_scale=1.0, bias=0.0),
        # layer 2: cross-sectional/context specialists
        *(dict(id=n, layer=2, role="alpha", activation="tanh", base_scale=1.0,
               bias=0.0) for n in ("sector_rotation", "news_sentiment",
                                    "congress_trades", "insider", "gap")),
        dict(id="macro_regime", layer=2, role="gate", activation="sigmoid",
             base_scale=1.0, bias=0.0),
        # learned interaction and output units
        *(dict(id=n, layer=3, role="interaction", activation="leaky_relu",
               base_scale=0.0, bias=0.0) for n in
          ("trend_confirmation", "catalyst_confirmation",
           "fundamental_confirmation", "risk_adjusted_conviction")),
        dict(id="output_5d", layer=4, role="output", activation="linear",
             base_scale=0.0, bias=0.0),
        dict(id="output_21d", layer=4, role="output", activation="linear",
             base_scale=0.0, bias=0.0),
    ]
    edges = []
    def connect(sources, target, weight):
        edges.extend({"source": s, "target": target, "weight": weight,
                      "pruned": False} for s in sources)
    connect(["momentum", "reversal", "vol_contraction", "neural"],
            "trend_confirmation", 0.25)
    connect(["earnings_drift", "news_sentiment", "gap"],
            "catalyst_confirmation", 0.30)
    connect(["fundamentals", "quality_value", "insider", "congress_trades"],
            "fundamental_confirmation", 0.22)
    connect(["sector_rotation", "macro_regime"],
            "risk_adjusted_conviction", 0.20)
    connect(["trend_confirmation", "catalyst_confirmation",
             "risk_adjusted_conviction"], "output_5d", 0.33)
    connect(["trend_confirmation", "fundamental_confirmation",
             "risk_adjusted_conviction"], "output_21d", 0.33)
    out = {"schema": "stonk.graph.v1", "nodes": list(nodes), "edges": edges}
    validate(out)
    return out


def validate(topology: dict) -> None:
    nodes = topology.get("nodes") or []
    edges = topology.get("edges") or []
    by_id = {n["id"]: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("duplicate graph node id")
    if FORBIDDEN & set(by_id):
        raise ValueError("risk/execution components cannot be graph nodes")
    if any(not 1 <= int(n["layer"]) <= MAX_LAYER for n in nodes):
        raise ValueError("graph layer outside 1..4")
    if len(edges) > max(1, len(nodes) * 4):
        raise ValueError("graph exceeds edge cap")
    fan_in: dict[str, int] = {}
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s not in by_id or t not in by_id:
            raise ValueError("edge references missing node")
        if by_id[s]["layer"] >= by_id[t]["layer"]:
            raise ValueError("graph must be acyclic and layer-forward")
        fan_in[t] = fan_in.get(t, 0) + 1
        if fan_in[t] > MAX_FAN_IN:
            raise ValueError("graph fan-in exceeds 8")
    for n in nodes:
        allowed = ({"sigmoid"} if n["role"] == "gate" else
                   {"linear"} if n["role"] == "output" else
                   {"tanh", "leaky_relu"})
        if n.get("activation") not in allowed:
            raise ValueError(f"activation incompatible with {n['role']}")


def _activate(kind: str, value: float) -> float:
    if kind == "tanh":
        return math.tanh(value)
    if kind == "sigmoid":
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))
    if kind == "leaky_relu":
        return value if value >= 0 else 0.05 * value
    return value


def event_bases(events: list[SignalEvent], symbol: str, regime: str) -> dict[str, float]:
    bases: dict[str, list[float]] = {}
    for e in events:
        if e.symbol != symbol:
            continue
        sign = 1.0 if e.direction in ("long", "long_call") else -1.0
        bases.setdefault(e.node_id, []).append(sign * e.score * e.confidence)
    out = {k: sum(v) / len(v) for k, v in bases.items()}
    out["macro_regime"] = {"risk_on": 1.0, "neutral": 0.3,
                           "risk_off": -0.5, "stress": -1.0}.get(regime, 0.0)
    # A missing filter result means neutral, not a free pass encoded as 1.
    out.setdefault("quality_value", 0.0)
    return out


def evaluate(topology: dict, bases: dict[str, float]) -> dict:
    validate(topology)
    incoming: dict[str, list[dict]] = {}
    for e in topology["edges"]:
        if not e.get("pruned"):
            incoming.setdefault(e["target"], []).append(e)
    activations, details = {}, {}
    for node in sorted(topology["nodes"], key=lambda n: (n["layer"], n["id"])):
        base = float(bases.get(node["id"], 0.0))
        contributions = [{"source": e["source"],
                          "value": activations.get(e["source"], 0.0) * e["weight"]}
                         for e in incoming.get(node["id"], [])]
        z = node.get("base_scale", 1.0) * base + node.get("bias", 0.0) + \
            sum(c["value"] for c in contributions)
        a = _activate(node["activation"], z)
        activations[node["id"]] = a
        details[node["id"]] = {"raw": round(base, 6), "preactivation": round(z, 6),
                               "activation": node["activation"],
                               "value": round(a, 6), "incoming": contributions}
    return {"outputs": {h: activations.get(f"output_{h}d", 0.0)
                        for h in (5, 21)},
            "activations": details}


def champion(store) -> dict:
    row = store.db.execute("SELECT * FROM graph_versions WHERE status='champion' "
                           "ORDER BY created_at DESC LIMIT 1").fetchone()
    if row:
        return {**dict(row), "topology": json.loads(row["topology"]),
                "metrics": json.loads(row["metrics"] or "{}")}
    return {"id": "default", "status": "shadow", "topology": default_topology(),
            "metrics": {"gating": "SHADOW — no validated champion"}}


def save_version(store, topology: dict, status: str = "challenger",
                 metrics: dict | None = None, parent_id: str | None = None,
                 data_as_of: str | None = None, checkpoint: str = "") -> str:
    validate(topology)
    vid = new_id()
    store.db.execute("INSERT INTO graph_versions VALUES(?,?,?,?,?,?,?,?)", (
        vid, datetime.now().astimezone().isoformat(timespec="seconds"), data_as_of,
        status, parent_id, json.dumps(topology), json.dumps(metrics or {}), checkpoint))
    store.db.commit()
    store.audit("graph_version_saved", {"id": vid, "status": status,
                                         "parent": parent_id, "metrics": metrics or {}})
    return vid


def promote(store, version_id: str) -> None:
    """Atomic champion swap. A failed transaction leaves the old champion."""
    with store.db:
        row = store.db.execute("SELECT id FROM graph_versions WHERE id=?",
                               (version_id,)).fetchone()
        if not row:
            raise ValueError("unknown graph version")
        store.db.execute("UPDATE graph_versions SET status='retired' "
                         "WHERE status='champion'")
        store.db.execute("UPDATE graph_versions SET status='champion' WHERE id=?",
                         (version_id,))
    store.audit("graph_champion_promoted", {"id": version_id})


def mutate(topology: dict, seed: int = 0) -> dict:
    """One bounded topology mutation; invalid proposals fall back unchanged."""
    rng, out = random.Random(seed), copy.deepcopy(topology)
    nodes, edges = out["nodes"], out["edges"]
    action = rng.choice(("remove", "add", "move", "activation"))
    if action == "remove" and edges:
        edges.pop(rng.randrange(len(edges)))
    elif action == "add":
        existing = {(e["source"], e["target"]) for e in edges}
        pairs = [(a["id"], b["id"]) for a in nodes for b in nodes
                 if a["layer"] < b["layer"] and (a["id"], b["id"]) not in existing]
        if pairs:
            s, t = rng.choice(pairs)
            edges.append({"source": s, "target": t, "weight": rng.uniform(-0.3, 0.3),
                          "pruned": False})
    elif action == "move":
        movable = [n for n in nodes if n["role"] in ("alpha", "interaction")]
        if movable:
            n = rng.choice(movable)
            legal = [x for x in range(1, 4) if x != n["layer"]]
            old = n["layer"]; n["layer"] = rng.choice(legal)
            try:
                validate(out)
            except ValueError:
                n["layer"] = old
    else:
        movable = [n for n in nodes if n["role"] in ("alpha", "interaction")]
        if movable:
            n = rng.choice(movable)
            n["activation"] = "leaky_relu" if n["activation"] == "tanh" else "tanh"
    validate(out)
    return out


def fit_weights(topology: dict, bases: list[dict[str, float]], targets,
                max_epochs: int = 50, patience: int = 5,
                prune_pct: float = 0.01) -> tuple[dict, dict]:
    """Backpropagate edge/base weights for one challenger topology.

    `bases` and two-column targets must already be chronological. The final
    15% is a sealed report set; only validation controls early stopping.
    """
    import numpy as np
    import torch
    validate(topology)
    out = copy.deepcopy(topology)
    nodes = sorted(out["nodes"], key=lambda n: (n["layer"], n["id"]))
    by = {n["id"]: i for i, n in enumerate(nodes)}
    X = torch.tensor([[row.get(n["id"], 0.0) for n in nodes] for row in bases],
                     dtype=torch.float32)
    Y = torch.tensor(np.asarray(targets, dtype=np.float32))
    if len(X) < 100 or Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError("graph training needs >=100 chronological 5d/21d samples")
    edge_w = torch.nn.Parameter(torch.tensor(
        [float(e.get("weight", 0.0)) for e in out["edges"]], dtype=torch.float32))
    scales = torch.nn.Parameter(torch.tensor(
        [float(n.get("base_scale", 1.0)) for n in nodes], dtype=torch.float32))
    biases = torch.nn.Parameter(torch.tensor(
        [float(n.get("bias", 0.0)) for n in nodes], dtype=torch.float32))
    params = [edge_w, scales, biases]
    opt = torch.optim.AdamW(params, lr=0.01, weight_decay=1e-4)
    split1, split2 = int(len(X) * .70), int(len(X) * .85)

    def forward(inp):
        acts = {}
        for idx, n in enumerate(nodes):
            z = scales[idx] * inp[:, idx] + biases[idx]
            for ei, e in enumerate(out["edges"]):
                if e["target"] == n["id"] and not e.get("pruned"):
                    z = z + edge_w[ei] * acts[e["source"]]
            if n["activation"] == "tanh": z = torch.tanh(z)
            elif n["activation"] == "sigmoid": z = torch.sigmoid(z)
            elif n["activation"] == "leaky_relu": z = torch.nn.functional.leaky_relu(z, .05)
            acts[n["id"]] = z
        return torch.stack((acts["output_5d"], acts["output_21d"]), dim=1)

    best, best_loss, stale = None, float("inf"), 0
    for _ in range(max_epochs):
        opt.zero_grad(); pred = forward(X[:split1])
        loss = torch.nn.functional.huber_loss(pred, Y[:split1], delta=.05)
        loss = loss + 1e-5 * edge_w.abs().sum(); loss.backward(); opt.step()
        with torch.no_grad():
            val = float(torch.nn.functional.huber_loss(
                forward(X[split1:split2]), Y[split1:split2], delta=.05))
        if val < best_loss - 1e-7:
            best_loss, stale = val, 0
            best = [p.detach().clone() for p in params]
        else:
            stale += 1
        if stale >= patience:
            break
    if best:
        with torch.no_grad():
            for p, v in zip(params, best): p.copy_(v)
    for i, e in enumerate(out["edges"]): e["weight"] = round(float(edge_w[i].detach()), 6)
    for i, n in enumerate(nodes):
        n["base_scale"] = round(float(scales[i].detach()), 6)
        n["bias"] = round(float(biases[i].detach()), 6)
    # Contribution-normalized pruning within each target; signed weak edges
    # disappear but remain in the topology so mutation can regrow them.
    for target in {e["target"] for e in out["edges"]}:
        group = [e for e in out["edges"] if e["target"] == target]
        denom = sum(abs(e["weight"]) for e in group) or 1.0
        for e in group: e["pruned"] = abs(e["weight"]) / denom < prune_pct
    with torch.no_grad(): test = forward(X[split2:]).numpy()
    truth = Y[split2:].numpy()
    corrs = [_safe_corr(test[:, i], truth[:, i]) for i in range(2)]
    metrics = {"validation_huber": round(best_loss, 6),
               "test_ic_5d": round(corrs[0], 4), "test_ic_21d": round(corrs[1], 4),
               "n_train": split1, "n_validation": split2 - split1,
               "n_test": len(X) - split2,
               "pruned_edges": sum(bool(e.get("pruned")) for e in out["edges"])}
    validate(out)
    return out, metrics


def _safe_corr(a, b) -> float:
    import numpy as np
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 5 and np.std(a) and np.std(b) else 0.0


def walk_forward_fit(topology: dict, bases: list[dict[str, float]], targets,
                     folds: int = 5, prune_pct: float = .01) -> tuple[dict, dict]:
    """Expanding-window topology fit; every fold scores only later rows."""
    import numpy as np
    if len(bases) < 180:
        learned, metrics = fit_weights(topology, bases, targets, prune_pct=prune_pct)
        metrics["walk_forward_folds"] = 0
        metrics["walk_forward_reason"] = "need at least 180 chronological samples"
        return learned, metrics
    initial = max(100, int(len(bases) * .45))
    width = max(10, (len(bases) - initial) // folds)
    fold_metrics = []
    for index in range(folds):
        train_end = initial + index * width
        test_end = len(bases) if index == folds - 1 else min(len(bases), train_end + width)
        learned, _ = fit_weights(topology, bases[:train_end], targets[:train_end],
                                 max_epochs=30, prune_pct=prune_pct)
        pred = np.asarray([[evaluate(learned, b)["outputs"][5],
                            evaluate(learned, b)["outputs"][21]]
                           for b in bases[train_end:test_end]])
        truth = np.asarray(targets[train_end:test_end])
        fold_metrics.append({"fold": index + 1, "train": train_end,
                             "test": test_end - train_end,
                             "ic_5d": round(_safe_corr(pred[:, 0], truth[:, 0]), 4),
                             "ic_21d": round(_safe_corr(pred[:, 1], truth[:, 1]), 4)})
    learned, metrics = fit_weights(topology, bases, targets, prune_pct=prune_pct)
    metrics.update(walk_forward_folds=folds, folds=fold_metrics,
                   median_fold_ic_5d=round(float(np.median(
                       [f["ic_5d"] for f in fold_metrics])), 4),
                   median_fold_ic_21d=round(float(np.median(
                       [f["ic_21d"] for f in fold_metrics])), 4))
    return learned, metrics


def blend_candidates(candidates, events, regime: str, cfg, store,
                     cycle_id: str | None = None) -> None:
    """Apply the validated graph as a capped score blend, in place."""
    champ = champion(store)
    blend = float(cfg.get("analog_graph", "live_blend", default=0.0) or 0.0)
    if champ["status"] != "champion":
        blend = 0.0
    snapshots = {}
    by_symbol = sorted({e.symbol for e in events} | {c.symbol for c in candidates} |
                       set(cfg.get("universe", "symbols", default=[])))
    for symbol in by_symbol:
        snapshots[symbol] = evaluate(
            champ["topology"], event_bases(events, symbol, regime))
    for c in candidates:
        result = snapshots[c.symbol]
        graph_score = math.tanh(float(result["outputs"][21]))
        if blend:
            c.final_score = round((1 - blend) * c.final_score + blend * graph_score, 4)
            c.thesis = (c.thesis + f"; analog_graph:{graph_score:+.3f}")[:400]
            c.contributing_nodes = sorted(set(c.contributing_nodes + ["analog_graph"]))
        snapshots[c.symbol] = {"graph_score": graph_score, **result}
    as_of = max((e.data_as_of.strftime("%Y-%m-%d") for e in events), default=None)
    store.kv_set("graph_last_activations", {
        "as_of": as_of, "cycle_id": cycle_id, "symbols": snapshots})
    # Evaluate the latest challenger in shadow across every signaled symbol,
    # not merely names selected for a trade. This supplies unbiased forward
    # labels for topology gates while leaving candidate scores untouched.
    challenger = store.db.execute(
        "SELECT * FROM graph_versions WHERE status='challenger' "
        "ORDER BY created_at DESC LIMIT 1").fetchone()
    if challenger:
        topology = json.loads(challenger["topology"])
        by_symbol = sorted({e.symbol for e in events})
        with store.db:
            for symbol in by_symbol:
                result = evaluate(topology, event_bases(events, symbol, regime))
                symbol_events = [e for e in events if e.symbol == symbol]
                spread = max(.01, min(.25, sum(e.expected_volatility for e in symbol_events)
                                      / max(1, len(symbol_events))))
                for h in (5, 21):
                    q50 = float(result["outputs"][h])
                    store.db.execute(
                        "INSERT OR IGNORE INTO model_forecasts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (challenger["id"], as_of, symbol, h, q50 - spread, q50,
                         q50 + spread, 1 / (1 + math.exp(-q50 / spread)),
                         None, None, "analog_graph"))


def maybe_promote(cfg, store) -> dict:
    """Forward-shadow gate for topology challengers; single-split graphs stay shadow."""
    row = store.db.execute("SELECT * FROM graph_versions WHERE status='challenger' "
                           "ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row:
        return {"action": "none"}
    from .neural import shadow_metrics
    sm, metrics = shadow_metrics(store, row["id"]), json.loads(row["metrics"] or "{}")
    hs = sm["horizons"]
    folds = metrics.get("folds") or []
    fold_gate = len(folds) >= 5 and \
        sum(f.get("ic_5d", 0) > 0 for f in folds) >= 4 and \
        sum(f.get("ic_21d", 0) > 0 for f in folds) >= 4 and \
        min(f.get("ic_5d", -1) for f in folds) > -.01 and \
        min(f.get("ic_21d", -1) for f in folds) > -.01 and \
        metrics.get("median_fold_ic_5d", 0) >= .015 and \
        metrics.get("median_fold_ic_21d", 0) >= .02
    passed = fold_gate and sm["sessions"] >= 30 and all(
        hs.get(str(h), {}).get("n", 0) >= 10_000 and
        hs[str(h)].get("ic", 0) >= .01 and hs[str(h)].get("top_decile_alpha", 0) > 0
        for h in (5, 21))
    if passed:
        promote(store, row["id"])
        return {"action": "promote", "id": row["id"], "metrics": sm}
    return {"action": "shadow", "id": row["id"], "metrics": sm}
