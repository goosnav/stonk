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

from .models import SignalEvent, new_id, signed_alpha

FORBIDDEN = {"risk_governor", "execution", "broker", "portfolio"}
MAX_LAYER = 4
MAX_FAN_IN = 8
MIN_SPECIALIST_EDGE_WEIGHT = 0.02
MIN_SPECIALIST_BASE_SCALE = 0.05


def default_topology() -> dict:
    nodes = [
        # layer 1: stock-local specialists
        *(dict(id=n, layer=1, role="alpha", activation="tanh", base_scale=1.0,
               bias=0.0) for n in ("momentum", "vol_contraction", "neural")),
        *(dict(id=n, layer=1, role="alpha", activation="tanh", base_scale=1.0,
               bias=0.0) for n in ("reversal", "earnings_drift",
                                   "business_fundamentals")),
        dict(id="quality_value", layer=1, role="gate", activation="sigmoid",
             base_scale=1.0, bias=0.0),
        # layer 2: cross-sectional/context specialists
        *(dict(id=n, layer=2, role="alpha", activation="tanh", base_scale=1.0,
               bias=0.0) for n in ("sector_rotation", "gap", "company_catalyst",
                                   "news_sentiment", "congress_trades", "insider",
                                   "hypothesis", "strategy_context")),
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
    connect(["earnings_drift", "company_catalyst", "news_sentiment", "gap",
             "congress_trades", "insider"],
            "catalyst_confirmation", 0.30)
    connect(["business_fundamentals", "quality_value"],
            "fundamental_confirmation", 0.22)
    connect(["sector_rotation", "macro_regime", "hypothesis", "strategy_context"],
            "risk_adjusted_conviction", 0.20)
    connect(["trend_confirmation", "catalyst_confirmation",
             "risk_adjusted_conviction"], "output_5d", 0.33)
    connect(["trend_confirmation", "fundamental_confirmation",
             "risk_adjusted_conviction"], "output_21d", 0.33)
    out = {"schema": "stonk.graph.v5", "nodes": list(nodes), "edges": edges,
           "excluded_experimental": []}
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
    outputs = {n["id"] for n in nodes if n["role"] == "output"}
    for node in (n for n in nodes if n["role"] in ("alpha", "gate")):
        seen, frontier, found = set(), [node["id"]], False
        while frontier:
            current = frontier.pop()
            if current in outputs:
                found = True; break
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(e["target"] for e in edges
                            if e["source"] == current and not e.get("pruned"))
        if not found:
            raise ValueError(f"analysis node {node['id']} has no forecast path")


def _activate(kind: str, value: float) -> float:
    if kind == "tanh":
        return math.tanh(value)
    if kind == "sigmoid":
        # Signed gate: unavailable/neutral input (0) must remain neutral (0),
        # not become a bullish +0.5 vote.  Negative gate evidence is retained.
        return 2.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value)))) - 1.0
    if kind == "leaky_relu":
        return value if value >= 0 else 0.05 * value
    return value


def event_bases(events: list[SignalEvent], symbol: str, regime: str) -> dict[str, float]:
    bases: dict[str, list[float]] = {}
    for e in events:
        if e.symbol != symbol:
            continue
        bases.setdefault(e.node_id, []).append(signed_alpha(e))
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
    lifecycle = {"challenger": "validation_candidate", "champion": "champion",
                 "retired": "retired"}.get(status, "validation_candidate")
    store.db.execute(
        "INSERT INTO graph_versions(id,created_at,data_as_of,status,parent_id,"
        "topology,metrics,checkpoint,lifecycle_state,temporal_model_id) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)", (
        vid, datetime.now().astimezone().isoformat(timespec="seconds"), data_as_of,
        status, parent_id, json.dumps(topology), json.dumps(metrics or {}), checkpoint,
        lifecycle, (metrics or {}).get("temporal_model_id")))
    store.db.commit()
    store.audit("graph_version_saved", {"id": vid, "status": status,
                                         "lifecycle_state": lifecycle,
                                         "parent": parent_id, "metrics": metrics or {}})
    return vid


def promote(store, version_id: str, *, reason: str = "promotion gates passed") -> None:
    """Atomic champion swap: activation is validated first, the previous
    champion retires in the same transaction that activates the successor."""
    from .ml import lifecycle as ml_lifecycle
    row = store.db.execute("SELECT * FROM graph_versions WHERE id=?",
                           (version_id,)).fetchone()
    if not row:
        raise ValueError("unknown graph version")
    json.loads(row["topology"])                # activation check before the swap
    prior = store.db.execute(
        "SELECT id FROM graph_versions WHERE lifecycle_state='champion' AND id<>?",
        (version_id,)).fetchall()
    with store.db:
        for old in prior:            # retire first: the unique champion index
            ml_lifecycle.transition(store, "graph_versions", old["id"], "retired",
                                    reason=f"superseded by {version_id}", in_tx=True)
        ml_lifecycle.transition(store, "graph_versions", version_id, "champion",
                                reason=reason, in_tx=True)
    store.audit("graph_champion_promoted", {"id": version_id, "reason": reason,
                                            "retired": [o["id"] for o in prior]})


def mutate(topology: dict, seed: int = 0) -> dict:
    """One bounded topology mutation; invalid proposals fall back unchanged."""
    rng, out = random.Random(seed), copy.deepcopy(topology)
    nodes, edges = out["nodes"], out["edges"]
    action = rng.choice(("remove", "add", "move", "activation"))
    if action == "remove" and edges:
        # Structure search may remove redundancy, never an analysis type's
        # last route to the forecast. Only the human node toggle owns complete
        # elimination.
        specialists = {node["id"] for node in nodes
                       if node["role"] in ("alpha", "gate")}
        outputs = {node["id"] for node in nodes if node["role"] == "output"}
        def reaches_output(source, candidate_edges):
            seen, frontier = set(), [source]
            while frontier:
                current = frontier.pop()
                if current in outputs:
                    return True
                if current in seen:
                    continue
                seen.add(current)
                frontier.extend(e["target"] for e in candidate_edges
                                if e["source"] == current)
            return False
        removable = []
        for index in range(len(edges)):
            candidate = edges[:index] + edges[index + 1:]
            if all(reaches_output(source, candidate) for source in specialists):
                removable.append(index)
        if removable:
            edges.pop(rng.choice(removable))
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
    sample_dates = [b.get("__date") for b in bases]
    unique_dates = sorted(set(sample_dates)) if all(sample_dates) else []
    if len(unique_dates) >= 20:
        val_date = unique_dates[int(len(unique_dates) * .70)]
        test_date = unique_dates[int(len(unique_dates) * .85)]
        split1 = next(i for i, d in enumerate(sample_dates) if d >= val_date)
        split2 = next(i for i, d in enumerate(sample_dates) if d >= test_date)
    else:
        split1, split2 = int(len(X) * .70), int(len(X) * .85)

    def forward(inp):
        acts = {}
        for idx, n in enumerate(nodes):
            z = scales[idx] * inp[:, idx] + biases[idx]
            for ei, e in enumerate(out["edges"]):
                if e["target"] == n["id"] and not e.get("pruned"):
                    z = z + edge_w[ei] * acts[e["source"]]
            if n["activation"] == "tanh": z = torch.tanh(z)
            elif n["activation"] == "sigmoid": z = 2.0 * torch.sigmoid(z) - 1.0
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
    original_weights = [float(e.get("weight", 0.0)) for e in topology["edges"]]
    for i, e in enumerate(out["edges"]): e["weight"] = round(float(edge_w[i].detach()), 6)
    for i, n in enumerate(nodes):
        n["base_scale"] = round(float(scales[i].detach()), 6)
        n["bias"] = round(float(biases[i].detach()), 6)
        if n["role"] in ("alpha", "gate") and abs(n["base_scale"]) < \
                MIN_SPECIALIST_BASE_SCALE:
            original_node = next(item for item in topology["nodes"]
                                 if item["id"] == n["id"])
            original = float(original_node.get("base_scale", 1.0))
            n["base_scale"] = round(math.copysign(
                MIN_SPECIALIST_BASE_SCALE, n["base_scale"] or original or 1.0), 6)
            n["deemphasized"] = True
    # Contribution-normalized soft pruning.  Interaction redundancy may be
    # pruned, but an enabled specialist is never allowed to lose its last path
    # through the model.  That final edge retains a small signed floor and is
    # labelled `deemphasized`; only the operator's node toggle can eliminate an
    # analysis type.
    for target in {e["target"] for e in out["edges"]}:
        group = [e for e in out["edges"] if e["target"] == target]
        denom = sum(abs(e["weight"]) for e in group) or 1.0
        for e in group:
            e["pruned"] = abs(e["weight"]) / denom < prune_pct
            e["deemphasized"] = bool(e["pruned"])
    node_by_id = {n["id"]: n for n in nodes}
    for source in {e["source"] for e in out["edges"]}:
        if node_by_id[source]["role"] not in ("alpha", "gate"):
            continue
        outgoing = [e for e in out["edges"] if e["source"] == source]
        if any(not e.get("pruned") for e in outgoing):
            continue
        keep = max(outgoing, key=lambda e: abs(e["weight"]))
        idx = out["edges"].index(keep)
        sign_source = keep["weight"] if keep["weight"] else original_weights[idx]
        keep["weight"] = round(math.copysign(
            max(abs(keep["weight"]), MIN_SPECIALIST_EDGE_WEIGHT),
            sign_source or 1.0), 6)
        keep["pruned"] = False
        keep["deemphasized"] = True
    outputs = {n["id"] for n in nodes if n["role"] == "output"}
    def active_path(source):
        frontier, seen = [source], set()
        while frontier:
            current = frontier.pop()
            if current in outputs:
                return True
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(e["target"] for e in out["edges"]
                            if e["source"] == current and not e.get("pruned"))
        return False
    def structural_path(source):
        frontier = [(source, [])]
        while frontier:
            current, path = frontier.pop()
            if current in outputs:
                return path
            for edge in out["edges"]:
                if edge["source"] == current and edge not in path:
                    frontier.append((edge["target"], path + [edge]))
        return []
    for node in (n for n in nodes if n["role"] in ("alpha", "gate")):
        if active_path(node["id"]):
            continue
        for edge in structural_path(node["id"]):
            idx = out["edges"].index(edge)
            sign_source = edge["weight"] if edge["weight"] else original_weights[idx]
            edge["weight"] = round(math.copysign(
                max(abs(edge["weight"]), MIN_SPECIALIST_EDGE_WEIGHT),
                sign_source or 1.0), 6)
            edge["pruned"] = False
            edge["deemphasized"] = True
    def frozen_forward(inp):
        """Inference from the exact finalized topology saved/deployed live."""
        acts = {}
        for idx, n in enumerate(nodes):
            z = float(n["base_scale"]) * inp[:, idx] + float(n["bias"])
            for e in out["edges"]:
                if e["target"] == n["id"] and not e.get("pruned"):
                    z = z + float(e["weight"]) * acts[e["source"]]
            if n["activation"] == "tanh": z = torch.tanh(z)
            elif n["activation"] == "sigmoid": z = 2.0 * torch.sigmoid(z) - 1.0
            elif n["activation"] == "leaky_relu":
                z = torch.nn.functional.leaky_relu(z, .05)
            acts[n["id"]] = z
        return torch.stack((acts["output_5d"], acts["output_21d"]), dim=1)
    with torch.no_grad():
        validation_tensor = frozen_forward(X[split1:split2])
        validation_pred = validation_tensor.numpy()
        test = frozen_forward(X[split2:]).numpy()
        finalized_validation_loss = float(torch.nn.functional.huber_loss(
            validation_tensor, Y[split1:split2], delta=.05))
    truth = Y[split2:].numpy()
    validation_truth = Y[split1:split2].numpy()
    corrs = [_safe_corr(test[:, i], truth[:, i]) for i in range(2)]
    coverage = {}
    for i, h in enumerate((5, 21)):
        residual = validation_truth[:, i] - validation_pred[:, i]
        lo, hi = np.quantile(residual, [.10, .90]) if len(residual) else (0, 0)
        coverage[f"coverage_{h}d"] = round(float(np.mean(
            (truth[:, i] >= test[:, i] + lo) &
            (truth[:, i] <= test[:, i] + hi))), 3) if len(residual) else 0.0
    # Exact parity guard between the vectorized evaluator used for metrics and
    # the scalar evaluator used in live graph activation.
    parity = 0.0
    for row_index in range(min(16, len(bases))):
        live = evaluate(out, bases[row_index])["outputs"]
        frozen = frozen_forward(X[row_index:row_index + 1]).detach().numpy()[0]
        parity = max(parity, abs(live[5] - float(frozen[0])),
                     abs(live[21] - float(frozen[1])))
    metrics = {"validation_huber": round(finalized_validation_loss, 6),
               "test_ic_5d": round(corrs[0], 4), "test_ic_21d": round(corrs[1], 4),
               "n_train": split1, "n_validation": split2 - split1,
               "n_test": len(X) - split2,
               "pruned_edges": sum(bool(e.get("pruned")) for e in out["edges"]),
               "live_evaluator_parity_max_abs": round(parity, 9),
               **coverage}
    validate(out)
    return out, metrics


def _safe_corr(a, b) -> float:
    import numpy as np
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 5 and np.std(a) and np.std(b) else 0.0


def _daily_rank_ic(pred, truth, dates) -> float:
    """Mean daily cross-sectional Spearman IC; never pool dates together."""
    import numpy as np
    values = []
    dates = np.asarray(dates)
    for day in np.unique(dates):
        mask = dates == day
        if mask.sum() < 8:
            continue
        pr = np.argsort(np.argsort(np.asarray(pred)[mask]))
        tr = np.argsort(np.argsort(np.asarray(truth)[mask]))
        values.append(_safe_corr(pr, tr))
    return float(np.mean(values)) if values else 0.0


# R6: the staggered-cohort policy metric moved to ml/portfolio_metrics so the
# graph, the TCN bakeoff and the promotion gate all score on ONE definition.
# These aliases keep the existing call sites (and their tests) pointed at it.
from .ml.portfolio_metrics import (            # noqa: E402
    MIN_COHORTS_PER_OFFSET as _MIN_COHORTS_PER_OFFSET,
    MIN_VALID_OFFSETS as _MIN_VALID_OFFSETS,
    cohort_returns as _cohort_returns,
    offset_metrics as _offset_metrics,
    staggered_portfolio_metrics as _staggered_portfolio_metrics,
)


def walk_forward_fit(topology: dict, bases: list[dict[str, float]], targets,
                     folds: int = 5, prune_pct: float = .01) -> tuple[dict, dict]:
    """Expanding-window topology fit with a real 21-session embargo."""
    import numpy as np
    dates = np.asarray([b.get("__date") for b in bases])
    unique_dates = sorted(set(dates)) if len(dates) and all(dates) else []
    if len(bases) < 180 or len(unique_dates) < 60:
        learned, metrics = fit_weights(topology, bases, targets, prune_pct=prune_pct)
        metrics["walk_forward_folds"] = 0
        metrics["walk_forward_reason"] = (
            "need at least 180 samples across 60 dated sessions")
        return learned, metrics
    embargo = 21
    initial = max(embargo + 20, int(len(unique_dates) * .45))
    width = max(1, (len(unique_dates) - initial) // folds)
    fold_metrics = []
    oos_pred21, oos_truth21, oos_dates = [], [], []   # pooled OOS for the portfolio metric
    for index in range(folds):
        test_start_pos = initial + index * width
        test_end_pos = len(unique_dates) if index == folds - 1 else min(
            len(unique_dates), test_start_pos + width)
        train_last = unique_dates[test_start_pos - embargo - 1]
        test_first, test_last = (unique_dates[test_start_pos],
                                 unique_dates[test_end_pos - 1])
        train_idx = np.flatnonzero(dates <= train_last)
        test_idx = np.flatnonzero((dates >= test_first) & (dates <= test_last))
        train_bases = [bases[i] for i in train_idx]
        train_targets = [targets[i] for i in train_idx]
        learned, _ = fit_weights(topology, train_bases, train_targets,
                                 max_epochs=30, prune_pct=prune_pct)
        pred = np.asarray([[evaluate(learned, b)["outputs"][5],
                            evaluate(learned, b)["outputs"][21]]
                           for b in (bases[i] for i in test_idx)])
        truth = np.asarray([targets[i] for i in test_idx])
        fold_dates = dates[test_idx]
        extra = {}
        for i, h in enumerate((5, 21)):
            cutoff = np.quantile(pred[:, i], .9)
            extra[f"net_alpha_{h}d"] = round(
                float(truth[pred[:, i] >= cutoff, i].mean() - .0016), 5)
        oos_pred21.append(pred[:, 1]); oos_truth21.append(truth[:, 1])
        oos_dates.append(fold_dates)
        fold_metrics.append({"fold": index + 1, "train": len(train_idx),
                             "train_end": train_last, "test_start": test_first,
                             "embargo": embargo, "embargo_unit": "sessions",
                             "test": len(test_idx),
                             "ic_5d": round(_daily_rank_ic(
                                 pred[:, 0], truth[:, 0], fold_dates), 4),
                             "ic_21d": round(_daily_rank_ic(
                                 pred[:, 1], truth[:, 1], fold_dates), 4),
                             **extra})
    learned, metrics = fit_weights(topology, bases, targets, prune_pct=prune_pct)
    # Portfolio utility is computed ONCE over the pooled out-of-sample span with
    # all staggered non-overlapping cohort alignments — per-fold test spans are
    # far too short to hold enough independent 21-session cohorts. Fails closed.
    port = _staggered_portfolio_metrics(
        np.concatenate(oos_pred21), np.concatenate(oos_truth21),
        np.concatenate(oos_dates), horizon=21, cost=.0016) if oos_dates else {}
    metrics.update(walk_forward_folds=folds, folds=fold_metrics,
                   median_fold_ic_5d=round(float(np.median(
                       [f["ic_5d"] for f in fold_metrics])), 4),
                   median_fold_ic_21d=round(float(np.median(
                       [f["ic_21d"] for f in fold_metrics])), 4), **port)
    return learned, metrics


def blend_candidates(candidates, events, regime: str, cfg, store,
                     cycle_id: str | None = None,
                     node_states: dict[str, str] | None = None,
                     symbol_states: dict[str, dict[str, str]] | None = None,
                     universe: list[str] | None = None) -> None:
    """Apply the validated graph as a capped score blend, in place."""
    champ = champion(store)
    blend = activation_state(cfg, store)["effective_blend"]
    if champ["status"] != "champion":
        blend = 0.0
    snapshots = {}
    node_states, symbol_states = node_states or {}, symbol_states or {}
    enabled = {node_id for node_id, node_cfg in
               (cfg.get("nodes", default={}) or {}).items()
               if node_cfg.get("enabled")}
    scan = (universe if universe is not None
            else cfg.get("universe", "symbols", default=[]))
    by_symbol = sorted({e.symbol for e in events} | {c.symbol for c in candidates} |
                       set(scan))
    for symbol in by_symbol:
        bases = event_bases(events, symbol, regime)
        from .strategy import contribution as strategy_contribution
        strategy = strategy_contribution(cfg, store, symbol)
        bases["strategy_context"] = float(strategy.get("value", 0.0))
        snapshots[symbol] = evaluate(
            champ["topology"], bases)
        snapshots[symbol]["node_states"] = {}
        for n in champ["topology"]["nodes"]:
            node_id = n["id"]
            if n["role"] not in ("alpha", "gate"):
                state = "running"
            elif node_id == "strategy_context":
                state = ("running" if strategy.get("state") == "running"
                         else "verified_neutral")
            elif node_id == "macro_regime":
                state = "running"
            elif node_id not in enabled:
                state = "human_disabled"
            else:
                state = (symbol_states.get(node_id, {}).get(symbol) or
                         node_states.get(node_id, "unavailable"))
            snapshots[symbol]["node_states"][node_id] = state
        snapshots[symbol]["activation_complete"] = not any(
            state in ("unavailable", "blocked")
            for state in snapshots[symbol]["node_states"].values())
    for c in candidates:
        result = snapshots[c.symbol]
        graph_score = math.tanh(float(result["outputs"][21]))
        if blend and result["activation_complete"]:
            prior = c.final_score
            c.final_score = round((1 - blend) * prior + blend * graph_score, 4)
            c.learned_contribution = round(c.final_score - prior, 6)
            c.thesis = (c.thesis + f"; analog_graph:{graph_score:+.3f}")[:400]
            c.contributing_nodes = sorted(set(c.contributing_nodes + ["analog_graph"]))
        snapshots[c.symbol] = {"graph_score": graph_score, **result}
    as_of = max((e.data_as_of.strftime("%Y-%m-%d") for e in events), default=None)
    store.kv_set("graph_last_activations", {
        "as_of": as_of, "cycle_id": cycle_id, "graph_version": champ["id"],
        "topology_schema": champ["topology"].get("schema"), "symbols": snapshots})
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
                shadow_bases = event_bases(events, symbol, regime)
                shadow_bases["strategy_context"] = float(
                    strategy_contribution(cfg, store, symbol).get("value", 0.0))
                result = evaluate(topology, shadow_bases)
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


def _graph_offline_gate(cfg, metrics: dict, current_metrics: dict) -> bool:
    """Offline eligibility for ONE graph version against persisted metrics."""
    folds = metrics.get("folds") or []
    fold_gate = len(folds) >= 5 and \
        sum(f.get("ic_5d", 0) > 0 for f in folds) >= 4 and \
        sum(f.get("ic_21d", 0) > 0 for f in folds) >= 4 and \
        min((f.get("ic_5d", -1) for f in folds), default=-1) > -.01 and \
        min((f.get("ic_21d", -1) for f in folds), default=-1) > -.01 and \
        metrics.get("median_fold_ic_5d", 0) >= .015 and \
        metrics.get("median_fold_ic_21d", 0) >= .02
    horizon_gate = all(
        .75 <= metrics.get(f"coverage_{h}d", 0) <= .85 and
        sum(f.get(f"net_alpha_{h}d", -1) > 0 for f in folds) >= 4
        for h in (5, 21))
    coverage = metrics.get("sample_coverage") or {}
    required = [n["id"] for n in default_topology()["nodes"]
                if n["role"] in ("alpha", "gate") and
                (n["id"] == "macro_regime" or
                 (n["id"] != "strategy_context" and
                  cfg.get("nodes", n["id"], "enabled", default=False)))]
    coverage_passed = all(coverage.get(node, 0) >= (.9 if node == "macro_regime" else .01)
                          for node in required)
    utility_passed = metrics.get("portfolio_utility", -1) > 0 and \
        metrics.get("portfolio_utility", -1) >= current_metrics.get("portfolio_utility", -1) and \
        metrics.get("oos_sharpe", -99) >= current_metrics.get("oos_sharpe", -99)
    return fold_gate and horizon_gate and coverage_passed and utility_passed


def maybe_promote(cfg, store) -> dict:
    """Lifecycle promotion over ALL eligible graph finalists (Sprint D).

    A graph can only activate against the exact TCN it was trained with, with
    complete metrics and sufficient (fail-closed) cohort utility evidence.
    Offline validity earns experimental_live — recorded, still zero live blend
    (activation_state keys on champion); the champion swap requires forward
    shadow evidence. The old promote_stage1 direct-championship is gone.
    """
    from .ml import lifecycle as ml_lifecycle
    from .neural import active_global_run, shadow_metrics
    rows = ml_lifecycle.finalists(
        store, "graph_versions",
        states=("validation_candidate",) + ml_lifecycle.FINALIST_STATES)
    if not rows:
        return {"action": "none"}
    active_tcn = active_global_run(store)
    current_row = store.db.execute(
        "SELECT metrics FROM graph_versions WHERE lifecycle_state='champion' "
        "ORDER BY created_at DESC LIMIT 1").fetchone()
    current_metrics = json.loads(current_row["metrics"] or "{}") if current_row else {}
    eligible = []
    for r in rows:
        metrics = json.loads(r["metrics"] or "{}")
        if not active_tcn or metrics.get("temporal_model_id") != active_tcn["id"]:
            continue                    # TCN dependency mismatch blocks activation
        if metrics.get("utility_evidence") != "ok":
            continue                    # insufficient cohort evidence fails closed
        if not _graph_offline_gate(cfg, metrics, current_metrics):
            continue
        eligible.append(r)
    if not eligible:
        return {"action": "shadow", "candidates": [r["id"] for r in rows]}
    eligible.sort(key=lambda r: ml_lifecycle.rank_key(r, primary="portfolio_utility"))
    top = eligible[0]
    if top["lifecycle_state"] == "validation_candidate":
        ml_lifecycle.transition(
            store, "graph_versions", top["id"], "experimental_live",
            reason="offline gates passed against active TCN "
                   f"{active_tcn['id']}; awaiting forward shadow evidence",
            evidence={"rank": 1, "eligible": len(eligible)})
        store.kv_set("graph_offline_gate", {"passed": True, "id": top["id"],
                                             "at": datetime.now().astimezone().isoformat()})
        return {"action": "experimental_live", "id": top["id"]}
    sm = shadow_metrics(store, top["id"])
    hs = sm["horizons"]
    forward = sm["sessions"] >= 30 and all(
        hs.get(str(h), {}).get("n", 0) >= 10_000 and
        hs[str(h)].get("ic", 0) >= .01 and hs[str(h)].get("top_decile_alpha", 0) > 0
        for h in (5, 21))
    if forward:
        promote(store, top["id"], reason="forward shadow gates passed")
        return {"action": "promote", "id": top["id"], "metrics": sm}
    return {"action": "shadow", "id": top["id"], "metrics": sm}


def activation_state(cfg, store, refresh_checkpoints: bool = True) -> dict:
    """Single source of truth for the 0→10→25→50 live ramp."""
    from .neural import refresh_compatibility, shadow_metrics
    if refresh_checkpoints:
        refresh_compatibility(store)
    graph_row = store.db.execute(
        "SELECT * FROM graph_versions WHERE status='champion' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    tcn = store.db.execute(
        "SELECT * FROM model_runs WHERE kind='global_tcn' AND status='champion' "
        "ORDER BY created_at DESC LIMIT 1").fetchone()
    if not graph_row or not tcn:
        missing = "GRAPH" if not graph_row else "GLOBAL TCN"
        return {"stage": 0, "effective_blend": 0.0, "ready": False,
                "block_reason": f"BLOCKED: {missing}",
                "graph_id": graph_row["id"] if graph_row else None,
                "global_tcn_id": tcn["id"] if tcn else None}
    graph_metrics = json.loads(graph_row["metrics"] or "{}")
    trained_with = graph_metrics.get("temporal_model_id")
    if trained_with != tcn["id"]:
        return {"stage": 0, "effective_blend": 0.0, "ready": False,
                "block_reason": "BLOCKED: GRAPH/TCN checkpoint mismatch",
                "graph_id": graph_row["id"], "global_tcn_id": tcn["id"],
                "graph_temporal_model_id": trained_with}
    try:
        validate(json.loads(graph_row["topology"]))
    except Exception as exc:
        return {"stage": 0, "effective_blend": 0.0, "ready": False,
                "block_reason": f"BLOCKED: GRAPH integrity ({type(exc).__name__})",
                "graph_id": graph_row["id"], "global_tcn_id": tcn["id"]}
    age = (datetime.now().date() - datetime.fromisoformat(tcn["created_at"]).date()).days
    if age > int(cfg.get("neural", "max_checkpoint_age_days", default=7)):
        return {"stage": 0, "effective_blend": 0.0, "ready": False,
                "block_reason": f"BLOCKED: GLOBAL TCN stale ({age}d)",
                "graph_id": graph_row["id"], "global_tcn_id": tcn["id"]}
    stage = max(1, int(store.kv_get("model_activation_stage", 1) or 1))
    gm, tm = shadow_metrics(store, graph_row["id"]), shadow_metrics(store, tcn["id"])
    metrics = [gm, tm]
    if stage < 2 and all(m["sessions"] >= 21 and all(
            m["horizons"].get(str(h), {}).get("n", 0) >= 5000 and
            m["horizons"][str(h)].get("ic", 0) >= .01 and
            m["horizons"][str(h)].get("top_decile_alpha", 0) > 0
            for h in (5, 21)) for m in metrics):
        stage = 2
    if stage < 3 and all(m["sessions"] >= 30 and all(
            m["horizons"].get(str(h), {}).get("n", 0) >= 10000 and
            .75 <= m["horizons"][str(h)].get("coverage", 0) <= .85 and
            m["horizons"][str(h)].get("top_decile_alpha", 0) > 0
            for h in (5, 21)) for m in metrics):
        stage = 3
    # Integrity/calibration decay drops back one stage, never below the
    # offline-approved 10% stage while both champions remain valid.
    if stage > 1 and any(any(
            m["horizons"].get(str(h), {}).get("n", 0) >= 1000 and
            (m["horizons"][str(h)].get("ic", 0) <= 0 or
             not .70 <= m["horizons"][str(h)].get("coverage", 0) <= .90)
            for h in (5, 21)) for m in metrics):
        stage -= 1
    store.kv_set("model_activation_stage", stage)
    return {"stage": stage, "effective_blend": {1: .10, 2: .25, 3: .50}[stage],
            "ready": True, "block_reason": None, "graph_id": graph_row["id"],
            "global_tcn_id": tcn["id"], "graph_shadow": gm, "tcn_shadow": tm}
