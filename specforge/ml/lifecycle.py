"""Explicit model lifecycle (Sprint D).

One authoritative field — `lifecycle_state` — replaces the overloaded
champion/challenger status. The legacy `status` column survives only as a
compatibility projection so existing queries and APIs keep working.

    training → validation_candidate → validation_winner → sealed_candidate
             → experimental_live → production_candidate → champion
    terminal / side states: rejected · incompatible · retired

Every transition is persisted to `model_transitions` with the evidence used,
so promotion decisions are reconstructable from the database alone.
"""
from __future__ import annotations

import json
from datetime import datetime

STATES = ("training", "validation_candidate", "validation_winner",
          "sealed_candidate", "experimental_live", "production_candidate",
          "champion", "rejected", "incompatible", "retired")

# States eligible to compete for promotion (finalists). validation_* stays out:
# validation-only evidence can never reach live influence.
FINALIST_STATES = ("sealed_candidate", "experimental_live", "production_candidate")
# States allowed to serve live inference, in priority order.
SERVING_STATES = ("champion", "production_candidate", "experimental_live")

_STATUS_PROJECTION = {"champion": "champion", "incompatible": "incompatible",
                      "retired": "retired"}

# R1: legal transitions only. None = legacy/new rows (backfill classification).
# retired → champion exists solely for the champion-rollback restore path.
ALLOWED_TRANSITIONS = {
    None: {"training", "validation_candidate", "sealed_candidate",
           "incompatible", "retired", "champion"},
    "training": {"validation_candidate", "rejected", "incompatible", "retired"},
    "validation_candidate": {"validation_winner", "sealed_candidate",
                             "experimental_live", "rejected", "incompatible",
                             "retired"},
    "validation_winner": {"sealed_candidate", "rejected", "incompatible",
                          "retired"},
    "sealed_candidate": {"experimental_live", "rejected", "incompatible",
                         "retired"},
    "experimental_live": {"production_candidate", "champion", "rejected",
                          "incompatible", "retired"},
    "production_candidate": {"champion", "rejected", "incompatible", "retired"},
    "champion": {"retired", "incompatible"},
    "retired": {"champion"},
    "rejected": {"incompatible", "retired"},
    "incompatible": {"retired"},
}


class LifecycleError(ValueError):
    """Illegal transition or lost compare-and-swap race."""


def project_status(state: str) -> str:
    """Legacy `status` value for a lifecycle state (compatibility only)."""
    return _STATUS_PROJECTION.get(state, "challenger")


def transition(store, model_table: str, model_id: str, new_state: str, *,
               reason: str, evidence: dict | None = None,
               permitted_blend: float | None = None, parent_id: str | None = None,
               in_tx: bool = False) -> None:
    """Move one model to `new_state`, project legacy status, and persist the
    full decision record. With in_tx=True the caller owns the transaction
    (atomic multi-row swaps); otherwise this commits."""
    if new_state not in STATES:
        raise ValueError(f"unknown lifecycle state {new_state!r}")
    if model_table not in ("model_runs", "graph_versions"):
        raise ValueError(f"unknown model table {model_table!r}")
    row = store.db.execute(f"SELECT * FROM {model_table} WHERE id=?",
                           (model_id,)).fetchone()
    if not row:
        raise ValueError(f"unknown model {model_id!r} in {model_table}")
    prior = row["lifecycle_state"]
    if new_state not in ALLOWED_TRANSITIONS.get(prior, set()):
        raise LifecycleError(
            f"illegal transition {prior!r} → {new_state!r} for {model_id}")
    sets, args = ["lifecycle_state=?", "status=?"], [new_state, project_status(new_state)]
    if model_table == "model_runs" and permitted_blend is not None:
        sets.append("permitted_blend=?"); args.append(float(permitted_blend))
    # Compare-and-swap on the observed prior state: a concurrent transition
    # between our read and this write loses exactly one of the two races.
    cursor = store.db.execute(
        f"UPDATE {model_table} SET {', '.join(sets)} "
        f"WHERE id=? AND lifecycle_state IS ?",
        args + [model_id, prior])
    if cursor.rowcount != 1:
        raise LifecycleError(
            f"lost transition race on {model_id}: state changed from {prior!r}")
    keys = row.keys()
    store.db.execute(
        "INSERT INTO model_transitions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (f"tr-{model_id}-{datetime.now().timestamp():.6f}", model_id, model_table,
         prior, new_state, reason, json.dumps(evidence or {}),
         permitted_blend, parent_id or (row["parent_id"] if "parent_id" in keys else None),
         row["architecture_hash"] if "architecture_hash" in keys else None,
         row["feature_hash"] if "feature_hash" in keys else None,
         json.loads(row["metrics"] or "{}").get("target_schema_hash")
         if "metrics" in keys else None,
         datetime.now().astimezone().isoformat(timespec="seconds")))
    if not in_tx:
        store.db.commit()
        store.audit("model_lifecycle_transition", {
            "model": model_id, "table": model_table, "from": prior,
            "to": new_state, "reason": reason,
            "permitted_blend": permitted_blend})


def history(store, model_id: str) -> list[dict]:
    """Pure read: the persisted transition trail, oldest first."""
    return [dict(r) for r in store.db.execute(
        "SELECT * FROM model_transitions WHERE model_id=? ORDER BY at, id",
        (model_id,))]


def finalists(store, model_table: str, *, kind: str | None = None,
              states: tuple = FINALIST_STATES) -> list[dict]:
    """Pure read: every promotion-eligible row (compatible, checkpointed),
    unranked. Ranking is the caller's job via `rank_key` on persisted metrics."""
    q = (f"SELECT * FROM {model_table} WHERE lifecycle_state IN "
         f"({','.join('?' * len(states))})")
    args: list = list(states)
    if model_table == "model_runs":
        q += " AND incompatibility_reason IS NULL AND checkpoint IS NOT NULL"
        if kind:
            q += " AND kind=?"; args.append(kind)
    return [dict(r) for r in store.db.execute(q, args)]


def rank_key(row: dict, primary: str = "validation_selection_score"):
    """Deterministic finalist ordering: persisted metric desc, then creation
    time asc (older first — a newer weak model can never hide an older
    qualified one), then id asc as the final stable tie-breaker."""
    metrics = json.loads(row.get("metrics") or "{}")
    score = metrics.get(primary)
    return (-(float(score) if score is not None else float("-inf")),
            row.get("created_at") or "", row.get("id") or "")
