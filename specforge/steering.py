"""Steering (V4/D34) — non-blocking strategic choices, Claude-plan-mode style.

A steering request is a strategic decision surfaced to the human with 2-4
options and an AI recommendation. Trading NEVER waits on one: scans keep
running under the current active hypothesis/config while requests pend, and
every request expires into a per-kind default (user-ratified tiers):

  adopt on expiry (agile tier):   hypothesis_adopt, watchlist_add,
                                  bootstrap north_star (no status quo exists)
  status quo on expiry (stable):  north_star_change, node_promotion,
                                  risk_suggestion

Apply paths are the ONLY trading influence: hypothesis activation (feeds the
hypothesis node) and config overrides through config.apply_override — the same
validated path the GUI uses, so Config.validate() still rejects dangerous
values regardless of who (human click or expiry default) triggers the apply.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from . import hypothesis as hypo_mod
from .config import apply_override
from .models import new_id
from .store import Store

# expiry-default tier per kind (design decision, dev/V4_PLAN.md)
EXPIRY_DEFAULTS = {
    "hypothesis_adopt": "adopt",
    "watchlist_add": "adopt",
    "north_star_change": "status_quo",
    "node_promotion": "status_quo",
    "risk_suggestion": "status_quo",
}
VALID_STATUSES = ("experimental", "probation", "production", "disabled")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def create(cfg, store: Store, kind: str, title: str, context: str,
           options: list[dict], recommended: str, payload: dict,
           default_on_expiry: str | None = None) -> dict:
    if kind not in EXPIRY_DEFAULTS:
        raise ValueError(f"unknown steering kind {kind}")
    ttl = cfg.get("hypothesis", "steering_ttl_hours", default=24)
    s = {
        "id": new_id(), "kind": kind, "created_at": _now(),
        "expires_at": (datetime.now().astimezone() + timedelta(hours=ttl)
                       ).isoformat(timespec="seconds"),
        "title": title, "context": context, "options": options,
        "recommended": recommended,
        "default_on_expiry": default_on_expiry or EXPIRY_DEFAULTS[kind],
        "status": "pending", "payload": payload,
    }
    store.save_steering(s)
    store.audit("steering_created", {"id": s["id"], "kind": kind, "title": title,
                                     "expires_at": s["expires_at"],
                                     "default_on_expiry": s["default_on_expiry"]})
    return s


def decide(cfg, store: Store, sid: str, key: str, via: str = "gui") -> dict:
    """Human (or expiry default) picks an option; applies it. Raises ValueError
    on unknown/late requests, ConfigError if a risk apply fails validation."""
    req = store.get_steering(sid)
    if not req:
        raise ValueError(f"unknown steering request {sid}")
    if req["status"] != "pending":
        raise ValueError(f"steering request {sid} already {req['status']}")
    keys = {o["key"] for o in json.loads(req["options"] or "[]")}
    if key not in keys:
        raise ValueError(f"invalid option {key!r}; choose one of {sorted(keys)}")
    _apply(cfg, store, req, key)
    store.update_steering(sid, status="decided", decided_key=key,
                          decided_at=_now(), decided_via=via)
    store.audit("steering_decided", {"id": sid, "kind": req["kind"],
                                     "key": key, "via": via})
    return store.get_steering(sid)


def sweep(cfg, store: Store, now_iso: str | None = None) -> list[dict]:
    """Expire past-due pending requests into their per-kind default. Called on
    every GUI read and at cycle start — cheap, deterministic, non-blocking."""
    now = now_iso or _now()
    out = []
    for req in store.steering_requests(status="pending"):
        if req["expires_at"] >= now:
            continue
        if req["default_on_expiry"] == "adopt":
            try:
                _apply(cfg, store, req, req["recommended"])
                store.update_steering(req["id"], status="decided",
                                      decided_key=req["recommended"],
                                      decided_at=now, decided_via="expiry")
            except Exception as e:              # noqa: BLE001 — an apply failure
                # must never wedge the sweep; fall back to status quo, loudly
                store.update_steering(req["id"], status="expired",
                                      decided_at=now, decided_via="expiry")
                store.audit("steering_apply_failed", {"id": req["id"], "err": str(e)})
        else:
            store.update_steering(req["id"], status="expired",
                                  decided_at=now, decided_via="expiry")
        store.audit("steering_expired", {"id": req["id"], "kind": req["kind"],
                                         "default": req["default_on_expiry"]})
        out.append(store.get_steering(req["id"]))
    return out


def _apply(cfg, store: Store, req: dict, key: str) -> None:
    """Execute the chosen option. 'keep'/'reject' style keys are no-ops beyond
    retiring a proposed hypothesis (so it archives instead of dangling)."""
    kind, payload = req["kind"], json.loads(req["payload"] or "{}")
    if kind in ("hypothesis_adopt", "north_star_change"):
        hid = payload.get("hypothesis_id", "")
        if key == "adopt":
            hypo_mod.activate(cfg, store, hid)
        else:                                   # keep current → archive proposal
            h = store.get_hypothesis(hid)
            if h and h["status"] == "proposed":
                store.update_hypothesis(hid, status="retired", retired_at=_now())
                hypo_mod.archive_file(cfg, store.get_hypothesis(hid))
    elif kind == "node_promotion":
        if key == "adopt":
            node, to = payload["node_id"], payload["to_status"]
            if to not in VALID_STATUSES:
                raise ValueError(f"invalid status {to}")
            apply_override(store, cfg.mode, ["nodes", node, "status"], to,
                           via="steering")
    elif kind == "risk_suggestion":
        if key == "adopt":
            # same validated path as the GUI — dangerous values still rejected
            apply_override(store, cfg.mode, list(payload["path"]),
                           payload["value"], via="steering")
    elif kind == "watchlist_add":
        if key == "adopt":
            h = store.active_hypothesis("short_term")
            if h:
                cap = cfg.get("hypothesis", "max_watchlist", default=8)
                wl = json.loads(h["watchlist"] or "[]")
                wl += [s for s in payload.get("symbols", []) if s not in wl]
                store.update_hypothesis(h["id"], watchlist=json.dumps(wl[:cap]))
                hypo_mod.write_current_file(cfg, store.get_hypothesis(h["id"]))


# ---------------- proposal flows (generation → steering) ----------------

def propose_short_term(cfg, store: Store, ai, ctx) -> dict | None:
    """Generate a short-term hypothesis and queue it for steering (auto-adopt
    tier). Returns the steering request, or None if generation declined."""
    h = hypo_mod.generate(cfg, store, ai, ctx, tier="short_term")
    if h is None:
        return None
    stances = json.loads(h["stances"] or "[]")
    return create(
        cfg, store, "hypothesis_adopt",
        title=f"Adopt new short-term hypothesis ({len(stances)} stances)",
        context=h["thesis"] + "\n\n**Invalidation:** " + (h["invalidation"] or "—"),
        options=[{"key": "adopt", "label": "Adopt (recommended)",
                  "detail": "Retire the current short-term hypothesis and trade on this one."},
                 {"key": "keep", "label": "Keep current",
                  "detail": "Archive this proposal; the active hypothesis stays in force."}],
        recommended="adopt", payload={"hypothesis_id": h["id"]})


def ensure_north_star(cfg, store: Store, ai, ctx) -> dict | None:
    """Bootstrap: if no north star exists, propose one whose expiry default is
    ADOPT (the sole exception to the status-quo tier — there is no status quo
    to keep). Subsequent changes go through propose_north_star_change."""
    if store.active_hypothesis("north_star"):
        return None
    h = hypo_mod.generate(cfg, store, ai, ctx, tier="north_star")
    if h is None:
        return None
    return create(
        cfg, store, "north_star_change",
        title="Ratify the initial north-star hypothesis",
        context=h["thesis"],
        options=[{"key": "adopt", "label": "Adopt (recommended)",
                  "detail": "This becomes the persistent thesis all short-term hypotheses align to."},
                 {"key": "keep", "label": "Reject",
                  "detail": "Archive it; the system runs without a north star until the next proposal."}],
        recommended="adopt", payload={"hypothesis_id": h["id"]},
        default_on_expiry="adopt")             # bootstrap exception (D34)


def maintain(cfg, store: Store, ai=None, ctx=None) -> dict:
    """Post-close hypothesis upkeep (scheduler + `specforge hypothesis` CLI):
    sweep expiries, bootstrap the north star, rotate a stale short-term
    hypothesis, review the north star on cadence. Everything routes through
    steering; nothing here touches orders. Returns a small summary."""
    out = {"swept": len(sweep(cfg, store))}
    if not (cfg.get("hypothesis", "enabled", default=False)
            and cfg.get("ai", "enabled", default=False)):
        out["skipped"] = "hypothesis or ai disabled"
        return out
    if ctx is None:
        from .data import MarketContext
        ctx = MarketContext(store, cfg)
    if ai is None:
        from .ai import AIClient
        ai = AIClient(cfg, store)
    from . import regime as regime_mod
    pending = {s["kind"] for s in store.steering_requests(status="pending")}

    if store.active_hypothesis("north_star") is None:
        if "north_star_change" not in pending:
            r = ensure_north_star(cfg, store, ai, ctx)
            out["north_star_proposed"] = bool(r)
    else:
        days = cfg.get("hypothesis", "north_star_review_days", default=30)
        ns = store.active_hypothesis("north_star")
        age = (datetime.now().astimezone()
               - datetime.fromisoformat(ns["activated_at"])).days
        if age >= days and "north_star_change" not in pending:
            r = propose_north_star_change(cfg, store, ai, ctx)
            out["north_star_review"] = bool(r)

    reg = regime_mod.classify(ctx, cfg)
    reason = hypo_mod.short_term_stale(cfg, store, reg.regime)
    if reason and "hypothesis_adopt" not in pending:
        r = propose_short_term(cfg, store, ai, ctx)
        out["short_term_proposed"] = bool(r)
        out["reason"] = reason
        if r:
            store.audit("hypothesis_regen", {"reason": reason, "steering": r["id"]})
    return out


def propose_north_star_change(cfg, store: Store, ai, ctx) -> dict | None:
    """Periodic north-star review (stable tier: keep status quo on expiry)."""
    h = hypo_mod.generate(cfg, store, ai, ctx, tier="north_star")
    if h is None:
        return None
    return create(
        cfg, store, "north_star_change",
        title="North-star review: proposed revision",
        context=h["thesis"],
        options=[{"key": "adopt", "label": "Adopt revision",
                  "detail": "Retire the current north star and adopt this one."},
                 {"key": "keep", "label": "Keep current (recommended default)",
                  "detail": "Archive the proposal; nothing changes."}],
        recommended="adopt", payload={"hypothesis_id": h["id"]})
