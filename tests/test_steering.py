"""V4/D34 steering: tiered expiry defaults, apply paths, validation, and the
non-blocking guarantee (scans run fine with pending requests)."""
from __future__ import annotations

import json

import pytest

from specforge import hypothesis as hypo
from specforge import steering
from specforge.config import ConfigError
from specforge.models import new_id

FUTURE = "2099-01-01T00:00:00-07:00"
PAST = "2020-01-01T00:00:00-07:00"


def _hypothesis(store, tier="short_term", **kw):
    h = {"id": new_id(), "tier": tier, "status": "proposed",
         "created_at": "2026-07-01T10:00:00-07:00", "thesis": "T",
         "stances": [{"symbol": "AAA", "direction": "long", "conviction": 0.7,
                      "horizon_days": 10, "rationale": "r"}],
         "watchlist": [], "invalidation": "x", "regime": "neutral",
         "source": "ai", "parent_id": "", **kw}
    store.save_hypothesis(h)
    return h


def _request(cfg, store, kind, payload, expired=False, **kw):
    s = steering.create(cfg, store, kind, title="t", context="c",
                        options=[{"key": "adopt", "label": "a", "detail": ""},
                                 {"key": "keep", "label": "k", "detail": ""}],
                        recommended="adopt", payload=payload, **kw)
    if expired:
        store.update_steering(s["id"], expires_at=PAST)
        s = store.get_steering(s["id"])
    return s


def test_expiry_tiers_adopt_vs_status_quo(cfg, store):
    h1, h2 = _hypothesis(store), _hypothesis(store, tier="north_star")
    # active north star exists → change requests are stable-tier
    ns = _hypothesis(store, tier="north_star")
    hypo.activate(cfg, store, ns["id"])

    r_adopt = _request(cfg, store, "hypothesis_adopt",
                       {"hypothesis_id": h1["id"]}, expired=True)
    r_keep = _request(cfg, store, "north_star_change",
                      {"hypothesis_id": h2["id"]}, expired=True)
    steering.sweep(cfg, store)

    # agile tier: expired → auto-adopted, hypothesis went active
    assert store.get_steering(r_adopt["id"])["status"] == "decided"
    assert store.get_steering(r_adopt["id"])["decided_via"] == "expiry"
    assert store.active_hypothesis("short_term")["id"] == h1["id"]
    # stable tier: expired → status quo, north star unchanged
    assert store.get_steering(r_keep["id"])["status"] == "expired"
    assert store.active_hypothesis("north_star")["id"] == ns["id"]


def test_gui_decide_applies_and_double_decide_rejected(cfg, store):
    h = _hypothesis(store)
    r = _request(cfg, store, "hypothesis_adopt", {"hypothesis_id": h["id"]})
    steering.decide(cfg, store, r["id"], "adopt", via="gui")
    assert store.active_hypothesis("short_term")["id"] == h["id"]
    with pytest.raises(ValueError, match="already decided"):
        steering.decide(cfg, store, r["id"], "keep")
    with pytest.raises(ValueError, match="invalid option"):
        steering.decide(cfg, store, _request(cfg, store, "hypothesis_adopt",
                                             {"hypothesis_id": h["id"]})["id"], "zzz")


def test_keep_archives_the_proposal(cfg, store):
    h = _hypothesis(store)
    r = _request(cfg, store, "hypothesis_adopt", {"hypothesis_id": h["id"]})
    steering.decide(cfg, store, r["id"], "keep")
    assert store.get_hypothesis(h["id"])["status"] == "retired"
    assert store.active_hypothesis("short_term") is None
    assert list((hypo.hypo_dir(cfg) / "archive").glob("*short_term*"))


def test_risk_suggestion_cannot_pass_dangerous_values(cfg, store):
    r = _request(cfg, store, "risk_suggestion",
                 {"path": ["risk", "max_daily_loss"], "value": 0.50})
    with pytest.raises(ConfigError):
        steering.decide(cfg, store, r["id"], "adopt")
    # a sane suggestion applies through the same path
    r2 = _request(cfg, store, "risk_suggestion",
                  {"path": ["risk", "max_daily_loss"], "value": 0.03})
    steering.decide(cfg, store, r2["id"], "adopt")
    assert (store.kv_get("config_overrides")["risk"]["max_daily_loss"]) == 0.03


def test_node_promotion_and_watchlist_add(cfg, store):
    r = _request(cfg, store, "node_promotion",
                 {"node_id": "momentum", "to_status": "probation"})
    steering.decide(cfg, store, r["id"], "adopt")
    assert store.kv_get("config_overrides")["nodes"]["momentum"]["status"] == "probation"

    h = _hypothesis(store, watchlist=["GE"])
    hypo.activate(cfg, store, h["id"])
    r2 = _request(cfg, store, "watchlist_add", {"symbols": ["CAT", "GE"]})
    steering.decide(cfg, store, r2["id"], "adopt")
    wl = json.loads(store.active_hypothesis("short_term")["watchlist"])
    assert wl == ["GE", "CAT"]                      # dedup, appended, capped


def test_scan_never_blocks_on_pending_steering(cfg, store):
    from specforge.engine import run_cycle
    cfg.data["hypothesis"]["enabled"] = True
    for i in range(10):
        _request(cfg, store, "risk_suggestion",
                 {"path": ["risk", "max_daily_loss"], "value": 0.03})
    summary = run_cycle(cfg, store, refresh_data=False)
    assert summary["signals"] > 0                   # traded normally
    assert len(store.steering_requests(status="pending")) == 10


def test_bootstrap_north_star_is_adopt_tier(cfg, store):
    class StubAI:
        def complete_json(self, *a, **k):
            return {"thesis": "durable", "summary": "s", "stances": [],
                    "watchlist": [], "invalidation": ""}
    from specforge.data import MarketContext
    cfg.data["hypothesis"]["enabled"] = True
    ctx = MarketContext(store, cfg, offline=True)
    r = steering.ensure_north_star(cfg, store, StubAI(), ctx)
    assert r["default_on_expiry"] == "adopt"        # bootstrap exception
    store.update_steering(r["id"], expires_at=PAST)
    steering.sweep(cfg, store)
    assert store.active_hypothesis("north_star") is not None
    # with one active, the next ensure is a no-op and changes are stable-tier
    assert steering.ensure_north_star(cfg, store, StubAI(), ctx) is None
