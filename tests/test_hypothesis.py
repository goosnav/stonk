"""V4/D34 hypothesis layer: lifecycle + archive, point-in-time reads, the
deterministic node, generation validation, and the engine watchlist merge.
All offline; a stub stands in for the AI client."""
from __future__ import annotations

import json

from specforge import hypothesis as hypo
from specforge.data import MarketContext
from specforge.models import new_id


class StubAI:
    def __init__(self, resp):
        self.resp = resp

    def complete_json(self, *a, **k):
        return self.resp


def _mk(store, tier="short_term", stances=None, wl=None, status="proposed"):
    h = {"id": new_id(), "tier": tier, "status": status,
         "created_at": "2026-07-01T10:00:00-07:00",
         "thesis": "test thesis", "stances": stances or [],
         "watchlist": wl or [], "invalidation": "x", "regime": "neutral",
         "source": "ai", "parent_id": ""}
    store.save_hypothesis(h)
    return h


def test_activate_retires_and_archives(cfg, store, tmp_path):
    h1 = _mk(store)
    hypo.activate(cfg, store, h1["id"], now_iso="2026-07-01T18:00:00-07:00")
    h2 = _mk(store)
    hypo.activate(cfg, store, h2["id"], now_iso="2026-07-03T18:00:00-07:00")

    assert store.active_hypothesis("short_term")["id"] == h2["id"]
    old = store.get_hypothesis(h1["id"])
    assert old["status"] == "retired" and old["retired_at"].startswith("2026-07-03")
    # current file + dated archive of the retired one (user-facing log)
    d = hypo.hypo_dir(cfg)
    assert (d / "short_term.md").exists()
    arch = list((d / "archive").glob("2026-07-03-short_term-*.md"))
    assert len(arch) == 1 and "test thesis" in arch[0].read_text()


def test_active_hypothesis_is_point_in_time(cfg, store):
    h = _mk(store)
    hypo.activate(cfg, store, h["id"], now_iso="2026-07-03T18:00:00-07:00")
    assert store.active_hypothesis("short_term", as_of="2026-07-02") is None
    assert store.active_hypothesis("short_term", as_of="2026-07-03")["id"] == h["id"]


def test_node_emits_stances_and_degrades_silently(cfg, store):
    from specforge.nodes.base import build_registry
    cfg.data["hypothesis"]["enabled"] = True
    cfg.data["nodes"]["hypothesis"] = {"enabled": True, "weight": 0.4,
                                       "status": "experimental", "horizon_days": 10}
    node = build_registry(cfg)["hypothesis"]
    ctx = MarketContext(store, cfg, offline=True)

    assert node.compute(ctx) == []                       # no active hypothesis
    h = _mk(store, stances=[
        {"symbol": "AAA", "direction": "long", "conviction": 0.8,
         "horizon_days": 10, "rationale": "r"},
        {"symbol": "ZZZZ", "direction": "long", "conviction": 0.9,
         "horizon_days": 10, "rationale": "no data → skipped"},
        {"symbol": "BBB", "direction": "avoid", "conviction": 0.5,
         "horizon_days": 5, "rationale": "r2"}])
    hypo.activate(cfg, store, h["id"], now_iso="2026-01-01T00:00:00-07:00")
    events = node.compute(ctx)
    by = {e.symbol: e for e in events}
    assert set(by) == {"AAA", "BBB"}                     # ZZZZ has no bars
    assert by["AAA"].score == 0.8 and by["AAA"].direction == "long"
    assert by["BBB"].score == 0.5 and by["BBB"].direction == "avoid"

    cfg.data["hypothesis"]["enabled"] = False            # master switch off
    assert node.compute(ctx) == []


def test_generate_validates_and_discards_garbage(cfg, store):
    cfg.data["hypothesis"]["enabled"] = True
    ctx = MarketContext(store, cfg, offline=True)
    good = {"thesis": "T", "summary": "s", "invalidation": "inv",
            "north_star_alignment": "aligned",
            "stances": [
                {"symbol": "aaa", "direction": "long", "conviction": 3.0,
                 "horizon_days": 99, "rationale": "clamped"},
                {"symbol": "$$$", "direction": "long", "conviction": 0.5,
                 "horizon_days": 10},                    # bad ticker → dropped
                {"symbol": "BBB", "direction": "sideways", "conviction": 0.5,
                 "horizon_days": 10}],                   # bad direction → dropped
            "watchlist": ["ge", "AAA", "GE", "ge!"]}     # dedup, drop in-universe/bad
    h = hypo.generate(cfg, store, StubAI(good), ctx)
    assert h["status"] == "proposed"
    st = json.loads(h["stances"])
    assert len(st) == 1 and st[0]["symbol"] == "AAA"
    assert st[0]["conviction"] == 1.0 and st[0]["horizon_days"] == 40   # clamped
    assert json.loads(h["watchlist"]) == ["GE"]

    assert hypo.generate(cfg, store, StubAI(None), ctx) is None          # AI down
    assert hypo.generate(cfg, store, StubAI({"nope": 1}), ctx) is None   # garbage
    cfg.data["hypothesis"]["enabled"] = False
    assert hypo.generate(cfg, store, StubAI(good), ctx) is None          # switch off


def test_north_star_has_no_stances_or_watchlist(cfg, store):
    cfg.data["hypothesis"]["enabled"] = True
    ctx = MarketContext(store, cfg, offline=True)
    resp = {"thesis": "durable philosophy", "summary": "s",
            "stances": [{"symbol": "AAA", "direction": "long",
                         "conviction": 0.9, "horizon_days": 10}],
            "watchlist": ["GE"]}
    h = hypo.generate(cfg, store, StubAI(resp), ctx, tier="north_star")
    assert json.loads(h["stances"]) == [] and json.loads(h["watchlist"]) == []


def test_engine_merges_watchlist(cfg, store):
    from conftest import synth_bars

    from specforge.engine import run_cycle
    store.upsert_bars("DDD", synth_bars(daily_drift=0.002), "test")
    cfg.data["hypothesis"]["enabled"] = True
    h = _mk(store, wl=["DDD"])
    hypo.activate(cfg, store, h["id"], now_iso="2026-01-01T00:00:00-07:00")
    before = list(cfg.get("universe", "symbols"))
    run_cycle(cfg, store, refresh_data=False)
    # The watchlist merges into the CYCLE-LOCAL universe (proven by the audit
    # below); shared config is immutable during execution (Sprint E1), so the
    # merge can never leak into later cycles or other services.
    assert cfg.get("universe", "symbols") == before
    merged = [a for a in store.audit_rows()
              if a["event_type"] == "hypothesis_watchlist_merged"]
    assert merged and "DDD" in json.loads(merged[0]["payload"])["added"]


def test_short_term_staleness(cfg, store):
    assert hypo.short_term_stale(cfg, store, "neutral") == "none active"
    h = _mk(store)
    hypo.activate(cfg, store, h["id"], now_iso="2026-07-01T18:00:00-07:00")
    assert hypo.short_term_stale(cfg, store, "neutral",
                                 now_iso="2026-07-02T18:00:00-07:00") is None
    assert "older" in hypo.short_term_stale(cfg, store, "neutral",
                                            now_iso="2026-07-09T18:00:00-07:00")
    assert "regime" in hypo.short_term_stale(cfg, store, "risk_off",
                                             now_iso="2026-07-02T18:00:00-07:00")
