"""No-lookahead guarantee + end-to-end paper loop on synthetic data (offline)."""
from __future__ import annotations

from specforge.data import MarketContext
from specforge.engine import run_cycle
from specforge.nodes.base import build_registry


def test_market_context_never_exposes_future(cfg, store):
    all_dates = [r["d"] for r in store.get_bars("AAA", "9999-12-31", 10000)]
    as_of = all_dates[len(all_dates) // 2]          # pick a mid-history date
    ctx = MarketContext(store, cfg, as_of=as_of)
    df = ctx.df("AAA", lookback=10000)
    assert df.index.max() <= as_of                   # invariant #3
    assert len(df) < len(all_dates)


def test_full_paper_cycle_and_audit_reconstruction(cfg, store):
    summary = run_cycle(cfg, store, refresh_data=False)
    assert summary["signals"] > 0                    # momentum sees the uptrend
    assert summary["equity"] > 0
    # audit trail must contain the full decision chain for the cycle
    events = {r["event_type"] for r in store.audit_rows(cycle_id=summary["cycle_id"])}
    assert {"cycle_start", "regime", "cycle_budget", "cycle_end"} <= events
    if any(v == "filled" for v in summary["entries"].values()):
        assert {"risk_decision", "broker_review", "order_filled"} <= events
        assert store.open_positions()
    # budget invariant: never deploy more than the cycle budget
    assert summary["budget_used"] <= summary["budget"] + 1e-6


def test_second_cycle_respects_duplicate_cooldown(cfg, store):
    s1 = run_cycle(cfg, store, refresh_data=False)
    filled = [s for s, v in s1["entries"].items() if v == "filled"]
    s2 = run_cycle(cfg, store, refresh_data=False)
    # same symbols must not be re-bought within the cooldown window
    for sym in filled:
        assert s2["entries"].get(sym) in (None, "rejected", "duplicate")


def test_exit_time_stop(cfg, store):
    """A position past 1.5× horizon gets closed by the time stop."""
    from specforge.broker.paper import KV_KEY
    from specforge.models import new_id
    # broker must actually hold the shares for the sell to fill
    store.kv_set(KV_KEY, {"cash": 955.0, "positions":
                          {"AAA": {"qty": 0.5, "avg_cost": 90.0,
                                   "opened_at": "2020-01-01T00:00:00"}}})
    store.save_position(new_id(), {
        "symbol": "AAA", "asset_type": "equity", "qty": 0.5, "avg_cost": 90.0,
        "opened_at": "2020-01-01T00:00:00", "horizon_days": 20, "stop_price": 1.0,
        "candidate_id": "x", "nodes": ["momentum"], "option_symbol": None,
        "status": "open"})
    summary = run_cycle(cfg, store, refresh_data=False)
    assert summary["exits"].get("AAA") == "filled"
    trades = store.trades()
    assert trades and trades[0]["exit_reason"].startswith("time_stop")
    assert trades[0]["regime"]                      # analog cell fields populated


def test_registry_skips_unimplemented_nodes(cfg):
    cfg.data["nodes"]["nonexistent_node"] = {"enabled": True, "weight": 0.5}
    reg = build_registry(cfg)
    assert "nonexistent_node" not in reg
    assert "momentum" in reg
