"""Bridge broker round-trip: queue → dump → report → reconcile → position."""
from __future__ import annotations

from datetime import datetime

from specforge.broker.bridge import (BridgeBroker, SNAPSHOT_KEY, bridge_dump,
                                     bridge_report)
from specforge.execution import Executor
from specforge.models import OrderIntent, TradeCandidate, new_id
from specforge.risk import Governor


def _candidate(symbol="AAA"):
    return TradeCandidate(
        id=new_id(), symbol=symbol, asset_type="equity", side="buy", thesis="t",
        final_score=0.4, target_notional=50, expected_return=0.02, ci_low=-0.05,
        ci_high=0.09, probability_positive=0.6, expected_apr=0.2, apr_ci_low=-0.4,
        apr_ci_high=0.9, horizon_days=20, max_loss=50, contributing_nodes=["momentum"])


def test_bridge_round_trip(cfg, store, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    monkeypatch.setenv("RH_ACCOUNT_WHITELIST", "RH123")
    cfg.data["live_trading_enabled"] = True
    cfg.data["broker"] = "robinhood_bridge"

    broker = BridgeBroker(cfg, store)

    # 1. no snapshot → review refuses (stale bridge = no trading)
    cand = _candidate()
    intent = OrderIntent.make(cand, qty=0.5, limit_price=100.0)
    assert not broker.review_order(intent).ok

    # 2. snapshot arrives via bridge_report → review passes
    bridge_report(store, {"account": {"equity": 1000, "cash": 900,
                                      "buying_power": 900, "positions": [],
                                      "quotes": {"AAA": 100.0}}, "orders": []})
    assert broker.review_order(intent).ok

    # 3. place → pending_relay, shows up in bridge_dump
    store.record_candidate(cand, "cyc1")
    store.record_order(intent)
    assert broker.place_order(intent) is None
    dump = bridge_dump(store, cfg)
    assert any(o["id"] == intent.id for o in dump["pending_intents"])

    # 4. bridge session reports the fill → reconcile creates the position
    bridge_report(store, {"orders": [{
        "intent_id": intent.id, "state": "filled", "qty": 0.5, "price": 100.05,
        "filled_at": datetime.now().astimezone().isoformat(),
        "broker_order_id": "rh-42"}]})
    ex = Executor(cfg, store, broker, Governor(cfg, store))
    res = ex.reconcile("cyc1")
    assert res == {"AAA": "filled"}
    pos = store.open_positions()
    assert pos and pos[0]["symbol"] == "AAA" and pos[0]["qty"] == 0.5
    orders = {o["id"]: o for o in store.orders_today()}
    assert orders[intent.id]["status"] == "filled"


def test_bridge_review_blocked_marks_dead(cfg, store, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    monkeypatch.setenv("RH_ACCOUNT_WHITELIST", "RH123")
    cfg.data["live_trading_enabled"] = True
    broker = BridgeBroker(cfg, store)
    store.kv_set(SNAPSHOT_KEY, {"equity": 1000, "cash": 900, "positions": [],
                                "as_of": datetime.now().astimezone().isoformat()})
    cand = _candidate("BBB")
    intent = OrderIntent.make(cand, qty=1, limit_price=50.0)
    store.record_order(intent)
    broker.place_order(intent)
    bridge_report(store, {"orders": [{"intent_id": intent.id,
                                      "state": "review_blocked",
                                      "note": "PDT warning"}]})
    ex = Executor(cfg, store, broker, Governor(cfg, store))
    assert ex.reconcile("cyc2") == {"BBB": "dead"}
    assert not store.open_positions()
