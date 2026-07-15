from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from specforge.data import MarketContext
from specforge.ensemble import score
from specforge.evidence import latest_dossier, persist_dossier
from specforge.models import SignalEvent, signed_alpha


def event(node, direction, magnitude, confidence=1.0):
    return SignalEvent("AAA", direction, magnitude, confidence, 21, .05, .1, -.1,
                       [node], datetime.now(), node)


def test_direction_owns_sign_and_legacy_negative_avoid_stays_negative():
    avoid = event("fundamentals", "avoid", .8, .8)
    assert signed_alpha(avoid) == pytest.approx(-.64)
    assert signed_alpha("avoid", -.8, .8) == pytest.approx(-.64)
    with pytest.raises(ValueError, match="magnitude"):
        event("fundamentals", "avoid", -.8, .8)


def test_cat_regression_fundamental_avoid_cannot_boost_momentum(cfg, store):
    cfg.data["ensemble"]["min_final_score"] = .15
    cfg.data["nodes"]["fundamentals"]["weight"] = .30
    ctx = MarketContext(store, cfg, as_of=store.latest_bar_date("AAA"), offline=True)
    events = [event("momentum", "long", .6935),
              event("business_fundamentals", "avoid", .8, .8)]
    candidates = score(events, "risk_on", cfg, store, [], ctx)
    assert not any(c.symbol == "AAA" for c in candidates)
    snapshot = store.kv_get("evidence_last_scores")["symbols"]["AAA"]
    business = next(d for d in snapshot["evidence"]
                    if d.get("node") == "family:business")
    assert business["family_contribution"] < 0


def test_verified_dossier_is_consumed_by_live_node_and_api(cfg, store):
    source = "SEC:0001:item_7"
    sources = {"catalog": [{"id": source, "type": "filing_section",
                             "text": "Cash generation improved."}],
               "bars_as_of": store.latest_bar_date("AAA")}
    report = {
        "fundamental": {"stance": "attractive", "confidence": .8,
                        "horizon_days": 60, "thesis": "Cash generation improved",
                        "contrary_evidence": ["cyclicality"], "catalysts": [],
                        "thesis_breakers": ["cash flow reversal"],
                        "citations": [{"source_id": source, "claim": "cash flow"}]},
        "catalyst": {"stance": "neutral", "confidence": .5,
                     "horizon_days": 14, "thesis": "No immediate catalyst",
                     "contrary_evidence": [], "catalysts": [],
                     "thesis_breakers": [], "citations": []},
    }
    dossier = persist_dossier(store, "AAA", sources["bars_as_of"], sources, [], report)
    assert dossier["status"] == "ready"
    assert latest_dossier(store, "AAA")["source_hash"] == dossier["source_hash"]

    from specforge.nodes.business_fundamentals import Node
    node = Node({"weight": .3, "horizon_days": 60})
    node.id = "business_fundamentals"
    ctx = MarketContext(store, cfg, as_of=sources["bars_as_of"], offline=True)
    emitted = node.compute(ctx)
    assert emitted and emitted[0].direction == "long"
    assert source in emitted[0].evidence[0]

    from specforge.app import create_app
    body = TestClient(create_app(cfg, store, with_scheduler=False)).get(
        "/api/evidence/AAA").json()
    assert body["available"] is True
    assert body["dossier"]["fundamental_memo"]["stance"] == "attractive"


def test_future_created_dossier_is_hidden_from_historical_replay(store):
    sources = {"bars_as_of": "2026-07-15", "filings": [], "news": []}
    dossier = persist_dossier(store, "AAA", "2026-07-15", sources, [], {})
    store.db.execute("UPDATE company_evidence SET created_at='2026-07-15T12:00:00' WHERE id=?",
                     (dossier["id"],))
    store.db.commit()
    assert latest_dossier(store, "AAA", "2026-07-14", "2026-07-14") is None
    assert latest_dossier(store, "AAA", "2026-07-15", "2026-07-15")["id"] == dossier["id"]

def test_legacy_backtests_remain_visible_but_not_live_analogs(store):
    base = {"symbol": "AAA", "entry_date": "2025-01-01", "exit_date": "2025-01-10",
            "entry_price": 10, "exit_price": 11, "qty": 1, "pnl": 1, "ret": .1,
            "score_bucket": "s2", "regime": "risk_on", "nodes": ["momentum"]}
    store.record_trade({**base, "source": "backtest"})
    store.record_trade({**base, "id": "live-one", "source": "live", "ret": -.02})
    assert len(store.trades(source="backtest")) == 1
    assert store.trades(source="backtest")[0]["qualified"] == 0
    assert store.analog_returns("s2", "risk_on") == [-.02]
    assert store.analog_returns("s2", "risk_on", evidence_version="evidence.v2",
                                horizon_days=20, asset_type="equity") == [-.02]
    assert store.analog_returns("s2", "risk_on", evidence_version="evidence.v3",
                                horizon_days=20, asset_type="equity") == []
