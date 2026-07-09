"""AI budget/parse discipline + attribution weight-update bounds. Offline."""
from __future__ import annotations

from specforge.ai import AIClient, _parse_json_block
from specforge.attribution import update_weights


def test_parse_json_block_strictness():
    assert _parse_json_block('{"a": 1}') == {"a": 1}
    assert _parse_json_block('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_block('Sure! Here: {"a": 1} hope that helps') == {"a": 1}
    assert _parse_json_block("[1,2,3]") is None          # arrays are not objects
    assert _parse_json_block("no json here") is None
    assert _parse_json_block("BUY NVDA NOW!!!") is None  # injection-ish → discard


def test_reserve_then_commit_budget(cfg, store, monkeypatch):
    cfg.data["ai"] = {"enabled": True, "daily_budget_usd": 1.0,
                      "model": "m", "prices": {"m": {"input": 100.0, "output": 100.0}}}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    ai = AIClient(cfg, store)
    assert ai.available()
    # a task costing ~$0.9 reserves fine; a second identical one must NOT
    # half-start — reserve refuses because 0.9 + 0.9 > 1.0
    assert ai.reserve(0.9)
    assert not ai.reserve(0.9)
    ai._release(0.9)
    # actual ledger spend also counts against the budget
    store.ai_log("m", "test", "node", 0, 0, 0.95, cache_hit=False, ok=True)
    assert not ai.reserve(0.1)
    assert ai.reserve(0.04)


def test_parse_failures_disable_ai_not_trading(cfg, store, monkeypatch):
    cfg.data["ai"] = {"enabled": True, "daily_budget_usd": 1.0, "model": "m",
                      "prices": {"m": {"input": 1, "output": 1}}}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    ai = AIClient(cfg, store)
    for _ in range(5):
        ai._record_parse_failure("test")
    assert not ai.available()                    # AI off for the day...
    from specforge.risk import Governor
    assert not Governor(cfg, store).active_switches()   # ...but trading unaffected


def test_weight_update_bounds_and_auto_disable(cfg, store):
    # winner node: 25 solid trades → multiplier rises but clamps at max 2.0
    for i in range(25):
        store.record_trade({"symbol": "AAA", "entry_date": "2026-01-01",
                            "exit_date": "2026-02-01", "entry_price": 100,
                            "exit_price": 104, "qty": 1, "pnl": 4, "ret": 0.04,
                            "nodes": ["momentum"], "source": "paper",
                            "regime": "risk_on", "score_bucket": "s2"})
    res = update_weights(cfg, store, log=lambda *a: None)
    mult = store.get_weight_multiplier("momentum")
    assert 1.0 < mult <= 2.0
    assert res["momentum"]["action"] == "updated"

    # loser node: 35 bad trades → auto-disabled via config override
    cfg.data["nodes"]["reversal"] = {"enabled": True, "weight": 0.1}
    for i in range(35):
        store.record_trade({"symbol": "BBB", "entry_date": "2026-01-01",
                            "exit_date": "2026-02-01", "entry_price": 100,
                            "exit_price": 97, "qty": 1, "pnl": -3, "ret": -0.03,
                            "nodes": ["reversal"], "source": "paper",
                            "regime": "risk_on", "score_bucket": "s1"})
    res = update_weights(cfg, store, log=lambda *a: None)
    assert "AUTO-DISABLED" in res["reversal"]["action"]
    ov = store.kv_get("config_overrides")
    assert ov["nodes"]["reversal"]["enabled"] is False
    assert store.get_weight_multiplier("reversal") == 0.3   # clamped at min

def test_regime_conditioned_multipliers(cfg, store):
    # 30 risk_on winners + 30 risk_off losers on one node: the regime cells
    # qualify (n >= regime_min_n) and must yield opposite-signed multipliers,
    # consumed instead of the global one; regimes without a sample fall back.
    for regime, ret, px in (("risk_on", 0.04, 104), ("risk_off", -0.03, 97)):
        for i in range(30):
            store.record_trade({"symbol": "CCC", "entry_date": "2026-01-01",
                                "exit_date": "2026-02-01", "entry_price": 100,
                                "exit_price": px, "qty": 1, "pnl": ret * 100,
                                "ret": ret + (i % 3) * 0.001,  # non-zero variance
                                "nodes": ["momentum"], "source": "paper",
                                "regime": regime, "score_bucket": "s2"})
    update_weights(cfg, store, log=lambda *a: None)
    rm = store.kv_get("regime_multipliers")["momentum"]
    assert rm["risk_on"] > 1.0 and rm["risk_off"] < 1.0
    assert 0.3 <= rm["risk_off"] and rm["risk_on"] <= 2.0

    from specforge.ensemble import s_node_weight
    base = float(cfg.get("nodes", "momentum", "weight", default=0.0))
    assert s_node_weight("momentum", cfg, store, "risk_on") == base * rm["risk_on"]
    assert s_node_weight("momentum", cfg, store, "risk_off") == base * rm["risk_off"]
    # neutral has no cell -> global multiplier path
    assert s_node_weight("momentum", cfg, store, "neutral") == \
        base * store.get_weight_multiplier("momentum")


def test_per_purpose_model_routing_and_monthly_caps(cfg, store, monkeypatch):
    """D36: purpose→model routing; monthly ceiling and per-purpose cap block
    reservations even when the daily budget has room."""
    cfg.data["ai"] = {
        "enabled": True, "daily_budget_usd": 100.0, "monthly_budget_usd": 10.0,
        "purpose_monthly_caps": {"hypothesis": 2.0},
        "model": "cheap", "models": {"hypothesis": "flagship"},
        "prices": {"cheap": {"input": 1, "output": 1},
                   "flagship": {"input": 100, "output": 100}}}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    ai = AIClient(cfg, store)
    assert ai.model_for("hypothesis") == "flagship"
    assert ai.model_for("headline_classification") == "cheap"   # falls to default
    # per-purpose monthly cap: $1.9 spent on hypothesis this month → $0.2 more refused
    store.ai_log("flagship", "hypothesis", "hypothesis", 0, 0, 1.9,
                 cache_hit=False, ok=True)
    assert not ai.reserve(0.2, "hypothesis")
    assert ai.reserve(0.05, "hypothesis"); ai._release(0.05)
    # monthly ceiling: total month spend 9.9 → 0.2 refused for ANY purpose
    store.ai_log("cheap", "other", "n", 0, 0, 8.0, cache_hit=False, ok=True)
    assert store.ai_spend_month() == 9.9
    assert not ai.reserve(0.2, "other")
    assert ai.reserve(0.05, "other")
