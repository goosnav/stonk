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


def test_broker_block_halts_remaining_entry_batch(cfg, store, monkeypatch):
    from datetime import datetime
    from specforge.models import AccountState, OrderReview, SignalEvent

    class BlockingBroker:
        def __init__(self):
            self.review_calls = 0

        def set_quotes(self, quotes):
            pass

        def get_account(self):
            return AccountState(equity=1000, cash=1000, buying_power=1000,
                                positions=[], as_of=datetime.now().isoformat())

        def review_order(self, intent):
            self.review_calls += 1
            return OrderReview(ok=False, warnings=["account_not_ready"])

        def place_order(self, intent):
            raise AssertionError("blocked review must never place")

    class Signals:
        id = "momentum"
        role = "alpha"
        degraded_reason = ""

        def compute(self, ctx):
            return [SignalEvent(
                symbol=s, direction="long", score=.8, confidence=.9,
                horizon_days=20, expected_return=.04, expected_volatility=.1,
                downside_estimate=-.1, evidence=["test"],
                data_as_of=datetime.now(), node_id="momentum")
                for s in ("AAA", "BBB", "CCC")]

    cfg.data["ensemble"]["min_final_score"] = -1
    cfg.data["nodes"]["quality_value"]["enabled"] = False
    cfg.data["risk"]["stale_data_max_age_days"] = 999
    # This test isolates broker batch behavior; classify the liquid synthetic
    # fixtures as fund-like so the independent company-dossier gate is not the
    # subject under test.
    monkeypatch.setattr("specforge.ensemble.ETF_SYMBOLS",
                        {"AAA", "BBB", "CCC", "SPY"})
    broker = BlockingBroker()
    summary = run_cycle(cfg, store, broker=broker, refresh_data=False,
                        registry={"momentum": Signals()})
    assert broker.review_calls == 1
    assert "broker_rejected" in summary["entries"].values()
    assert "skipped_broker_block" in summary["entries"].values()
    assert any(r["event_type"] == "entry_batch_halted"
               for r in store.audit_rows(cycle_id=summary["cycle_id"]))


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


def test_resting_sell_fill_closes_position_and_records_trade(cfg, store):
    from datetime import datetime
    from specforge.execution import Executor
    from specforge.models import AccountState, Fill, OrderReview, new_id
    from specforge.risk import Governor

    class RestingBroker:
        def get_account(self):
            return AccountState(equity=100, cash=0, buying_power=0, positions=[],
                                as_of=datetime.now().isoformat())

        def review_order(self, intent):
            return OrderReview(ok=True, warnings=[])

        def place_order(self, intent):
            return None

        def poll_order(self, broker_order_id, intent):
            return Fill(order_id=intent.id, symbol=intent.symbol, side="sell",
                        qty=intent.qty, price=110,
                        filled_at=datetime.now().isoformat())

    pid = new_id()
    store.save_position(pid, {
        "symbol": "AAA", "asset_type": "equity", "qty": 1, "avg_cost": 100,
        "opened_at": "2026-01-01T00:00:00", "horizon_days": 20,
        "stop_price": 90, "candidate_id": "", "nodes": ["momentum"],
        "option_symbol": None, "status": "open", "mode": "paper"})
    broker = RestingBroker()
    ex = Executor(cfg, store, broker, Governor(cfg, store))
    assert ex.execute_exit(store.open_positions("paper")[0], 110, "time_stop",
                           broker.get_account(), "c1", "neutral") == "resting"
    assert ex.reconcile("c2")["AAA"] == "filled"
    assert store.open_positions("paper") == []
    assert store.trades(source="paper")[0]["exit_reason"] == "time_stop"


def test_partial_sell_reduces_position_without_closing(cfg, store):
    from specforge.broker.paper import KV_KEY, PaperBroker
    from specforge.execution import Executor
    from specforge.models import new_id
    from specforge.risk import Governor
    store.kv_set(KV_KEY, {"cash": 0, "positions": {
        "AAA": {"qty": 2, "avg_cost": 100, "asset_type": "equity"}}})
    pid = new_id()
    store.save_position(pid, {
        "symbol": "AAA", "asset_type": "equity", "qty": 2, "avg_cost": 100,
        "opened_at": "2026-01-01T00:00:00", "horizon_days": 20,
        "stop_price": 80, "candidate_id": "", "nodes": ["momentum"],
        "option_symbol": None, "status": "open", "mode": "paper"})
    broker = PaperBroker(cfg, store); broker.set_quotes({"AAA": 110})
    ex = Executor(cfg, store, broker, Governor(cfg, store))
    assert ex.execute_exit(store.open_positions("paper")[0], 110, "rebalance trim",
                           broker.get_account(), "c1", "neutral", qty=.5) == "filled"
    remaining = store.open_positions("paper")
    assert len(remaining) == 1 and abs(remaining[0]["qty"] - 1.5) < 1e-8
    assert store.trades(source="paper")[0]["qty"] == .5
    assert any(r["event_type"] == "position_trimmed" for r in store.audit_rows())


def test_live_model_failure_keeps_production_evidence_pipeline_alive(cfg, store):
    """A shadow learned component cannot freeze the production evidence path."""
    import json
    from datetime import datetime
    from specforge.models import AccountState, SignalEvent

    class Broker:
        def set_quotes(self, quotes): pass
        def get_account(self):
            return AccountState(100, 100, 100, [], datetime.now().isoformat())

    class Signals:
        id, role, degraded_reason = "momentum", "alpha", ""
        def compute(self, ctx):
            return [SignalEvent("SPY", "long", .9, .9, 21, .05, .1, -.1,
                                ["test"], datetime.now(), "momentum")]

    cfg.data["mode"] = "live"
    cfg.data["ensemble"]["min_final_score"] = -1
    cfg.data["nodes"]["quality_value"]["enabled"] = False
    summary = run_cycle(cfg, store, broker=Broker(), refresh_data=False,
                        registry={"momentum": Signals()})
    # The learned reason remains explicit while the production path attempts
    # the governed order (this minimal fake broker rejects at review).
    assert summary["model_state"]["block_reason"] == "BLOCKED: GRAPH"
    assert summary["model_state"]["effective_blend"] == 0.0
    assert summary["model_state"]["production_evidence"] is True
    assert summary["entries"] == {"SPY": "rejected"}
    assert store.db.execute(
        "SELECT payload FROM audit WHERE event_type='model_entries_blocked'"
    ).fetchone() is None


def test_audit_redacts_account_and_secret_fields(store):
    import json
    store.audit("broker_payload", {"account_number": "934803396",
                                    "nested": {"access_token": "very-secret"},
                                    "safe": "kept"})
    payload = json.loads(store.audit_rows()[0]["payload"])
    assert payload["account_number"] == "[redacted]"
    assert payload["nested"]["access_token"] == "[redacted]"
    assert payload["safe"] == "kept"


def test_audit_file_mirror(store, tmp_path):
    import json
    from specforge.store import configure_file_logging
    path = configure_file_logging("paper", tmp_path)
    store.audit("test_event", {"ok": True}, "cycle")
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["event"] == "test_event" and row["payload"]["ok"] is True


def test_operator_console_frame_is_quiet_and_informative(monkeypatch):
    from specforge import cli
    payloads = {
        "/api/health": {"readiness": {"trading": True, "reasons": []},
                        "broker": {"adapter": "paper", "connected": True},
                        "engine": {"heartbeat_age_s": 5},
                        "market": {"session": "regular", "et": "12:00 ET"},
                        "pending_approvals": 0},
        "/api/status": {"mode": "paper", "equity": 1000, "cash": 900,
                        "buying_power": 900, "day_pnl": 2, "net_pnl": 3,
                        "realized_pnl": 1, "unrealized_pnl": 2,
                        "drawdown_from_peak": .01, "regime": "risk_on",
                        "deployment_multiplier": 1, "cycle_budget": 100,
                        "kill_switches": {}, "positions": [{
                            "symbol": "AAA", "qty": 1, "avg_cost": 98,
                            "last": 100, "value": 100, "pnl_usd": 2,
                            "pnl_pct": .0204}]},
        "/api/engine": {"state": {"phase": "idle", "detail": "cycle done"}},
        "/api/today": {"hypothesis": "momentum remains constructive",
                       "news": {"items": [{"symbol": "AAA", "sentiment": .4,
                                            "summary": "positive catalyst"}]},
                       "fundamentals": {"items": []}},
        "/api/decisions": {"cycle": {"id": "c1", "budget": 100,
                                        "budget_used": 20},
                           "considered": [{"symbol": "AAA", "score": .6,
                                           "result": "filled", "thesis": "momentum"}],
                           "working": []},
    }
    monkeypatch.setattr(cli, "_console_api", lambda port, path, timeout=20: payloads[path])
    frame = cli._console_frame(8420, color=False)
    assert "equity $1,000.00" in frame
    assert "POSITIONS (1)" in frame and "AAA" in frame
    assert "AI COMMENTARY" in frame and "positive catalyst" in frame
    assert "no kill switches active" in frame
    assert "rh_mcp_call" not in frame


def test_registry_skips_unimplemented_nodes(cfg):
    cfg.data["nodes"]["nonexistent_node"] = {"enabled": True, "weight": 0.5}
    reg = build_registry(cfg)
    assert "nonexistent_node" not in reg
    assert "momentum" in reg


def test_gap_node_replays_open_without_future_close(cfg, store):
    from specforge.data import MarketContext
    latest = store.latest_bar_date("AAA")
    rows = store.get_bars("AAA", latest, 2)
    prev = rows[-2]["close"]
    store.db.execute("UPDATE bars SET open=?,high=? WHERE symbol='AAA' AND d=?",
                     (prev * 1.03, prev * 1.04, latest)); store.db.commit()
    ctx = MarketContext(store, cfg, as_of=latest, offline=True)
    gap = build_registry(cfg)["gap"]
    events = gap.compute(ctx)
    assert any(e.symbol == "AAA" and "opening gap" in e.evidence[0] for e in events)


def test_paper_positions_invisible_to_live_mode(cfg, store):
    """Paper and live share one DB; a live scan must not see paper positions."""
    from specforge.models import new_id
    store.save_position(new_id(), {
        "symbol": "AAA", "asset_type": "equity", "qty": 1.0, "avg_cost": 100.0,
        "opened_at": "2026-01-01T00:00:00", "horizon_days": 20, "stop_price": 0,
        "candidate_id": "x", "nodes": [], "option_symbol": None,
        "status": "open", "mode": "paper"})
    assert len(store.open_positions(mode="paper")) == 1
    assert store.open_positions(mode="live") == []


def test_paper_orders_invisible_to_live_mode(store):
    """D26: a paper-mode order must not block a live entry (duplicate cooldown)
    or inflate live daily caps (orders_today)."""
    from specforge.models import OrderIntent, new_id
    now = "2026-07-09T09:30:00-07:00"
    intent = OrderIntent(
        id=new_id(), candidate_id="c", symbol="CAT", asset_type="equity",
        side="buy", qty=1.0, limit_price=100.0, notional=100.0,
        idempotency_key=new_id(), created_at=now, status="filled")
    assert store.record_order(intent, mode="paper")
    # duplicate cooldown: paper order invisible to a live check, visible to paper
    assert store.recent_order_exists("CAT", "buy", 60, now_iso=now, mode="live") is False
    assert store.recent_order_exists("CAT", "buy", 60, now_iso=now, mode="paper") is True
    # daily-cap counting is mode-scoped too
    assert store.orders_today("buy", day="2026-07-09", mode="live") == []
    assert len(store.orders_today("buy", day="2026-07-09", mode="paper")) == 1


def test_current_config_survives_cross_mode_override(cfg, store):
    """D38: a live-only override (single position 30%, legal under live.yaml's
    advanced_override) must NOT crash a paper-mode load — the server refuses
    the override, keeps the safe file config, and audits it."""
    from specforge.app import current_config
    from specforge.config import ConfigError, load_config
    store.kv_set("config_overrides", {"risk": {"max_single_equity_position": 0.30}})
    # direct load still raises (validate is not weakened)
    import pytest as _pt
    with _pt.raises(ConfigError):
        load_config("paper", overrides=store.kv_get("config_overrides"))
    # but the server path survives, on the SAFE default
    c = current_config(store, "paper")
    assert c.get("risk", "max_single_equity_position") <= 0.25
    assert any(a["event_type"] == "config_override_rejected" for a in store.audit_rows())


def test_health_endpoint(cfg, store):
    """V3 truth contract: reasons NEVER empty when not trading; broker honest;
    heartbeat tracked; never throws."""
    from fastapi.testclient import TestClient
    from specforge.app import create_app
    client = TestClient(create_app(cfg, store, with_scheduler=False))
    body = client.get("/api/health").json()
    assert body["mode"] == "paper"
    assert body["broker"]["adapter"] == "paper" and body["broker"]["connected"]
    assert "SIMULATION" in body["broker"]["detail"]
    assert body["readiness"]["trading"] is False
    assert any("PAPER" in r for r in body["readiness"]["reasons"])       # sim labeled
    assert any("heartbeat" in r for r in body["readiness"]["reasons"])   # never ran
    assert body["pending_approvals"] == 0
    # heartbeat write clears the never-ran reason
    from specforge.health import write_heartbeat
    write_heartbeat(store, "cyc1", "paper", source="cron")
    body2 = client.get("/api/health").json()
    assert not any("heartbeat" in r for r in body2["readiness"]["reasons"])
    assert body2["engine"]["heartbeat_source"] == "cron"


def test_health_flags_stalled_cadence(cfg, store, monkeypatch):
    from datetime import datetime, timedelta
    from specforge import health

    old = (datetime.now().astimezone() - timedelta(minutes=31)).isoformat()
    store.kv_set("heartbeat", {"at": old, "cycle_id": "stuck",
                               "mode": "paper", "source": "serve"})
    monkeypatch.setattr(health, "_market_clock", lambda: {
        "open": True, "session": "regular", "et": "12:00 ET"})
    monkeypatch.setattr(health, "_broker_health", lambda cfg, store: {
        "adapter": "paper", "connected": True, "detail": "SIMULATION"})
    body = health.system_health(cfg, store, scheduler_alive=True)
    assert any("expected about every 10m" in r
               for r in body["readiness"]["reasons"])


def test_set_env_var_upserts_and_preserves(tmp_path, monkeypatch):
    """D27: the .env upsert must replace only the target key and keep the rest,
    and apply the value to os.environ live."""
    import specforge.config as config
    env = tmp_path / ".env"
    env.write_text("LIVE_TRADING_ENABLED=false\nAI_API_KEY=old\nFRED_API_KEY=abc\n")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    config.set_env_var("AI_API_KEY", "newkey")
    text = env.read_text()
    assert "AI_API_KEY=newkey" in text
    assert "AI_API_KEY=old" not in text
    assert "LIVE_TRADING_ENABLED=false" in text and "FRED_API_KEY=abc" in text  # untouched
    import os
    assert os.environ["AI_API_KEY"] == "newkey"                                 # live
    config.set_env_var("NEW_ONLY", "1")                                         # append path
    assert "NEW_ONLY=1" in env.read_text()


def test_ai_provider_endpoint_never_leaks_key(cfg, store, tmp_path, monkeypatch):
    """D27: GET returns only a masked hint; POST persists to .env, never echoes
    the key, and audits provider without the secret."""
    import specforge.config as config
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("AI_BASE_URL", raising=False)
    from fastapi.testclient import TestClient
    from specforge.app import create_app
    client = TestClient(create_app(cfg, store, with_scheduler=False))

    r = client.post("/api/ai/provider",
                    json={"provider": "anthropic", "api_key": "sk-secret-123456"})
    assert r.status_code == 200
    assert "sk-secret-123456" not in r.text                    # POST never echoes the key
    body = client.get("/api/ai/provider").json()
    assert body["provider"] == "anthropic"
    assert body["base_url"] == "https://api.anthropic.com/v1"
    assert body["key_set"] is True and body["key_hint"] == "…3456"
    assert "sk-secret-123456" not in client.get("/api/ai/provider").text
    # key landed in .env, audit recorded provider but not the secret
    assert "AI_API_KEY=sk-secret-123456" in (tmp_path / ".env").read_text()
    ev = [a for a in store.audit_rows() if a["event_type"] == "ai_provider_set"]
    assert ev and "sk-secret-123456" not in ev[0]["payload"]
    # bad base_url rejected
    assert client.post("/api/ai/provider",
                       json={"provider": "custom", "base_url": "ftp://x"}).status_code == 400


def test_portfolio_value_and_steering_endpoints(cfg, store):
    """V4: portfolio series merges daily + intraday marks; steering API lists,
    decides, and rejects bad choices."""
    from fastapi.testclient import TestClient

    from specforge import steering
    from specforge.app import create_app
    client = TestClient(create_app(cfg, store, with_scheduler=False))

    from datetime import date, timedelta as td
    store.record_equity(1000.0, 500.0, "paper",
                        d=(date.today() - td(days=1)).isoformat())
    store.record_equity(1005.0, 500.0, "paper")          # today's scan mark
    assert store.record_intraday_mark(1010.0, 500.0, "paper") is True
    assert store.record_intraday_mark(1011.0, 500.0, "paper") is False   # throttled
    pv = client.get("/api/portfolio_value?range=1W").json()
    # yesterday's daily + today's intraday; today's daily superseded by intraday
    assert pv["current"] == 1010.0 and len(pv["points"]) == 2

    r = steering.create(cfg, store, "risk_suggestion", title="t", context="c",
                        options=[{"key": "adopt", "label": "a", "detail": ""},
                                 {"key": "keep", "label": "k", "detail": ""}],
                        recommended="adopt",
                        payload={"path": ["risk", "max_daily_loss"], "value": 0.03})
    body = client.get("/api/steering").json()
    assert [p["id"] for p in body["pending"]] == [r["id"]]
    assert client.post(f"/api/steering/{r['id']}", json={"choice": "zzz"}).status_code == 400
    assert client.post(f"/api/steering/{r['id']}", json={"choice": "adopt"}).status_code == 200
    assert client.get("/api/steering").json()["pending"] == []
    # dangerous suggested value still bounced by the shared validated path
    r2 = steering.create(cfg, store, "risk_suggestion", title="t", context="c",
                         options=[{"key": "adopt", "label": "a", "detail": ""}],
                         recommended="adopt",
                         payload={"path": ["risk", "max_daily_loss"], "value": 0.5})
    assert client.post(f"/api/steering/{r2['id']}", json={"choice": "adopt"}).status_code == 400


def test_live_quotes_price_orders_not_stale_close(cfg, store):
    """D35 root cause: limits must price off live quotes when provided, not
    yesterday's daily close (stale limits = resting unfilled orders)."""
    as_of = store.latest_bar_date("AAA")     # synth data ends before today
    ctx = MarketContext(store, cfg, as_of=as_of)
    live = {s: round(ctx.close(s) * 1.03, 4) for s in ["SPY", "AAA", "BBB", "CCC"]}
    summary = run_cycle(cfg, store, as_of=as_of, refresh_data=False,
                        live_quotes=live)
    buys = [o for o in store.orders_today("buy", day=as_of)
            if o["status"] in ("filled", "reviewed", "placed")]
    assert buys, f"no buys placed: {summary['entries']}"
    off = cfg.get("execution", "limit_offset_pct", default=0.001)
    for o in buys:
        assert abs(o["limit_price"] - live[o["symbol"]] * (1 + off)) < 0.01, \
            f"{o['symbol']} limit {o['limit_price']} not from live px {live[o['symbol']]}"


def test_today_digest_endpoint(cfg, store):
    """D35: the Today panel is real audit/orders/kv data — scans, candidates,
    order outcomes, and the AI reads when present."""
    from fastapi.testclient import TestClient

    from specforge.app import create_app
    client = TestClient(create_app(cfg, store, with_scheduler=False))
    run_cycle(cfg, store, refresh_data=False)
    store.kv_set("news_synopsis", {"ts": "2026-07-09T10:00:00-07:00", "items": [
        {"symbol": "AAA", "sentiment": 0.6, "catalyst": "earnings",
         "summary": "beat", "already_priced": False}]})
    t = client.get("/api/today").json()
    assert t["scans"] >= 1 and isinstance(t["orders"], dict)
    assert t["news"]["items"][0]["symbol"] == "AAA"
    assert t["hypothesis"] is None                      # none active → honest null


def test_pnl_series_is_deposit_independent(cfg, store):
    """D36: the P&L chart never moves on deposits — only realized trades and
    marked unrealized P&L feed it."""
    from datetime import date, timedelta as td

    from fastapi.testclient import TestClient

    from specforge.app import create_app
    client = TestClient(create_app(cfg, store, with_scheduler=False))
    y = (date.today() - td(days=1)).isoformat()
    store.record_trade({"symbol": "AAA", "entry_date": y, "exit_date": y,
                        "entry_price": 100, "exit_price": 105, "qty": 1,
                        "pnl": 5.0, "ret": 0.05, "source": "paper"})
    # deposit-like equity jump: mark equity way up but pnl only +2 unrealized
    store.record_intraday_mark(5000.0, 4000.0, "paper", pnl=7.0)
    pl = client.get("/api/pnl?range=1W").json()
    assert pl["current"] == 7.0                          # 5 realized + 2 unrealized
    assert [p["pnl"] for p in pl["points"]] == [5.0, 7.0]
    assert all(p["pnl"] < 100 for p in pl["points"])     # the 5000 never leaks in


def test_decisions_endpoint(cfg, store):
    """D36: the decisions feed shows every considered move with the governor's
    verdict and any working orders."""
    from fastapi.testclient import TestClient

    from specforge.app import create_app
    client = TestClient(create_app(cfg, store, with_scheduler=False))
    run_cycle(cfg, store, refresh_data=False)
    d = client.get("/api/decisions").json()
    assert d["cycle"]["regime"]
    assert d["considered"], "cycle produced candidates but decisions shows none"
    row = d["considered"][0]
    assert {"symbol", "score", "verdict", "reasons", "result"} <= set(row)
    assert isinstance(d["working"], list)


def test_model_endpoint(cfg, store):
    """V4: the model view exposes every configured node with effective weight
    = base × learned multiplier, plus regime and hypothesis link."""
    from fastapi.testclient import TestClient

    from specforge.app import create_app
    client = TestClient(create_app(cfg, store, with_scheduler=False))
    store.set_weight_multiplier("momentum", 1.5, note="test")
    m = client.get("/api/model").json()
    by = {n["id"]: n for n in m["nodes"]}
    assert by["momentum"]["multiplier"] == 1.5
    assert by["momentum"]["effective_weight"] == round(
        by["momentum"]["base_weight"] * 1.5, 4)
    assert m["regime"] and "min_final_score" in m["ensemble"]
    assert m["hypothesis"]["enabled"] is False and m["hypothesis"]["short_term"] is None
