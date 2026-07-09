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


def test_commit_reports_snapshots_dev_reports(store, tmp_path):
    """Nightly report snapshot: commits new files under dev/reports, audits it,
    and is a no-op when nothing changed."""
    import subprocess

    from specforge.app import _commit_reports

    root = tmp_path / "repo"
    (root / "dev" / "reports").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "--allow-empty", "-m", "init"],
                   cwd=root, check=True)
    (root / "dev" / "reports" / "r.json").write_text("{}")
    # commit identity via repo config so _commit_reports's plain git works
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)

    _commit_reports(store, root=root)
    log = subprocess.run(["git", "log", "--oneline"], cwd=root,
                         capture_output=True, text=True).stdout
    assert "nightly dev/reports snapshot" in log
    assert store.db.execute(
        "select count(*) from audit where event_type='reports_committed'"
    ).fetchone()[0] == 1

    _commit_reports(store, root=root)  # nothing new → no second commit/audit
    log2 = subprocess.run(["git", "log", "--oneline"], cwd=root,
                          capture_output=True, text=True).stdout
    assert log2.count("nightly") == 1


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
    live = {s: round(ctx.close(s) * 1.03, 4) for s in ["AAA", "BBB", "CCC"]}
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
