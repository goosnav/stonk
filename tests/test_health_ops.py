"""Reliability/observability contract tests (2026-07-10 rebase).

Covers: the app-health rollup (ok/degraded/stale/error — distinct from
readiness.trading, which is legitimately false outside market hours), the
/health liveness and /api/metrics monitor endpoints, sanitized last-error,
cycle counters, /api/status backward compatibility, and the operator
checker scripts/check_health.py (exit-code contract Hermes/watchdogs use).
All offline: fixture DBs, kv-seeded broker probes, monkeypatched clock.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from specforge import health as health_mod
from specforge.app import create_app
from specforge.config import load_config
from specforge.health import _redact, rollup, system_health, write_heartbeat
from specforge.risk import Governor
from specforge.store import Store

ROOT = Path(__file__).resolve().parent.parent


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clock(open_: bool):
    return lambda: {"open": open_, "session": "regular" if open_ else "closed",
                    "et": "12:00 ET"}


def _fresh_spy(store):
    """Synthetic fixture bars end ~2 weeks ago (conftest walks weekdays from a
    fixed offset); tests asserting status=ok need a benchmark bar from today
    so the data-staleness alert stays quiet."""
    store.upsert_bars("SPY", [{"d": datetime.now().date().isoformat(),
                               "open": 100, "high": 101, "low": 99,
                               "close": 100.5, "volume": 1_000_000}], "test")


@pytest.fixture()
def client(cfg, store):
    return TestClient(create_app(cfg, store, with_scheduler=False))


# ---------------- rollup: the app-health verdict ----------------

def _base_health() -> dict:
    """Minimal healthy system_health()-shaped dict (market closed)."""
    return {
        "mode": "paper",
        "broker": {"adapter": "paper", "connected": True, "detail": ""},
        "engine": {"heartbeat_age_s": 60, "scheduler_alive": True,
                   "heartbeat_stale_s": 1800},
        "market": {"open": False, "session": "closed", "et": "20:00 ET"},
        "data": {"newest_bar": "2026-07-10", "age_days": 0, "stale_limit_days": 4},
        "kill_switches": [],
        "broker_block": None,
        "last_error": None,
    }


def test_rollup_ok_when_market_closed_and_not_trading():
    status, alerts = rollup(_base_health())
    assert status == "ok" and alerts == []


def test_rollup_degraded_on_broker_disconnect_and_kill_switch():
    h = _base_health()
    h["broker"] = {"adapter": "robinhood_mcp", "connected": False, "detail": "auth expired"}
    h["kill_switches"] = ["rejected_orders"]
    status, alerts = rollup(h)
    assert status == "degraded"
    assert any("broker" in a for a in alerts)
    assert any("kill switch" in a for a in alerts)


def test_rollup_degraded_on_broker_block_and_stale_data():
    h = _base_health()
    h["broker_block"] = "investor profile incomplete"
    h["data"]["age_days"] = 9
    status, alerts = rollup(h)
    assert status == "degraded" and len(alerts) == 2


def test_rollup_stale_when_market_open_and_no_recent_scan():
    h = _base_health()
    h["market"]["open"] = True
    h["engine"]["heartbeat_age_s"] = 7200
    status, alerts = rollup(h)
    assert status == "stale"
    h["engine"]["heartbeat_age_s"] = None          # never scanned
    assert rollup(h)[0] == "stale"


def test_rollup_error_beats_stale_and_none_scheduler_is_unknown():
    h = _base_health()
    h["market"]["open"] = True
    h["engine"]["heartbeat_age_s"] = 7200
    h["engine"]["scheduler_alive"] = False
    assert rollup(h)[0] == "error"
    h["engine"]["scheduler_alive"] = None          # cron/TUI caller: not an error
    assert rollup(h)[0] == "stale"


def test_rollup_recent_error_degrades_but_old_error_does_not():
    h = _base_health()
    h["last_error"] = {"event": "scheduler_error", "age_s": 120, "detail": "boom"}
    assert rollup(h)[0] == "degraded"
    h["last_error"]["age_s"] = 86400
    assert rollup(h)[0] == "ok"


# ---------------- sanitization ----------------

def test_redact_strips_tokens_account_numbers_and_long_blobs():
    dirty = ("failed for account 934803396 with key=sk-abcDEF0123456789 and "
             "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload blob")
    clean = _redact(dirty)
    assert "934803396" not in clean
    assert "sk-abcDEF0123456789" not in clean
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in clean
    assert "failed for account" in clean            # prose survives
    # timestamps, cycle ids, and hostnames/paths survive redaction
    assert _redact("2026-07-10T12:54:29 cycle 727edae9855b") == \
        "2026-07-10T12:54:29 cycle 727edae9855b"
    assert _redact("connect to agent.robinhood.com/mcp/trading failed") == \
        "connect to agent.robinhood.com/mcp/trading failed"


def test_system_health_last_error_is_sanitized_and_aged(cfg, store, monkeypatch):
    monkeypatch.setattr(health_mod, "_market_clock", _clock(False))
    store.audit("scheduler_error", {"error": "quote fetch died: account 934803396 token=abcdef0123456789ABCDEF012345"})
    h = system_health(cfg, store, scheduler_alive=True)
    le = h["last_error"]
    assert le and le["event"] == "scheduler_error" and le["age_s"] < 60
    assert "934803396" not in le["detail"]
    assert "abcdef0123456789ABCDEF012345" not in le["detail"]
    assert h["status"] == "degraded"                # fresh error → operator attention
    assert all("934803396" not in a for a in h["alerts"])


def test_closed_market_scheduler_is_idle_not_dead(cfg, store, monkeypatch):
    monkeypatch.setattr(health_mod, "_market_clock", _clock(False))
    h = system_health(cfg, store, scheduler_alive=True)
    assert h["engine"]["operational_state"] == "closed_idle"
    store.kv_set("research_state", {"phase": "tcn", "detail": "training"})
    h = system_health(cfg, store, scheduler_alive=True)
    assert h["engine"]["operational_state"] == "researching"
    h = system_health(cfg, store, scheduler_alive=False)
    assert h["engine"]["operational_state"] == "offline"


def test_running_operator_job_surfaces_as_researching(cfg, store, monkeypatch):
    monkeypatch.setattr(health_mod, "_market_clock", _clock(False))
    now = _now_iso()
    store.db.execute("INSERT INTO research_jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                     ("j1", "train_holdings", "running", 10, now, now, None,
                      "{}", '{"symbol":"AAPL","index":1,"total":7}', None, None, 1))
    store.db.commit()
    h = system_health(cfg, store, scheduler_alive=True)
    assert h["engine"]["operational_state"] == "researching"
    assert h["engine"]["active_research_job"]["progress"]["symbol"] == "AAPL"


# ---------------- endpoints ----------------

def test_liveness_endpoint_is_dependency_free(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["pid"] > 0 and body["uptime_s"] >= 0
    assert body["mode"] == "paper"


def test_m1a_health_aliases_and_cross_origin_write_guard(client):
    live = client.get("/health/live")
    ready = client.get("/health/ready")
    assert live.status_code == 200 and live.json()["ok"] is True
    assert ready.status_code == 200 and ready.json()["ready"] is True
    refused = client.post("/api/research/jobs", json={"kind": "discover"},
                          headers={"Origin": "https://malicious.example"})
    assert refused.status_code == 403
    allowed = client.post("/api/research/jobs", json={"kind": "discover"},
                          headers={"Origin": "http://testserver"})
    assert allowed.status_code == 200


def test_metrics_contract_healthy(client, cfg, store, monkeypatch):
    monkeypatch.setattr(health_mod, "_market_clock", _clock(False))
    _fresh_spy(store)
    write_heartbeat(store, "cycle123", "paper", source="test")
    r = client.get("/api/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["schema"] == "stonk.metrics.v1"
    assert m["status"] == "ok" and m["alerts"] == []
    assert m["mode"] == "paper" and m["as_of"]
    assert m["process"]["pid"] > 0 and m["process"]["uptime_s"] >= 0
    assert m["health"]["readiness"]["trading"] is False      # market closed
    assert m["cycles"]["last_scan_at"]                       # heartbeat visible
    assert m["positions_open"] == 0


def test_metrics_cycle_counters(client, store, monkeypatch):
    monkeypatch.setattr(health_mod, "_market_clock", _clock(False))
    store.audit("cycle_end", {"mode": "paper", "candidates": 3}, "c1")
    store.audit("cycle_end", {"mode": "paper", "candidates": 1}, "c2")
    store.audit("cycle_end", {"mode": "live", "candidates": 1}, "c3")   # other mode
    store.audit("scheduler_error", {"error": "boom"})
    m = client.get("/api/metrics").json()
    assert m["cycles"]["today"] == 2                # paper only
    assert m["cycles"]["errors_today"] == 1


def test_metrics_stale_and_kill_switch_visibility(client, cfg, store, monkeypatch):
    monkeypatch.setattr(health_mod, "_market_clock", _clock(True))
    m = client.get("/api/metrics").json()
    assert m["status"] == "stale"                   # market open, never scanned
    write_heartbeat(store, "cyc", "paper", source="test")
    Governor(cfg, store).trip("manual", "operator drill", auto_clear_days=1)
    m = client.get("/api/metrics").json()
    assert m["status"] == "degraded"
    assert any("kill switch" in a for a in m["alerts"])


def test_broker_down_visible_and_detail_redacted(tmp_path, store, monkeypatch):
    monkeypatch.setattr(health_mod, "_market_clock", _clock(False))
    cfg_rh = load_config("paper", overrides={
        "db_path": store.path, "broker": "robinhood_mcp",
        "universe": {"symbols": ["AAA", "SPY"], "benchmark": "SPY"}})
    store.kv_set("broker_health", {
        "adapter": "robinhood_mcp", "connected": False,
        "detail": "MCP auth failed token=abcdef0123456789ABCDEF012345 acct 934803396",
        "as_of": _now_iso()})                       # fresh → probe cache hit, no network
    h = system_health(cfg_rh, store, scheduler_alive=True)
    assert h["status"] == "degraded"
    assert h["broker"]["connected"] is False
    dumped = json.dumps(h)
    assert "934803396" not in dumped
    assert "abcdef0123456789ABCDEF012345" not in dumped


def test_api_status_backward_compatible(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    for key in ("mode", "broker", "equity", "cash", "buying_power", "day_pnl",
                "net_pnl", "kill_switches", "positions", "projection",
                "schedule", "as_of"):
        assert key in body, f"legacy /api/status key missing: {key}"


def test_api_health_backward_compatible(client, monkeypatch):
    monkeypatch.setattr(health_mod, "_market_clock", _clock(False))
    body = client.get("/api/health").json()
    for key in ("mode", "broker", "engine", "market", "data", "kill_switches",
                "pending_approvals", "readiness", "as_of"):
        assert key in body, f"legacy /api/health key missing: {key}"
    assert body["status"] in ("ok", "degraded", "stale", "error")   # new field


# ---------------- operator checker (scripts/check_health.py) ----------------

def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_health", ROOT / "scripts" / "check_health.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_checker_evaluate_prefers_server_rollup_and_falls_back():
    ck = _load_checker()
    h = _base_health()
    h["status"], h["alerts"] = "degraded", ["kill switch active: manual"]
    assert ck.evaluate(h) == ("degraded", ["kill switch active: manual"])
    legacy = _base_health()                          # pre-metrics server: no status
    assert ck.evaluate(legacy)[0] == "ok"
    legacy["market"]["open"] = True
    legacy["engine"]["heartbeat_age_s"] = 99999
    assert ck.evaluate(legacy)[0] == "stale"


def test_checker_exit_codes_down_and_malformed(capsys):
    ck = _load_checker()
    with socket.socket() as s:                       # grab a port nothing serves
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    assert ck.main(["--url", f"http://127.0.0.1:{dead}", "--timeout", "1",
                    "--json"]) == 3
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "down" and out["exit_code"] == 3


def test_checker_against_real_server(cfg, store, monkeypatch, capsys):
    monkeypatch.setattr(health_mod, "_market_clock", _clock(False))
    _fresh_spy(store)
    import uvicorn
    app = create_app(cfg, store, with_scheduler=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    try:
        ck = _load_checker()
        code = ck.main(["--url", f"http://127.0.0.1:{port}", "--json"])
        out = json.loads(capsys.readouterr().out)
        assert code == 0 and out["status"] == "ok"
        assert out["source"] == "metrics"            # new endpoint was used
        code = ck.main(["--url", f"http://127.0.0.1:{port}"])
        human = capsys.readouterr().out
        assert code == 0 and "OK" in human
    finally:
        server.should_exit = True
