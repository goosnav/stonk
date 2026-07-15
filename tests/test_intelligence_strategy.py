"""Offline vertical slices for the local intelligence and strategy plane."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from specforge.ai import AIClient, _safe_env
from specforge import intelligence, strategy
from specforge.app import create_app


def test_codex_stdin_primary_then_claude_fallback(cfg, store, monkeypatch):
    cfg.data["intelligence"] = {
        "enabled": True, "default_provider": "codex",
        "default_models": {"cheap": "gpt-cheap", "advanced": "gpt-advanced"},
        "local_fallback": {"provider": "claude", "models": {
            "cheap": "haiku", "advanced": "opus"}},
        "api_fallback": {"enabled": False},
        "daily_local_limits": {"cheap": 24, "advanced": 4},
    }
    monkeypatch.setattr("specforge.ai.shutil.which", lambda provider: f"/bin/{provider}")
    calls = []

    def complete(route, system, user, schema):
        calls.append((route["provider"], route["model"], system, user))
        if route["provider"] == "codex":
            raise RuntimeError("configured alias is unavailable")
        return {"ok": True}, {"input_tokens": 2, "output_tokens": 1}

    client = AIClient(cfg, store)
    monkeypatch.setattr(client, "_complete_cli", complete)
    assert client.complete_json("strategic_synthesis", "strategy", "system", "direction") == {
        "ok": True}
    assert [(x[0], x[1]) for x in calls] == [("codex", "gpt-advanced"),
                                             ("claude", "opus")]
    last = store.kv_get("intelligence_last_call")
    assert last["provider"] == "claude" and "RuntimeError" in last["fallback_reason"]


def test_codex_command_uses_stdin_empty_workspace_and_no_app_secrets(cfg, store, monkeypatch):
    cfg.data["intelligence"] = {"enabled": True, "request_timeout_seconds": 3}
    monkeypatch.setenv("ROBINHOOD_TOKEN", "must-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("CODEX_HOME", "/tmp/codex-auth")
    observed = {}

    def run(args, prompt, cwd, timeout):
        observed.update(args=args, prompt=prompt, cwd=cwd, timeout=timeout)
        output = args[args.index("--output-last-message") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("specforge.ai.shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr("specforge.ai._run_process", run)
    result, _ = AIClient(cfg, store)._complete_cli(
        {"provider": "codex", "model": "gpt-test", "channel": "cli", "tier": "cheap"},
        "SYSTEM SECRET", "USER PAYLOAD", {"type": "object"})
    assert result == {"ok": True}
    assert observed["args"][-1] == "-" and "SYSTEM SECRET" not in observed["args"]
    assert observed["prompt"] == "SYSTEM SECRET\n\nUSER PAYLOAD"
    assert "--ephemeral" in observed["args"] and "read-only" in observed["args"]
    assert "stonk-intelligence-" in observed["cwd"]
    assert "ROBINHOOD_TOKEN" not in _safe_env() and "OPENAI_API_KEY" not in _safe_env()
    assert _safe_env()["CODEX_HOME"] == "/tmp/codex-auth"


def test_strategy_requires_synthesis_then_activation_and_caps_influence(cfg, store):
    class FakeAI:
        def complete_json(self, *args, **kwargs):
            return {
                "thesis": "Prefer broad technology only when company evidence agrees.",
                "accepted_user_points": ["research technology"],
                "modified_user_points": [], "rejected_user_points": ["buy immediately"],
                "favored_themes": ["technology"], "favored_sectors": ["technology"],
                "favored_symbols": ["SPY", "NOTREAL"], "avoided_themes": [],
                "avoided_sectors": [], "avoided_symbols": [],
                "research_priorities": ["SEC cash flow"],
                "portfolio_tilts": [{"symbol": "SPY", "direction": "favor",
                                      "confidence": .8, "rationale": "broad confirmation"}],
                "horizon_days": 21, "confidence": .8,
                "contrary_evidence": ["valuation"],
                "invalidation_conditions": ["credit stress"],
                "expiry": "2099-01-01T00:00:00+00:00", "summary": "bounded tilt",
            }

    message = strategy.submit(store, "Buy immediately; focus technology")
    proposed = strategy.analyze(cfg, store, message["id"], ai=FakeAI())
    assert strategy.contribution(cfg, store, "SPY")["value"] == 0
    assert "NOTREAL" not in proposed["payload"]["favored_symbols"]
    strategy.activate(store, proposed["id"])
    vote = strategy.contribution(cfg, store, "SPY")
    assert vote["value"] == pytest.approx(.12)  # .8 confidence × .15 hard cap
    assert vote["mandate_id"] == proposed["id"]
    assert strategy.message(store, message["id"])["text"].startswith("Buy immediately")


def test_news_job_is_durable_deduplicated_and_aggregated(cfg, store):
    cfg.data["intelligence"] = {"enabled": True}

    def fetch(symbol, limit=12):
        return [{"id": f"{symbol}-1", "published": "2099-01-01T00:00:00+00:00",
                 "title": "Material company update", "summary": "Results improved",
                 "url": f"https://example.test/{symbol}", "provider": "fixture"}]

    class FakeAI:
        def complete_json(self, purpose, node_id, system, user, max_out_tokens):
            articles = json.loads(user)["articles"]
            return {"items": [{"id": row["id"], "symbol": row["symbol"],
                                "stance": .6, "confidence": .8, "catalyst": "earnings",
                                "novelty": .7, "reliability": .9, "contradiction": ""}
                               for row in articles]}

    first = intelligence.refresh_news(cfg, store, fetcher=fetch, ai=FakeAI())
    second = intelligence.refresh_news(cfg, store, fetcher=fetch, ai=FakeAI())
    assert first["classified"] > 0 and second["inserted"] == 0
    aggregate = store.kv_get("news_intelligence")
    assert aggregate["symbols"]["SPY"]["score"] == pytest.approx(.6)
    one = intelligence.enqueue(store, "news_refresh")
    two = intelligence.enqueue(store, "news_refresh")
    assert one["id"] == two["id"]


def test_routing_and_strategy_public_interfaces_persist_without_restart(cfg, store):
    client = TestClient(create_app(cfg, store, with_scheduler=False))
    update = client.put("/api/ai/routing", json={
        "enabled": True, "default_provider": "claude",
        "default_models": {"cheap": "haiku", "advanced": "opus"},
        "local_fallback": {"provider": "codex", "models": {
            "cheap": "gpt-5.4-mini", "advanced": "gpt-5.5"}},
        "api_fallback": {"enabled": False, "provider": "openrouter",
                         "models": {"cheap": "", "advanced": ""}},
        "purpose_routes": {}, "daily_local_limits": {"cheap": 12, "advanced": 2},
        "request_timeout_seconds": 90, "cache_ttl_hours": 6,
    })
    assert update.status_code == 200
    routing = client.get("/api/ai/routing").json()
    assert routing["config"]["default_provider"] == "claude"
    queued = client.post("/api/strategy/directives/analyze",
                         json={"text": "Research durable cash flow, not hype."})
    assert queued.status_code == 200
    body = queued.json()
    assert body["message"]["status"] == "queued"
    assert body["job"]["kind"] == "strategic_synthesis"
    jobs = client.get("/api/intelligence/jobs").json()["jobs"]
    assert any(job["id"] == body["job"]["id"] for job in jobs)
