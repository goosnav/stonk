"""Structured intelligence router: Codex CLI first, Claude CLI second, API last.

Every caller uses :meth:`AIClient.complete_json`; providers are adapters behind
that one port. Local clients run as one-shot, tool-free processes in an empty
temporary directory. They never receive broker credentials and never initiate
authentication. Provider failure degrades the evidence input, never trading.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

from .store import Store

MAX_PARSE_FAILURES_PER_DAY = 5
MAX_PROVIDER_OUTPUT_BYTES = 2_000_000
_LOCAL_GATE = threading.BoundedSemaphore(1)

PURPOSE_TIERS = {
    "headline_classification": "cheap",
    "news_batch": "cheap",
    "investment_memo": "advanced",
    "hypothesis": "advanced",
    "strategic_synthesis": "advanced",
}

# Provider-side structural constraint. Domain validators remain authoritative.
PURPOSE_SCHEMAS = {
    "headline_classification": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "number"}, "confidence": {"type": "number"},
            "catalyst": {"type": "string"}, "horizon_days": {"type": "integer"},
            "already_priced": {"type": "boolean"}, "summary": {"type": "string"},
        },
        "required": ["sentiment", "confidence", "catalyst", "horizon_days",
                     "already_priced", "summary"],
        "additionalProperties": False,
    },
    "news_batch": {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {
            "type": "object", "properties": {
                "id": {"type": "string"}, "symbol": {"type": "string"},
                "stance": {"type": "number"}, "confidence": {"type": "number"},
                "catalyst": {"type": "string"}, "novelty": {"type": "number"},
                "reliability": {"type": "number"}, "contradiction": {"type": "string"}},
            "required": ["id", "symbol", "stance", "confidence", "catalyst",
                         "novelty", "reliability", "contradiction"],
            "additionalProperties": False}}},
        "required": ["items"], "additionalProperties": False,
    },
    "investment_memo": {
        "type": "object", "properties": {
            name: {"type": "object", "properties": {
                "stance": {"type": "string", "enum": ["attractive", "neutral", "avoid"]},
                "confidence": {"type": "number"}, "horizon_days": {"type": "integer"},
                "thesis": {"type": "string"},
                "contrary_evidence": {"type": "array", "items": {"type": "string"}},
                "catalysts": {"type": "array", "items": {"type": "string"}},
                "thesis_breakers": {"type": "array", "items": {"type": "string"}},
                "citations": {"type": "array", "items": {"type": "object",
                    "properties": {"source_id": {"type": "string"},
                                   "claim": {"type": "string"}},
                    "required": ["source_id", "claim"], "additionalProperties": False}},
            }, "required": ["stance", "confidence", "horizon_days", "thesis",
                              "contrary_evidence", "catalysts", "thesis_breakers",
                              "citations"], "additionalProperties": False}
            for name in ("fundamental", "catalyst")},
        "required": ["fundamental", "catalyst"], "additionalProperties": False,
    },
    "strategic_synthesis": {
        "type": "object",
        "properties": {
            "thesis": {"type": "string"},
            "accepted_user_points": {"type": "array", "items": {"type": "string"}},
            "modified_user_points": {"type": "array", "items": {"type": "string"}},
            "rejected_user_points": {"type": "array", "items": {"type": "string"}},
            "favored_themes": {"type": "array", "items": {"type": "string"}},
            "favored_sectors": {"type": "array", "items": {"type": "string"}},
            "favored_symbols": {"type": "array", "items": {"type": "string"}},
            "avoided_themes": {"type": "array", "items": {"type": "string"}},
            "avoided_sectors": {"type": "array", "items": {"type": "string"}},
            "avoided_symbols": {"type": "array", "items": {"type": "string"}},
            "research_priorities": {"type": "array", "items": {"type": "string"}},
            "portfolio_tilts": {"type": "array", "items": {
                "type": "object", "properties": {
                    "symbol": {"type": "string"},
                    "direction": {"type": "string", "enum": ["favor", "avoid"]},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"}},
                "required": ["symbol", "direction", "confidence", "rationale"],
                "additionalProperties": False}},
            "horizon_days": {"type": "integer"}, "confidence": {"type": "number"},
            "contrary_evidence": {"type": "array", "items": {"type": "string"}},
            "invalidation_conditions": {"type": "array", "items": {"type": "string"}},
            "expiry": {"type": "string"}, "summary": {"type": "string"},
        },
        "required": ["thesis", "accepted_user_points", "modified_user_points",
                     "rejected_user_points", "favored_themes", "favored_sectors",
                     "favored_symbols", "avoided_themes", "avoided_sectors",
                     "avoided_symbols", "research_priorities", "portfolio_tilts",
                     "horizon_days", "confidence", "contrary_evidence",
                     "invalidation_conditions", "expiry", "summary"],
        "additionalProperties": False,
    },
}
GENERIC_SCHEMA = {"type": "object", "additionalProperties": True}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_env() -> dict[str, str]:
    """Allow client auth/config while withholding all application secrets."""
    allowed = {"HOME", "PATH", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG",
               "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "XDG_CONFIG_HOME",
               "CODEX_HOME", "CLAUDE_CONFIG_DIR", "TERM"}
    return {k: v for k, v in os.environ.items() if k in allowed}


def _provider_key(provider: str) -> str:
    names = {
        "openrouter": ("OPENROUTER_API_KEY", "AI_API_KEY"),
        "openai": ("OPENAI_API_KEY", "AI_API_KEY"),
        "anthropic": ("ANTHROPIC_API_KEY", "AI_API_KEY"),
        "custom": ("AI_API_KEY",),
    }
    return next((os.environ.get(k, "") for k in names.get(provider, ("AI_API_KEY",))
                 if os.environ.get(k)), "")


class AIClient:
    def __init__(self, cfg, store: Store):
        self.config = cfg
        self.legacy = cfg.get("ai", default={}) or {}
        self.cfg = cfg.get("intelligence", default={}) or {}
        self.store = store
        self._reserved = 0.0

    # ---------------- routing / availability ----------------
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False) or self.legacy.get("enabled", False))

    def tier_for(self, purpose: str) -> str:
        return str((self.cfg.get("purpose_tiers") or {}).get(
            purpose, PURPOSE_TIERS.get(purpose, "cheap")))

    def _models(self, block: dict, tier: str) -> str:
        models = block.get("models") or block.get("default_models") or {}
        return str(models.get(tier) or block.get("model") or "")

    def route_chain(self, purpose: str) -> list[dict]:
        """Resolve global defaults plus a complete or partial purpose override."""
        tier = self.tier_for(purpose)
        # Migration compatibility: installations/tests that explicitly disable
        # the new router but enable the legacy API block retain their exact API
        # behavior until the settings migration is accepted.
        if not self.cfg.get("enabled", False) and self.legacy.get("enabled", False):
            provider = self._legacy_api_provider()
            models = self.legacy.get("models") or {}
            model = str(models.get(purpose) or self.legacy.get("model", ""))
            return [{"channel": "api", "provider": provider, "model": model,
                     "base_url": self._legacy_base_url(provider), "tier": tier}]
        override = (self.cfg.get("purpose_routes") or {}).get(purpose) or {}
        default_provider = str(self.cfg.get(
            "default_provider", self.cfg.get("default_local_provider", "codex")))
        defaults = self.cfg.get("default_models") or {}
        primary = {"provider": default_provider,
                   "channel": "api" if default_provider in
                   {"openrouter", "openai", "anthropic", "custom", "api"} else "cli",
                   "model": str(defaults.get(tier) or
                                ("gpt-5.4-mini" if tier == "cheap" else "gpt-5.5"))}
        if override.get("primary"):
            primary.update(override["primary"])
        elif override.get("provider") or override.get("model"):
            primary.update({k: override[k] for k in ("provider", "model", "channel")
                            if k in override})
        if primary["provider"] == "api":
            primary["provider"] = str((self.cfg.get("api_fallback") or {}).get(
                "provider", "openrouter"))
            primary["channel"] = "api"

        fallback_cfg = self.cfg.get("local_fallback") or {
            "provider": "claude", "models": {"cheap": "haiku", "advanced": "opus"}}
        local_fallback = {"channel": "cli",
                          "provider": str(fallback_cfg.get("provider", "claude")),
                          "model": self._models(fallback_cfg, tier)}
        if override.get("local_fallback"):
            local_fallback.update(override["local_fallback"])

        api_cfg = self.cfg.get("api_fallback") or {}
        api_provider = str(api_cfg.get("provider") or self._legacy_api_provider())
        legacy_models = self.legacy.get("models") or {}
        api_model = self._models(api_cfg, tier) or str(
            legacy_models.get(purpose) or self.legacy.get("model", ""))
        api = {"channel": "api", "provider": api_provider, "model": api_model,
               "base_url": str(api_cfg.get("base_url") or self._legacy_base_url(api_provider))}
        if override.get("api_fallback"):
            api.update(override["api_fallback"])

        chain = [primary]
        if local_fallback.get("provider") and local_fallback != primary:
            chain.append(local_fallback)
        api_enabled = bool(api_cfg.get("enabled", True))
        if api_enabled and api.get("model") and not any(
                x["channel"] == "api" and x["provider"] == api["provider"] and
                x["model"] == api["model"] for x in chain):
            chain.append(api)
        return [{**r, "tier": tier} for r in chain if r.get("model")]

    def _legacy_api_provider(self) -> str:
        base = os.environ.get("AI_BASE_URL", self.legacy.get(
            "base_url", "https://openrouter.ai/api/v1"))
        if "api.anthropic.com" in base:
            return "anthropic"
        if "api.openai.com" in base:
            return "openai"
        return "openrouter" if "openrouter.ai" in base else "custom"

    def _legacy_base_url(self, provider: str) -> str:
        defaults = {"openrouter": "https://openrouter.ai/api/v1",
                    "openai": "https://api.openai.com/v1",
                    "anthropic": "https://api.anthropic.com/v1"}
        return os.environ.get("AI_BASE_URL", defaults.get(provider,
                              self.legacy.get("base_url", "")))

    def model_for(self, purpose: str) -> str:
        chain = self.route_chain(purpose)
        return chain[0]["model"] if chain else ""

    def available(self) -> bool:
        if not self.enabled():
            return False
        disabled_until = self.store.kv_get("ai_disabled_until")
        if disabled_until and disabled_until > datetime.now().isoformat():
            return False
        return any(self._route_available(r) for r in self.route_chain("headline_classification"))

    def _route_available(self, route: dict) -> bool:
        if route["channel"] == "cli":
            return bool(shutil.which(route["provider"])) and not self._circuit_open(route)
        return bool(_provider_key(route["provider"])) and not self._circuit_open(route)

    def _circuit_key(self, route: dict) -> str:
        digest = hashlib.sha256(f"{route['provider']}|{route['model']}".encode()).hexdigest()[:12]
        return f"intelligence_circuit_{digest}"

    def _circuit_open(self, route: dict) -> bool:
        state = self.store.kv_get(self._circuit_key(route), {}) or {}
        return bool(state.get("until") and state["until"] > _now())

    def _trip(self, route: dict, error: str) -> None:
        key = self._circuit_key(route)
        prior = self.store.kv_get(key, {}) or {}
        failures = int(prior.get("failures", 0)) + 1
        delay = min(30, 2 ** min(failures, 4))
        until = (datetime.now().astimezone() + timedelta(minutes=delay)).isoformat(
            timespec="seconds")
        self.store.kv_set(key, {"failures": failures, "until": until,
                                "error": error[:240], "at": _now()})

    def _clear_circuit(self, route: dict) -> None:
        self.store.kv_set(self._circuit_key(route), {})

    # ---------------- budget ----------------
    def _prices(self, model: str | None = None) -> dict:
        return (self.legacy.get("prices") or {}).get(model or self.legacy.get("model"),
                                                     {"input": 1.0, "output": 3.0})

    def estimate_cost(self, in_tokens: int, out_tokens: int,
                      model: str | None = None) -> float:
        p = self._prices(model)
        return in_tokens / 1e6 * p.get("input", 1.0) + out_tokens / 1e6 * p.get("output", 3.0)

    def reserve(self, est_cost: float, purpose: str = "") -> bool:
        if self.store.ai_spend_today() + self._reserved + est_cost > float(
                self.legacy.get("daily_budget_usd", 1.0)):
            return False
        if self.store.ai_spend_month() + self._reserved + est_cost > float(
                self.legacy.get("monthly_budget_usd", 40.0)):
            return False
        cap = (self.legacy.get("purpose_monthly_caps") or {}).get(purpose)
        if cap and self.store.ai_spend_month(purpose) + est_cost > float(cap):
            return False
        self._reserved += est_cost
        return True

    def _release(self, est_cost: float) -> None:
        self._reserved = max(0.0, self._reserved - est_cost)

    def _local_budget(self, tier: str) -> bool:
        limits = self.cfg.get("daily_local_limits") or {"cheap": 24, "advanced": 4}
        return self.store.ai_local_calls_today(tier) < int(limits.get(tier, 0))

    # ---------------- completion ----------------
    def complete_json(self, purpose: str, node_id: str, system: str, user: str,
                      max_out_tokens: int = 500) -> dict | None:
        if not self.enabled():
            return None
        schema = PURPOSE_SCHEMAS.get(purpose, GENERIC_SCHEMA)
        chain = self.route_chain(purpose)
        cache_key = "ai_cache_" + hashlib.sha256(json.dumps({
            "chain": chain, "purpose": purpose, "system": system, "user": user,
            "schema": schema}, sort_keys=True).encode()).hexdigest()[:24]
        cached = self.store.kv_get(cache_key)
        ttl = timedelta(hours=float(self.cfg.get(
            "cache_ttl_hours", self.legacy.get("cache_ttl_hours", 24))))
        if cached and cached.get("at", "") > (datetime.now() - ttl).isoformat():
            route = cached.get("route") or (chain[0] if chain else {})
            self.store.ai_log(route.get("model", "cache"), purpose, node_id, 0, 0, 0.0,
                              cache_hit=True, ok=True, provider=route.get("provider", "cache"),
                              channel=route.get("channel", "cache"), tier=self.tier_for(purpose))
            return cached.get("data")

        fallback_reason = ""
        for route in chain:
            if not self._route_available(route):
                fallback_reason = f"{route['provider']} unavailable"
                continue
            if route["channel"] == "cli" and not self._local_budget(route["tier"]):
                fallback_reason = f"{route['tier']} local invocation limit reached"
                continue
            started = time.monotonic()
            try:
                if route["channel"] == "cli":
                    data, usage = self._complete_cli(route, system, user, schema)
                    cost = 0.0
                else:
                    data, usage, cost = self._complete_api(
                        route, purpose, system, user, max_out_tokens)
                latency = int((time.monotonic() - started) * 1000)
                if data is None:
                    raise ValueError("provider returned no JSON object")
                self.store.ai_log(route["model"], purpose, node_id,
                                  int(usage.get("input_tokens", 0)),
                                  int(usage.get("output_tokens", 0)), cost,
                                  cache_hit=False, ok=True, provider=route["provider"],
                                  channel=route["channel"], tier=route["tier"],
                                  latency_ms=latency, fallback_reason=fallback_reason)
                self.store.kv_set(cache_key, {"at": _now(), "data": data, "route": route})
                self.store.kv_set("intelligence_last_call", {
                    "at": _now(), "purpose": purpose, "provider": route["provider"],
                    "model": route["model"], "channel": route["channel"],
                    "latency_ms": latency, "ok": True, "fallback_reason": fallback_reason})
                self._clear_circuit(route)
                return data
            except Exception as exc:  # provider boundary: fall through safely
                latency = int((time.monotonic() - started) * 1000)
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
                self._trip(route, error)
                self.store.ai_log(route["model"], purpose, node_id, 0, 0, 0.0,
                                  cache_hit=False, ok=False, provider=route["provider"],
                                  channel=route["channel"], tier=route["tier"],
                                  latency_ms=latency, fallback_reason=fallback_reason,
                                  error=error)
                self.store.audit("intelligence_route_failed", {
                    "purpose": purpose, "provider": route["provider"],
                    "model": route["model"], "channel": route["channel"],
                    "error": error})
                fallback_reason = error
        self._record_parse_failure(purpose)
        self.store.kv_set("intelligence_last_call", {
            "at": _now(), "purpose": purpose, "ok": False,
            "error": fallback_reason or "no configured provider available"})
        return None

    def _complete_cli(self, route: dict, system: str, user: str,
                      schema: dict) -> tuple[dict, dict]:
        timeout = float(self.cfg.get("request_timeout_seconds", 120))
        if not _LOCAL_GATE.acquire(timeout=min(timeout, 5)):
            raise TimeoutError("another local intelligence call is running")
        try:
            with tempfile.TemporaryDirectory(prefix="stonk-intelligence-") as td:
                schema_path = Path(td) / "schema.json"
                schema_path.write_text(json.dumps(schema))
                prompt = system.strip() + "\n\n" + user.strip()
                if route["provider"] == "codex":
                    output = Path(td) / "result.json"
                    args = [shutil.which("codex") or "codex", "exec", "-m", route["model"],
                            "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
                            "--ignore-user-config", "--output-schema", str(schema_path),
                            "--output-last-message", str(output), "--json", "-C", td, "-"]
                elif route["provider"] == "claude":
                    args = [shutil.which("claude") or "claude", "-p", "--model", route["model"],
                            "--safe-mode", "--disable-slash-commands", "--tools", "",
                            "--permission-mode", "dontAsk", "--no-session-persistence",
                            "--no-chrome", "--mcp-config", '{"mcpServers":{}}',
                            "--strict-mcp-config",
                            "--output-format", "json", "--json-schema", json.dumps(schema)]
                    output = None
                else:
                    raise ValueError(f"unsupported CLI provider {route['provider']}")
                completed = _run_process(args, prompt, td, timeout)
                raw = output.read_text() if output and output.exists() else completed.stdout
                if len(raw.encode()) > MAX_PROVIDER_OUTPUT_BYTES:
                    raise ValueError("provider output exceeded size limit")
                wrapper = _parse_json_block(raw)
                if route["provider"] == "claude" and wrapper:
                    candidate = wrapper.get("structured_output") or wrapper.get("result") or wrapper
                    if isinstance(candidate, str):
                        wrapper = _parse_json_block(candidate)
                    elif isinstance(candidate, dict):
                        wrapper = candidate
                if not isinstance(wrapper, dict):
                    raise ValueError("unparseable structured provider output")
                usage = _usage_from_output(completed.stdout)
                return wrapper, usage
        finally:
            _LOCAL_GATE.release()

    def _complete_api(self, route: dict, purpose: str, system: str, user: str,
                      max_out_tokens: int) -> tuple[dict, dict, float]:
        key = _provider_key(route["provider"])
        if not key:
            raise ValueError(f"no API key configured for {route['provider']}")
        est_in = max(1, len(system + user) // 3)
        est_cost = self.estimate_cost(est_in, max_out_tokens, route["model"])
        if not self.reserve(est_cost, purpose):
            raise RuntimeError("API budget exhausted")
        try:
            timeout = float((self.cfg.get("request_timeout_seconds", 120)
                             if self.cfg.get("enabled", False) else
                             self.legacy.get("request_timeout_seconds", 20)))
            if route["provider"] == "anthropic":
                base = route.get("base_url") or "https://api.anthropic.com/v1"
                response = httpx.post(f"{base.rstrip('/')}/messages",
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                    json={"model": route["model"], "max_tokens": max_out_tokens,
                          "system": system, "messages": [{"role": "user", "content": user}]},
                    timeout=timeout)
                response.raise_for_status(); body = response.json()
                usage = body.get("usage", {})
                text = "".join(x.get("text", "") for x in body.get("content", [])
                               if x.get("type") == "text")
                normalized = {"input_tokens": usage.get("input_tokens", est_in),
                              "output_tokens": usage.get("output_tokens", max_out_tokens)}
            else:
                base = route.get("base_url") or self._legacy_base_url(route["provider"])
                req = {"model": route["model"], "max_tokens": max_out_tokens,
                       "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}
                effort = (self.legacy.get("reasoning_effort") or {}).get(purpose)
                if effort:
                    req["reasoning"] = {"effort": effort}
                response = httpx.post(f"{base.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"}, json=req, timeout=timeout)
                response.raise_for_status(); body = response.json()
                usage = body.get("usage", {})
                text = body["choices"][0]["message"]["content"]
                normalized = {"input_tokens": usage.get("prompt_tokens", est_in),
                              "output_tokens": usage.get("completion_tokens", max_out_tokens)}
            data = _parse_json_block(text)
            cost = self.estimate_cost(normalized["input_tokens"], normalized["output_tokens"],
                                      route["model"])
            return data, normalized, round(cost, 6)
        finally:
            self._release(est_cost)

    # ---------------- status / diagnostics ----------------
    def status(self, probe_auth: bool = False) -> dict:
        purposes = sorted(set(PURPOSE_TIERS) | set((self.cfg.get("purpose_routes") or {})))
        binaries = {}
        for provider in ("codex", "claude"):
            path = shutil.which(provider)
            item = {"path": path, "installed": bool(path)}
            if probe_auth and path:
                args = [path, "login", "status"] if provider == "codex" else [path, "auth", "status"]
                try:
                    p = subprocess.run(args, capture_output=True, text=True, timeout=8,
                                       env=_safe_env(), cwd=tempfile.gettempdir())
                    raw_detail = (p.stdout or p.stderr)[-300:].strip()
                    if provider == "claude":
                        try:
                            auth = json.loads(p.stdout or "{}")
                            raw_detail = (f"logged in via {auth.get('authMethod', 'unknown')} · "
                                          f"{auth.get('subscriptionType', 'unknown')} plan")
                        except json.JSONDecodeError:
                            raw_detail = "authenticated" if p.returncode == 0 else "unavailable"
                    item.update(authenticated=p.returncode == 0,
                                detail=raw_detail)
                except Exception as exc:
                    item.update(authenticated=False, detail=str(exc)[:200])
            binaries[provider] = item
        api_cfg = self.cfg.get("api_fallback") or {}
        api_provider = str(api_cfg.get("provider") or self._legacy_api_provider())
        return {"enabled": self.enabled(), "available": self.available(),
                "default_provider": self.cfg.get(
                    "default_provider", self.cfg.get("default_local_provider", "codex")),
                "routes": {p: self.route_chain(p) for p in purposes},
                "binaries": binaries,
                "api": {"provider": api_provider, "key_set": bool(_provider_key(api_provider)),
                        "base_url": api_cfg.get("base_url") or self._legacy_base_url(api_provider)},
                "limits": self.cfg.get("daily_local_limits") or {"cheap": 24, "advanced": 4},
                "usage_today": {t: self.store.ai_local_calls_today(t)
                                for t in ("cheap", "advanced")},
                "last_call": self.store.kv_get("intelligence_last_call")}

    def _record_parse_failure(self, purpose: str) -> None:
        key = f"ai_parse_failures_{date.today().isoformat()}"
        n = (self.store.kv_get(key) or 0) + 1
        self.store.kv_set(key, n)
        if n >= MAX_PARSE_FAILURES_PER_DAY:
            until = (datetime.now() + timedelta(hours=24)).isoformat()
            self.store.kv_set("ai_disabled_until", until)
            self.store.audit("ai_auto_disabled", {"failures": n, "until": until,
                                                    "purpose": purpose})


def _run_process(args: list[str], prompt: str, cwd: str, timeout: float):
    process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, cwd=cwd,
                               env=_safe_env(), start_new_session=True)
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise TimeoutError(f"provider exceeded {timeout:g}s timeout")
    if process.returncode:
        raise RuntimeError((stderr or stdout or f"exit {process.returncode}")[-600:])
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def _usage_from_output(text: str) -> dict:
    """Best-effort only: subscription-backed usage is not assigned a dollar cost."""
    for line in reversed(text.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = row.get("usage") or row.get("token_usage") or {}
        if usage:
            return {"input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                    "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0))}
    return {}


def _parse_json_block(text: str) -> dict | None:
    for candidate in (text, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)):
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            continue
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group())
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    return None
