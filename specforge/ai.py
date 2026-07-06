"""AI enrichment layer — OpenRouter-compatible chat client with a hard daily
budget (D7). AI is optional garnish: every caller must have a deterministic
fallback, and AI output only ever becomes a *structured input* to deterministic
scoring — it never sizes or places anything.

Reserve-then-commit: a task pre-estimates its full cost and reserves it against
(today's actual spend + outstanding reservations). If the whole task doesn't
fit, it is skipped cleanly — never started-then-abandoned mid-budget.

Parse-failure policy (deviation from AGENTS.md §14.2 "kill switch", logged in
dev/DECISIONS.md D14): repeated unparseable AI output disables AI for the rest
of the day (kv 'ai_disabled_until') instead of halting trading — the
deterministic pipeline is unaffected by a broken enrichment feed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime, timedelta

import httpx

from .store import Store

MAX_PARSE_FAILURES_PER_DAY = 5


class AIClient:
    def __init__(self, cfg, store: Store):
        self.cfg = cfg.get("ai", default={}) or {}
        self.store = store
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = os.environ.get("AI_BASE_URL",
                                       self.cfg.get("base_url",
                                                    "https://openrouter.ai/api/v1"))
        self.model = self.cfg.get("model", "deepseek/deepseek-chat-v3-0324")
        self._reserved = 0.0            # in-flight reservations (single process)

    # ---------------- availability & budget ----------------
    def available(self) -> bool:
        if not self.cfg.get("enabled") or not self.api_key:
            return False
        disabled_until = self.store.kv_get("ai_disabled_until")
        if disabled_until and disabled_until > datetime.now().isoformat():
            return False
        return True

    def _prices(self) -> dict:
        return (self.cfg.get("prices") or {}).get(self.model,
                                                  {"input": 1.0, "output": 3.0})

    def estimate_cost(self, in_tokens: int, out_tokens: int) -> float:
        p = self._prices()
        return in_tokens / 1e6 * p.get("input", 1.0) + out_tokens / 1e6 * p.get("output", 3.0)

    def reserve(self, est_cost: float) -> bool:
        budget = float(self.cfg.get("daily_budget_usd", 1.0))
        if self.store.ai_spend_today() + self._reserved + est_cost > budget:
            return False
        self._reserved += est_cost
        return True

    def _release(self, est_cost: float) -> None:
        self._reserved = max(0.0, self._reserved - est_cost)

    # ---------------- the one entry point ----------------
    def complete_json(self, purpose: str, node_id: str, system: str, user: str,
                      max_out_tokens: int = 500) -> dict | None:
        """Chat call that must return a JSON object. None ⇒ caller falls back
        to deterministic behavior (budget exhausted, disabled, or parse fail)."""
        if not self.available():
            return None
        cache_key = "ai_cache_" + hashlib.sha256(
            f"{self.model}|{system}|{user}".encode()).hexdigest()[:24]
        cached = self.store.kv_get(cache_key)
        ttl = timedelta(hours=self.cfg.get("cache_ttl_hours", 24))
        if cached and cached["at"] > (datetime.now() - ttl).isoformat():
            self.store.ai_log(self.model, purpose, node_id, 0, 0, 0.0,
                              cache_hit=True, ok=True)
            return cached["data"]

        est_in = len(system + user) // 3            # ~3 chars/token, conservative
        est_cost = self.estimate_cost(est_in, max_out_tokens)
        if not self.reserve(est_cost):
            self.store.audit("ai_budget_skip", {"purpose": purpose,
                                                "est_cost": round(est_cost, 5)})
            return None
        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "max_tokens": max_out_tokens,
                      "response_format": {"type": "json_object"},
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
                timeout=60)
            r.raise_for_status()
            body = r.json()
            usage = body.get("usage", {})
            cost = self.estimate_cost(usage.get("prompt_tokens", est_in),
                                      usage.get("completion_tokens", max_out_tokens))
            text = body["choices"][0]["message"]["content"]
            data = _parse_json_block(text)
            ok = data is not None
            self.store.ai_log(self.model, purpose, node_id,
                              usage.get("prompt_tokens", 0),
                              usage.get("completion_tokens", 0),
                              round(cost, 6), cache_hit=False, ok=ok)
            if not ok:
                self._record_parse_failure(purpose)
                return None
            self.store.kv_set(cache_key, {"at": datetime.now().isoformat(),
                                          "data": data})
            return data
        except Exception as e:                      # noqa: BLE001
            self.store.ai_log(self.model, purpose, node_id, 0, 0, 0.0,
                              cache_hit=False, ok=False)
            self.store.audit("ai_error", {"purpose": purpose, "error": str(e)})
            return None
        finally:
            self._release(est_cost)

    def _record_parse_failure(self, purpose: str) -> None:
        key = f"ai_parse_failures_{date.today().isoformat()}"
        n = (self.store.kv_get(key) or 0) + 1
        self.store.kv_set(key, n)
        if n >= MAX_PARSE_FAILURES_PER_DAY:
            until = (datetime.now() + timedelta(hours=24)).isoformat()
            self.store.kv_set("ai_disabled_until", until)
            self.store.audit("ai_auto_disabled",
                             {"failures": n, "until": until, "purpose": purpose})


def _parse_json_block(text: str) -> dict | None:
    """Strict-ish: accept a bare JSON object or one inside a ```json fence.
    Anything else is discarded (AGENTS.md §34.15)."""
    for candidate in (text, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```",
                                        text, re.S)):
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group())
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None
