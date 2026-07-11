"""FastAPI backend + scheduler. Serves static/dashboard.html and a JSON API.

Runtime config edits from the GUI persist in kv['config_overrides'] and are
merged on every config load (so scheduled scans pick them up immediately).
Every mutation is audit-logged. Dangerous risk values are rejected by
Config.validate() exactly like file edits — the GUI has no privileged path.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .config import OVERRIDES_KEY, ConfigError, apply_override, load_config
from .data import MarketContext
from .forecast import portfolio_projection
from .risk import Governor
from .store import Store

STATIC = Path(__file__).resolve().parent.parent / "static"

# Known AI providers → OpenAI-compatible base URL. All four speak the same
# chat-completions shape (Anthropic via its OpenAI-compat endpoint), so only the
# base_url + key change; ai.py's request code is provider-agnostic. 'custom'
# lets the user paste any other OpenAI-compatible base_url.
AI_PROVIDERS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}


def current_config(store: Store, mode: str):
    overrides = store.kv_get(OVERRIDES_KEY, {}) or {}
    try:
        return load_config(mode, overrides=overrides)
    except ConfigError as e:
        # D38: config_overrides is a mode-agnostic kv blob, but a value set in
        # the GUI while live (where live.yaml's advanced_override permits e.g.
        # single-position 30%) is DANGEROUS when the same DB loads in paper
        # mode. Refuse the override and keep the SAFE committed file config —
        # never take the server down over it. This is stricter, not weaker:
        # the dangerous value is rejected exactly as validate() intends.
        # ponytail: drops the whole blob; per-key pruning if this ever bites a
        # mode where some overrides are safe and others aren't worth keeping.
        store.audit("config_override_rejected", {"mode": mode, "error": str(e)})
        return load_config(mode)


def create_app(cfg, store: Store, with_scheduler: bool = True) -> FastAPI:
    import os

    from .quotes import QuoteService
    app = FastAPI(title="Stonk Terminal", docs_url="/api/docs")
    mode = cfg.mode
    quotes = QuoteService(cfg)          # provider chain: broker→stooq→yfinance
    app.state.started_at = datetime.now().astimezone()

    def _process() -> dict:
        now = datetime.now().astimezone()
        return {"pid": os.getpid(),
                "started_at": app.state.started_at.isoformat(timespec="seconds"),
                "uptime_s": int((now - app.state.started_at).total_seconds())}

    def _scheduler_alive():
        """None = no scheduler in this process (embedded/tests) — unknown, not
        dead. False only when the scheduler object exists and stopped."""
        sched = getattr(app.state, "scheduler", None)
        return bool(sched.running) if sched else None

    def fresh_cfg():
        return current_config(store, mode)

    def broker_and_ctx():
        from .broker.base import make_broker
        c = fresh_cfg()
        ctx = MarketContext(store, c)
        broker = make_broker(c, store)
        if hasattr(broker, "set_quotes"):
            broker.set_quotes(ctx.prices())
        return c, broker, ctx

    def _positions_marked(acct, daily_prices):
        """Positions with LIVE marks where the quote chain has them; falls
        back to last daily close, labeled by source either way."""
        open_pos = [p for p in acct.positions if p.qty > 0]
        live = quotes.get([p.symbol for p in open_pos]) if open_pos else {}
        out = []
        for p in open_pos:
            q = live.get(p.symbol)
            mark = q["price"] if q else daily_prices.get(p.symbol, p.avg_cost)
            mult = 100.0 if p.asset_type == "option" else 1.0
            out.append({
                "symbol": p.option_symbol or p.symbol, "qty": p.qty,
                "asset_type": p.asset_type, "avg_cost": round(p.avg_cost, 2),
                "last": round(mark, 2),
                "quote_source": q["source"] if q else "daily close",
                "quote_as_of": q["as_of"] if q else None,
                "pnl_pct": round(mark / p.avg_cost - 1, 4) if p.avg_cost else 0,
                "pnl_usd": round((mark - p.avg_cost) * p.qty * mult, 2),
                "value": round(p.qty * mark * mult, 2)})
        return out

    # ---------------- pages ----------------
    @app.get("/")
    def index():
        return FileResponse(STATIC / "dashboard.html")

    # ---------------- live data ----------------
    @app.get("/api/quotes")
    def api_quotes(symbols: str):
        return quotes.get([s.strip() for s in symbols.split(",") if s.strip()])

    @app.get("/api/market")
    def market():
        """The Overview market strip + regime + next scan, in one call."""
        c, _, ctx = broker_and_ctx()
        from . import regime as regime_mod
        strip = quotes.get(["SPY", "QQQ", "IWM", "DIA",
                            c.get("universe", "vix_symbol", default="^VIX")])
        reg = regime_mod.classify(ctx, c)
        sched = c.get("schedule", default={}) or {}
        # structured trend inputs for the indicator tiles (D35) — the same
        # numbers regime.classify uses, not a parallel calculation
        bench = c.get("universe", "benchmark", default="SPY")
        closes = ctx.closes(bench)
        trend = {}
        if len(closes) >= 200:
            trend = {"bench": bench, "px": round(float(closes.iloc[-1]), 2),
                     "sma50": round(float(closes.rolling(50).mean().iloc[-1]), 2),
                     "sma200": round(float(closes.rolling(200).mean().iloc[-1]), 2)}
        return {"strip": strip, "regime": reg.regime, "trend": trend,
                "vix": ctx.vix(),
                "regime_evidence": reg.evidence,
                "deployment_multiplier": reg.deployment_multiplier,
                "breadth_above_50sma": ctx.breadth_above_sma(50),
                "scan_times": sched.get("scans", []),
                "post_close": sched.get("post_close"),
                "timezone": sched.get("timezone"),
                "last_cycle": (store.audit_rows(limit=1000) and next(
                    (json.loads(r["payload"]) for r in store.audit_rows(limit=200)
                     if r["event_type"] == "cycle_end"), None))}

    # ---------------- broker connect flow ----------------
    @app.get("/api/broker/status")
    def broker_status():
        c = fresh_cfg()
        probe = store.kv_get("broker_probe") or {"connected": False,
                                                 "state": "never_attempted"}
        # a 'connecting' older than the OAuth window means the thread died
        # (server restart mid-login) — say so instead of spinning forever
        if probe.get("state") == "connecting":
            started = probe.get("started_at", "")
            from datetime import timedelta
            cutoff = (datetime.now().astimezone() - timedelta(minutes=6)).isoformat()
            if not started or started < cutoff:
                probe = {**probe, "state": "interrupted",
                         "error": "previous attempt was interrupted (likely a "
                                  "server restart mid-login) — click Connect again"}
        ok, why = c.live_trading_allowed()
        return {"configured_broker": c.get("broker"), "probe": probe,
                "live_gate_ok": ok, "live_gate_reason": why}

    @app.post("/api/broker/connect")
    def broker_connect():
        """Kick off the Robinhood OAuth probe in a background thread — the
        browser opens on this machine for login. Poll /api/broker/status."""
        import threading

        def _run():
            store.kv_set("broker_probe", {"connected": False, "state": "connecting",
                                          "started_at": datetime.now().astimezone()
                                          .isoformat(timespec="seconds")})
            try:
                from .broker.robinhood_mcp import RobinhoodMCPBroker
                b = RobinhoodMCPBroker(fresh_cfg(), store)
                result = b.probe()
                result["state"] = "connected"
                store.kv_set("broker_probe", result)
                store.audit("broker_probe_ok", result)
            except Exception as e:                  # noqa: BLE001
                store.kv_set("broker_probe", {"connected": False, "state": "error",
                                              "error": str(e)[:500]})
                store.audit("broker_probe_failed", {"error": str(e)[:500]})

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "note": "OAuth window should open in your browser; "
                                    "poll /api/broker/status"}

    @app.get("/api/proposals")
    def proposals():
        return store.kv_get("promotion_proposals", [])

    @app.get("/health")
    def liveness():
        """Process liveness ONLY — touches no store/config/broker, so it
        answers fast even when the engine or DB is wedged. Readiness/why-not-
        trading lives in /api/health; the monitor contract in /api/metrics."""
        return {"ok": True, "mode": mode, **_process()}

    def _system_health():
        from .health import system_health
        sched = getattr(app.state, "scheduler", None)
        jobs = {j.id: str(j.next_run_time) for j in sched.get_jobs()} if sched else {}
        return system_health(fresh_cfg(), store, next_runs=jobs,
                             scheduler_alive=_scheduler_alive())

    @app.get("/api/health")
    def health():
        """Truth aggregator (CONTROL_CENTER_V3): mode, real broker
        connectivity, heartbeat, market clock, and — always — WHY we are not
        trading whenever we aren't. Broker probe is kv-cached 60s. `status`/
        `alerts` carry the app-health rollup (ok|degraded|stale|error)."""
        return _system_health()

    @app.get("/api/metrics")
    def metrics():
        """Monitor contract (schema stonk.metrics.v1, documented in
        RUNBOOK.md): read-only, sanitized, stable keys. status/alerts judge
        APP health — market-closed is `ok`; health.readiness answers the
        separate question 'why is it not trading right now'."""
        h = _system_health()
        src = "live" if mode == "live" else "paper"
        today = datetime.now().astimezone().date().isoformat()
        cycles_today = store.db.execute(
            "SELECT COUNT(*) n FROM audit WHERE event_type='cycle_end' "
            "AND substr(ts,1,10)=? AND json_extract(payload,'$.mode')=?",
            (today, src)).fetchone()["n"]
        errors_today = store.db.execute(
            "SELECT COUNT(*) n FROM audit WHERE event_type='scheduler_error' "
            "AND substr(ts,1,10)=?", (today,)).fetchone()["n"]
        positions_open = store.db.execute(
            "SELECT COUNT(*) n FROM positions WHERE status='open' AND mode=?",
            (src,)).fetchone()["n"]
        lq = store.db.execute(
            "SELECT ts FROM audit WHERE event_type='live_quotes' "
            "ORDER BY id DESC LIMIT 1").fetchone()
        from . import __version__
        return {"schema": "stonk.metrics.v1", "as_of": h["as_of"],
                "status": h["status"], "alerts": h["alerts"],
                "mode": mode, "version": __version__, "process": _process(),
                "cycles": {"today": cycles_today, "errors_today": errors_today,
                           "last_scan_at": h["engine"].get("last_scan_at"),
                           "last_cycle_id": h["engine"].get("last_cycle_id"),
                           "next_runs": h["engine"].get("next_runs", {})},
                "positions_open": positions_open,
                "last_quote_cycle_at": lq["ts"] if lq else None,
                "last_error": h["last_error"],
                "health": h}

    @app.get("/api/version")
    def version():
        from . import __version__
        import subprocess
        try:
            rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=5,
                                 cwd=STATIC.parent).stdout.strip()
        except Exception:                       # noqa: BLE001 — shipped w/o git
            rev = None
        return {"version": __version__, "git": rev, "mode": mode}

    # ---------------- read API ----------------
    @app.get("/api/status")
    def status():
        c, broker, ctx = broker_and_ctx()
        from . import regime as regime_mod
        acct = broker.get_account()
        gov = Governor(c, store)
        reg = regime_mod.classify(ctx, c)
        prices = ctx.prices()
        curve = store.equity_curve(c.mode if c.mode == "live" else "paper")
        # drawdown vs the governor's HWM baseline (resets on drawdown clears, D17)
        reset_d = store.kv_get("dd_peak_reset_d", "") or ""
        peak = max([r["equity"] for r in curve if r["d"] >= reset_d],
                   default=acct.equity)
        # D36: P&L is trading-only (realized from closed trades + unrealized on
        # open positions) — NEVER equity deltas, which deposits distort.
        src = "live" if c.mode == "live" else "paper"
        pos_marked = _positions_marked(acct, prices)
        realized = store.db.execute(
            "SELECT COALESCE(SUM(pnl),0) s FROM trades WHERE source=?",
            (src,)).fetchone()["s"]
        unrealized = round(sum(p["pnl_usd"] for p in pos_marked), 2)
        net_pnl = round(realized + unrealized, 2)
        today_d = datetime.now().astimezone().date().isoformat()
        prev = store.db.execute(
            "SELECT pnl FROM equity_intraday WHERE source=? AND pnl IS NOT NULL "
            "AND ts < ? ORDER BY ts DESC LIMIT 1", (src, today_d)).fetchone()
        if prev:
            day_pnl = round(net_pnl - prev["pnl"], 2)
        else:                                  # no prior marks: realized-only basis
            r_prev = store.db.execute(
                "SELECT COALESCE(SUM(pnl),0) s FROM trades WHERE source=? "
                "AND exit_date < ?", (src, today_d)).fetchone()["s"]
            day_pnl = round(net_pnl - r_prev, 2)
        if acct.equity > 0:                    # intraday mark (V4), throttled in store
            store.record_intraday_mark(acct.equity, acct.cash, src, pnl=net_pnl)
        return {
            "mode": c.mode, "broker": c.get("broker"),
            "equity": round(acct.equity, 2), "cash": round(acct.cash, 2),
            "buying_power": round(acct.buying_power, 2),
            "day_pnl": day_pnl, "net_pnl": net_pnl,
            "realized_pnl": round(realized, 2), "unrealized_pnl": unrealized,
            "drawdown_from_peak": round(1 - acct.equity / peak, 4) if peak else 0,
            "regime": reg.regime, "regime_evidence": reg.evidence,
            "deployment_multiplier": reg.deployment_multiplier,
            "kill_switches": gov.active_switches(),
            "options_unlocked": gov.options_unlocked(acct),
            "cycle_budget": round(gov.cycle_budget(acct, reg.deployment_multiplier), 2),
            "approval_mode": c.get("risk", "approval_mode"),
            "positions": pos_marked,
            "projection": portfolio_projection(store, c.mode),
            "ai_spend_today": round(store.ai_spend_today(), 4),
            "schedule": c.get("schedule", default={}),
            "as_of": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    @app.get("/api/nodes")
    def nodes():
        c = fresh_cfg()
        trades = store.trades()
        by_node: dict[str, list[float]] = {}
        for t in trades:
            for nd in json.loads(t["nodes"] or "[]"):
                by_node.setdefault(nd, []).append(t["ret"])
        # most recent degradation per node (data feed failures surface here)
        degraded: dict[str, str] = {}
        for r in store.audit_rows(limit=400):
            if r["event_type"] == "node_degraded":
                p = json.loads(r["payload"] or "{}")
                degraded.setdefault(p.get("node", ""), p.get("error", ""))
        out = []
        for node_id, nc in (c.get("nodes", default={}) or {}).items():
            rets = by_node.get(node_id, [])
            out.append({
                "id": node_id, **nc,
                "weight_multiplier": store.get_weight_multiplier(node_id),
                "n_trades": len(rets),
                "expectancy": round(sum(rets) / len(rets), 5) if rets else None,
                "win_rate": round(sum(1 for r in rets if r > 0) / len(rets), 3) if rets else None,
                "degraded": degraded.get(node_id),
            })
        return out

    @app.get("/api/freshness")
    def freshness():
        """Per-symbol bar age — the governor refuses stale data; this shows why."""
        c = fresh_cfg()
        out = []
        for sym in c.get("universe", "symbols", default=[]) + \
                [c.get("universe", "vix_symbol", default="^VIX")]:
            out.append({"symbol": sym, "latest_bar": store.latest_bar_date(sym)})
        stale_limit = c.get("risk", "stale_data_max_age_days", default=4)
        return {"symbols": out, "stale_data_max_age_days": stale_limit}

    @app.get("/api/candidates")
    def candidates():
        rows = store.db.execute(
            "SELECT payload FROM candidates WHERE cycle_id="
            "(SELECT cycle_id FROM candidates ORDER BY ts DESC LIMIT 1) "
            "ORDER BY final_score DESC LIMIT 25").fetchall()
        return [json.loads(r["payload"]) for r in rows]

    @app.get("/api/approvals")
    def approvals():
        return store.pending_approvals()

    @app.get("/api/trades")
    def trades(source: str | None = None, limit: int = 100):
        return store.trades(source=source, limit=limit)

    @app.get("/api/equity_curve")
    def equity_curve():
        c = fresh_cfg()
        return store.equity_curve(c.mode if c.mode == "live" else "paper")

    @app.get("/api/portfolio_value")
    def portfolio_value(range: str = "1M"):
        """Total portfolio value series: daily scan marks merged with throttled
        intraday marks (V4). Points: [{t, equity}] ascending."""
        src = "live" if mode == "live" else "paper"
        days = {"1D": 1, "1W": 7, "1M": 31, "ALL": 36500}.get(range, 31)
        since = datetime.now().astimezone() - timedelta(days=days)
        marks = store.intraday_marks(src, since_ts=since.isoformat())
        intra_days = {r["ts"][:10] for r in marks}
        # intraday marks supersede the daily scan mark for days they cover —
        # the daily row's synthetic 16:00 stamp must not outrank fresher marks
        daily = [{"t": r["d"] + "T16:00:00", "equity": r["equity"]}
                 for r in store.equity_curve(src)
                 if r["d"] >= since.date().isoformat() and r["d"] not in intra_days]
        intra = [{"t": r["ts"][:19], "equity": r["equity"]} for r in marks]
        pts = sorted(daily + intra, key=lambda p: p["t"])
        return {"points": pts, "range": range, "source": src,
                "current": pts[-1]["equity"] if pts else None}

    @app.get("/api/audit")
    def audit(limit: int = 100):
        return store.audit_rows(limit=limit)

    @app.get("/api/risk")
    def risk_get():
        return fresh_cfg().get("risk", default={})

    @app.get("/api/costs")
    def costs():
        c = fresh_cfg()
        ai_cfg = c.get("ai", default={})
        prices = ai_cfg.get("prices", {}).get(ai_cfg.get("model", ""), {})
        n_stocks = len([s for s in c.get("universe", "symbols", default=[])
                        if not s.startswith("^")])
        est = {}
        for node_id, nc in (c.get("nodes", default={}) or {}).items():
            if not nc.get("ai") or not nc.get("enabled") or not ai_cfg.get("enabled"):
                continue
            # rough: one classification call per stock per day, ~5 headlines each
            tokens = n_stocks * ai_cfg.get("est_tokens_per_headline", 220) * 5
            est[node_id] = round(tokens / 1e6 * (prices.get("input", 0.5)
                                                 + 0.3 * prices.get("output", 1.0)), 3)
        rows = store.db.execute(
            "SELECT day, SUM(cost_usd) c, COUNT(*) n FROM ai_ledger "
            "GROUP BY day ORDER BY day DESC LIMIT 30").fetchall()
        friction_bps = (c.get("execution", "spread_cost_bps", default=3)
                        + c.get("execution", "slippage_bps", default=5)) * 2
        return {
            "ai_enabled": bool(ai_cfg.get("enabled")),
            "ai_model": ai_cfg.get("model"),
            "ai_models": ai_cfg.get("models", {}),
            "ai_daily_budget_usd": ai_cfg.get("daily_budget_usd"),
            "ai_monthly_budget_usd": ai_cfg.get("monthly_budget_usd"),
            "ai_purpose_monthly_caps": ai_cfg.get("purpose_monthly_caps", {}),
            "ai_spend_month": round(store.ai_spend_month(), 4),
            "ai_spend_month_hypothesis": round(store.ai_spend_month("hypothesis"), 4),
            "ai_spend_today": round(store.ai_spend_today(), 4),
            "estimated_daily_by_node": est,
            "estimated_daily_total": round(sum(est.values()), 3),
            "history": [dict(r) for r in rows],
            "friction_round_trip_bps": friction_bps,
        }

    @app.get("/api/montecarlo")
    def montecarlo(horizon_days: int = 20):
        from .montecarlo import from_positions, simulate
        c, broker, ctx = broker_and_ctx()
        acct = broker.get_account()
        out = simulate(from_positions(store, ctx, acct, horizon_days))
        return out.__dict__

    # ---------------- mutations (all audit-logged) ----------------
    def _set_override(path: list[str], value):
        # validates merged result BEFORE persisting — GUI can't sneak past
        # governor (shared with steering: config.apply_override)
        apply_override(store, mode, path, value, via="gui")

    @app.post("/api/nodes/{node_id}")
    def set_node(node_id: str, body: dict):
        c = fresh_cfg()
        if node_id not in (c.get("nodes", default={}) or {}):
            raise HTTPException(404, f"unknown node {node_id}")
        if "status" in body and body["status"] not in (
                "experimental", "probation", "production", "disabled"):
            raise HTTPException(400, f"invalid status {body['status']}")
        for key in ("enabled", "weight", "status"):
            if key in body:
                try:
                    _set_override(["nodes", node_id, key], body[key])
                except ConfigError as e:
                    raise HTTPException(400, str(e)) from e
        return {"ok": True, "node": node_id}

    @app.post("/api/risk")
    def set_risk(body: dict):
        allowed = {"time_step_budget_pct", "time_step_budget_abs_cap",
                   "max_single_equity_position", "max_daily_loss", "max_weekly_loss",
                   "kill_switch_drawdown", "approval_mode",
                   "approval_notional_threshold_pct", "max_daily_new_positions",
                   "max_open_positions", "max_account_deployment", "options_enabled"}
        for k, v in body.items():
            if k not in allowed:
                raise HTTPException(400, f"risk key not editable via GUI: {k}")
            try:
                _set_override(["risk", k], v)
            except ConfigError as e:
                raise HTTPException(400, str(e)) from e
        return {"ok": True}

    @app.post("/api/ai")
    def set_ai(body: dict):
        allowed = {"enabled", "model", "daily_budget_usd", "monthly_budget_usd"}
        for k, v in body.items():
            if k == "models" and isinstance(v, dict):   # per-purpose routing (D36)
                for purpose, model_id in v.items():
                    if purpose not in ("headline_classification", "hypothesis"):
                        raise HTTPException(400, f"unknown ai purpose: {purpose}")
                    _set_override(["ai", "models", purpose], str(model_id))
                continue
            if k not in allowed:
                raise HTTPException(400, f"ai key not editable via GUI: {k}")
            _set_override(["ai", k], v)
        return {"ok": True}

    @app.get("/api/ai/provider")
    def get_ai_provider():
        """Current provider + whether a key is set. NEVER returns the key —
        only a masked last-4 hint (it is a secret)."""
        import os
        base = os.environ.get("AI_BASE_URL", "https://openrouter.ai/api/v1")
        key = os.environ.get("AI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        provider = next((p for p, u in AI_PROVIDERS.items() if u == base), "custom")
        return {"provider": provider, "base_url": base, "key_set": bool(key),
                "key_hint": ("…" + key[-4:]) if len(key) >= 4 else "",
                "providers": AI_PROVIDERS}

    @app.post("/api/ai/provider")
    def set_ai_provider(body: dict):
        """Persist provider base_url (+ optional new key) to .env and apply live.
        A blank api_key keeps the current one (so you can switch provider without
        re-typing the key)."""
        from .config import set_env_var
        provider = body.get("provider", "custom")
        base_url = (AI_PROVIDERS.get(provider) or body.get("base_url", "") or "").strip()
        if not base_url.startswith(("http://", "https://")):
            raise HTTPException(400, "base_url must be a valid http(s) URL")
        set_env_var("AI_BASE_URL", base_url)
        key = (body.get("api_key") or "").strip()
        if key:
            set_env_var("AI_API_KEY", key)
        # audit records the provider/base_url but NEVER the key value
        store.audit("ai_provider_set", {"provider": provider, "base_url": base_url,
                                        "key_changed": bool(key), "via": "gui"})
        return {"ok": True, "provider": provider, "base_url": base_url}

    @app.post("/api/approvals/{intent_id}")
    def decide(intent_id: str, body: dict):
        decision = body.get("decision")
        if decision not in ("approved", "rejected"):
            raise HTTPException(400, "decision must be approved|rejected")
        try:
            store.decide_approval(intent_id, decision)
        except ValueError as e:                    # approving an expired intent
            store.audit("approval_expired", {"intent": intent_id, "via": "gui"})
            raise HTTPException(409, str(e))
        if decision == "rejected":
            store.update_order(intent_id, status="rejected")
        store.audit("approval_decided", {"intent": intent_id, "decision": decision,
                                         "via": "gui"})
        return {"ok": True}

    @app.post("/api/kill/{name}/reset")
    def reset_kill(name: str):
        Governor(fresh_cfg(), store).reset(name)
        return {"ok": True}

    # ---------------- steering (V4/D34, non-blocking) ----------------
    @app.get("/api/steering")
    def steering_list():
        from . import steering as steering_mod
        c = fresh_cfg()
        steering_mod.sweep(c, store)           # expire-on-read
        rows = store.steering_requests(limit=30)
        return {"pending": [r for r in rows if r["status"] == "pending"],
                "recent": [r for r in rows if r["status"] != "pending"][:10]}

    @app.post("/api/steering/{sid}")
    def steering_decide(sid: str, body: dict):
        from . import steering as steering_mod
        try:
            return steering_mod.decide(fresh_cfg(), store, sid,
                                       body.get("choice", ""), via="gui")
        except (ValueError, ConfigError) as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/hypotheses")
    def hypotheses_list():
        out = {"north_star": store.active_hypothesis("north_star"),
               "short_term": store.active_hypothesis("short_term"),
               "history": store.hypotheses(limit=20)}
        return out

    @app.get("/api/pnl")
    def pnl(range: str = "1M"):
        """Net trading P&L series (D36): cumulative realized (closed trades) +
        marked unrealized going forward. Deposit/withdrawal-independent by
        construction — never derived from equity deltas."""
        src = "live" if mode == "live" else "paper"
        days = {"1D": 1, "1W": 7, "1M": 31, "ALL": 36500}.get(range, 31)
        since = datetime.now().astimezone() - timedelta(days=days)
        marks = [m for m in store.intraday_marks(src, since_ts=since.isoformat())
                 if m.get("pnl") is not None]
        intra_days = {m["ts"][:10] for m in marks}
        cum, daily = 0.0, []
        for r in store.db.execute("SELECT exit_date d, SUM(pnl) p FROM trades "
                                  "WHERE source=? GROUP BY exit_date ORDER BY d",
                                  (src,)):
            cum += r["p"]
            if r["d"] >= since.date().isoformat() and r["d"] not in intra_days:
                daily.append({"t": r["d"] + "T16:00:00", "pnl": round(cum, 2)})
        pts = sorted(daily + [{"t": m["ts"][:19], "pnl": m["pnl"]} for m in marks],
                     key=lambda p: p["t"])
        return {"points": pts, "range": range, "source": src,
                "current": pts[-1]["pnl"] if pts else 0.0}

    @app.get("/api/decisions")
    def decisions():
        """D36: what the engine is considering and what's queued — the last
        completed cycle's candidates with governor verdicts + reasons, plus
        every working (resting/relayed/awaiting-approval) order."""
        src = "live" if mode == "live" else "paper"
        r = store.db.execute(
            "SELECT * FROM audit WHERE event_type='cycle_end' "
            "AND json_extract(payload,'$.mode')=? ORDER BY id DESC LIMIT 1",
            (src,)).fetchone()
        last = dict(r) if r else None
        out = {"cycle": None, "considered": [], "working": [], "exits": {}}
        if last:
            summ = json.loads(last["payload"] or "{}")
            out["cycle"] = {"id": last["cycle_id"], "ts": last["ts"],
                            "as_of": summ.get("as_of"), "regime": summ.get("regime"),
                            "budget": summ.get("budget"),
                            "budget_used": summ.get("budget_used")}
            out["exits"] = summ.get("exits", {})
            verdicts = {}
            for r in store.audit_rows(cycle_id=last["cycle_id"], limit=500):
                if r["event_type"] == "risk_decision":
                    p = json.loads(r["payload"] or "{}")
                    verdicts[p.get("symbol")] = p
            entries = summ.get("entries", {})
            for c in store.db.execute(
                    "SELECT symbol, final_score, payload FROM candidates "
                    "WHERE cycle_id=? ORDER BY final_score DESC",
                    (last["cycle_id"],)):
                pl = json.loads(c["payload"] or "{}")
                v = verdicts.get(c["symbol"], {})
                out["considered"].append({
                    "symbol": c["symbol"], "score": round(c["final_score"], 3),
                    "thesis": (pl.get("thesis") or "")[:200],
                    "notional": pl.get("target_notional"),
                    "expected_return": pl.get("expected_return"),
                    "ci": [pl.get("ci_low"), pl.get("ci_high")],
                    "verdict": v.get("verdict"),
                    "reasons": v.get("reasons", []),
                    "result": entries.get(c["symbol"])})
        out["working"] = [dict(r) for r in store.db.execute(
            "SELECT symbol, side, qty, limit_price, notional, status, created_at "
            "FROM orders WHERE mode=? AND status IN "
            "('placed','pending_relay','relayed','pending_approval') "
            "ORDER BY created_at DESC", (src,))]
        return out

    @app.get("/api/today")
    def today():
        """D35: 'what did the system DO today' digest — scans, candidates,
        order outcomes, top veto reasons — plus the latest AI reads (news
        synopsis + active hypothesis). All from audit/orders/kv: real data or
        empty, never invented."""
        day = datetime.now().astimezone().date().isoformat()
        rows = store.db.execute(
            "SELECT event_type, payload FROM audit "
            "WHERE date(ts,'localtime')=? AND event_type IN "
            "('cycle_end','risk_decision') ORDER BY id DESC LIMIT 2000",
            (day,)).fetchall()
        scans, candidates, veto_reasons = 0, 0, {}
        for r in rows:
            p = json.loads(r["payload"] or "{}")
            if r["event_type"] == "cycle_end":
                scans += 1
                candidates += p.get("candidates", 0)
            elif p.get("verdict") == "REJECTED":
                for reason in p.get("reasons", [])[:1]:   # first reason = the blocker
                    key = reason.split("(")[0].split(":")[0].strip()[:60]
                    veto_reasons[key] = veto_reasons.get(key, 0) + 1
        by_status = {}
        for o in store.orders_today(mode="live" if mode == "live" else "paper"):
            by_status[o["status"]] = by_status.get(o["status"], 0) + 1
        st = store.active_hypothesis("short_term")
        return {
            "date": day, "scans": scans, "candidates": candidates,
            "orders": by_status,
            "top_vetoes": sorted(veto_reasons.items(), key=lambda x: -x[1])[:4],
            "news": store.kv_get("news_synopsis"),
            "fundamentals": store.kv_get("fundamentals_synopsis"),
            "hypothesis": ((st["thesis"] or "").strip().splitlines()[0][:160]
                           if st else None),
        }

    @app.get("/api/model")
    def model():
        """The model's current shape (V4): every node with its base weight ×
        learned multiplier = effective weight, measured scorecard, recent
        signal activity — plus ensemble params, regime, and hypothesis link."""
        from . import regime as regime_mod
        from .attribution import node_scorecard
        from .ensemble import s_node_weight
        c = fresh_cfg()
        ctx = MarketContext(store, c)
        reg = regime_mod.classify(ctx, c)
        sig_counts = {r["node_id"]: r["n"] for r in store.db.execute(
            "SELECT node_id, COUNT(*) n FROM signals "
            "WHERE ts >= datetime('now', '-7 days') GROUP BY node_id")}
        nodes = []
        for node_id, nc in (c.get("nodes", default={}) or {}).items():
            sc = node_scorecard(store, node_id)
            nodes.append({
                "id": node_id, "enabled": bool(nc.get("enabled")),
                "role": nc.get("role", "alpha"), "status": nc.get("status"),
                "ai": bool(nc.get("ai")) or node_id == "hypothesis",
                "base_weight": nc.get("weight", 0.0),
                "multiplier": round(store.get_weight_multiplier(node_id), 3),
                "effective_weight": round(s_node_weight(node_id, c, store,
                                                        reg.regime), 4),
                "signals_7d": sig_counts.get(node_id, 0),
                "trades_n": sc.get("n", 0),
                "expectancy": sc.get("expectancy"),
                "hit_rate": sc.get("hit_rate"),
                "per_trade_ir": sc.get("per_trade_ir"),
            })
        st = store.active_hypothesis("short_term")
        from .graph import champion as graph_champion
        graph = graph_champion(store)
        return {
            "regime": reg.regime,
            "deployment_multiplier": reg.deployment_multiplier,
            "kill_switches": sorted(Governor(c, store).active_switches()),
            "ensemble": {"min_final_score": c.get("ensemble", "min_final_score"),
                         "conflict_dispersion_penalty":
                             c.get("ensemble", "conflict_dispersion_penalty"),
                         "weight_learning": c.get("ensemble", "weight_learning")},
            "hypothesis": {
                "enabled": bool(c.get("hypothesis", "enabled", default=False)),
                "short_term": {"id": st["id"][:8], "activated_at": st["activated_at"],
                               "stances": len(json.loads(st["stances"] or "[]")),
                               "watchlist": json.loads(st["watchlist"] or "[]")}
                if st else None,
                "north_star_active": store.active_hypothesis("north_star") is not None},
            "nodes": sorted(nodes, key=lambda n: -n["effective_weight"]),
            "neural": store.kv_get("neural_status"),
            "research": store.kv_get("research_report"),
            "analog_graph": {"id": graph["id"], "status": graph["status"],
                             "metrics": graph.get("metrics", {}),
                             "live_blend": c.get("analog_graph", "live_blend", default=0)},
        }

    @app.get("/api/model/graph")
    def model_graph(symbol: str = "SPY", horizon: int = 21):
        """Actual deployed topology plus symbol-specific live activations."""
        from .graph import champion as graph_champion
        c = fresh_cfg(); graph = graph_champion(store)
        if horizon not in (5, 21):
            raise HTTPException(400, "horizon must be 5 or 21")
        snapshots = store.kv_get("graph_last_activations", {}) or {}
        return {"schema": "stonk.graph.v1", "version": graph["id"],
                "status": graph["status"], "metrics": graph.get("metrics", {}),
                "live_blend": c.get("analog_graph", "live_blend", default=0),
                "symbol": symbol.upper(), "horizon": horizon,
                "topology": graph["topology"],
                "snapshot": snapshots.get(symbol.upper())}

    @app.get("/api/research")
    def research_status():
        from .research import status
        return status(store)

    @app.get("/api/universe")
    def universe_status(tier: str = "active", limit: int = 250, offset: int = 0):
        if tier not in ("research", "active"):
            raise HTTPException(400, "tier must be research or active")
        limit = min(500, max(1, limit)); offset = max(0, offset)
        d = store.db.execute("SELECT MAX(as_of) d FROM universe_membership WHERE tier=?",
                             (tier,)).fetchone()["d"]
        rows = [] if not d else [dict(r) for r in store.db.execute(
            "SELECT u.*,i.name,i.exchange,i.security_type FROM universe_membership u "
            "LEFT JOIN instruments i ON i.symbol=u.symbol WHERE u.as_of=? AND u.tier=? "
            "ORDER BY u.rank LIMIT ? OFFSET ?", (d, tier, limit, offset))]
        for row in rows:
            row["metrics"] = json.loads(row["metrics"] or "{}")
        return {"as_of": d, "tier": tier, "limit": limit, "offset": offset,
                "rows": rows}

    @app.get("/api/engine")
    def engine_status():
        """D39: the state-machine view. Current phase (kv stamped by run_cycle
        at every transition), the last cycle's phase trace, and cadence."""
        from .health import _market_clock
        c = fresh_cfg()
        sched = getattr(app.state, "scheduler", None)
        jobs = {j.id: str(j.next_run_time) for j in sched.get_jobs()} if sched else {}
        state = store.kv_get("engine_state") or {}
        if state.get("mode") != mode:
            state = {"phase": "awaiting_cycle",
                     "detail": f"no {mode} cycle in this process yet; scheduler is armed",
                     "at": None, "trace": [], "mode": mode}
        return {"state": state,
                "market": _market_clock(),
                "interval_minutes": c.get("schedule", "scan_interval_minutes"),
                "scan_times": c.get("schedule", "scans", default=[]),
                "next_runs": jobs,
                "heartbeat": store.kv_get("heartbeat")}

    @app.post("/api/scan")
    def manual_scan():
        from .engine import run_cycle
        from .health import write_heartbeat
        summary = run_cycle(fresh_cfg(), store)
        if not summary.get("skipped"):
            write_heartbeat(store, summary["cycle_id"], mode, source="serve")
        return summary

    # ---------------- scheduler ----------------
    if with_scheduler:
        _start_scheduler(app, store, mode)
    return app


def _commit_reports(store: Store, root: Path | None = None) -> None:
    """Nightly git snapshot of dev/reports (ROADMAP Sprint D). Best-effort:
    skips silently when there is nothing new or git is unavailable."""
    import subprocess
    root = root or Path(__file__).resolve().parent.parent
    if not (root / ".git").is_dir():
        return
    try:
        subprocess.run(["git", "add", "dev/reports"], cwd=root,
                       timeout=15, capture_output=True, check=True)
        dirty = subprocess.run(["git", "diff", "--cached", "--quiet",
                                "--", "dev/reports"], cwd=root,
                               timeout=15, capture_output=True)
        if dirty.returncode == 0:
            return                              # nothing new under dev/reports
        subprocess.run(["git", "commit", "-m",
                        "chore: nightly dev/reports snapshot",
                        "--", "dev/reports"], cwd=root,
                       timeout=15, capture_output=True, check=True)
        store.audit("reports_committed", {})
    except Exception as e:                      # noqa: BLE001
        store.audit("scheduler_error", {"job": "commit_reports",
                                        "error": str(e)})


def _start_scheduler(app: FastAPI, store: Store, mode: str) -> None:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    def scan_job():
        from .engine import run_cycle
        cfg = current_config(store, mode)
        try:
            summary = run_cycle(cfg, store)
            if summary.get("skipped"):          # another cycle holds the lock
                return
            from .health import write_heartbeat
            write_heartbeat(store, summary["cycle_id"], cfg.mode, source="serve")
            if summary.get("kill_switches"):
                _notify("Stonk Terminal kill switch",
                        f"active: {', '.join(summary['kill_switches'])} — "
                        f"open the dashboard")
            # D29: intents were silently expiring (D25) because nothing told
            # the human they'd been queued — surface them at queue time
            pending = sum(1 for s in summary.get("entries", {}).values()
                          if s == "pending_approval")
            if pending:
                _notify("Stonk Terminal: trades await approval",
                        f"{pending} intent(s) pending — approve in the "
                        f"dashboard before they expire")
        except Exception as e:                      # noqa: BLE001
            store.audit("scheduler_error", {"error": str(e)})
            _notify("Stonk Terminal scan FAILED", str(e)[:120])
            print(f"[scheduler] scan FAILED: {e}")

    def _notify(title: str, msg: str) -> None:
        """Best-effort local desktop notification (macOS); silent elsewhere."""
        import subprocess
        try:
            subprocess.run(["osascript", "-e",
                            f'display notification "{msg}" with title "{title}"'],
                           timeout=5, capture_output=True)
        except Exception:                           # noqa: BLE001
            pass

    def post_close_job():
        """Mark-to-market + attribution: the self-improvement heartbeat."""
        from .attribution import propose_promotions, update_weights
        from .engine import run_cycle
        cfg = current_config(store, mode)
        try:
            run_cycle(cfg, store)              # final scan marks equity + exits
            update_weights(cfg, store)
            # hypothesis upkeep (V4/D34): regen/review through steering; a
            # failure here must never break attribution or the backup below
            try:
                from . import steering as steering_mod
                hs = steering_mod.maintain(cfg, store)
                if hs.get("short_term_proposed") or hs.get("north_star_proposed"):
                    _notify("Stonk Terminal: hypothesis proposal",
                            "a strategic choice awaits (or auto-applies at "
                            "expiry) — see the dashboard")
            except Exception as e:              # noqa: BLE001
                store.audit("scheduler_error", {"job": "hypothesis", "error": str(e)})
            proposals = propose_promotions(cfg, store)
            if proposals:
                store.audit("promotion_proposals", proposals)
                store.kv_set("promotion_proposals", proposals)
                _notify("Stonk Terminal", f"{len(proposals)} node promotion proposal(s) "
                                     f"await your review")
            _backup_db(store)
            _commit_reports(store)
        except Exception as e:                  # noqa: BLE001
            store.audit("scheduler_error", {"job": "post_close", "error": str(e)})
            print(f"[scheduler] post-close FAILED: {e}")

    def _backup_db(store: Store) -> None:
        """Nightly sqlite online backup; keep the newest 14."""
        from datetime import date as _date
        bdir = Path(store.path).parent / "backups"
        bdir.mkdir(exist_ok=True)
        dest = bdir / f"specforge_{_date.today().isoformat()}.db"
        import sqlite3 as _sq
        with _sq.connect(dest) as out:
            store.db.backup(out)
        for old in sorted(bdir.glob("specforge_*.db"))[:-14]:
            old.unlink()
        store.audit("db_backup", {"path": str(dest)})

    cfg = current_config(store, mode)
    tz = cfg.get("schedule", "timezone", default="America/New_York")
    sched = BackgroundScheduler(timezone=tz)
    # misfire_grace_time: a scan fired up to 30 min late (laptop wake) still
    # runs; later than that APScheduler drops it and the listener below alerts.
    for hhmm in cfg.get("schedule", "scans", default=[]):
        h, m = hhmm.split(":")
        sched.add_job(scan_job, CronTrigger(day_of_week="mon-fri", hour=int(h),
                                            minute=int(m), timezone=tz),
                      misfire_grace_time=1800, id=f"scan_{hhmm}")
    pc = cfg.get("schedule", "post_close", default="16:30")
    h, m = pc.split(":")
    sched.add_job(post_close_job, CronTrigger(day_of_week="mon-fri", hour=int(h),
                                              minute=int(m), timezone=tz),
                  misfire_grace_time=1800, id="post_close")

    from apscheduler.triggers.interval import IntervalTrigger

    # D39 continuous heartbeat: full cycle every N minutes while the market is
    # open (the engine.py cycle lock makes overlap with the cron scans a no-op)
    # and a visible "sleeping" stamp when it's closed — the state machine is
    # always demonstrably alive, never a 3-hour black box.
    iv = cfg.get("schedule", "scan_interval_minutes")
    if iv:
        def interval_job():
            from .health import _market_clock
            mkt = _market_clock()
            if mkt["open"]:
                scan_job()
            else:
                from datetime import datetime as _dt
                store.kv_set("engine_state", {
                    "phase": "sleeping", "cycle_id": None, "trace": [],
                    "mode": mode,
                    "detail": f"market {mkt['session']} ({mkt['et']}) — engine "
                              f"alive, cycles resume at the next open",
                    "at": _dt.now().astimezone().isoformat(timespec="seconds")})

        sched.add_job(
            interval_job, IntervalTrigger(minutes=int(iv)),
            next_run_time=datetime.now().astimezone() + timedelta(seconds=3),
            misfire_grace_time=120, id="scan_interval")

    # Closed-market research plane: one bounded, idempotent task per tick.
    # It trains challengers, never mutates live champions in place.
    def research_job():
        from .health import _market_clock
        c = current_config(store, mode)
        if _market_clock()["open"]:
            return
        try:
            from .research import run_next
            run_next(c, store)
        except Exception as e:                  # noqa: BLE001
            store.audit("scheduler_error", {"job": "research",
                                            "error": str(e)[:300]})

    sched.add_job(research_job, IntervalTrigger(
        minutes=int(cfg.get("research", "interval_minutes", default=15))),
        next_run_time=datetime.now().astimezone() + timedelta(seconds=8),
        misfire_grace_time=600, id="research")

    # D40: the weekend research loop — schedule slot existed since V1 but was
    # never wired. A 2y walk-forward backtest of the CURRENT config every
    # Saturday: config drift vs measured edge surfaces within a week.
    wr = cfg.get("schedule", "weekend_research", default="SAT 10:00")

    def weekend_research_job():
        c = current_config(store, mode)
        try:
            from .backtest import run_backtest
            rep = run_backtest(c, years=2, tag="weekly_research")
            slim = {"window": rep.get("window"), "n_trades": rep.get("n_trades"),
                    "win_rate": rep.get("win_rate"),
                    "overall": rep.get("overall"),
                    "out_of_sample_30pct": rep.get("out_of_sample_30pct"),
                    "benchmark_buy_hold_return": rep.get("benchmark_buy_hold_return"),
                    "ran_at": datetime.now().astimezone().isoformat(timespec="seconds")}
            store.kv_set("research_report", slim)
            store.audit("weekend_research", slim)
            print(f"[scheduler] weekend research: {slim}")
        except Exception as e:                  # noqa: BLE001
            store.audit("scheduler_error", {"job": "weekend_research",
                                            "error": str(e)[:300]})

    try:
        wd, hhmm = wr.split()
        h2, m2 = hhmm.split(":")
        sched.add_job(weekend_research_job,
                      CronTrigger(day_of_week=wd.lower()[:3], hour=int(h2),
                                  minute=int(m2), timezone=tz),
                      misfire_grace_time=6 * 3600, id="weekend_research")
    except ValueError:
        store.audit("scheduler_error", {"job": "weekend_research",
                                        "error": f"bad schedule string {wr!r}"})

    def _on_missed(event):
        """Watchdog (ROADMAP Sprint D): a scheduled scan was silently skipped
        (machine asleep past the grace window) — make it loud."""
        store.audit("scheduler_missed", {"scheduled": str(event.scheduled_run_time)})
        _notify("Stonk Terminal missed a scan",
                f"scheduled {event.scheduled_run_time:%H:%M} never ran — "
                f"machine asleep?")

    from apscheduler.events import EVENT_JOB_MISSED
    sched.add_listener(_on_missed, EVENT_JOB_MISSED)
    sched.start()
    app.state.scheduler = sched
