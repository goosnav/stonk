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
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

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


def _public_probe(value):
    """Return broker-connect state without account identifiers or secrets."""
    if isinstance(value, list):
        return [_public_probe(item) for item in value]
    if isinstance(value, str):
        from .health import _redact
        return _redact(value)
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        lowered = str(key).lower()
        if any(token in lowered for token in ("whitelist", "credential", "token", "secret")):
            continue
        if "account" in lowered and isinstance(item, (str, int)):
            text = str(item)
            out[key] = f"••••{text[-4:]}" if text else ""
        else:
            out[key] = _public_probe(item)
    return out


def _job_summary(job: dict) -> dict:
    """Compact polling representation; detailed job endpoints retain results."""
    allowed = ("id", "kind", "status", "state", "priority", "resource_class",
               "requested_at", "started_at", "completed_at", "updated_at",
               "heartbeat_at", "lease_expires_at", "next_retry_at", "wait_reason",
               "error", "depends_on_id", "progress")
    out = {key: job.get(key) for key in allowed if key in job}
    if out.get("error"):
        out["error"] = str(out["error"])[:240]
    return out


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
    import threading

    from .quotes import QuoteService
    app = FastAPI(title="Stonk Terminal", docs_url="/api/docs")
    # A loopback service is still a control plane.  Never derive trust from the
    # request Host header: that makes DNS rebinding look same-origin.
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
    )
    mode = cfg.mode
    quotes = QuoteService(cfg)          # provider chain: broker→stooq→yfinance
    app.state.started_at = datetime.now().astimezone()
    app.state.account_cache = {"at": None, "account": None, "persisted": False}
    persisted_account = store.kv_get("account_snapshot_cache") or {}
    try:
        from .models import AccountState, Position
        raw_account = persisted_account["account"]
        app.state.account_cache = {
            "at": datetime.fromisoformat(persisted_account["cached_at"]),
            "persisted": True,
            "account": AccountState(
                equity=float(raw_account["equity"]), cash=float(raw_account["cash"]),
                buying_power=float(raw_account["buying_power"]),
                positions=[Position(**p) for p in raw_account.get("positions", [])],
                as_of=raw_account["as_of"])}
    except (KeyError, TypeError, ValueError):
        pass
    app.state.account_cache_lock = threading.Lock()
    app.state.broker_connect_lock = threading.Lock()

    @app.middleware("http")
    async def same_origin_mutations(request: Request, call_next):
        """Block browser cross-site writes to the loopback control plane.

        Non-browser CLI calls normally omit Origin and remain supported. A
        browser that supplies Origin must itself be a loopback origin;
        restrictive CORS headers alone do not stop form/fetch CSRF.
        """
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin:
                parsed = urlparse(origin)
                if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
                        "127.0.0.1", "localhost", "::1", "testserver"}:
                    return JSONResponse({"detail": "cross-origin mutation refused"},
                                        status_code=403)
        return await call_next(request)

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

    def cached_account(broker):
        from .health import _market_clock
        ttl = 10 if _market_clock()["open"] else 60
        with app.state.account_cache_lock:
            cached = app.state.account_cache
            age = ((datetime.now().astimezone() - cached["at"]).total_seconds()
                   if cached["at"] else float("inf"))
            hit = cached["account"] is not None and age < ttl and \
                not cached.get("persisted", False)
            stale = False
            if not hit:
                try:
                    account = broker.get_account()
                    cached.update(account=account, at=datetime.now().astimezone(),
                                  persisted=False)
                    from .models import to_json_dict
                    store.kv_set("account_snapshot_cache", {
                        "cached_at": cached["at"].isoformat(timespec="seconds"),
                        "account": to_json_dict(account)})
                except Exception:
                    if cached["account"] is None:
                        raise
                    stale = True
            return cached["account"], cached["at"], hit, stale

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
        broker_health = store.kv_get("broker_health") or {}
        if broker_health.get("adapter") == c.get("broker") and \
                broker_health.get("connected"):
            # A successful automatic read is stronger, newer evidence than an
            # old explicit Connect error. Show the recovered state so the UI
            # does not invite another unnecessary OAuth attempt.
            probe = {"connected": True, "state": "connected",
                     "as_of": broker_health.get("as_of"),
                     "note": "session verified by automatic broker read"}
            store.kv_set("broker_probe", probe)
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
        return {"configured_broker": c.get("broker"), "probe": _public_probe(probe),
                "live_gate_ok": ok, "live_gate_reason": why}

    @app.post("/api/broker/connect")
    def broker_connect():
        """Kick off the Robinhood OAuth probe in a background thread — the
        browser opens on this machine for login. Poll /api/broker/status."""
        c = fresh_cfg()
        if c.get("broker") != "robinhood_mcp":
            raise HTTPException(409, "Robinhood connect is unavailable for this broker/mode")
        probe = store.kv_get("broker_probe") or {}
        if probe.get("connected") and probe.get("state") == "connected":
            connected_at = (probe.get("probed_at") or probe.get("as_of") or
                            probe.get("finished_at"))
            try:
                if datetime.now().astimezone() - datetime.fromisoformat(connected_at) < \
                        timedelta(seconds=60):
                    return {"ok": True, "state": "already_connected",
                            "note": "A fresh read already verified this session"}
            except (TypeError, ValueError):
                pass
        # Persist a short cooldown as well as the process-local lock.  This
        # prevents repeated clicks/restarts from opening an OAuth tab storm.
        try:
            last_start = datetime.fromisoformat(probe.get("started_at", ""))
            if datetime.now().astimezone() - last_start < timedelta(seconds=30):
                return {"ok": True, "state": "already_connecting",
                        "note": "A recent Robinhood login attempt is still settling"}
        except (TypeError, ValueError):
            pass
        if not app.state.broker_connect_lock.acquire(blocking=False):
            return {"ok": True, "state": "already_connecting",
                    "note": "One Robinhood login is already in progress"}

        def _run():
            attempt_started = datetime.now().astimezone().isoformat(timespec="seconds")
            store.kv_set("broker_probe", {"connected": False, "state": "connecting",
                                          "started_at": attempt_started})
            try:
                from .broker.robinhood_mcp import RobinhoodMCPBroker
                b = RobinhoodMCPBroker(fresh_cfg(), store, interactive_auth=True)
                result = _public_probe(b.probe())
                result["state"] = "connected"
                store.kv_set("broker_probe", result)
                store.audit("broker_probe_ok", result)
            except Exception as e:                  # noqa: BLE001
                from .health import _redact
                error = _redact(str(e))[:500]
                store.kv_set("broker_probe", {"connected": False, "state": "error",
                                              "error": error,
                                              "started_at": attempt_started,
                                              "finished_at": datetime.now().astimezone()
                                              .isoformat(timespec="seconds")})
                store.audit("broker_probe_failed", {"error": error})
            finally:
                app.state.broker_connect_lock.release()

        threading.Thread(target=_run, name="robinhood-oauth", daemon=True).start()
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

    @app.get("/health/live")
    def health_live():
        return liveness()

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

    @app.get("/health/ready")
    def health_ready():
        h = _system_health()
        ready = h.get("status") not in {"error", "offline"}
        payload = {"ready": ready, "mode": mode, "status": h.get("status"),
                   "scheduler_alive": h.get("engine", {}).get("scheduler_alive"),
                   **_process()}
        return payload if ready else JSONResponse(payload, status_code=503)

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
        acct, account_at, account_cached, account_stale = cached_account(broker)
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
            "account_as_of": account_at.isoformat(timespec="seconds"),
            "account_stale": account_stale,
            "account_source": "stale_cache" if account_stale else
                              ("cache" if account_cached else "broker"),
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
        from .ensemble import s_node_multiplier
        c = fresh_cfg()
        src = "live" if c.mode == "live" else "paper"
        trades = store.trades(source=src)
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
            status = str(nc.get("status", "experimental"))
            floor_key = ("experimental_floor" if status == "experimental"
                         else "production_floor")
            floor = float(c.get("ensemble", "weight_learning", floor_key,
                                default=.25 if status == "experimental" else .50))
            learned_multiplier = s_node_multiplier(node_id, c, store)
            out.append({
                "id": node_id, **nc,
                "stored_weight_multiplier": store.get_weight_multiplier(node_id),
                "learned_weight_multiplier": learned_multiplier,
                "weight_multiplier": (learned_multiplier
                                      if nc.get("enabled") else 0.0),
                "automated_floor": floor,
                "human_disabled": not bool(nc.get("enabled")),
                "learning_state": ("human_disabled" if not nc.get("enabled") else
                                   "deemphasized" if learned_multiplier < .999 else
                                   "full"),
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
    def trades(source: str | None = None, limit: int = 100,
               evidence_version: str | None = None):
        if source not in (None, "live", "paper", "backtest", "backtest_validated"):
            raise HTTPException(400, "invalid trade source")
        return store.trades(source=source, limit=min(1000, max(1, limit)),
                            evidence_version=evidence_version)

    @app.get("/api/evidence/{symbol}")
    def company_evidence(symbol: str):
        from .evidence import ai_budget_status, latest_dossier
        c = fresh_cfg()
        dossier = latest_dossier(store, symbol)
        return {"symbol": symbol.upper(), "dossier": dossier,
                "available": bool(dossier and dossier.get("status") == "ready"),
                "budget": ai_budget_status(c, store)}

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
        from .ai import AIClient
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
            "ai_enabled": AIClient(c, store).enabled(),
            "ai_available": AIClient(c, store).available(),
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
        if not 1 <= horizon_days <= 252:
            raise HTTPException(400, "horizon_days must be between 1 and 252")
        c, broker, ctx = broker_and_ctx()
        acct, _, _, _ = cached_account(broker)
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
                    if purpose not in ("headline_classification", "hypothesis",
                                       "investment_memo"):
                        raise HTTPException(400, f"unknown ai purpose: {purpose}")
                    _set_override(["ai", "models", purpose], str(model_id))
                    tier = "cheap" if purpose == "headline_classification" else "advanced"
                    _set_override(["intelligence", "api_fallback", "models", tier],
                                  str(model_id))
                continue
            if k not in allowed:
                raise HTTPException(400, f"ai key not editable via GUI: {k}")
            _set_override(["ai", k], v)
            if k == "enabled":
                # Backward-compatible control: the original /api/ai switch now
                # controls the router too instead of appearing to save while
                # Codex/Claude continue running.
                _set_override(["intelligence", "enabled"], bool(v))
            elif k == "model":
                _set_override(["intelligence", "api_fallback", "models", "cheap"], str(v))
        return {"ok": True}

    @app.get("/api/ai/routing")
    def ai_routing():
        from .ai import AIClient
        c = fresh_cfg()
        return {"config": c.get("intelligence", default={}),
                "status": AIClient(c, store).status(probe_auth=False)}

    @app.put("/api/ai/routing")
    def set_ai_routing(body: dict):
        allowed = {"enabled", "default_provider", "default_models", "local_fallback",
                   "api_fallback", "purpose_tiers", "purpose_routes",
                   "daily_local_limits", "request_timeout_seconds", "cache_ttl_hours"}
        unknown = set(body) - allowed
        if unknown:
            raise HTTPException(400, f"unknown intelligence setting(s): {sorted(unknown)}")
        providers = {"codex", "claude", "api", "openrouter", "openai",
                     "anthropic", "custom"}
        provider = body.get("default_provider")
        if provider is not None and provider not in providers:
            raise HTTPException(400, f"unsupported default provider {provider}")
        fallback = body.get("local_fallback") or {}
        if fallback.get("provider") and fallback["provider"] not in {"codex", "claude"}:
            raise HTTPException(400, "local fallback must be codex or claude")
        api = body.get("api_fallback") or {}
        if api.get("provider") and api["provider"] not in {
                "openrouter", "openai", "anthropic", "custom"}:
            raise HTTPException(400, "unsupported API fallback provider")
        for purpose, route in (body.get("purpose_routes") or {}).items():
            if purpose not in {"headline_classification", "news_batch", "investment_memo",
                                "hypothesis", "strategic_synthesis"}:
                raise HTTPException(400, f"unknown intelligence purpose {purpose}")
            blocks = [route.get("primary") or route]
            if route.get("local_fallback"): blocks.append(route["local_fallback"])
            if route.get("api_fallback"): blocks.append(route["api_fallback"])
            for block in blocks:
                if block.get("provider") and block["provider"] not in providers:
                    raise HTTPException(400, f"unsupported provider in {purpose}")
        limits = body.get("daily_local_limits") or {}
        if any(not isinstance(v, int) or v < 0 or v > 1000 for v in limits.values()):
            raise HTTPException(400, "daily local limits must be integers from 0 to 1000")
        timeout = body.get("request_timeout_seconds")
        if timeout is not None and (not isinstance(timeout, (int, float)) or not 5 <= timeout <= 1800):
            raise HTTPException(400, "request timeout must be 5-1800 seconds")
        for key, value in body.items():
            try:
                _set_override(["intelligence", key], value)
            except ConfigError as exc:
                raise HTTPException(400, str(exc)) from exc
        store.audit("intelligence_routing_updated", {
            "keys": sorted(body), "default_provider": body.get("default_provider"),
            "api_provider": api.get("provider")})
        from .ai import AIClient
        return {"ok": True, "status": AIClient(fresh_cfg(), store).status(False)}

    @app.get("/api/ai/health")
    def ai_health():
        from .ai import AIClient
        return AIClient(fresh_cfg(), store).status(probe_auth=False)

    @app.post("/api/ai/doctor")
    def ai_doctor():
        from .ai import AIClient
        return AIClient(fresh_cfg(), store).status(probe_auth=True)

    @app.post("/api/ai/test")
    def ai_test(body: dict):
        from .ai import AIClient
        from .config import Config
        import copy
        purpose = str(body.get("purpose", "headline_classification"))
        if purpose not in {"headline_classification", "investment_memo", "hypothesis",
                           "strategic_synthesis", "news_batch"}:
            raise HTTPException(400, "unknown intelligence purpose")
        # Use a purpose-compatible harmless schema test; never pass market data
        # or contact the broker from a provider diagnostic.
        test_purpose = "headline_classification"
        c = fresh_cfg()
        requested_provider = str(body.get("provider", "")).strip().lower()
        if requested_provider:
            if requested_provider not in {"codex", "claude", "openrouter", "openai",
                                           "anthropic", "custom"}:
                raise HTTPException(400, "unsupported provider")
            # Provider-only diagnostics intentionally disable fallbacks so the
            # operator learns whether this exact provider/model works.
            data = copy.deepcopy(c.data)
            intel = data.setdefault("intelligence", {})
            intel["enabled"] = True
            model = str(body.get("model", "")).strip() or (
                (intel.get("default_models") or {}).get("cheap") or "gpt-5.4-mini")
            intel["purpose_routes"] = {"headline_classification": {"primary": {
                "provider": requested_provider,
                "channel": "cli" if requested_provider in {"codex", "claude"} else "api",
                "model": model}}}
            intel["local_fallback"] = {"provider": "", "models": {}}
            intel["api_fallback"] = {**(intel.get("api_fallback") or {}), "enabled": False}
            c = Config(data)
        result = AIClient(c, store).complete_json(
            test_purpose, "provider_test",
            "Return only valid JSON in the requested shape. This is a connectivity test.",
            "Classify the neutral sentence 'Test completed.' with sentiment 0, "
            "confidence 1, catalyst 'test', horizon_days 1, already_priced true, "
            "and summary 'provider route operational'.", 120)
        if result is None:
            raise HTTPException(503, "all configured intelligence routes failed")
        return {"ok": True, "requested_purpose": purpose,
                "requested_provider": requested_provider or None, "result": result,
                "last_call": store.kv_get("intelligence_last_call")}

    @app.get("/api/intelligence/jobs")
    def intelligence_jobs():
        from .intelligence import jobs
        return {"jobs": jobs(store), "state": store.kv_get("intelligence_last_call")}

    @app.get("/api/strategy/directives")
    def strategy_directives():
        from .strategy import active, mandates, messages
        return {"active": active(store), "messages": messages(store),
                "mandates": mandates(store)}

    @app.post("/api/strategy/directives/analyze")
    def strategy_analyze(body: dict):
        from .intelligence import enqueue
        from .strategy import submit
        try:
            msg = submit(store, body.get("text", ""))
            job = enqueue(store, "strategic_synthesis", {"message_id": msg["id"]}, priority=100)
            return {"message": msg, "job": job}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/strategy/directives/{mandate_id}/activate")
    def strategy_activate(mandate_id: str):
        from .strategy import activate
        try:
            return activate(store, mandate_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/strategy/directives/{mandate_id}/deactivate")
    def strategy_deactivate(mandate_id: str):
        from .strategy import deactivate
        try:
            return deactivate(store, mandate_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

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
        from .config import set_env_vars
        provider = body.get("provider", "custom")
        base_url = (AI_PROVIDERS.get(provider) or body.get("base_url", "") or "").strip()
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or \
                parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise HTTPException(400, "base_url must be a credential-free http(s) origin/path")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise HTTPException(400, "non-loopback custom providers must use https")
        key = (body.get("api_key") or "").strip()
        settings = {"AI_BASE_URL": base_url}
        if key:
            settings["AI_API_KEY"] = key
        try:
            set_env_vars(settings)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
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
        out = {"cycle": None, "considered": [], "working": [], "exits": {},
               "rebalance": None}
        if last:
            summ = json.loads(last["payload"] or "{}")
            out["cycle"] = {"id": last["cycle_id"], "ts": last["ts"],
                            "as_of": summ.get("as_of"), "regime": summ.get("regime"),
                            "budget": summ.get("budget"),
                            "budget_used": summ.get("budget_used")}
            out["exits"] = summ.get("exits", {})
            rr = next((x for x in store.audit_rows(cycle_id=last["cycle_id"], limit=500)
                       if x["event_type"] == "rebalance_plan"), None)
            out["rebalance"] = json.loads(rr["payload"] or "{}") if rr else None
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
                    "evidence_version": pl.get("evidence_version"),
                    "evidence_coverage": pl.get("evidence_coverage"),
                    "evidence": pl.get("evidence_details", []),
                    "production_score": pl.get("production_score", c["final_score"]),
                    "strategy_contribution": pl.get("strategy_contribution", 0),
                    "learned_contribution": pl.get("learned_contribution", 0),
                    "strategy_mandate_id": pl.get("strategy_mandate_id"),
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
        dossiers = []
        seen = set()
        for row in store.db.execute(
                "SELECT symbol,created_at,quality,status,fundamental_memo,catalyst_memo "
                "FROM company_evidence ORDER BY created_at DESC LIMIT 50"):
            if row["symbol"] in seen:
                continue
            seen.add(row["symbol"])
            item = dict(row)
            for key in ("fundamental_memo", "catalyst_memo"):
                try:
                    item[key] = json.loads(item[key] or "null")
                except json.JSONDecodeError:
                    item[key] = None
            dossiers.append(item)
            if len(dossiers) == 5:
                break
        return {
            "date": day, "scans": scans, "candidates": candidates,
            "orders": by_status,
            "top_vetoes": sorted(veto_reasons.items(), key=lambda x: -x[1])[:4],
            "news": store.kv_get("news_synopsis"),
            "fundamentals": store.kv_get("fundamentals_synopsis"),
            "company_evidence": dossiers,
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
        from .ensemble import s_node_multiplier, s_node_weight
        c = fresh_cfg()
        ctx = MarketContext(store, c)
        reg = regime_mod.classify(ctx, c)
        sig_counts = {r["node_id"]: r["n"] for r in store.db.execute(
            "SELECT node_id, COUNT(*) n FROM signals "
            "WHERE ts >= datetime('now', '-7 days') GROUP BY node_id")}
        nodes = []
        for node_id, nc in (c.get("nodes", default={}) or {}).items():
            sc = node_scorecard(store, node_id,
                                sources=("live" if c.mode == "live" else "paper",))
            nodes.append({
                "id": node_id, "enabled": bool(nc.get("enabled")),
                "role": nc.get("role", "alpha"), "status": nc.get("status"),
                "ai": bool(nc.get("ai")) or node_id == "hypothesis",
                "base_weight": nc.get("weight", 0.0),
                "stored_multiplier": round(store.get_weight_multiplier(node_id), 3),
                "multiplier": (round(s_node_multiplier(node_id, c, store,
                                                        reg.regime), 3)
                               if nc.get("enabled") else 0.0),
                "effective_weight": (round(s_node_weight(node_id, c, store,
                                                         reg.regime), 4)
                                     if nc.get("enabled") else 0.0),
                "signals_7d": sig_counts.get(node_id, 0),
                "trades_n": sc.get("n", 0),
                "expectancy": sc.get("expectancy"),
                "hit_rate": sc.get("hit_rate"),
                "per_trade_ir": sc.get("per_trade_ir"),
            })
        st = store.active_hypothesis("short_term")
        from .graph import activation_state, champion as graph_champion
        graph = graph_champion(store)
        activation = activation_state(c, store)
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
                             "live_blend": activation["effective_blend"],
                             "activation": activation},
        }

    @app.get("/api/model/graph")
    def model_graph(symbol: str = "SPY", horizon: int = 21):
        """Actual deployed topology plus symbol-specific live activations."""
        from .graph import activation_state, champion as graph_champion
        from .neural import describe as describe_neural
        c = fresh_cfg(); graph = graph_champion(store)
        if horizon not in (5, 21):
            raise HTTPException(400, "horizon must be 5 or 21")
        snapshots = store.kv_get("graph_last_activations", {}) or {}
        symbol = symbol.upper()
        if "symbols" in snapshots:
            snapshot = snapshots.get("symbols", {}).get(symbol)
            if snapshot:
                snapshot = {**snapshot, "as_of": snapshots.get("as_of"),
                            "cycle_id": snapshots.get("cycle_id")}
        else:                                  # compatibility with D41 snapshots
            snapshot = snapshots.get(symbol)
        # Never pair activations from an older topology with the deployed DAG.
        # Legacy snapshots lack an explicit schema, so detect removed node IDs
        # as well. A directed empty state is more truthful than a plausible lie.
        topology_ids = {n["id"] for n in graph["topology"]["nodes"]}
        snapshot_ids = set(((snapshot or {}).get("activations") or {}).keys())
        snapshot_schema = snapshots.get("topology_schema")
        if snapshot and ((snapshot_schema and snapshot_schema != graph["topology"].get("schema"))
                         or snapshot_ids - topology_ids):
            snapshot = None
        configured = float(c.get("analog_graph", "live_blend", default=0) or 0)
        activation = activation_state(c, store)
        effective = activation["effective_blend"]
        enabled = {n for n, nc in (c.get("nodes", default={}) or {}).items()
                   if nc.get("enabled")}
        snapshot_states = (snapshot or {}).get("node_states", {})
        node_state = {}
        for n in graph["topology"]["nodes"]:
            fallback = ("shadow" if n["role"] in ("interaction", "output") and
                        graph["status"] != "champion" else
                        "neutral" if n["id"] in enabled or
                        n["id"] in ("macro_regime", "quality_value") else "unavailable")
            node_state[n["id"]] = snapshot_states.get(n["id"], fallback)
        return {"schema": graph["topology"].get("schema", "stonk.graph.v1"),
                "version": graph["id"],
                "status": graph["status"], "metrics": graph.get("metrics", {}),
                "live_blend": configured,
                "configured_live_blend": configured,
                "effective_live_blend": effective,
                "ramp_stage": activation["stage"],
                "current_live_blend": effective,
                "entry_block_reason": None,
                "learned_block_reason": activation["block_reason"],
                "production_evidence_version": "evidence.v2",
                "production_evidence_active": True,
                "activation_completeness": ((snapshot or {}).get(
                    "activation_complete") if snapshot else False),
                "activation_state": activation, "node_state": node_state,
                "topology_provenance": "learned" if graph["status"] == "champion"
                                       else "initial_prior",
                "symbol": symbol, "horizon": horizon,
                "topology": graph["topology"],
                "snapshot": snapshot,
                "temporal": describe_neural(c, store, symbol)}

    @app.get("/api/research")
    def research_status():
        from .research import status
        return status(store, fresh_cfg())

    @app.post("/api/research/jobs")
    def create_research_job(body: dict):
        from .research import PUBLIC_JOB_KINDS, enqueue_job
        try:
            kind = str(body.get("kind", ""))
            if kind not in PUBLIC_JOB_KINDS:
                raise HTTPException(400, f"unknown operator research job: {kind}")
            if kind == "deep_research":
                from .ai import AIClient
                if not AIClient(fresh_cfg(), store).available():
                    raise HTTPException(409, "Deep Research requires an enabled, available "
                                           "Codex/Claude/API route with remaining limits")
            return enqueue_job(store, kind, body.get("payload") or {}, priority=10,
                               requested_by="operator", force=bool(body.get("force")))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/research/jobs")
    def research_jobs(limit: int = 20):
        from .research import list_jobs
        return list_jobs(store, min(100, max(1, limit)))

    @app.post("/api/research/jobs/{job_id}/cancel")
    def cancel_research_job(job_id: str):
        from .research import cancel_job
        try:
            return cancel_job(store, job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/research/reports")
    def research_reports(symbol: str | None = None, limit: int = 20):
        sql, params = "SELECT * FROM research_reports", []
        if symbol:
            sql += " WHERE symbol=?"; params.append(symbol.upper())
        sql += " ORDER BY created_at DESC LIMIT ?"; params.append(min(100, max(1, limit)))
        rows = [dict(r) for r in store.db.execute(sql, tuple(params))]
        for row in rows:
            row["sources"] = json.loads(row["sources"] or "{}")
            row["report"] = json.loads(row["report"] or "{}")
        return rows

    @app.get("/api/universe")
    def universe_status(tier: str = "active", limit: int = 250, offset: int = 0):
        if tier not in ("research", "active", "shortlist"):
            raise HTTPException(400, "tier must be research, active, or shortlist")
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
        from .research import list_jobs
        research_state = store.kv_get("research_state") or {"phase": "never_ran"}
        research_jobs = list_jobs(store, 10)
        from .intelligence import jobs as list_intelligence_jobs
        intelligence_jobs = list_intelligence_jobs(store, 10)
        active_intelligence = next((j for j in intelligence_jobs
                                    if j.get("status") == "running"), None)
        intelligence_state = ({"phase": active_intelligence["kind"],
                               "detail": (active_intelligence.get("progress") or {}).get(
                                   "phase", "working"),
                               "progress": active_intelligence.get("progress") or {},
                               "at": active_intelligence.get("started_at")}
                              if active_intelligence else
                              {"phase": "idle", "detail": "waiting for cached-news refresh "
                               "or Strategy AI direction", "progress": {},
                               "at": (store.kv_get("intelligence_last_call") or {}).get("at")})
        worker_snapshot = (getattr(app.state, "worker_snapshot", lambda: {})())
        return {"state": state,
                "processes": {
                    "trading": {"state": state, "serial": True},
                    "research": {"state": research_state,
                                 "jobs": [_job_summary(j) for j in research_jobs],
                                 "workers": worker_snapshot,
                                 "serial": True,
                                 "detail": "one research task at a time; trading remains responsive"},
                    "intelligence": {"state": intelligence_state,
                                     "jobs": [_job_summary(j) for j in intelligence_jobs],
                                     "serial": True,
                                     "detail": "one sandboxed provider call at a time; trading reads cache"}},
                "market": _market_clock(),
                "interval_minutes": c.get("schedule", "scan_interval_minutes"),
                "scan_times": c.get("schedule", "scans", default=[]),
                "next_runs": jobs,
                "heartbeat": store.kv_get("heartbeat")}

    @app.post("/api/scan")
    def manual_scan():
        from .health import _market_clock
        market = _market_clock()
        if not market["open"]:
            return {"skipped": f"market {market['session']}; use Discover Opportunities "
                               "for closed-market analysis"}
        from .engine import run_cycle
        from .health import write_heartbeat
        summary = run_cycle(fresh_cfg(), store)
        if not summary.get("skipped"):
            write_heartbeat(store, summary["cycle_id"], mode, source="serve")
        return summary

    # ---------------- scheduler ----------------
    if with_scheduler:
        _start_scheduler(app, store, mode)

        @app.on_event("shutdown")
        def stop_scheduler():
            import os
            import signal
            app.state.stopping = True
            scheduler = getattr(app.state, "scheduler", None)
            if scheduler and scheduler.running:
                scheduler.shutdown(wait=False)
            for proc in list(getattr(app.state, "worker_processes", {}).values()):
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
    return app


def _start_scheduler(app: FastAPI, store: Store, mode: str) -> None:
    import os
    import subprocess
    import sys
    import threading
    import time
    import signal

    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    app.state.worker_processes = {}
    app.state.worker_deadlines = {}
    app.state.worker_lock = threading.Lock()

    def _worker_snapshot() -> dict:
        snapshot = {}
        with app.state.worker_lock:
            for lane, proc in list(app.state.worker_processes.items()):
                code = proc.poll()
                deadline = app.state.worker_deadlines.get(lane)
                if code is None and deadline is not None and time.monotonic() > deadline:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    snapshot[lane] = {"pid": proc.pid, "state": "timed_out",
                                      "exit_code": None}
                    store.kv_set(f"worker_process:{lane}", {
                        **snapshot[lane], "timed_out_at": datetime.now().astimezone()
                        .isoformat(timespec="seconds")})
                    store.audit("research_worker_timed_out", {"lane": lane,
                                                               "pid": proc.pid})
                    del app.state.worker_processes[lane]
                    app.state.worker_deadlines.pop(lane, None)
                    continue
                snapshot[lane] = {"pid": proc.pid,
                                  "state": "running" if code is None else "exited",
                                  "exit_code": code}
                if code is not None:
                    del app.state.worker_processes[lane]
                    app.state.worker_deadlines.pop(lane, None)
        return snapshot

    app.state.worker_snapshot = _worker_snapshot

    def _spawn_worker(lane: str, max_seconds: int = 600) -> dict:
        """Start at most one short-lived process per durable worker lane.

        Pandas/Torch allocations and provider clients are intentionally kept
        out of Uvicorn's trading/control process; process exit returns their
        allocator high-water memory to the OS.
        """
        with app.state.worker_lock:
            existing = app.state.worker_processes.get(lane)
            if existing and existing.poll() is None:
                return {"lane": lane, "state": "running", "pid": existing.pid}
            if existing:
                del app.state.worker_processes[lane]
                app.state.worker_deadlines.pop(lane, None)
            log_dir = Path(__file__).resolve().parent.parent / "logs"
            log_dir.mkdir(exist_ok=True)
            log = (log_dir / f"worker-{mode}.log").open("a", encoding="utf-8")
            broker_secret_prefixes = ("RH_", "ROBINHOOD_", "MCP_")
            env = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(broker_secret_prefixes)}
            env["PYTHONUNBUFFERED"] = "1"
            env["STONK_RESEARCH_WORKER"] = "1"
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "specforge.cli", "--mode", mode,
                     "worker", lane, "--max-seconds", str(max_seconds)],
                    cwd=Path(__file__).resolve().parent.parent,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=env, shell=False, start_new_session=True)
            finally:
                log.close()
            app.state.worker_processes[lane] = proc
            app.state.worker_deadlines[lane] = time.monotonic() + max_seconds + 15
        detail = {"lane": lane, "state": "running", "pid": proc.pid,
                  "started_at": datetime.now().astimezone().isoformat(timespec="seconds")}
        store.kv_set(f"worker_process:{lane}", detail)
        return detail

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

    def broker_session_job():
        """Premarket non-interactive auth/read probe; never opens a browser."""
        from .health import _broker_health
        c = current_config(store, mode)
        state = _broker_health(c, store, force=True)
        store.audit("broker_session_check", {
            "connected": bool(state.get("connected")),
            "detail": str(state.get("detail") or "")[:200]})
        if not state.get("connected"):
            _notify("Stonk Terminal: Robinhood login required",
                    "Broker session check failed before open; use Connect Robinhood once")

    cfg = current_config(store, mode)
    from .research import recover_jobs
    recover_jobs(store)
    from .intelligence import recover as recover_intelligence_jobs
    recover_intelligence_jobs(store)
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
    sched.add_job(broker_session_job,
                  CronTrigger(day_of_week="mon-fri", hour=9, minute=10, timezone=tz),
                  misfire_grace_time=900, id="broker_session_check")

    from apscheduler.triggers.interval import IntervalTrigger

    # Enforce worker wall-clock budgets even when no dashboard is polling.
    sched.add_job(_worker_snapshot, IntervalTrigger(seconds=5),
                  next_run_time=datetime.now().astimezone() + timedelta(seconds=5),
                  max_instances=1, coalesce=True, misfire_grace_time=30,
                  id="worker_watchdog")

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

    # Closed-market research is one atomic unit per short-lived subprocess.
    # It never shares Uvicorn's heap/GIL with the trading/control plane.
    def research_job():
        from .health import _market_clock
        if _market_clock()["open"]:
            return
        import pandas as _pd
        import exchange_calendars as _xcals
        seconds_to_open = (_xcals.get_calendar("XNYS").next_open(
            _pd.Timestamp.now(tz="UTC")) - _pd.Timestamp.now(tz="UTC")).total_seconds()
        budget = min(840, max(0, int(seconds_to_open - 15 * 60)))
        if budget < 30:
            return
        try:
            _spawn_worker("autonomous", min(600, budget))
        except Exception as e:                  # noqa: BLE001
            store.audit("scheduler_error", {"job": "research",
                                            "error": str(e)[:300]})

    sched.add_job(research_job, IntervalTrigger(
        minutes=int(cfg.get("research", "interval_minutes", default=15))),
        next_run_time=datetime.now().astimezone() + timedelta(seconds=8),
        misfire_grace_time=600, id="research")

    # Operator buttons should start promptly after the current atomic research
    # unit finishes. Polling SQLite is cheap; the shared research mutex keeps
    # this from overlapping autonomous backfill/training.
    def operator_lane(resource: str):
        """Run one resource lane.

        Discovery and intelligence are independent market-safe workloads. A
        slow model call must not strand a cached discovery scan in QUEUED, and
        neither may occupy the training lane. Per-resource SQLite leases in
        research.py preserve single execution across multiple app processes.
        """
        try:
            from .research import recover_jobs
            recover_jobs(store)
            state = store.kv_get("engine_state", {}) or {}
            if state.get("phase") not in (None, "idle", "sleeping", "never_ran"):
                return                    # trading always wins the start boundary
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            due = store.db.execute(
                "SELECT 1 FROM research_jobs WHERE status='queued' AND resource_class=? "
                "AND (eligible_at IS NULL OR eligible_at<=?) "
                "AND (next_retry_at IS NULL OR next_retry_at<=?) LIMIT 1",
                (resource, now, now)).fetchone()
            if not due:
                return
            _spawn_worker(f"operator-{resource}", 600)
        except Exception as e:             # noqa: BLE001
            store.audit("scheduler_error", {"job": f"operator_{resource}",
                                            "error": str(e)[:300]})

    for lane_index, resource in enumerate(("discovery", "intelligence", "training")):
        sched.add_job(
            operator_lane, trigger=IntervalTrigger(seconds=5), args=[resource],
            next_run_time=datetime.now().astimezone() + timedelta(seconds=5 + lane_index),
            # One instance may spend minutes inside a model call.  Permit a
            # second cheap poll: the durable SQLite lease makes it return
            # immediately, while avoiding APScheduler's repeated noisy
            # "maximum instances" warnings during healthy long jobs.
            max_instances=1, coalesce=True, misfire_grace_time=30,
            id=f"operator_{resource}")

    # Intelligence is market-safe: it never contacts the broker and trading
    # reads only committed cached results. Strategy requests take priority;
    # news refreshes are deduplicated and only classify newly ingested items.
    def intelligence_job():
        try:
            from .intelligence import enqueue, jobs, recover
            recover(store)
            rows = jobs(store, limit=5)
            active = any(j.get("status") in ("queued", "running") for j in rows)
            state = store.kv_get("news_intelligence", {}) or {}
            stale = True
            try:
                stale = datetime.now().astimezone() - datetime.fromisoformat(
                    state.get("as_of", "")) >= timedelta(minutes=30)
            except (ValueError, TypeError):
                pass
            if not active and stale:
                enqueue(store, "news_refresh", priority=1)
                rows = jobs(store, limit=5)
            if any(j.get("status") == "queued" for j in rows):
                _spawn_worker("news", 600)
        except Exception as exc:              # intelligence never wedges scheduler
            store.audit("scheduler_error", {"job": "intelligence",
                                            "error": str(exc)[:300]})

    sched.add_job(intelligence_job, IntervalTrigger(seconds=10),
                  next_run_time=datetime.now().astimezone() + timedelta(seconds=12),
                  # run_next owns the same cross-process intelligence lease;
                  # a second scheduler poll is safe and stays short.
                  max_instances=1, coalesce=True, misfire_grace_time=60,
                  id="intelligence")

    # D40: the weekend research loop — schedule slot existed since V1 but was
    # never wired. A 2y walk-forward backtest of the CURRENT config every
    # Saturday: config drift vs measured edge surfaces within a week.
    wr = cfg.get("schedule", "weekend_research", default="SAT 10:00")

    def weekend_research_job():
        try:
            _spawn_worker("weekend", 1800)
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
