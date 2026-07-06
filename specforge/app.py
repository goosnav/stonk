"""FastAPI backend + scheduler. Serves static/dashboard.html and a JSON API.

Runtime config edits from the GUI persist in kv['config_overrides'] and are
merged on every config load (so scheduled scans pick them up immediately).
Every mutation is audit-logged. Dangerous risk values are rejected by
Config.validate() exactly like file edits — the GUI has no privileged path.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .config import ConfigError, load_config
from .data import MarketContext
from .forecast import portfolio_projection
from .risk import Governor
from .store import Store

STATIC = Path(__file__).resolve().parent.parent / "static"
OVERRIDES_KEY = "config_overrides"


def current_config(store: Store, mode: str):
    return load_config(mode, overrides=store.kv_get(OVERRIDES_KEY, {}))


def create_app(cfg, store: Store, with_scheduler: bool = True) -> FastAPI:
    app = FastAPI(title="SpecForge", docs_url="/api/docs")
    mode = cfg.mode

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

    # ---------------- pages ----------------
    @app.get("/")
    def index():
        return FileResponse(STATIC / "dashboard.html")

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
        day_pnl = None
        if len(curve) >= 2:
            day_pnl = round(acct.equity - curve[-2]["equity"], 2)
        return {
            "mode": c.mode, "broker": c.get("broker"),
            "equity": round(acct.equity, 2), "cash": round(acct.cash, 2),
            "day_pnl": day_pnl,
            "drawdown_from_peak": round(1 - acct.equity / peak, 4) if peak else 0,
            "regime": reg.regime, "regime_evidence": reg.evidence,
            "deployment_multiplier": reg.deployment_multiplier,
            "kill_switches": gov.active_switches(),
            "options_unlocked": gov.options_unlocked(acct),
            "cycle_budget": round(gov.cycle_budget(acct, reg.deployment_multiplier), 2),
            "approval_mode": c.get("risk", "approval_mode"),
            "positions": [{
                "symbol": p.symbol, "qty": p.qty, "avg_cost": round(p.avg_cost, 2),
                "last": prices.get(p.symbol),
                "pnl_pct": round(prices.get(p.symbol, p.avg_cost) / p.avg_cost - 1, 4),
                "value": round(p.qty * prices.get(p.symbol, p.avg_cost), 2),
            } for p in acct.positions if p.qty > 0],
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
        out = []
        for node_id, nc in (c.get("nodes", default={}) or {}).items():
            rets = by_node.get(node_id, [])
            out.append({
                "id": node_id, **nc,
                "weight_multiplier": store.get_weight_multiplier(node_id),
                "n_trades": len(rets),
                "expectancy": round(sum(rets) / len(rets), 5) if rets else None,
                "win_rate": round(sum(1 for r in rets if r > 0) / len(rets), 3) if rets else None,
            })
        return out

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
            "ai_daily_budget_usd": ai_cfg.get("daily_budget_usd"),
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
        ov = store.kv_get(OVERRIDES_KEY, {}) or {}
        cur = ov
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        cur[path[-1]] = value
        # validate merged result BEFORE persisting — GUI can't sneak past governor
        load_config(mode, overrides=ov)
        store.kv_set(OVERRIDES_KEY, ov)
        store.audit("config_override", {"path": path, "value": value, "via": "gui"})

    @app.post("/api/nodes/{node_id}")
    def set_node(node_id: str, body: dict):
        c = fresh_cfg()
        if node_id not in (c.get("nodes", default={}) or {}):
            raise HTTPException(404, f"unknown node {node_id}")
        for key in ("enabled", "weight"):
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
        allowed = {"enabled", "model", "daily_budget_usd"}
        for k, v in body.items():
            if k not in allowed:
                raise HTTPException(400, f"ai key not editable via GUI: {k}")
            _set_override(["ai", k], v)
        return {"ok": True}

    @app.post("/api/approvals/{intent_id}")
    def decide(intent_id: str, body: dict):
        decision = body.get("decision")
        if decision not in ("approved", "rejected"):
            raise HTTPException(400, "decision must be approved|rejected")
        store.decide_approval(intent_id, decision)
        if decision == "rejected":
            store.update_order(intent_id, status="rejected")
        store.audit("approval_decided", {"intent": intent_id, "decision": decision,
                                         "via": "gui"})
        return {"ok": True}

    @app.post("/api/kill/{name}/reset")
    def reset_kill(name: str):
        Governor(fresh_cfg(), store).reset(name)
        return {"ok": True}

    @app.post("/api/scan")
    def manual_scan():
        from .engine import run_cycle
        summary = run_cycle(fresh_cfg(), store)
        return summary

    # ---------------- scheduler ----------------
    if with_scheduler:
        _start_scheduler(app, store, mode)
    return app


def _start_scheduler(app: FastAPI, store: Store, mode: str) -> None:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    def scan_job():
        from .engine import run_cycle
        cfg = current_config(store, mode)
        try:
            summary = run_cycle(cfg, store)
            print(f"[scheduler] scan done: {summary['cycle_id']} "
                  f"entries={summary['entries']} exits={summary['exits']}")
        except Exception as e:                      # noqa: BLE001
            store.audit("scheduler_error", {"error": str(e)})
            print(f"[scheduler] scan FAILED: {e}")

    def post_close_job():
        """Mark-to-market + attribution: the self-improvement heartbeat."""
        from .attribution import propose_promotions, update_weights
        from .engine import run_cycle
        cfg = current_config(store, mode)
        try:
            run_cycle(cfg, store)              # final scan marks equity + exits
            update_weights(cfg, store)
            proposals = propose_promotions(cfg, store)
            if proposals:
                store.audit("promotion_proposals", proposals)
                store.kv_set("promotion_proposals", proposals)
        except Exception as e:                  # noqa: BLE001
            store.audit("scheduler_error", {"job": "post_close", "error": str(e)})
            print(f"[scheduler] post-close FAILED: {e}")

    cfg = current_config(store, mode)
    tz = cfg.get("schedule", "timezone", default="America/New_York")
    sched = BackgroundScheduler(timezone=tz)
    for hhmm in cfg.get("schedule", "scans", default=[]):
        h, m = hhmm.split(":")
        sched.add_job(scan_job, CronTrigger(day_of_week="mon-fri", hour=int(h),
                                            minute=int(m), timezone=tz))
    pc = cfg.get("schedule", "post_close", default="16:30")
    h, m = pc.split(":")
    sched.add_job(post_close_job, CronTrigger(day_of_week="mon-fri", hour=int(h),
                                              minute=int(m), timezone=tz))
    sched.start()
    app.state.scheduler = sched
