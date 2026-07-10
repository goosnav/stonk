"""System truth aggregator (CONTROL_CENTER_V3 chunk 2).

One function, system_health(), answers: what mode is this, is the broker
really connected, is anything actually running, is the market open, is the
data fresh, and — the contract — WHY we are not trading whenever we aren't.
Used by GET /api/health and `stonk tui`. Must NEVER raise: every probe
degrades to connected=false + the error string (fail loudly, not silently).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .store import Store

BROKER_PROBE_TTL_S = 60
HEARTBEAT_STALE_S = 6 * 3600
ET = ZoneInfo("America/New_York")


def write_heartbeat(store: Store, cycle_id: str, mode: str, source: str) -> None:
    """Called after EVERY scan cycle (serve scheduler AND cron `scan`)."""
    store.kv_set("heartbeat", {"at": datetime.now().astimezone().isoformat(timespec="seconds"),
                               "cycle_id": cycle_id, "mode": mode, "source": source})


def _broker_health(cfg, store: Store) -> dict:
    """Cheap cached connectivity probe. Paper = honest 'simulation' label."""
    adapter = cfg.get("broker", default="paper")
    if adapter == "paper":
        return {"adapter": "paper", "connected": True,
                "detail": "SIMULATION — not real money", "as_of": _now_iso()}
    cached = store.kv_get("broker_health")
    if cached and cached.get("adapter") == adapter and \
            _age_s(cached.get("as_of")) < BROKER_PROBE_TTL_S:
        return cached
    out = {"adapter": adapter, "as_of": _now_iso()}
    try:
        from .broker.base import make_broker
        broker = make_broker(cfg, store)
        if adapter == "robinhood_bridge":
            fresh = broker._snapshot_fresh()        # noqa: SLF001 — same package
            out.update(connected=fresh,
                       detail="" if fresh else "bridge account snapshot stale/missing")
        else:
            q = broker.get_quotes(["SPY"])
            out.update(connected=bool(q),
                       detail="" if q else "quote probe returned no data")
    except Exception as e:                          # noqa: BLE001 — degrade loudly
        out.update(connected=False, detail=str(e)[:300])
    store.kv_set("broker_health", out)
    return out


def _market_clock() -> dict:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return {"open": False, "session": "weekend", "et": now.strftime("%H:%M ET")}
    hm = now.strftime("%H:%M")
    open_ = "09:30" <= hm < "16:00"
    # ponytail: US market holidays not modeled; worst case = one no-fill scan day
    return {"open": open_, "session": "regular" if open_ else "closed",
            "et": now.strftime("%H:%M ET")}


def system_health(cfg, store: Store, next_runs: dict | None = None,
                  scheduler_alive: bool | None = None) -> dict:
    mode = cfg.mode
    broker = _broker_health(cfg, store)
    market = _market_clock()

    hb = store.kv_get("heartbeat") or {}
    hb_age = _age_s(hb.get("at")) if hb else None
    engine = {"last_scan_at": hb.get("at"), "last_cycle_id": hb.get("cycle_id"),
              "heartbeat_mode": hb.get("mode"), "heartbeat_source": hb.get("source"),
              "heartbeat_age_s": None if hb_age is None else int(hb_age),
              "scheduler_alive": scheduler_alive,
              "next_runs": next_runs or {}}

    bench = cfg.get("universe", "benchmark", default="SPY")
    newest = store.latest_bar_date(bench)
    stale_limit = cfg.get("risk", "stale_data_max_age_days", default=4)
    data_age = (datetime.now().date() - datetime.strptime(newest, "%Y-%m-%d").date()).days \
        if newest else None
    data = {"newest_bar": newest, "age_days": data_age, "stale_limit_days": stale_limit}

    from .risk import Governor
    switches = Governor(cfg, store).active_switches()

    # --- the contract: reasons is NEVER empty when trading=false ---
    reasons: list[str] = []
    if mode != "live":
        reasons.append("mode is PAPER — all numbers are simulation, no real orders")
    if not broker["connected"]:
        reasons.append(f"broker disconnected: {broker.get('detail', 'unknown')}")
    if mode == "live":
        ok, why = cfg.live_trading_allowed()
        if not ok:
            reasons.append(f"live gate closed: {why}")
    if switches:
        reasons.append(f"kill switch active: {', '.join(sorted(switches))}")
    if not market["open"]:
        reasons.append(f"market {market['session']} ({market['et']}) — orders queue for open")
    if hb_age is None:
        reasons.append("no scan heartbeat ever — engine has not run; start the "
                       "daemon or cron (see OPS panel)")
    elif hb_age > HEARTBEAT_STALE_S:
        reasons.append(f"no scan heartbeat in {int(hb_age/3600)}h — is the "
                       f"daemon/cron actually running?")
    if data_age is not None and data_age > stale_limit:
        reasons.append(f"market data stale ({data_age}d old) — governor will veto entries")
    # Broker review blocks/refusals carry the actionable explanation. Query
    # them directly rather than scanning a bounded audit tail that busy MCP
    # traffic can push out in minutes. A newer fill proves the block cleared.
    import json as _json
    today = datetime.now().astimezone().date().isoformat()
    refusal = store.db.execute(
        "SELECT * FROM audit WHERE event_type='broker_place_refused' "
        "AND substr(ts,1,10)=? ORDER BY id DESC LIMIT 1", (today,)).fetchone()
    review = store.db.execute(
        "SELECT * FROM audit WHERE event_type='broker_review' "
        "AND substr(ts,1,10)=? AND payload LIKE '%\"ok\": false%' "
        "ORDER BY id DESC LIMIT 1", (today,)).fetchone()
    fill = store.db.execute(
        "SELECT * FROM audit WHERE event_type IN "
        "('order_filled','order_filled_reconciled') ORDER BY id DESC LIMIT 1"
    ).fetchone()
    blocked = max((r for r in (refusal, review) if r),
                  key=lambda r: r["id"], default=None)
    if blocked and (not fill or fill["id"] < blocked["id"]):
        payload = _json.loads(blocked["payload"] or "{}")
        detail = payload.get("response") or ", ".join(payload.get("warnings", []))
        reasons.append(f"broker blocked entries — {str(detail)[:260]}")

    return {"mode": mode, "broker": broker, "engine": engine, "market": market,
            "data": data, "kill_switches": sorted(switches),
            "pending_approvals": len(store.pending_approvals()),
            "readiness": {"trading": not reasons, "reasons": reasons},
            "as_of": _now_iso()}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _age_s(iso: str | None) -> float:
    if not iso:
        return float("inf")
    try:
        return (datetime.now().astimezone() - datetime.fromisoformat(iso)).total_seconds()
    except ValueError:
        return float("inf")
