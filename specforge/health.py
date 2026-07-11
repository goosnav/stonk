"""System truth aggregator (CONTROL_CENTER_V3 chunk 2).

One function, system_health(), answers: what mode is this, is the broker
really connected, is anything actually running, is the market open, is the
data fresh, and — the contract — WHY we are not trading whenever we aren't.
Used by GET /api/health and `stonk tui`. Must NEVER raise: every probe
degrades to connected=false + the error string (fail loudly, not silently).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .store import Store

BROKER_PROBE_TTL_S = 60
MIN_HEARTBEAT_STALE_S = 30 * 60
ET = ZoneInfo("America/New_York")

# Sanitizer for any operator-facing detail string that may embed broker/AI
# error text: token-ish assignments, account-number-length digit runs, and
# long base64/JWT-ish blobs. Timestamps (digit runs ≤4), 12-char cycle ids,
# and dotted/slashed hostnames+paths (segments stay short) survive.
# Applied to broker detail, last_error, and rollup alerts.
_SECRETISH = [
    re.compile(r"(?i)(?:token|key|secret|bearer|authorization|password)"
               r"[\"'=:\s]+[^\s\"']{6,}"),
    re.compile(r"\d{7,}"),
    re.compile(r"[A-Za-z0-9+_\-]{28,}"),
]


def _redact(text: str) -> str:
    for pat in _SECRETISH:
        text = pat.sub("[redacted]", text)
    return text


STATUS_ORDER = {"ok": 0, "degraded": 1, "stale": 2, "error": 3}


def rollup(h: dict) -> tuple[str, list[str]]:
    """App-health verdict from structured system_health() fields — the machine
    contract monitors act on (RUNBOOK.md). Distinct from readiness.trading,
    which is legitimately false outside market hours or in paper mode:
      ok        nothing needs anyone (even if not trading right now)
      degraded  app alive, something needs the OPERATOR (broker down/blocked,
                kill switch, stale data, fresh cycle error) — do NOT restart
      stale     scans not happening while the market is open — restart may help
      error     scheduler dead in the serving process — restart
    scheduler_alive=None means "no scheduler in this caller" (cron/TUI/tests)
    and is unknown, not an error.
    """
    status, alerts = "ok", []

    def worse(sev: str, msg: str) -> None:
        nonlocal status
        alerts.append(_redact(msg))
        if STATUS_ORDER[sev] > STATUS_ORDER[status]:
            status = sev

    b = h.get("broker") or {}
    if not b.get("connected"):
        worse("degraded", f"broker disconnected: {b.get('detail') or 'unknown'}")
    for name in h.get("kill_switches") or []:
        worse("degraded", f"kill switch active: {name}")
    if h.get("broker_block"):
        worse("degraded", f"broker blocking entries: {h['broker_block']}")
    d = h.get("data") or {}
    limit = d.get("stale_limit_days") or 4
    if d.get("age_days") is not None and d["age_days"] > limit:
        worse("degraded", f"market data stale ({d['age_days']}d old, "
                          f"limit {limit}d) — governor vetoes entries")
    e = h.get("engine") or {}
    stale_s = e.get("heartbeat_stale_s") or MIN_HEARTBEAT_STALE_S
    le = h.get("last_error")
    if le and le.get("age_s") is not None and le["age_s"] < stale_s:
        worse("degraded", f"recent {le.get('event', 'error')} "
                          f"({le['age_s']}s ago): {le.get('detail', '')}")
    if (h.get("market") or {}).get("open"):
        hb = e.get("heartbeat_age_s")
        if hb is None or hb > stale_s:
            worse("stale", "no completed scan "
                  + (f"in {int(hb / 60)}m" if hb is not None else "ever")
                  + f" with the market open (limit {int(stale_s / 60)}m)")
    if e.get("scheduler_alive") is False:
        worse("error", "scheduler is not running in the serving process")
    return status, alerts


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
    if broker.get("detail"):                 # adapter errors can quote raw
        broker = {**broker, "detail": _redact(str(broker["detail"]))}
    market = _market_clock()

    hb = store.kv_get("heartbeat") or {}
    if hb.get("mode") != mode:                 # paper/live share the same DB
        hb = {}
    hb_age = _age_s(hb.get("at")) if hb else None
    interval_min = int(cfg.get("schedule", "scan_interval_minutes", default=10) or 10)
    heartbeat_stale_s = max(MIN_HEARTBEAT_STALE_S, interval_min * 3 * 60)
    engine = {"last_scan_at": hb.get("at"), "last_cycle_id": hb.get("cycle_id"),
              "heartbeat_mode": hb.get("mode"), "heartbeat_source": hb.get("source"),
              "heartbeat_age_s": None if hb_age is None else int(hb_age),
              "heartbeat_stale_s": heartbeat_stale_s,
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
    elif market["open"] and hb_age > heartbeat_stale_s:
        reasons.append(f"no completed scan in {int(hb_age/60)}m — expected about "
                       f"every {interval_min}m; inspect the engine phase")
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
    block_detail = None
    if blocked and (not fill or fill["id"] < blocked["id"]):
        payload = _json.loads(blocked["payload"] or "{}")
        detail = payload.get("response") or ", ".join(payload.get("warnings", []))
        block_detail = _redact(str(detail)[:260])
        reasons.append(f"broker blocked entries — {block_detail}")

    # last engine-level failure, sanitized: what broke most recently and how
    # long ago. A fresh one degrades app health (rollup); a stale one is
    # history. Never a raw traceback over HTTP.
    err = store.db.execute(
        "SELECT ts, event_type, payload FROM audit WHERE event_type IN "
        "('scheduler_error','broker_probe_failed') "
        "ORDER BY id DESC LIMIT 1").fetchone()
    last_error = None
    if err:
        p = _json.loads(err["payload"] or "{}")
        detail = p.get("error") or p.get("detail") or ""
        age = _age_s(err["ts"])
        last_error = {"event": err["event_type"], "at": err["ts"],
                      "age_s": None if age == float("inf") else int(age),
                      "detail": _redact(str(detail)[:200])}

    out = {"mode": mode, "broker": broker, "engine": engine, "market": market,
           "data": data, "kill_switches": sorted(switches),
           "broker_block": block_detail, "last_error": last_error,
           "pending_approvals": len(store.pending_approvals()),
           "readiness": {"trading": not reasons, "reasons": reasons},
           "as_of": _now_iso()}
    out["status"], out["alerts"] = rollup(out)
    return out


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _age_s(iso: str | None) -> float:
    if not iso:
        return float("inf")
    try:
        return (datetime.now().astimezone() - datetime.fromisoformat(iso)).total_seconds()
    except ValueError:
        return float("inf")
