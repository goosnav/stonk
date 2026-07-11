#!/usr/bin/env python3
"""Stonk Terminal operator health check — stdlib only, read-only, no venv
needed. The judgment contract Hermes, cron watchdogs, and humans share.

    scripts/check_health.py [--url http://127.0.0.1:8420] [--json]
                            [--timeout 15] [--quiet]

Exit codes (a watchdog restarts ONLY on 2/3; 1 means page the operator,
never bounce the app — the problem is broker/data/safe-mode, not the process):
  0  ok         app healthy; "not trading" for benign reasons (market closed,
                paper mode) still exits 0
  1  degraded   app alive but needs the OPERATOR: broker disconnected/blocked,
                kill switch active, stale market data, fresh cycle error
  2  app failure  scans stale with the market open, or scheduler dead
  3  down       server unreachable (connection refused / timeout)
  4  malformed  server answered but not with the expected contract

Reads /api/metrics (new servers) and falls back to /api/health (older
builds). Prints one human line by default; --json emits the full verdict.
Never mutates the app; never prints secrets (defense-in-depth redaction).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime

DEFAULT_URL = "http://127.0.0.1:8420"
STATUS_EXIT = {"ok": 0, "degraded": 1, "stale": 2, "error": 2, "down": 3,
               "malformed": 4}
_ORDER = {"ok": 0, "degraded": 1, "stale": 2, "error": 3}

# mirror of specforge.health._SECRETISH — this script must run standalone
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


def evaluate(h: dict) -> tuple[str, list[str]]:
    """App-health verdict from an /api/health payload. New servers ship
    status/alerts precomputed; otherwise compute the same rollup client-side.
    ponytail: the fallback branch is a compat shim for pre-metrics servers —
    delete once the live server has restarted onto the 2026-07-10 build."""
    if h.get("status") in _ORDER and isinstance(h.get("alerts"), list):
        return h["status"], h["alerts"]

    status, alerts = "ok", []

    def worse(sev: str, msg: str) -> None:
        nonlocal status
        alerts.append(_redact(msg))
        if _ORDER[sev] > _ORDER[status]:
            status = sev

    b = h.get("broker") or {}
    if not b.get("connected"):
        worse("degraded", f"broker disconnected: {b.get('detail') or 'unknown'}")
    for name in h.get("kill_switches") or []:
        worse("degraded", f"kill switch active: {name}")
    d = h.get("data") or {}
    limit = d.get("stale_limit_days") or 4
    if d.get("age_days") is not None and d["age_days"] > limit:
        worse("degraded", f"market data stale ({d['age_days']}d old, limit {limit}d)")
    e = h.get("engine") or {}
    stale_s = e.get("heartbeat_stale_s") or 1800
    if (h.get("market") or {}).get("open"):
        hb = e.get("heartbeat_age_s")
        if hb is None or hb > stale_s:
            worse("stale", "no completed scan "
                  + (f"in {int(hb / 60)}m" if hb is not None else "ever")
                  + " with the market open")
    if e.get("scheduler_alive") is False:
        worse("error", "scheduler is not running in the serving process")
    return status, alerts


def _fetch(url: str, timeout: float) -> dict | None:
    """GET url as JSON. None on HTTP 404 (endpoint doesn't exist yet);
    raises urllib errors upward for down-detection."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=DEFAULT_URL, help="server base URL")
    p.add_argument("--timeout", type=float, default=15.0,
                   help="seconds; /api/health can take ~3s on a cold broker probe")
    p.add_argument("--json", action="store_true", dest="as_json")
    p.add_argument("--quiet", action="store_true", help="exit code only")
    args = p.parse_args(argv)
    base = args.url.rstrip("/")
    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")

    out = {"checked_at": checked_at, "url": base}
    try:
        metrics = _fetch(f"{base}/api/metrics", args.timeout)
        health = metrics["health"] if metrics else _fetch(f"{base}/api/health",
                                                          args.timeout)
        if health is None:
            raise KeyError("no /api/metrics or /api/health endpoint")
        status, alerts = evaluate(health)
        out.update(
            status=status, alerts=alerts, source="metrics" if metrics else "health",
            mode=health.get("mode"),
            trading=(health.get("readiness") or {}).get("trading"),
            not_trading_reasons=(health.get("readiness") or {}).get("reasons", []),
            market=(health.get("market") or {}).get("session"),
            broker_adapter=(health.get("broker") or {}).get("adapter"),
            broker_connected=(health.get("broker") or {}).get("connected"),
            heartbeat_age_s=(health.get("engine") or {}).get("heartbeat_age_s"),
            kill_switches=health.get("kill_switches") or [],
            last_error=health.get("last_error"),
        )
        if metrics:
            out.update(uptime_s=metrics.get("process", {}).get("uptime_s"),
                       pid=metrics.get("process", {}).get("pid"),
                       version=metrics.get("version"),
                       cycles_today=metrics.get("cycles", {}).get("today"),
                       errors_today=metrics.get("cycles", {}).get("errors_today"),
                       positions_open=metrics.get("positions_open"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        out.update(status="down", alerts=[_redact(f"unreachable: {e}")])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        out.update(status="malformed",
                   alerts=[_redact(f"unexpected response: {type(e).__name__}: {e}")])

    code = STATUS_EXIT[out["status"]]
    out["exit_code"] = code
    if args.quiet:
        return code
    if args.as_json:
        print(json.dumps(out, indent=2))
        return code

    line = f"STONK {out['status'].upper()}"
    if out.get("mode"):
        hb = out.get("heartbeat_age_s")
        line += (f" mode={out['mode']} market={out.get('market')} "
                 f"broker={out.get('broker_adapter')}"
                 f"{'✓' if out.get('broker_connected') else '✗'} "
                 f"trading={'yes' if out.get('trading') else 'no'} "
                 f"last-scan={'never' if hb is None else f'{hb // 60}m ago'}")
        if out.get("uptime_s") is not None:
            line += f" uptime={out['uptime_s'] // 3600}h{out['uptime_s'] % 3600 // 60}m"
    print(_redact(line))
    for a in out.get("alerts", []):
        print(f"  ! {_redact(str(a))}")
    if code == 0 and not out.get("trading"):
        for r in out.get("not_trading_reasons", [])[:3]:
            print(f"  · {_redact(str(r))}")
    return code


if __name__ == "__main__":
    sys.exit(main())
