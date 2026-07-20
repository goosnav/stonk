"""CLI entry point. `stonk <cmd>` (or `.venv/bin/python -m specforge.cli`).

Commands:
  data       refresh daily bars (--full re-pulls entire history)
  scan       run one full scan cycle (paper unless --mode live)
  status     account, kill switches, projection, pending approvals
  backtest   walk-forward backtest (--years N) → report + analog trades
  research   inspect or run the bounded closed-market research queue
  tui        quiet terminal dashboard; attaches to or runs the daemon
  serve      start the quiet GUI/headless server (FastAPI on --port)
  approve/reject <intent_id>   decide a queued order
  reset-kill <name>            clear a manual kill switch after review
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from .config import load_config
from .store import Store, configure_file_logging


def _store(cfg) -> Store:
    return Store(cfg.get("db_path", default="data/specforge.db"))


def cmd_data(args, cfg, store):
    from .data import refresh
    symbols = list(cfg.get("universe", "symbols", default=[]))
    symbols.append(cfg.get("universe", "vix_symbol", default="^VIX"))
    res = refresh(store, symbols, full=args.full)
    ok = sum(1 for v in res.values() if v)
    print(f"refreshed {ok}/{len(res)} symbols "
          f"({sum(res.values())} rows)")


def cmd_scan(args, cfg, store):
    from .engine import run_cycle
    from .health import write_heartbeat
    summary = run_cycle(cfg, store, as_of=args.as_of,
                        refresh_data=not args.no_refresh)
    write_heartbeat(store, summary["cycle_id"], cfg.mode, source="cron")
    if getattr(args, "post_close", False):
        # cron run-model equivalent of the serve post-close job (attribution)
        from .attribution import propose_promotions, update_weights
        update_weights(cfg, store)
        props = propose_promotions(cfg, store)
        if props:
            store.kv_set("promotion_proposals", props)
            store.audit("promotion_proposals", props)
    print(json.dumps(summary, indent=2))


def cmd_status(args, cfg, store):
    from .broker.base import make_broker
    from .data import MarketContext
    from .forecast import portfolio_projection
    from .risk import Governor
    ctx = MarketContext(store, cfg)
    broker = make_broker(cfg, store)
    if hasattr(broker, "set_quotes"):
        broker.set_quotes(ctx.prices())
    acct = broker.get_account()
    gov = Governor(cfg, store)
    out = {
        "mode": cfg.mode, "broker": cfg.get("broker"),
        "equity": round(acct.equity, 2), "cash": round(acct.cash, 2),
        "positions": [{"symbol": p.symbol, "qty": p.qty, "avg_cost": p.avg_cost}
                      for p in acct.positions],
        "kill_switches": gov.active_switches(),
        "options_unlocked": gov.options_unlocked(acct),
        "pending_approvals": store.pending_approvals(),
        "projection": portfolio_projection(store, cfg.mode),
    }
    print(json.dumps(out, indent=2, default=str))


def cmd_backtest(args, cfg, store):
    from .backtest import run_backtest
    report = run_backtest(cfg, years=args.years, tag=args.tag, scale=args.scale,
                          copy_analogs_to=store if args.save_analogs else None)
    print(json.dumps(report, indent=2, default=str))


def cmd_research(args, cfg, store):
    from .research import (enqueue_job, list_jobs, run_next, run_operator_job,
                           status)
    if args.enqueue:
        print(json.dumps(enqueue_job(store, args.enqueue,
                                     requested_by="operator", force=args.force),
                         indent=2, default=str))
    elif args.jobs:
        print(json.dumps(list_jobs(store, 50), indent=2, default=str))
    elif args.status:
        print(json.dumps(status(store, cfg), indent=2, default=str))
    else:
        result = run_operator_job(cfg, store) or \
            run_next(cfg, store, max_seconds=args.max_minutes * 60)
        print(json.dumps(result,
                         indent=2, default=str))


def cmd_bars_audit(args, cfg, store):
    """Report — and optionally repair — adjustment seams in the bars table.

    Report-only by default on purpose: a repair rewrites stored price history,
    so the operator should see what would change before it does.
    """
    from . import bars_audit
    report = bars_audit.audit(store, symbols=args.symbols, repair=args.repair,
                              limit=args.limit)
    print(f"scanned {report['symbols_scanned']} symbols")
    print(f"  seams (auto-repairable): {report['seams']} "
          f"across {len(report['symbols_affected'])} symbols")
    print(f"  suspicious (reported, never auto-repaired): {report['suspicious']}")
    seams = [f for f in report["findings"] if f["kind"] == "seam"]
    for finding in sorted(seams, key=lambda f: -abs(f["ratio"]))[:args.show]:
        print(f"    {finding['symbol']:8} {finding['d']}  x{finding['ratio']:<12.4f}"
              f" {finding['prior_close']} -> {finding['close']}")
    if not args.repair:
        if seams:
            print("\nreport only. re-run with --repair to rewrite these symbols.")
        return None
    print(f"\nrepaired {len(report['repaired'])} symbols")
    failed = [r for r in report["repaired"] if r["status"] != "repaired"]
    for r in failed:
        print(f"    FAILED {r['symbol']}: {r['status']} {r.get('error', '')}")
    if report["still_seamed"]:
        # The provider itself is serving a mixed series; retrying will not help.
        print(f"    STILL SEAMED after refetch: {report['still_seamed']}")
    return None


def cmd_worker(args, cfg, store):
    """Internal one-shot worker used by the scheduler's process boundary."""
    from .config import load_config_with_stored_overrides
    cfg = load_config_with_stored_overrides(cfg.mode, store)
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    if args.lane == "autonomous":
        from .research import run_next
        result = run_next(cfg, store, max_seconds=args.max_seconds)
    elif args.lane.startswith("operator-"):
        from .research import run_operator_job
        resource = args.lane.removeprefix("operator-")
        result = run_operator_job(cfg, store, {resource})
    elif args.lane == "news":
        from .intelligence import recover, run_next
        recover(store)
        result = run_next(cfg, store)
    elif args.lane == "weekend":
        from datetime import datetime
        from .backtest import run_backtest
        report = run_backtest(cfg, years=2, tag="weekly_research")
        result = {"window": report.get("window"), "n_trades": report.get("n_trades"),
                  "win_rate": report.get("win_rate"), "overall": report.get("overall"),
                  "out_of_sample_30pct": report.get("out_of_sample_30pct"),
                  "benchmark_buy_hold_return": report.get("benchmark_buy_hold_return"),
                  "ran_at": datetime.now().astimezone().isoformat(timespec="seconds")}
        store.kv_set("research_report", result)
        store.audit("weekend_research", result)
    else:  # argparse constrains this; retain a fail-closed worker boundary.
        raise ValueError(f"unknown worker lane {args.lane}")
    print(json.dumps(result, default=str))
    return 0


def cmd_serve(args, cfg, store):
    import socket

    import uvicorn
    from .app import create_app
    instance_key = f"service_instance:{cfg.mode}"
    instance_token = f"{os.getpid()}:{uuid.uuid4().hex}"
    try:
        store.db.execute("BEGIN IMMEDIATE")
        row = store.db.execute("SELECT value FROM kv WHERE key=?", (instance_key,)).fetchone()
        previous = json.loads(row["value"]) if row else {}
        previous_pid = int(previous.get("pid", 0) or 0)
        alive = False
        if previous_pid and previous_pid != os.getpid():
            try:
                os.kill(previous_pid, 0); alive = True
            except (ProcessLookupError, PermissionError):
                pass
        if alive:
            store.db.rollback()
            print(f"a {cfg.mode} Stonk service is already running (pid {previous_pid}); "
                  "refusing to start a second trading scheduler", file=sys.stderr)
            return 2
        payload = json.dumps({"pid": os.getpid(), "token": instance_token})
        store.db.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) "
                         "DO UPDATE SET value=excluded.value", (instance_key, payload))
        store.db.commit()
    except Exception:
        store.db.rollback()
        raise
    def release_instance():
        try:
            store.db.execute("BEGIN IMMEDIATE")
            row = store.db.execute("SELECT value FROM kv WHERE key=?", (instance_key,)).fetchone()
            current = json.loads(row["value"]) if row else {}
            if current.get("token") == instance_token:
                store.db.execute("DELETE FROM kv WHERE key=?", (instance_key,))
            store.db.commit()
        except Exception:
            store.db.rollback()
    store.scrub_audit_history()
    # Pre-bind once and pass the socket into Uvicorn: choosing a fallback and
    # releasing it before server startup creates a check-then-bind race.
    listener = None
    for port in range(args.port, args.port_range_end + 1):
        candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            candidate.bind(("127.0.0.1", port))
            candidate.listen(128)
            listener = candidate
            break
        except OSError:
            candidate.close()
    if listener is None:
        print(f"no free loopback port in {args.port}-{args.port_range_end}", file=sys.stderr)
        release_instance()
        return 2
    effective_port = listener.getsockname()[1]
    # Publish the bound port so alternate headless clients can attach instead
    # of starting a second scheduler when the preferred port was occupied.
    try:
        store.db.execute("BEGIN IMMEDIATE")
        row = store.db.execute("SELECT value FROM kv WHERE key=?", (instance_key,)).fetchone()
        current = json.loads(row["value"]) if row else {}
        if current.get("token") == instance_token:
            current["effective_port"] = effective_port
            store.db.execute("UPDATE kv SET value=? WHERE key=?",
                             (json.dumps(current), instance_key))
        store.db.commit()
    except Exception:
        store.db.rollback()
        listener.close(); release_instance(); raise
    detail = {"mode": cfg.mode, "preferred_port": args.port,
              "effective_port": effective_port,
              "fallback": effective_port != args.port}
    store.audit("service_starting", detail)
    print(f"STONK_URL=http://127.0.0.1:{effective_port}", flush=True)
    app = None
    try:
        app = create_app(cfg, store)
        server = uvicorn.Server(uvicorn.Config(
            app, log_level="info" if args.verbose else "warning",
            access_log=args.verbose))
        server.run(sockets=[listener])
    except KeyboardInterrupt:
        pass
    finally:
        scheduler = getattr(app.state, "scheduler", None) if app else None
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)
        listener.close()
        release_instance()
        store.audit("service_stopped", {"mode": cfg.mode,
                                         "effective_port": effective_port})


def cmd_approve(args, cfg, store):
    try:
        store.decide_approval(args.intent_id, "approved")
    except ValueError as e:                        # approving an expired intent
        store.audit("approval_expired", {"intent": args.intent_id, "via": "cli"})
        print(f"REFUSED: {e}")
        return
    store.audit("approval_decided", {"intent": args.intent_id, "decision": "approved",
                                     "via": "cli"})
    print(f"approved {args.intent_id}; it will place on the next scan cycle")


def cmd_reject(args, cfg, store):
    store.decide_approval(args.intent_id, "rejected")
    store.update_order(args.intent_id, status="rejected")
    store.audit("approval_decided", {"intent": args.intent_id, "decision": "rejected",
                                     "via": "cli"})
    print(f"rejected {args.intent_id}")


def _console_api(port: int, path: str, timeout: int = 20):
    from urllib.request import urlopen
    with urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
        return json.load(response)


def _console_server(args, cfg, store):
    """Attach to the published service, or start it through the same lock."""
    try:
        version = _console_api(args.port, "/api/version", timeout=2)
        if version.get("mode") != cfg.mode:
            raise RuntimeError(f"port {args.port} is serving {version.get('mode')} mode; "
                               f"requested {cfg.mode}")
        return None, args.port
    except RuntimeError:
        raise
    except Exception:
        instance = store.kv_get(f"service_instance:{cfg.mode}") or {}
        published_port = int(instance.get("effective_port", 0) or 0)
        published_pid = int(instance.get("pid", 0) or 0)
        if published_port and published_pid:
            try:
                os.kill(published_pid, 0)
                version = _console_api(published_port, "/api/version", timeout=2)
                if version.get("mode") == cfg.mode:
                    return None, published_port
            except (ProcessLookupError, PermissionError, OSError):
                pass
        # Starting via `serve` preserves its cross-process instance lock and
        # atomic fallback-port publication; the TUI never owns a hidden second
        # scheduler again.
        import subprocess
        import time
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log = (log_dir / f"tui-service-{cfg.mode}.log").open("a", encoding="utf-8")
        server = subprocess.Popen(
            [sys.executable, "-m", "specforge.cli", "--mode", cfg.mode,
             "serve", "--port", str(args.port), "--port-range-end", str(args.port + 10)],
            cwd=Path(__file__).resolve().parent.parent,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            shell=False, start_new_session=True)
        log.close()
        for _ in range(100):
            try:
                instance = store.kv_get(f"service_instance:{cfg.mode}") or {}
                effective = int(instance.get("effective_port", 0) or args.port)
                version = _console_api(effective, "/api/version", timeout=1)
                if version.get("mode") == cfg.mode:
                    return server, effective
            except Exception:
                time.sleep(0.1)
            if server.poll() is not None:
                break
        if server.poll() is None:
            import signal
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        raise RuntimeError(f"headless server did not start on port {args.port}")


def _money(value) -> str:
    return "—" if value is None else f"${value:,.2f}"


def _pct(value) -> str:
    return "—" if value is None else f"{value:+.2%}"


def _console_frame(port: int, color: bool) -> str:
    from datetime import datetime
    h = _console_api(port, "/api/health")
    s = _console_api(port, "/api/status")
    e = _console_api(port, "/api/engine")
    today = _console_api(port, "/api/today")
    decisions = _console_api(port, "/api/decisions")
    try:
        research = _console_api(port, "/api/research")
    except Exception:
        research = {}
    try:
        live_model = _console_api(port, "/api/model/graph?symbol=SPY&horizon=21")
    except Exception:
        live_model = {}
    try:
        strategy_state = _console_api(port, "/api/strategy/directives")
        intelligence_state = _console_api(port, "/api/intelligence/jobs")
        routing_state = _console_api(port, "/api/ai/health")
    except Exception:
        strategy_state, intelligence_state, routing_state = {}, {}, {}

    G, R, A, D, B, X = (("\033[32m", "\033[31m", "\033[33m", "\033[2m",
                          "\033[1m", "\033[0m") if color else ("",) * 6)
    rd, broker, engine = h["readiness"], h["broker"], h["engine"]
    hb = engine.get("heartbeat_age_s")
    heartbeat = "never" if hb is None else (f"{hb}s" if hb < 120 else f"{hb//60}m")
    phase = engine.get("operational_state") or (e.get("state") or {}).get("phase", "unknown")
    deployment = 1 - s["cash"] / s["equity"] if s.get("equity") else 0
    state = G + "TRADING" + X if rd["trading"] else A + "PAUSED" + X

    lines = [f"{B}STONK TERMINAL · {s['mode'].upper()} · "
             f"{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}{X}",
             f"{state}  broker:{broker['adapter']} "
             f"{'OK' if broker['connected'] else 'DOWN'}  "
             f"market:{h['market']['session']} {h['market']['et']}  "
             f"autonomy:{phase}" +
             (f"  last-scan:{heartbeat} ago" if h["market"].get(
                 "open", h["market"].get("session") == "regular") else ""),
             f"{B}ACCOUNT{X}  equity {_money(s['equity'])}  cash {_money(s['cash'])}  "
             f"buying-power {_money(s.get('buying_power'))}  deployed {deployment:.1%}",
             f"P&L  day {_money(s.get('day_pnl'))}  total {_money(s.get('net_pnl'))}  "
             f"realized {_money(s.get('realized_pnl'))}  unrealized {_money(s.get('unrealized_pnl'))}  "
             f"drawdown {s.get('drawdown_from_peak', 0):.2%}",
             f"REGIME {s['regime']} ×{s['deployment_multiplier']}  "
             f"cycle risk ceiling {_money(s['cycle_budget'])}  approvals {h['pending_approvals']}"]

    if not rd["trading"]:
        lines.append(R + "WHY PAUSED  " + " | ".join(rd.get("reasons") or ["unknown"]) + X)

    lines += ["", B + f"POSITIONS ({len(s['positions'])})" + X,
              D + "SYMBOL       QTY          AVG       LAST      VALUE       P&L" + X]
    for p in s["positions"][:12]:
        pnl_color = G if p["pnl_usd"] >= 0 else R
        lines.append(f"{p['symbol']:<10} {p['qty']:>10.6f}  {p['avg_cost']:>9.2f}  "
                     f"{p['last']:>9.2f}  {p['value']:>9.2f}  "
                     f"{pnl_color}{p['pnl_usd']:>+8.2f} ({p['pnl_pct']:+.2%}){X}")
    if not s["positions"]:
        lines.append(D + "no open positions" + X)

    working = decisions.get("working") or []
    lines += ["", B + f"WORKING ORDERS ({len(working)})" + X]
    for o in working[:8]:
        lines.append(f"{o['symbol']:<8} {o['side']:<4} {_money(o['notional']):>10}  "
                     f"{o['status']:<16} qty {o['qty']:.6f}")
    if not working:
        lines.append(D + "none" + X)

    cycle = decisions.get("cycle") or {}
    considered = decisions.get("considered") or []
    lines += ["", B + "ENGINE / LAST CYCLE" + X,
              f"{(e.get('state') or {}).get('detail', '—')}"]
    if cycle:
        lines.append(f"cycle {cycle.get('id')}  candidates {len(considered)}  "
                     f"budget {_money(cycle.get('budget_used'))}/{_money(cycle.get('budget'))}")
    for c in considered[:5]:
        verdict = c.get("result") or c.get("verdict") or "not_selected"
        evidence = " ".join(
            f"{item.get('node')}:{float(item.get('signed_alpha', 0)):+.2f}"
            for item in (c.get("evidence") or [])
            if item.get("state") == "running" and
            not str(item.get("node", "")).startswith("family:"))
        lines.append(f"{c['symbol']:<6} score {c['score']:.3f}  {verdict:<18} "
                     f"coverage {float(c.get('evidence_coverage') or 0):.0%}  "
                     f"{evidence[:95]}")

    rs = research.get("state") or {}
    us = research.get("universe") or {}
    tiers = us.get("tiers") or {}
    graph = research.get("graph") or {}
    active_jobs = [j for j in research.get("jobs", [])
                   if j.get("status") in ("queued", "running")]
    lines += ["", B + "OFF-HOURS RESEARCH" + X,
              f"{rs.get('phase', 'unknown')}: {rs.get('detail', '—')}  "
              f"catalog {((us.get('catalog') or {}).get('count') or 0):,} · "
              f"research {(tiers.get('research') or 0):,} · active {(tiers.get('active') or 0):,}",
              f"analog graph {graph.get('id', 'default')} {graph.get('status', 'shadow')} · "
              f"TCN {((research.get('neural') or {}).get('status') or 'shadow')} · "
              f"PRODUCTION EVIDENCE {live_model.get('production_evidence_version', 'evidence.v2')} · "
              f"LEARNED stage {live_model.get('ramp_stage', 0)} "
              f"{float(live_model.get('effective_live_blend', 0)):.0%} "
              f"{live_model.get('learned_block_reason') or 'validated'}"]
    evidence_status = research.get("company_evidence") or {}
    evidence_budget = evidence_status.get("budget") or {}
    lines.append(f"company dossiers {evidence_status.get('counts') or {}} · "
                 f"AI month {_money(evidence_budget.get('spent_month_usd'))}/"
                 f"{_money(evidence_budget.get('monthly_budget_usd'))}")
    for job in active_jobs:
        p = job.get("progress") or {}
        detail = (f"{p.get('phase', '')} {p.get('symbol', '')} "
                  f"{p.get('index', '')}/{p.get('total', '')}").strip(" /")
        wait = job.get("wait_reason") or ""
        lines.append(f"{job.get('resource_class', 'research'):<12} "
                     f"{job['kind']} {job.get('state') or job['status']} · "
                     f"{detail or wait or 'queued'}")

    mandate = strategy_state.get("active") or {}
    mandate_payload = mandate.get("payload") or {}
    intel_job = next((j for j in intelligence_state.get("jobs", [])
                      if j.get("status") in ("queued", "running")), None)
    last_intel = routing_state.get("last_call") or intelligence_state.get("state") or {}
    lines += ["", B + "STRATEGY / INTELLIGENCE" + X,
              (f"active mandate {mandate.get('id', '')[:8]} · "
               f"{mandate_payload.get('summary') or mandate_payload.get('thesis', '')[:100]}"
               if mandate else "no active Strategy AI mandate; raw operator text has no effect"),
              f"route {routing_state.get('default_provider', '—')} · last "
              f"{last_intel.get('provider', '—')}/{last_intel.get('model', '—')} "
              f"{last_intel.get('purpose', '')} "
              f"{'OK' if last_intel.get('ok') else 'idle/unavailable'}"]
    if intel_job:
        progress = intel_job.get("progress") or {}
        lines.append(f"intelligence {intel_job.get('kind')} {intel_job.get('status')} · "
                     f"{progress.get('phase', 'queued')} "
                     f"{progress.get('completed', 0)}/{progress.get('total', '—')}")

    switches = s.get("kill_switches") or {}
    lines += ["", B + "RISK / RECOVERY" + X]
    if not switches:
        lines.append(G + "no kill switches active" + X)
    for name, item in switches.items():
        recovery = (f"auto-resets {item.get('clear_at') or item.get('clear_on')}"
                    if item.get("auto_clear") else f"MANUAL: stonk reset-kill {name}")
        lines.append(f"{R}{name}{X}: {item.get('reason')} · {recovery}")

    news = (today.get("news") or {}).get("items") or []
    fundamentals = (today.get("fundamentals") or {}).get("items") or []
    lines += ["", B + "AI COMMENTARY" + X]
    if today.get("hypothesis"):
        lines.append("hypothesis: " + today["hypothesis"])
    for item in news[:3]:
        lines.append(f"news {item['symbol']:<5} {item.get('sentiment', 0):+0.2f}  "
                     f"{item.get('summary', '')[:100]}")
    for item in fundamentals[:2]:
        lines.append(f"fund {item['symbol']:<5} {item.get('direction', '?'):<7}  "
                     f"{item.get('summary', '')[:100]}")
    if not news and not fundamentals and not today.get("hypothesis"):
        lines.append(D + "AI disabled, over budget, or no current commentary" + X)

    lines += ["", D + "Ctrl-C exits · --once prints one agent-friendly snapshot · "
              f"verbose diagnostics stay in logs/audit-{s.get('mode', 'paper')}.jsonl" + X]
    return "\n".join(lines)


def cmd_tui(args, cfg, store):
    """Quiet operator console. Attaches to a server or becomes the daemon."""
    import time
    server, effective_port = _console_server(args, cfg, store)
    color = bool(sys.stdout.isatty() and not args.no_color and not args.once)
    try:
        while True:
            try:
                frame = _console_frame(effective_port, color)
            except Exception as e:                 # transient feed/API errors heal
                frame = ("STONK TERMINAL · CONSOLE DEGRADED\n"
                         f"{type(e).__name__}: {e}\nretrying in {args.interval}s; "
                         "autonomous server remains running")
            print(("\033[2J\033[H" if color else "") + frame, flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            import signal
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass


def cmd_bridge_dump(args, cfg, store):
    from .broker.bridge import bridge_dump
    print(json.dumps(bridge_dump(store, cfg), indent=2, default=str))


def cmd_bridge_report(args, cfg, store):
    from .broker.bridge import bridge_report
    payload = json.loads(Path(args.file).read_text() if args.file != "-"
                         else sys.stdin.read())
    print(json.dumps(bridge_report(store, payload)))


def cmd_reset_kill(args, cfg, store):
    from .risk import Governor
    Governor(cfg, store).reset(args.name)
    print(f"kill switch '{args.name}' cleared")


def cmd_hypothesis(args, cfg, store):
    """Manual hypothesis upkeep (V4/D34): sweep + bootstrap/rotate/review via
    steering. Same code path the post-close scheduler runs."""
    from .steering import maintain
    print(json.dumps(maintain(cfg, store), indent=2, default=str))


def cmd_ai(args, cfg, store):
    from .ai import AIClient
    client = AIClient(cfg, store)
    if args.action == "doctor":
        out = client.status(probe_auth=True)
    elif args.action == "test":
        result = client.complete_json(
            "headline_classification", "cli_provider_test",
            "Return only the requested JSON. This is a harmless connectivity test.",
            "Return sentiment 0, confidence 1, catalyst 'test', horizon_days 1, "
            "already_priced true, summary 'provider route operational'.", 120)
        out = {"ok": result is not None, "result": result,
               "last_call": store.kv_get("intelligence_last_call")}
    else:
        out = client.status(probe_auth=False)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out.get("available", out.get("ok", False)) else 1


def cmd_strategy(args, cfg, store):
    from . import strategy
    if args.action == "analyze":
        msg = strategy.submit(store, args.text)
        out = strategy.analyze(cfg, store, msg["id"])
    elif args.action == "activate":
        out = strategy.activate(store, args.id or args.text)
    elif args.action == "deactivate":
        out = strategy.deactivate(store, args.id or args.text)
    else:
        out = {"active": strategy.active(store), "messages": strategy.messages(store),
               "mandates": strategy.mandates(store)}
    print(json.dumps(out, indent=2, default=str))


def cmd_intelligence(args, cfg, store):
    from .intelligence import jobs
    print(json.dumps({"jobs": jobs(store),
                      "last_call": store.kv_get("intelligence_last_call"),
                      "news": store.kv_get("news_intelligence")}, indent=2, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(prog="stonk")
    p.add_argument("--mode", default=None, help="config overlay: paper|live")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data"); s.add_argument("--full", action="store_true")
    s = sub.add_parser("scan")
    s.add_argument("--as-of", default=None)
    s.add_argument("--no-refresh", action="store_true")
    s.add_argument("--post-close", action="store_true", dest="post_close",
                   help="also run attribution/weight updates (cron run-model)")
    s = sub.add_parser("tui")
    s.add_argument("--interval", type=int, default=5)
    s.add_argument("--port", type=int, default=8420)
    s.add_argument("--once", action="store_true",
                   help="print one stable snapshot for agents/scripts, then exit")
    s.add_argument("--no-color", action="store_true")
    sub.add_parser("status")
    s = sub.add_parser("backtest")
    s.add_argument("--years", type=int, default=10)
    s.add_argument("--tag", default="default")
    s.add_argument("--scale", choices=("live", "research"), default="research")
    s.add_argument("--save-analogs", action="store_true", default=True)
    s = sub.add_parser("research")
    s.add_argument("--status", action="store_true")
    s.add_argument("--jobs", action="store_true",
                   help="show durable operator jobs with progress and wait reasons")
    s.add_argument("--enqueue", choices=("discover", "deep_research", "train_holdings"),
                   help="queue an operator research action")
    s.add_argument("--force", action="store_true",
                   help="legacy flag; market-hours training is intentionally rejected")
    s.add_argument("--max-minutes", type=int, default=10)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8420)
    s.add_argument("--port-range-end", type=int, default=8420,
                   help="last loopback port allowed for atomic startup fallback")
    s.add_argument("--verbose", action="store_true",
                   help="stream HTTP access logs (default is quiet; audit stays on disk)")
    s = sub.add_parser("approve"); s.add_argument("intent_id")
    s = sub.add_parser("reject"); s.add_argument("intent_id")
    s = sub.add_parser("reset-kill"); s.add_argument("name")
    sub.add_parser("hypothesis")
    s = sub.add_parser("ai")
    s.add_argument("action", choices=("status", "doctor", "test"), default="status",
                   nargs="?")
    s.add_argument("--json", action="store_true", help="stable JSON output (default)")
    s = sub.add_parser("strategy")
    s.add_argument("action", choices=("status", "analyze", "activate", "deactivate"))
    s.add_argument("text", nargs="?", default="")
    s.add_argument("--id", default="")
    s = sub.add_parser("intelligence")
    s.add_argument("--status", action="store_true", default=True)
    s = sub.add_parser("worker", help=argparse.SUPPRESS)
    s.add_argument("lane", choices=("autonomous", "operator-discovery",
                                    "operator-intelligence", "operator-training",
                                    "news", "weekend"))
    s.add_argument("--max-seconds", type=int, default=600)
    sub.add_parser("bridge-dump")
    s = sub.add_parser("bridge-report")
    s.add_argument("--file", default="-", help="results JSON path, or - for stdin")

    s = sub.add_parser("bars-audit",
                       help="find split-adjustment seams in stored bars")
    s.add_argument("--repair", action="store_true",
                   help="rewrite affected symbols from a single full fetch "
                        "(default is report-only: this changes price history)")
    s.add_argument("--symbols", nargs="*",
                   help="limit to these symbols (default: every symbol with bars)")
    s.add_argument("--limit", type=int,
                   help="repair at most N symbols this run")
    s.add_argument("--show", type=int, default=20,
                   help="how many findings to print")

    args = p.parse_args(argv)
    if args.cmd == "tui" and args.mode is None:
        try:
            args.mode = _console_api(args.port, "/api/version", timeout=1).get("mode")
        except Exception:
            pass
    cfg = load_config(args.mode)
    configure_file_logging(cfg.mode)
    store = _store(cfg)
    return globals()[f"cmd_{args.cmd.replace('-', '_')}"](args, cfg, store)


if __name__ == "__main__":
    sys.exit(main())
