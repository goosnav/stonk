"""CLI entry point. `stonk <cmd>` (or `.venv/bin/python -m specforge.cli`).

Commands:
  data       refresh daily bars (--full re-pulls entire history)
  scan       run one full scan cycle (paper unless --mode live)
  status     account, kill switches, projection, pending approvals
  backtest   walk-forward backtest (--years N) → report + analog trades
  tui        quiet terminal dashboard; attaches to or runs the daemon
  serve      start the quiet GUI/headless server (FastAPI on --port)
  approve/reject <intent_id>   decide a queued order
  reset-kill <name>            clear a manual kill switch after review
"""
from __future__ import annotations

import argparse
import json
import sys
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
    report = run_backtest(cfg, years=args.years, tag=args.tag,
                          copy_analogs_to=store if args.save_analogs else None)
    print(json.dumps(report, indent=2, default=str))


def cmd_serve(args, cfg, store):
    import socket

    import uvicorn
    from .app import create_app
    # Refuse to double-serve BEFORE create_app: a second instance would start
    # a second scheduler against the shared DB in the window before uvicorn
    # discovers the port is taken. Duplicate live engines must never race.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            print(f"port {args.port} is already serving — another Stonk Terminal "
                  f"instance? `stonk tui` attaches to it; "
                  f"scripts/check_health.py reports its health", file=sys.stderr)
            return 2
    store.audit("service_starting", {"mode": cfg.mode, "port": args.port})
    uvicorn.run(create_app(cfg, store), host="127.0.0.1", port=args.port,
                log_level="info" if args.verbose else "warning",
                access_log=args.verbose)


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
    """Attach to the local server, or quietly start it in this TUI process."""
    try:
        version = _console_api(args.port, "/api/version", timeout=2)
        if version.get("mode") != cfg.mode:
            raise RuntimeError(f"port {args.port} is serving {version.get('mode')} mode; "
                               f"requested {cfg.mode}")
        return None
    except RuntimeError:
        raise
    except Exception:                         # no server: TUI becomes the daemon
        import threading
        import time
        import uvicorn
        from .app import create_app
        server = uvicorn.Server(uvicorn.Config(
            create_app(cfg, store), host="127.0.0.1", port=args.port,
            log_level="warning", access_log=False))
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(100):
            try:
                _console_api(args.port, "/api/version", timeout=1)
                return server
            except Exception:
                time.sleep(0.1)
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

    G, R, A, D, B, X = (("\033[32m", "\033[31m", "\033[33m", "\033[2m",
                          "\033[1m", "\033[0m") if color else ("",) * 6)
    rd, broker, engine = h["readiness"], h["broker"], h["engine"]
    hb = engine.get("heartbeat_age_s")
    heartbeat = "never" if hb is None else (f"{hb}s" if hb < 120 else f"{hb//60}m")
    phase = (e.get("state") or {}).get("phase", "unknown")
    deployment = 1 - s["cash"] / s["equity"] if s.get("equity") else 0
    state = G + "TRADING" + X if rd["trading"] else A + "PAUSED" + X

    lines = [f"{B}STONK TERMINAL · {s['mode'].upper()} · "
             f"{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}{X}",
             f"{state}  broker:{broker['adapter']} "
             f"{'OK' if broker['connected'] else 'DOWN'}  "
             f"market:{h['market']['session']} {h['market']['et']}  "
             f"engine:{phase}  last-scan:{heartbeat} ago",
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
        lines.append(f"{c['symbol']:<6} score {c['score']:.3f}  {verdict:<18} "
                     f"{(c.get('thesis') or '')[:75]}")

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
              "verbose diagnostics stay in logs/audit-live.jsonl" + X]
    return "\n".join(lines)


def cmd_tui(args, cfg, store):
    """Quiet operator console. Attaches to a server or becomes the daemon."""
    import time
    server = _console_server(args, cfg, store)
    color = bool(sys.stdout.isatty() and not args.no_color and not args.once)
    try:
        while True:
            try:
                frame = _console_frame(args.port, color)
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
            server.should_exit = True


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
    s.add_argument("--save-analogs", action="store_true", default=True)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8420)
    s.add_argument("--verbose", action="store_true",
                   help="stream HTTP access logs (default is quiet; audit stays on disk)")
    s = sub.add_parser("approve"); s.add_argument("intent_id")
    s = sub.add_parser("reject"); s.add_argument("intent_id")
    s = sub.add_parser("reset-kill"); s.add_argument("name")
    sub.add_parser("hypothesis")
    sub.add_parser("bridge-dump")
    s = sub.add_parser("bridge-report")
    s.add_argument("--file", default="-", help="results JSON path, or - for stdin")

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
