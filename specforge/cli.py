"""CLI entry point. `specforge <cmd>` (or `.venv/bin/python -m specforge.cli`).

Commands:
  data       refresh daily bars (--full re-pulls entire history)
  scan       run one full scan cycle (paper unless --mode live)
  status     account, kill switches, projection, pending approvals
  backtest   walk-forward backtest (--years N) → report + analog trades
  serve      start the GUI (FastAPI on --port)
  approve/reject <intent_id>   decide a queued order
  reset-kill <name>            clear a manual kill switch after review
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .store import Store


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
    summary = run_cycle(cfg, store, as_of=args.as_of,
                        refresh_data=not args.no_refresh)
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
    import uvicorn
    from .app import create_app
    uvicorn.run(create_app(cfg, store), host="127.0.0.1", port=args.port)


def cmd_approve(args, cfg, store):
    store.decide_approval(args.intent_id, "approved")
    store.audit("approval_decided", {"intent": args.intent_id, "decision": "approved",
                                     "via": "cli"})
    print(f"approved {args.intent_id}; it will place on the next scan cycle")


def cmd_reject(args, cfg, store):
    store.decide_approval(args.intent_id, "rejected")
    store.update_order(args.intent_id, status="rejected")
    store.audit("approval_decided", {"intent": args.intent_id, "decision": "rejected",
                                     "via": "cli"})
    print(f"rejected {args.intent_id}")


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


def main(argv=None):
    p = argparse.ArgumentParser(prog="specforge")
    p.add_argument("--mode", default=None, help="config overlay: paper|live")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data"); s.add_argument("--full", action="store_true")
    s = sub.add_parser("scan")
    s.add_argument("--as-of", default=None)
    s.add_argument("--no-refresh", action="store_true")
    sub.add_parser("status")
    s = sub.add_parser("backtest")
    s.add_argument("--years", type=int, default=10)
    s.add_argument("--tag", default="default")
    s.add_argument("--save-analogs", action="store_true", default=True)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8420)
    s = sub.add_parser("approve"); s.add_argument("intent_id")
    s = sub.add_parser("reject"); s.add_argument("intent_id")
    s = sub.add_parser("reset-kill"); s.add_argument("name")
    sub.add_parser("bridge-dump")
    s = sub.add_parser("bridge-report")
    s.add_argument("--file", default="-", help="results JSON path, or - for stdin")

    args = p.parse_args(argv)
    cfg = load_config(args.mode)
    store = _store(cfg)
    return globals()[f"cmd_{args.cmd.replace('-', '_')}"](args, cfg, store)


if __name__ == "__main__":
    sys.exit(main())
