"""Walk-forward backtest — runs THE SAME run_cycle() the live engine uses, once
per historical trading day, against a separate DB seeded with the main DB's
bars. Backtests are hostile evidence (AGENTS.md §24): costs included, no
lookahead (MarketContext as_of slicing + injected clock), OOS split reported,
SPY buy-hold comparison mandatory.

Outputs:
- report dict (also written to dev/reports/backtest_<tag>.json)
- analog trades (source='backtest') optionally copied into the live DB so
  forecast.py can put real error bars on live candidates from day one.

Known coverage caveats (also in dev/PROGRESS.md): earnings_drift only has
~2 years of yfinance history; quality_value uses CURRENT fundamentals snapshots
(survivor-ish bias for the filter role — acceptable for a veto-only node on a
megacap universe, revisit if the universe widens).
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from .engine import run_cycle
from .nodes.base import build_registry
from .store import Store

WARMUP_BARS = 270          # momentum needs 260; give margin


def _seed_backtest_db(src_path: str, bt_path: Path) -> Store:
    if bt_path.exists():
        bt_path.unlink()
    bt = Store(bt_path)
    bt.db.execute("ATTACH DATABASE ? AS src", (str(src_path),))
    bt.db.execute("INSERT INTO bars SELECT * FROM src.bars")
    # carry over kv caches for flaky external data (earnings/fundamentals)
    bt.db.execute("INSERT OR IGNORE INTO kv SELECT * FROM src.kv WHERE "
                  "key LIKE 'earnings_%' OR key LIKE 'fundamentals_%'")
    bt.db.commit()
    bt.db.execute("DETACH DATABASE src")
    return bt


def run_backtest(cfg, years: int = 10, tag: str = "default",
                 copy_analogs_to: Store | None = None, log=print) -> dict:
    src_db = cfg.get("db_path", default="data/specforge.db")
    bt_path = Path(f"data/backtest_{tag}.db")
    bt = _seed_backtest_db(src_db, bt_path)

    # backtest config: the caller's merged config (so `--mode aggressive
    # backtest` tests that risk profile) with broker forced to paper and the
    # approval queue off (no human in a simulation)
    from .config import Config, _deep_merge
    bt_cfg = Config(_deep_merge(cfg.data, {
        "db_path": str(bt_path),
        "mode": "paper", "broker": "paper", "live_trading_enabled": False,
        "risk": {"approval_mode": "auto"},
        "paper": {"starting_cash": 10000.0},
    }))
    bt_cfg.validate()

    bench = bt_cfg.get("universe", "benchmark", default="SPY")
    all_dates = [r["d"] for r in bt.get_bars(bench, "9999-12-31", 20000)]
    start_cut = (date.today() - timedelta(days=365 * years)).isoformat()
    dates = [d for d in all_dates if d >= start_cut]
    if len(all_dates) - len(dates) < WARMUP_BARS:
        dates = all_dates[WARMUP_BARS:]           # ensure indicator warmup
    if not dates:
        return {"error": "not enough history — run `stonk data --full` first"}

    registry = build_registry(bt_cfg)             # build once, reuse across days
    from .broker.paper import PaperBroker
    broker = PaperBroker(bt_cfg, bt)

    log(f"backtest[{tag}]: {dates[0]} → {dates[-1]} ({len(dates)} sessions)")
    for i, d in enumerate(dates):
        run_cycle(bt_cfg, bt, broker=broker, as_of=d, refresh_data=False,
                  registry=registry, log=lambda *a: None)
        if i % 250 == 0:
            eq = bt.equity_curve("paper")
            log(f"  {d}: equity ${eq[-1]['equity']:.0f}" if eq else f"  {d}")

    # liquidate remaining positions at final close so every trade is a round-trip
    _liquidate(bt, bt_cfg, broker, dates[-1])

    report = _report(bt, bt_cfg, dates, years, tag)
    out_dir = Path("dev/reports"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"backtest_{tag}.json").write_text(json.dumps(report, indent=2, default=str))

    # push earnings/fundamentals kv caches back to the main DB so the next
    # backtest (bt DB is recreated each run) doesn't refetch from yfinance
    _sync_caches_back(bt, src_db)

    if copy_analogs_to is not None:
        n = _copy_analogs(bt_path, copy_analogs_to)
        report["analogs_copied_to_live_db"] = n
    return report


def _sync_caches_back(bt: Store, src_db: str) -> None:
    try:
        bt.db.execute("ATTACH DATABASE ? AS src", (str(src_db),))
        bt.db.execute("INSERT OR REPLACE INTO src.kv SELECT * FROM kv WHERE "
                      "key LIKE 'earnings_%' OR key LIKE 'fundamentals_%'")
        bt.db.commit()
        bt.db.execute("DETACH DATABASE src")
    except sqlite3.OperationalError as e:   # locked live DB is non-fatal
        print(f"backtest: cache sync-back skipped ({e})")


def _liquidate(bt: Store, bt_cfg, broker, last_day: str) -> None:
    from .data import MarketContext
    from .execution import Executor
    from .risk import Governor
    ctx = MarketContext(bt, bt_cfg, as_of=last_day)
    broker.set_quotes(ctx.prices())
    gov = Governor(bt_cfg, bt, now_iso=f"{last_day}T20:00:00")
    ex = Executor(bt_cfg, bt, broker, gov)
    for pos in bt.open_positions():
        px = ctx.close(pos["symbol"])
        if px:
            ex.execute_exit(pos, px, "backtest_end", broker.get_account(),
                            "bt_final", "n/a")


def _metrics(curve: list[dict]) -> dict:
    if len(curve) < 3:
        return {}
    eq = [c["equity"] for c in curve]
    rets = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq))]
    n = len(rets)
    mean_d = sum(rets) / n
    var = sum((r - mean_d) ** 2 for r in rets) / max(1, n - 1)
    sd = math.sqrt(var)
    downside = [r for r in rets if r < 0]
    dsd = math.sqrt(sum(r * r for r in downside) / max(1, len(downside)))
    peak, max_dd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        max_dd = max(max_dd, 1 - v / peak)
    years_span = n / 252
    cagr = (eq[-1] / eq[0]) ** (1 / max(years_span, 1e-9)) - 1
    return {
        "total_return": round(eq[-1] / eq[0] - 1, 4),
        "cagr": round(cagr, 4),
        "vol_annual": round(sd * math.sqrt(252), 4),
        "sharpe": round((mean_d / sd * math.sqrt(252)) if sd else 0.0, 2),
        "sortino": round((mean_d / dsd * math.sqrt(252)) if dsd else 0.0, 2),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(cagr / max_dd, 2) if max_dd else None,
        "sessions": n,
    }


def _report(bt: Store, bt_cfg, dates: list[str], years: int, tag: str) -> dict:
    curve = bt.equity_curve("paper")
    trades = bt.trades(source="paper")            # engine writes mode-named source
    for t in trades:                              # rebrand as backtest analogs
        bt.db.execute("UPDATE trades SET source='backtest' WHERE id=?", (t["id"],))
    bt.db.commit()

    # SPY buy-hold over the same window, same cost assumptions on entry/exit
    bench = bt_cfg.get("universe", "benchmark", default="SPY")
    spy = {r["d"]: r["close"] for r in bt.get_bars(bench, dates[-1], 20000)}
    spy_in = [spy[d] for d in dates if d in spy]
    friction = (bt_cfg.get("execution", "spread_cost_bps", default=3)
                + bt_cfg.get("execution", "slippage_bps", default=5)) * 2 / 10000
    spy_return = (spy_in[-1] / spy_in[0]) * (1 - friction) - 1 if len(spy_in) > 1 else None

    # OOS: first 70% vs last 30% of sessions
    split = int(len(curve) * 0.7)
    wins = [t for t in trades if t["ret"] > 0]
    by_regime: dict[str, list[float]] = {}
    for t in trades:
        by_regime.setdefault(t["regime"] or "unknown", []).append(t["ret"])
    by_node: dict[str, list[float]] = {}
    for t in trades:
        for nd in json.loads(t["nodes"] or "[]"):
            by_node.setdefault(nd, []).append(t["ret"])

    gross_win = sum(t["ret"] for t in wins)
    gross_loss = -sum(t["ret"] for t in trades if t["ret"] <= 0)
    return {
        "tag": tag, "window": [dates[0], dates[-1]], "years_requested": years,
        "overall": _metrics(curve),
        "in_sample_70pct": _metrics(curve[:split]),
        "out_of_sample_30pct": _metrics(curve[split:]),
        "benchmark_buy_hold_return": round(spy_return, 4) if spy_return is not None else None,
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3) if trades else None,
        "avg_win": round(sum(t["ret"] for t in wins) / len(wins), 4) if wins else None,
        "avg_loss": round(-gross_loss / max(1, len(trades) - len(wins)), 4) if trades else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "avg_holding_days": round(sum(t["horizon_days"] or 0 for t in trades)
                                  / len(trades), 1) if trades else None,
        "by_regime": {k: {"n": len(v), "avg_ret": round(sum(v) / len(v), 4)}
                      for k, v in sorted(by_regime.items())},
        "by_node": {k: {"n": len(v), "avg_ret": round(sum(v) / len(v), 4)}
                    for k, v in sorted(by_node.items())},
        "costs_included": True,
        "exit_reasons": _count(trades, "exit_reason"),
    }


def _count(trades: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for t in trades:
        k = (t[key] or "").split(" ")[0]
        out[k] = out.get(k, 0) + 1
    return out


def _copy_analogs(bt_path: Path, live: Store) -> int:
    """Copy backtest analog trades into the live DB (replacing prior set for
    the same source) so live forecasts get error bars immediately."""
    live.db.execute("DELETE FROM trades WHERE source='backtest'")
    live.db.execute("ATTACH DATABASE ? AS bt", (str(bt_path),))
    cur = live.db.execute(
        "INSERT INTO trades SELECT * FROM bt.trades WHERE source='backtest'")
    live.db.commit()
    n = cur.rowcount
    live.db.execute("DETACH DATABASE bt")
    live.audit("analogs_imported", {"n": n, "from": str(bt_path)})
    return n
