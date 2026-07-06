"""Scan-cycle orchestrator — the pipeline in dev/ARCHITECTURE.md, one function.

run_cycle() is used verbatim by: the CLI (`specforge scan`), the scheduler
(app.py), and the backtester (with as_of set and refresh_data=False). One code
path for live/paper/backtest is the lookahead guarantee (D8).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from . import data as data_mod
from . import ensemble as ensemble_mod
from . import forecast as forecast_mod
from . import portfolio as portfolio_mod
from . import regime as regime_mod
from .broker.base import make_broker
from .data import MarketContext
from .execution import Executor
from .models import new_id
from .nodes.base import build_registry
from .risk import CycleState, Governor
from .store import Store


def run_cycle(cfg, store: Store, broker=None, as_of: str | None = None,
              refresh_data: bool = True, registry=None, log=print) -> dict:
    cycle_id = new_id()
    source = "live" if cfg.mode == "live" else "paper"
    store.audit("cycle_start", {"as_of": as_of, "mode": cfg.mode}, cycle_id)

    # 1. data
    symbols = list(cfg.get("universe", "symbols", default=[]))
    aux = [cfg.get("universe", "vix_symbol", default="^VIX")]
    if refresh_data:
        data_mod.refresh(store, symbols + aux, log=log)
    ctx = MarketContext(store, cfg, as_of)

    # 2. account + broker
    broker = broker or make_broker(cfg, store)
    if hasattr(broker, "set_quotes"):          # paper broker has no live feed
        broker.set_quotes(ctx.prices())
    account = broker.get_account()

    # 3. safety rails up front
    governor = Governor(cfg, store)
    switches = governor.check_kill_switches(account, source)
    reg = regime_mod.classify(ctx, cfg)
    store.audit("regime", {"regime": reg.regime, "mult": reg.deployment_multiplier,
                           "evidence": reg.evidence}, cycle_id)

    executor = Executor(cfg, store, broker, governor)

    # 4. exits first — freeing risk budget before spending it
    exits = _check_exits(ctx, store, executor, account, cycle_id, reg.regime)

    # 5. human-approved intents from the queue
    approvals = executor.process_approval_queue(account, ctx, cycle_id, reg.regime)

    # 6. signals
    registry = registry if registry is not None else build_registry(cfg)
    events = []
    filters = []
    for node in registry.values():
        if node.role == "filter":
            filters.append(node)
            continue
        try:
            node_events = node.compute(ctx)
        except Exception as e:                 # noqa: BLE001 — node isolation
            node.degraded_reason = str(e)
            store.audit("node_degraded", {"node": node.id, "error": str(e)}, cycle_id)
            continue
        for ev in node_events:
            store.record_signal(ev, cycle_id)
        events.extend(node_events)

    # 7. ensemble → forecast → portfolio
    candidates = ensemble_mod.score(events, reg.regime, cfg, store, filters, ctx)
    forecast_mod.attach_intervals(candidates, store, ctx.prices())
    for c in candidates:
        store.record_candidate(c, cycle_id)

    account = broker.get_account()             # refresh after exits
    targets = portfolio_mod.construct(candidates, account, ctx, cfg)

    # 8. risk-gated execution under the time-step budget
    cycle = CycleState(governor.cycle_budget(account, reg.deployment_multiplier))
    store.audit("cycle_budget", {"budget": cycle.budget,
                                 "regime_mult": reg.deployment_multiplier}, cycle_id)
    entry_results = {}
    for cand, notional in targets:
        price = ctx.close(cand.symbol)
        if not price:
            continue
        status = executor.execute_entry(
            cand, notional, price, account, cycle,
            ctx.data_age_days(cand.symbol), cycle_id, reg.regime)
        entry_results[cand.symbol] = status
        if status == "filled":
            account = broker.get_account()     # keep caps honest within cycle

    # 9. mark equity
    account = broker.get_account()
    store.record_equity(account.equity, account.cash, source, d=ctx.as_of)

    summary = {
        "cycle_id": cycle_id, "as_of": ctx.as_of, "regime": reg.regime,
        "kill_switches": sorted(switches), "signals": len(events),
        "candidates": len(candidates), "entries": entry_results,
        "exits": exits, "approvals_processed": len(approvals),
        "budget": round(cycle.budget, 2), "budget_used": round(cycle.budget_used, 2),
        "equity": round(account.equity, 2), "cash": round(account.cash, 2),
    }
    store.audit("cycle_end", summary, cycle_id)
    return summary


def _check_exits(ctx: MarketContext, store: Store, executor: Executor,
                 account, cycle_id: str, regime: str) -> dict:
    """Stop-loss and time-stop exits (AGENTS.md §28 MVP subset; score-decay and
    regime exits arrive with attribution in Phase 5)."""
    results = {}
    as_of_dt = datetime.strptime(ctx.as_of, "%Y-%m-%d")
    for pos in store.open_positions():
        price = ctx.close(pos["symbol"])
        if price is None:
            continue
        reason = None
        if pos["stop_price"] and price <= pos["stop_price"]:
            reason = f"stop_loss ({price:.2f} <= {pos['stop_price']:.2f})"
        else:
            opened = datetime.strptime(pos["opened_at"][:10], "%Y-%m-%d")
            if as_of_dt - opened >= timedelta(days=int(pos["horizon_days"] * 1.5)):
                reason = f"time_stop ({(as_of_dt - opened).days}d held)"
        if reason:
            results[pos["symbol"]] = executor.execute_exit(
                pos, price, reason, account, cycle_id, regime)
    return results
