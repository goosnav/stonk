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
    # hypothesis watchlist merge (V4/D34): the active short-term hypothesis may
    # add a bounded set of symbols to this cycle's scan universe. In-memory
    # config mutation only — the file/override universe is untouched.
    if cfg.get("hypothesis", "enabled", default=False):
        from . import hypothesis as hypo_mod
        extra = [s for s in hypo_mod.watchlist(
                     store, as_of=as_of,
                     cap=cfg.get("hypothesis", "max_watchlist", default=8))
                 if s not in symbols]
        if extra:
            symbols = symbols + extra
            cfg.data["universe"]["symbols"] = symbols
            store.audit("hypothesis_watchlist_merged", {"added": extra}, cycle_id)
    aux = [cfg.get("universe", "vix_symbol", default="^VIX")]
    if refresh_data:
        data_mod.refresh(store, symbols + aux, log=log)
    ctx = MarketContext(store, cfg, as_of, offline=not refresh_data)

    # 2. account + broker
    broker = broker or make_broker(cfg, store)
    if hasattr(broker, "set_quotes"):          # paper broker has no live feed
        broker.set_quotes(ctx.prices())
    account = broker.get_account()

    # 3. safety rails up front. Logical clock = as_of date + real time-of-day:
    # identical to wall clock for live scans, historical for backtests.
    now_iso = f"{ctx.as_of}T{datetime.now().astimezone().isoformat()[11:]}"
    governor = Governor(cfg, store, now_iso=now_iso)
    switches = governor.check_kill_switches(account, source)
    reg = regime_mod.classify(ctx, cfg)
    store.audit("regime", {"regime": reg.regime, "mult": reg.deployment_multiplier,
                           "evidence": reg.evidence}, cycle_id)

    # steering expiry sweep (V4/D34): cheap + deterministic. An auto-adopt
    # tier request past its TTL activates here, so this very cycle trades on
    # it — trading never waits on a human.
    if cfg.get("hypothesis", "enabled", default=False):
        from . import steering as steering_mod
        steering_mod.sweep(cfg, store, now_iso=now_iso)

    executor = Executor(cfg, store, broker, governor)

    # 4. settle async fills from prior cycles, then exits (free budget before
    #    spending it)
    reconciled = executor.reconcile(cycle_id)
    mismatches = _position_mismatch(store, account, mode=source)
    if mismatches:
        # engine thinks it holds something the broker doesn't (or vice versa):
        # trading blind on wrong state is how phantom orders happen. Close the
        # orphan engine records, surface loudly, let the operator inspect.
        store.audit("position_mismatch", mismatches, cycle_id)
        for pid in mismatches.get("engine_only_ids", []):
            store.close_position(pid)
    exits = _check_exits(ctx, store, executor, account, cycle_id, reg.regime, mode=source)

    # 5. human-approved intents from the queue
    approvals = executor.process_approval_queue(account, ctx, cycle_id, reg.regime)

    # 6. signals (AI client injected for ai-flagged nodes; they degrade to
    #    silence when it's disabled/over budget)
    if registry is None:
        from .ai import AIClient
        registry = build_registry(cfg, ai_client=AIClient(cfg, store))
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

    # 7.5 convexity overlay: maybe swap the top equity target for a bounded-
    #     premium long call (no-op unless options_vol enabled AND account
    #     unlocked AND §22 conditions pass; never runs in backtests)
    from .nodes.options_vol import convexity_overlay
    targets = convexity_overlay(targets, ctx, account, cfg, governor, store, log=log)

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
        "exits": exits, "reconciled": reconciled,
        "approvals_processed": len(approvals),
        "budget": round(cycle.budget, 2), "budget_used": round(cycle.budget_used, 2),
        "equity": round(account.equity, 2), "cash": round(account.cash, 2),
    }
    store.audit("cycle_end", summary, cycle_id)
    return summary


def _position_mismatch(store: Store, account, mode: str = 'paper') -> dict:
    """Compare engine position metadata vs broker truth. Broker wins: engine
    rows without broker backing get closed (audited, no trade recorded);
    broker holdings the engine doesn't know are reported for the operator."""
    if account.equity <= 0 and not account.positions:
        return {}   # dead/missing feed (e.g. bridge snapshot absent) — don't
                    # mistake "no data" for "no positions" and wipe state
    broker_syms = {(p.option_symbol or p.symbol) for p in account.positions if p.qty > 0}
    engine = store.open_positions(mode=mode)
    engine_only = [p for p in engine
                   if (p["option_symbol"] or p["symbol"]) not in broker_syms]
    engine_syms = {(p["option_symbol"] or p["symbol"]) for p in engine}
    broker_only = sorted(broker_syms - engine_syms)
    if not engine_only and not broker_only:
        return {}
    return {"engine_only": [p["symbol"] for p in engine_only],
            "engine_only_ids": [p["id"] for p in engine_only],
            "broker_only_untracked": broker_only}


def _check_exits(ctx: MarketContext, store: Store, executor: Executor,
                 account, cycle_id: str, regime: str, mode: str = 'paper') -> dict:
    """Stop-loss and time-stop exits (AGENTS.md §28 MVP subset; score-decay and
    regime exits arrive with attribution in Phase 5)."""
    results = {}
    as_of_dt = datetime.strptime(ctx.as_of, "%Y-%m-%d")
    for pos in store.open_positions(mode=mode):
        is_option = pos["asset_type"] == "option"
        price = _option_mark(ctx, pos) if is_option else ctx.close(pos["symbol"])
        if price is None:
            continue
        reason = None
        # options exit on time only: premium isn't reliably markable, and max
        # loss is already bounded at the premium paid (nodes/options_vol.py)
        if not is_option and pos["stop_price"] and price <= pos["stop_price"]:
            reason = f"stop_loss ({price:.2f} <= {pos['stop_price']:.2f})"
        else:
            opened = datetime.strptime(pos["opened_at"][:10], "%Y-%m-%d")
            grace = 1.0 if is_option else 1.5
            if as_of_dt - opened >= timedelta(days=int(pos["horizon_days"] * grace)):
                reason = f"time_stop ({(as_of_dt - opened).days}d held)"
        if reason:
            results[pos["option_symbol"] or pos["symbol"]] = executor.execute_exit(
                pos, price, reason, account, cycle_id, regime)
    return results


def _option_mark(ctx: MarketContext, pos: dict) -> float | None:
    """Exit premium for a long option: live chain bid if reachable, else
    intrinsic value from the underlying close (conservative for a long)."""
    occ = pos["option_symbol"] or ""
    try:
        # OCC: SYMBOL + YYMMDD + C/P + strike*1000 (8 digits)
        body = occ[len(pos["symbol"]):]
        expiry = f"20{body[0:2]}-{body[2:4]}-{body[4:6]}"
        is_call = body[6] == "C"
        strike = int(body[7:15]) / 1000.0
    except (IndexError, ValueError):
        return None
    if not ctx.offline:
        try:
            import yfinance as yf
            chain = yf.Ticker(pos["symbol"]).option_chain(expiry)
            side = chain.calls if is_call else chain.puts
            row = side[side["strike"] == strike]
            if len(row) and float(row.iloc[0]["bid"] or 0) > 0:
                return float(row.iloc[0]["bid"])
        except Exception:                      # noqa: BLE001 — fall to intrinsic
            pass
    spot = ctx.close(pos["symbol"])
    if spot is None:
        return None
    return round(max(0.01, (spot - strike) if is_call else (strike - spot)), 2)
