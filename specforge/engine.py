"""Scan-cycle orchestrator — the pipeline in dev/ARCHITECTURE.md, one function.

run_cycle() is used verbatim by: the CLI (`stonk scan`), the scheduler
(app.py), and the backtester (with as_of set and refresh_data=False). One code
path for live/paper/backtest is the lookahead guarantee (D8).
"""
from __future__ import annotations

import threading
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


# D39: one cycle at a time. Scan-now racing the scheduler ran two cycles
# concurrently on 2026-07-10, doubled the orders, and tripped the
# rejected_orders kill switch. Overlap now skips instead of racing.
_CYCLE_LOCK = threading.Lock()


def _stamp(store: Store, phase: str, detail: str = "",
           cycle_id: str | None = None, trace: list | None = None,
           mode: str | None = None) -> None:
    """D39 live visibility: the GUI Engine tab polls kv['engine_state'] to show
    exactly where the state machine is. `trace` accumulates this cycle's
    phase timeline (written whole each stamp — it's tiny)."""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    if trace is not None:
        trace.append({"phase": phase, "detail": detail, "at": now})
    store.kv_set("engine_state", {"phase": phase, "detail": detail,
                                  "cycle_id": cycle_id, "at": now,
                                  "trace": trace or [], "mode": mode})


def run_cycle(cfg, store: Store, broker=None, as_of: str | None = None,
              refresh_data: bool = True, registry=None, log=print,
              live_quotes: dict | None = None) -> dict:
    if not _CYCLE_LOCK.acquire(blocking=False):
        store.audit("cycle_skipped_overlap", {"as_of": as_of, "mode": cfg.mode})
        return {"skipped": "a cycle is already running — this one was not started"}
    try:
        return _run_cycle(cfg, store, broker, as_of, refresh_data, registry,
                          log, live_quotes)
    finally:
        _CYCLE_LOCK.release()


def _run_cycle(cfg, store: Store, broker=None, as_of: str | None = None,
               refresh_data: bool = True, registry=None, log=print,
               live_quotes: dict | None = None) -> dict:
    cycle_id = new_id()
    trace: list = []
    source = "live" if cfg.mode == "live" else "paper"
    store.audit("cycle_start", {"as_of": as_of, "mode": cfg.mode}, cycle_id)

    # backtests spin hundreds of cycles — don't churn the live-visibility kv
    def st(phase: str, detail: str = ""):
        if refresh_data:
            _stamp(store, phase, detail, cycle_id, trace, source)

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
        st("data", f"refreshing daily bars for {len(symbols) + len(aux)} symbols")
        data_mod.refresh(store, symbols + aux, log=log)
    ctx = MarketContext(store, cfg, as_of, offline=not refresh_data)

    # 1.5 live prices (D35). Without these, limit prices come from the LAST
    # DAILY CLOSE — live orders rest unfilled all day (the GE order, D26).
    # Stooq/yfinance quotes are ~15min delayed; that still beats yesterday.
    # Backtests (refresh_data=False, no injection) stay lookahead-clean.
    live_px = dict(live_quotes or {})
    if refresh_data and not live_px:
        st("quotes", "fetching live quotes (broker → stooq → yfinance)")
        try:
            from .quotes import QuoteService
            live_px = {s: q["price"] for s, q in
                       QuoteService(cfg).get(symbols).items() if q.get("price")}
        except Exception as e:                 # noqa: BLE001 — quotes are garnish
            store.audit("live_quotes_failed", {"error": str(e)[:200]}, cycle_id)
    if live_px:
        store.audit("live_quotes", {"n": len(live_px)}, cycle_id)

    # 2. account + broker
    st("account", "reading account state from broker")
    broker = broker or make_broker(cfg, store)
    if hasattr(broker, "set_quotes"):          # paper broker has no live feed
        broker.set_quotes({**ctx.prices(), **live_px})
    account = broker.get_account()

    # 3. safety rails up front. Logical clock = as_of date + real time-of-day:
    # identical to wall clock for live scans, historical for backtests.
    now_iso = f"{ctx.as_of}T{datetime.now().astimezone().isoformat()[11:]}"
    governor = Governor(cfg, store, now_iso=now_iso)
    switches = governor.check_kill_switches(account, source)
    reg = regime_mod.classify(ctx, cfg)
    st("risk_gate", f"regime {reg.regime} ×{reg.deployment_multiplier}"
                    + (f" · KILL: {sorted(switches)}" if switches else " · no kill switches"))
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
    st("settle", "reconciling resting orders + checking stops/time exits")
    reconciled = executor.reconcile(cycle_id)
    mismatches = _position_mismatch(store, account, mode=source)
    if mismatches:
        # engine thinks it holds something the broker doesn't (or vice versa):
        # trading blind on wrong state is how phantom orders happen. Close the
        # orphan engine records, surface loudly, let the operator inspect.
        store.audit("position_mismatch", mismatches, cycle_id)
        for pid in mismatches.get("engine_only_ids", []):
            store.close_position(pid)
    exits = _check_exits(ctx, store, executor, account, cycle_id, reg.regime,
                         mode=source, live_px=live_px)

    # 5. human-approved intents from the queue
    account = broker.get_account()
    cycle = CycleState(governor.cycle_budget(account, reg.deployment_multiplier))
    store.audit("cycle_budget", {"budget": cycle.budget,
                                 "regime_mult": reg.deployment_multiplier}, cycle_id)
    approvals = executor.process_approval_queue(account, ctx, cycle, cycle_id, reg.regime)

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
        st("signals", f"node {node.id} computing")
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
    st("ensemble", f"scoring {len(events)} signals across nodes")
    candidates = ensemble_mod.score(events, reg.regime, cfg, store, filters, ctx)
    forecast_mod.attach_intervals(candidates, store, ctx.prices())
    for c in candidates:
        store.record_candidate(c, cycle_id)

    st("sizing", f"{len(candidates)} candidates → position sizing")
    account = broker.get_account()             # refresh after exits
    targets = portfolio_mod.construct(candidates, account, ctx, cfg)

    # 7.5 convexity overlay: maybe swap the top equity target for a bounded-
    #     premium long call (no-op unless options_vol enabled AND account
    #     unlocked AND §22 conditions pass; never runs in backtests)
    from .nodes.options_vol import convexity_overlay
    targets = convexity_overlay(targets, ctx, account, cfg, governor, store, log=log)

    # 8. Fact-check the whole desired batch against cash, buying power,
    # deployment room, and the remaining cycle budget before sending order 1.
    requested = round(sum(n for _, n in targets), 2)
    targets = portfolio_mod.fit_to_capacity(targets, account, cfg, cycle.budget_left)
    store.audit("batch_allocation", {
        "requested": requested, "allocated": round(sum(n for _, n in targets), 2),
        "cash": account.cash, "buying_power": account.buying_power,
        "cycle_budget_left": cycle.budget_left,
        "orders": [{"symbol": c.symbol, "notional": n} for c, n in targets],
    }, cycle_id)
    entry_results = {}
    broker_blocked = "broker_rejected" in approvals
    for i, (cand, notional) in enumerate(targets):
        if broker_blocked:
            entry_results[cand.symbol] = "skipped_broker_block"
            continue
        price = live_px.get(cand.symbol) or ctx.close(cand.symbol)
        if not price:
            continue
        st("execute", f"{cand.symbol}: governor review + order ${notional:.0f}")
        status = executor.execute_entry(
            cand, notional, price, account, cycle,
            ctx.data_age_days(cand.symbol), cycle_id, reg.regime)
        entry_results[cand.symbol] = status
        if status == "broker_rejected":
            # One shared account/broker problem should produce one diagnostic,
            # not a burst of identical rejected orders and a self-inflicted
            # kill switch. The next scheduled cycle may probe again.
            broker_blocked = True
            for remaining, _ in targets[i + 1:]:
                entry_results[remaining.symbol] = "skipped_broker_block"
            store.audit("entry_batch_halted", {
                "symbol": cand.symbol, "reason": "broker rejected first attempted entry",
                "skipped": [c.symbol for c, _ in targets[i + 1:]],
            }, cycle_id)
            break
        if status == "filled":
            account = broker.get_account()     # keep caps honest within cycle

    # 9. mark equity + net P&L (D37: the engine stamps a pnl mark every cycle
    # so the P&L chart populates even when no dashboard is open; realized from
    # closed trades + unrealized at current marks — deposit-independent)
    st("mark", "stamping equity + net P&L")
    account = broker.get_account()
    store.record_equity(account.equity, account.cash, source, d=ctx.as_of)
    if refresh_data:                        # live/paper scans only, not backtests
        realized = store.db.execute(
            "SELECT COALESCE(SUM(pnl),0) s FROM trades WHERE source=?",
            (source,)).fetchone()["s"]
        px = {**ctx.prices(), **live_px}
        unreal = sum(
            ((px.get(p.option_symbol or p.symbol) or p.avg_cost) - p.avg_cost)
            * p.qty * (100.0 if p.asset_type == "option" else 1.0)
            for p in account.positions if p.qty > 0)
        store.record_intraday_mark(account.equity, account.cash, source,
                                   pnl=round(realized + unreal, 2))

    summary = {
        "cycle_id": cycle_id, "mode": source, "as_of": ctx.as_of,
        "regime": reg.regime,
        "kill_switches": sorted(switches), "signals": len(events),
        "candidates": len(candidates), "entries": entry_results,
        "exits": exits, "reconciled": reconciled,
        "approvals_processed": len(approvals),
        "budget": round(cycle.budget, 2), "budget_used": round(cycle.budget_used, 2),
        "equity": round(account.equity, 2), "cash": round(account.cash, 2),
    }
    store.audit("cycle_end", summary, cycle_id)
    st("idle", f"cycle done: {len(events)} signals → {len(candidates)} candidates → "
               f"entries {entry_results or 'none'} · exits {exits or 'none'}")
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
                 account, cycle_id: str, regime: str, mode: str = 'paper',
                 live_px: dict | None = None) -> dict:
    """Stop-loss and time-stop exits (AGENTS.md §28 MVP subset; score-decay and
    regime exits arrive with attribution in Phase 5). live_px (D35) lets stops
    fire on intraday prices instead of waiting for the next daily bar."""
    live_px = live_px or {}
    results = {}
    as_of_dt = datetime.strptime(ctx.as_of, "%Y-%m-%d")
    for pos in store.open_positions(mode=mode):
        is_option = pos["asset_type"] == "option"
        price = (_option_mark(ctx, pos) if is_option
                 else live_px.get(pos["symbol"]) or ctx.close(pos["symbol"]))
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
