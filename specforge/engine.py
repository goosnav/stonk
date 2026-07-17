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
        store.audit("cycle_skipped_overlap", {"as_of": as_of, "mode": cfg.mode,
                                              "scope": "in_process"})
        return {"skipped": "a cycle is already running — this one was not started"}
    lease_resource = f"trading_cycle:{cfg.mode}"
    lease_owner = None
    try:
        # Cross-process authority (Sprint E1): the SQLite lease — not the
        # thread lock — decides. Daemon, CLI, GUI, and a restarted process can
        # never run concurrent cycles; a crashed worker's lease expires.
        lease_owner = store.acquire_lease(
            lease_resource, float(cfg.get("risk", "cycle_lease_seconds", default=900)))
        if not lease_owner:
            store.audit("cycle_skipped_overlap", {"as_of": as_of, "mode": cfg.mode,
                                                  "scope": "cross_process"})
            return {"skipped": "another process holds the trading-cycle lease"}
        return _run_cycle(cfg, store, broker, as_of, refresh_data, registry,
                          log, live_quotes, lease=(lease_resource, lease_owner))
    finally:
        if lease_owner:
            store.release_lease(lease_resource, lease_owner)
        _CYCLE_LOCK.release()


def _run_cycle(cfg, store: Store, broker=None, as_of: str | None = None,
               refresh_data: bool = True, registry=None, log=print,
               live_quotes: dict | None = None,
               lease: tuple[str, str] | None = None) -> dict:
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
    if refresh_data:
        # Closed-market research precomputes the broad-to-active funnel. A
        # missing/incomplete snapshot preserves the proven configured universe.
        from .universe import symbols as tier_symbols
        active = tier_symbols(store, "active")
        if active:
            symbols = active
    # hypothesis watchlist merge (V4/D34): the active short-term hypothesis may
    # add a bounded set of symbols to this cycle's scan universe. The cycle
    # universe is CYCLE-LOCAL (passed into MarketContext below) — shared
    # config is never mutated during execution (Sprint E1).
    if cfg.get("hypothesis", "enabled", default=False):
        from . import hypothesis as hypo_mod
        extra = [s for s in hypo_mod.watchlist(
                     store, as_of=as_of,
                     cap=cfg.get("hypothesis", "max_watchlist", default=8))
                 if s not in symbols]
        if extra:
            symbols = symbols + extra
            store.audit("hypothesis_watchlist_merged", {"added": extra}, cycle_id)
    volatility_symbols = (cfg.get("universe", "volatility_symbols", default={}) or {}).values()
    aux = list(dict.fromkeys([cfg.get("universe", "vix_symbol", default="^VIX"),
                              *volatility_symbols,
                              *cfg.get("universe", "context_symbols", default=[])]))
    if refresh_data:
        # Daily histories are settled-data inputs, not a ten-minute quote
        # feed. During an open session reuse the last completed snapshot; the
        # post-close/premarket research plane refreshes it. This removes 175
        # sequential provider calls (and their politeness sleeps) from every
        # live cycle. Missing history still gets one bootstrap attempt.
        from .health import _market_clock
        latest_benchmark = store.latest_bar_date(
            cfg.get("universe", "benchmark", default="SPY"))
        if _market_clock()["open"] and latest_benchmark:
            st("data", f"using cached settled bars ({latest_benchmark}) for "
                       f"{len(symbols) + len(aux)} symbols")
        else:
            st("data", f"refreshing daily bars for {len(symbols) + len(aux)} symbols")
            data_mod.refresh(store, symbols + aux, log=log)
    ctx = MarketContext(store, cfg, as_of, offline=not refresh_data,
                        symbols=symbols)

    # 1.5 live prices (D35). Without these, limit prices come from the LAST
    # DAILY CLOSE — live orders rest unfilled all day (the GE order, D26).
    # Stooq/yfinance quotes are ~15min delayed; that still beats yesterday.
    # Backtests (refresh_data=False, no injection) stay lookahead-clean.
    live_px = dict(live_quotes or {})
    if refresh_data and not live_px:
        st("quotes", "fetching live quotes (broker → stooq → yfinance)")
        try:
            from .quotes import QuoteService
            broker = broker or make_broker(cfg, store)
            live_px = {s: q["price"] for s, q in
                       QuoteService(cfg, broker=broker).get(symbols).items()
                       if q.get("price")}
        except Exception as e:                 # noqa: BLE001 — quotes are garnish
            store.audit("live_quotes_failed", {"error": str(e)[:200]}, cycle_id)
    if live_px:
        store.audit("live_quotes", {"n": len(live_px)}, cycle_id)
    ctx.live_px = live_px      # D40: intraday-aware nodes (gap) read these;
    #                            empty in backtests → those nodes stay silent

    # 2. account + broker
    st("account", "reading account state from broker")
    broker = broker or make_broker(cfg, store)
    if hasattr(broker, "set_quotes"):          # paper broker has no live feed
        broker.set_quotes({**ctx.prices(), **live_px})
    account = broker.get_account()
    if refresh_data and (cfg.get("intelligence", "enabled", default=False) or
                         cfg.get("ai", "enabled", default=False)):
        # Event-driven evidence refresh is asynchronous and deduplicated. It
        # never stalls the market cycle or touches the broker.
        last_request = store.kv_get("evidence_refresh_requested_at")
        due = True
        if last_request:
            try:
                due = (datetime.now().astimezone() -
                       datetime.fromisoformat(last_request)).total_seconds() >= 6 * 3600
            except ValueError:
                pass
        if due:
            from .research import enqueue_job
            enqueue_job(store, "deep_research", {"reason": "six-hour evidence refresh"},
                        priority=1, requested_by="autonomous")
            store.kv_set("evidence_refresh_requested_at",
                         datetime.now().astimezone().isoformat(timespec="seconds"))

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
    from .graph import activation_state, default_topology
    model_state = activation_state(cfg, store)
    model_state = {**model_state, "production_evidence": True,
                   "production_evidence_version": "evidence.v2"}

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
    if reconciled or exits:
        account = broker.get_account()
    cycle = CycleState(governor.cycle_budget(account, reg.deployment_multiplier))
    store.audit("cycle_budget", {"budget": cycle.budget,
                                 "regime_mult": reg.deployment_multiplier}, cycle_id)
    # D41: an unproven learned model gates the BLEND (zero weight), never
    # trading itself — approvals passed the governor and get re-reviewed at
    # placement; the deterministic ensemble is the proven fallback.
    approvals = executor.process_approval_queue(
        account, ctx, cycle, cycle_id, reg.regime)
    if approvals:
        account = broker.get_account()

    # 6. signals (AI client injected for ai-flagged nodes; they degrade to
    #    silence when it's disabled/over budget)
    if registry is None:
        from .ai import AIClient
        registry = build_registry(cfg, ai_client=AIClient(cfg, store))
    events = []
    filters = []
    node_states = {node_id: "unavailable" for node_id, node_cfg in
                   (cfg.get("nodes", default={}) or {}).items()
                   if node_cfg.get("enabled")}
    node_states["macro_regime"] = "running"
    symbol_states = {}
    for node in registry.values():
        if node.role == "filter":
            filters.append(node)
            continue
        st("signals", f"node {node.id} computing")
        try:
            node_events = node.compute(ctx)
        except Exception as e:                 # noqa: BLE001 — node isolation
            node.degraded_reason = str(e)
            node_states[node.id] = "blocked"
            store.audit("node_degraded", {"node": node.id, "error": str(e)}, cycle_id)
            continue
        node_states[node.id] = ("running" if node_events else
                                "unavailable" if node.degraded_reason else
                                "verified_neutral")
        symbol_states[node.id] = dict(getattr(node, "symbol_states", {}) or {})
        for ev in node_events:
            store.record_signal(ev, cycle_id)
        events.extend(node_events)

    # Filter equations remain outside the deterministic alpha sum, but their
    # signed point-in-time outputs are real specialist activations for the
    # outer graph and historical replay.
    graph_events = list(events)
    graph_symbols = sorted(set(ctx.universe) | {e.symbol for e in events})
    for node in filters:
        filter_states = {}
        for symbol in graph_symbols:
            try:
                event = node.graph_signal(ctx, symbol) if hasattr(node, "graph_signal") else None
            except Exception as exc:  # one missing issuer must not erase every symbol
                store.audit("filter_graph_degraded", {"node": node.id, "symbol": symbol,
                                                       "error": str(exc)[:160]}, cycle_id)
                filter_states[symbol] = "blocked"
                continue
            if event:
                store.record_signal(event, cycle_id)
                graph_events.append(event)
                filter_states[symbol] = "running"
            else:
                filter_states[symbol] = "unavailable"
        symbol_states[node.id] = filter_states
        node_states[node.id] = ("running" if any(
            state == "running" for state in filter_states.values()) else "unavailable")

    # 7. ensemble → forecast → portfolio
    st("ensemble", f"scoring {len(events)} signals across nodes")
    ensemble_events = ([e for e in events if e.node_id != "neural"]
                       if model_state["effective_blend"] else events)
    candidates = ensemble_mod.score(ensemble_events, reg.regime, cfg, store, filters, ctx,
                                    node_states=node_states,
                                    symbol_states=symbol_states)
    # The analog-neural graph is a bounded learned overlay. Specialist
    # equations remain unchanged and the deterministic ensemble stays the
    # fallback; an unvalidated graph has a zero live blend.
    from .graph import blend_candidates
    blend_candidates(candidates, graph_events, reg.regime, cfg, store, cycle_id,
                     node_states=node_states, symbol_states=symbol_states,
                     universe=ctx.universe)
    required_graph_nodes = {n["id"] for n in default_topology()["nodes"]
                            if n["role"] in ("alpha", "gate")}
    failed_nodes = sorted(n for n, state in node_states.items()
                          if n in required_graph_nodes and state == "blocked")
    if failed_nodes and model_state["ready"]:
        model_state = {**model_state, "ready": False, "effective_blend": 0.0,
                       "block_reason": f"BLOCKED: NODE {failed_nodes[0]}",
                       "failed_nodes": failed_nodes}
    # Direct bounded neural blend (Stage C1): graph-independent, audited, zero
    # with deterministic fallback when the model is unavailable. Reuses the
    # forecasts the neural node already computed this cycle.
    from .ml.policy import apply_neural_blend
    neural_node = registry.get("neural")
    neural_forecasts = getattr(neural_node, "last_forecasts", None) or {}
    neural_meta = getattr(neural_node, "last_meta", None) or {}
    # Cycle binding (C2 audit): a stash stamped for a different as_of is stale
    # and must be inert — fail closed to the deterministic path.
    if getattr(neural_node, "last_forecast_as_of", None) != ctx.as_of:
        neural_forecasts, neural_meta = {}, {}
    direct = apply_neural_blend(candidates, neural_forecasts, cfg, store, cycle_id,
                                graph_blend=model_state["effective_blend"],
                                as_of=ctx.as_of, meta=neural_meta)
    model_state = {**model_state, "direct_neural_blend": direct["blend"],
                   "direct_neural_reason": direct["reason"]}
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    forecast_mod.attach_intervals(candidates, store, ctx.prices())
    for c in candidates:
        if c.expected_return <= 0:
            c.risk_flags.append("nonpositive_after_cost_expected_return")
        store.record_candidate(c, cycle_id)
    rejected_edge = [c.symbol for c in candidates if c.expected_return <= 0]
    if rejected_edge:
        store.audit("candidates_rejected_nonpositive_edge", {
            "symbols": rejected_edge,
            "reason": "post-forecast expected return must remain positive after costs",
        }, cycle_id)
    candidates = [c for c in candidates if c.expected_return > 0]

    st("sizing", f"{len(candidates)} candidates → position sizing")
    rebalance = {"weights": {}, "actual_weights": {}, "target_weights": {},
                 "sells": [], "buys": [], "deferred_sells": [], "turnover": 0,
                 "turnover_cap": account.equity * .30}
    rebalance = portfolio_mod.rebalance_plan(candidates, account, ctx, cfg)
    for sell in rebalance["sells"]:
        price = live_px.get(sell["symbol"]) or ctx.close(sell["symbol"])
        if not price or price <= 0:
            sell["result"] = "unavailable_price"
            store.audit("rebalance_sell_skipped", {
                "symbol": sell["symbol"], "reason": "price unavailable"}, cycle_id)
            continue
        reason = (f"evidence rebalance {sell['current_weight']:.1%}→"
                  f"{sell['target_weight']:.1%}; score {sell['held_score']:.3f} "
                  f"vs {sell['best_new_score']:.3f}")
        sell["result"] = executor.execute_exit(
            sell["position"], price, reason, account, cycle_id, reg.regime,
            qty=sell["qty"])
    if rebalance["sells"]:
        # Only a broker refresh after the sell attempts may expose confirmed
        # proceeds; a theoretical target is never cash.
        account = broker.get_account()
    targets = portfolio_mod.fit_to_capacity(
        rebalance["buys"], account, cfg, cycle.budget_left)
    store.audit("rebalance_plan", {
        "model": model_state, "weights": rebalance["weights"],
        "actual_weights": rebalance.get("actual_weights", {}),
        "target_weights": rebalance.get("target_weights", {}),
        "turnover": rebalance["turnover"], "turnover_cap": rebalance["turnover_cap"],
        "funding_order": [s["symbol"] for s in rebalance["sells"]],
        "deferred_sells": rebalance.get("deferred_sells", []),
        "sells": [{k: v for k, v in s.items() if k != "position"}
                  for s in rebalance["sells"]],
        "buys": [{"symbol": c.symbol, "notional": n,
                  "actual_weight": rebalance.get("actual_weights", {}).get(c.symbol, 0),
                  "target_weight": rebalance.get("target_weights", {}).get(c.symbol, 0)}
                 for c, n in rebalance["buys"]],
    }, cycle_id)

    # 7.5 convexity overlay: maybe swap the top equity target for a bounded-
    #     premium long call (no-op unless options_vol enabled AND account
    #     unlocked AND §22 conditions pass; never runs in backtests)
    from .nodes.options_vol import convexity_overlay
    targets = convexity_overlay(targets, ctx, account, cfg, governor, store, log=log)

    # 8. Fact-check the whole desired batch against cash, buying power,
    # deployment room, and the remaining cycle budget before sending order 1.
    requested = round(sum(n for _, n in targets), 2)
    targets = portfolio_mod.fit_to_capacity(targets, account, cfg, cycle.budget_left)
    # 8.1 bounded neural exploration probe (Stage C2): at most one dedicated
    # slot beyond the deterministic batch, sized once via construct() and
    # capped by the exploration budget fraction and remaining cash headroom.
    # The governor below remains the final authority on every limit.
    from .ml.policy import select_exploration_probe
    probe = select_exploration_probe(
        candidates, targets, neural_forecasts, neural_meta, cfg, store, cycle_id,
        account, ctx, as_of=ctx.as_of, allocated=sum(n for _, n in targets))
    if probe:
        targets = targets + [probe]
    store.audit("batch_allocation", {
        "requested": requested, "allocated": round(sum(n for _, n in targets), 2),
        "cash": account.cash, "buying_power": account.buying_power,
        "cycle_budget_left": cycle.budget_left,
        "orders": [{"symbol": c.symbol, "notional": n} for c, n in targets],
    }, cycle_id)
    # Fence (Sprint E1): a worker that lost the cross-process lease between
    # planning and placement must not commit orders — another process may
    # already be trading. Protective exits above are deliberately NOT fenced.
    if lease and targets and not store.holds_lease(*lease):
        store.audit("cycle_fenced_lost_lease", {
            "resource": lease[0], "dropped_orders": len(targets)}, cycle_id)
        st("fenced", "trading lease lost before order placement — no orders sent")
        targets = []
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
    if entry_results and not any(status == "filled" for status in entry_results.values()):
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
        "model_state": model_state, "rebalance_turnover": round(rebalance["turnover"], 2),
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
