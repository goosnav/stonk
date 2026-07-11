"""Bounded closed-market research queue.

There is no speculative queue table: each task derives whether it is due from
durable catalog/model/forecast watermarks, runs idempotently, and stamps one
operator-visible research_state record. One process lock prevents overlap.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime

_LOCK = threading.Lock()


def _stamp(store, phase: str, detail: str, **extra) -> dict:
    state = {"phase": phase, "detail": detail,
             "at": datetime.now().astimezone().isoformat(timespec="seconds"), **extra}
    store.kv_set("research_state", state)
    return state


def _missing_history(store, limit: int) -> list[str]:
    return [r["symbol"] for r in store.db.execute(
        "SELECT i.symbol,COUNT(b.d) n FROM instruments i LEFT JOIN bars b "
        "ON b.symbol=i.symbol WHERE i.active=1 GROUP BY i.symbol HAVING n<260 "
        "ORDER BY n DESC,i.symbol LIMIT ?", (limit,))]


def resolve_forecasts(store) -> int:
    rows = store.db.execute(
        "SELECT * FROM model_forecasts WHERE resolved_at IS NULL ORDER BY as_of LIMIT 5000"
    ).fetchall()
    resolved = 0
    for r in rows:
        future = store.db.execute(
            "SELECT d,close FROM bars WHERE symbol=? AND d>? ORDER BY d LIMIT ?",
            (r["symbol"], r["as_of"], r["horizon"])).fetchall()
        bench = store.db.execute(
            "SELECT d,close FROM bars WHERE symbol='SPY' AND d>? ORDER BY d LIMIT ?",
            (r["as_of"], r["horizon"])).fetchall()
        start = store.db.execute("SELECT close FROM bars WHERE symbol=? AND d<=? "
                                 "ORDER BY d DESC LIMIT 1", (r["symbol"], r["as_of"])).fetchone()
        bstart = store.db.execute("SELECT close FROM bars WHERE symbol='SPY' AND d<=? "
                                  "ORDER BY d DESC LIMIT 1", (r["as_of"],)).fetchone()
        if len(future) < r["horizon"] or len(bench) < r["horizon"] or not start or not bstart:
            continue
        realized = (future[-1]["close"] / start["close"] - 1) - \
            (bench[-1]["close"] / bstart["close"] - 1)
        store.db.execute("UPDATE model_forecasts SET resolved_at=?,realized_excess=? "
                         "WHERE model_id=? AND as_of=? AND symbol=? AND horizon=?",
                         (future[-1]["d"], realized, r["model_id"], r["as_of"],
                          r["symbol"], r["horizon"]))
        resolved += 1
    store.db.commit()
    if resolved:
        store.audit("shadow_forecasts_resolved", {"count": resolved})
    return resolved


def record_shadow_forecasts(cfg, store) -> int:
    from .data import MarketContext
    from .neural import predict_run
    from .universe import symbols
    syms = symbols(store, "research")
    if not syms:
        return 0
    # Research inference uses an in-memory universe only.
    old = cfg.data["universe"]["symbols"]
    cfg.data["universe"]["symbols"] = syms
    try:
        # Shadow the newest challenger.  A champion is only the fallback: if
        # we always preferred it, challengers could never accumulate the
        # forward evidence required to replace it.
        row = store.db.execute("SELECT id FROM model_runs WHERE kind='global_tcn' "
                               "AND status IN ('champion','challenger') "
                               "ORDER BY CASE status WHEN 'challenger' THEN 0 ELSE 1 END, "
                               "created_at DESC LIMIT 1").fetchone()
        if not row:
            return 0
        preds, meta = predict_run(cfg, store, MarketContext(store, cfg), row["id"])
    finally:
        cfg.data["universe"]["symbols"] = old
    model_id = meta.get("model_id")
    if not model_id:
        return 0
    as_of = store.latest_bar_date(cfg.get("universe", "benchmark", default="SPY"))
    n = 0
    with store.db:
        for sym, hs in preds.items():
            for horizon, p in hs.items():
                store.db.execute(
                    "INSERT OR IGNORE INTO model_forecasts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (model_id, as_of, sym, int(horizon), p["q10"], p["q50"], p["q90"],
                     p["probability_positive"], None, None, None)); n += 1
    if n: store.audit("shadow_forecasts_recorded", {"model": model_id, "count": n})
    return n


def graph_samples(store) -> tuple[list[dict], list[list[float]]]:
    """Resolve historical live/paper specialist activations into graph labels."""
    rows = store.db.execute(
        "SELECT substr(ts,1,10) d,symbol,node_id,direction,score,confidence "
        "FROM signals ORDER BY ts").fetchall()
    grouped: dict[tuple[str, str], dict] = {}
    for r in rows:
        sign = 1 if r["direction"] in ("long", "long_call") else -1
        grouped.setdefault((r["d"], r["symbol"]), {})[r["node_id"]] = \
            sign * r["score"] * r["confidence"]
    bases, targets = [], []
    for (d, sym), b in grouped.items():
        start = store.db.execute("SELECT close FROM bars WHERE symbol=? AND d<=? "
                                 "ORDER BY d DESC LIMIT 1", (sym, d)).fetchone()
        bs = store.db.execute("SELECT close FROM bars WHERE symbol='SPY' AND d<=? "
                              "ORDER BY d DESC LIMIT 1", (d,)).fetchone()
        future = store.db.execute("SELECT close FROM bars WHERE symbol=? AND d>? "
                                  "ORDER BY d LIMIT 21", (sym, d)).fetchall()
        bench = store.db.execute("SELECT close FROM bars WHERE symbol='SPY' AND d>? "
                                 "ORDER BY d LIMIT 21", (d,)).fetchall()
        if not start or not bs or len(future) < 21 or len(bench) < 21:
            continue
        y5 = (future[4]["close"] / start["close"] - 1) - (bench[4]["close"] / bs["close"] - 1)
        y21 = (future[20]["close"] / start["close"] - 1) - (bench[20]["close"] / bs["close"] - 1)
        bases.append(b); targets.append([y5, y21])
    return bases, targets


def train_graph_challenger(cfg, store) -> dict:
    from .graph import champion, fit_weights, mutate, save_version
    snapshot = store.latest_bar_date(cfg.get("universe", "benchmark", default="SPY"))
    key = f"graph_trials_{snapshot}"
    used = int(store.kv_get(key, 0) or 0)
    cap = int(cfg.get("analog_graph", "max_topology_trials_per_snapshot", default=24))
    if used >= cap:
        return {"status": "caught_up", "kind": "analog_graph", "trials": used}
    bases, targets = graph_samples(store)
    if len(bases) < 100:
        return {"status": "waiting", "kind": "analog_graph",
                "reason": f"need 100 resolved signal snapshots; have {len(bases)}"}
    parent = champion(store)
    topology = mutate(parent["topology"], seed=used)
    learned, metrics = fit_weights(
        topology, bases, targets,
        prune_pct=float(cfg.get("analog_graph", "prune_contribution_pct", default=.01)))
    store.kv_set(key, used + 1)
    vid = save_version(store, learned, "challenger", metrics, parent["id"], snapshot)
    return {"id": vid, "status": "challenger", "metrics": metrics, "trial": used + 1}


def current_scenarios(cfg, store) -> dict:
    import pandas as pd
    from .data import MarketContext
    from .montecarlo import block_bootstrap
    ctx = MarketContext(store, cfg)
    syms = [r["symbol"] for r in store.db.execute(
        "SELECT symbol,MAX(ts) t FROM candidates GROUP BY symbol ORDER BY t DESC LIMIT 25")]
    series = {s: ctx.closes(s, lookback=270).pct_change().dropna().tail(252)
              for s in syms}
    df = pd.DataFrame(series).dropna()
    if len(df) < 60 or not syms:
        return {"status": "waiting", "reason": "not enough aligned candidate returns"}
    curve = store.equity_curve("live" if cfg.mode == "live" else "paper")
    equity = float(curve[-1]["equity"]) if curve else 100.0
    out = block_bootstrap(equity, df.values, [.9 / len(syms)] * len(syms),
                          horizon_days=21, n_paths=10_000).__dict__
    store.kv_set("research_scenarios", {"at": datetime.now().astimezone().isoformat(),
                                        "symbols": syms, "shortlist": out})
    return out


def run_next(cfg, store, max_seconds: int | None = None) -> dict:
    if not _LOCK.acquire(blocking=False):
        return {"status": "skipped", "reason": "research task already running"}
    max_seconds = max_seconds or int(cfg.get("research", "max_task_seconds", default=600))
    try:
        from . import neural
        from .data import refresh
        from .universe import (ingest_next_filing_facts, refresh_membership,
                               status as universe_status, symbols as tier_symbols,
                               sync_catalog)
        today = date.today().isoformat()
        cat = store.kv_get("catalog_status") or {}
        if not cat.get("at", "").startswith(today):
            _stamp(store, "catalog", "synchronizing official listings")
            return {"task": "catalog", **sync_catalog(store)}
        newest = store.latest_bar_date(cfg.get("universe", "benchmark", default="SPY"))
        if (store.kv_get("universe_status") or {}).get("as_of") != newest:
            _stamp(store, "universe", "ranking research and active tiers")
            return {"task": "universe", **refresh_membership(cfg, store, newest)}
        n = resolve_forecasts(store)
        if n:
            _stamp(store, "resolve", f"resolved {n} matured forecasts")
            return {"task": "resolve", "count": n}
        # One global challenger per queue turn; trial caps prevent overtraining.
        _stamp(store, "tcn", "training bounded global TCN challenger")
        trained = neural.train_challenger(
            cfg, store, symbols=tier_symbols(store, "research") or None,
            max_seconds=min(max_seconds, 300))
        if trained.get("status") not in ("caught_up",):
            shadow = record_shadow_forecasts(cfg, store)
            return {"task": "tcn", "shadow_forecasts": shadow, **trained}
        has_global = store.db.execute("SELECT 1 FROM model_runs WHERE kind='global_tcn' "
                                      "AND status='champion' LIMIT 1").fetchone()
        if has_global:
            holdings = {p["symbol"] for p in store.open_positions(mode="live")}
            with store.db:
                sql = ("UPDATE model_runs SET status='archived' "
                       "WHERE kind='holding_tcn' "
                       "AND status IN ('champion','challenger')")
                if holdings:
                    sql += f" AND symbol NOT IN ({','.join('?' for _ in holdings)})"
                store.db.execute(sql, tuple(holdings))
            for symbol in sorted(holdings):
                h = neural.train_challenger(cfg, store, symbol=symbol,
                                            max_seconds=min(max_seconds, 180))
                if h.get("status") not in ("caught_up", "waiting"):
                    return {"task": "holding_tcn", **h}
        _stamp(store, "graph", "training bounded analog-graph challenger")
        graph = train_graph_challenger(cfg, store)
        if graph.get("status") not in ("caught_up", "waiting"):
            return {"task": "graph", **graph}
        shadow = record_shadow_forecasts(cfg, store)
        if shadow:
            return {"task": "shadow", "count": shadow}
        promotion = neural.maybe_promote(cfg, store)
        if promotion.get("action") not in ("none", "shadow"):
            return {"task": "model_gate", **promotion}
        from .graph import maybe_promote as maybe_promote_graph
        graph_gate = maybe_promote_graph(cfg, store)
        if graph_gate.get("action") not in ("none", "shadow"):
            return {"task": "graph_gate", **graph_gate}
        missing = _missing_history(store, int(cfg.get("research", "backfill_batch_size", default=50)))
        if missing:
            _stamp(store, "backfill", f"fetching history for {len(missing)} symbols",
                   progress={"symbols": missing})
            result = refresh(store, missing, full=True, log=lambda *a: None)
            # The next queue turn reranks the progressively wider catalog.
            store.kv_set("universe_status", {})
            return {"task": "backfill", "symbols": len(missing), "rows": sum(result.values())}
        facts = ingest_next_filing_facts(store)
        if facts.get("status") != "caught_up":
            _stamp(store, "filings", f"ingested SEC facts for {facts.get('symbol')}")
            return {"task": "filings", **facts}
        _stamp(store, "scenarios", "running portfolio bootstrap scenarios")
        current_scenarios(cfg, store)
        state = _stamp(store, "caught_up", "all useful research work is current",
                       universe=universe_status(store))
        return {"task": "caught_up", **state}
    except Exception as e:
        state = _stamp(store, "error", f"{type(e).__name__}: {str(e)[:240]}")
        store.audit("research_error", state)
        return state
    finally:
        state = store.kv_get("research_state") or {}
        if state.get("phase") not in ("error", "caught_up"):
            _stamp(store, "idle", "waiting for the next closed-market research tick",
                   last_task={"phase": state.get("phase"), "detail": state.get("detail"),
                              "completed_at": datetime.now().astimezone().isoformat(
                                  timespec="seconds")})
        _LOCK.release()


def status(store) -> dict:
    from .graph import champion
    from .universe import status as universe_status
    neural = store.kv_get("neural_status")
    if neural and "epoch" in neural:              # rejected D40 checkpoint schema
        neural = {"status": "legacy_rejected", "reason": "continuous MLP overfit",
                  "last_val_ic": neural.get("val_ic"), "last_epoch": neural.get("epoch"),
                  "at": neural.get("at")}
    return {"state": store.kv_get("research_state") or {"phase": "never_ran"},
            "universe": universe_status(store),
            "graph": {k: v for k, v in champion(store).items() if k != "topology"},
            "neural": neural,
            "scenarios": store.kv_get("research_scenarios"),
            "weekly": store.kv_get("research_report")}
