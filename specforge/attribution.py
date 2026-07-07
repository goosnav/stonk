"""Attribution + self-improvement (AGENTS.md §12): statistical, not mystical.

Runs post-close. For each node: rolling scorecard from closed trades it
contributed to, then a Bayesian-shrunk weight multiplier update bounded to
[min_multiplier, max_multiplier] (config ensemble.weight_learning). The system
may ONLY move multipliers and auto-disable clearly failing nodes; promotions
and anything else on the §12.2 list require a human.

Multiplier math (deliberately simple, upgrade path = regime-conditioned
multipliers once per-regime samples are meaningful):
  edge      = mean(trade returns) / std(trade returns)      (per-trade IR)
  shrunk    = edge × n / (n + shrinkage_n)                   (toward zero edge)
  multiplier= clamp(1 + shrunk × 2, min, max)
A node with no live sample keeps multiplier 1.0 — the backtest already set its
base weight; learning only reacts to measured live/paper outcomes.
"""
from __future__ import annotations

import json
import math
from datetime import datetime

from .store import Store

MIN_TRADES_TO_DISABLE = 30


def _sd(v: list[float]) -> float:
    m = sum(v) / len(v)
    return math.sqrt(sum((r - m) ** 2 for r in v) / max(1, len(v) - 1)) or 1e-9


def node_scorecard(store: Store, node_id: str,
                   sources: tuple = ("paper", "live")) -> dict:
    trades = [t for t in store.trades()
              if t["source"] in sources and node_id in json.loads(t["nodes"] or "[]")]
    if not trades:
        return {"node_id": node_id, "n": 0}
    rets = [t["ret"] for t in trades]
    n = len(rets)
    wins = [r for r in rets if r > 0]
    mean = sum(rets) / n
    sd = _sd(rets)
    gross_loss = -sum(r for r in rets if r <= 0)
    by_regime: dict[str, list[float]] = {}
    for t in trades:
        by_regime.setdefault(t["regime"] or "unknown", []).append(t["ret"])
    return {
        "node_id": node_id, "n": n,
        "expectancy": round(mean, 5),
        "hit_rate": round(len(wins) / n, 3),
        "avg_win": round(sum(wins) / len(wins), 4) if wins else None,
        "avg_loss": round(-gross_loss / max(1, n - len(wins)), 4),
        "profit_factor": round(sum(wins) / gross_loss, 2) if gross_loss else None,
        "per_trade_ir": round(mean / sd, 3),
        "by_regime": {k: {"n": len(v), "avg": round(sum(v) / len(v), 4),
                          "ir": round((sum(v) / len(v)) / _sd(v), 3)}
                      for k, v in by_regime.items()},
    }


def update_weights(cfg, store: Store, log=print) -> dict:
    """Post-close job. Returns {node_id: {multiplier, action}} for the audit."""
    wl = cfg.get("ensemble", "weight_learning", default={}) or {}
    if not wl.get("enabled", True):
        return {}
    lo = wl.get("min_multiplier", 0.3)
    hi = wl.get("max_multiplier", 2.0)
    min_n = wl.get("min_trades_before_update", 20)
    shrink_n = max(1, int(min_n * (wl.get("shrinkage", 0.7) / (1 - wl.get("shrinkage", 0.7)))))

    results = {}
    for node_id, ncfg in (cfg.get("nodes", default={}) or {}).items():
        if ncfg.get("role") in ("filter", "gate"):
            continue
        card = node_scorecard(store, node_id)
        store.db.execute("INSERT OR REPLACE INTO node_stats VALUES(?,?,?)",
                         (node_id, datetime.now().date().isoformat(),
                          json.dumps(card)))
        store.db.commit()
        n = card.get("n", 0)
        if n < min_n:
            results[node_id] = {"multiplier": store.get_weight_multiplier(node_id),
                                "action": f"hold (n={n} < {min_n})"}
            continue
        edge = card["per_trade_ir"]
        shrunk = edge * n / (n + shrink_n)
        mult = max(lo, min(hi, 1 + 2 * shrunk))
        store.set_weight_multiplier(node_id, round(mult, 3),
                                    note=f"n={n} ir={edge} shrunk={shrunk:.3f}")
        # regime-conditioned multipliers (ROADMAP Sprint D): same shrunk-IR
        # formula per regime cell, only where the cell has a meaningful sample.
        # Consumed INSTEAD of (not on top of) the global multiplier, so the
        # [lo, hi] governor bound holds trivially.
        regime_min_n = wl.get("regime_min_n", 30)
        regime_mults = {
            reg: round(max(lo, min(hi, 1 + 2 * (c["ir"] * c["n"] / (c["n"] + shrink_n)))), 3)
            for reg, c in card["by_regime"].items()
            if c["n"] >= regime_min_n and reg != "unknown"
        }
        all_rm = store.kv_get("regime_multipliers", {}) or {}
        if regime_mults:
            all_rm[node_id] = regime_mults
            store.kv_set("regime_multipliers", all_rm)
        elif node_id in all_rm:            # sample fell below threshold (e.g. window change)
            del all_rm[node_id]
            store.kv_set("regime_multipliers", all_rm)
        action = "updated"
        # auto-disable is allowed (§12.1) when the live evidence is damning
        if n >= MIN_TRADES_TO_DISABLE and card["expectancy"] < 0 and edge < -0.1:
            store.kv_set(f"node_auto_disabled_{node_id}",
                         {"at": datetime.now().isoformat(), "card": card})
            ov = store.kv_get("config_overrides", {}) or {}
            ov.setdefault("nodes", {}).setdefault(node_id, {})["enabled"] = False
            store.kv_set("config_overrides", ov)
            action = "AUTO-DISABLED (negative expectancy on meaningful sample)"
        results[node_id] = {"multiplier": mult, "action": action}
        log(f"attribution: {node_id} n={n} expectancy={card['expectancy']} "
            f"→ multiplier {mult:.2f} ({action})")
    store.audit("weight_update", results)
    return results


def propose_promotions(cfg, store: Store) -> list[dict]:
    """The system PROPOSES status changes; humans decide (AGENTS.md §12.6).
    Surfaced in the GUI/status; never applied automatically."""
    proposals = []
    for node_id, ncfg in (cfg.get("nodes", default={}) or {}).items():
        card = node_scorecard(store, node_id)
        status = ncfg.get("status", "experimental")
        if status == "experimental" and card.get("n", 0) >= 30 and \
                (card.get("expectancy") or 0) > 0:
            proposals.append({"node_id": node_id, "from": status, "to": "probation",
                              "basis": card})
        elif status == "probation" and card.get("n", 0) >= 100 and \
                (card.get("profit_factor") or 0) > 1.2:
            proposals.append({"node_id": node_id, "from": status, "to": "production",
                              "basis": card})
    return proposals
