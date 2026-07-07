"""Execution: approved candidates → broker orders. Deliberately boring.

Path for every order: build limit intent → governor review → (maybe) approval
queue → idempotent record → broker review_order → place → record fill →
position bookkeeping. Every step audits. Duplicate idempotency keys and broker
review warnings stop the order cold.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from .models import Fill, OrderIntent, TradeCandidate, new_id
from .risk import CycleState, Governor
from .store import Store


def score_bucket(score: float) -> str:
    """Coarse buckets so analog-trade cells (bucket × regime) accumulate sample."""
    a = abs(score)
    if a < 0.15: return "s0"
    if a < 0.30: return "s1"
    if a < 0.50: return "s2"
    return "s3"


class Executor:
    def __init__(self, cfg, store: Store, broker, governor: Governor,
                 now_iso: str | None = None):
        self.cfg = cfg
        self.store = store
        self.broker = broker
        self.governor = governor
        # logical clock (matches governor's) — see dev/PROGRESS.md clock injection
        self.now_iso = now_iso or governor.now_iso

    def _limit_price(self, last_price: float, side: str) -> float:
        off = self.cfg.get("execution", "limit_offset_pct", default=0.001)
        return round(last_price * (1 + off if side == "buy" else 1 - off), 4)

    def execute_entry(self, cand: TradeCandidate, target_notional: float,
                      last_price: float, account, cycle: CycleState,
                      data_age_days: int | None, cycle_id: str,
                      regime: str) -> str:
        """Returns final status string (for the cycle summary)."""
        if cand.asset_type == "option":
            # overlay computed whole contracts + premium; notional = premium×100
            det = cand.option_details or {}
            limit = det.get("premium", last_price)
            qty = float(det.get("contracts", 0))
            intent = OrderIntent.make(cand, qty=qty, limit_price=limit,
                                      now_iso=self.now_iso)
            intent.notional = round(qty * limit * 100, 2)
        else:
            limit = self._limit_price(last_price, cand.side)
            qty = round(target_notional / limit, 6)      # fractional shares (D11)
            intent = OrderIntent.make(cand, qty=qty, limit_price=limit,
                                      now_iso=self.now_iso)

        decision = self.governor.review(intent, cand, account, cycle, data_age_days)
        self.store.audit("risk_decision", {"intent": intent.id, "symbol": cand.symbol,
                                           "verdict": decision.verdict,
                                           "reasons": decision.reasons}, cycle_id)
        if decision.verdict == "REJECTED":
            # governor "no" is healthy — status 'vetoed', distinct from broker
            # 'rejected', so routine vetoes never trip the rejected-order storm
            # kill switch (that switch exists for broker bounces)
            intent.status = "vetoed"
            self.store.record_order(intent)
            return "rejected"

        # governor may have shrunk the size
        if decision.approved_notional < intent.notional:
            intent.qty = round(decision.approved_notional / limit, 6)
            intent.notional = round(intent.qty * limit, 2)

        if decision.verdict == "REQUIRES_HUMAN_APPROVAL":
            intent.status = "pending_approval"
            if not self.store.record_order(intent):
                return "duplicate"
            expires = (datetime.fromisoformat(self.now_iso) + timedelta(
                hours=self.cfg.get("risk", "approval_timeout_hours", default=24))).isoformat()
            self.store.queue_approval(intent.id, expires)
            self.store.audit("approval_queued", {"intent": intent.id,
                                                 "notional": intent.notional}, cycle_id)
            return "pending_approval"

        if not self.store.record_order(intent):
            self.store.audit("duplicate_order_blocked",
                             {"key": intent.idempotency_key}, cycle_id)
            return "duplicate"
        return self._review_and_place(intent, cand, cycle, cycle_id, regime)

    def _review_and_place(self, intent: OrderIntent, cand: TradeCandidate | None,
                          cycle: CycleState | None, cycle_id: str, regime: str) -> str:
        review = self.broker.review_order(intent)
        self.store.audit("broker_review", {"intent": intent.id, "ok": review.ok,
                                           "warnings": review.warnings}, cycle_id)
        if not review.ok:
            # unknown/severe warning → never place (AGENTS.md §34.16)
            self.store.update_order(intent.id, status="rejected")
            return "broker_rejected"
        self.store.update_order(intent.id, status="reviewed")

        fill = self.broker.place_order(intent)
        if fill is None:
            self.store.update_order(intent.id, status="placed")   # resting limit
            self.store.audit("order_resting", {"intent": intent.id}, cycle_id)
            return "resting"

        self.store.record_fill(fill)
        self.store.update_order(intent.id, status="filled")
        self.store.audit("order_filled", {"intent": intent.id, "price": fill.price,
                                          "qty": fill.qty}, cycle_id)
        if intent.side == "buy" and cand is not None:
            if cycle:
                cycle.budget_used += intent.notional
                cycle.new_positions += 1
            stop_mult = self.cfg.get("sizing", "atr_stop_multiple", default=1.8)
            atr_frac = (cand.option_details or {}).get("atr_pct") if cand.option_details else None
            stop = round(fill.price * (1 - stop_mult * (atr_frac or 0.02)), 4)
            self.store.save_position(new_id(), {
                "symbol": intent.symbol, "asset_type": intent.asset_type,
                "qty": fill.qty, "avg_cost": fill.price, "opened_at": self.now_iso,
                "horizon_days": cand.horizon_days, "stop_price": stop,
                "candidate_id": cand.id, "nodes": cand.contributing_nodes,
                "option_symbol": intent.option_symbol, "status": "open"})
        return "filled"

    def execute_exit(self, position: dict, last_price: float, reason: str,
                     account, cycle_id: str, regime: str) -> str:
        """Close an open position (sell). Exits bypass entry caps by design."""
        limit = self._limit_price(last_price, "sell")
        cand = TradeCandidate(
            id=new_id(), symbol=position["symbol"], asset_type=position["asset_type"],
            side="sell", thesis=f"exit: {reason}", final_score=0.0,
            target_notional=position["qty"] * limit, expected_return=0, ci_low=0,
            ci_high=0, probability_positive=0, expected_apr=0, apr_ci_low=0,
            apr_ci_high=0, horizon_days=0, max_loss=0, contributing_nodes=[],
            option_symbol=position.get("option_symbol"))
        intent = OrderIntent.make(cand, qty=position["qty"], limit_price=limit,
                                  now_iso=self.now_iso)
        if not self.store.record_order(intent):
            return "duplicate"
        status = self._review_and_place(intent, None, None, cycle_id, regime)
        if status == "filled":
            entry_px, exit_px = position["avg_cost"], limit
            pnl = (exit_px - entry_px) * position["qty"]
            self.store.close_position(position["id"])
            # link the round-trip back to the entry candidate: its score/bucket
            # defines the analog cell this outcome feeds (forecast error bars)
            entry_cand = self._load_candidate(position.get("candidate_id", ""))
            entry_score = entry_cand.final_score if entry_cand else 0.0
            import json as _json
            self.store.record_trade({
                "score": entry_score, "score_bucket": score_bucket(entry_score),
                "symbol": position["symbol"], "asset_type": position["asset_type"],
                "entry_date": position["opened_at"][:10],
                "exit_date": self.now_iso[:10],
                "entry_price": entry_px, "exit_price": exit_px, "qty": position["qty"],
                "pnl": round(pnl, 4), "ret": round(exit_px / entry_px - 1, 6),
                "horizon_days": position["horizon_days"],
                "nodes": _json.loads(position["nodes"] or "[]"),
                "source": self.cfg.mode if self.cfg.mode == "live" else "paper",
                "exit_reason": reason,
                # analog cell keys on ENTRY regime (that's the state the signal
                # fired in); exit regime is only context
                "regime": entry_cand.regime if entry_cand else regime,
            })
            self.store.audit("position_closed", {"symbol": position["symbol"],
                                                 "reason": reason, "pnl": pnl}, cycle_id)
        return status

    def reconcile(self, cycle_id: str) -> dict:
        """Settle resting/relayed orders from prior cycles (live brokers fill
        asynchronously; paper never rests). Fills create positions/trades via
        the same bookkeeping as immediate fills."""
        if not hasattr(self.broker, "poll_order"):
            return {}
        results = {}
        rows = [dict(r) for r in self.store.db.execute(
            "SELECT * FROM orders WHERE status IN ('placed','pending_relay','relayed')")]
        for o in rows:
            intent = OrderIntent(
                id=o["id"], candidate_id=o["candidate_id"], symbol=o["symbol"],
                asset_type=o["asset_type"], side=o["side"], qty=o["qty"],
                limit_price=o["limit_price"], notional=o["notional"],
                idempotency_key=o["idempotency_key"], created_at=o["created_at"],
                option_symbol=o["option_symbol"])
            res = self.broker.poll_order(o["broker_order_id"], intent)
            if res is None:
                continue
            if res == "dead":
                self.store.update_order(o["id"], status="cancelled")
                self.store.audit("order_dead", {"intent": o["id"]}, cycle_id)
                results[o["symbol"]] = "dead"
                continue
            fill = res
            self.store.record_fill(fill)
            self.store.update_order(o["id"], status="filled")
            self.store.audit("order_filled_reconciled",
                             {"intent": o["id"], "price": fill.price}, cycle_id)
            if o["side"] == "buy":
                cand = self._load_candidate(o["candidate_id"])
                stop_mult = self.cfg.get("sizing", "atr_stop_multiple", default=1.8)
                self.store.save_position(new_id(), {
                    "symbol": o["symbol"], "asset_type": o["asset_type"],
                    "qty": fill.qty, "avg_cost": fill.price, "opened_at": self.now_iso,
                    "horizon_days": cand.horizon_days if cand else 20,
                    "stop_price": round(fill.price * (1 - stop_mult * 0.02), 4),
                    "candidate_id": o["candidate_id"],
                    "nodes": cand.contributing_nodes if cand else [],
                    "option_symbol": o["option_symbol"], "status": "open"})
            results[o["symbol"]] = "filled"
        return results

    def process_approval_queue(self, account, ctx, cycle_id: str, regime: str) -> list[str]:
        """Place human-approved intents; expire stale ones. Called each cycle."""
        results = []
        now = self.now_iso
        for row in self.store.pending_approvals():
            if row["expires_at"] < now:
                self.store.decide_approval(row["intent_id"], "expired")
                self.store.update_order(row["intent_id"], status="expired")
                self.store.audit("approval_expired", {"intent": row["intent_id"]}, cycle_id)
        approved = [dict(r) for r in self.store.db.execute(
            "SELECT o.* FROM orders o JOIN approvals a ON a.intent_id=o.id "
            "WHERE a.status='approved' AND o.status='pending_approval'")]
        for o in approved:
            intent = OrderIntent(
                id=o["id"], candidate_id=o["candidate_id"], symbol=o["symbol"],
                asset_type=o["asset_type"], side=o["side"], qty=o["qty"],
                limit_price=o["limit_price"], notional=o["notional"],
                idempotency_key=o["idempotency_key"], created_at=o["created_at"],
                option_symbol=o["option_symbol"])
            # reload the original candidate so the fill still creates position
            # metadata (stop/horizon/nodes) for exit logic + attribution
            cand = self._load_candidate(o["candidate_id"])
            # note: approved intents re-run broker review but not entry caps —
            # the human decision supersedes size checks made at queue time
            results.append(self._review_and_place(intent, cand, None, cycle_id, regime))
        return results

    def _load_candidate(self, candidate_id: str) -> TradeCandidate | None:
        import json as _json
        row = self.store.db.execute("SELECT payload FROM candidates WHERE id=?",
                                    (candidate_id,)).fetchone()
        return TradeCandidate(**_json.loads(row["payload"])) if row else None
