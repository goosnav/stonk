"""Execution: approved candidates → broker orders. Deliberately boring.

Path for every order: build limit intent → governor review → (maybe) approval
queue → idempotent record → broker review_order → place → record fill →
position bookkeeping. Every step audits. Duplicate idempotency keys and broker
review warnings stop the order cold.
"""
from __future__ import annotations

import math
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
        # positions/trades in the shared DB are tagged by mode (paper|live)
        self.mode = "live" if cfg.mode == "live" else "paper"

    def _limit_price(self, last_price: float, side: str) -> float:
        off = self.cfg.get("execution", "limit_offset_pct", default=0.001)
        return round(last_price * (1 + off if side == "buy" else 1 - off), 4)

    @staticmethod
    def _resize_intent(intent: OrderIntent, approved_notional: float) -> bool:
        mult = 100.0 if intent.asset_type == "option" else 1.0
        qty = approved_notional / (intent.limit_price * mult)
        intent.qty = float(math.floor(qty)) if intent.asset_type == "option" else round(qty, 6)
        intent.notional = round(intent.qty * intent.limit_price * mult, 2)
        return intent.qty > 0

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
            self.store.record_order(intent, self.mode)
            return "rejected"

        # governor may have shrunk the size
        if decision.approved_notional < intent.notional:
            if not self._resize_intent(intent, decision.approved_notional):
                intent.status = "vetoed"
                self.store.record_order(intent, self.mode)
                return "rejected"

        if decision.verdict == "REQUIRES_HUMAN_APPROVAL":
            intent.status = "pending_approval"
            if not self.store.record_order(intent, self.mode):
                return "duplicate"
            expires = (datetime.fromisoformat(self.now_iso) + timedelta(
                hours=self.cfg.get("risk", "approval_timeout_hours", default=24))).isoformat()
            self.store.queue_approval(intent.id, expires)
            self.store.audit("approval_queued", {"intent": intent.id,
                                                 "notional": intent.notional}, cycle_id)
            return "pending_approval"

        if not self.store.record_order(intent, self.mode):
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

        try:
            fill = self.broker.place_order(intent)
        except Exception as e:                 # noqa: BLE001 — refusal or transport
            # D39: a refused/failed placement is a broker rejection, never a
            # crashed cycle. The adapter already audited the broker's words.
            self.store.update_order(intent.id, status="rejected")
            self.store.audit("order_place_failed",
                             {"intent": intent.id, "symbol": intent.symbol,
                              "error": str(e)[:300]}, cycle_id)
            return "broker_rejected"
        if fill is None:
            self.store.update_order(intent.id, status="placed")   # resting limit
            self.store.audit("order_resting", {"intent": intent.id}, cycle_id)
            if intent.side == "buy" and cycle:
                # D39: a resting BUY has spent the cycle budget the moment it's
                # placed. Without this, 12 resting orders totaling ~3× cash all
                # passed review in one cycle (2026-07-10, cycle 976cab145c00).
                cycle.budget_used += intent.notional
                cycle.new_positions += 1
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
                "option_symbol": intent.option_symbol, "status": "open",
                "mode": self.mode})
        return "filled"

    def execute_exit(self, position: dict, last_price: float, reason: str,
                     account, cycle_id: str, regime: str) -> str:
        """Close an open position (sell). Exits bypass entry caps by design."""
        limit = self._limit_price(last_price, "sell")
        cand = TradeCandidate(
            id=new_id(), symbol=position["symbol"], asset_type=position["asset_type"],
            side="sell", thesis=f"exit: {reason}", final_score=0.0,
            target_notional=position["qty"] * limit *
            (100 if position["asset_type"] == "option" else 1),
            expected_return=0, ci_low=0,
            ci_high=0, probability_positive=0, expected_apr=0, apr_ci_low=0,
            apr_ci_high=0, horizon_days=0, max_loss=0, contributing_nodes=[],
            option_symbol=position.get("option_symbol"))
        # Persist the exit thesis so an asynchronous fill can reconstruct the
        # reason and attribution on a later cycle.
        self.store.record_candidate(cand, cycle_id)
        intent = OrderIntent.make(cand, qty=position["qty"], limit_price=limit,
                                  now_iso=self.now_iso)
        if not self.store.record_order(intent, self.mode):
            return "duplicate"
        status = self._review_and_place(intent, None, None, cycle_id, regime)
        if status == "filled":
            row = self.store.db.execute(
                "SELECT price FROM fills WHERE order_id=? ORDER BY filled_at DESC LIMIT 1",
                (intent.id,)).fetchone()
            self._record_close(position, row["price"] if row else limit,
                               reason, cycle_id, regime)
        return status

    def reconcile(self, cycle_id: str) -> dict:
        """Settle resting/relayed orders from prior cycles (live brokers fill
        asynchronously; paper never rests). Fills create positions/trades via
        the same bookkeeping as immediate fills."""
        if not hasattr(self.broker, "poll_order"):
            return {}
        results = {}
        rows = [dict(r) for r in self.store.db.execute(
            "SELECT * FROM orders WHERE status IN ('placed','pending_relay','relayed') "
            "AND mode=?", (self.mode,))]
        for o in rows:
            intent = OrderIntent(
                id=o["id"], candidate_id=o["candidate_id"], symbol=o["symbol"],
                asset_type=o["asset_type"], side=o["side"], qty=o["qty"],
                limit_price=o["limit_price"], notional=o["notional"],
                idempotency_key=o["idempotency_key"], created_at=o["created_at"],
                option_symbol=o["option_symbol"])
            try:
                res = self.broker.poll_order(o["broker_order_id"], intent)
            except Exception as e:             # noqa: BLE001 — one bad poll
                # must not kill the cycle (D39: a hung/errored order query
                # took the whole 10:56 cycle down with it)
                self.store.audit("reconcile_poll_failed",
                                 {"intent": o["id"], "error": str(e)[:200]}, cycle_id)
                continue
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
                    "option_symbol": o["option_symbol"], "status": "open",
                    "mode": self.mode})
            else:
                key = o["option_symbol"] or o["symbol"]
                position = next((p for p in self.store.open_positions(mode=self.mode)
                                 if (p["option_symbol"] or p["symbol"]) == key), None)
                if position:
                    exit_cand = self._load_candidate(o["candidate_id"])
                    reason = (exit_cand.thesis.removeprefix("exit: ")
                              if exit_cand else "resting_exit_fill")
                    self._record_close(position, fill.price, reason, cycle_id,
                                       exit_cand.regime if exit_cand else "unknown")
            results[o["symbol"]] = "filled"
        return results

    def process_approval_queue(self, account, ctx, cycle: CycleState,
                               cycle_id: str, regime: str) -> list[str]:
        """Revalidate then place human-approved intents against current funds."""
        results = []
        now = self.now_iso
        for row in self.store.pending_approvals():
            if row["expires_at"] < now:
                self.store.decide_approval(row["intent_id"], "expired")
                self.store.update_order(row["intent_id"], status="expired")
                self.store.audit("approval_expired", {"intent": row["intent_id"]}, cycle_id)
        approved = [dict(r) for r in self.store.db.execute(
            "SELECT o.* FROM orders o JOIN approvals a ON a.intent_id=o.id "
            "WHERE a.status='approved' AND o.status='pending_approval' AND o.mode=?",
            (self.mode,))]
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
            if cand is None:
                self.store.update_order(intent.id, status="vetoed")
                results.append("rejected")
                continue
            account = self.broker.get_account()
            decision = self.governor.review(
                intent, cand, account, cycle, ctx.data_age_days(intent.symbol),
                skip_duplicate=True)
            self.store.audit("approved_order_revalidated", {
                "intent": intent.id, "symbol": intent.symbol,
                "verdict": decision.verdict, "reasons": decision.reasons,
                "original_notional": intent.notional,
                "approved_notional": decision.approved_notional,
            }, cycle_id)
            if decision.verdict == "REJECTED":
                self.store.update_order(intent.id, status="vetoed")
                results.append("rejected")
                continue
            if decision.approved_notional < intent.notional and not self._resize_intent(
                    intent, decision.approved_notional):
                self.store.update_order(intent.id, status="vetoed")
                results.append("rejected")
                continue
            status = self._review_and_place(intent, cand, cycle, cycle_id, regime)
            results.append(status)
            if status == "broker_rejected":
                break
        return results

    def _record_close(self, position: dict, exit_price: float, reason: str,
                      cycle_id: str, regime: str) -> None:
        entry_px = position["avg_cost"]
        mult = 100.0 if position["asset_type"] == "option" else 1.0
        pnl = (exit_price - entry_px) * position["qty"] * mult
        self.store.close_position(position["id"])
        entry_cand = self._load_candidate(position.get("candidate_id", ""))
        entry_score = entry_cand.final_score if entry_cand else 0.0
        import json as _json
        nodes = position["nodes"] or "[]"
        self.store.record_trade({
            "score": entry_score, "score_bucket": score_bucket(entry_score),
            "symbol": position["symbol"], "asset_type": position["asset_type"],
            "entry_date": position["opened_at"][:10], "exit_date": self.now_iso[:10],
            "entry_price": entry_px, "exit_price": exit_price, "qty": position["qty"],
            "pnl": round(pnl, 4), "ret": round(exit_price / entry_px - 1, 6),
            "horizon_days": position["horizon_days"],
            "nodes": _json.loads(nodes) if isinstance(nodes, str) else nodes,
            "source": self.cfg.mode if self.cfg.mode == "live" else "paper",
            "exit_reason": reason,
            "regime": entry_cand.regime if entry_cand else regime,
        })
        self.store.audit("position_closed", {"symbol": position["symbol"],
                                             "reason": reason, "pnl": pnl}, cycle_id)

    def _load_candidate(self, candidate_id: str) -> TradeCandidate | None:
        import json as _json
        row = self.store.db.execute("SELECT payload FROM candidates WHERE id=?",
                                    (candidate_id,)).fetchone()
        return TradeCandidate(**_json.loads(row["payload"])) if row else None
