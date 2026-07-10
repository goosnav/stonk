"""Bridge broker (D6 fallback): the engine still decides everything, but a
scheduled Claude Code session — which already has the Robinhood MCP connected —
acts as the transport for orders and account state.

Protocol (all through the shared SQLite DB + CLI):
1. Engine cycle: place_order() marks the order 'pending_relay' and returns None
   (resting). review_order() approves locally IF the account snapshot is fresh;
   the real RH review happens in the bridge session before placing.
2. Bridge session (see scripts/bridge_prompt.md) runs:
     stonk bridge-dump              → JSON: pending intents + snapshot request
   relays each intent through RH MCP tools (review → place → poll), then:
     stonk bridge-report --file results.json
   which records fills (kv 'bridge_fill_<order_id>'), account snapshot
   (kv 'bridge_account'), and review rejections.
3. Next engine cycle: Executor.reconcile() polls us; we serve the recorded
   fills; positions/trades get created by the normal engine path.

Staleness rule: if the account snapshot is older than max_snapshot_age_hours,
review_order refuses — trading blind against a dead bridge is how phantom
positions happen.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from ..models import AccountState, Fill, OrderIntent, OrderReview, Position

SNAPSHOT_KEY = "bridge_account"
MAX_SNAPSHOT_AGE_HOURS = 20


class BridgeBroker:
    name = "robinhood_bridge"

    def __init__(self, cfg, store):
        self.cfg = cfg
        self.store = store
        ok, why = cfg.live_trading_allowed()
        self._live_ok, self._live_why = ok, why

    def _snapshot(self) -> dict | None:
        return self.store.kv_get(SNAPSHOT_KEY)

    def _snapshot_fresh(self) -> bool:
        s = self._snapshot()
        if not s:
            return False
        age = datetime.now().astimezone() - datetime.fromisoformat(s["as_of"])
        return age < timedelta(hours=MAX_SNAPSHOT_AGE_HOURS)

    def get_account(self) -> AccountState:
        s = self._snapshot()
        if not s:
            # empty account forces the governor to reject everything — safe default
            return AccountState(equity=0.0, cash=0.0, buying_power=0.0,
                                positions=[], as_of="1970-01-01T00:00:00+00:00")
        return AccountState(
            equity=s["equity"], cash=s["cash"], buying_power=s.get("buying_power", s["cash"]),
            positions=[Position(symbol=p["symbol"], asset_type="equity",
                                qty=p["qty"], avg_cost=p["avg_cost"],
                                opened_at=p.get("opened_at", ""))
                       for p in s.get("positions", [])],
            as_of=s["as_of"])

    def get_quotes(self, symbols: list[str]) -> dict[str, float]:
        s = self._snapshot() or {}
        return {k: v for k, v in (s.get("quotes") or {}).items() if k in symbols}

    def review_order(self, intent: OrderIntent) -> OrderReview:
        if not self._live_ok:
            return OrderReview(ok=False, warnings=[f"live gate: {self._live_why}"])
        if not self._snapshot_fresh():
            return OrderReview(ok=False, warnings=[
                f"bridge account snapshot stale/missing (max {MAX_SNAPSHOT_AGE_HOURS}h) "
                f"— run the bridge session (scripts/bridge_prompt.md)"])
        return OrderReview(ok=True, warnings=[])   # real RH review runs bridge-side

    def place_order(self, intent: OrderIntent) -> Fill | None:
        self.store.update_order(intent.id, status="pending_relay")
        self.store.audit("bridge_intent_queued", {"intent": intent.id,
                                                  "symbol": intent.symbol,
                                                  "notional": intent.notional})
        return None                                 # bridge session will act

    def poll_order(self, broker_order_id: str, intent: OrderIntent) -> Fill | str | None:
        rec = self.store.kv_get(f"bridge_fill_{intent.id}")
        if not rec:
            return None
        if rec.get("state") == "filled":
            return Fill(order_id=intent.id, symbol=intent.symbol, side=intent.side,
                        qty=rec["qty"], price=rec["price"], filled_at=rec["filled_at"])
        if rec.get("state") in ("cancelled", "rejected", "failed", "review_blocked"):
            return "dead"
        return None

    def cancel_order(self, broker_order_id: str) -> bool:
        return False                                # bridge session handles cancels


# ---------- CLI-side helpers (used by `stonk bridge-dump/-report`) ----------

def bridge_dump(store, cfg) -> dict:
    """Everything the bridge session needs, as one JSON blob."""
    pending = [dict(r) for r in store.db.execute(
        "SELECT * FROM orders WHERE status IN ('pending_relay','relayed')")]
    return {
        "instructions": "See scripts/bridge_prompt.md. Review each intent via "
                        "review_equity_order; place with place_equity_order using "
                        "ref_id=idempotency_key; report results via "
                        "`stonk bridge-report --file <results.json>`.",
        "account_whitelist_env": "RH_ACCOUNT_WHITELIST",
        "pending_intents": pending,
        "want_account_snapshot": True,
        "want_quotes_for": list(cfg.get("universe", "symbols", default=[])),
    }


def bridge_report(store, payload: dict) -> dict:
    """Ingest bridge session results. Payload schema:
    {account: {equity, cash, buying_power, positions:[{symbol,qty,avg_cost}], quotes:{sym:px}},
     orders: [{intent_id, state, qty, price, filled_at, broker_order_id, note}]}"""
    n_orders = 0
    if payload.get("account"):
        snap = payload["account"]
        snap["as_of"] = datetime.now().astimezone().isoformat()
        store.kv_set(SNAPSHOT_KEY, snap)
    for o in payload.get("orders", []):
        store.kv_set(f"bridge_fill_{o['intent_id']}", {
            "state": o.get("state", "unknown"), "qty": o.get("qty"),
            "price": o.get("price"),
            "filled_at": o.get("filled_at") or datetime.now().astimezone().isoformat(),
            "note": o.get("note", "")})
        if o.get("broker_order_id"):
            store.update_order(o["intent_id"], broker_order_id=o["broker_order_id"],
                               status="relayed")
        n_orders += 1
    store.audit("bridge_report", {"orders": n_orders,
                                  "account_updated": bool(payload.get("account"))})
    return {"ok": True, "orders_ingested": n_orders}
