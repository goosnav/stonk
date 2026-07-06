"""Paper broker: simulated account persisted in the store's kv table.

Fill model: limit orders fill immediately at limit price plus the configured
spread + slippage cost model (execution.spread_cost_bps / slippage_bps). That is
deliberately pessimistic-simple — same cost model the backtest uses, so paper
and backtest results are comparable.
"""
from __future__ import annotations

from datetime import datetime

from ..models import AccountState, Fill, OrderIntent, OrderReview, Position
from ..store import Store

KV_KEY = "paper_account"


class PaperBroker:
    name = "paper"

    def __init__(self, cfg, store: Store):
        self.cfg = cfg
        self.store = store
        if store.kv_get(KV_KEY) is None:
            store.kv_set(KV_KEY, {"cash": cfg.get("paper", "starting_cash", default=1000.0),
                                  "positions": {}})   # symbol -> {qty, avg_cost}
        self._quotes: dict[str, float] = {}

    # engine injects latest known prices each cycle (paper has no live feed)
    def set_quotes(self, prices: dict[str, float]) -> None:
        self._quotes.update(prices)

    def _acct(self) -> dict:
        return self.store.kv_get(KV_KEY)

    def _save(self, acct: dict) -> None:
        self.store.kv_set(KV_KEY, acct)

    def get_account(self) -> AccountState:
        acct = self._acct()
        positions = [Position(symbol=s, asset_type="equity", qty=p["qty"],
                              avg_cost=p["avg_cost"], opened_at=p.get("opened_at", ""))
                     for s, p in acct["positions"].items() if p["qty"] > 0]
        equity = acct["cash"] + sum(
            p.qty * self._quotes.get(p.symbol, p.avg_cost) for p in positions)
        return AccountState(equity=equity, cash=acct["cash"],
                            buying_power=acct["cash"], positions=positions,
                            as_of=datetime.now().astimezone().isoformat())

    def get_quotes(self, symbols: list[str]) -> dict[str, float]:
        return {s: self._quotes[s] for s in symbols if s in self._quotes}

    def review_order(self, intent: OrderIntent) -> OrderReview:
        acct = self._acct()
        warnings = []
        if intent.side == "buy" and intent.notional > acct["cash"]:
            warnings.append("insufficient_cash")
        if intent.side == "sell":
            held = acct["positions"].get(intent.symbol, {}).get("qty", 0)
            if intent.qty > held + 1e-9:
                warnings.append("insufficient_shares")
        return OrderReview(ok=not warnings, warnings=warnings)

    def place_order(self, intent: OrderIntent) -> Fill | None:
        # cost model: adverse fill by half-spread + slippage
        bps = (self.cfg.get("execution", "spread_cost_bps", default=3)
               + self.cfg.get("execution", "slippage_bps", default=5)) / 10000.0
        px = intent.limit_price * (1 + bps if intent.side == "buy" else 1 - bps)
        acct = self._acct()
        pos = acct["positions"].setdefault(intent.symbol, {"qty": 0.0, "avg_cost": 0.0})
        if intent.side == "buy":
            cost = intent.qty * px
            if cost > acct["cash"] + 1e-9:
                return None
            new_qty = pos["qty"] + intent.qty
            pos["avg_cost"] = (pos["qty"] * pos["avg_cost"] + cost) / new_qty
            pos["qty"] = new_qty
            pos.setdefault("opened_at", intent.created_at)
            acct["cash"] -= cost
        else:
            sell_qty = min(intent.qty, pos["qty"])
            if sell_qty <= 0:
                return None
            pos["qty"] -= sell_qty
            acct["cash"] += sell_qty * px
            if pos["qty"] <= 1e-9:
                acct["positions"].pop(intent.symbol, None)
        self._save(acct)
        return Fill(order_id=intent.id, symbol=intent.symbol, side=intent.side,
                    qty=intent.qty, price=round(px, 4),
                    filled_at=datetime.now().astimezone().isoformat())

    def cancel_order(self, broker_order_id: str) -> bool:
        return True   # paper orders never rest
