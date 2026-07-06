"""Broker adapter protocol. The engine core only ever talks to this interface;
broker-specific details (MCP tool names, REST endpoints) live in adapters only.
"""
from __future__ import annotations

from typing import Protocol

from ..models import AccountState, Fill, OrderIntent, OrderReview


class BrokerAdapter(Protocol):
    name: str

    def get_account(self) -> AccountState: ...

    def get_quotes(self, symbols: list[str]) -> dict[str, float]: ...

    def review_order(self, intent: OrderIntent) -> OrderReview:
        """MUST be called before place_order. Adapters surface every broker
        warning; execution rejects on unknown/severe ones."""
        ...

    def place_order(self, intent: OrderIntent) -> Fill | None:
        """Returns Fill if (simulated-)immediately filled, None if resting."""
        ...

    def cancel_order(self, broker_order_id: str) -> bool: ...


def make_broker(cfg, store, ctx_prices=None) -> "BrokerAdapter":
    """Factory. Import adapters lazily so e.g. `mcp` is only needed for live."""
    kind = cfg.get("broker", default="paper")
    if kind == "paper":
        from .paper import PaperBroker
        return PaperBroker(cfg, store)
    if kind == "robinhood_mcp":
        from .robinhood_mcp import RobinhoodMCPBroker
        return RobinhoodMCPBroker(cfg, store)
    if kind == "robinhood_bridge":
        from .bridge import BridgeBroker
        return BridgeBroker(cfg, store)
    raise ValueError(f"unknown broker: {kind}")
