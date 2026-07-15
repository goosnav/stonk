"""Signal node base + registry.

A node turns MarketContext into SignalEvent forecasts. It NEVER sizes, orders,
or touches the broker (dev/ARCHITECTURE.md invariant #1). Node enablement,
base weight, role and status come from config `nodes:`; learned weight
multipliers come from the store (attribution layer, Phase 5).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..data import MarketContext
from ..models import SignalEvent


class SignalNode(ABC):
    id: str = ""                 # must match key in config nodes:
    version: str = "1"
    role: str = "alpha"          # alpha | filter | gate
    requires_ai: bool = False

    def __init__(self, node_cfg: dict):
        self.cfg = node_cfg
        self.horizon_days = int(node_cfg.get("horizon_days", 20))
        self.base_weight = float(node_cfg.get("weight", 0.0))
        self.status = node_cfg.get("status", "experimental")
        self.degraded_reason: str | None = None   # set when data source failed
        # Per-symbol truth used by scoring/graph APIs. A missing event can mean
        # verified neutral or unavailable data; callers must not guess.
        self.symbol_states: dict[str, str] = {}

    @abstractmethod
    def compute(self, ctx: MarketContext) -> list[SignalEvent]:
        """Return forecasts for symbols in ctx.universe. Must only read data
        through ctx (as_of-sliced). Raise nothing: catch data errors, set
        self.degraded_reason, and return []."""

    # filter-role nodes implement this instead of / in addition to compute()
    def passes(self, ctx: MarketContext, symbol: str) -> bool:  # noqa: ARG002
        return True


def build_registry(cfg, ai_client=None) -> dict[str, SignalNode]:
    """Instantiate enabled nodes from config. Import lazily per node so an
    optional dependency or a broken node file can't take down the engine."""
    from importlib import import_module

    registry: dict[str, SignalNode] = {}
    for node_id, node_cfg in (cfg.get("nodes", default={}) or {}).items():
        if not node_cfg.get("enabled"):
            continue
        try:
            mod = import_module(f".{node_id}", package=__package__)
            node = mod.Node(node_cfg)
            node.id = node_id
            if node.requires_ai:
                node.ai = ai_client   # None → node must degrade gracefully
            registry[node_id] = node
        except ModuleNotFoundError:
            # config lists a node that isn't implemented yet — skip loudly
            print(f"nodes: {node_id} enabled in config but not implemented; skipping")
        except Exception as e:                     # noqa: BLE001
            print(f"nodes: {node_id} failed to load: {e}; skipping")
    return registry
