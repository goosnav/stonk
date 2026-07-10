"""D37 fundamentals node: validation, caching (no token re-spend), signal
emission, and degradation. Offline: _brief is stubbed, AI is a stub."""
from __future__ import annotations

from specforge.data import MarketContext
from specforge.nodes.base import build_registry


class StubAI:
    def __init__(self, resp):
        self.resp, self.calls = resp, 0

    def available(self):
        return True

    def complete_json(self, *a, **k):
        self.calls += 1
        return self.resp


GOOD = {"valuation": "undervalued", "direction": "long", "conviction": 0.7,
        "horizon_days": 40, "thesis": "cheap vs cash flow", "red_flags": []}


def _node(cfg, ai):
    cfg.data["nodes"]["fundamentals"] = {"enabled": True, "weight": 0.15,
                                         "status": "experimental", "ai": True,
                                         "horizon_days": 40}
    node = build_registry(cfg, ai_client=ai)["fundamentals"]
    node._brief = lambda ctx, sym: f"{sym} fundamentals: pe 10, margins 0.3"
    return node


def test_emits_signal_and_caches(cfg, store):
    ai = StubAI(GOOD)
    node = _node(cfg, ai)
    # online ctx (offline would silence the node) but stubbed brief/AI
    ctx = MarketContext(store, cfg, as_of=store.latest_bar_date("AAA"))
    ctx.offline = False
    events = node.compute(ctx)
    by = {e.symbol: e for e in events}
    assert "AAA" in by and by["AAA"].score == 0.7 and by["AAA"].direction == "long"
    assert "SPY" not in by                            # ETF skipped
    assert "undervalued" in by["AAA"].evidence[0]
    n_first = ai.calls
    assert node.compute(ctx) and ai.calls == n_first   # kv cache: zero new calls
    syn = store.kv_get("fundamentals_synopsis")
    assert syn and syn["items"][0]["valuation"] == "undervalued"


def test_avoid_and_neutral_and_garbage(cfg, store):
    ctx = MarketContext(store, cfg, as_of=store.latest_bar_date("AAA"))
    ctx.offline = False
    node = _node(cfg, StubAI({**GOOD, "direction": "avoid", "conviction": 0.5}))
    ev = node.compute(ctx)
    assert ev and ev[0].score == -0.5 and ev[0].direction == "avoid"

    for sym in ["AAA", "BBB", "CCC"]:                 # clear analysis cache
        store.kv_set(f"fund_view_{sym}", None)
    node = _node(cfg, StubAI({**GOOD, "direction": "neutral"}))
    assert node.compute(ctx) == []                    # neutral → no signal

    for sym in ["AAA", "BBB", "CCC"]:
        store.kv_set(f"fund_view_{sym}", None)
    node = _node(cfg, StubAI({"direction": "sideways", "conviction": 2}))
    assert node.compute(ctx) == []                    # garbage → discarded

    node = _node(cfg, StubAI(GOOD))
    assert node.compute(MarketContext(store, cfg, offline=True)) == []  # backtest


def test_validate_clamps(cfg):
    from specforge.nodes.fundamentals import Node
    v = Node._validate({"direction": "long", "conviction": 9,
                        "horizon_days": 500, "valuation": "undervalued",
                        "red_flags": ["a"] * 10})
    assert v["conviction"] == 1.0 and v["horizon_days"] == 90
    assert len(v["red_flags"]) == 4
    assert Node._validate(None) is None
    assert Node._validate({"conviction": 0.5}) is None
