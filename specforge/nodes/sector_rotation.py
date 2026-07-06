"""Sector rotation node (AGENTS.md §10.6): single-stock signals work better when
sector flow agrees. Emits mild long signals on leading sector ETFs AND a sector
tailwind/headwind signal for member stocks (via a static sector map — good
enough for a 45-name universe; swap for a data source if universe grows).
"""
from __future__ import annotations

import math
from datetime import datetime

from ..data import MarketContext
from ..models import SignalEvent
from .base import SignalNode

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]

# ponytail: static membership map for the default universe; replace with a
# fundamentals data source when the universe becomes dynamic.
SECTOR_OF = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "AVGO": "XLK",
    "ORCL": "XLK", "CRM": "XLK", "QCOM": "XLK", "TXN": "XLK", "MU": "XLK",
    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "MCD": "XLY",
    "JPM": "XLF", "V": "XLF",
    "UNH": "XLV", "LLY": "XLV",
    "XOM": "XLE", "CVX": "XLE",
    "WMT": "XLP", "COST": "XLP", "KO": "XLP", "PEP": "XLP",
    "CAT": "XLI", "BA": "XLI", "GE": "XLI",
}


class Node(SignalNode):
    version = "1"
    role = "alpha"

    def compute(self, ctx: MarketContext) -> list[SignalEvent]:
        bench = ctx.cfg.get("universe", "benchmark", default="SPY")
        bench_c = ctx.closes(bench)
        if len(bench_c) < 64:
            return []
        bench_r = bench_c.iloc[-1] / bench_c.iloc[-64] - 1

        # sector relative strength vs benchmark, 1m + 3m blend
        rel: dict[str, float] = {}
        for etf in SECTOR_ETFS:
            c = ctx.closes(etf)
            if len(c) < 64:
                continue
            r63 = c.iloc[-1] / c.iloc[-64] - 1
            r21 = c.iloc[-1] / c.iloc[-22] - 1
            rel[etf] = 0.6 * (r63 - bench_r) + 0.4 * (r21 - (bench_c.iloc[-1] / bench_c.iloc[-22] - 1))

        if not rel:
            return []
        as_of = datetime.strptime(ctx.as_of, "%Y-%m-%d")
        ranked = sorted(rel.items(), key=lambda kv: kv[1], reverse=True)
        top = {etf for etf, _ in ranked[:3]}
        bottom = {etf for etf, _ in ranked[-3:]}

        events = []
        for sym in ctx.universe:
            sector = SECTOR_OF.get(sym) or (sym if sym in rel else None)
            if sector is None or sector not in rel:
                continue
            strength = math.tanh(rel[sector] / 0.06)
            if sector in top and strength > 0:
                direction, score = "long", strength
            elif sector in bottom and strength < 0:
                direction, score = "avoid", strength
            else:
                continue
            vol = (ctx.atr_pct(sym) or 0.02) * math.sqrt(self.horizon_days)
            events.append(SignalEvent(
                symbol=sym, direction=direction, score=round(score, 4),
                confidence=0.6, horizon_days=self.horizon_days,
                expected_return=round(score * vol * 0.4, 5),
                expected_volatility=round(vol, 5),
                downside_estimate=round(-2 * vol, 5),
                evidence=[f"sector {sector} rel={rel[sector]:+.1%} rank "
                          f"{'top3' if sector in top else 'bottom3'}"],
                data_as_of=as_of, node_id=self.id, node_version=self.version))
        return events
