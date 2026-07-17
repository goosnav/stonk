"""Post-earnings announcement drift node (AGENTS.md §10.3): markets underreact
to true beat-and-raise events. Long when: recent report (≤10 sessions), solid
EPS surprise, and positive day-after price confirmation.

Data: yfinance earnings_dates (flaky, ~8 quarters of history). Results are
cached in kv for 3 days to spare the endpoint; on fetch failure the node
degrades to silence (never blocks the pipeline). Backtest caveat documented in
dev/PROGRESS.md: coverage only extends as far as yfinance history, so this
node's backtest sample is thin — treat its scorecard accordingly.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..data import MarketContext
from ..models import SignalEvent
from .base import SignalNode

CACHE_DAYS = 3
MIN_SURPRISE = 0.04       # 4% EPS beat


class Node(SignalNode):
    version = "1"
    role = "alpha"

    def _earnings(self, ctx: MarketContext, sym: str) -> list[dict]:
        """[{date, surprise_pct}] from cache or yfinance. [] on failure."""
        key = f"earnings_{sym}"
        cached = ctx.store.kv_get(key)
        if ctx.offline:                                   # backtest: cache only
            return cached["rows"] if cached else []
        if cached and cached.get("fetched_at", "") >= \
                (datetime.now() - timedelta(days=CACHE_DAYS)).isoformat():
            return cached["rows"]
        try:
            import yfinance as yf
            df = yf.Ticker(sym).earnings_dates
            rows = []
            if df is not None:
                for idx, row in df.iterrows():
                    surp = row.get("Surprise(%)")
                    if surp is None or surp != surp:      # NaN
                        continue
                    rows.append({"date": idx.strftime("%Y-%m-%d"),
                                 "surprise_pct": float(surp) / 100.0})
            ctx.store.kv_set(key, {"fetched_at": datetime.now().isoformat(), "rows": rows})
            return rows
        except Exception as e:                            # noqa: BLE001
            self.degraded_reason = f"earnings fetch failed: {e}"
            # cache the failure too — otherwise a rate-limited endpoint gets
            # hammered once per symbol per cycle (found via profiling)
            rows = cached["rows"] if cached else []
            ctx.store.kv_set(key, {"fetched_at": datetime.now().isoformat(),
                                   "rows": rows, "failed": str(e)})
            return rows

    def compute(self, ctx: MarketContext) -> list[SignalEvent]:
        as_of = datetime.strptime(ctx.as_of, "%Y-%m-%d")
        events = []
        from .quality_value import ETFISH
        for sym in ctx.universe:
            if sym.startswith("^") or sym in ETFISH:
                continue
            c = ctx.closes(sym)
            if len(c) < 30:
                self.symbol_states[sym] = "unavailable"
                continue
            rows = self._earnings(ctx, sym)
            self.symbol_states[sym] = ("unavailable" if self.degraded_reason and not rows
                                       else "verified_neutral")
            for ev in rows:
                ev_date = ev["date"]
                if ev_date > ctx.as_of:                   # future event: not tradable
                    continue
                days_ago = (as_of - datetime.strptime(ev_date, "%Y-%m-%d")).days
                if days_ago > 14 or ev["surprise_pct"] < MIN_SURPRISE:
                    continue
                # day-after confirmation: close after report vs close before
                dates = list(c.index)
                after = [d for d in dates if d > ev_date]
                before = [d for d in dates if d <= ev_date]
                if not after or not before:
                    continue
                reaction = c[after[0]] / c[before[-1]] - 1
                if reaction <= 0.005:
                    continue                              # no confirmation, no trade
                score = min(1.0, ev["surprise_pct"] / 0.15 * 0.6 + min(reaction, 0.08) / 0.08 * 0.4)
                vol = (ctx.atr_pct(sym) or 0.02) * (self.horizon_days ** 0.5)
                events.append(SignalEvent(
                    symbol=sym, direction="long", score=round(score, 4),
                    confidence=0.6, horizon_days=self.horizon_days,
                    expected_return=round(0.3 * reaction + 0.2 * ev["surprise_pct"], 5),
                    expected_volatility=round(vol, 5),
                    downside_estimate=round(-2 * vol, 5),
                    evidence=[f"EPS surprise {ev['surprise_pct']:+.1%} on {ev_date}, "
                              f"reaction {reaction:+.1%}, {days_ago}d ago"],
                    data_as_of=as_of, node_id=self.id, node_version=self.version))
                self.symbol_states[sym] = "running"
                break                                     # one signal per symbol
        return events
