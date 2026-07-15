"""Production company catalyst/news vote backed by verified source citations."""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from ..evidence import latest_dossier
from ..models import SignalEvent
from .base import SignalNode
from .quality_value import ETFISH


class Node(SignalNode):
    version = "1"
    role = "alpha"

    def compute(self, ctx) -> list[SignalEvent]:
        events = []
        news_state = ctx.store.kv_get("news_intelligence", {}) or {}
        news_by_symbol = news_state.get("symbols", {})
        for symbol in ctx.universe:
            if symbol in ETFISH or symbol.startswith("^"):
                continue
            dossier = latest_dossier(ctx.store, symbol, ctx.as_of)
            if dossier:
                try:
                    if datetime.now().astimezone() - datetime.fromisoformat(
                            dossier["created_at"]) > timedelta(hours=6):
                        self.degraded_reason = "catalyst dossier is older than six hours"
                        dossier = None       # fresh news may still supply the catalyst vote
                except (KeyError, ValueError):
                    continue
            memo = (dossier or {}).get("catalyst_memo") or {}
            news = news_by_symbol.get(symbol) or {}
            memo_signed = 0.0
            if memo and memo.get("stance") != "neutral":
                memo_signed = (1 if memo.get("stance") == "attractive" else -1) * float(
                    memo.get("confidence", 0))
            news_signed = float(news.get("score", 0))
            signed = (.65 * memo_signed + .35 * news_signed if memo and news else
                      memo_signed if memo else news_signed)
            if not signed:
                continue
            confidence = min(1.0, abs(signed))
            if confidence <= 0:
                continue
            quality = float((dossier or {}).get("quality", 0))
            if not dossier:
                quality = min(1.0, float(news.get("weight", 0)) / 3.0)
            horizon = min(45, int(memo.get("horizon_days", self.horizon_days))) if memo else 10
            direction = "long" if signed > 0 else "avoid"
            vol = (ctx.atr_pct(symbol) or .02) * math.sqrt(horizon)
            citations = ", ".join(c["source_id"] for c in memo.get("citations", [])[:3])
            thesis = memo.get("thesis", "") if memo else (
                f"{news.get('articles', 0)} classified company articles; "
                f"catalysts {news.get('catalysts', {})}")
            events.append(SignalEvent(
                symbol=symbol, direction=direction, score=confidence,
                confidence=max(.1, quality), horizon_days=horizon,
                expected_return=round((1 if direction == "long" else -1) *
                                      confidence * vol * .30, 5),
                expected_volatility=round(vol, 5), downside_estimate=round(-2 * vol, 5),
                evidence=[f"{thesis[:180]}" + (f" [{citations}]" if citations else "")],
                data_as_of=datetime.strptime(ctx.as_of, "%Y-%m-%d"),
                node_id=self.id, node_version=self.version))
        return events
