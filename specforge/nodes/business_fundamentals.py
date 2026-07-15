"""Production AI business-quality vote backed by a versioned SEC dossier."""
from __future__ import annotations

import math
from datetime import datetime

from ..evidence import latest_dossier
from ..models import SignalEvent
from .base import SignalNode
from .quality_value import ETFISH


class Node(SignalNode):
    version = "1"
    role = "alpha"

    def compute(self, ctx) -> list[SignalEvent]:
        events, missing = [], 0
        for symbol in ctx.universe:
            if symbol in ETFISH or symbol.startswith("^"):
                continue
            dossier = latest_dossier(ctx.store, symbol, ctx.as_of)
            memo = (dossier or {}).get("fundamental_memo") or {}
            if not memo or memo.get("stance") == "neutral":
                missing += 1
                continue
            confidence = float(memo.get("confidence", 0))
            quality = float((dossier or {}).get("quality", 0))
            if confidence <= 0:
                continue
            horizon = int(memo.get("horizon_days", self.horizon_days))
            direction = "long" if memo["stance"] == "attractive" else "avoid"
            vol = (ctx.atr_pct(symbol) or .02) * math.sqrt(horizon)
            citations = ", ".join(c["source_id"] for c in memo.get("citations", [])[:3])
            events.append(SignalEvent(
                symbol=symbol, direction=direction, score=confidence,
                confidence=max(.1, quality), horizon_days=horizon,
                expected_return=round((1 if direction == "long" else -1) *
                                      confidence * vol * .35, 5),
                expected_volatility=round(vol, 5), downside_estimate=round(-2 * vol, 5),
                evidence=[f"{memo.get('thesis','')[:180]} [{citations}]"],
                data_as_of=datetime.strptime(ctx.as_of, "%Y-%m-%d"),
                node_id=self.id, node_version=self.version))
        if missing:
            self.degraded_reason = f"no verified current dossier for {missing} symbol(s)"
        return events
