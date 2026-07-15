"""Core typed models. Nodes emit SignalEvent; ensemble emits TradeCandidate;
risk emits RiskDecision; execution emits OrderIntent/Fill. Nodes never order."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Literal, Optional

Direction = Literal["long", "short_bias", "avoid", "hedge", "long_call", "long_put"]
AssetType = Literal["equity", "option"]
Side = Literal["buy", "sell"]
RiskVerdict = Literal["APPROVED", "APPROVED_WITH_SIZE_REDUCTION", "REJECTED", "REQUIRES_HUMAN_APPROVAL"]
Regime = Literal["risk_on", "neutral", "risk_off", "stress"]


def new_id() -> str:
    return uuid.uuid4().hex[:12]


_POSITIVE_DIRECTIONS = {"long", "long_call"}


def direction_sign(direction: Direction | str) -> float:
    """Canonical sign contract: direction owns sign; score owns magnitude."""
    allowed = _POSITIVE_DIRECTIONS | {
        "short_bias", "avoid", "hedge", "long_put",
    }
    if direction not in allowed:
        raise ValueError(f"unsupported signal direction: {direction}")
    return 1.0 if direction in _POSITIVE_DIRECTIONS else -1.0


def signed_alpha(event_or_direction, score: float | None = None,
                 confidence: float | None = None) -> float:
    """Signed activation for live objects and legacy persisted signal rows.

    `abs(score)` repairs rows written under the old mixed signed/unsigned
    convention without allowing a negative avoid score to become positive.
    """
    if isinstance(event_or_direction, SignalEvent):
        direction = event_or_direction.direction
        magnitude = event_or_direction.score
        certainty = event_or_direction.confidence
    else:
        direction = event_or_direction
        magnitude = 0.0 if score is None else score
        certainty = 1.0 if confidence is None else confidence
    return direction_sign(direction) * abs(float(magnitude)) * float(certainty)


@dataclass
class SignalEvent:
    symbol: str
    direction: Direction
    score: float                 # magnitude 0..1; direction owns sign
    confidence: float            # 0..1
    horizon_days: int
    expected_return: float       # simple return over horizon, pre-cost
    expected_volatility: float   # horizon return stddev
    downside_estimate: float     # modeled bad-case horizon return (negative)
    evidence: list[str]
    data_as_of: datetime
    node_id: str
    node_version: str = "1"

    def __post_init__(self) -> None:
        direction_sign(self.direction)
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("SignalEvent.score must be a magnitude in [0,1]")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("SignalEvent.confidence must be in [0,1]")
        if self.horizon_days < 1:
            raise ValueError("SignalEvent.horizon_days must be positive")


@dataclass
class TradeCandidate:
    id: str
    symbol: str
    asset_type: AssetType
    side: Side
    thesis: str
    final_score: float
    target_notional: float
    expected_return: float       # horizon, after modeled costs
    ci_low: float                # 80% interval
    ci_high: float
    probability_positive: float
    expected_apr: float          # secondary, annualized
    apr_ci_low: float
    apr_ci_high: float
    horizon_days: int
    max_loss: float              # $ worst case (notional for equity, premium for option)
    contributing_nodes: list[str]
    risk_flags: list[str] = field(default_factory=list)
    confidence_label: str = "low"   # low | medium | high (basis size)
    regime: str = "neutral"
    evidence_version: str = "evidence.v1"
    evidence_coverage: float = 0.0
    evidence_details: list[dict] = field(default_factory=list)
    production_score: float = 0.0
    strategy_contribution: float = 0.0
    learned_contribution: float = 0.0
    strategy_mandate_id: Optional[str] = None
    # option-only fields
    option_symbol: Optional[str] = None
    option_details: Optional[dict] = None


@dataclass
class RiskDecision:
    verdict: RiskVerdict
    reasons: list[str]
    approved_notional: float = 0.0


@dataclass
class OrderIntent:
    id: str
    candidate_id: str
    symbol: str
    asset_type: AssetType
    side: Side
    qty: float                   # shares (fractional ok) or contracts
    limit_price: float
    notional: float
    idempotency_key: str
    created_at: str              # iso
    status: str = "pending"      # pending|pending_approval|reviewed|placed|filled|rejected|expired|cancelled
    broker_order_id: Optional[str] = None
    option_symbol: Optional[str] = None

    @staticmethod
    def make(candidate: TradeCandidate, qty: float, limit_price: float,
             now_iso: str | None = None) -> "OrderIntent":
        # now_iso lets the backtester run this exact code path at historical
        # timestamps (dev/PROGRESS.md "clock injection"); live passes None.
        now = now_iso or datetime.now().astimezone().isoformat()
        key = f"{candidate.symbol}-{candidate.side}-{now[:10]}-{candidate.id}"
        return OrderIntent(
            id=new_id(), candidate_id=candidate.id, symbol=candidate.symbol,
            asset_type=candidate.asset_type, side=candidate.side, qty=qty,
            limit_price=limit_price,
            notional=round(qty * limit_price *
                           (100 if candidate.asset_type == "option" else 1), 2),
            idempotency_key=key, created_at=now, option_symbol=candidate.option_symbol)


@dataclass
class OrderReview:
    ok: bool
    warnings: list[str]
    raw: dict = field(default_factory=dict)


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    filled_at: str
    fees: float = 0.0


@dataclass
class Position:
    symbol: str
    asset_type: AssetType
    qty: float
    avg_cost: float
    opened_at: str
    horizon_days: int = 20
    stop_price: float = 0.0
    candidate_id: str = ""
    contributing_nodes: list[str] = field(default_factory=list)
    option_symbol: Optional[str] = None

    @property
    def cost_basis(self) -> float:
        mult = 100.0 if self.asset_type == "option" else 1.0
        return self.qty * self.avg_cost * mult


@dataclass
class AccountState:
    equity: float
    cash: float
    buying_power: float
    positions: list[Position]
    as_of: str

    def market_value(self, prices: dict[str, float]) -> float:
        mv = self.cash
        for p in self.positions:
            mult = 100.0 if p.asset_type == "option" else 1.0
            mv += p.qty * prices.get(p.option_symbol or p.symbol, p.avg_cost) * mult
        return mv


def to_json_dict(obj) -> dict:
    return asdict(obj)
