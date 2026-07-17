"""Typed neural forecast contract.

The model predicts BOTH absolute and benchmark-excess return distributions.
Conflating the two is the semantic bug this contract exists to prevent: a stock
that returns -5% while SPY returns -10% has +5% *excess* but a -5% *absolute*
outcome — it is not a long. Trade eligibility must key off the absolute edge
after cost; excess only confirms cross-sectional strength.

Validation is strict and FAILS LOUD — it never silently clamps. Silent clamping
would conceal broken inference or calibration. Calibration is a separate,
explicit transformation applied to raw model output *before* a NeuralForecast is
constructed; by the time values reach this contract they must already be valid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Horizons the model is trained to emit. A forecast for any other horizon is a
# construction bug, not a value to tolerate.
SUPPORTED_HORIZONS = (5, 21)


@dataclass(frozen=True)
class NeuralForecast:
    symbol: str
    as_of: str                       # decision date, YYYY-MM-DD
    horizon_sessions: int

    absolute_q10: float
    absolute_q50: float
    absolute_q90: float

    excess_q10: float
    excess_q50: float
    excess_q90: float

    probability_absolute_edge_positive: float   # P(absolute return > round-trip cost)
    probability_excess_positive: float           # P(excess return > 0)

    model_id: str
    dataset_manifest_id: str
    feature_schema_hash: str

    def __post_init__(self) -> None:
        if self.horizon_sessions <= 0 or self.horizon_sessions not in SUPPORTED_HORIZONS:
            raise ValueError(
                f"horizon_sessions must be one of {SUPPORTED_HORIZONS}, "
                f"got {self.horizon_sessions}")

        quantiles = {
            "absolute_q10": self.absolute_q10, "absolute_q50": self.absolute_q50,
            "absolute_q90": self.absolute_q90, "excess_q10": self.excess_q10,
            "excess_q50": self.excess_q50, "excess_q90": self.excess_q90,
        }
        probabilities = {
            "probability_absolute_edge_positive": self.probability_absolute_edge_positive,
            "probability_excess_positive": self.probability_excess_positive,
        }
        for name, value in {**quantiles, **probabilities}.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")

        for kind, lo, mid, hi in (
            ("absolute", self.absolute_q10, self.absolute_q50, self.absolute_q90),
            ("excess", self.excess_q10, self.excess_q50, self.excess_q90),
        ):
            if not (lo <= mid <= hi):
                raise ValueError(
                    f"{kind} quantiles must be ordered q10<=q50<=q90, "
                    f"got {lo}, {mid}, {hi}")

        for name, p in probabilities.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"{name} out of [0,1]: {p}")

        for name in ("model_id", "dataset_manifest_id", "feature_schema_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} provenance is required, got {value!r}")

    def absolute_edge_after_cost(self, cost: float) -> float:
        """Median absolute return net of modeled round-trip cost. This — not
        excess — is what a long candidate must clear to be eligible."""
        return self.absolute_q50 - cost
