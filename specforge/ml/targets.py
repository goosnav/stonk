"""Explicit neural target definitions — absolute AND benchmark-excess returns.

Kept in the ML schema layer so the target contract (what the model is asked to
predict) is auditable in one place. The central invariant this module exists to
protect: absolute and excess returns are DIFFERENT quantities. A stock that
returns -5% while the benchmark returns -10% has +5% excess but a -5% absolute
outcome — it is not a long. Trade eligibility keys off absolute-after-cost;
excess only confirms cross-sectional strength.

Both families share the identical decision date `t` and horizon `h`; the only
difference is the benchmark subtraction.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

HORIZONS = (5, 21)
RETURN_FAMILIES = ("absolute", "excess")
QUANTILES = (0.1, 0.5, 0.9)

TARGET_SCHEMA = {
    "horizons": list(HORIZONS),
    "families": list(RETURN_FAMILIES),
    "quantiles": list(QUANTILES),
    "absolute": "close[t+h]/close[t]-1",
    "excess": "absolute_h - benchmark_forward_return_h",
    "absolute_prob": "absolute_return > per_sample_round_trip_cost",
    "excess_prob": "excess_return > 0",
    "cost_model": "per-session (spread proxy + slippage + sqrt participation impact)",
}
# A cost estimate above this is a liquidity failure, not a tradable sample; the
# cap keeps one microcap print from manufacturing an untradable label.
MAX_SAMPLE_COST = 0.05
TARGET_SCHEMA_HASH = hashlib.sha256(
    repr(sorted(TARGET_SCHEMA.items())).encode()).hexdigest()[:16]


def round_trip_cost(cfg) -> float:
    """Round-trip friction as a return fraction from the repo's own cost model:
    (half-spread + slippage) per side × two sides. For the default
    spread_cost_bps=3, slippage_bps=5 this is (3+5)·2/1e4 = 0.0016 — the same
    constant montecarlo.py and _top_decile_alpha already use, so cost is never
    deducted on a different basis in different places."""
    spread = float(cfg.get("execution", "spread_cost_bps", default=3))
    slippage = float(cfg.get("execution", "slippage_bps", default=5))
    return (spread + slippage) * 2 / 10_000


def sample_costs(cfg, bars: pd.DataFrame) -> pd.Series:
    """Per-session round-trip cost fraction for one symbol.

    `round_trip_cost` is a single constant for the whole market, which prices a
    thin microcap exactly like SPY. This estimates each session's own cost from
    that session's own bar:

      spread_bps  ← configured spread · sqrt(reference / dollar volume), i.e.
                    spread widens as liquidity thins, floored at the configured
                    value for names at or above the reference
      impact_bps  ← coefficient · sqrt(assumed order notional / 21-session
                    median dollar volume) — the standard square-root law
      cost        ← (spread + slippage + impact) · 2 sides

    Floored at `round_trip_cost(cfg)` and capped at MAX_SAMPLE_COST so a bad
    print can neither make a trade look free nor manufacture an absurd label.

    ponytail: dollar volume is the proxy, not the high/low range — a range is
    dominated by drift, so half-range reads a mega-cap's spread at ~75bp when
    it is under 1bp. Real historical bid/ask would beat this; the live quote
    path already has it, history does not. Upgrade when a quote archive exists.
    """
    floor = round_trip_cost(cfg)
    slippage = float(cfg.get("execution", "slippage_bps", default=5))
    spread_floor = float(cfg.get("execution", "spread_cost_bps", default=3))
    coefficient = float(cfg.get("execution", "impact_coefficient", default=10.0))
    notional = float(cfg.get("execution", "assumed_order_notional", default=2_000.0))
    reference = float(cfg.get("execution", "reference_dollar_volume", default=1e8))
    close = bars["close"].astype(float).replace(0, np.nan)
    dollar_volume = (close * bars["volume"].astype(float)).rolling(
        21, min_periods=1).median()
    thinness = (reference / dollar_volume.replace(0, np.nan)).clip(lower=1.0)
    spread_bps = spread_floor * np.sqrt(thinness.fillna(1e4))
    participation = (notional / dollar_volume.replace(0, np.nan)).clip(lower=0, upper=1)
    impact_bps = coefficient * np.sqrt(participation.fillna(1.0)) * 1e2
    cost = (spread_bps + slippage + impact_bps) * 2 / 1e4
    return cost.fillna(floor).clip(lower=floor, upper=MAX_SAMPLE_COST)


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """h-session forward simple return; NaN for the last h rows (no lookahead)."""
    return close.shift(-horizon) / close - 1


def build_targets(close: pd.Series, benchmark_close: pd.Series,
                  horizons=HORIZONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Y_absolute, Y_excess), columns = horizons, indexed like `close`.

    Excess reproduces the prior inline definition exactly:
    absolute_h - (benchmark forward return over the same window).
    """
    bench = benchmark_close.reindex(close.index).ffill().astype(float)
    absolute = pd.DataFrame({h: forward_return(close, h) for h in horizons})
    bench_fwd = pd.DataFrame({h: forward_return(bench, h) for h in horizons})
    excess = absolute - bench_fwd
    return absolute, excess


def probability_labels(absolute, excess, cost: float):
    """(absolute_edge_positive, excess_positive) boolean labels.

    Absolute uses the modeled round-trip cost threshold — hence
    'absolute_edge_positive', NOT 'absolute_positive'. Excess uses zero.
    """
    absolute, excess = np.asarray(absolute), np.asarray(excess)
    return absolute > cost, excess > 0
