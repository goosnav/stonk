"""Shared fixtures: in-memory store, synthetic bar data, offline config.

Synthetic universe: 3 trending symbols + SPY + fake VIX — enough to drive
momentum/regime/ensemble without any network access. Tests must NEVER hit the
network or the real data/ DB.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from specforge.config import load_config
from specforge.store import Store

SYMS = ["AAA", "BBB", "CCC"]


def synth_bars(n_days: int = 320, start_price: float = 100.0,
               daily_drift: float = 0.001, wiggle: float = 0.01):
    """Deterministic pseudo-random walk with drift; returns store-ready rows."""
    rows, px = [], start_price
    d = date.today() - timedelta(days=int(n_days * 1.45))  # ~calendar→trading days
    i = 0
    while len(rows) < n_days:
        d += timedelta(days=1)
        if d.weekday() >= 5:
            continue
        i += 1
        px *= 1 + daily_drift + wiggle * math.sin(i * 0.7) * 0.5
        rows.append({"d": d.isoformat(), "open": px * 0.999, "high": px * 1.01,
                     "low": px * 0.99, "close": px, "volume": 1_000_000})
    return rows


@pytest.fixture()
def cfg(tmp_path):
    return load_config("paper", overrides={
        "db_path": str(tmp_path / "test.db"),
        "universe": {"symbols": SYMS + ["SPY"], "benchmark": "SPY", "vix_symbol": "^VIX"},
        # only deterministic offline nodes in tests
        "nodes": {"momentum": {"enabled": True, "weight": 0.5, "horizon_days": 20,
                               "status": "production"}},
    })


@pytest.fixture()
def store(cfg):
    s = Store(cfg.get("db_path"))
    for sym in SYMS:
        s.upsert_bars(sym, synth_bars(daily_drift=0.002), "test")
    s.upsert_bars("SPY", synth_bars(daily_drift=0.0008), "test")
    vix = [{**r, "close": 15.0, "open": 15, "high": 16, "low": 14} for r in synth_bars()]
    s.upsert_bars("^VIX", vix, "test")
    return s
