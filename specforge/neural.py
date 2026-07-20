"""Causal temporal neural node with immutable global/holding champions.

The model predicts 5d/21d excess-return quantiles from 60-session multivariate
windows. Research always writes a challenger first; live inference reads only
a model_runs row explicitly marked champion. Repeating the same snapshot is
bounded by a persisted trial counter and can never mutate that champion.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .ml import lifecycle as ml_lifecycle
from .ml import targets as ml_targets


class NeuralModelOutput(NamedTuple):
    """Structured dual-family model output (tensors).

    absolute_quantiles / excess_quantiles: [batch, n_horizons, 3] (q10,q50,q90)
    probability_* : [batch, n_horizons]
    """
    absolute_quantiles: Any
    excess_quantiles: Any
    probability_absolute_edge_positive: Any
    probability_excess_positive: Any

FEATURES = ["r1", "range", "gap", "volume_z", "vol21", "rsi14",
            "atr14", "breakout60", "sma50_d", "sma200_d", "spy_r1", "spy_r21",
            "sector_relative_r21", "vix", "valuation", "valuation_missing",
            "event_proximity", "event_missing", "vix9d", "vix3m", "vix6m",
            "vvix", "vix_curve_9d_3m", "vix_curve_1m_3m",
            "implied_realized_spread", "hyg_r21", "tlt_r21",
            "vol_context_missing", "revenue_growth", "revenue_growth_missing",
            "operating_margin", "operating_margin_missing", "fcf_margin",
            "fcf_margin_missing", "debt_assets", "debt_assets_missing",
            "dilution", "dilution_missing", "accruals", "accruals_missing",
            "liquidity", "liquidity_missing", "news_sentiment", "news_missing"]
QUANTILES = (0.1, 0.5, 0.9)

# Feature split (B3): the temporal branch sees ONLY sequence-varying market
# features across all 60 sessions; the context branch sees ONLY point-in-time
# company/event state, and only from the latest session. Annual fundamentals
# were previously fed as 60 identical copies through the TCN — meaningless
# repetition the sequence model had to learn to ignore.
TEMPORAL_FEATURES = ("r1", "range", "gap", "volume_z", "vol21", "rsi14",
                     "atr14", "breakout60", "sma50_d", "sma200_d", "spy_r1",
                     "spy_r21", "sector_relative_r21", "vix", "vix9d", "vix3m",
                     "vix6m", "vvix", "vix_curve_9d_3m", "vix_curve_1m_3m",
                     "implied_realized_spread", "hyg_r21", "tlt_r21",
                     "vol_context_missing")
CONTEXT_FEATURES = ("valuation", "valuation_missing", "event_proximity",
                    "event_missing", "revenue_growth", "revenue_growth_missing",
                    "operating_margin", "operating_margin_missing", "fcf_margin",
                    "fcf_margin_missing", "debt_assets", "debt_assets_missing",
                    "dilution", "dilution_missing", "accruals", "accruals_missing",
                    "liquidity", "liquidity_missing", "news_sentiment",
                    "news_missing")
TEMPORAL_IDX = tuple(FEATURES.index(f) for f in TEMPORAL_FEATURES)
CONTEXT_IDX = tuple(FEATURES.index(f) for f in CONTEXT_FEATURES)
TEMPORAL_HASH = hashlib.sha256("|".join(TEMPORAL_FEATURES).encode()).hexdigest()[:16]
CONTEXT_HASH = hashlib.sha256("|".join(CONTEXT_FEATURES).encode()).hexdigest()[:16]

MODEL_SCHEMA = 6           # dual-target (absolute+excess) dual-branch checkpoints
FEATURE_HASH = hashlib.sha256("|".join(FEATURES).encode()).hexdigest()[:16]
ARCHITECTURE_HASH = hashlib.sha256(
    b"tcn-v8:temporal(24)conv32:k3:d1,2,4,8,16:gelu:dropout.1:context(20)mlp16:"
    b"dual-heads(absolute+excess):qheads:probheads:rank-loss:calibrated"
).hexdigest()[:16]

TRIAL_SPECS = (
    {"lr": 1e-3, "weight_decay": 1e-4, "rank_weight": .03},
    {"lr": 5e-4, "weight_decay": 1e-4, "rank_weight": .05},
    {"lr": 3e-4, "weight_decay": 3e-4, "rank_weight": .08},
    {"lr": 1e-3, "weight_decay": 1e-3, "rank_weight": .05},
    {"lr": 2e-4, "weight_decay": 1e-4, "rank_weight": .10},
    {"lr": 7e-4, "weight_decay": 5e-4, "rank_weight": .08},
)

# A 60 x 44 float32 window is roughly 10 KiB before targets, masks, and
# framework tensors.  Accepting the historical 250k-window setting can
# therefore consume several GiB while the live service is still resident.
# This is a process-safety ceiling, not a tuning knob: configuration may lower
# it, but cannot raise it without a reviewed code change.
SAFE_MAX_TRAINING_WINDOWS = 12_000


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _code_commit() -> str | None:
    """Best-effort training-code commit for checkpoint provenance; None if git
    is unavailable. Never raises — provenance is a nice-to-have, not a gate."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=2,
                             cwd=os.path.dirname(os.path.dirname(__file__)))
        return out.stdout.strip()[:12] or None if out.returncode == 0 else None
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    """Hash a checkpoint without duplicating the entire artifact in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(cfg, symbol: str | None = None, challenger: bool = False,
          run_id: str | None = None) -> Path:
    if symbol:
        root = Path(cfg.get("neural", "holdings_dir", default="data/models/holdings"))
        return root / (f"{symbol}.{run_id}.pt" if challenger and run_id else f"{symbol}.pt")
    p = Path(cfg.get("neural", "checkpoint", default="data/models/global_tcn.pt"))
    return p.with_name(f"{p.stem}.{run_id}.pt") if challenger and run_id else p


def _save(path: Path, payload: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _bars(store, symbol: str, since: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT d,open,high,low,close,volume FROM bars WHERE symbol=? AND d>=? "
        "ORDER BY d", store.db, params=(symbol, since)).set_index("d")


def _valuation_series(store, symbol: str, index, prices) -> tuple[pd.Series, pd.Series]:
    """Historical P/E from then-known annual diluted EPS; never current Yahoo P/E."""
    inst = store.db.execute("SELECT cik FROM instruments WHERE symbol=?", (symbol,)).fetchone()
    if not inst or not inst["cik"]:
        return pd.Series(0.0, index=index), pd.Series(1.0, index=index)
    rows = store.db.execute(
        "SELECT filed,period_end,value FROM filing_facts WHERE cik=? "
        "AND tag='EarningsPerShareDiluted' "
        "AND form IN ('10-K','10-K/A') ORDER BY filed", (str(inst["cik"]),)).fetchall()
    if not rows:
        return pd.Series(0.0, index=index), pd.Series(1.0, index=index)
    # A 10-K repeats comparative prior years under the same filing date. Keep
    # the latest period that was actually reported on each availability date.
    by_filed = {}
    for r in rows:
        current = by_filed.get(r["filed"])
        if current is None or r["period_end"] > current[0]:
            by_filed[r["filed"]] = (r["period_end"], float(r["value"]))
    known = pd.Series({filed: value for filed, (_, value) in by_filed.items()})
    eps = known.reindex(known.index.union(index)).sort_index().ffill().reindex(index)
    pe = (prices / eps.replace(0, np.nan)).clip(-100, 100) / 25.0
    return pe.fillna(0.0), pe.isna().astype(float)


def _event_series(store, symbol: str, index) -> tuple[pd.Series, pd.Series]:
    """Post-filing event proximity using only then-public filing dates."""
    inst = store.db.execute("SELECT cik FROM instruments WHERE symbol=?", (symbol,)).fetchone()
    if not inst or not inst["cik"]:
        return pd.Series(0.0, index=index), pd.Series(1.0, index=index)
    rows = store.db.execute(
        "SELECT DISTINCT filed FROM filing_facts WHERE cik=? AND form IN "
        "('10-K','10-K/A','10-Q','10-Q/A') ORDER BY filed", (str(inst["cik"]),)).fetchall()
    if not rows:
        return pd.Series(0.0, index=index), pd.Series(1.0, index=index)
    dates = np.asarray([np.datetime64(r["filed"]) for r in rows])
    observed = np.asarray(index, dtype="datetime64[D]")
    positions = np.searchsorted(dates, observed, side="right") - 1
    missing = positions < 0
    safe = np.maximum(positions, 0)
    days = (observed - dates[safe]).astype("timedelta64[D]").astype(float)
    # np.where evaluates both branches. Clip pre-filing negative offsets before
    # exponentiation so missing history cannot create overflow warnings during
    # broad shadow inference; the mask still makes those rows explicitly zero.
    decay = np.exp(-np.clip(days, 0.0, 30.0) / 10.0)
    proximity = np.where(missing | (days > 30), 0.0, decay)
    return (pd.Series(proximity, index=index),
            pd.Series(missing.astype(float), index=index))


def _fundamental_series(store, symbol: str, index) -> dict[str, pd.Series]:
    """Then-known SEC facts transformed only on their filing availability date."""
    inst = store.db.execute("SELECT cik FROM instruments WHERE symbol=?", (symbol,)).fetchone()
    names = ("revenue_growth", "operating_margin", "fcf_margin", "debt_assets",
             "dilution", "accruals", "liquidity")
    empty = {n: pd.Series(0.0, index=index) for n in names}
    empty.update({n + "_missing": pd.Series(1.0, index=index) for n in names})
    if not inst or not inst["cik"]:
        return empty
    tags = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
            "OperatingIncomeLoss", "NetCashProvidedByUsedInOperatingActivities",
            "PaymentsToAcquirePropertyPlantAndEquipment", "LongTermDebtCurrent",
            "LongTermDebtNoncurrent", "LongTermDebt", "Assets", "NetIncomeLoss",
            "AssetsCurrent", "LiabilitiesCurrent", "CommonStocksIncludingAdditionalPaidInCapital",
            "CommonStockSharesOutstanding")
    marks = ",".join("?" for _ in tags)
    rows = store.db.execute(
        f"SELECT filed,period_end,tag,value FROM filing_facts WHERE cik=? "
        "AND form IN ('10-K','10-K/A') "
        f"AND tag IN ({marks}) "
        "ORDER BY filed,period_end", (str(inst["cik"]), *tags)).fetchall()
    if not rows:
        return empty
    grouped: dict[str, dict[str, tuple[str, float]]] = {}
    for row in rows:
        current = grouped.setdefault(row["filed"], {}).get(row["tag"])
        if current is None or row["period_end"] >= current[0]:
            grouped[row["filed"]][row["tag"]] = (row["period_end"], float(row["value"]))
    state, prior = {}, {}; snapshots = {}
    for filed, facts in sorted(grouped.items()):
        for tag, (_, value) in facts.items(): state[tag] = value
        revenue = state.get("RevenueFromContractWithCustomerExcludingAssessedTax",
                            state.get("Revenues"))
        # Paid-in capital is a dollar balance, not a share count.  Treat a
        # missing share fact as missing instead of manufacturing dilution from
        # dimensionally incompatible SEC data.
        shares = state.get("CommonStockSharesOutstanding")
        assets = state.get("Assets")
        cfo = state.get("NetCashProvidedByUsedInOperatingActivities")
        capex = state.get("PaymentsToAcquirePropertyPlantAndEquipment")
        debt = sum(v for v in (state.get("LongTermDebtCurrent"),
                               state.get("LongTermDebtNoncurrent")) if v is not None)
        if not debt: debt = state.get("LongTermDebt")
        values = {
            "revenue_growth": ((revenue / prior["revenue"] - 1) if revenue is not None and
                               prior.get("revenue") not in (None, 0) else None),
            "operating_margin": (state.get("OperatingIncomeLoss") / revenue
                                 if revenue not in (None, 0) and
                                 state.get("OperatingIncomeLoss") is not None else None),
            "fcf_margin": ((cfo - (capex or 0)) / revenue if cfo is not None and
                           revenue not in (None, 0) else None),
            "debt_assets": (debt / assets if debt is not None and assets not in (None, 0) else None),
            "dilution": ((shares / prior["shares"] - 1) if shares is not None and
                         prior.get("shares") not in (None, 0) else None),
            "accruals": ((state.get("NetIncomeLoss") - cfo) / assets
                         if state.get("NetIncomeLoss") is not None and cfo is not None and
                         assets not in (None, 0) else None),
            "liquidity": (state.get("AssetsCurrent") / state.get("LiabilitiesCurrent") - 1
                          if state.get("AssetsCurrent") is not None and
                          state.get("LiabilitiesCurrent") not in (None, 0) else None),
        }
        snapshots[filed] = {k: (max(-3.0, min(3.0, v)) if v is not None else None)
                            for k, v in values.items()}
        if revenue is not None: prior["revenue"] = revenue
        if shares is not None: prior["shares"] = shares
    out = {}
    for name in names:
        series = pd.Series({d: value[name] for d, value in snapshots.items()}, dtype=float)
        aligned = series.reindex(series.index.union(index)).sort_index().ffill().reindex(index)
        out[name] = aligned.fillna(0.0)
        out[name + "_missing"] = aligned.isna().astype(float)
    return out


_NEWS_KNOWN_AT = ("MAX(substr(published_at,1,10),"
                  "substr(COALESCE(classified_at,published_at),1,10))")


def news_pit_stats(store) -> dict:
    """Provenance for the news feature: how much of it was scored after the fact.

    A classification lag of days or years means the sentiment score was NOT
    available on the publication date. The feature keys off `known_at`, so
    retro-scored history is simply absent — this reports how much.
    """
    row = store.db.execute(
        "SELECT COUNT(*) n,"
        " SUM(julianday(substr(classified_at,1,10))-julianday(substr(published_at,1,10))>1) retro,"
        " MAX(julianday(substr(classified_at,1,10))-julianday(substr(published_at,1,10))) lag "
        "FROM news_intelligence WHERE classified_at IS NOT NULL").fetchone()
    return {"classified": int(row["n"] or 0),
            "retro_classified": int(row["retro"] or 0),
            "max_lag_days": float(row["lag"] or 0.0)}


def _news_series(store, symbol: str, index) -> tuple[pd.Series, pd.Series]:
    # known_at, not published_at: the score exists only once the classifier ran.
    rows = store.db.execute(
        f"SELECT {_NEWS_KNOWN_AT} d,AVG(stance*confidence*reliability) score "
        "FROM news_intelligence WHERE symbol=? AND classified_at IS NOT NULL GROUP BY 1",
        (symbol,)).fetchall()
    if not rows:
        return pd.Series(0.0, index=index), pd.Series(1.0, index=index)
    series = pd.Series({r["d"]: float(r["score"] or 0) for r in rows})
    aligned = series.reindex(series.index.union(index)).sort_index().rolling(3, min_periods=1).mean()
    aligned = aligned.reindex(index)
    return aligned.fillna(0.0), aligned.isna().astype(float)


def _features(b: pd.DataFrame, spy: pd.DataFrame, vix: pd.DataFrame,
              store=None, symbol: str = "",
              sectors: dict[str, pd.DataFrame] | None = None,
              context: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    c = b["close"].astype(float)
    r = c.pct_change()
    f = pd.DataFrame(index=b.index)
    f["r1"] = r
    f["range"] = (b["high"] - b["low"]) / c.replace(0, np.nan)
    f["gap"] = b["open"] / c.shift(1) - 1
    lv = np.log1p(b["volume"].astype(float))
    f["volume_z"] = (lv - lv.rolling(21).mean()) / (lv.rolling(21).std() + 1e-8)
    f["vol21"] = r.rolling(21).std()
    prev = c.shift(1)
    tr = pd.concat(((b["high"] - b["low"]), (b["high"] - prev).abs(),
                    (b["low"] - prev).abs()), axis=1).max(axis=1)
    f["atr14"] = tr.rolling(14).mean() / c.replace(0, np.nan)
    low60, high60 = c.rolling(60).min(), c.rolling(60).max()
    f["breakout60"] = (c - low60) / (high60 - low60 + 1e-8) - .5
    gain = r.clip(lower=0).rolling(14).mean()
    loss = (-r.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = gain / (gain + loss + 1e-8) - 0.5
    f["sma50_d"] = c / c.rolling(50).mean() - 1
    f["sma200_d"] = c / c.rolling(200).mean() - 1
    sp = spy["close"].reindex(f.index).ffill().astype(float)
    f["spy_r1"] = sp.pct_change()
    f["spy_r21"] = sp.pct_change(21)
    sector_returns = {}
    for sector_symbol, sector_bars in (sectors or {}).items():
        if len(sector_bars):
            sector_close = sector_bars["close"].reindex(f.index).ffill().astype(float)
            sector_returns[sector_symbol] = sector_close.pct_change()
    if sector_returns:
        sr = pd.DataFrame(sector_returns, index=f.index)
        # Each date chooses the sector ETF whose prior 60 sessions best match
        # this stock. Rolling correlations and 21d returns use no future rows.
        correlations = sr.apply(lambda series: r.rolling(60).corr(series))
        valid = correlations.notna().any(axis=1)
        chosen = pd.Series(index=f.index, dtype=object)
        if valid.any():
            chosen.loc[valid] = correlations.loc[valid].idxmax(axis=1)
        sector_r21 = sr.add(1).rolling(21).apply(np.prod, raw=True) - 1
        proxy = pd.Series(index=f.index, dtype=float)
        for sector_symbol in sector_r21.columns:
            mask = chosen == sector_symbol
            proxy.loc[mask] = sector_r21.loc[mask, sector_symbol]
        f["sector_relative_r21"] = c.pct_change(21) - proxy
    else:
        f["sector_relative_r21"] = c.pct_change(21) - sp.pct_change(21)
    if len(vix):
        vix_series = vix["close"].reindex(f.index).ffill()
        f["vix"] = vix_series / 20.0 - 1
    else:
        vix_series = pd.Series(np.nan, index=f.index)
        f["vix"] = 0.0
    context = context or {}
    def aligned(name):
        frame = context.get(name)
        return (frame["close"].reindex(f.index).ffill().astype(float)
                if frame is not None and len(frame) else pd.Series(np.nan, index=f.index))
    vix9d, vix3m, vix6m, vvix = (aligned("vix9d"), aligned("vix3m"),
                                 aligned("vix6m"), aligned("vvix"))
    f["vix9d"] = (vix9d / 20.0 - 1).fillna(0.0)
    f["vix3m"] = (vix3m / 20.0 - 1).fillna(0.0)
    f["vix6m"] = (vix6m / 20.0 - 1).fillna(0.0)
    f["vvix"] = (vvix / 100.0 - 1).fillna(0.0)
    f["vix_curve_9d_3m"] = (vix9d / vix3m - 1).fillna(0.0)
    f["vix_curve_1m_3m"] = (vix_series / vix3m - 1).fillna(0.0)
    realized = sp.pct_change().rolling(21).std() * math.sqrt(252) * 100
    f["implied_realized_spread"] = ((vix_series - realized) / 20.0).fillna(0.0)
    hyg, tlt = aligned("hyg"), aligned("tlt")
    f["hyg_r21"] = hyg.pct_change(21).fillna(0.0)
    f["tlt_r21"] = tlt.pct_change(21).fillna(0.0)
    f["vol_context_missing"] = (vix9d.isna() | vix3m.isna() | vix6m.isna() |
                                vvix.isna()).astype(float)
    if store is not None and symbol:
        f["valuation"], f["valuation_missing"] = _valuation_series(
            store, symbol, f.index, c)
    else:
        f["valuation"], f["valuation_missing"] = 0.0, 1.0
    if store is not None and symbol:
        f["event_proximity"], f["event_missing"] = _event_series(store, symbol, f.index)
        for name, values in _fundamental_series(store, symbol, f.index).items():
            f[name] = values
        f["news_sentiment"], f["news_missing"] = _news_series(store, symbol, f.index)
    else:
        f["event_proximity"], f["event_missing"] = 0.0, 1.0
        for name in ("revenue_growth", "operating_margin", "fcf_margin", "debt_assets",
                     "dilution", "accruals", "liquidity"):
            f[name], f[name + "_missing"] = 0.0, 1.0
        f["news_sentiment"], f["news_missing"] = 0.0, 1.0
    return f.replace([np.inf, -np.inf], np.nan)


def _training_window_limit(cfg) -> int:
    requested = int(cfg.get("neural", "max_training_windows", default=12_000))
    return max(100, min(requested, SAFE_MAX_TRAINING_WINDOWS))


def _dataset_should_yield(deadline: float | None, cancelled) -> bool:
    return bool((deadline is not None and time.monotonic() >= deadline) or
                (cancelled is not None and cancelled()))


def build_dataset(cfg, store, symbols: list[str] | None = None, progress=None,
                  *, deadline: float | None = None, cancelled=None) -> dict:
    window = int(cfg.get("neural", "input_sessions", default=60))
    horizons = tuple(cfg.get("neural", "horizons", default=[5, 21]))
    since = cfg.get("neural", "train_since", default="2011-01-01")
    # Today's configured list is survivor-biased: it cannot contain a name that
    # was delisted before today. Union it with every symbol that was ever a
    # member so the losers stay in the panel.
    from . import universe as _universe
    symbols = symbols or sorted(
        {s for s in cfg.get("universe", "symbols", default=[]) if not s.startswith("^")}
        | set(_universe.historical_symbols(store)))
    pit_dates = sorted(pit_history := _universe.membership_history(store))
    bench = cfg.get("universe", "benchmark", default="SPY")
    spy, vix = _bars(store, bench, since), _bars(
        store, cfg.get("universe", "vix_symbol", default="^VIX"), since)
    vol_symbols = cfg.get("universe", "volatility_symbols", default={}) or {}
    context = {name: _bars(store, ticker, since) for name, ticker in vol_symbols.items()
               if name != "vix"}
    context.update({"hyg": _bars(store, "HYG", since), "tlt": _bars(store, "TLT", since)})
    sector_symbols = cfg.get("universe", "sector_etfs", default=[
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"])
    sectors = {s: _bars(store, s, since) for s in sector_symbols}
    if len(spy) < window + max(horizons) + 100:
        return {"error": "not enough benchmark history"}
    X, Y, Y_abs, dates, owners, sample_cost = [], [], [], [], [], []
    cost_floor = ml_targets.round_trip_cost(cfg)
    pit_covered = pit_dropped = pit_uncovered = 0
    window_limit = _training_window_limit(cfg)
    per_symbol_cap = min(int(cfg.get("neural", "max_windows_per_symbol", default=500)),
                         max(1, window_limit // max(1, len(symbols))))
    for symbol_index, sym in enumerate(symbols, 1):
        if _dataset_should_yield(deadline, cancelled):
            return {"status": "yielded", "reason": "dataset build deadline reached",
                    "windows_built": len(X), "symbols_completed": symbol_index - 1}
        if progress:
            progress({"phase": "dataset", "symbol": sym, "index": symbol_index,
                      "total": len(symbols),
                      "fraction": .2 * symbol_index / max(1, len(symbols))})
        b = _bars(store, sym, since)
        if len(b) < window + max(horizons) + 100:
            continue
        f = _features(b, spy, vix, store, sym, sectors, context)
        c = b["close"].astype(float)
        sp = spy["close"].reindex(c.index).ffill().astype(float)
        # Dual targets: absolute stock return AND benchmark-excess return, same
        # decision date and horizon. Excess reproduces the prior definition.
        abs_df, exc_df = ml_targets.build_targets(c, spy["close"], horizons)
        vals = f[FEATURES].to_numpy(np.float32)
        yvals = exc_df.to_numpy(np.float32)
        yvals_abs = abs_df.to_numpy(np.float32)
        row_valid = np.isfinite(vals).all(axis=1).astype(np.int16)
        # A window is usable only when all 60 rows and both future targets are
        # finite. Select evenly across history instead of materializing every
        # overlapping window (which grows to many GB at 1,500 symbols).
        complete = np.convolve(row_valid, np.ones(window, dtype=np.int16),
                               mode="valid") == window
        indices = np.flatnonzero(complete) + window - 1
        indices = indices[(indices < len(f) - max(horizons)) &
                          np.isfinite(yvals[indices]).all(axis=1) &
                          np.isfinite(yvals_abs[indices]).all(axis=1)]
        if pit_dates:
            # known_at <= decision_at for universe membership: a window only
            # counts on a date the symbol was actually in the tradable panel.
            keep = []
            for i in indices:
                position = bisect.bisect_right(pit_dates, f.index[i]) - 1
                if position < 0:
                    pit_uncovered += 1; keep.append(i)          # predates coverage
                elif sym in pit_history[pit_dates[position]]:
                    pit_covered += 1; keep.append(i)
                else:
                    pit_dropped += 1
            indices = np.asarray(keep, dtype=int)
            if not len(indices):
                continue
        costs = ml_targets.sample_costs(cfg, b).to_numpy(np.float32)
        if len(indices) > per_symbol_cap:
            indices = indices[np.linspace(0, len(indices) - 1,
                                          per_symbol_cap, dtype=int)]
        remaining = window_limit - len(X)
        if remaining <= 0:
            break
        for offset, i in enumerate(indices[:remaining]):
            if offset % 128 == 0 and _dataset_should_yield(deadline, cancelled):
                return {"status": "yielded", "reason": "dataset build deadline reached",
                        "windows_built": len(X), "symbols_completed": symbol_index - 1}
            X.append(vals[i - window + 1:i + 1])
            Y.append(yvals[i]); Y_abs.append(yvals_abs[i])
            dates.append(f.index[i]); owners.append(sym)
            sample_cost.append(costs[i])
    if len(X) < 100:
        return {"error": f"not enough training windows ({len(X)})"}
    X, Y, Y_abs = np.stack(X), np.stack(Y), np.stack(Y_abs)
    sample_cost = np.asarray(sample_cost, dtype=np.float32)
    median_cost = float(np.median(sample_cost))
    unique = sorted(set(dates))
    if len(unique) < 180:
        return {"error": f"not enough distinct dates ({len(unique)})"}
    # Chronological train/validation/test with horizon embargoes.
    test_start = unique[int(len(unique) * 0.85)]
    val_start = unique[int(len(unique) * 0.70)]
    embargo = max(horizons)
    val_pos, test_pos = unique.index(val_start), unique.index(test_start)
    # Strict embargo: a training target ending exactly on the first validation
    # or test session is still leakage. Leave one additional settled session.
    train_end = unique[max(0, val_pos - embargo - 1)]
    val_end = unique[max(val_pos, test_pos - embargo - 1)]
    d = np.asarray(dates)
    masks = {"train": d <= train_end,
             "val": (d >= val_start) & (d <= val_end),
             "test": d >= test_start}
    mean = X[masks["train"]].mean((0, 1), keepdims=True)
    std = X[masks["train"]].std((0, 1), keepdims=True) + 1e-6
    target_scale = np.maximum(Y[masks["train"]].std(axis=0), .005).astype(np.float32)
    target_scale_absolute = np.maximum(
        Y_abs[masks["train"]].std(axis=0), .005).astype(np.float32)
    # Normalize the already-owned float32 array in place.  The prior expression
    # allocated another full dataset-sized array at peak memory.
    np.subtract(X, mean, out=X)
    np.divide(X, std, out=X)
    return {"X": X, "Y": Y, "Y_excess": Y, "Y_absolute": Y_abs, "dates": d,
            "owners": np.asarray(owners), "masks": masks,
            "mean": mean, "std": std, "horizons": horizons,
            "target_scale": target_scale,
            "target_scale_absolute": target_scale_absolute,
            "sample_cost": sample_cost,
            "cost_floor": cost_floor,
            # The scalar survives only as the calibration threshold and as the
            # reported central cost; labels use each sample's own estimate.
            "round_trip_cost": median_cost, "cost_threshold": median_cost,
            "pit": {"universe_snapshots": len(pit_dates),
                    "universe_covered_windows": pit_covered,
                    "universe_dropped_windows": pit_dropped,
                    "universe_uncovered_windows": pit_uncovered,
                    "news": news_pit_stats(store)},
            "target_schema_hash": ml_targets.TARGET_SCHEMA_HASH,
            "data_as_of": unique[-1], "train_end": train_end,
            "val_start": val_start, "val_end": val_end,
            "test_start": test_start}


def _make_model(n_features: int, n_horizons: int,
                temporal_idx=None, context_idx=None):
    import torch
    import torch.nn as nn
    temporal_idx = list(TEMPORAL_IDX if temporal_idx is None else temporal_idx)
    context_idx = list(CONTEXT_IDX if context_idx is None else context_idx)

    class CausalBlock(nn.Module):
        def __init__(self, inp, out, dilation):
            super().__init__()
            pad = (3 - 1) * dilation
            self.pad = pad
            self.conv = nn.Conv1d(inp, out, 3, padding=pad, dilation=dilation)
            self.proj = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()
            self.drop = nn.Dropout(0.1)

        def forward(self, x):
            y = self.conv(x)[..., :-self.pad] if self.pad else self.conv(x)
            return torch.nn.functional.gelu(self.drop(y) + self.proj(x))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            # Five dilated blocks → receptive field 1+2·(1+2+4+8+16)=63 sessions,
            # so the full 60-session window can influence the output. Three
            # blocks (d1,2,4) reached only ~15 sessions. See NN_REPAIR plan B2.
            self.temporal_idx = temporal_idx
            self.context_idx = context_idx
            self.blocks = nn.Sequential(CausalBlock(len(temporal_idx), 32, 1),
                                        CausalBlock(32, 32, 2),
                                        CausalBlock(32, 32, 4),
                                        CausalBlock(32, 32, 8),
                                        CausalBlock(32, 32, 16))
            self.context = nn.Sequential(nn.Linear(len(context_idx), 16), nn.GELU(),
                                         nn.Dropout(.1))
            # Dual return families, one quantile + one probability head per
            # horizon each. Absolute and excess are separate heads so the node
            # can read a genuine absolute forecast, not excess reinterpreted.
            self.excess_quantile_heads = nn.ModuleList(
                nn.Linear(48, 3) for _ in range(n_horizons))
            self.excess_probability_heads = nn.ModuleList(
                nn.Linear(48, 1) for _ in range(n_horizons))
            self.absolute_quantile_heads = nn.ModuleList(
                nn.Linear(48, 3) for _ in range(n_horizons))
            self.absolute_probability_heads = nn.ModuleList(
                nn.Linear(48, 1) for _ in range(n_horizons))

        def encoded(self, x):
            # Temporal branch: only sequence-varying features, all sessions.
            # Context branch: only point-in-time features, latest session.
            temporal_in = x[:, :, self.temporal_idx].transpose(1, 2)
            temporal = self.blocks(temporal_in)[..., -1]
            return torch.cat((temporal, self.context(x[:, -1, self.context_idx])), dim=1)

        @staticmethod
        def _quantiles(z, heads):
            raw = torch.stack([head(z) for head in heads], dim=1)
            q50 = raw[..., 1]
            q10 = q50 - torch.nn.functional.softplus(raw[..., 0])
            q90 = q50 + torch.nn.functional.softplus(raw[..., 2])
            return torch.stack((q10, q50, q90), dim=-1)

        @staticmethod
        def _probs(z, heads):
            return torch.sigmoid(torch.cat([head(z) for head in heads], dim=1))

        def forward_structured(self, x):
            """Full dual-family output from one encoder pass."""
            z = self.encoded(x)
            return NeuralModelOutput(
                absolute_quantiles=self._quantiles(z, self.absolute_quantile_heads),
                excess_quantiles=self._quantiles(z, self.excess_quantile_heads),
                probability_absolute_edge_positive=self._probs(
                    z, self.absolute_probability_heads),
                probability_excess_positive=self._probs(
                    z, self.excess_probability_heads))

        def forward_all(self, x):
            """Legacy excess-only (quantiles, probabilities). Transitional —
            unmigrated consumers (inference, calibration, metrics) read this
            until B4C migrates them to forward_structured."""
            z = self.encoded(x)
            return (self._quantiles(z, self.excess_quantile_heads),
                    self._probs(z, self.excess_probability_heads))

        def forward_legacy_excess(self, x):
            return self._quantiles(self.encoded(x), self.excess_quantile_heads)

        def forward(self, x):
            return self._quantiles(self.encoded(x), self.excess_quantile_heads)

        def probability(self, x):
            return self._probs(self.encoded(x), self.excess_probability_heads)
    return TCN()


def _pinball(pred, target):
    import torch
    q = torch.tensor(QUANTILES, device=pred.device).view(1, 1, 3)
    err = target.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1) * err).mean()


def _rank_loss(pred, target, dates) -> object:
    """Small same-session pairwise loss, aligned with live cross-sectional rank."""
    import torch
    losses = []
    for day in np.unique(dates):
        loc = np.flatnonzero(dates == day)
        if len(loc) < 2:
            continue
        # Adjacent deterministic pairs cap quadratic cost while retaining sign.
        a = torch.as_tensor(loc[:-1], device=pred.device)
        b = torch.as_tensor(loc[1:], device=pred.device)
        truth = torch.sign(target[a] - target[b])
        mask = truth != 0
        if mask.any():
            losses.append(torch.nn.functional.softplus(
                -truth[mask] * (pred[a][mask] - pred[b][mask])).mean())
    return torch.stack(losses).mean() if losses else pred.sum() * 0


def _training_device(cfg, torch) -> str:
    requested = str(cfg.get("neural", "device", default="auto")).lower()
    if requested in ("cpu", "mps", "cuda"):
        if requested == "cuda" and not torch.cuda.is_available(): return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available(): return "cpu"
        return requested
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def _predict_batches(model, X, indices, scale, device: str, batch_size: int = 2048):
    import torch
    predictions, probabilities = [], []
    model.eval()
    with torch.no_grad():
        for chunk in torch.as_tensor(indices).split(batch_size):
            xb = X[chunk].to(device)
            prediction, probability = model.forward_all(xb)
            predictions.append(prediction.cpu().numpy())
            probabilities.append(probability.cpu().numpy())
    pred = np.concatenate(predictions) * np.asarray(scale).reshape(1, -1, 1)
    return pred, np.concatenate(probabilities)


def _predict_structured(model, X, indices, scale_excess, scale_absolute,
                        device: str, batch_size: int = 2048):
    """Both return families in one pass: (abs_pred, abs_prob, exc_pred, exc_prob),
    each unscaled by its own train-only volatility scale."""
    import torch
    ap, aq, ep, eq = [], [], [], []
    model.eval()
    with torch.no_grad():
        for chunk in torch.as_tensor(indices).split(batch_size):
            out = model.forward_structured(X[chunk].to(device))
            ap.append(out.absolute_quantiles.cpu().numpy())
            aq.append(out.probability_absolute_edge_positive.cpu().numpy())
            ep.append(out.excess_quantiles.cpu().numpy())
            eq.append(out.probability_excess_positive.cpu().numpy())
    sa = np.asarray(scale_absolute).reshape(1, -1, 1)
    se = np.asarray(scale_excess).reshape(1, -1, 1)
    return (np.concatenate(ap) * sa, np.concatenate(aq),
            np.concatenate(ep) * se, np.concatenate(eq))


def _calibration(pred, probability, truth, prob_threshold: float = 0.0) -> dict:
    """Validation-only calibration for one return family. Pure function of its
    arguments — it never sees sealed/forward outcomes because they are not
    passed in. Per horizon: q50 median-bias offset, q10/q90 coverage offsets,
    and a probability logit offset toward the observed rate of exceeding
    `prob_threshold` (0 for excess-positive, round-trip cost for absolute-edge).
    """
    out = {"quantile_offsets": [], "probability_logit_offsets": [],
           "prob_threshold": float(prob_threshold)}
    for i in range(truth.shape[1]):
        out["quantile_offsets"].append([
            float(np.quantile(truth[:, i] - pred[:, i, 0], .10)),
            float(np.median(truth[:, i] - pred[:, i, 1])),   # q50 systematic bias
            float(np.quantile(truth[:, i] - pred[:, i, 2], .90))])
        observed = np.clip((truth[:, i] > prob_threshold).mean(), 1e-4, 1 - 1e-4)
        expected = np.clip(probability[:, i].mean(), 1e-4, 1 - 1e-4)
        out["probability_logit_offsets"].append(float(
            np.log(observed / (1 - observed)) - np.log(expected / (1 - expected))))
    return out


def _apply_calibration(pred, probability, calibration):
    pred = np.asarray(pred).copy(); probability = np.asarray(probability).copy()
    for i, offsets in enumerate((calibration or {}).get("quantile_offsets", [])):
        pred[:, i, :] += np.asarray(offsets)
        pred[:, i, 0] = np.minimum(pred[:, i, 0], pred[:, i, 1])
        pred[:, i, 2] = np.maximum(pred[:, i, 2], pred[:, i, 1])
    for i, offset in enumerate((calibration or {}).get("probability_logit_offsets", [])):
        p = np.clip(probability[:, i], 1e-5, 1 - 1e-5)
        probability[:, i] = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + offset)))
    return pred, probability


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 5 and np.std(a) and np.std(b) else 0.0


def _rank_ic(pred: np.ndarray, truth: np.ndarray, dates) -> float:
    if dates is None:
        return _corr(pred, truth)
    values = []
    dates = np.asarray(dates)
    for day in np.unique(dates):
        mask = dates == day
        if mask.sum() < 8:
            continue
        pr = pd.Series(pred[mask]).rank(method="average").to_numpy()
        tr = pd.Series(truth[mask]).rank(method="average").to_numpy()
        values.append(_corr(pr, tr))
    return float(np.mean(values)) if values else 0.0


def _top_decile_alpha(pred: np.ndarray, truth: np.ndarray, dates=None,
                      cost: float = .0016) -> float:
    """After-cost mean return of each session's highest-ranked decile.

    Pooling symbols across dates rewards market timing instead of the
    cross-sectional stock selection this model is used for.  Forward shadow
    and tournament metrics therefore use the same per-session definition.
    """
    pred, truth = np.asarray(pred), np.asarray(truth)
    if dates is None:
        if not len(pred):
            return 0.0
        selected = truth[pred >= np.quantile(pred, .9)]
        return float(selected.mean() - cost) if len(selected) else 0.0
    values = []
    dates = np.asarray(dates)
    for day in np.unique(dates):
        mask = dates == day
        if mask.sum() < 8:
            continue
        day_pred, day_truth = pred[mask], truth[mask]
        chosen = day_truth[day_pred >= np.quantile(day_pred, .9)]
        if len(chosen):
            values.append(float(chosen.mean() - cost))
    return float(np.mean(values)) if values else 0.0


def _metrics(pred: np.ndarray, y: np.ndarray, horizons, dates=None) -> dict:
    out = {}
    losses = []
    for i, h in enumerate(horizons):
        q10, q50, q90 = pred[:, i, 0], pred[:, i, 1], pred[:, i, 2]
        loss = float(np.mean(np.maximum(
            np.asarray(QUANTILES) * (y[:, i, None] - pred[:, i, :]),
            (np.asarray(QUANTILES) - 1) * (y[:, i, None] - pred[:, i, :]))))
        losses.append(loss)
        out[str(h)] = {"pinball": round(loss, 6),
                       "correlation": round(_rank_ic(q50, y[:, i], dates), 4),
                       "top_decile_alpha_after_cost": round(
                           _top_decile_alpha(q50, y[:, i], dates), 5),
                       "directional_accuracy": round(float((np.sign(q50) == np.sign(y[:, i])).mean()), 3),
                       "coverage": round(float(((y[:, i] >= q10) & (y[:, i] <= q90)).mean()), 3)}
    out["pinball"] = round(float(np.mean(losses)), 6)
    out["correlation_kind"] = "daily_rank_ic" if dates is not None else "time_series"
    return out


def _selection_score(metrics: dict) -> float:
    """Validation-only tournament utility aligned with the live ranker."""
    hs = [metrics.get(str(h), {}) for h in (5, 21)]
    if any("correlation" not in h for h in hs):
        return float("-inf")
    rank_ic = sum(float(h.get("correlation", 0)) for h in hs) / 2
    alpha = sum(float(h.get("top_decile_alpha_after_cost", 0)) for h in hs) / 2
    calibration = sum(abs(float(h.get("coverage", 0)) - .80) for h in hs) / 2
    pinball = sum(float(h.get("pinball", 1)) for h in hs) / 2
    return rank_ic + 4 * alpha - .25 * calibration - pinball


def _baseline_metrics(ds: dict, eval_idx: np.ndarray) -> dict:
    """Honest local-compute baselines evaluated on the identical untouched rows."""
    y = ds["Y"][eval_idx]
    horizons = ds["horizons"]
    dates = ds["dates"][eval_idx]
    raw = ds["X"] * ds["std"] + ds["mean"]
    train = ds["masks"]["train"]

    def bands(median, residual_source):
        pred = np.zeros((len(median), len(horizons), 3), dtype=np.float32)
        for i in range(len(horizons)):
            residual = residual_source[:, i]
            lo, hi = np.quantile(residual, [.10, .90])
            pred[:, i, 0], pred[:, i, 1], pred[:, i, 2] = \
                median[:, i] + lo, median[:, i], median[:, i] + hi
        return pred

    zero_median = np.zeros_like(y)
    zero = bands(zero_median, ds["Y"][train])
    last_r1 = raw[:, -1, FEATURES.index("r1")]
    momentum_all = np.column_stack(
        [last_r1 * math.sqrt(max(1, h)) for h in horizons]).astype(np.float32)
    momentum = bands(momentum_all[eval_idx], ds["Y"][train] - momentum_all[train])

    # Closed-form ridge on the latest context row. It is deliberately simple
    # and deterministic; a TCN that cannot beat it has not earned complexity.
    x = raw[:, -1, :].astype(np.float64)
    mean, std = x[train].mean(0), x[train].std(0) + 1e-6
    xn = (x - mean) / std
    design = np.column_stack((np.ones(train.sum()), xn[train]))
    reg = np.eye(design.shape[1]) * 1e-2; reg[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ ds["Y"][train])
    ridge_all = np.column_stack((np.ones(len(xn)), xn)) @ coef
    ridge = bands(ridge_all[eval_idx], ds["Y"][train] - ridge_all[train])
    return {"zero": _metrics(zero, y, horizons, dates),
            "momentum": _metrics(momentum, y, horizons, dates),
            "ridge": _metrics(ridge, y, horizons, dates)}


def _fold_windows(n_unique: int, folds: int, embargo: int) -> list[tuple[int, int, int]]:
    """Expanding purged walk-forward fold index positions over `n_unique`
    sorted sessions. Returns (train_pos, test_start_pos, test_end_pos) with a
    HALF-OPEN test range [test_start_pos, test_end_pos): adjacent folds share no
    session and none reach into the sealed block (>= 0.85·n). See NN_REPAIR A2.
    """
    initial, sealed = int(n_unique * .45), int(n_unique * .85)
    width = max(1, (sealed - initial - embargo) // folds)
    out: list[tuple[int, int, int]] = []
    for fold in range(folds):
        train_pos = initial + fold * width
        test_start_pos = train_pos + embargo + 1
        test_end_pos = sealed if fold == folds - 1 else min(sealed, test_start_pos + width)
        if test_start_pos >= test_end_pos:
            continue
        out.append((train_pos, test_start_pos, test_end_pos))
    return out


def _walk_forward_metrics(cfg, ds: dict, trial_spec: dict | None = None,
                          max_seconds: float | None = None) -> tuple[dict, list[tuple]]:
    """Five expanding, embargoed TCN folds before the final sealed block."""
    import torch
    if max_seconds is not None and max_seconds <= 0:
        return {"walk_forward_folds": 0, "folds": [],
                **{f"median_fold_ic_{h}d": 0.0 for h in ds["horizons"]}}, []
    dates, unique = ds["dates"], sorted(set(ds["dates"]))
    trial_spec = trial_spec or TRIAL_SPECS[0]
    device = _training_device(cfg, torch)
    folds = int(cfg.get("neural", "walk_forward_folds", default=5))
    embargo = max(ds["horizons"])
    windows = _fold_windows(len(unique), folds, embargo)
    raw = ds["X"] * ds["std"] + ds["mean"]
    results, oos = [], []
    started = time.time()
    for fold, (train_pos, test_start_pos, test_end_pos) in enumerate(windows):
        train_mask = dates <= unique[train_pos]
        # Half-open test range [test_start_pos, test_end_pos): adjacent folds
        # share no session and the last fold never reaches the sealed block.
        test_mask = (dates >= unique[test_start_pos]) & (dates < unique[test_end_pos])
        mean = raw[train_mask].mean((0, 1), keepdims=True)
        std = raw[train_mask].std((0, 1), keepdims=True) + 1e-6
        X = torch.from_numpy(((raw - mean) / std).astype(np.float32))
        scale_np = np.maximum(ds["Y"][train_mask].std(axis=0), .005).astype(np.float32)
        scale = torch.from_numpy(scale_np)
        Yraw = torch.from_numpy(ds["Y"])
        Y = Yraw / scale
        tr = torch.from_numpy(np.flatnonzero(train_mask))
        te = np.flatnonzero(test_mask)
        model = _make_model(len(FEATURES), len(ds["horizons"])).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=trial_spec["lr"],
                                weight_decay=trial_spec["weight_decay"])
        remaining_folds = max(1, len(windows) - fold)
        fold_deadline = (time.time() + max(1.0, (max_seconds - (time.time() - started)) /
                         remaining_folds)) if max_seconds else None
        epochs = min(int(cfg.get("neural", "walk_forward_epochs", default=6)),
                     int(cfg.get("neural", "max_epochs", default=50)))
        for _ in range(epochs):
            model.train()
            for idx in tr[torch.randperm(len(tr))].split(512):
                xb, yb, yr = X[idx].to(device), Y[idx].to(device), Yraw[idx].to(device)
                opt.zero_grad()
                prediction, positive_probability = model.forward_all(xb)
                loss = _pinball(prediction, yb) + .1 * \
                    torch.nn.functional.binary_cross_entropy(
                        positive_probability, (yr > 0).float())
                rank = sum(_rank_loss(prediction[:, h, 1], yb[:, h], dates[idx.numpy()])
                           for h in range(len(ds["horizons"]))) / len(ds["horizons"])
                loss = loss + float(trial_spec["rank_weight"]) * rank
                loss.backward(); opt.step()
                if fold_deadline and time.time() >= fold_deadline:
                    break
            if fold_deadline and time.time() >= fold_deadline:
                break
        pred, probability = _predict_batches(model, X, te, scale_np, device)
        m = _metrics(pred, ds["Y"][te], ds["horizons"], dates[te])
        fold_payload = {}
        for i, h in enumerate(ds["horizons"]):
            q50, truth = pred[:, i, 1], ds["Y"][te, i]
            fold_payload[f"net_alpha_{h}d"] = round(
                _top_decile_alpha(q50, truth, dates[te]), 5)
        results.append({"fold": fold + 1, "train_end": unique[train_pos],
                        "test_start": unique[test_start_pos], "n": len(te),
                        **{f"ic_{h}d": m[str(h)]["correlation"]
                           for h in ds["horizons"]}, **fold_payload})
        for row_index, sample_index in enumerate(te):
            for horizon_index, horizon in enumerate(ds["horizons"]):
                oos.append((str(dates[sample_index]), str(ds["owners"][sample_index]),
                            int(horizon), *map(float, pred[row_index, horizon_index]),
                            float(probability[row_index, horizon_index]),
                            float(ds["Y"][sample_index, horizon_index])))
    return {"walk_forward_folds": len(results), "folds": results,
            **{f"median_fold_ic_{h}d": round(float(np.median(
                [f[f'ic_{h}d'] for f in results])), 4) if results else 0.0
               for h in ds["horizons"]}}, oos


def train_challenger(cfg, store, symbols: list[str] | None = None,
                     symbol: str | None = None, max_seconds: int | None = None,
                     progress=None, cancelled=None) -> dict:
    try:
        import torch
    except ImportError:
        return {"error": "torch not installed — install .[neural]"}
    torch.set_num_threads(int(cfg.get("research", "max_cpu_threads", default=4)))
    market_snapshot = store.latest_bar_date(
        cfg.get("universe", "benchmark", default="SPY")) or "no-market-snapshot"
    tournament_key = f"{market_snapshot}_{MODEL_SCHEMA}_{FEATURE_HASH}_{ARCHITECTURE_HASH}"
    if not symbol and symbols:
        tournament = store.kv_get("neural_active_tournament")
        if tournament and tournament.get("key") == tournament_key:
            symbols = tournament["symbols"]
        else:
            tournament = {"key": tournament_key, "symbols": sorted(set(symbols)),
                          "started_at": _now()}
            store.kv_set("neural_active_tournament", tournament)
    if symbol:
        observed = store.db.execute("SELECT COUNT(*) n FROM bars WHERE symbol=?",
                                    (symbol,)).fetchone()["n"]
        required = int(cfg.get("neural", "holding_min_bars", default=1250))
        if observed < required:
            return {"status": "waiting", "kind": "holding_tcn", "symbol": symbol,
                    "reason": f"need {required} settled observations; have {observed}"}
    overall_deadline = (time.monotonic() + max_seconds
                        if max_seconds is not None else None)
    ds = build_dataset(cfg, store, symbols=[symbol] if symbol else symbols,
                       progress=progress, deadline=overall_deadline,
                       cancelled=cancelled)
    if ds.get("status") == "yielded":
        return ds
    if "error" in ds:
        return ds
    snapshot = ds["data_as_of"]
    fact_state = store.db.execute(
        "SELECT COUNT(*) n,COALESCE(MAX(rowid),0) newest FROM filing_facts").fetchone()
    data_fingerprint = hashlib.sha256(json.dumps({
        "owners": sorted(set(ds["owners"])), "data_as_of": snapshot,
        "facts": [fact_state["n"], fact_state["newest"]],
        "features": FEATURE_HASH}, sort_keys=True).encode()).hexdigest()[:16]
    # The settled market snapshot is the trial boundary.  Filing ingestion or
    # universe maintenance must not reset the six-trial cap and repeatedly
    # expose the same sealed test (the previous fingerprint-based key did).
    trial_key = (f"neural_trials_{symbol or 'global'}_{market_snapshot}_"
                 f"{MODEL_SCHEMA}_{FEATURE_HASH}_{ARCHITECTURE_HASH}")
    trials = int(store.kv_get(trial_key, 0) or 0)
    cap = int(cfg.get("neural", "max_trials_per_snapshot", default=6))
    if trials >= cap:
        return {"status": "caught_up", "reason": f"{trials}/{cap} trials used",
                "data_as_of": snapshot}
    torch.manual_seed(trials)
    trial_spec = dict(TRIAL_SPECS[trials % len(TRIAL_SPECS)])
    device = _training_device(cfg, torch)
    model = _make_model(len(FEATURES), len(ds["horizons"]))
    # Holding nets are complete trainable clones of the global champion.
    parent_row = parent = None
    if symbol:
        candidate_parent = store.db.execute(
                "SELECT * FROM model_runs WHERE kind='global_tcn' "
                "AND status IN ('champion','challenger') "
                "ORDER BY CASE status WHEN 'champion' THEN 0 ELSE 1 END,created_at DESC LIMIT 1"
            ).fetchone()
        if candidate_parent:
            parent, parent_model, reason = _load_checked(
                Path(candidate_parent["checkpoint"]), candidate_parent["checkpoint_sha256"])
            if parent is not None:
                parent_row = candidate_parent
                model.load_state_dict(parent_model.state_dict())
        if parent_row is None:
            # Holding networks are fine-tunes, never random standalone models.
            # Return the reserved trial because no training occurred.
            return {"status": "waiting", "kind": "holding_tcn", "symbol": symbol,
                    "reason": "need a compatible global TCN champion or challenger"}
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=trial_spec["lr"],
                            weight_decay=trial_spec["weight_decay"])
    X = torch.from_numpy(ds["X"])
    Yraw = torch.from_numpy(ds["Y"])
    target_scale = torch.from_numpy(ds["target_scale"])
    Y = Yraw / target_scale
    # Absolute-family supervision (added in B1B; excess path above unchanged).
    Yraw_abs = torch.from_numpy(ds["Y_absolute"])
    Y_abs = Yraw_abs / torch.from_numpy(ds["target_scale_absolute"])
    rtc = float(ds["round_trip_cost"])
    # Per-sample cost labels (R5): the absolute-edge label asks whether the
    # return beat THIS name's cost on THIS session, not a market-wide constant.
    cost_vector = torch.from_numpy(
        np.asarray(ds["sample_cost"], dtype=np.float32)).unsqueeze(1)
    tr = torch.from_numpy(np.flatnonzero(ds["masks"]["train"]))
    va = torch.from_numpy(np.flatnonzero(ds["masks"]["val"]))
    te = np.flatnonzero(ds["masks"]["test"])
    best, best_loss, stale = None, float("inf"), 0
    started = time.monotonic()
    max_epochs = int(cfg.get("neural", "max_epochs", default=50))
    sealed_epoch = str(market_snapshot)[:7]
    sealed_key = (f"neural_sealed_inspection_{sealed_epoch}_{MODEL_SCHEMA}_"
                  f"{FEATURE_HASH}_{ARCHITECTURE_HASH}")
    sealed_available = not bool(store.kv_get(sealed_key, False))
    final_trial = cap >= 5 and trials == cap - 1 and (bool(symbol) or sealed_available)
    # Preserve enough of the bounded final trial for the five chronological
    # folds; the ordinary validation challengers can use the whole budget.
    remaining_seconds = (max(0.0, overall_deadline - time.monotonic())
                         if overall_deadline is not None else None)
    if remaining_seconds is not None and remaining_seconds <= 0:
        return {"status": "yielded", "reason": "training deadline reached",
                "data_as_of": snapshot}
    training_seconds = (remaining_seconds * .4
                        if remaining_seconds is not None and final_trial
                        else remaining_seconds)
    for epoch in range(max_epochs):
        model.train()
        for idx in tr[torch.randperm(len(tr))].split(512):
            xb, yb = X[idx].to(device), Y[idx].to(device)
            yr = Yraw[idx].to(device)
            yb_abs, yr_abs = Y_abs[idx].to(device), Yraw_abs[idx].to(device)
            opt.zero_grad()
            out = model.forward_structured(xb)
            bce = torch.nn.functional.binary_cross_entropy
            # Multi-task: absolute + excess pinball, absolute-edge + excess-positive
            # BCE, and a date-grouped ranking loss on the EXCESS median only
            # (ranking is a cross-sectional-selection objective; absolute is not).
            loss = (_pinball(out.excess_quantiles, yb)
                    + _pinball(out.absolute_quantiles, yb_abs)
                    + .1 * bce(out.probability_excess_positive, (yr > 0).float())
                    + .1 * bce(out.probability_absolute_edge_positive,
                              (yr_abs > cost_vector[idx].to(device)).float()))
            rank = sum(_rank_loss(out.excess_quantiles[:, h, 1], yb[:, h],
                                  ds["dates"][idx.numpy()])
                       for h in range(len(ds["horizons"]))) / len(ds["horizons"])
            loss = loss + float(trial_spec["rank_weight"]) * rank
            loss.backward(); opt.step()
            if (cancelled is not None and cancelled()) or \
                    (training_seconds and time.monotonic() - started >= training_seconds):
                break
        model.eval()
        total_loss = total_rows = 0
        with torch.no_grad():
            for idx in va.split(2048):
                xb, yb, yr = X[idx].to(device), Y[idx].to(device), Yraw[idx].to(device)
                vb_abs, vr_abs = Y_abs[idx].to(device), Yraw_abs[idx].to(device)
                out = model.forward_structured(xb)
                bce = torch.nn.functional.binary_cross_entropy
                # R1 declared joint objective: the model is selected on BOTH
                # families it will be judged (and traded) on — never excess only.
                objective = (_pinball(out.excess_quantiles, yb)
                             + _pinball(out.absolute_quantiles, vb_abs)
                             + .1 * bce(out.probability_excess_positive, (yr > 0).float())
                             + .1 * bce(out.probability_absolute_edge_positive,
                                        (vr_abs > cost_vector[idx].to(device)).float()))
                total_loss += float(objective) * len(idx)
                total_rows += len(idx)
        val_loss = total_loss / max(1, total_rows)
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if progress:
            progress({"epoch": epoch + 1, "max_epochs": max_epochs,
                      "best_loss": round(best_loss, 6), "patience": stale,
                      "fraction": .2 + .8 * (epoch + 1) / max_epochs})
        if stale >= int(cfg.get("neural", "patience", default=5)) or \
                (training_seconds and time.monotonic() - started >= training_seconds):
            break
    model.load_state_dict(best or model.state_dict()); model.eval()
    tournament_winner_id = None
    winner_trial_spec = dict(trial_spec)
    if final_trial and not symbol:
        # Validation chooses the tournament winner. Only this cloned winner is
        # allowed to inspect the sealed test and walk-forward report folds.
        current_val_pred, _ = _predict_batches(
            model, X, va.numpy(), ds["target_scale"], device)
        current_val_metrics = _metrics(
            current_val_pred, ds["Y"][va.numpy()], ds["horizons"], ds["dates"][va.numpy()])
        winner_score = _selection_score(current_val_metrics)
        winner_loss = best_loss
        for row in store.db.execute(
                "SELECT * FROM model_runs WHERE kind='global_tcn' AND status='challenger' "
                "AND schema_version=? AND feature_hash=? AND architecture_hash=?",
                (MODEL_SCHEMA, FEATURE_HASH, ARCHITECTURE_HASH)).fetchall():
            prior_metrics = json.loads(row["metrics"] or "{}")
            if prior_metrics.get("tournament_key") != tournament_key or \
                    prior_metrics.get("evaluation_split") != "validation":
                continue
            score = prior_metrics.get("validation_selection_score")
            if score is None or float(score) <= winner_score:
                continue
            _, prior_model, _ = _load_checked(Path(row["checkpoint"]),
                                               row["checkpoint_sha256"])
            if prior_model is not None:
                model = prior_model.to(device)
                winner_score, tournament_winner_id = float(score), row["id"]
                winner_loss = float(prior_metrics.get("validation_objective", winner_loss))
                winner_trial_spec = dict(prior_metrics.get("trial_spec") or trial_spec)
        best_loss = winner_loss
    eval_idx = te if final_trial else va.numpy()
    # Fit the transform on validation only; the sealed block and live calls
    # receive the same frozen calibration without seeing their outcomes.
    validation_pred, validation_probability = _predict_batches(
        model, X, va.numpy(), ds["target_scale"], device)
    # Dual-family calibration, validation rows only. The flat `calibration`
    # stays the excess one for the legacy inference path; `calibration_structured`
    # carries both families for the B4C structured inference migration.
    val_abs_pred, val_abs_prob, _, _ = _predict_structured(
        model, X, va.numpy(), ds["target_scale"], ds["target_scale_absolute"], device)
    excess_cal = _calibration(validation_pred, validation_probability,
                              ds["Y_excess"][va.numpy()], prob_threshold=0.0)
    absolute_cal = _calibration(val_abs_pred, val_abs_prob,
                                ds["Y_absolute"][va.numpy()],
                                prob_threshold=ds["round_trip_cost"])
    calibration = excess_cal
    calibration_structured = {"excess": excess_cal, "absolute": absolute_cal}
    pred, probability = _predict_batches(
        model, X, eval_idx, ds["target_scale"], device)
    pred, probability = _apply_calibration(pred, probability, calibration)
    metric_dates = None if symbol else ds["dates"][eval_idx]
    metrics = _metrics(pred, ds["Y"][eval_idx], ds["horizons"], metric_dates)
    # R1: the ABSOLUTE head is what trades, so it is evaluated first-class on
    # the same rows, calibrated with its own validation-only transform.
    abs_pred, abs_prob, _, _ = _predict_structured(
        model, X, eval_idx, ds["target_scale"], ds["target_scale_absolute"], device)
    abs_pred, abs_prob = _apply_calibration(abs_pred, abs_prob, absolute_cal)
    metrics["absolute"] = _metrics(abs_pred, ds["Y_absolute"][eval_idx],
                                   ds["horizons"], metric_dates)
    metrics["absolute"]["evaluated_on"] = "structured_absolute_heads"
    metrics.update(evaluation_split="sealed_test" if final_trial else "validation",
                   validation_objective=round(best_loss, 6),
                   validation_selection_score=round(
                       winner_score if final_trial and not symbol else _selection_score(metrics), 6),
                   tournament_key=tournament_key,
                   data_fingerprint=data_fingerprint,
                   device=device, trial_spec=winner_trial_spec,
                   calibration_source="validation")
    if not symbol:
        metrics["baselines"] = _baseline_metrics(ds, eval_idx)
        metrics["beats_baselines"] = all(
            metrics["pinball"] < baseline.get("pinball", float("inf")) and
            _selection_score(metrics) > _selection_score(baseline)
            for baseline in metrics["baselines"].values())
    if final_trial:
        metrics["tournament_winner_source"] = tournament_winner_id or "final_trial"
    feature_std = (ds["X"][ds["masks"]["train"]] * ds["std"] + ds["mean"]).std(
        axis=(0, 1))
    active_features = [name for name, value in zip(FEATURES, feature_std) if value > 1e-8]
    metrics["feature_diagnostics"] = {
        "active": active_features,
        "inactive": [name for name in FEATURES if name not in active_features]}
    for i, h in enumerate(ds["horizons"]):
        metrics[str(h)]["probability_brier"] = round(float(np.mean(
            (probability[:, i] - (ds["Y"][eval_idx, i] > 0)) ** 2)), 5)
    # Only the final permitted trial pays for the five-fold tournament. Earlier
    # challengers remain cheap; none can promote without these real folds.
    oos_predictions = []
    if final_trial:
        remaining = (None if overall_deadline is None else
                     max(0.0, overall_deadline - time.monotonic()))
        walk_metrics, oos_predictions = _walk_forward_metrics(
            cfg, ds, trial_spec=winner_trial_spec, max_seconds=remaining)
        metrics.update(walk_metrics)
        # The validation-selected final model has never seen the sealed block.
        # Its per-symbol sealed predictions are therefore legitimate OOS graph
        # inputs.  Without them the graph replay starts after the TCN series
        # ends and the neural specialist has exactly zero sample coverage.
        for row_index, sample_index in enumerate(eval_idx):
            for horizon_index, horizon in enumerate(ds["horizons"]):
                oos_predictions.append((
                    str(ds["dates"][sample_index]), str(ds["owners"][sample_index]),
                    int(horizon), *map(float, pred[row_index, horizon_index]),
                    float(probability[row_index, horizon_index]),
                    float(ds["Y"][sample_index, horizon_index])))
    if symbol and parent_row and parent:
        try:
            baseline_model = _make_model(len(parent["features"]), len(parent["horizons"]))
            baseline_model.load_state_dict(parent["model"]); baseline_model.eval()
            raw = ds["X"] * ds["std"] + ds["mean"]
            bx = (raw - parent["mean"]) / parent["std"]
            with torch.no_grad():
                baseline_pred, baseline_probability = baseline_model.forward_all(
                    torch.from_numpy(bx[eval_idx].astype(np.float32)))
                baseline_pred = baseline_pred.numpy() * \
                    np.asarray(parent["target_scale"]).reshape(1, -1, 1)
                baseline_probability = baseline_probability.numpy()
            baseline_pred, _ = _apply_calibration(
                baseline_pred, baseline_probability, parent.get("calibration"))
            metrics["global_baseline"] = _metrics(
                baseline_pred, ds["Y"][eval_idx], ds["horizons"])
        except Exception:
            metrics["global_baseline"] = {"error": "incompatible parent checkpoint"}
    rid = hashlib.sha256(
        f"{tournament_key}|{symbol}|{snapshot}|{trials}|{MODEL_SCHEMA}|"
        f"{FEATURE_HASH}|{data_fingerprint}".encode()
    ).hexdigest()[:12]
    payload = {"schema_version": MODEL_SCHEMA, "architecture_hash": ARCHITECTURE_HASH,
               "feature_hash": FEATURE_HASH,
               "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "mean": ds["mean"], "std": ds["std"],
               "target_scale": ds["target_scale"],
               "target_scale_absolute": ds["target_scale_absolute"],
               "round_trip_cost": ds["round_trip_cost"],
               "target_schema_hash": ds["target_schema_hash"],
               "calibration": calibration,
               "calibration_structured": calibration_structured,
               "trial_spec": winner_trial_spec,
               "features": FEATURES, "horizons": ds["horizons"],
               "temporal_features": list(TEMPORAL_FEATURES),
               "context_features": list(CONTEXT_FEATURES),
               "temporal_hash": TEMPORAL_HASH, "context_hash": CONTEXT_HASH,
               "dataset_manifest_id": data_fingerprint, "code_commit": _code_commit(),
               "active_features": active_features,
               "metrics": metrics, "trained_at": _now(), "data_as_of": snapshot,
               "data_fingerprint": data_fingerprint,
               "parent_id": parent_row["id"] if parent_row else tournament_winner_id,
               "symbol": symbol, "window": int(cfg.get("neural", "input_sessions", default=60))}
    path = _path(cfg, symbol, challenger=True, run_id=rid); _save(path, payload)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    store.kv_set(trial_key, trials + 1)
    kind = "holding_tcn" if symbol else "global_tcn"
    store.db.execute(
        "INSERT OR REPLACE INTO model_runs(id,kind,symbol,created_at,data_as_of,status,"
        "parent_id,metrics,checkpoint,feature_hash,schema_version,architecture_hash,"
        "checkpoint_sha256,incompatibility_reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        rid, kind, symbol, _now(), snapshot, "challenger",
        parent_row["id"] if parent_row else tournament_winner_id, json.dumps(metrics), str(path),
        FEATURE_HASH, MODEL_SCHEMA, ARCHITECTURE_HASH, sha, None))
    store.db.commit()
    ml_lifecycle.transition(
        store, "model_runs", rid,
        "sealed_candidate" if metrics.get("evaluation_split") == "sealed_test"
        else "validation_candidate",
        reason=f"trained; evaluated on {metrics.get('evaluation_split', 'validation')}",
        evidence={"validation_selection_score": metrics.get("validation_selection_score"),
                  "evaluation_split": metrics.get("evaluation_split")},
        parent_id=parent_row["id"] if parent_row else tournament_winner_id)
    if oos_predictions:
        with store.db:
            store.db.executemany(
                "INSERT OR REPLACE INTO model_forecasts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                [(rid, as_of, owner, horizon, q10, q50, q90, probability,
                  "historical_oos", realized, FEATURE_HASH)
                 for as_of, owner, horizon, q10, q50, q90, probability, realized
                 in oos_predictions])
    if final_trial and not symbol:
        store.kv_set(sealed_key, {"model_id": rid, "inspected_at": _now(),
                                  "market_snapshot": market_snapshot})
    status = {"id": rid, "kind": kind, "symbol": symbol, "status": "challenger",
              "data_as_of": snapshot, "metrics": metrics, "trial": trials + 1,
              "at": _now()}
    if symbol and (not parent_row or parent_row["status"] != "champion"):
        status["provisional"] = True
    store.kv_set("neural_status" if not symbol else f"holding_model_{symbol}", status)
    store.audit("neural_challenger_trained", status)
    if symbol and parent_row and parent_row["status"] == "champion" and \
            holding_gate_passed(metrics):
        # Sprint D: validation-only evidence can NEVER auto-promote a holding
        # model. It stays validation_candidate until genuine out-of-sample
        # shadow history at both horizons exists (enforced inside promote()).
        status["validation_gate_passed"] = True
        store.audit("holding_promotion_blocked", {
            "id": rid, "symbol": symbol,
            "reason": "validation-only evidence; holding promotion requires "
                      "out-of-sample shadow observations at both horizons"})
    return status


def describe(cfg, store, symbol: str | None = None) -> dict:
    """Operator-facing truth; no checkpoint load and no invented metrics."""
    refresh_compatibility(store)
    global_run = store.db.execute(
        "SELECT * FROM model_runs WHERE kind='global_tcn' "
        "AND status IN ('champion','challenger') "
        "ORDER BY CASE status WHEN 'champion' THEN 0 ELSE 1 END,created_at DESC LIMIT 1"
    ).fetchone()
    holdings = []
    for position in store.open_positions(mode="live"):
        sym = position["symbol"]
        run = store.db.execute("SELECT * FROM model_runs WHERE kind='holding_tcn' "
                               "AND symbol=? ORDER BY created_at DESC LIMIT 1", (sym,)).fetchone()
        bars = store.db.execute("SELECT COUNT(*) n,MAX(d) latest FROM bars WHERE symbol=?",
                                (sym,)).fetchone()
        holdings.append({"symbol": sym, "bars": bars["n"], "latest_bar": bars["latest"],
                         "eligible": bars["n"] >= int(cfg.get(
                             "neural", "holding_min_bars", default=1250)),
                         "run": ({**dict(run), "metrics": json.loads(run["metrics"] or "{}")}
                                 if run else None),
                         "effective_blend": float(cfg.get("neural", "holding_blend", default=.25))
                         if run and run["status"] == "champion" else 0.0})
    forecast = None
    if symbol and global_run:
        rows = store.db.execute("SELECT * FROM model_forecasts WHERE model_id=? AND symbol=? "
                                "ORDER BY as_of DESC,horizon", (global_run["id"], symbol)).fetchall()
        forecast = [dict(r) for r in rows[:2]]
    return {"architecture": {"type": "causal_tcn_dual_branch", "window": int(cfg.get(
                "neural", "input_sessions", default=60)), "blocks": [
                {"channels": 32, "kernel": 3, "dilation": d, "activation": "GELU"}
                for d in (1, 2, 4, 8, 16)], "dropout": .1, "receptive_field": 63,
                "temporal_branch": f"{len(TEMPORAL_FEATURES)} sequence features → TCN",
                "context_branch": f"{len(CONTEXT_FEATURES)} point-in-time features "
                                  "(latest session) → Linear(16) + GELU",
                "target_scaling": "train-only per-horizon volatility",
                "return_families": list(ml_targets.RETURN_FAMILIES),
                "heads": ["absolute 5d/21d q10/q50/q90", "excess 5d/21d q10/q50/q90",
                          "absolute-edge 5d/21d probability",
                          "excess-positive 5d/21d probability"]},
            "temporal_features": list(TEMPORAL_FEATURES),
            "context_features": list(CONTEXT_FEATURES),
            "features": FEATURES, "global": ({**dict(global_run),
                "metrics": json.loads(global_run["metrics"] or "{}")} if global_run else None),
            "selected_forecast": forecast, "holdings": holdings}


def train_burst(cfg, store, max_seconds: int | None = None) -> dict:
    """Compatibility entry point: bounded challenger, never champion mutation."""
    return train_challenger(cfg, store, max_seconds=max_seconds)


def holding_gate_passed(metrics: dict) -> bool:
    baseline = metrics.get("global_baseline") or {}
    try:
        return all(
            metrics[str(h)]["pinball"] <= .95 * baseline[str(h)]["pinball"] and
            metrics[str(h)]["correlation"] > 0 and
            metrics[str(h)]["directional_accuracy"] > .52 and
            .75 <= metrics[str(h)]["coverage"] <= .85 for h in (5, 21))
    except (KeyError, TypeError, ZeroDivisionError):
        return False


def holding_forward_gate(store, run_id: str, min_observations: int = 200) -> bool:
    """Holding models may promote only on genuine out-of-sample evidence:
    resolved forward-shadow observations at BOTH horizons with positive IC."""
    sm = shadow_metrics(store, run_id)
    hs = sm["horizons"]
    return all(hs.get(str(h), {}).get("n", 0) >= min_observations and
               hs[str(h)].get("ic", 0) > 0 for h in (5, 21))


def promote(cfg, store, run_id: str, *, reason: str = "promotion gates passed",
            evidence: dict | None = None) -> None:
    """Atomic champion swap. Activation (checkpoint load) is verified BEFORE
    the transaction; the previous champion is retired in the same transaction
    that activates the new one, so a failure leaves it fully intact."""
    row = store.db.execute("SELECT * FROM model_runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        raise ValueError("unknown model run")
    if row["symbol"] and not holding_forward_gate(store, run_id):
        raise ValueError("holding model lacks out-of-sample shadow evidence "
                         "at both horizons; validation-only results cannot promote")
    payload, _, load_reason = _load_checked(Path(row["checkpoint"]), row["checkpoint_sha256"])
    if payload is None:
        raise ValueError(f"incompatible checkpoint: {load_reason}")
    symbol = row["symbol"]
    prior = store.db.execute(
        "SELECT id FROM model_runs WHERE kind=? AND COALESCE(symbol,'')=COALESCE(?,'') "
        "AND lifecycle_state='champion' AND id<>?",
        (row["kind"], symbol, run_id)).fetchall()
    with store.db:
        # Retire-then-crown inside ONE transaction: activation was validated
        # BEFORE the tx (checkpoint load), the unique champion index holds at
        # every statement, and any failure rolls the predecessor back intact.
        for old in prior:
            ml_lifecycle.transition(store, "model_runs", old["id"], "retired",
                                    reason=f"superseded by {run_id}", in_tx=True)
        ml_lifecycle.transition(store, "model_runs", run_id, "champion",
                                reason=reason, evidence=evidence, in_tx=True)
    store.audit("neural_champion_promoted", {"id": run_id, "symbol": symbol,
                                             "reason": reason,
                                             "retired": [o["id"] for o in prior]})


def _latest_window(cfg, store, ctx, symbol: str, payload: dict) -> np.ndarray | None:
    window = payload["window"]
    since = "1900-01-01"
    b = _bars(store, symbol, since)
    spy = _bars(store, cfg.get("universe", "benchmark", default="SPY"), since)
    vix = _bars(store, cfg.get("universe", "vix_symbol", default="^VIX"), since)
    vol_symbols = cfg.get("universe", "volatility_symbols", default={}) or {}
    context = {name: _bars(store, ticker, since) for name, ticker in vol_symbols.items()
               if name != "vix"}
    context.update({"hyg": _bars(store, "HYG", since), "tlt": _bars(store, "TLT", since)})
    sectors = {s: _bars(store, s, since) for s in cfg.get(
        "universe", "sector_etfs", default=[
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU",
            "XLB", "XLRE", "XLC"])}
    f = _features(b, spy, vix, store, symbol, sectors, context)[FEATURES].dropna()
    if len(f) < window:
        return None
    x = f.iloc[-window:].to_numpy(np.float32)
    return ((x - payload["mean"].reshape(1, -1)) /
            payload["std"].reshape(1, -1)).astype(np.float32)


def _load_checked(path: Path, expected_sha: str | None = None):
    import torch
    if not path.exists():
        return None, None, "checkpoint missing"
    try:
        actual_sha = _sha256_file(path)
        if expected_sha and actual_sha != expected_sha:
            return None, None, "checkpoint hash mismatch"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != MODEL_SCHEMA:
            return None, None, f"schema {payload.get('schema_version', 1)} != {MODEL_SCHEMA}"
        if payload.get("feature_hash") != FEATURE_HASH or payload.get("features") != FEATURES:
            return None, None, "feature schema mismatch"
        if payload.get("architecture_hash") != ARCHITECTURE_HASH:
            return None, None, "architecture mismatch"
        if payload.get("target_schema_hash") != ml_targets.TARGET_SCHEMA_HASH:
            return None, None, "target schema mismatch"
        if "target_scale" not in payload or len(payload["target_scale"]) != len(payload["horizons"]):
            return None, None, "target scaling metadata missing"
        if "target_scale_absolute" not in payload:
            return None, None, "absolute target scaling metadata missing"
        model = _make_model(len(payload["features"]), len(payload["horizons"]))
        model.load_state_dict(payload["model"]); model.eval()
        return payload, model, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {str(exc)[:120]}"


def _load(path: Path):
    payload, model, _ = _load_checked(path)
    return payload, model


def refresh_compatibility(store) -> dict:
    """Persist cheap checkpoint truth without importing/deserializing Torch.

    Full payload validation still happens at the inference/promotion boundary.
    Polling the Model view only needs to prove that the immutable file hash and
    the metadata recorded when that file was created still match the running
    schema.  This keeps a dashboard GET from loading every model into the web
    process.
    """
    out = {"compatible": 0, "incompatible": 0}
    for row in store.db.execute("SELECT * FROM model_runs WHERE status!='retired'").fetchall():
        path = Path(row["checkpoint"] or "")
        reason = None
        if not row["checkpoint"] or not path.exists():
            reason = "checkpoint missing"
        elif not row["checkpoint_sha256"]:
            reason = "checkpoint hash missing"
        elif _sha256_file(path) != row["checkpoint_sha256"]:
            reason = "checkpoint hash mismatch"
        elif int(row["schema_version"] or 0) != MODEL_SCHEMA:
            reason = f"schema {row['schema_version'] or 0} != {MODEL_SCHEMA}"
        elif row["feature_hash"] != FEATURE_HASH:
            reason = "feature schema mismatch"
        elif row["architecture_hash"] != ARCHITECTURE_HASH:
            reason = "architecture mismatch"
        incompatible = reason is not None
        status = "incompatible" if incompatible else row["status"]
        keys = row.keys()
        lifecycle = ("incompatible" if incompatible else
                     (row["lifecycle_state"] if "lifecycle_state" in keys else None))
        store.db.execute(
            "UPDATE model_runs SET status=?,incompatibility_reason=?,"
            "lifecycle_state=COALESCE(?,lifecycle_state) WHERE id=?",
            (status, reason, lifecycle, row["id"]))
        out["incompatible" if incompatible else "compatible"] += 1
    store.db.commit()
    return out


def build_neural_forecast(*, symbol, as_of, horizon, i, abs_q, abs_p, exc_q,
                          exc_p, meta) -> "NeuralForecast":
    """The ONE tensor-index → typed-forecast mapping. Every inference path (live,
    shadow, tests) builds forecasts here so the absolute/excess column mapping
    exists in exactly one place instead of being re-derived per consumer."""
    from .ml import NeuralForecast
    return NeuralForecast(
        symbol=symbol, as_of=as_of, horizon_sessions=int(horizon),
        absolute_q10=float(abs_q[i, 0]), absolute_q50=float(abs_q[i, 1]),
        absolute_q90=float(abs_q[i, 2]),
        excess_q10=float(exc_q[i, 0]), excess_q50=float(exc_q[i, 1]),
        excess_q90=float(exc_q[i, 2]),
        probability_absolute_edge_positive=float(abs_p[i]),
        probability_excess_positive=float(exc_p[i]),
        model_id=str(meta["model_id"]),
        dataset_manifest_id=str(meta.get("dataset_manifest_id") or meta["model_id"]),
        feature_schema_hash=str(meta.get("feature_hash") or FEATURE_HASH))


def _structured_calibrated(model, x, payload, torch):
    """(abs_q[H,3], abs_p[H], exc_q[H,3], exc_p[H]) — scaled by each family's
    train-only volatility and calibrated with the frozen validation transform."""
    with torch.no_grad():
        out = model.forward_structured(torch.from_numpy(x[None, ...]))
        abs_q = out.absolute_quantiles.numpy()[0] * np.asarray(
            payload["target_scale_absolute"]).reshape(-1, 1)
        abs_p = out.probability_absolute_edge_positive.numpy()[0]
        exc_q = out.excess_quantiles.numpy()[0] * np.asarray(
            payload["target_scale"]).reshape(-1, 1)
        exc_p = out.probability_excess_positive.numpy()[0]
    cal = payload.get("calibration_structured") or {}
    abs_q, abs_p = _apply_calibration(abs_q[None], abs_p[None], cal.get("absolute"))
    exc_q, exc_p = _apply_calibration(exc_q[None], exc_p[None], cal.get("excess"))
    return abs_q[0], abs_p[0], exc_q[0], exc_p[0]


def _forecast_meta(row, payload) -> dict:
    keys = row.keys()
    state = row["lifecycle_state"] if "lifecycle_state" in keys else None
    # A full champion serves under the config's whole blend range; ramp states
    # are capped by the blend persisted at their lifecycle transition.
    permitted = (None if state == "champion" else
                 float(row["permitted_blend"] or 0.0)
                 if "permitted_blend" in keys else None)
    return {"model_id": row["id"], "feature_hash": payload.get("feature_hash"),
            "target_schema_hash": payload.get("target_schema_hash"),
            "dataset_manifest_id": payload.get("dataset_manifest_id"),
            "lifecycle_state": state, "permitted_blend": permitted}


def active_global_run(store):
    """The global model currently allowed to serve live inference, or None.
    Serving priority (Sprint D): champion > production_candidate >
    experimental_live. Validation-only rows never serve."""
    return store.db.execute(
        "SELECT * FROM model_runs WHERE kind='global_tcn' AND symbol IS NULL "
        "AND incompatibility_reason IS NULL AND lifecycle_state IN "
        "('champion','production_candidate','experimental_live') "
        "ORDER BY CASE lifecycle_state WHEN 'champion' THEN 0 "
        "WHEN 'production_candidate' THEN 1 ELSE 2 END, created_at DESC LIMIT 1"
    ).fetchone()


def replay_forecasts(cfg, store, as_of: str) -> tuple[dict[str, dict], dict]:
    """Offline (backtest) replay of IMMUTABLE dual-family forecasts (R3).

    Serves the same {symbol: {horizon: NeuralForecast}} contract as
    predict_today, but from persisted model_forecasts_v2 rows for the serving
    model at exactly this decision date — no torch, no lookahead: each row was
    produced from data <= as_of when it was recorded. Rows whose provenance
    fails the fail-closed policy validation downstream are simply inert.
    """
    from .ml import NeuralForecast
    row = active_global_run(store)
    if not row:
        return {}, {"silent": "no serving model to replay"}
    out: dict[str, dict] = {}
    for r in store.db.execute(
            "SELECT * FROM model_forecasts_v2 WHERE model_id=? AND as_of=?",
            (row["id"], as_of)).fetchall():
        try:
            nf = NeuralForecast(
                symbol=r["symbol"], as_of=r["as_of"],
                horizon_sessions=int(r["horizon"]),
                absolute_q10=r["absolute_q10"], absolute_q50=r["absolute_q50"],
                absolute_q90=r["absolute_q90"],
                excess_q10=r["excess_q10"], excess_q50=r["excess_q50"],
                excess_q90=r["excess_q90"],
                probability_absolute_edge_positive=r["probability_absolute_edge_positive"],
                probability_excess_positive=r["probability_excess_positive"],
                model_id=r["model_id"],
                dataset_manifest_id=r["dataset_manifest_id"] or r["model_id"],
                feature_schema_hash=r["feature_hash"])
        except ValueError:
            continue                       # malformed persisted row: inert
        out.setdefault(r["symbol"], {})[str(r["horizon"])] = nf
    payload_stub = {"feature_hash": row["feature_hash"],
                    "target_schema_hash": None, "dataset_manifest_id": None}
    meta = {**_forecast_meta(row, payload_stub), "replayed": True, "as_of": as_of}
    return out, (meta if out else {**meta, "silent": f"no v2 forecasts at {as_of}"})


def predict_today(cfg, store, ctx) -> tuple[dict[str, dict], dict]:
    try:
        import torch
    except ImportError:
        return {}, {"silent": "torch not installed"}
    # An experimental_live model serves at its BOUNDED permitted blend (the
    # influence ramp); a full champion serves under the whole config range.
    row = active_global_run(store)
    if not row:
        return {}, {"silent": "no validated global TCN champion"}
    payload, model, reason = _load_checked(Path(row["checkpoint"]), row["checkpoint_sha256"])
    if payload is None:
        return {}, {"silent": f"global champion unavailable: {reason}"}
    age = (datetime.now().astimezone() - datetime.fromisoformat(payload["trained_at"])).days
    if age > int(cfg.get("neural", "max_checkpoint_age_days", default=7)):
        return {}, {"silent": f"global champion stale ({age}d)"}
    meta = {**_forecast_meta(row, payload), "checkpoint_age_days": age}
    out = {}
    for sym in ctx.universe:
        x = _latest_window(cfg, store, ctx, sym, payload)
        if x is None:
            continue
        abs_q, abs_p, exc_q, exc_p = _structured_calibrated(model, x, payload, torch)
        # A validated holding champion blends within this node only (both families).
        holding_row = store.db.execute(
            "SELECT * FROM model_runs WHERE kind='holding_tcn' AND symbol=? "
            "AND status='champion' ORDER BY created_at DESC LIMIT 1", (sym,)).fetchone()
        if holding_row:
            hp, hm, _ = _load_checked(Path(holding_row["checkpoint"]),
                                      holding_row["checkpoint_sha256"])
            hx = _latest_window(cfg, store, ctx, sym, hp) if hp is not None else None
            if hx is not None:
                ha_q, ha_p, he_q, he_p = _structured_calibrated(hm, hx, hp, torch)
                w = float(cfg.get("neural", "holding_blend", default=0.25))
                abs_q, abs_p = (1 - w) * abs_q + w * ha_q, (1 - w) * abs_p + w * ha_p
                exc_q, exc_p = (1 - w) * exc_q + w * he_q, (1 - w) * exc_p + w * he_p
        out[sym] = {str(h): build_neural_forecast(
                        symbol=sym, as_of=ctx.as_of, horizon=h, i=i, abs_q=abs_q,
                        abs_p=abs_p, exc_q=exc_q, exc_p=exc_p, meta=meta)
                    for i, h in enumerate(payload["horizons"])}
    return out, {**meta, "metrics": json.loads(row["metrics"] or "{}")}


def predict_run(cfg, store, ctx, run_id: str) -> tuple[dict[str, dict], dict]:
    """Shadow inference for an immutable challenger checkpoint."""
    try:
        import torch
    except ImportError:
        return {}, {"silent": "torch not installed"}
    row = store.db.execute("SELECT * FROM model_runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return {}, {"silent": "unknown model run"}
    payload, model, reason = _load_checked(
        Path(row["checkpoint"]), row["checkpoint_sha256"])
    if payload is None:
        return {}, {"silent": f"challenger checkpoint unavailable: {reason}"}
    meta = _forecast_meta(row, payload)
    out = {}
    for sym in ctx.universe:
        x = _latest_window(cfg, store, ctx, sym, payload)
        if x is None:
            continue
        abs_q, abs_p, exc_q, exc_p = _structured_calibrated(model, x, payload, torch)
        out[sym] = {str(h): build_neural_forecast(
                        symbol=sym, as_of=ctx.as_of, horizon=h, i=i, abs_q=abs_q,
                        abs_p=abs_p, exc_q=exc_q, exc_p=exc_p, meta=meta)
                    for i, h in enumerate(payload["horizons"])}
    return out, {**meta, "metrics": json.loads(row["metrics"] or "{}")}


def shadow_metrics(store, model_id: str) -> dict:
    rows = store.db.execute("SELECT * FROM model_forecasts WHERE model_id=? "
                            "AND resolved_at IS NOT NULL "
                            "AND resolved_at!='historical_oos'", (model_id,)).fetchall()
    out = {"sessions": len({r["as_of"] for r in rows}), "total": len(rows), "horizons": {}}
    for h in (5, 21):
        hs = [r for r in rows if r["horizon"] == h]
        pred = np.asarray([r["q50"] for r in hs]); actual = np.asarray([r["realized_excess"] for r in hs])
        if not hs:
            out["horizons"][str(h)] = {"n": 0}; continue
        dates = np.asarray([r["as_of"] for r in hs])
        coverage = np.mean([(r["q10"] <= r["realized_excess"] <= r["q90"]) for r in hs])
        out["horizons"][str(h)] = {"n": len(hs),
                                     "ic": round(_rank_ic(pred, actual, dates), 4),
                                     "top_decile_alpha": round(
                                         _top_decile_alpha(pred, actual, dates), 5),
                                     "coverage": round(float(coverage), 3)}
    # Absolute-family quality, reported separately from the excess `horizons`
    # above (which keeps the existing promotion-gate contract).
    v2 = store.db.execute("SELECT * FROM model_forecasts_v2 WHERE model_id=? "
                          "AND resolved_at IS NOT NULL", (model_id,)).fetchall()
    out["families"] = ["excess", "absolute"]
    out["absolute"] = {}
    for h in (5, 21):
        hh = [r for r in v2 if r["horizon"] == h]
        if not hh:
            out["absolute"][str(h)] = {"n": 0}; continue
        ap = np.asarray([r["absolute_q50"] for r in hh])
        aa = np.asarray([r["realized_absolute"] for r in hh])
        ad = np.asarray([r["as_of"] for r in hh])
        acov = np.mean([(r["absolute_q10"] <= r["realized_absolute"] <= r["absolute_q90"])
                        for r in hh])
        out["absolute"][str(h)] = {"n": len(hh),
                                   "ic": round(_rank_ic(ap, aa, ad), 4),
                                   "coverage": round(float(acov), 3)}
    return out


def _offline_gate(metrics: dict) -> bool:
    """Stage-one gate shared by promotion and champion integrity checks."""
    folds = metrics.get("folds") or []
    fold_gate = len(folds) >= 5 and all(
        sum(f.get(f"ic_{h}d", 0) > 0 for f in folds) >= 4 and
        min(f.get(f"ic_{h}d", -1) for f in folds) > -.01 and
        metrics.get(f"median_fold_ic_{h}d", 0) >= (.015 if h == 5 else .02) and
        sum(f.get(f"net_alpha_{h}d", -1) > 0 for f in folds) >= 4
        for h in (5, 21))
    excess_gate = all(
        float(metrics.get(str(h), {}).get("correlation", 0)) > 0 and
        float(metrics.get(str(h), {}).get("top_decile_alpha_after_cost", 0)) > 0 and
        .75 <= float(metrics.get(str(h), {}).get("coverage", 0)) <= .85
        for h in (5, 21))
    # R1: the absolute head is what trades — a model whose absolute after-cost
    # selection is not positive can NEVER pass, whatever its excess rank IC.
    # Missing absolute metrics (legacy rows) fail closed.
    absolute = metrics.get("absolute") or {}
    absolute_gate = bool(absolute) and all(
        float(absolute.get(str(h), {}).get("correlation", 0)) > 0 and
        float(absolute.get(str(h), {}).get("top_decile_alpha_after_cost", 0)) > 0
        for h in (5, 21))
    return (bool(metrics.get("beats_baselines")) and fold_gate
            and excess_gate and absolute_gate)


def maybe_promote(cfg, store) -> dict:
    """Objective-gated global promotion/rollback; never relaxes the governor."""
    refresh_compatibility(store)
    champion_row = store.db.execute("SELECT * FROM model_runs WHERE kind='global_tcn' "
                                    "AND status='champion' ORDER BY created_at DESC LIMIT 1").fetchone()
    if champion_row:
        champion_metrics = json.loads(champion_row["metrics"] or "{}")
        sm = shadow_metrics(store, champion_row["id"])
        hs = sm["horizons"]
        stale = (datetime.now().date() - datetime.fromisoformat(champion_row["created_at"]).date()).days > \
            int(cfg.get("neural", "max_checkpoint_age_days", default=7))
        decay = any(hs.get(str(h), {}).get("n", 0) >= 1000 and
                    (hs[str(h)].get("ic", 0) <= 0 or hs[str(h)].get("top_decile_alpha", 0) <= 0)
                    for h in (5, 21))
        integrity_failed = not _offline_gate(champion_metrics)
        if stale or decay or integrity_failed:
            retired_id = champion_row["id"]
            rollback_reason = ("stale" if stale else
                               "forward decay" if decay else "offline gate failed")
            with store.db:
                ml_lifecycle.transition(store, "model_runs", retired_id, "retired",
                                        reason=f"champion rollback: {rollback_reason}",
                                        evidence={"shadow": hs}, in_tx=True)
                predecessor = None
                for candidate in store.db.execute(
                        "SELECT * FROM model_runs WHERE kind='global_tcn' AND status='retired' "
                        "AND id<>? AND incompatibility_reason IS NULL ORDER BY created_at DESC",
                        (retired_id,)).fetchall():
                    candidate_metrics = json.loads(candidate["metrics"] or "{}")
                    candidate_age = (datetime.now().date() -
                                     datetime.fromisoformat(candidate["created_at"]).date()).days
                    if _offline_gate(candidate_metrics) and candidate_age <= int(
                            cfg.get("neural", "max_checkpoint_age_days", default=7)):
                        predecessor = candidate
                        break
                restored_graph = None
                if predecessor:
                    ml_lifecycle.transition(store, "model_runs", predecessor["id"],
                                            "champion",
                                            reason=f"restored after rollback of {retired_id}",
                                            in_tx=True)
                    for graph_row in store.db.execute(
                            "SELECT * FROM graph_versions WHERE status='retired' "
                            "ORDER BY created_at DESC").fetchall():
                        graph_metrics = json.loads(graph_row["metrics"] or "{}")
                        if graph_metrics.get("temporal_model_id") == predecessor["id"]:
                            for old_graph in store.db.execute(
                                    "SELECT id FROM graph_versions WHERE "
                                    "lifecycle_state='champion'").fetchall():
                                ml_lifecycle.transition(
                                    store, "graph_versions", old_graph["id"], "retired",
                                    reason="TCN dependency rolled back", in_tx=True)
                            ml_lifecycle.transition(
                                store, "graph_versions", graph_row["id"], "champion",
                                reason=f"restored with TCN {predecessor['id']}", in_tx=True)
                            restored_graph = graph_row["id"]
                            break
            store.audit("neural_champion_rolled_back", {"id": retired_id,
                                                         "stale": stale,
                                                         "offline_gate_failed": integrity_failed,
                                                         "restored": predecessor["id"] if predecessor else None,
                                                         "restored_graph": restored_graph,
                                                         "metrics": sm})
            return {"action": "rollback_restore" if predecessor else "rollback",
                    "id": retired_id,
                    "restored": predecessor["id"] if predecessor else None,
                    "restored_graph": restored_graph,
                    "offline_gate_failed": integrity_failed, "metrics": sm}
    # ---- finalist evaluation (Sprint D): ALL eligible models compete, ranked
    # deterministically on persisted metrics — a newer weak model can never
    # conceal an older qualified finalist. Validation-only rows are not
    # finalists; incompatible/missing-checkpoint rows are filtered upstream.
    rows = [r for r in ml_lifecycle.finalists(store, "model_runs", kind="global_tcn")
            if not r["symbol"]]
    eligible = [r for r in rows if _offline_gate(json.loads(r["metrics"] or "{}"))]
    if not eligible:
        return ({"action": "none"} if not rows else
                {"action": "shadow", "candidates": [r["id"] for r in rows]})
    eligible.sort(key=ml_lifecycle.rank_key)
    top = eligible[0]
    state = top["lifecycle_state"]
    blend = float(cfg.get("neural", "experimental_blend", default=0.15))
    if state == "sealed_candidate":
        # Offline validity earns a BOUNDED experimental-live blend — never a
        # silent full championship (the old promote_stage1 hole is closed).
        ml_lifecycle.transition(
            store, "model_runs", top["id"], "experimental_live",
            reason="offline gates passed on sealed evaluation",
            evidence={"rank": 1, "eligible": len(eligible)},
            permitted_blend=blend)
        store.kv_set("tcn_offline_gate", {"passed": True, "id": top["id"], "at": _now()})
        return {"action": "experimental_live", "id": top["id"],
                "permitted_blend": blend}
    sm = shadow_metrics(store, top["id"]); hs = sm["horizons"]
    ab = sm.get("absolute") or {}
    forward = sm["sessions"] >= 30 and all(
        hs.get(str(h), {}).get("n", 0) >= 10_000 and
        hs[str(h)].get("ic", 0) >= .01 and hs[str(h)].get("top_decile_alpha", 0) > 0 and
        .75 <= hs[str(h)].get("coverage", 0) <= .85 for h in (5, 21)) and all(
        # R1: resolved DUAL-FAMILY (v2) evidence — the absolute head must show
        # positive forward rank quality on its own outcomes before any ramp-up.
        ab.get(str(h), {}).get("n", 0) >= 200 and
        ab[str(h)].get("ic", 0) > 0 for h in (5, 21))
    if not forward:
        return {"action": "shadow", "id": top["id"], "metrics": sm}
    if state == "experimental_live":
        ml_lifecycle.transition(
            store, "model_runs", top["id"], "production_candidate",
            reason="forward shadow gates passed",
            evidence={"sessions": sm["sessions"], "horizons": hs},
            permitted_blend=blend)
        return {"action": "production_candidate", "id": top["id"], "metrics": sm}
    promote(cfg, store, top["id"], reason="forward gates re-confirmed as "
            "production_candidate", evidence={"sessions": sm["sessions"]})
    return {"action": "promote", "id": top["id"], "metrics": sm}
