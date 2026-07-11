"""Causal temporal neural node with immutable global/holding champions.

The model predicts 5d/21d excess-return quantiles from 60-session multivariate
windows. Research always writes a challenger first; live inference reads only
a model_runs row explicitly marked champion. Repeating the same snapshot is
bounded by a persisted trial counter and can never mutate that champion.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["r1", "range", "gap", "volume_z", "vol21", "rsi14",
            "sma50_d", "sma200_d", "spy_r1", "spy_r21", "vix",
            "valuation", "valuation_missing"]
QUANTILES = (0.1, 0.5, 0.9)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _path(cfg, symbol: str | None = None, challenger: bool = False) -> Path:
    if symbol:
        root = Path(cfg.get("neural", "holdings_dir", default="data/models/holdings"))
        suffix = ".challenger.pt" if challenger else ".pt"
        return root / f"{symbol}{suffix}"
    p = Path(cfg.get("neural", "checkpoint", default="data/models/global_tcn.pt"))
    return p.with_suffix(".challenger.pt") if challenger else p


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


def _features(b: pd.DataFrame, spy: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    c = b["close"].astype(float)
    r = c.pct_change()
    f = pd.DataFrame(index=b.index)
    f["r1"] = r
    f["range"] = (b["high"] - b["low"]) / c.replace(0, np.nan)
    f["gap"] = b["open"] / c.shift(1) - 1
    lv = np.log1p(b["volume"].astype(float))
    f["volume_z"] = (lv - lv.rolling(21).mean()) / (lv.rolling(21).std() + 1e-8)
    f["vol21"] = r.rolling(21).std()
    gain = r.clip(lower=0).rolling(14).mean()
    loss = (-r.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = gain / (gain + loss + 1e-8) - 0.5
    f["sma50_d"] = c / c.rolling(50).mean() - 1
    f["sma200_d"] = c / c.rolling(200).mean() - 1
    sp = spy["close"].reindex(f.index).ffill().astype(float)
    f["spy_r1"] = sp.pct_change()
    f["spy_r21"] = sp.pct_change(21)
    if len(vix):
        f["vix"] = vix["close"].reindex(f.index).ffill() / 20.0 - 1
    else:
        f["vix"] = 0.0
    # Point-in-time valuation arrives through SEC research later. Missing is
    # explicit; current Yahoo ratios are never leaked backward.
    f["valuation"] = 0.0
    f["valuation_missing"] = 1.0
    return f.replace([np.inf, -np.inf], np.nan)


def build_dataset(cfg, store, symbols: list[str] | None = None) -> dict:
    window = int(cfg.get("neural", "input_sessions", default=60))
    horizons = tuple(cfg.get("neural", "horizons", default=[5, 21]))
    since = cfg.get("neural", "train_since", default="2011-01-01")
    symbols = symbols or [s for s in cfg.get("universe", "symbols", default=[])
                          if not s.startswith("^")]
    bench = cfg.get("universe", "benchmark", default="SPY")
    spy, vix = _bars(store, bench, since), _bars(
        store, cfg.get("universe", "vix_symbol", default="^VIX"), since)
    if len(spy) < window + max(horizons) + 100:
        return {"error": "not enough benchmark history"}
    X, Y, dates, owners = [], [], [], []
    for sym in symbols:
        b = _bars(store, sym, since)
        if len(b) < window + max(horizons) + 100:
            continue
        f = _features(b, spy, vix)
        c = b["close"].astype(float)
        sp = spy["close"].reindex(c.index).ffill().astype(float)
        targets = pd.DataFrame({h: (c.shift(-h) / c - 1) -
                                    (sp.shift(-h) / sp - 1) for h in horizons})
        vals = f[FEATURES].to_numpy(np.float32)
        for i in range(window - 1, len(f) - max(horizons)):
            x = vals[i - window + 1:i + 1]
            y = targets.iloc[i].to_numpy(np.float32)
            if np.isfinite(x).all() and np.isfinite(y).all():
                X.append(x); Y.append(y); dates.append(f.index[i]); owners.append(sym)
    if len(X) < 100:
        return {"error": f"not enough training windows ({len(X)})"}
    X, Y = np.stack(X), np.stack(Y)
    unique = sorted(set(dates))
    if len(unique) < 180:
        return {"error": f"not enough distinct dates ({len(unique)})"}
    # Chronological train/validation/test with horizon embargoes.
    test_start = unique[int(len(unique) * 0.85)]
    val_start = unique[int(len(unique) * 0.70)]
    embargo = max(horizons)
    val_pos, test_pos = unique.index(val_start), unique.index(test_start)
    train_end = unique[max(0, val_pos - embargo)]
    val_end = unique[max(val_pos, test_pos - embargo)]
    d = np.asarray(dates)
    masks = {"train": d <= train_end,
             "val": (d >= val_start) & (d <= val_end),
             "test": d >= test_start}
    mean = X[masks["train"]].mean((0, 1), keepdims=True)
    std = X[masks["train"]].std((0, 1), keepdims=True) + 1e-6
    X = (X - mean) / std
    return {"X": X.astype(np.float32), "Y": Y, "dates": d,
            "owners": np.asarray(owners), "masks": masks,
            "mean": mean, "std": std, "horizons": horizons,
            "data_as_of": unique[-1], "train_end": train_end,
            "val_start": val_start, "test_start": test_start}


def _make_model(n_features: int, n_horizons: int):
    import torch
    import torch.nn as nn

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
            self.blocks = nn.Sequential(CausalBlock(n_features, 32, 1),
                                        CausalBlock(32, 32, 2),
                                        CausalBlock(32, 32, 4))
            self.head = nn.Linear(32, n_horizons * 3)

        def forward(self, x):
            z = self.blocks(x.transpose(1, 2))[..., -1]
            raw = self.head(z).view(-1, n_horizons, 3)
            q50 = raw[..., 1]
            q10 = q50 - torch.nn.functional.softplus(raw[..., 0])
            q90 = q50 + torch.nn.functional.softplus(raw[..., 2])
            return torch.stack((q10, q50, q90), dim=-1)
    return TCN()


def _pinball(pred, target):
    import torch
    q = torch.tensor(QUANTILES, device=pred.device).view(1, 1, 3)
    err = target.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1) * err).mean()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 5 and np.std(a) and np.std(b) else 0.0


def _metrics(pred: np.ndarray, y: np.ndarray, horizons) -> dict:
    out = {}
    losses = []
    for i, h in enumerate(horizons):
        q10, q50, q90 = pred[:, i, 0], pred[:, i, 1], pred[:, i, 2]
        loss = float(np.mean(np.maximum(
            np.asarray(QUANTILES) * (y[:, i, None] - pred[:, i, :]),
            (np.asarray(QUANTILES) - 1) * (y[:, i, None] - pred[:, i, :]))))
        losses.append(loss)
        out[str(h)] = {"pinball": round(loss, 6),
                       "correlation": round(_corr(q50, y[:, i]), 4),
                       "directional_accuracy": round(float((np.sign(q50) == np.sign(y[:, i])).mean()), 3),
                       "coverage": round(float(((y[:, i] >= q10) & (y[:, i] <= q90)).mean()), 3)}
    out["pinball"] = round(float(np.mean(losses)), 6)
    return out


def train_challenger(cfg, store, symbols: list[str] | None = None,
                     symbol: str | None = None, max_seconds: int | None = None) -> dict:
    try:
        import torch
    except ImportError:
        return {"error": "torch not installed — install .[neural]"}
    torch.set_num_threads(int(cfg.get("research", "max_cpu_threads", default=4)))
    if symbol:
        observed = store.db.execute("SELECT COUNT(*) n FROM bars WHERE symbol=?",
                                    (symbol,)).fetchone()["n"]
        required = int(cfg.get("neural", "holding_min_bars", default=1250))
        if observed < required:
            return {"status": "waiting", "kind": "holding_tcn", "symbol": symbol,
                    "reason": f"need {required} settled observations; have {observed}"}
    ds = build_dataset(cfg, store, symbols=[symbol] if symbol else symbols)
    if "error" in ds:
        return ds
    snapshot = ds["data_as_of"]
    trial_key = f"neural_trials_{symbol or 'global'}_{snapshot}"
    trials = int(store.kv_get(trial_key, 0) or 0)
    cap = int(cfg.get("neural", "max_trials_per_snapshot", default=6))
    if trials >= cap:
        return {"status": "caught_up", "reason": f"{trials}/{cap} trials used",
                "data_as_of": snapshot}
    store.kv_set(trial_key, trials + 1)
    torch.manual_seed(trials)
    model = _make_model(len(FEATURES), len(ds["horizons"]))
    # Holding nets are complete trainable clones of the global champion.
    if symbol and _path(cfg).exists():
        try:
            parent = torch.load(_path(cfg), map_location="cpu", weights_only=False)
            model.load_state_dict(parent["model"])
        except Exception:
            pass
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.from_numpy(ds["X"]); Y = torch.from_numpy(ds["Y"])
    tr = torch.from_numpy(np.flatnonzero(ds["masks"]["train"]))
    va = torch.from_numpy(np.flatnonzero(ds["masks"]["val"]))
    te = np.flatnonzero(ds["masks"]["test"])
    best, best_loss, stale = None, float("inf"), 0
    started = time.time()
    for epoch in range(int(cfg.get("neural", "max_epochs", default=50))):
        model.train()
        for idx in tr[torch.randperm(len(tr))].split(512):
            opt.zero_grad(); loss = _pinball(model(X[idx]), Y[idx]); loss.backward(); opt.step()
            if max_seconds and time.time() - started >= max_seconds:
                break
        model.eval()
        with torch.no_grad():
            val_loss = float(_pinball(model(X[va]), Y[va]))
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(cfg.get("neural", "patience", default=5)) or \
                (max_seconds and time.time() - started >= max_seconds):
            break
    model.load_state_dict(best or model.state_dict()); model.eval()
    with torch.no_grad():
        pred = model(X[te]).numpy()
    metrics = _metrics(pred, ds["Y"][te], ds["horizons"])
    payload = {"model": model.state_dict(), "mean": ds["mean"], "std": ds["std"],
               "features": FEATURES, "horizons": ds["horizons"],
               "metrics": metrics, "trained_at": _now(), "data_as_of": snapshot,
               "symbol": symbol, "window": int(cfg.get("neural", "input_sessions", default=60))}
    path = _path(cfg, symbol, challenger=True); _save(path, payload)
    rid = hashlib.sha256(f"{symbol}|{snapshot}|{trials}|{metrics}".encode()).hexdigest()[:12]
    kind = "holding_tcn" if symbol else "global_tcn"
    store.db.execute("INSERT OR REPLACE INTO model_runs VALUES(?,?,?,?,?,?,?,?,?,?)", (
        rid, kind, symbol, _now(), snapshot, "challenger", None, json.dumps(metrics),
        str(path), hashlib.sha256("|".join(FEATURES).encode()).hexdigest()[:16]))
    store.db.commit()
    status = {"id": rid, "kind": kind, "symbol": symbol, "status": "challenger",
              "data_as_of": snapshot, "metrics": metrics, "trial": trials + 1,
              "at": _now()}
    store.kv_set("neural_status" if not symbol else f"holding_model_{symbol}", status)
    store.audit("neural_challenger_trained", status)
    return status


def train_burst(cfg, store, max_seconds: int | None = None) -> dict:
    """Compatibility entry point: bounded challenger, never champion mutation."""
    return train_challenger(cfg, store, max_seconds=max_seconds)


def promote(cfg, store, run_id: str) -> None:
    import torch
    row = store.db.execute("SELECT * FROM model_runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        raise ValueError("unknown model run")
    symbol = row["symbol"]
    source, dest = Path(row["checkpoint"]), _path(cfg, symbol)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _save(dest, payload)
    with store.db:
        store.db.execute("UPDATE model_runs SET status='retired' WHERE kind=? AND "
                         "COALESCE(symbol,'')=COALESCE(?,'') AND status='champion'",
                         (row["kind"], symbol))
        store.db.execute("UPDATE model_runs SET status='champion', checkpoint=? WHERE id=?",
                         (str(dest), run_id))
    store.audit("neural_champion_promoted", {"id": run_id, "symbol": symbol})


def _latest_window(cfg, store, ctx, symbol: str, payload: dict) -> np.ndarray | None:
    window = payload["window"]
    since = "1900-01-01"
    b = _bars(store, symbol, since)
    spy = _bars(store, cfg.get("universe", "benchmark", default="SPY"), since)
    vix = _bars(store, cfg.get("universe", "vix_symbol", default="^VIX"), since)
    f = _features(b, spy, vix)[FEATURES].dropna()
    if len(f) < window:
        return None
    x = f.iloc[-window:].to_numpy(np.float32)
    return ((x - payload["mean"].reshape(1, -1)) /
            payload["std"].reshape(1, -1)).astype(np.float32)


def _load(path: Path):
    import torch
    if not path.exists():
        return None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = _make_model(len(payload["features"]), len(payload["horizons"]))
    model.load_state_dict(payload["model"]); model.eval()
    return payload, model


def predict_today(cfg, store, ctx) -> tuple[dict[str, dict], dict]:
    try:
        import torch
    except ImportError:
        return {}, {"silent": "torch not installed"}
    row = store.db.execute("SELECT * FROM model_runs WHERE kind='global_tcn' "
                           "AND status='champion' ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row:
        return {}, {"silent": "no validated global TCN champion"}
    payload, model = _load(_path(cfg))
    if payload is None:
        return {}, {"silent": "global champion checkpoint missing"}
    age = (datetime.now().astimezone() - datetime.fromisoformat(payload["trained_at"])).days
    if age > int(cfg.get("neural", "max_checkpoint_age_days", default=7)):
        return {}, {"silent": f"global champion stale ({age}d)"}
    out = {}
    for sym in ctx.universe:
        x = _latest_window(cfg, store, ctx, sym, payload)
        if x is None:
            continue
        with torch.no_grad():
            pred = model(torch.from_numpy(x[None, ...])).numpy()[0]
        # A validated holding champion blends within this node only.
        hp, hm = _load(_path(cfg, sym))
        if hp is not None:
            hx = _latest_window(cfg, store, ctx, sym, hp)
            if hx is not None:
                with torch.no_grad():
                    local = hm(torch.from_numpy(hx[None, ...])).numpy()[0]
                w = float(cfg.get("neural", "holding_blend", default=0.25))
                pred = (1 - w) * pred + w * local
        out[sym] = {str(h): {"q10": float(pred[i, 0]), "q50": float(pred[i, 1]),
                             "q90": float(pred[i, 2]),
                             "probability_positive": float(1 / (1 + math.exp(
                                 -pred[i, 1] / max(1e-4, (pred[i, 2] - pred[i, 0]) / 2))))}
                    for i, h in enumerate(payload["horizons"])}
    return out, {"model_id": row["id"], "metrics": json.loads(row["metrics"] or "{}"),
                 "checkpoint_age_days": age}


def predict_run(cfg, store, ctx, run_id: str) -> tuple[dict[str, dict], dict]:
    """Shadow inference for an immutable challenger checkpoint."""
    try:
        import torch
    except ImportError:
        return {}, {"silent": "torch not installed"}
    row = store.db.execute("SELECT * FROM model_runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return {}, {"silent": "unknown model run"}
    payload, model = _load(Path(row["checkpoint"]))
    if payload is None:
        return {}, {"silent": "challenger checkpoint missing"}
    out = {}
    for sym in ctx.universe:
        x = _latest_window(cfg, store, ctx, sym, payload)
        if x is None:
            continue
        with torch.no_grad(): pred = model(torch.from_numpy(x[None, ...])).numpy()[0]
        out[sym] = {str(h): {"q10": float(pred[i, 0]), "q50": float(pred[i, 1]),
                             "q90": float(pred[i, 2]),
                             "probability_positive": float(1 / (1 + math.exp(
                                 -pred[i, 1] / max(1e-4, (pred[i, 2] - pred[i, 0]) / 2))))}
                    for i, h in enumerate(payload["horizons"])}
    return out, {"model_id": row["id"], "metrics": json.loads(row["metrics"] or "{}")}


def shadow_metrics(store, model_id: str) -> dict:
    rows = store.db.execute("SELECT * FROM model_forecasts WHERE model_id=? "
                            "AND resolved_at IS NOT NULL", (model_id,)).fetchall()
    out = {"sessions": len({r["as_of"] for r in rows}), "total": len(rows), "horizons": {}}
    for h in (5, 21):
        hs = [r for r in rows if r["horizon"] == h]
        pred = np.asarray([r["q50"] for r in hs]); actual = np.asarray([r["realized_excess"] for r in hs])
        if not hs:
            out["horizons"][str(h)] = {"n": 0}; continue
        cutoff = np.quantile(pred, .9)
        top = actual[pred >= cutoff]
        coverage = np.mean([(r["q10"] <= r["realized_excess"] <= r["q90"]) for r in hs])
        out["horizons"][str(h)] = {"n": len(hs), "ic": round(_corr(pred, actual), 4),
                                     "top_decile_alpha": round(float(top.mean()), 5),
                                     "coverage": round(float(coverage), 3)}
    return out


def maybe_promote(cfg, store) -> dict:
    """Objective-gated global promotion/rollback; never relaxes the governor."""
    champion_row = store.db.execute("SELECT * FROM model_runs WHERE kind='global_tcn' "
                                    "AND status='champion' ORDER BY created_at DESC LIMIT 1").fetchone()
    if champion_row:
        sm = shadow_metrics(store, champion_row["id"])
        hs = sm["horizons"]
        stale = (datetime.now().date() - datetime.fromisoformat(champion_row["created_at"]).date()).days > \
            int(cfg.get("neural", "max_checkpoint_age_days", default=7))
        decay = any(hs.get(str(h), {}).get("n", 0) >= 1000 and
                    (hs[str(h)].get("ic", 0) <= 0 or hs[str(h)].get("top_decile_alpha", 0) <= 0)
                    for h in (5, 21))
        if stale or decay:
            with store.db:
                store.db.execute("UPDATE model_runs SET status='retired' WHERE id=?",
                                 (champion_row["id"],))
            store.audit("neural_champion_rolled_back", {"id": champion_row["id"],
                                                         "stale": stale, "metrics": sm})
            return {"action": "rollback", "id": champion_row["id"], "metrics": sm}
    row = store.db.execute("SELECT * FROM model_runs WHERE kind='global_tcn' "
                           "AND status='challenger' ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row:
        return {"action": "none"}
    test = json.loads(row["metrics"] or "{}")
    sm = shadow_metrics(store, row["id"]); hs = sm["horizons"]
    # Five embargoed folds are intentionally mandatory. The initial single-
    # split challenger therefore remains shadow until research attaches them.
    passed = test.get("walk_forward_folds", 0) >= 5 and sm["sessions"] >= 30 and all(
        hs.get(str(h), {}).get("n", 0) >= 10_000 and
        hs[str(h)].get("ic", 0) >= .01 and hs[str(h)].get("top_decile_alpha", 0) > 0 and
        .75 <= hs[str(h)].get("coverage", 0) <= .85 for h in (5, 21))
    if passed:
        promote(cfg, store, row["id"])
        return {"action": "promote", "id": row["id"], "metrics": sm}
    return {"action": "shadow", "id": row["id"], "metrics": sm}
