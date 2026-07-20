"""Model bakeoff: does the TCN earn its complexity?

The R6 gate: a temporal convolutional network is only allowed to influence
trading if it beats genuinely simple models on **net out-of-sample policy
return** — not on pinball loss, not on rank IC. A model that forecasts more
accurately but selects worse baskets has not earned anything.

Every candidate here is fit on the identical training rows and scored on the
identical untouched evaluation rows with the identical policy
(`portfolio_metrics.staggered_portfolio_metrics`) and the identical R5
per-sample costs. The only thing that varies is the model.

Candidates, in ascending order of "if this wins, we did not need a network":

  zero          — predict nothing. Beats a surprising number of real models.
  momentum      — last session's return, scaled by sqrt(horizon).
  ridge         — closed-form L2 on the latest context row.
  elastic_net   — L1+L2 via proximal gradient; sparse, so it also shows which
                  features carry signal at all.
  boosted_tree  — gradient-boosted depth-1 stumps on quantile-binned features;
                  the nonlinear-but-not-deep control.

Implemented in numpy on purpose: sklearn/lightgbm are not dependencies of this
repo and a few dozen lines of readable numpy is cheaper to audit than a new
supply-chain edge. These are controls, not production models — they need to be
honest and fixed, not state of the art.
"""
from __future__ import annotations

import math

import numpy as np

from . import portfolio_metrics

MODELS = ("zero", "momentum", "ridge", "elastic_net", "boosted_tree")
FAMILIES = ("absolute", "excess")


# ── simple models ─────────────────────────────────────────────────────────────

def _ridge(x_train, y_train, x_all, penalty: float = 1e-2) -> np.ndarray:
    design = np.column_stack((np.ones(len(x_train)), x_train))
    reg = np.eye(design.shape[1]) * penalty
    reg[0, 0] = 0                                   # never penalize the intercept
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y_train)
    return np.column_stack((np.ones(len(x_all)), x_all)) @ coef


def _elastic_net(x_train, y_train, x_all, alpha: float = 1e-3,
                 l1_ratio: float = .5, iterations: int = 200) -> np.ndarray:
    """Proximal-gradient (ISTA) elastic net — L1 shrinkage plus L2 ridge.

    Plain gradient step on the smooth part, soft-threshold for the L1 part.
    The step size is 1/L for the Lipschitz constant of the smooth gradient, so
    it converges without a line search.
    """
    n = len(x_train)
    intercept = y_train.mean(0)
    centered = y_train - intercept
    lipschitz = float(np.linalg.norm(x_train, 2) ** 2 / n + alpha) + 1e-12
    weights = np.zeros((x_train.shape[1], y_train.shape[1]))
    threshold = alpha * l1_ratio / lipschitz
    for _ in range(iterations):
        gradient = (x_train.T @ (x_train @ weights - centered) / n
                    + alpha * (1 - l1_ratio) * weights)
        weights = weights - gradient / lipschitz
        weights = np.sign(weights) * np.maximum(np.abs(weights) - threshold, 0)
    return x_all @ weights + intercept


def _quantile_bins(x_train, x_all, bins: int = 16):
    """Bin every feature by TRAIN quantiles — edges never see evaluation rows."""
    edges = np.quantile(x_train, np.linspace(0, 1, bins + 1)[1:-1], axis=0)
    binned = np.empty(x_all.shape, dtype=np.int16)
    for j in range(x_all.shape[1]):
        binned[:, j] = np.searchsorted(edges[:, j], x_all[:, j], side="left")
    return binned, bins


def _boosted_trees(binned_train, y_train, binned_all, bins: int,
                   rounds: int = 60, learning_rate: float = .05,
                   min_leaf: int = 20) -> np.ndarray:
    """Gradient-boosted depth-1 stumps, one additive model per target column.

    Per round and per feature the per-bin residual sums come from a single
    bincount, so a split search costs O(n) rather than O(n·thresholds).
    """
    n_features = binned_train.shape[1]
    prediction_train = np.zeros_like(y_train, dtype=float)
    prediction_all = np.zeros((len(binned_all), y_train.shape[1]), dtype=float)
    base = y_train.mean(0)
    prediction_train += base
    prediction_all += base
    for column in range(y_train.shape[1]):
        for _ in range(rounds):
            residual = y_train[:, column] - prediction_train[:, column]
            best = None
            for feature in range(n_features):
                codes = binned_train[:, feature]
                counts = np.bincount(codes, minlength=bins)
                sums = np.bincount(codes, weights=residual, minlength=bins)
                left_n, left_sum = np.cumsum(counts), np.cumsum(sums)
                right_n = left_n[-1] - left_n
                right_sum = left_sum[-1] - left_sum
                usable = (left_n >= min_leaf) & (right_n >= min_leaf)
                if not usable.any():
                    continue
                # SSE reduction of a two-leaf split, standard boosting gain.
                gain = np.where(usable,
                                left_sum ** 2 / np.maximum(left_n, 1)
                                + right_sum ** 2 / np.maximum(right_n, 1), -np.inf)
                cut = int(np.argmax(gain))
                if best is None or gain[cut] > best[0]:
                    best = (gain[cut], feature, cut,
                            left_sum[cut] / max(left_n[cut], 1),
                            right_sum[cut] / max(right_n[cut], 1))
            if best is None:
                break
            _, feature, cut, left_value, right_value = best
            for codes, target in ((binned_train, prediction_train),
                                  (binned_all, prediction_all)):
                go_left = codes[:, feature] <= cut
                target[:, column] += learning_rate * np.where(
                    go_left, left_value, right_value)
    return prediction_all


# ── panel assembly ────────────────────────────────────────────────────────────

def context_design(ds) -> np.ndarray:
    """The latest session of each window, de-normalized back to real units.

    Simple models get one row per window (not the full 60-session tensor) on
    purpose: that IS the control. If a linear model on the last session matches
    the TCN, the sequence model bought nothing.
    """
    from .. import neural
    return neural.context_rows(ds)


def simple_predictions(ds, family: str = "absolute", models=MODELS,
                       context=None) -> dict[str, np.ndarray]:
    """{model: (n_windows, n_horizons) median prediction} for every simple model.

    `context` overrides the derived (n, n_features) context matrix — that is
    how ablation knocks a feature family out without copying the panel.
    """
    from .. import neural
    y = ds["Y_absolute"] if family == "absolute" else ds["Y_excess"]
    y = np.asarray(y, dtype=np.float64)
    horizons = list(ds["horizons"])
    train = ds["masks"]["train"]
    x = context_design(ds) if context is None else np.asarray(context, dtype=np.float64)
    mean, std = x[train].mean(0), x[train].std(0) + 1e-6
    xn = (x - mean) / std
    out: dict[str, np.ndarray] = {}
    if "zero" in models:
        out["zero"] = np.zeros_like(y)
    if "momentum" in models:
        last_return = x[:, neural.FEATURES.index("r1")]
        out["momentum"] = np.column_stack(
            [last_return * math.sqrt(max(1, h)) for h in horizons])
    if "ridge" in models:
        out["ridge"] = _ridge(xn[train], y[train], xn)
    if "elastic_net" in models:
        out["elastic_net"] = _elastic_net(xn[train], y[train], xn)
    if "boosted_tree" in models:
        binned_all, bins = _quantile_bins(xn[train], xn)
        out["boosted_tree"] = _boosted_trees(binned_all[train], y[train],
                                             binned_all, bins)
    return out


# ── scoring ───────────────────────────────────────────────────────────────────

def policy_return(prediction, ds, eval_idx, family: str = "absolute",
                  costs=None) -> dict:
    """Net OOS policy return per horizon for one model's median predictions.

    `prediction` is (n_windows, n_horizons); only the evaluation rows are read.
    """
    truth = np.asarray(ds["Y_absolute"] if family == "absolute"
                       else ds["Y_excess"], dtype=np.float64)
    dates = np.asarray(ds["dates"])[eval_idx]
    if costs is None:
        costs = ds.get("sample_cost", ds.get("round_trip_cost", .0016))
    cost = (np.asarray(costs)[eval_idx] if not np.isscalar(costs)
            and np.ndim(costs) else costs)
    out = {}
    for i, horizon in enumerate(ds["horizons"]):
        out[str(horizon)] = portfolio_metrics.staggered_portfolio_metrics(
            np.asarray(prediction)[eval_idx, i], truth[eval_idx, i], dates,
            horizon=int(horizon), cost=cost)
    usable = [m for m in out.values() if m.get("utility_evidence") == "ok"]
    # Fail closed, and mean it: a model that carries evidence at ONE horizon and
    # none at the other has not been measured. Averaging over just the horizon
    # that happened to have enough cohorts hands it a real positive utility on
    # partial evidence — which the gate would then read as a win.
    complete = len(usable) == len(ds["horizons"])
    out["evidence"] = "ok" if complete else "insufficient"
    out["policy_utility"] = (round(float(np.mean(
        [m["portfolio_utility"] for m in usable])), 5) if complete else -1.0)
    return out


FEATURE_FAMILIES = {
    "price": ("r1", "range", "gap", "volume_z", "vol21", "rsi14", "atr14",
              "breakout60", "sma50_d", "sma200_d"),
    "market": ("spy_r1", "spy_r21", "sector_relative_r21", "hyg_r21", "tlt_r21"),
    "volatility": ("vix", "vix9d", "vix3m", "vix6m", "vvix", "vix_curve_9d_3m",
                   "vix_curve_1m_3m", "implied_realized_spread",
                   "vol_context_missing"),
    "valuation": ("valuation", "valuation_missing"),
    "event": ("event_proximity", "event_missing"),
    "fundamentals": ("revenue_growth", "revenue_growth_missing", "operating_margin",
                     "operating_margin_missing", "fcf_margin", "fcf_margin_missing",
                     "debt_assets", "debt_assets_missing", "dilution",
                     "dilution_missing", "accruals", "accruals_missing",
                     "liquidity", "liquidity_missing"),
    "news": ("news_sentiment", "news_missing"),
}


def ablate(ds, eval_idx, family: str = "absolute", model: str | None = None,
           families=None) -> dict:
    """Policy-return cost of removing each feature family, one at a time.

    Ablation runs on a simple model on purpose — it is fast, deterministic, and
    the question ("does this family carry signal at all?") does not need a
    network to answer. A family whose removal does not hurt is not evidence of
    a subtle deep interaction; it is a family to suspect.

    Returns {family_name: {"policy_utility": .., "delta": ..}} where a NEGATIVE
    delta means removing the family hurt, i.e. the family was carrying weight.
    """
    from .. import neural
    families = families or FEATURE_FAMILIES
    context = context_design(ds)
    if model is None:
        # Ablate the STRONGEST control, not a hardcoded one. Knocking a family
        # out of a weaker model understates its value: elastic-net scored +0.805
        # where ridge scored +0.683, so ridge's deltas describe a model nobody
        # would use.
        scored = {name: policy_return(prediction, ds, eval_idx, family)
                  ["policy_utility"]
                  for name, prediction in simple_predictions(
                      ds, family, ("ridge", "elastic_net", "boosted_tree")).items()}
        model = max(scored, key=scored.get)
    train_mean = context[ds["masks"]["train"]].mean(0)
    full = policy_return(simple_predictions(ds, family, (model,), context)[model],
                         ds, eval_idx, family)
    out = {"_full": {"policy_utility": full["policy_utility"],
                     "evidence": full["evidence"]}}
    for name, members in families.items():
        columns = [neural.FEATURES.index(f) for f in members
                   if f in neural.FEATURES]
        if not columns:
            continue
        # Knock the family out on the (n, n_features) context matrix, not on the
        # full panel — copying the panel once per family was the largest
        # allocation in the diagnostic. Ablated columns are pinned to the TRAIN
        # mean, which is what "this feature tells us nothing" actually means;
        # zero would inject a fictitious value in de-normalized units.
        knocked = np.array(context, copy=True)
        knocked[:, columns] = train_mean[columns]
        scored = policy_return(
            simple_predictions(ds, family, (model,), knocked)[model],
            ds, eval_idx, family)
        out[name] = {
            "policy_utility": scored["policy_utility"],
            "evidence": scored["evidence"],
            "delta": round(scored["policy_utility"] - full["policy_utility"], 5)}
    out["basis"] = "net_oos_policy_return_staggered_cohorts"
    out["ablated_model"] = model
    return out


def candidate_cohort_matrix(ds, eval_idx, family: str = "absolute",
                            tcn_predictions=None, horizon: int | None = None,
                            models=MODELS):
    """(n_cohorts, n_candidates) non-overlapping returns, one column per trial.

    This is the input the Probability of Backtest Overfitting needs: it asks
    whether picking the in-sample best among THESE candidates predicts their
    out-of-sample ranking. Columns must be the real alternatives that were
    actually considered, otherwise PBO measures a search nobody performed.
    """
    truth = np.asarray(ds["Y_absolute"] if family == "absolute"
                       else ds["Y_excess"], dtype=np.float64)
    dates = np.asarray(ds["dates"])[eval_idx]
    costs = ds.get("sample_cost", ds.get("round_trip_cost", .0016))
    cost = (np.asarray(costs)[eval_idx]
            if not np.isscalar(costs) and np.ndim(costs) else costs)
    horizons = list(ds["horizons"])
    # Shortest horizon by default: it yields the most independent cohorts, and
    # a deflated Sharpe on a dozen observations is not worth computing.
    index = 0 if horizon is None else horizons.index(horizon)
    candidates = dict(simple_predictions(ds, family, models))
    candidates.update((tcn_predictions or {}).get(family, {}))
    columns, names = [], []
    for name, prediction in candidates.items():
        series = portfolio_metrics.cohort_returns(
            np.asarray(prediction)[eval_idx, index], truth[eval_idx, index],
            dates, int(horizons[index]), cost=cost, offset=0)
        columns.append(series); names.append(name)
    width = min((len(c) for c in columns), default=0)
    if width < 4:
        return np.empty((0, 0)), names
    return np.column_stack([c[:width] for c in columns]), names


def compare(ds, eval_idx, tcn_predictions: dict[str, np.ndarray] | None = None,
            families=FAMILIES, models=MODELS) -> dict:
    """Full bakeoff table: every simple model and every TCN seed, both families.

    `tcn_predictions` maps a label (e.g. "tcn_seed_0") to (n, n_horizons)
    medians for that family — pass {} to score the controls alone.

    The TCN's entry is the MEDIAN seed, never the best: three seeds exist
    precisely so a lucky draw cannot be presented as the model's ability.
    """
    table: dict[str, dict] = {}
    for family in families:
        entries = {name: policy_return(pred, ds, eval_idx, family)
                   for name, pred in simple_predictions(ds, family, models).items()}
        raw_seeds = (tcn_predictions or {}).get(family, {})
        seeds = {name: policy_return(pred, ds, eval_idx, family)
                 for name, pred in raw_seeds.items()}
        # The deployable artifact is the ENSEMBLE, not one arbitrary draw. Seed
        # spread exceeded the median seed's own utility in the first real run,
        # which is exactly the variance an average removes. Scored alongside the
        # individual seeds so the gain (or absence of one) is visible.
        ensemble = None
        if len(raw_seeds) > 1:
            mean_prediction = np.mean([np.asarray(p) for p in raw_seeds.values()],
                                      axis=0)
            ensemble = policy_return(mean_prediction, ds, eval_idx, family)
        best_control = max((e["policy_utility"] for e in entries.values()),
                           default=-1.0)
        summary = {"controls": entries, "best_control_utility": best_control,
                   "seeds": seeds, "ensemble": ensemble}
        if seeds:
            utilities = sorted(e["policy_utility"] for e in seeds.values())
            median = float(np.median(utilities))
            # The gate reads the ensemble when there is one, else the median
            # seed — never the best seed, which is a lucky draw dressed up as
            # ability. The ensemble is a legitimate model, not cherry-picking:
            # it is fixed before seeing the scores and is what would ship.
            decisive = (ensemble["policy_utility"] if ensemble is not None
                        else median)
            summary.update(
                tcn_median_utility=round(median, 5),
                tcn_best_utility=round(utilities[-1], 5),
                tcn_worst_utility=round(utilities[0], 5),
                tcn_seed_spread=round(utilities[-1] - utilities[0], 5),
                tcn_ensemble_utility=(round(ensemble["policy_utility"], 5)
                                      if ensemble is not None else None),
                n_seeds=len(utilities),
                decisive_utility=round(decisive, 5),
                beats_controls=bool(decisive > best_control))
        else:
            summary.update(n_seeds=0, beats_controls=False)
        table[family] = summary
    table["verdict"] = all(table[f].get("beats_controls") for f in families)
    table["basis"] = "net_oos_policy_return_staggered_cohorts"
    return table
