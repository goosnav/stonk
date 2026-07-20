"""Experiment governance — what a backtest number is actually worth.

A Sharpe ratio reported without saying how many strategies were tried to find
it is not a measurement. Search 100 variants on noise and the best one shows a
Sharpe near 2.5 with no skill whatsoever; the number is real, the skill is not.
Everything here exists to attach that missing context.

* `deflated_sharpe` — Bailey & López de Prado. Discounts an observed Sharpe by
  the expected MAXIMUM Sharpe achievable under the null given the number of
  trials, and corrects for non-normal returns (skew and fat tails inflate a
  naive Sharpe). Answers: "given that I searched this hard, how surprised
  should I be?"
* `probability_of_backtest_overfitting` — CSCV. Splits the trial-performance
  matrix every possible way into in-sample and out-of-sample halves, and asks
  how often the in-sample winner lands below the OOS median. A PBO near 0.5
  means selection carries no information: the winner is a coin flip.
* `block_bootstrap_ci` — resamples CONTIGUOUS blocks, because daily returns are
  autocorrelated and an i.i.d. bootstrap invents precision that is not there.
* `holdout_ledger` — a sealed test set examined ten times is not a holdout. The
  count is recorded so the erosion is visible instead of forgotten.

No scipy in this repo, so the two normal functions are implemented here.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, ~1e-9 absolute)."""
    if not 0.0 < p < 1.0:
        return -np.inf if p <= 0 else np.inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected MAXIMUM Sharpe from `n_trials` independent strategies with no skill.

    This is the bar an observed Sharpe has to clear before it means anything.
    It grows with the number of trials, which is why an unrecorded search is
    unfalsifiable: without the trial count there is no bar.
    """
    n_trials = max(1, int(n_trials))
    if n_trials == 1:
        return 0.0
    scale = math.sqrt(max(sharpe_variance, 0.0))
    return scale * ((1 - EULER_MASCHERONI) * _norm_ppf(1 - 1.0 / n_trials)
                    + EULER_MASCHERONI * _norm_ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe(returns, n_trials: int, sharpe_variance: float | None = None,
                    benchmark_sharpe: float | None = None) -> dict:
    """Probability the observed Sharpe reflects skill rather than search.

    Returns `deflated_sharpe` in [0, 1] — a PROBABILITY, not a ratio. Below
    ~0.95 the result is not distinguishable from the best of N lucky draws.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        return {"observed_sharpe": 0.0, "deflated_sharpe": 0.0, "n_trials": n_trials,
                "n_observations": n, "evidence": "insufficient"}
    std = r.std(ddof=1)
    if std <= 0:
        return {"observed_sharpe": 0.0, "deflated_sharpe": 0.0, "n_trials": n_trials,
                "n_observations": n, "evidence": "degenerate"}
    observed = float(r.mean() / std)
    centered = (r - r.mean()) / std
    skew = float((centered ** 3).mean())
    kurtosis = float((centered ** 4).mean())
    if sharpe_variance is None:
        # Across-trial dispersion is unknown when only one track record is
        # supplied; the standard fallback is the sampling variance of Sharpe.
        sharpe_variance = (1 + 0.5 * observed ** 2) / max(1, n - 1)
    bar = (expected_max_sharpe(n_trials, sharpe_variance)
           if benchmark_sharpe is None else benchmark_sharpe)
    denominator = math.sqrt(max(1e-12,
                                1 - skew * observed + 0.25 * (kurtosis - 1) * observed ** 2))
    statistic = (observed - bar) * math.sqrt(n - 1) / denominator
    return {"observed_sharpe": round(observed, 4),
            "expected_max_sharpe_under_null": round(bar, 4),
            "deflated_sharpe": round(_norm_cdf(statistic), 4),
            "skew": round(skew, 4), "kurtosis": round(kurtosis, 4),
            "n_trials": int(n_trials), "n_observations": n, "evidence": "ok"}


def probability_of_backtest_overfitting(performance, n_splits: int = 10,
                                        max_combinations: int = 2000) -> dict:
    """CSCV: how often does the in-sample winner underperform out of sample?

    `performance` is (n_observations, n_trials) — one column per strategy tried.
    PBO near 0.5 means picking the in-sample best tells you nothing about OOS
    rank. Selecting on a backtest is then indistinguishable from selecting at
    random, however good the winner's numbers look.
    """
    matrix = np.asarray(performance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return {"pbo": 1.0, "evidence": "insufficient", "n_trials": int(matrix.shape[1])
                if matrix.ndim == 2 else 0}
    n_splits -= n_splits % 2                       # CSCV needs an even split count
    if n_splits < 4 or len(matrix) < n_splits * 2:
        return {"pbo": 1.0, "evidence": "insufficient", "n_trials": matrix.shape[1]}
    blocks = np.array_split(np.arange(len(matrix)), n_splits)
    logits = []
    half = n_splits // 2
    for i, chosen in enumerate(combinations(range(n_splits), half)):
        if i >= max_combinations:
            break
        in_rows = np.concatenate([blocks[b] for b in chosen])
        out_rows = np.concatenate([blocks[b] for b in range(n_splits)
                                   if b not in chosen])
        in_score = matrix[in_rows].mean(0)
        out_score = matrix[out_rows].mean(0)
        best = int(np.argmax(in_score))
        # Relative OOS rank of the in-sample winner, in (0, 1).
        rank = float((out_score <= out_score[best]).sum()) / (matrix.shape[1] + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))
    if not logits:
        return {"pbo": 1.0, "evidence": "insufficient", "n_trials": matrix.shape[1]}
    logits = np.asarray(logits)
    return {"pbo": round(float((logits <= 0).mean()), 4),
            "median_logit": round(float(np.median(logits)), 4),
            "n_combinations": len(logits), "n_trials": int(matrix.shape[1]),
            "evidence": "ok"}


def block_bootstrap_ci(returns, block_size: int = 21, samples: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> dict:
    """Circular block bootstrap CI for the mean and the Sharpe.

    Contiguous blocks preserve autocorrelation. An i.i.d. bootstrap on serially
    correlated returns reports a confidence interval far tighter than reality.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < block_size * 2:
        return {"evidence": "insufficient", "n_observations": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(samples, n_blocks))
    offsets = np.arange(block_size)
    means = np.empty(samples)
    sharpes = np.empty(samples)
    for s in range(samples):
        idx = ((starts[s][:, None] + offsets[None, :]) % n).ravel()[:n]
        sample = r[idx]
        means[s] = sample.mean()
        std = sample.std(ddof=1)
        sharpes[s] = sample.mean() / std if std > 0 else 0.0
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"mean": round(float(r.mean()), 6),
            "mean_ci": [round(float(np.percentile(means, lo)), 6),
                        round(float(np.percentile(means, hi)), 6)],
            "sharpe_ci": [round(float(np.percentile(sharpes, lo)), 4),
                          round(float(np.percentile(sharpes, hi)), 4)],
            "prob_mean_positive": round(float((means > 0).mean()), 4),
            "block_size": block_size, "samples": samples,
            "n_observations": n, "evidence": "ok"}


def trial_adjusted_summary(returns, n_trials: int, performance=None,
                           block_size: int = 21) -> dict:
    """Everything a displayed result must carry: search cost and uncertainty.

    `verdict` is True only when the deflated Sharpe clears 0.95, PBO is below
    0.5, and the bootstrap mean CI excludes zero. Any missing piece fails
    closed — an unmeasured dimension is not a passed one.
    """
    deflated = deflated_sharpe(returns, n_trials)
    bootstrap = block_bootstrap_ci(returns, block_size=block_size)
    overfitting = (probability_of_backtest_overfitting(performance)
                   if performance is not None else
                   {"pbo": 1.0, "evidence": "not_computed"})
    passes = (deflated.get("evidence") == "ok"
              and bootstrap.get("evidence") == "ok"
              and deflated["deflated_sharpe"] >= 0.95
              and overfitting.get("evidence") == "ok"
              and overfitting["pbo"] < 0.5
              and bootstrap["mean_ci"][0] > 0)
    return {"deflated": deflated, "bootstrap": bootstrap, "overfitting": overfitting,
            "n_trials": int(n_trials), "verdict": bool(passes),
            "basis": "trial_adjusted_deflated_sharpe_pbo_block_bootstrap"}
