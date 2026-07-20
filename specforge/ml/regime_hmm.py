"""Fold-local Gaussian HMM regime challenger — FILTERED states only.

`specforge/regime.py` is the deterministic baseline: benchmark trend, VIX level
and curve, breadth, credit proxy → a label and a deployment multiplier. This
module is its challenger, and it is retained only if it earns retention.

Three constraints define this module; each one exists because breaking it is
the standard way an HMM regime layer produces a beautiful, false backtest.

**1. Filtered, never smoothed.**
The usual `hmmlearn`-shaped call returns the SMOOTHED posterior
P(state_t | x_1..x_T) — conditioned on the WHOLE series, including the future.
Label a 2008 session "crisis" using 2009 data and the backtest looks
clairvoyant, because it is. `filter_states` runs the forward recursion only:
P(state_t | x_1..x_t). The backward pass exists here solely to FIT parameters
on a closed training window, which is legitimate; it never labels a decision.
There is deliberately no public smoothed-posterior function to reach for.

**2. Market-level inputs only.**
Benchmark return and realized vol, VIX level and slope, breadth, credit spread
proxy, cross-sectional dispersion. No per-stock features. A regime layer that
sees individual names stops being a regime layer and starts being a second,
unvalidated alpha model wearing a macro costume.

**3. Regimes move deployment and thresholds — never direction.**
The output is a multiplier on how much capital may be deployed, not a view on
any stock. Nothing here can make the system want to own something.

Retention gate: net OOS policy utility must improve AND states must be stable
across seeds. An HMM will happily fit 3 states to noise and relabel them on
every restart; unstable states are noise with a Greek letter attached.
"""
from __future__ import annotations

import numpy as np

FEATURES = ("bench_return", "bench_vol", "vix_level", "vix_slope",
            "breadth", "credit", "dispersion")
MIN_SESSIONS_PER_STATE = 30


# ── Gaussian HMM ──────────────────────────────────────────────────────────────

def _log_gaussian(x, means, variances):
    """(T, K) log density of each session under each state's diagonal Gaussian."""
    x = np.asarray(x, dtype=np.float64)
    diff = x[:, None, :] - means[None, :, :]
    return -0.5 * (np.log(2 * np.pi * variances).sum(1)[None, :]
                   + (diff ** 2 / variances[None, :, :]).sum(2))


def _forward(log_b, start, transition):
    """Filtered log-alpha and log-likelihood. Forward recursion ONLY.

    alpha_t(i) = P(state_t = i, x_1..x_t). Nothing after t is touched, which is
    the property the whole module exists to guarantee.
    """
    n, k = log_b.shape
    log_alpha = np.empty((n, k))
    log_alpha[0] = np.log(start + 1e-300) + log_b[0]
    log_transition = np.log(transition + 1e-300)
    for t in range(1, n):
        stacked = log_alpha[t - 1][:, None] + log_transition
        peak = stacked.max(0)
        log_alpha[t] = peak + np.log(np.exp(stacked - peak).sum(0)) + log_b[t]
    peak = log_alpha[-1].max()
    return log_alpha, float(peak + np.log(np.exp(log_alpha[-1] - peak).sum()))


def _backward(log_b, transition):
    n, k = log_b.shape
    log_beta = np.zeros((n, k))
    log_transition = np.log(transition + 1e-300)
    for t in range(n - 2, -1, -1):
        stacked = log_transition + (log_b[t + 1] + log_beta[t + 1])[None, :]
        peak = stacked.max(1)
        log_beta[t] = peak + np.log(np.exp(stacked - peak[:, None]).sum(1))
    return log_beta


def fit(x, n_states: int = 3, seed: int = 0, iterations: int = 50) -> dict:
    """Baum-Welch on a CLOSED training window.

    Using the backward pass here is not leakage: every observation in `x` is in
    the past relative to any decision this model will later label. Leakage
    would be applying smoothed states to those decisions, which `filter_states`
    structurally cannot do.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    rng = np.random.default_rng(seed)
    # Random-restart initialization: `n_states` distinct observed sessions as
    # the initial means. Seeding the init deterministically (quantile slices
    # with a whisker of jitter) makes every seed converge to the same optimum,
    # which would make `seed_stability` report perfect agreement even on pure
    # noise — a stability check that cannot fail is not a check. Genuine
    # restarts expose a multi-modal likelihood surface, which is exactly the
    # symptom of states fitted to noise.
    picks = rng.choice(n, size=min(n_states, n), replace=False)
    means = x[picks].astype(np.float64).copy()
    if len(means) < n_states:                        # degenerate tiny windows
        means = np.vstack([means] * n_states)[:n_states]
    means += rng.normal(scale=1e-6, size=means.shape)
    variances = np.maximum(np.tile(x.var(0), (n_states, 1)), 1e-6)
    start = np.full(n_states, 1.0 / n_states)
    transition = np.full((n_states, n_states), 1.0 / n_states)
    previous = -np.inf
    for _ in range(iterations):
        log_b = _log_gaussian(x, means, variances)
        log_alpha, loglik = _forward(log_b, start, transition)
        log_beta = _backward(log_b, transition)
        log_gamma = log_alpha + log_beta
        log_gamma -= log_gamma.max(1, keepdims=True)
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(1, keepdims=True)
        # Expected transitions.
        xi = np.zeros((n_states, n_states))
        log_transition = np.log(transition + 1e-300)
        for t in range(n - 1):
            block = (log_alpha[t][:, None] + log_transition
                     + (log_b[t + 1] + log_beta[t + 1])[None, :])
            block -= block.max()
            weights = np.exp(block)
            xi += weights / weights.sum()
        start = gamma[0] / gamma[0].sum()
        transition = xi / np.maximum(xi.sum(1, keepdims=True), 1e-300)
        weight = np.maximum(gamma.sum(0), 1e-300)
        means = (gamma.T @ x) / weight[:, None]
        variances = np.maximum(
            (gamma.T @ (x ** 2)) / weight[:, None] - means ** 2, 1e-6)
        if abs(loglik - previous) < 1e-4:
            break
        previous = loglik
    return {"start": start, "transition": transition, "means": means,
            "variances": variances, "n_states": n_states, "seed": seed,
            "loglik": previous, "n_train": n}


def filter_states(x, params) -> tuple[np.ndarray, np.ndarray]:
    """(states, posterior) using ONLY data up to each session.

    This is the sole labeling entry point. There is no smoothed counterpart on
    purpose — a smoothed label at t is a function of observations after t.
    """
    x = np.asarray(x, dtype=np.float64)
    log_b = _log_gaussian(x, params["means"], params["variances"])
    log_alpha, _ = _forward(log_b, params["start"], params["transition"])
    log_alpha -= log_alpha.max(1, keepdims=True)
    posterior = np.exp(log_alpha)
    posterior /= posterior.sum(1, keepdims=True)
    return posterior.argmax(1), posterior


# ── stability ─────────────────────────────────────────────────────────────────

def _permutations(k: int):
    from itertools import permutations
    return permutations(range(k))


def state_agreement(a, b, n_states: int) -> float:
    """Best-permutation agreement between two state sequences.

    HMM state indices are arbitrary: two identical fits can label the same
    regime 0 and 2. Comparing raw indices would report a perfect model as
    unstable, so the best relabeling wins — which also means a HIGH score here
    is a real claim about the partition, not about the numbering.
    """
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or len(a) != len(b):
        return 0.0
    return max(float((a == np.asarray([p[s] for s in b])).mean())
               for p in _permutations(n_states))


def seed_stability(x_train, x_eval, n_states: int = 3, seeds: int = 3) -> dict:
    """Refit under several seeds; how much do the filtered partitions agree?"""
    sequences = [filter_states(x_eval, fit(x_train, n_states, seed=s))[0]
                 for s in range(seeds)]
    pairs = [state_agreement(sequences[i], sequences[j], n_states)
             for i in range(len(sequences)) for j in range(i + 1, len(sequences))]
    occupancy = [float((sequences[0] == s).mean()) for s in range(n_states)]
    return {"mean_agreement": round(float(np.mean(pairs)), 4) if pairs else 0.0,
            "worst_agreement": round(float(min(pairs)), 4) if pairs else 0.0,
            "occupancy": [round(o, 4) for o in occupancy],
            "degenerate": bool(min(occupancy) * len(x_eval) < MIN_SESSIONS_PER_STATE),
            "n_seeds": seeds}


# ── deployment mapping ────────────────────────────────────────────────────────

def deployment_multipliers(x_train, params, floor: float = 0.0,
                           ceiling: float = 1.0) -> np.ndarray:
    """One multiplier per state, ranked by that state's TRAIN-window volatility.

    Calmer state → more deployment. Derived from train data only, and it is a
    scalar per state: the regime layer can throttle exposure, never point at a
    stock. Monotone in volatility by construction, so it cannot invent a
    "buy aggressively into chaos" state by overfitting.
    """
    states, _ = filter_states(x_train, params)
    volatility_column = FEATURES.index("bench_vol")
    x_train = np.asarray(x_train, dtype=np.float64)
    levels = []
    for state in range(params["n_states"]):
        mask = states == state
        levels.append(float(x_train[mask, volatility_column].mean())
                      if mask.any() else np.inf)
    order = np.argsort(levels)                      # calmest first
    multipliers = np.full(params["n_states"], floor)
    if params["n_states"] > 1:
        steps = np.linspace(ceiling, floor, params["n_states"])
        for rank, state in enumerate(order):
            multipliers[state] = steps[rank]
    else:
        multipliers[:] = ceiling
    return multipliers
