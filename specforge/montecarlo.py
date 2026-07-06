"""Monte Carlo portfolio simulator (AGENTS.md §25) — callable by the risk
governor, the GUI (/api/montecarlo), and research code.

Simulates correlated GBM-ish daily returns for current positions + cash,
applies the drawdown stop if configured, reports percentile outcomes. Numpy,
seeded → deterministic for identical inputs (GUI results are reproducible).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MonteCarloInput:
    starting_equity: float
    expected_returns: list[float]        # per-position DAILY mean return
    volatilities: list[float]            # per-position DAILY stddev
    correlation_matrix: list[list[float]]
    position_weights: list[float]        # fraction of equity per position
    horizon_days: int = 20
    n_paths: int = 2000
    transaction_costs: float = 0.0016    # round-trip friction on deployed capital
    drawdown_stop: float | None = None   # e.g. 0.15 → path stops deploying at -15%
    seed: int = 7


@dataclass
class MonteCarloOutput:
    expected_terminal_equity: float
    median_terminal_equity: float
    ci_5: float
    ci_25: float
    ci_75: float
    ci_95: float
    probability_loss: float
    probability_drawdown_gt_5: float
    probability_drawdown_gt_10: float
    expected_max_drawdown: float
    percentile_paths: dict = field(default_factory=dict)   # p5/p50/p95 equity paths for charting


def simulate(inp: MonteCarloInput) -> MonteCarloOutput:
    rng = np.random.default_rng(inp.seed)
    k = len(inp.position_weights)
    eq0 = inp.starting_equity

    if k == 0:                                   # all cash: flat line
        flat = [eq0] * (inp.horizon_days + 1)
        return MonteCarloOutput(eq0, eq0, eq0, eq0, eq0, eq0, 0.0, 0.0, 0.0, 0.0,
                                {"p5": flat, "p50": flat, "p95": flat})

    mu = np.array(inp.expected_returns)
    sigma = np.array(inp.volatilities)
    w = np.array(inp.position_weights)
    corr = np.array(inp.correlation_matrix)
    # nearest-PSD guard: correlation estimates from short windows can be broken
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eigval, eigvec = np.linalg.eigh(corr)
        corr = eigvec @ np.diag(np.clip(eigval, 1e-6, None)) @ eigvec.T
        d = np.sqrt(np.diag(corr)); corr = corr / np.outer(d, d)
        chol = np.linalg.cholesky(corr)

    z = rng.standard_normal((inp.n_paths, inp.horizon_days, k)) @ chol.T
    daily = mu + z * sigma                       # (paths, days, positions)
    port_daily = daily @ w                       # cash drags implicitly via w sum < 1
    cost_drag = inp.transaction_costs * w.sum() / max(1, inp.horizon_days)
    port_daily -= cost_drag

    equity = np.empty((inp.n_paths, inp.horizon_days + 1))
    equity[:, 0] = eq0
    stopped = np.zeros(inp.n_paths, dtype=bool)
    peak = np.full(inp.n_paths, eq0)
    for t in range(inp.horizon_days):
        step = np.where(stopped, 0.0, port_daily[:, t])
        equity[:, t + 1] = equity[:, t] * (1 + step)
        peak = np.maximum(peak, equity[:, t + 1])
        if inp.drawdown_stop:
            stopped |= equity[:, t + 1] < peak * (1 - inp.drawdown_stop)

    terminal = equity[:, -1]
    running_peak = np.maximum.accumulate(equity, axis=1)
    max_dd = ((running_peak - equity) / running_peak).max(axis=1)
    pct = lambda q: float(np.percentile(terminal, q))  # noqa: E731
    path_pct = lambda q: np.percentile(equity, q, axis=0).round(2).tolist()  # noqa: E731

    return MonteCarloOutput(
        expected_terminal_equity=round(float(terminal.mean()), 2),
        median_terminal_equity=round(pct(50), 2),
        ci_5=round(pct(5), 2), ci_25=round(pct(25), 2),
        ci_75=round(pct(75), 2), ci_95=round(pct(95), 2),
        probability_loss=round(float((terminal < eq0).mean()), 3),
        probability_drawdown_gt_5=round(float((max_dd > 0.05).mean()), 3),
        probability_drawdown_gt_10=round(float((max_dd > 0.10).mean()), 3),
        expected_max_drawdown=round(float(max_dd.mean()), 4),
        percentile_paths={"p5": path_pct(5), "p50": path_pct(50), "p95": path_pct(95)},
    )


def from_positions(store, ctx, account, horizon_days: int = 20) -> MonteCarloInput:
    """Build MC inputs from current open positions using trailing 120d stats."""
    import pandas as pd
    positions = [p for p in account.positions if p.qty > 0]
    rets = {}
    for p in positions:
        c = ctx.closes(p.symbol, lookback=140)
        if len(c) > 30:
            rets[p.symbol] = c.pct_change().dropna().tail(120)
    syms = list(rets)
    if not syms:
        return MonteCarloInput(account.equity, [], [], [], [], horizon_days)
    df = pd.DataFrame(rets).dropna()
    prices = ctx.prices()
    weights = [(p.qty * prices.get(p.symbol, p.avg_cost)) / account.equity
               for p in positions if p.symbol in syms]
    return MonteCarloInput(
        starting_equity=account.equity,
        expected_returns=[float(df[s].mean()) for s in syms],
        volatilities=[float(df[s].std()) for s in syms],
        correlation_matrix=df.corr().values.tolist(),
        position_weights=weights, horizon_days=horizon_days)
