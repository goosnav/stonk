# Stonk Terminal Architecture (working notes)

> Canonical spec: [../AGENTS.md](../AGENTS.md). Approved build plan: [PLAN.md](PLAN.md).
> Decision log: [DECISIONS.md](DECISIONS.md). Current status: [PROGRESS.md](PROGRESS.md).
> This file is the *implementation-level* map: what each module actually does and how data flows.

## One-paragraph summary

Stonk Terminal is a deterministic trading operating system with optional AI attachments.
Strategy **nodes** (momentum, earnings drift, congress trades, …) each turn market
data into `SignalEvent` forecasts — never orders. A regime-conditioned weighted
**ensemble** merges them into `TradeCandidate`s with bootstrap error bars. A
deterministic **risk governor** (the only component that can say yes) enforces the
per-scan-cycle **time-step budget** — worst-case loss of everything opened in a
cycle must fit inside it — plus position/sector/drawdown caps and kill switches.
**Execution** always calls broker `review_order` before `place_order`. Fills flow
back through **attribution**, which updates node scorecards and (within bounded
multipliers) node weights: that is the self-improvement loop. The broker is an
adapter (`paper`, `robinhood_mcp`, `robinhood_bridge`, later `alpaca`); the core
never imports MCP details.

## Pipeline (one scan cycle)

```
cli.scan / scheduler tick
  data.refresh()                    # daily bars via Stooq→yfinance, as_of-stamped, into sqlite
  ctx = MarketContext(store, cfg, as_of)   # lazy per-symbol DataFrames, ONLY rows <= as_of
  exits = engine.check_exits(ctx)   # stops / time stops / score decay on open positions
  regime = regime.classify(ctx)     # risk_on | neutral | risk_off | stress
  events = registry.compute_all(ctx)          # each enabled node → [SignalEvent]
  candidates = ensemble.score(events, regime) # weighted merge + conflict/cost penalty
  candidates = forecast.attach_intervals(...) # bootstrap CIs from backtest analog trades
  targets = portfolio.construct(candidates)   # rank, vol-size, caps, cash reserve
  for t in targets: risk.governor.review(t)   # APPROVED / REDUCED / REJECTED / NEEDS_HUMAN
  execution.execute(approved)       # build limit order → broker.review → place → record fill
  store.audit(everything)           # every step above writes an audit row
```

Post-close job: mark-to-market equity curve, close matured trade records,
`attribution.update()` node stats + bounded weight updates.

## Module map (specforge/)

| File | Responsibility | Key types |
|---|---|---|
| `models.py` | All dataclasses crossing module boundaries | SignalEvent, TradeCandidate, RiskDecision, OrderIntent, Fill, Position, AccountState |
| `config.py` | default.yaml ⊕ <mode>.yaml ⊕ runtime overrides; rejects dangerous values; triple-gate check for live trading | Config |
| `store.py` | SQLite (WAL). Bars, signals, candidates, orders, fills, positions, equity curve, node stats, audit log, kv state, AI ledger | Store |
| `data.py` | Daily OHLCV ingestion (Stooq primary, yfinance fallback), staleness checks, MarketContext with as_of slicing (lookahead guard) | MarketContext |
| `nodes/base.py` | SignalNode ABC + registry built from config `nodes:` section | SignalNode |
| `nodes/*.py` | One strategy node per file. Emit forecasts only | — |
| `regime.py` | SPY trend + VIX + breadth → regime + deployment multiplier | — |
| `ensemble.py` | AGENTS.md §11 step-5 scoring formula | TradeCandidate |
| `forecast.py` | Error bars: bootstrap over analog trades (same score bucket × regime) | — |
| `portfolio.py` | Ranking, ATR vol-sizing, caps, cash reserve | — |
| `risk.py` | Governor + kill switches + option validation + time-step budget | RiskDecision |
| `execution.py` | OrderIntent build, review-before-place, idempotency, fill recording | — |
| `broker/` | Adapter protocol + paper + robinhood_mcp + bridge | BrokerAdapter |
| `backtest.py` | Walk-forward daily sim, costs, SPY baseline, writes analog trades used by forecast.py | — |
| `attribution.py` | Fill→node linkage, scorecards, Bayesian-shrunk bounded weight updates, pruning | — |
| `ai.py` | OpenRouter-compatible client, reserve-then-commit budget ledger, cache | — |
| `hypothesis.py` | V4/D34: two-tier AI hypotheses (north star + short term), file mirror + dated archive, strict-JSON generation, staleness | — |
| `steering.py` | V4/D34: non-blocking strategic-choice queue, tiered expiry defaults, validated apply paths, post-close maintenance | — |
| `montecarlo.py` | Portfolio path simulation (AGENTS.md §25) for GUI/risk | — |
| `app.py` | FastAPI: dashboard page + JSON API + APScheduler startup; `/health` liveness + `/api/metrics` monitor contract | — |
| `health.py` | Truth aggregator `system_health()` (broker probe, heartbeat, market clock, readiness reasons, sanitized last error) + app-health `rollup()` → ok/degraded/stale/error. Operator side: RUNBOOK.md, `scripts/check_health.py` | — |
| `cli.py` | `stonk scan|paper|backtest|status|serve|...`; serve refuses a taken port before starting a duplicate scheduler | — |

## Non-negotiable invariants (tested)

1. Nodes never place orders; only `execution.py` talks to brokers.
2. Risk governor is deterministic and final; nothing bypasses it (AI included).
3. `MarketContext` only ever exposes rows with `date <= as_of` — same code path
   powers live and backtest, so backtest cannot look ahead.
4. Live orders need config flag + `LIVE_TRADING_ENABLED=true` env + account whitelist.
5. Bounded risk only: long equity/ETF, long calls/puts. Max loss = notional/premium.
6. Every order carries an idempotency key; duplicates are rejected by governor.
7. AI output that fails schema parse is discarded (and counted toward a kill switch).
8. Weight self-updates clamp to [min_multiplier, max_multiplier] × base weight.

## Data storage

Single SQLite file `data/specforge.db` (WAL). Backtests use a separate DB file
(`data/backtest_<tag>.db`) but identical schema, so analog-trade queries work on
either. No ORM — schema in `store.py`, plain SQL.

## Self-improvement loop (Phase 5)

- Every closed trade → per-node attributed PnL rows.
- Nightly: recompute node scorecards (hit rate, expectancy, IC, drawdown, by-regime).
- Weight update: `w_new = clamp(base_w × shrunk_expectancy_multiplier, bounds)`,
  shrinkage toward zero edge until `min_trades_before_update` reached.
- Pruning: auto-disable on negative live expectancy with n≥30, node drawdown breach,
  or cost > edge. Promotion (experimental→probation→production) requires human
  approval via GUI; the system only *proposes*.
