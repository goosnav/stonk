# SpecForge — Automated Speculation Engine (Architecture + MVP Build Plan)

## Context

Build an automated stock-trading system in `/Users/jbs/Documents/code/stonks` whose first live broker interface is the Robinhood Agentic Trading MCP (`https://agent.robinhood.com/mcp/trading`), with the trading "brain" fully broker-portable. The brain is a **switchboard of explainable strategy nodes** (momentum, earnings drift, congress trades, sentiment, options-vol, etc.) combined by a weighted, regime-conditioned ensemble, gated by a deterministic risk governor, and improved over time by statistical attribution — not an LLM picking stocks. AI is an optional, budgeted enrichment layer.

[AGENTS.md](/Users/jbs/Documents/code/stonks/AGENTS.md) in the project folder is the canonical architecture spec — it is sound and we adopt it wholesale. [speculation_cheat_sheet.txt](/Users/jbs/Documents/code/stonks/speculation_cheat_sheet.txt) (33 sections of indicators/formulas) is the reference library for node math. This plan records the **deltas** from user decisions, the concrete tech stack, and the build sequence.

**Honest expectations (agreed with user):** this is a calculated-risk experiment with money the user can afford to lose. Target is outsized returns (the stack that plausibly works: liquid universe + momentum/revision/earnings-drift + sector confirmation + macro risk gate + volatility sizing + strict loss control — realistically low-teens-to-~25% APR *if* edge materializes, with wide error bars the system itself must compute and display). The system's first job is to measure whether it has edge and size accordingly, never to fake precision.

## Locked decisions (user answers)

1. **Capital:** starts <$1k → fractional equities only at launch. System is scale-aware: capability tiers unlock by account equity (options node auto-unlocks when `account_equity × max_single_option_premium_risk ≥ min_viable_premium` (default $75, i.e. ~$5k account); threshold configurable).
2. **Execution:** standalone Python MCP client (OAuth 2.1 against `agent.robinhood.com/mcp/trading`) as primary; if Robinhood allowlists clients and blocks us, fall back to a **bridge adapter**: engine writes reviewed order-intents to a queue, a scheduled Claude Code session (which already has the RH MCP connected — verified in this session, options tools included) relays them via MCP tools and writes back fills.
3. **Autonomy:** autonomous by default; orders above a configurable size threshold require human approval; full-auto selectable. **The time-step budget is the primary safety check**: each scan cycle has a hard dollar deployment cap, and worst-case loss of all new positions opened in a cycle must be ≤ that cycle's budget. Plus the AGENTS.md kill switches (daily loss, drawdown, stale data, duplicate orders, etc.).
4. **AI:** default <$1/day via OpenRouter (cheap models e.g. DeepSeek/GLM/Qwen; model + budget configurable from GUI). Budget ledger with **reserve-then-commit**: a node pre-estimates a task's token cost, reserves it, and if the remaining budget can't cover the whole task it skips gracefully (deterministic fallback) rather than half-spending mid-process.

## Architecture (summary — full detail in AGENTS.md)

```
Scheduler → Data Ingestion → Market Context → Signal Nodes (switchboard)
  → Regime Detector → Ensemble Scorer → Forecast Distribution (error bars)
  → Portfolio Constructor → Risk Governor (deterministic, final veto)
  → Broker Order Review → Execution → Fill Reconciliation
  → Attribution → Weight Update (self-improvement) → Dashboard/API
```

Key invariants (from AGENTS.md §34, enforced in code + tests):
- Nodes emit forecasts (`SignalEvent`), never orders. Ensemble + risk governor decide.
- Risk governor is deterministic; AI cannot bypass, modify configs, or approve its own changes.
- Broker abstraction: core never imports MCP details; only `broker/robinhood_mcp.py` knows tool names. Runtime tool discovery (RH says capabilities evolve).
- Bounded risk only: long equities/ETFs, long calls/puts (max loss = premium). No shorting, no naked options, no margin.
- Every trade reproducible from the audit log; every data point carries `as_of` timestamps (no lookahead — congress trades keyed to *filing* date).
- Self-improvement = statistical weight updates within bounds + pruning + promotion gates (`idea → backtest → shadow → paper → small_live → production`), never self-modifying live code.

## Tech stack

- **Python 3.12+**, `uv` for env. Single package `specforge/`.
- **SQLite** (stdlib `sqlite3`) for all state: bars, signals, trades, audit log, node scorecards, AI cost ledger. No ORM.
- **FastAPI + one vanilla HTML/JS dashboard page** (htmx-style fetch polling; no React build chain).
- **APScheduler** for scan schedule; `httpx` for data fetch; `numpy`/`pandas` for node math; `mcp` Python SDK for the Robinhood MCP client.
- Research/backtest data: **Stooq + yfinance** (free daily OHLCV); **SEC EDGAR** (filings/insiders), **FRED** (macro), **Capitol Trades/Quiver public pages** (congress trades) — added in Phase 5. Live quotes/account: Robinhood MCP.
- AI: OpenRouter-compatible client (also works for Anthropic API) with per-call cost logging.

## Repo layout (leaner than AGENTS.md §7 — collapse until a file earns a split)

```
stonks/
  AGENTS.md, README.md, TUTORIAL.md, run.sh, run.command, run.ps1
  pyproject.toml, .env.example
  configs/            default.yaml, paper.yaml, live.yaml  (risk limits, node registry, AI budget inline)
  specforge/
    models.py         # dataclasses: SignalEvent, TradeCandidate, OrderIntent, Fill, ... (AGENTS.md §8)
    config.py         # yaml load + validation + dangerous-setting warnings
    store.py          # sqlite schema + accessors + audit log
    data.py           # ingestion: stooq/yfinance bars, quotes via broker, as_of stamping
    nodes/            # one file per signal node + base.py + registry.py
    regime.py         # macro regime classifier (gate node)
    ensemble.py       # weighted regime-conditioned scoring, conflict/cost penalties
    forecast.py       # bootstrap/Bayesian-shrinkage intervals, prob_positive
    montecarlo.py     # MonteCarloInput/Output per AGENTS.md §25 (callable by ensemble, risk, GUI)
    portfolio.py      # ranking, vol sizing, caps, cash reserve
    risk.py           # governor: limits, time-step budget, option validation, kill switches
    execution.py      # order build → broker review → place → reconcile; idempotency keys
    broker/           # base.py (Protocol), paper.py, robinhood_mcp.py, bridge.py, alpaca.py(later)
    backtest.py       # walk-forward, costs/slippage, regime breakdown, report
    attribution.py    # fill→signals linkage, node scorecards, weight updates, pruning
    ai.py             # openrouter client, budget ledger (reserve-then-commit), cache
    app.py            # FastAPI: dashboard + JSON API + scheduler startup
    cli.py            # specforge paper|live|backtest|scan|status
  static/dashboard.html
  tests/              # risk governor, option validation, kill switches, dup orders, stale data, backtest no-lookahead, paper loop e2e
  scripts/bridge_prompt.md   # the prompt a scheduled Claude session runs for the bridge adapter
```

## Build phases

### Phase 1 — Spine (paper-trading closed loop, no live money)
`models.py`, `config.py`, `store.py` (schema + audit), `data.py` (daily bars for a ~40-symbol liquid universe: SPY/QQQ/IWM/DIA, sector ETFs, top liquid megacaps), `broker/paper.py` (fills at next bar open ± slippage + spread cost), `nodes/base.py` + registry, `risk.py` governor core (time-step budget, position caps, daily-loss, kill switches), `execution.py`, `cli.py`. **Exit criteria:** `specforge paper` runs a full scan→signal→risk→order→fill→log cycle on real downloaded data; audit log reconstructs the run; risk tests pass.

### Phase 2 — Deterministic alpha + validation
Nodes: `momentum.py`, `reversal.py`, `sector_rotation.py`, `earnings_drift.py` (earnings dates via yfinance/EDGAR), `quality_value.py` (filter role), `regime.py` (SPY trend + VIX + breadth → risk-on/off/stress). `ensemble.py` (AGENTS.md §11 step-5 formula), `forecast.py` (bootstrap CIs from historical analog trades), `portfolio.py`, `backtest.py` with walk-forward + costs + SPY baseline comparison. **Exit criteria:** 10-year backtest report with Sharpe/drawdown/regime breakdown vs SPY; out-of-sample split; parameter-robustness sweep. This is the go/no-go gate for what gets default-enabled.

### Phase 3 — GUI + cost/return communication
`app.py` + `dashboard.html`: dashboard (equity, PnL, drawdown, regime, kill-switch status, next scan), node switchboard (enable/weight/cap/status/live-vs-backtest expectancy), candidate trades with **horizon return + 80% interval + prob-positive + max modeled loss** (annualized APR shown secondary, per AGENTS.md §13), risk controls editor (governor rejects dangerous values), cost meter (AI $/day by node + friction drag), approval queue (approve/reject above-threshold orders), Monte Carlo portfolio projection chart. Config changes audit-logged.

### Phase 4 — Robinhood live
`broker/robinhood_mcp.py`: MCP client with OAuth (probe `agent.robinhood.com/mcp/trading`; MCP-spec OAuth 2.1 + dynamic client registration attempt), runtime tool discovery, account/portfolio/quote sync, `review_*_order` always before `place_*_order`, reject on unknown/severe review warnings. If OAuth blocked for custom clients → `broker/bridge.py`: order-intent queue table + `scripts/bridge_prompt.md` for a scheduled Claude Code session to relay intents through its connected RH MCP tools and write back results (engine still does all decisions + review-gating). Live requires `LIVE_TRADING_ENABLED=true` + account-ID whitelist + config flag. Start tiny (time-step budget ≈ $20–50). **Exit criteria:** AGENTS.md §32 safety-gate checklist green in paper first; then live probation.

### Phase 5 — Self-improvement + exotic data + AI + options
`attribution.py` (rolling scorecards, Bayesian shrinkage toward zero edge, regime-conditioned multipliers, pruning rules, promotion gates — auto-updates bounded per AGENTS.md §12.1, everything else requires approval), `ai.py` + `nodes/news_sentiment.py` (headlines → structured catalyst scores; deterministic scoring of AI output; parse-failure → discard + kill-switch counter), `nodes/congress_trades.py` (filing-date keyed), `nodes/insider.py` (EDGAR Form 4), `nodes/options_vol.py` (long calls/puts only, AGENTS.md §22 constraints, gated behind the equity threshold), `montecarlo.py` wired into GUI.

Each phase lands with its tests and updates README/TUTORIAL; final delivery meets the MVP completion contract (run.sh trio, smoke test, CLAUDE_RUN_SUMMARY.md).

## What I am deliberately NOT building (YAGNI, revisit only on demand)

- Alpaca/IBKR adapters (interface exists; build when needed).
- Intraday/minute bars in MVP (daily + a premarket/midday/near-close scan schedule per AGENTS.md §11 — daily bars carry the validated signals; minute data adds cost before it adds edge).
- Deep-learning ensemble, mean-variance optimizer, tax lots, transcript diffing — all post-MVP per AGENTS.md.
- Mobile/hosted GUI. Local FastAPI only.

## Verification

1. `pytest` — risk governor rejections, option bounded-risk validation, time-step budget enforcement, duplicate-order prevention, stale-data rejection, kill switches, backtest no-lookahead (assert every feature timestamp ≤ decision time), AI parse-failure discard.
2. `specforge backtest --years 10` — report must show costs included, out-of-sample results, SPY comparison.
3. `specforge paper` for a multi-day run — closed loop with real data, audit log complete, dashboard live at localhost.
4. Robinhood read-only integration test (get_accounts/portfolio/quotes) before any order path is enabled; `review_order` round-trip with a 1-share intent in approval mode as the first live action.
5. Smoke test in `run.sh`: boots app, hits `/api/status`, runs one paper scan cycle, asserts audit rows written.
