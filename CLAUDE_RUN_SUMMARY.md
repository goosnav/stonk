# CLAUDE_RUN_SUMMARY — SpecForge build session (2026-07-06)

## What was built

Complete, tested, runnable MVP of SpecForge: a broker-portable speculation
engine (switchboard of strategy nodes → regime-gated ensemble with bootstrap
error bars → deterministic risk governor with per-cycle time-step budget →
review-gated execution → attribution/self-improvement), with a local web GUI,
walk-forward backtester, Robinhood MCP adapter + Claude-session bridge
fallback, AI enrichment layer (OpenRouter, reserve-then-commit budget), and
Monte Carlo risk projection. ~4,300 lines across `specforge/`, `tests/`,
`configs/`, `static/`, `dev/`.

## Files (map in dev/ARCHITECTURE.md)

- Engine: `specforge/{engine,models,config,store,data,regime,ensemble,forecast,portfolio,risk,execution,attribution,ai,montecarlo,backtest,app,cli}.py`
- Nodes: `specforge/nodes/{base,momentum,reversal,sector_rotation,earnings_drift,quality_value,news_sentiment,congress_trades,insider,options_vol}.py`
- Brokers: `specforge/broker/{base,paper,robinhood_mcp,bridge}.py`
- GUI: `static/dashboard.html` · Configs: `configs/{default,paper,live,aggressive}.yaml`
- Docs: `README.md`, `TUTORIAL.md`, `dev/{PLAN,ARCHITECTURE,DECISIONS,PROGRESS,ROADMAP}.md`
- Launchers: `run.sh`, `run.command`, `run.ps1` (Unix ones executable)
- Ops: `scripts/bridge_prompt.md`, `.env.example`, `.claude/launch.json`

## Exact commands run + results (this session)

- `.venv/bin/pytest tests/ -q` → **25 passed** (governor, time-step budget,
  kill switches, duplicates, stale data, option validation, no-lookahead,
  e2e paper loop, bridge round-trip, AI budget/parse, weight bounds, D17)
- `.venv/bin/specforge data --full` → 46/46 symbols, 431,250 daily bars
- `.venv/bin/specforge scan` (paper) → risk_on regime, 71 signals → 29
  candidates → 2 fills, $100 cycle budget enforced, 3rd order rejected on
  budget exhaustion; audit trail reconstructs the cycle
- `.venv/bin/specforge backtest --years 10 --tag v3` →
  `dev/reports/backtest_v3.json`: **CAGR 7.55%, Sharpe 0.76, maxDD 16.8%,
  PF 1.39, 1,756 trades, OOS (8.8%) > in-sample (7.1%)** — costs included
- `--mode aggressive backtest` → 15.6% in-sample but **−0.8% OOS** → rejected
  as default (D18)
- `bash run.sh` → tests + data check + smoke scan ("smoke OK") + GUI boot
- GUI verified over HTTP: all 9 API endpoints 200 with live data; node
  toggle round-trip works; dangerous risk value (max_daily_loss 0.5)
  **rejected 400 by the governor**; sane value saved + audit-logged
- Server currently running: `http://127.0.0.1:8420` (paper mode)

## Known limitations (honest list)

- Robinhood OAuth for custom MCP clients is UNVERIFIED (needs interactive
  login; may be allowlisted). The bridge fallback is tested and documented.
- Options: overlay + validation + paper accounting done; RH option order
  placement not implemented (account is far below the options unlock anyway).
- earnings_drift cannot be honestly backtested (yfinance history ~2y);
  judge it on paper/live scorecards only.
- quality_value uses current fundamentals snapshots (mild survivorship bias
  as a veto-only filter on megacaps; documented in backtest.py docstring).
- Backtest fills at same-day close ± spread/slippage model (not next-open).
- Paper broker can't mark option premiums between chain fetches.

## Next task

Follow `dev/ROADMAP.md` Sprint A: keep the paper campaign running for 2–4
weeks, watch `/api/audit` for scheduler errors, enable congress/insider nodes
after the first clean week. Sprint B (live probation) requires the human to
fund the Robinhood Agentic account and set `.env`.
