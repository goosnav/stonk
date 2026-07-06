# Progress / handoff state

> Update this file whenever a module lands or a decision changes.
> To pick up: read PLAN.md (phases) → ARCHITECTURE.md (module map) → this file (what exists).

## Status: Phase 1 (spine) — DONE ✅ · Phase 2 (backtest) — IN PROGRESS

### Phase 1 done (2026-07-06, commit 9c77cdc)
- [x] Scaffold, .venv, configs (default/paper/live), dev/ docs
- [x] models.py, config.py (dangerous-value rejection + live triple-gate)
- [x] store.py (full schema + audit + analog-trade queries)
- [x] data.py (Stooq→yfinance, 46 symbols × full history = 431k bars ingested;
      MarketContext with as_of slicing)
- [x] broker/base.py + paper.py (kv-persisted account, spread+slippage cost model)
- [x] nodes/: base+registry, momentum, reversal, sector_rotation, earnings_drift,
      quality_value(filter) — Phase-2 nodes written early since the engine needed one
- [x] regime.py, ensemble.py, forecast.py (bootstrap CIs), portfolio.py (vol sizing)
- [x] risk.py governor + kill switches; execution.py; engine.py run_cycle; cli.py
- [x] tests/: 18 passing (governor rejections, budget enforcement, dup orders,
      stale data, kill switches, option validation, no-lookahead, e2e paper loop)
- [x] VERIFIED: `specforge scan` end-to-end on real data — risk_on regime,
      71 signals → 29 candidates → 2 fills within $100 cycle budget, 3rd order
      correctly rejected on budget exhaustion; audit trail reconstructs cycle.

### Notable behaviors (for whoever picks up)
- Approval threshold (0.10) compares REQUESTED notional pre-reduction (see
  risk.py comment); routine cap-respecting entries run autonomously.
- earnings_drift/quality_value degrade gracefully when yfinance flakes
  (kv-cached; fail-open for the filter, fail-silent for drift).
- Resetting the paper account: do NOT delete data/specforge.db (bars live
  there too); clear the kv key 'paper_account' instead.
- Run tests: `.venv/bin/pytest tests/ -q` (offline, synthetic data).

### Phase 2 remaining
- [ ] Engine clock injection — wall-clock leaks that break backtests:
      OrderIntent.created_at, duplicate-cooldown query, kill-switch date math,
      approval expiry, position opened_at. Fix: thread an as_of-derived
      timestamp through Governor (`.today`) and Executor (`.now_iso`).
- [ ] backtest.py — walk-forward: copy bars into data/backtest_<tag>.db, run
      run_cycle(as_of=d) over SPY trading days, force approval_mode=auto,
      close remaining positions at end, report (CAGR/Sharpe/maxDD/regime split)
      vs SPY buy-hold, 70/30 OOS split, write analog trades for forecast.py.
- [ ] 10-year backtest run → go/no-go gate for default-enabled nodes/weights
- [ ] Analog trades copied into live DB so live candidates get real error bars.

### Then
Phase 3 (GUI) → 4 (Robinhood live) → 5 (self-improvement/AI/options). See PLAN.md.

## Environment notes
- Run everything with `.venv/bin/python` / `.venv/bin/pytest` (no uv on machine).
- No API keys needed until Phase 5 (AI) / Phase 4 live (Robinhood OAuth interactive).
- DB lives at data/specforge.db (gitignored).

## Known risks / open questions
- Robinhood MCP custom-client OAuth may be allowlisted → bridge fallback designed (D6).
- yfinance earnings history only reaches back ~2 years → earnings_drift backtest
  sample is thin; judge that node mostly on paper/live scorecards.
- Stooq occasionally rate-limits bursts → data.py sleeps 0.2s/symbol.
