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

### Phases 2–5 built (2026-07-06, afternoon)
- [x] Clock injection (Governor.now_iso / Executor.now_iso) — backtests replay
      the EXACT live code path at historical timestamps
- [x] backtest.py walk-forward + report + analog-trade export; caches synced
      back; --mode aggressive backtests that risk profile
- [x] GUI: app.py (FastAPI, all endpoints verified 200 via curl; dangerous
      config rejected with 400) + static/dashboard.html; scheduler with
      post-close attribution job
- [x] montecarlo.py (§25) wired to /api/montecarlo
- [x] broker/robinhood_mcp.py — OAuth 2.1 MCP client; EXACT tool schemas
      encoded from live session (D12): string params, ref_id idempotency,
      fractional⇒market+regular_hours. OAuth flow UNTESTED against RH (needs
      interactive login; may be allowlisted → bridge)
- [x] broker/bridge.py + scripts/bridge_prompt.md — VERIFIED round-trip in tests
- [x] Executor.reconcile() for async live fills (D13)
- [x] attribution.py — scorecards, bounded Bayesian weight multipliers,
      auto-disable, promotion proposals (human-gated)
- [x] ai.py — OpenRouter client, reserve-then-commit, parse-failure → AI-off-
      for-day (D14); news_sentiment, congress_trades (pub-date keyed),
      insider (cluster buys), options_vol convexity overlay (§22-gated)
- [x] 24 tests passing; README/TUTORIAL/run.sh|command|ps1 written

### Backtest findings (D15/D16 — IMPORTANT context)
- v1 (dev/reports/backtest_v1.json): per-trade edge REAL after costs
  (PF 1.34, momentum +0.74%/trade × 988) but drawdown kill switch tripped in
  the 2022 bear and froze entries for 3 years (manual-reset semantics).
  → Fixed: drawdown_cooldown_days (default 10) auto-resume.
- v2 rerun in progress (same defaults + fix). configs/aggressive.yaml added
  as the sizing knob for outsized-return attempts — validate in paper.
- earnings_drift never fires in backtest (no deep earnings history from
  yfinance) — its scorecard only accumulates from paper/live.

### Validation gate — PASSED (2026-07-06, D18)
- v3 backtest (default risk): CAGR 7.55%, Sharpe 0.76, maxDD 16.8%, PF 1.39,
  1756 trades, OOS > in-sample. Report: dev/reports/backtest_v3.json.
- aggressive profile: failed OOS (−0.8%) → kept as experiment, not default.
- reversal node defaulted off (flat in both runs); analogs (v3) loaded into
  the live DB → GUI projection shows 12.9% APR [9.0%, 16.4%] confidence=low.

## BUILD COMPLETE — system is in OPERATION phase
Everything from here is running/measuring/scaling, not construction.
→ **Next steps live in [ROADMAP.md](ROADMAP.md)** (Sprint A: paper campaign;
Sprint B: live probation — needs human to fund RH agentic account + .env;
Sprint C: learn/scale; Sprint D: hardening backlog).
Session deliverable summary: [../CLAUDE_RUN_SUMMARY.md](../CLAUDE_RUN_SUMMARY.md).
GUI was left running at http://127.0.0.1:8420 (paper mode, scheduler active).

## Environment notes
- Run everything with `.venv/bin/python` / `.venv/bin/pytest` (no uv on machine).
- No API keys needed until Phase 5 (AI) / Phase 4 live (Robinhood OAuth interactive).
- DB lives at data/specforge.db (gitignored).

## Known risks / open questions
- Robinhood MCP custom-client OAuth may be allowlisted → bridge fallback designed (D6).
- yfinance earnings history only reaches back ~2 years → earnings_drift backtest
  sample is thin; judge that node mostly on paper/live scorecards.
- Stooq occasionally rate-limits bursts → data.py sleeps 0.2s/symbol.
