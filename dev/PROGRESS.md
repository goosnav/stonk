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

## Sprint E — Control Center v2 (2026-07-06 evening, commits 32ae062 + c885163)
- [x] specforge/quotes.py — QuoteService provider chain (broker→stooq→yfinance),
      30s cache, every quote stamped {price, change_pct, as_of, source}.
      VERIFIED live: /api/quotes returned real prices (source-labeled).
- [x] app.py: /api/market (strip+regime+breadth+scan times), /api/quotes,
      /api/broker/status, /api/broker/connect (background OAuth probe → kv
      broker_probe; RH adapter got read-only probe()), /api/proposals; status
      positions now marked with live quotes incl. P&L $ and quote provenance.
      All 14 endpoints curl-verified 200.
- [x] static/dashboard.html v2 — full control center: 5 tabs (Overview /
      Trading / Switchboard / Risk & Budget / Activity), live market strip,
      next-scan countdown, alert rail, per-section plain-English explainers,
      node descriptions, $-equivalents on risk %, Connect Robinhood card,
      audit filter. JS `node --check` clean; NOT yet eyeballed in a browser
      (Chrome extension was disconnected) — first human look may find layout
      nits, logic is contract-tested.
- [x] Bug fixes with tests: tz-safe daily order counting (evening UTC-shift),
      governor vetoes ≠ broker rejections (D19, kill-switch livelock),
      engine↔broker position mismatch guard (D20, self-heals orphan state).
- Product vision + scope fences: dev/PRODUCT.md. Remaining polish list:
  ROADMAP Sprint E step 4 (tooltips, freshness panel, promotion Approve
  button, empty states) — all small, no unwritten load-bearing blocks.
- USER ACTION WAITING: $50 is in the Robinhood account. Click "Connect
  Robinhood" on the Overview tab (or POST /api/broker/connect) — OAuth opens
  in the browser; on allowlist error switch to the bridge (TUTORIAL §5).

## LIVE MILESTONE (2026-07-06 21:11, D22) — Robinhood connected for real
- User completed OAuth via the GUI Connect card; standalone Python MCP client
  verified against the real agentic account 934803396: equity $50 read live,
  real-time quotes, review_equity_order dry-run passed (order-arg mapping
  correct: fractional⇒market, string params). Bridge = fallback only.
- .env autoloaded by config.py; whitelist + LIVE_TRADING_ENABLED set; third
  gate (config flag) opens only with `--mode live`.
- order_checks.alertType parsing added: unknown alert types block orders.
- Concurrency fix D21 (per-thread sqlite). GUI render-verified headless
  (tests/test_gui.py, needs playwright; screenshots dev/reports/gui_*.png).
- scripts/install_service.sh — launchd persistence (login start, crash restart).
- NEXT LIVE STEP (human): during market hours run
  `.venv/bin/specforge --mode live serve`, approve the first queued intent in
  the GUI, verify the fill in the Robinhood app + next-cycle reconcile.

## BUILD COMPLETE — system is in OPERATION phase
Everything from here is running/measuring/scaling, not construction.
→ **Next steps live in [ROADMAP.md](ROADMAP.md)** (Sprint A: paper campaign;
Sprint B: live probation — needs human to fund RH agentic account + .env;
Sprint C: learn/scale; Sprint D: hardening backlog).
Session deliverable summary: [../CLAUDE_RUN_SUMMARY.md](../CLAUDE_RUN_SUMMARY.md).
GUI was left running at http://127.0.0.1:8420 (paper mode, scheduler active).

## Sprint D progress (2026-07-07, scheduled session)
- Sprint A daily check: server up, scheduler armed for 09:45 ET; zero
  scheduler_errors; yesterday's kill-switch churn in the audit was the D19
  dev session itself (fix verified holding: post-21:00 cycles show governor
  vetoes with kill_switches=[]). 27 -> 28 tests green.
- [x] Regime-conditioned weight multipliers (D24) — per-(node, regime)
  shrunk-IR multiplier, kv `regime_multipliers`, replaces the global
  multiplier when the cell has >=30 trades (config regime_min_n). Inert
  until paper data accumulates; test added.
- [x] D25: expired intents can no longer be approved (guard in
  store.decide_approval; GUI 409 / CLI REFUSED; order+approval marked
  expired). 28 -> 29 tests green.

## Environment notes
- Run everything with `.venv/bin/python` / `.venv/bin/pytest` (no uv on machine).
- No API keys needed until Phase 5 (AI) / Phase 4 live (Robinhood OAuth interactive).
- DB lives at data/specforge.db (gitignored).

## Known risks / open questions
- Robinhood MCP custom-client OAuth may be allowlisted → bridge fallback designed (D6).
- yfinance earnings history only reaches back ~2 years → earnings_drift backtest
  sample is thin; judge that node mostly on paper/live scorecards.
- Stooq occasionally rate-limits bursts → data.py sleeps 0.2s/symbol.

## Sprint D progress (2026-07-07 afternoon, scheduled session #2)
- Daily check: live server healthy, 4 scans ran (09:45/12:30/15:30 + probes),
  zero scheduler_errors, nightly db_backup fired, 10 intents pending approval
  (they expire per D25 — approve during market hours or let them lapse).
- [x] D26: `_commit_reports()` in app.py — post-close job now git-commits
  dev/reports when it changed (best-effort, audited `reports_committed`).
  Hoisted to module level for testability; test in test_pipeline.py.
- [x] D27: missed-scan watchdog — scan/post-close jobs get
  misfire_grace_time=1800 (fires up to 30 min late after laptop wake) and an
  EVENT_JOB_MISSED listener that audits `scheduler_missed` + desktop-notifies.
  Kill-switch and scan-failure notifications already existed (Sprint E).
- 30 tests green. NOTE: the running live server (pid from before this session)
  still runs the old code — restart `specforge --mode live serve` outside
  market hours to pick up D26/D27. Not restarted automatically to avoid
  disturbing the live broker session.
- Remaining Sprint D backlog: EDGAR point-in-time earnings, Alpaca adapter
  (both are big, deliberate builds — not scheduled-session material).

## Sprint D progress (2026-07-07 evening, scheduled session #3)
- Ops: restarted the live server outside market hours (23:05 ET) so it now
  runs D26/D27 code (previous pid was pre-commit 99ab8db). Verified live
  broker readback ($50 equity) and scheduler armed post-restart. NOTE: no
  launchd agent is actually installed — scripts/install_service.sh exists
  but was never run; server is a plain nohup process. Human decision:
  run the install script for crash/reboot persistence.
- Daily check: 4 cycles today, zero scheduler_errors, db_backup fired
  13:30 PT, 10 intents queued then expired per D25 (nobody approved —
  expected, human wasn't at the GUI during market hours).
- [x] D28: `/api/health` endpoint — liveness probe (no broker calls) with
  scheduler_running + named next_runs (scan jobs now have readable ids
  `scan_HH:MM` / `post_close`). Test in test_pipeline.py. 30 tests green.
- Remaining Sprint D backlog unchanged: EDGAR point-in-time earnings,
  Alpaca adapter — deliberate builds, not scheduled-session material.

## Sprint D progress (2026-07-08 ~00:05 PT, scheduled session #4)
- Daily check: /api/health OK (live mode, scheduler armed for today's
  09:45/12:30/15:30 + post_close), zero scheduler_errors yesterday,
  db_backup fired, 4 cycles ran; 10 approval_queued intents expired
  unapproved again (third day running) — root cause: nobody is told.
- [x] D29: desktop notification when a scan queues intents for approval —
  D25 made silent expiry safe, but nothing ever TOLD the human intents
  were waiting; scan_job now counts pending_approval entries in the cycle
  summary and fires _notify. No new test: two-line count over
  already-tested summary data inside the scheduler closure. 31 tests green.
- Ops: restarted live server ~00:10 PT (outside market hours) onto D29.
- Still outstanding (human): launchd persistence — scripts/install_service.sh
  has never been run; server remains a plain nohup process.

## Sprint D progress (2026-07-08 morning, scheduled session #5)
- Daily check: /api/health OK (live, D29 code, scheduler armed for today),
  zero scheduler_errors since 07-07, db_backup + weight_update fired,
  4 cycles ran; 10 pending approvals are within their TTL window (the
  cycle-start sweep + D25 guard handle expiry correctly — verified, no bug).
- [x] D30: live approval TTL 24h -> 6h (configs/live.yaml). Approved intents
  are placed at their original queue-time limit/qty (no re-pricing), so a
  24h window let live orders fire at day-old prices. Config-only; 31 tests
  green. Upgrade path if longer TTL is ever wanted: re-quote at placement.
- Backfilled DECISIONS.md D26-D29 (sessions #2-#4 logged them in PROGRESS
  only, violating ground rule 5).
- NOTE: running live server predates this change but reads config per-run?
  No — config is loaded at startup; restart outside market hours to pick up
  D30. Still outstanding (human): launchd persistence never installed.

## Sprint D progress (2026-07-08 ~10:05 PT, scheduled session #6)
- Daily check: /api/health OK (live, scheduler armed, 15:30 scan +
  post_close pending today); server WAS restarted onto D30 at 05:06 PT
  (session #5's "restart outstanding" note is stale — pid start time equals
  the D30 commit time). Zero scheduler_errors; two scans today queued
  CAT/GE/AMD intents (13 pending_approval, inside the new 6h TTL).
- [x] D31: silenced the benign "Session termination failed: 400" warning
  (mcp lib teardown vs RH server) via logger level in robinhood_mcp.py —
  see DECISIONS.md D31. 31 tests green.
- Remaining backlog unchanged: EDGAR point-in-time earnings, Alpaca adapter
  (deliberate builds). Still outstanding (human): launchd persistence —
  scripts/install_service.sh never run; server is a plain nohup process.
- Note: server restart not needed for D31 urgently (log noise only); it will
  be picked up at the next routine restart.
