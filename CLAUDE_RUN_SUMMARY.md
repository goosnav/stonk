# CLAUDE_RUN_SUMMARY — Stonk Terminal (final, 2026-07-06 end of day)

## What exists

Stonk Terminal v0.1.0: a validated, live-connected trading engine plus a Control
Center GUI intended as sellable software. 21 commits, 27 tests (26 offline +
1 Playwright GUI render test), decision log D1–D23.

- **Engine**: node switchboard → regime-gated ensemble (bootstrap error bars)
  → deterministic risk governor (per-cycle time-step budget, kill switches
  with cooldown/baseline-reset semantics) → review-gated execution →
  attribution with bounded weight learning.
- **Validation**: 10-year walk-forward backtest, costs included — CAGR 7.55%,
  Sharpe 0.76, maxDD 16.8%, PF 1.39, out-of-sample BETTER than in-sample
  (dev/reports/backtest_v3.json). Aggressive profile failed OOS → not default.
- **Control Center** (http://127.0.0.1:8420): 5 tabs, live market strip +
  quotes with per-number source/age provenance (broker→stooq→yfinance chain),
  positions with live P&L, error-barred projections, switchboard with node
  descriptions/degraded badges/promotion apply, risk editor with $-equivalents
  (governor-validated), AI cost meter, approvals queue, audit log, data
  freshness, Monte Carlo fan. Render-tested headless; screenshots in
  dev/reports/gui_*.png and docs/.
- **Robinhood LIVE (D22)**: standalone Python MCP client OAuth VERIFIED against
  the real agentic account 934803396 ($50 funded): account/positions/quotes
  read live; review_equity_order dry-run passed; placement deliberately not
  yet exercised. Tokens ~/.specforge/. Bridge adapter = tested fallback.
- **Ops**: run.sh/.command/.ps1 (smoke-tested end-to-end), launchd service
  installer, nightly DB backup (keep 14), macOS notifications on kill
  switches/failures/proposals, /api/version, LICENSE (placeholder), CHANGELOG.

## Verification evidence (all from this session)

- `.venv/bin/pytest tests/ -q` → 26 passed; `tests/test_gui.py` → 1 passed
- `bash run.sh` → 27 passed, smoke scan OK, GUI 200
- 140-request concurrency hammer → 0 failures (D21 fix)
- Live: `specforge --mode live status` → equity $50.00 via robinhood_mcp
- Paper scan: 71 signals → 29 candidates → budget-capped fills, full audit trail

## What is NOT done (honest, human-gated)

1. First live order placement (market hours + your approval click):
   `.venv/bin/stonk --mode live serve` → approve intent in GUI.
2. Paper/live soak time (calendar) before raising the $50/cycle live cap.
3. Options order path on RH adapter (account far below unlock anyway).
4. Sprint C learning items (need live trade data that doesn't exist yet).
5. Commercial decisions: pricing, real license terms, distribution channel.

## Resume pointers

dev/ROADMAP.md (sprints + ground rules) → dev/PROGRESS.md (state) →
dev/DECISIONS.md (D1–D23, every non-obvious choice) → dev/PRODUCT.md (product
spec + scope fences). Server likely running at :8420; service installer at
scripts/install_service.sh.
