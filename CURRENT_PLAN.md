# CURRENT PLAN — 2026-07-10 — Engine visibility + why-no-trades fixes (D39)

Read this before touching the repo. Written by the agent session of 2026-07-10
after auditing `data/specforge.db`. Supersedes nothing; extends D35–D38.

## Diagnosis (from the live audit log, not speculation)

The user saw "no trades, one small trade yesterday." The engine WAS running
and generating 27–36 candidates per cycle. Trades were blocked by a chain of
four real defects:

1. **July 9: cycle budget computed as $1.80** on a ~$100 live account (old
   kv overrides), so every entry was "time-step budget exhausted." Overrides
   have since been raised (pct 0.8, cap $50) — budget is now $50/cycle.
2. **July 10: Robinhood bounced every order** with pre-trade alert
   `order_check:EQUITY_SUITABILITY`. The MCP adapter treats any unknown
   alertType as a hard block (AGENTS.md §34.16), so 100% of approved orders
   died at broker review. A later placement attempt proved this alert maps to
   the incomplete investor-profile requirement; it must remain a hard block.
3. **Two cycles ran concurrently** (Scan-now raced the scheduler; interleaved
   cycle_ends 81af25b40c1c / 0afce5c63551 at 10:25). Both placed the same
   orders → 12 broker rejections → `rejected_orders` kill switch tripped.
   There is no cycle mutex.
4. **`rejected_orders` re-trips forever after reset**: the check counts ALL
   of today's rejections, so a manual reset is undone at the next cycle
   until midnight. The user's 10:36 reset was futile.

Plus two pure-GUI defects:

5. **Invisible P&L line**: `refreshCurve` passes `var(--green)` as a canvas
   `strokeStyle`. Canvas can't resolve CSS vars → assignment silently fails →
   the line renders in the last-set color `#232630` (the gridline gray).
6. **No state-machine visibility**: the engine's phases exist only in the
   audit log; nothing shows "what is it doing RIGHT NOW / where in the cycle."

## RESOLVED — final root cause chain (verified live, 2026-07-10 ~11:20 PT)

Executing the plan surfaced two deeper defects the audit log couldn't show
until orders actually reached the broker:

7. **RH throttles order bursts**: `place_equity_order` returns prose
   ("API error 429 … available in N seconds") which the adapter mistook for
   a resting order — 12 phantom "placed" rows that never existed at RH.
   Fixed: 429-aware retry honoring the broker's wait hint (idempotent via
   ref_id), refusals recorded as rejections with the broker's words audited
   (`broker_place_refused`), phantom rows buried by reconcile.
8. **THE final blocker — investor profile** (verbatim from RH): *"We're
   required to have you answer some questions about your investing goals
   before we can allow you to continue using Robinhood. … the user has not
   completed their investor profile for this agentic account."*
   **USER ACTION REQUIRED**: open the Robinhood app → agentic account
   934803396 → complete the investor-profile questionnaire. Until then the
   status bar shows NOT TRADING with this exact message (new health check).
   The moment it's done, the next 10-minute cycle trades — no restart needed.

## Work items (small patches, in order)

- [x] P1 `static/dashboard.html` — P&L line: real hex colors, green when net
      P&L ≥ 0 / red when < 0, thicker line + soft area fill.
- [x] P2 `specforge/engine.py` — module-level cycle lock; overlapping
      run_cycle returns `{"skipped": ...}` instead of racing. Callers guard.
- [x] P3 `specforge/risk.py` — rejected_orders counts only rejections since
      the last manual reset; trips with auto_clear_days=1.
- [x] P4 `specforge/broker/robinhood_mcp.py` + `configs/default.yaml` —
      acknowledgeable-alert list. Follow-up evidence proved
      `EQUITY_SUITABILITY` was an account-eligibility block, so the default is
      empty and suitability blocks until Robinhood reports the profile ready.
- [x] P5 `specforge/app.py` + `configs/live.yaml` — continuous heartbeat:
      `schedule.scan_interval_minutes` runs a full cycle every N minutes
      during market hours (cycle lock makes overlap with cron scans benign);
      outside hours it stamps an idle state so the GUI shows liveness.
- [x] P6 `specforge/engine.py` — engine_state kv stamped at every phase
      transition (data → signals(node) → ensemble → sizing → execute(symbol)
      → mark → idle) with a per-cycle trace.
- [x] P7 `specforge/app.py` — GET /api/engine: current phase, trace of last
      cycle, cadence, market clock, next runs.
- [x] P8 `static/dashboard.html` — new ENGINE tab: live state-machine strip,
      phase trace with timings, what's being considered/rejected (verdicts +
      reasons), cadence panel. Polls every 2.5s.
- [x] P9 config tuning for "full-time trader, no margin" directive:
      live.yaml regime multipliers (neutral 0.8, risk_off 0.4, stress 0.15),
      max_daily_new_positions 12 (+ kv override sync), max_rejected_orders 10.
      Margin/borrowing remains impossible: notional is always clamped to cash
      and max_account_deployment > 1.0 is a validation error.
- [x] P10 restart the live server so the fixes take effect; verify with a
      real cycle + /api/engine.

## Invariants preserved

- Governor remains the only gate; AI still can't place orders.
- Backtests untouched (run_cycle inner logic unchanged; lock is outer).
- No leverage: cash-clamp + deployment cap unchanged.

## D40 follow-up — 2026-07-10 11:55 PT

The rejection investigation exposed four additional correctness gaps, now
fixed and verified:

- Whole-batch preflight scales the top three ranked entries to the lower of
  cycle budget, cash, broker buying power, and remaining deployment room.
- A shared broker rejection halts the rest of that cycle's batch after one
  diagnostic instead of submitting eleven identical failures.
- Previously approved orders are revalidated against current cash, buying
  power, deployment, stale-data, kill-switch, and cycle-budget state.
- Resting sell fills now close the position and record the realized trade;
  option notional/P&L uses the required 100× contract multiplier.

Operational logging now lives under `logs/`: rotating `audit-live.jsonl` plus
`runtime-live.log`. The live service was restarted on the corrected code at
11:54 PT; broker connected, scheduler alive, no kill switches, next cycle due
at 12:04 PT. Entry readiness remains false only because Robinhood still reports
the incomplete investor profile. Verification: 65 offline tests + 1 rendered
dashboard test passed.
