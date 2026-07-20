# R-Series Sprint Plan (2026-07-17) — THE current plan

Source: external review of `experimental_NN_repair` at `ad3d871` (post-repair).
Supersedes every planning doc now in `dev/archive/`. One sprint per session,
suite green + local commit per sprint, exit gate proven before moving on.
Running records stay in `dev/NN_REPAIR_STAGE_B_REPORT_07.15.2026.md` (addenda)
and `dev/NN_REPAIR_COMPLETION_REPORT_07.15.2026.md`.

## The core finding to keep in view (P0)
The absolute head can TRADE (direct blend + probes) but models are still
SELECTED on excess-only evidence: validation early-stopping, tournament
selection, walk-forward folds, baselines, and persisted OOS forecasts all run
through legacy `forward_all()` (excess heads only), and `_offline_gate()`
reads those excess metrics. v2 dual-family shadow forecasts are recorded but
`resolve_forecasts_v2` has **no production call site**, so absolute forward
evidence never accumulates. The permission system and the trading system are
measuring different quantities. **Until R1 closes this, learned influence
stays hard-zero (done in R0).**

Other confirmed P0/P1s: backtest fills same-bar at close (not t−1 features →
t execution); `_seed_backtest_db` seeds no model runs/v2 forecasts so policy
comparison coincides by construction; shadow recorder picks newest
'challenger'-projected row (can starve an older finalist); RH adapter turns
fractional orders into unprotected MARKET orders; options intents can reach
the equity adapter (`options_enabled: auto`); `max_sector_exposure` and
`max_total_options_premium_risk` unenforced; governor uses cost basis not
marked notional; live profile aggressive without validation (40%/cycle, 98%
deployment, auto approvals — and cycle/deployment knobs carry NO max_equity
cap); lifecycle.transition validates state names but not transition LEGALITY
(no adjacency map, no compare-and-swap, no one-champion uniqueness);
`engine._run_cycle` F/111 and `train_challenger` F/94 still monolithic;
run.sh installs `.[dev]` but the suite needs `.[neural]` (torch) + Playwright.

## Sprints

| # | Scope | Exit gate |
|---|-------|-----------|
| **R0 — Containment + branch hygiene** (THIS SPRINT) | Rebase the 8 post-merge commits onto main; focused draft PR; `experimental_blend: 0.0`, `exploration.enabled: false`, options hard-off (yaml + kv override keys); truth-up completion report (branch pushed, test count). | Branch cleanly based on main; effective learned influence provably zero; options cannot reach the broker. |
| **R1 — Dual-family validation** | `forward_structured()` in validation early-stop (declared joint objective), tournament, walk-forward folds; absolute AND excess metrics separately, each vs its own baselines (zero/momentum/ridge per family); OOS persistence → v2; wire `resolve_forecasts_v2` into the research loop next to v1; shadow EVERY eligible finalist (or fixed prospective cohort), not newest-'challenger'; promotion gates require absolute-after-cost AND excess evidence from v2; sample size = distinct resolved sessions, not ticker-rows; lifecycle gets adjacency map + `UPDATE … WHERE lifecycle_state=expected` CAS + rowcount==1 + one-champion uniqueness + concurrency tests; hard regression: +excess/−absolute can never promote. | No model can enter experimental_live unless absolute-after-cost and excess gates BOTH pass on structured outputs. |
| **R2 — Executable backtest** | Formal decision timestamp; features through t−1; decision on session t; fill at t open/VWAP or simulated executable quote; resting limits, gap-through stops, partial/non-fills; dynamic per-symbol costs; cash-flow-adjusted NAV; retarget labels to modeled entry price. | No same-bar fill; future-bar mutation cannot affect earlier decisions; doubling costs cannot improve results. |
| **R3 — Real policy comparison** | Seed backtest DB with immutable fold-specific v2 forecasts (replay); isolated deterministic/neural/graph/blend ledgers; deterministic override disables the GRAPH too; identical entry/exit rules everywhere. | Injected forecasts cause expected policy divergence; every policy has an independent ledger. |
| **R4 — Governor + broker completion** | Sector/cluster/options aggregate caps enforced; marked-notional + pending-order exposure; fractional-order quote guard (bid/ask, max quote age, deviation + spread thresholds, slippage monitor + auto-halt); options: separate contract-tested adapter or permanent hard-disable; per-launch API bearer token; deposits/withdrawals normalized out of loss switches. | Every configured risk limit has a failing property test; wrong-asset submission impossible. |
| **R5 — Point-in-time data** | event_at/known_at/ingested_at on every non-price feature; historical membership + delistings + corporate actions; feature provenance; per-sample cost labels (spread/slippage/impact, not flat 16bps). | Every feature and universe membership passes known_at ≤ decision_at. |
| **R6 — Model laboratory** | Streaming panel (drop 12k-window cap); date-grouped cross-sectional sampler; ≥3 seeds; elastic-net + boosted-tree + TCN bakeoff; shared model + small adapters replacing per-holding fine-tunes; feature-family ablations. | TCN must beat simple models on net OOS POLICY return, not just forecast loss. |
| **R7 — Regime layer** | Deterministic regime rules vs fold-local FILTERED HMM (market-level inputs only: bench ret/vol, breadth, dispersion, VIX level+slope, credit/rates); regimes adjust deployment/thresholds/strategy weights only — never direct stock direction; no smoothed (future-peeking) states. Per review: do NOT start this before R6. | HMM retained only if net OOS utility improves and states are stable across seeds/relabeling. |
| **R8 — Experiment governance** | Immutable trial registry; purged eval; block bootstrap; Deflated Sharpe; Probability of Backtest Overfitting; sealed-holdout consumption ledger. | Every displayed result carries every attempted trial + trial-adjusted uncertainty. |
| **R9 — Shadow + tiny-live probation** | 60–90 sessions, 200–500 decisions, isolated counterfactual books, manual approvals, no options, automated decay/slippage rollback. | Positive incremental evidence vs deterministic and exposure-matched benchmarks; clean ops. |

## Merge policy (from the review)
Merge after **R0–R4** as a *software-correctness milestone* only: learned
influence off by default, options hard-disabled, no predictive claim,
executable backtest + dual-family gates required before any activation.
"The immediate priority is not training a champion. It is closing the gap
between the absolute forecast that can trade and the excess-only evidence
that currently grants it permission."

## Status log
- 2026-07-17 R0: in progress this session.
- 2026-07-17 R1 DONE (332 tests): joint dual-family early-stop; absolute
  metrics first-class in sealed/tournament eval; _offline_gate + forward gate
  require BOTH families (fail closed on legacy metrics); resolve_forecasts_v2
  wired into the research unit; shadow recorder covers ALL finalists;
  lifecycle = strict state machine (adjacency map, CAS, one-champion DB
  uniqueness, retire-before-crown). Remainder → fold-internal structured
  migration + per-family baselines tracked in R6.
- 2026-07-17 R2 DONE (336 tests): executable decision convention — features
  through t−1 (cycle as_of=t−1), decision on t, fill at t OPEN injected as the
  executable quote through the SAME live_quotes path live cycles use; entries
  limit off the open, gap-through stops fill AT the open, paper fills add
  adverse half-spread+slippage. Exit gates proven: no same-bar fill (every
  buy fill ∈ open-derived set, ∉ close-derived), future-bar mutation is
  bit-identical-immune before the boundary, doubled costs cannot improve
  results (and provably bind). Convention persisted in every report.
  Honest remainder: partial/non-fill modeling needs intrabar data (R5/R6);
  label retargeting to modeled entry price rides with R5's per-sample costs
  (one dataset schema bump); sim has no external cash flows — deposit
  normalization for LIVE loss switches stays an R4 item.
- 2026-07-19 R2.5 ops: Errno 24 FD exhaustion killed the weekend server
  (quotes "delisted" noise, yfinance cache OperationalError, broker probe
  down). Launchers now ulimit -n 4096; /health exposes open_fds (83e7ad8).
- 2026-07-19 R3 DONE (340 tests): policy comparison is a real experiment.
  _seed_backtest_db copies model_runs + model_forecasts_v2 (immutable
  replay evidence); neural.replay_forecasts serves persisted dual-family
  forecasts for EXACTLY the decision date (no torch, provenance re-validated
  fail-closed downstream); the node stashes them offline when
  neural.backtest_replay is on. deterministic policy now provably darkens
  EVERY learned pathway (blend 0, replay off, exploration off, neural node
  off, analog_graph off); fixed_blend/neural_only replay. Exit gate proven:
  injected forecasts flip actual fills between independent per-policy
  ledgers, blend audits engage in the blend book and stay silent in the
  deterministic book. Honest note: graph-policy book still N/A (no graph
  champion exists; graph events are not synthesized offline).
- 2026-07-19 R4 DONE (350 tests): governor exposure is MARKED value (not cost
  basis) + open-order reservation; max_sector_exposure enforced (unknown
  sector exempt-but-audited-once until R5 classifies; instruments.sector
  column added); max_total_options_premium_risk aggregate cap binds; loss
  switches flow-normalized (deposit can't mask a loss, withdrawal can't fake
  one; drawdown peak stays gross — noted); slippage-breach halt (3/day).
  RH adapter: structural equity-only invariant (review refuses, place
  RAISES); fractional⇒market orders pass a quote guard (fresh, aged ≤120s,
  spread ≤ max, deviation ≤1% of approved reference) + post-fill slippage
  monitor. Per-launch X-Session-Token required on every mutating /api route
  (dashboard fetches it from /api/session behind loopback+origin checks).
  Every new limit has a failing property test. R0-R4 milestone reached —
  merge policy per review applies.
