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
- 2026-07-20 R5 DONE (356 tests): point-in-time data.
  * News was the ONE feature violating known_at: `_news_series` keyed the
    LLM sentiment score to `published_at` while the score itself came into
    existence at `classified_at`. Now grouped by
    `known_at = MAX(published_at, classified_at)`, so retro-scored history is
    simply absent instead of silently backdated. `news_pit_stats()` reports
    how much of the corpus was scored after the fact (retro count, max lag).
  * Audited every other non-price feature: valuation, fundamentals and event
    proximity already key on SEC `filed` (= acceptance/known_at). No further
    violations found — the feature half of the exit gate holds.
  * Universe membership is now point-in-time: `universe.membership_as_of`
    returns the latest snapshot <= the date and **None** when history predates
    coverage (uncovered != empty != today's list). `build_dataset` drops
    windows on dates the symbol was not a member, counts covered/dropped/
    uncovered windows into `ds["pit"]`, and unions the configured symbol list
    with `universe.historical_symbols()` so delisted names stay in the panel —
    the survivor-bias fix.
  * Per-sample cost labels replace the flat 16bps: `ml_targets.sample_costs`
    prices each session from its own liquidity (spread widens as
    `sqrt(reference/dollar_volume)`, plus a sqrt-participation impact term),
    floored at the old constant and capped at MAX_SAMPLE_COST. The absolute-
    edge BCE label (train AND validation) now asks whether the return beat
    THIS name's cost on THIS session. TARGET_SCHEMA_HASH bumped accordingly;
    `ds["round_trip_cost"]` is now the median estimate and `ds["cost_floor"]`
    carries the old constant.
  Honest remainder: (a) the spread estimator is a dollar-volume proxy — real
  historical bid/ask would beat it, the live quote path has quotes but there
  is no quote archive; (b) `instruments.cik`/`sector` are still current-state
  lookups, not point-in-time (a re-sectored or re-CIK'd name leaks its present
  classification into its past); (c) corporate actions are still whatever the
  bar vendor already adjusted — no independent split/dividend ledger; (d) the
  universe PIT filter only binds where snapshots exist, and this deployment
  has snapshots only from its own runtime forward, so nearly all training
  history is `universe_uncovered_windows` — labeled, not fixed. Real
  point-in-time membership needs a historical security master (R6 input).
  * Label retargeting to modeled entry price (carried from R2) is NOT done —
    it lands with R6's dataset rebuild rather than bumping the schema twice.
- 2026-07-20 R6 PARTIAL (368 tests): model laboratory. Exit gate MET as a
  mechanism; NOT yet exercised on a real model (no trained champion exists).
  * `ml/portfolio_metrics.py`: the staggered non-overlapping cohort metric
    moved out of graph.py so the graph, the bakeoff and the promotion gate all
    score on ONE definition; it now takes R5 per-sample cost arrays.
  * `ml/bakeoff.py`: zero / momentum / ridge / elastic-net / boosted-stump
    controls, implemented in numpy (sklearn and lightgbm are not dependencies
    and ~40 auditable lines beat a new supply-chain edge for a control).
    Every candidate is fit on identical train rows, scored on identical eval
    rows, same policy, same costs. Both families scored independently — the
    R1 tail.
  * `neural.seed_predictions`: >=3 independent TCN draws. The gate reads the
    MEDIAN seed; the spread is reported. A lucky seed cannot promote.
  * `_offline_gate` now requires `bakeoff.verdict is True` AND the declared
    basis string, so legacy rows whose `beats_baselines` meant "lower pinball"
    fail closed instead of inheriting a permission they were never measured
    for. The old forecast-loss comparison survives as
    `beats_baselines_forecast_loss`, a diagnostic.
  * `neural.date_grouped_batches`: batches are now whole decision dates. Proof
    of the defect it fixes: a uniform 512-row batch over a 2500-date x 400-name
    panel lands ~460 distinct dates and leaves ~52 usable pairs out of 511 —
    the cross-sectional rank loss, the objective aligned with how the model is
    actually used, was ~90% discarded. Applied to the main loop AND the folds.
  * Walk-forward folds migrated to `forward_structured` with the joint
    dual-family objective and per-sample cost labels (the rest of the R1 tail).
    Folds no longer train and select on excess-only evidence.
  * `bakeoff.ablate`: per-feature-family policy-return deltas, reported in
    metrics. A family whose removal costs nothing is a family to suspect.
  Two real bugs found while checking the new tests were not vacuous:
    (a) `policy_return` returned a large POSITIVE utility while its own
        evidence field said "insufficient" — a model measured at one horizon
        and not the other read as a win. Now fails closed.
    (b) `MIN_VALID_OFFSETS=8` is unsatisfiable at horizon 5, which has only 5
        phases. Inherited from graph.py and latent because graph only ever
        called the metric at horizon 21. Now requires every phase when fewer
        than 8 exist.
  NOT DONE, carried forward — these are capacity/architecture items that do
  not gate the comparison, and I am not calling R6 complete:
    * Streaming panel: the 12,000-window cap and `max_windows_per_symbol` are
      untouched. R5 added delisted names to the panel, so the cap now bites
      HARDER than before — the bakeoff currently runs on a truncated panel.
      This is the most important R6 remainder.
    * Per-holding models are still full fine-tunes of the global champion; the
      shared-model-plus-small-adapters refactor is not started.
    * Ablation runs on ridge, not on the TCN — cheap and deterministic, but it
      cannot see a family that only a network can exploit.
- 2026-07-20 R6c DONE (372 tests): the panel stops copying itself; the window
  cap becomes a memory budget. This closes the streaming-panel remainder.
  * The training path de-normalized the whole 60-session panel THREE times to
    read data it then mostly discarded. `_baseline_metrics` and
    `bakeoff.context_design` wanted only each window's last session — both now
    slice first and scale after (`neural.context_rows`). Walk-forward folds
    materialized `raw` plus a renormalized copy PER FOLD — `NormalizedPanel`
    applies fold statistics lazily per batch, with the fold mean/std derived
    algebraically from the stored panel. A test pins the lazy and eager paths
    as numerically identical rather than trusting the algebra.
  * `bakeoff.ablate` copied the entire panel once per feature family; it now
    knocks columns out on the (n, n_features) context matrix, pinned to the
    TRAIN MEAN rather than zero — zero injects a fictitious value once the
    matrix is in de-normalized units.
  * With the copies gone the ceiling can be the thing that actually binds: a
    memory budget re-derived from the real window and feature count, instead
    of a hand-set 12,000 that rots whenever either changes (R5 grew both).
    12,000 -> 203,360 windows at a 2 GB budget; SAFE_PANEL_MEMORY_GB = 8 is
    the process ceiling config may only lower.
  * Trap worth recording: removing the hard-coded ceiling in code changed
    NOTHING, because configs/default.yaml still pinned max_training_windows to
    12000 and capped the result downward. The win was invisible until both
    went. A regression test now asserts the shipped config does not re-pin it.
    Same shape as the R5 "grep every caller" lesson: a limit expressed in two
    places is only lifted when both are.
  Still open from R6: per-holding models remain full fine-tunes (the
  shared-model-plus-adapters refactor is not started), and ablation still runs
  on ridge rather than the TCN.
- 2026-07-20 LOG FORENSICS: silent SEC fact starvation (375 tests).
  The 2026-07-17 `tcn_repair` run in logs/worker-live.log carried
  `feature_diagnostics.inactive` = 15 of 44 features: EVERY fundamental except
  revenue_growth, plus both news features. Root cause was three components
  disagreeing about which SEC tags matter:
    * the ingester's tag list was widened on 2026-07-15 (14f7966), but issuers
      already marked complete re-fetch only WEEKLY, so five days later not one
      had refreshed and the store still held the old narrow set;
    * `_fundamental_series` asked for 14 tags; the store had 6 of them, and a
      tag that is never fetched yields 0.0 forever — indistinguishable from a
      real zero once a missing-flag is set beside it;
    * `min_fundamental_coverage` counted issuers with ANY fact row. Measured
      against the tags the features actually need, the 213 "covered" issuers
      are ZERO — not one carries even half the required set. The gate was 100%
      wrong and the training run sailed through it.
  Fix: `ml/facts.py` is the single tag contract; ingester, feature builder and
  coverage gate all import it. A test asserts every tag the feature builder
  consumes is one the ingester fetches. Metrics now name dead feature FAMILIES
  and report `live_fraction` instead of burying 15 constant inputs in a flat
  list nobody read.
  Also checked and found HISTORICAL, not current: the ConfigError storm (6844
  in worker-live, 1068 in runtime-live) and the BrokenPipeError storm both stop
  well before the end of their logs, with successful jobs immediately after —
  they predate the ad3d871 config fix and are resolved.
  NOT fixed, and it needs a decision: news covers 2026-02-05 → 2026-07-20 (62
  days, 52 symbols) against a panel starting 2011. The news feature is ~0 for
  99% of training history and R5's known_at correction makes it sparser still.
  It is honest but near-useless as a training input at this coverage.
- 2026-07-20 R7 PARTIAL (381 tests): regime layer challenger.
  `ml/regime_hmm.py` — fold-local Gaussian HMM as a CHALLENGER to the existing
  deterministic `specforge/regime.py` (which already uses exactly the
  market-level inputs the review specified: trend, VIX level/curve, breadth,
  HYG-TLT credit proxy).
  * FILTERED states only. `filter_states` runs the forward recursion alone;
    the backward pass exists solely to fit parameters on a closed training
    window and never labels a decision. There is deliberately no public
    smoothed-posterior function — the usual library call returns
    P(state_t | x_1..x_T), which labels a 2008 session using 2009 data. The
    headline test mutates everything after t and asserts labels before t are
    bit-identical, mirroring the R2 future-bar gate.
  * Market-level inputs only, asserted by test; no per-stock surface exists.
  * Output is a scalar deployment multiplier per state, monotone in that
    state's train-window volatility by construction, so the layer can throttle
    exposure but can never point at a stock.
  * `state_agreement` compares partitions under the best relabeling, since HMM
    state indices are arbitrary.
  Bug found in my own first draft: `seed_stability` was theater. Initialization
  was deterministic (quantile slices + 1e-3 jitter), so every seed converged
  identically and agreement was ~1.0 even on pure noise — a stability check
  that cannot fail. Switched to genuine random restarts; it now reads 1.00 on
  planted regimes vs 0.64 on noise.
  NOT DONE — R7 is not complete: the HMM is not wired into `engine.run_cycle`
  and nothing consumes its multiplier yet. The retention gate ("retained only
  if net OOS utility improves") is specified but NOT run, because running it
  honestly needs the R3 policy backtest driven under both regime sources, and
  the deterministic baseline has never been measured either. Wiring it in
  before that measurement exists would grant it influence it has not earned —
  the exact failure mode R0-R1 were written to prevent.

- 2026-07-20 R8 DONE (388 tests): experiment governance.
  `ml/governance.py` - deflated_sharpe (Bailey & Lopez de Prado: discount the
  observed Sharpe by the expected MAXIMUM under the null given the trial count,
  corrected for skew/kurtosis; returns a PROBABILITY), CSCV
  probability_of_backtest_overfitting (how often the IS winner lands below the
  OOS median; ~0.5 means selection is uninformative), block_bootstrap_ci
  (contiguous blocks - an i.i.d. bootstrap on autocorrelated returns invents
  precision), and trial_adjusted_summary which fails closed on any missing
  dimension. Acklam normal-inverse + erf CDF locally; no scipy.
  `Store.record_holdout_use` / `holdout_uses` - append-only sealed-block
  consumption ledger, so an eroded holdout cannot present itself as fresh.
  `bakeoff.candidate_cohort_matrix` builds the PBO input from the alternatives
  actually considered (controls + every TCN seed).
  Test correction worth recording: my first trial-adjustment test asserted a
  strong edge should FAIL at 100k trials. It does not, and should not - a ~3.0
  annualized Sharpe survives that deflation correctly. Rewrote it around a
  MARGINAL edge, which is where search cost actually decides.

## 2026-07-20 - THE STANDING CAVEAT, MEASURED

First honest end-to-end run on real data. 200 symbols with >=1500 sessions,
203,200 windows over 3,285 distinct sessions; train <= 2022-03-07, val from
2022-04-08, sealed test from 2024-05-03 (29,761 sealed windows).

Net OOS policy utility on the sealed block (higher is better):

| candidate                 | absolute   | excess   |
|---------------------------|------------|----------|
| zero                      | -0.311     | -0.604   |
| momentum                  | -0.771     | -1.023   |
| ridge                     | -0.176     | -0.330   |
| elastic_net               | **-0.145** | -0.349   |
| boosted_tree              | -0.464     | -0.727   |
| **TCN, median of 3 seeds**| **-0.556** | **-0.548** |

**BAKEOFF VERDICT: False.** The TCN loses to every control in the absolute
family and to ridge in excess, and is worse than predicting ZERO. Seed spread
is 0.18-0.24 - wider than most gaps between candidates, so the TCN's own
number is unstable. The R6 gate refuses it. That is the gate working.

R8 governance on the same run (178 registry trials + 8 candidates):

| family   | best candidate | observed Sharpe | bar under null | deflated | PBO    |
|----------|----------------|-----------------|----------------|----------|--------|
| absolute | ridge          | 0.0239          | 0.3803         | 0.0039   | 0.3175 |
| excess   | ridge          | -0.0645         | 0.3806         | 0.0019   | 0.6905 |

After declaring the search honestly, a candidate needs a per-cohort Sharpe of
~0.38 just to match the best of random search. The best we have is 0.024.
Deflated Sharpe 0.004 means there is essentially no evidence of skill. In the
excess family PBO = 0.69: the in-sample winner lands BELOW the OOS median more
often than not, i.e. selecting on this backtest is worse than not selecting.

Read the LEVELS with care: utility = median annualized - 0.5 * max drawdown
over staggered cohorts, and for a losing series the drawdown term dominates
and approaches 1. These rank candidates honestly; they are not returns.

Three things this measurement does NOT license concluding:
1. **Every candidate is negative, controls included.** This is not "the TCN is
   bad, ridge is good". At the R5 median round-trip cost of 43.5 bps (vs the
   old flat 16) nothing in the current feature set shows cross-sectional edge
   over 2024-2026. Weak features, a cost model now correctly killing marginal
   strategies, or both.
2. **The model trained WITHOUT fundamentals or news.** The SEC fact starvation
   fix landed the same day; the store still holds the old narrow tag set and
   refills at one issuer per research task. The panel was effectively
   price/volatility/market-context only.
3. **Survivor bias is present and labeled, not fixed.** Only 5 universe
   snapshots exist, all recent, so essentially every candidate window predates
   point-in-time coverage.

The caveat is no longer an assertion, it is a measurement. There is still no
champion and learned influence remains hard-zero - now because that is the
correct answer, not merely the safe default.

## Next
R9 (shadow + tiny-live probation) CANNOT start: it requires positive
incremental evidence vs deterministic, and the evidence is negative. The real
next step is R6/R5 remediation - refill SEC facts, backfill news or drop the
feature, build historical universe membership - then re-run this measurement.
