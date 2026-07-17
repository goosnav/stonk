# DUAL-NETWORK UPGRADE ROADMAP — 2026-07-11

## Corrected ground truth

The first D40 MLP was not proven. It briefly reported validation rank-IC 0.043,
then repeated 30-minute training on the same rows overwrote the checkpoint and
drove IC to 0.0076 after 22,315 epochs. Its gate correctly made it silent, but
the training design was wrong. The two-year weekly strategy report also gained
only 0.54% while SPY gained 38.65%. These results are the baseline, not success.

## Architecture now

Stonk Terminal has two separate learning systems:

1. **Analog neural graph** — existing specialist equations emit signed base
   activations. A bounded DAG applies learned edge weights, biases, base scales,
   and role-compatible nonlinear activations. Topology challengers may rewire
   only as acyclic, layer-forward graphs. Risk, execution, and the broker can
   never enter this graph. The deterministic ensemble remains the fallback.
2. **Temporal neural specialist** — a causal 60-session TCN consumes
   multivariate price/volume, technical, benchmark and VIX context. Reserved
   valuation channels stay explicitly missing until point-in-time SEC facts
   exist; current Yahoo ratios are never copied backward. It predicts 5d/21d
   excess-return q10/q50/q90. A confirmed holding
   may receive a complete global-model clone with every layer fine-tuned on that
   symbol, but it is still only one specialist inside the outer graph.

Champions are immutable. Training writes challengers; promotion is an atomic
database/checkpoint swap after chronological tests and forward shadow gates.

## Research plane

The daemon runs one bounded task every 15 minutes while the market is closed:
official Nasdaq/SEC catalog sync, resumable history backfill, SEC fact ingest,
universe ranking, shadow-label resolution, global/holding TCN challengers,
analog-graph challengers, scenarios, and weekly walk-forward reports. Repeating
an unchanged snapshot has fixed trial caps. When useful work is exhausted the
state is `caught_up` rather than fake activity.

Universe tiers are catalog → 1,500 liquid stocks/ADRs → 250 active names → 25
execution candidates plus holdings. Free historical data remains explicitly
survivorship-limited; live influence additionally requires forward shadow data.

## Rollout gates

- No graph/TCN champion: learned live blend is zero.
- Experimental analog graph: maximum 25% of candidate score.
- Holding TCN: maximum 25% of the temporal-neural activation.
- At least five embargoed folds, positive majority-fold IC, calibrated
  quantiles, 30 shadow sessions, and 10,000 resolved forecasts per horizon.
- Rolling decay, stale checkpoints, or integrity failure atomically roll back.
- Production promotion and any governor change remain human decisions.

## Remaining measured work

The catalog/bootstrap may take multiple closed-market runs because free feeds
are rate-limited. Point-in-time SEC coverage will fill incrementally. Until the
historical and shadow gates are actually satisfied, both learned systems remain
visible but non-influential shadows.

## Operator research controls — 2026-07-11

The dashboard and TUI now distinguish the trading loop from the closed-market
research worker. A weekend trading heartbeat is not evidence of a dead engine.
The graph API reports configured and effective blend separately; initial prior
edges are not presented as learned weights.

Operator discovery, deep research, and holding training use a durable SQLite
job queue processed by the same single research worker as autonomous work.
Discovery is deterministic and never trades. AI deep reads honor the ordinary
budget. Holding jobs create challengers and cannot mutate live champions.

Breadth now precedes optimization: below 500 research-ready companies the
queue prioritizes official-catalog history backfill. Five-fold graph/TCN
tournaments, point-in-time valuation channels, probability calibration, and
bootstrap sizing remain fail-closed behind the existing deployment gates.

## D41 — 2026-07-13 — the passivity deadlock, fixed and verified live

The node-network rollout gates above accidentally created a triple deadlock:
live entries were HARD-BLOCKED until graph+TCN champions existed; champions
need training (closed-market only) plus >=25 resolved shadow forecasts
(5-21 days to mature); the research plane AND the operator buttons refused
to run while the market was open; and SEC-filings ingestion starved model
training even when the plane did run. Observed 2026-07-13 morning: 14
cycles, ~130 candidates each, 0 orders, 3 operator jobs frozen in `queued`.
All prior model_runs are schema-incompatible (1 vs 3) so champions were
structurally WEEKS away while trading sat frozen.

Fixes (deployed + live-verified 11:05 PT — cycle 6e215773d885 bought ACA via
deterministic fallback while deep_research executed concurrently):
1. engine.py — a missing/failed learned model zeroes the BLEND, never blocks
   trading; the deterministic ensemble is the fallback. Policy encoded in
   test_live_model_failure_falls_back_to_deterministic.
2. app.py — research plane runs 24/7 in back-to-back bounded slices (600s
   budget per tick while open, 840s closed); operator buttons execute
   immediately at any hour.
3. research.py — model repair outranks filings ingestion; filings batch
   10→25; the repair/backfill alternation counter advances on filings ticks
   so coverage gaps can never pin repair off.

The champion/challenger + shadow-forecast honesty gates are UNCHANGED — they
still decide when the learned blend turns ON; they no longer hold trading
hostage. Rollout gates above should be read with this amendment.

## D46 — 2026-07-15 — production evidence replaces the weak fallback

The former fallback was not adequate: it could double-sign `avoid` signals,
ignored stored SEC/AI reports, and let momentum dominate because evidence was
not budgeted by family. Production now uses a fixed-weight company evidence
score (30% verified business memo, 20% verified catalyst memo, 15% point-in-time
financial quality, 15% market/sector context, 20% price/liquidity behavior).
Missing dossiers stay missing and reduce sizing. Live, paper, and legacy
backtest outcomes are isolated.

The TCN is now schema 4 with 28 market/company/context/missingness inputs and
must beat zero, momentum, and ridge baselines on the same untouched rows.
Graph v3 includes the real business, catalyst, earnings, insider, Congress,
regime, and temporal specialists. These learned components remain visible
shadows until their existing objective gates pass; the production evidence
ensemble—not the old flat heuristic blend—continues autonomous trading.
