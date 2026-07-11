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
