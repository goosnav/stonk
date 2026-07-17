# NN Repair — Stage B Report (dual-target migration COMPLETE)

Branch: `experimental_NN_repair`. No push, no live restart, no broker, no
production-DB or live-config mutation. Safety tag at the pre-migration HEAD:
`nn-repair-pre-dual-targets-20260715` → `476731e`.

Status legend: **[impl]** implemented · **[test]** covered by a test ·
**[smoke]** exercised end-to-end · **[unproven]** statistically unproven.

The central defect is fixed: a positive benchmark-relative excess forecast can
no longer be read as a positive absolute expected return. Trade eligibility now
keys off the **absolute** forecast after modeled cost; the **excess** forecast
only informs cross-sectional ranking.

---

## Local commits (small, independently green; none pushed)

| Commit | Sub-stage | Tests |
|--------|-----------|------:|
| `3b6c4f6` | Stage A (folds, graph cohort metric, NeuralForecast) | 225 |
| `72e0f3a` | B2 receptive field (63 sessions) | 226 |
| `14ac547` | B1A absolute + excess targets | 234 |
| `bf94a54` | B1B dual output heads + multi-task loss | 238 |
| `6124eed` | B3 temporal/context feature split | 244 |
| `e37c5dc` | B4A schema v6 + dual calibration (q50 bias fixed) | 250 |
| `38987c3` | B4B `model_forecasts_v2` persistence | 254 |
| `5d1fe68` | B4C structured inference + node/graph semantics | 262 |
| (this doc) | smoke test + report | 263 |

(`06d8741`, an automated "Tentative Unstable Update" by the branch owner, landed
between B4B and B4C; it only committed two pre-existing dev/ docs and touched no
migration code.)

## Architecture: before → after

Before: single-tensor TCN, 3 dilated blocks (~15-session field), all 44 features
in both branches, **excess-only** heads, excess q50 read as absolute in the node.

After: dual-branch causal TCN.
- Temporal branch: 24 sequence features → 5 dilated blocks (d=1,2,4,8,16),
  receptive field **63** ≥ 60. Context branch: 20 point-in-time features from the
  **latest** session → Linear(16). Fused → 4 quantile heads + 4 probability heads.
- Predicts **absolute** and **excess** return distributions per horizon.

## B1 target migration [impl][test][smoke]
Per horizon h∈{5,21}, same decision date t:
```
absolute_h = close[t+h]/close[t] − 1
excess_h   = absolute_h − (benchmark[t+h]/benchmark[t] − 1)
```
Dataset carries `Y_absolute[n,2]` and `Y_excess[n,2]` (excess unchanged from the
prior inline definition). `round_trip_cost = (spread_bps+slippage_bps)·2/1e4 =
0.0016` — the repo's existing constant; deducted exactly once. Probability labels:
`absolute > round_trip_cost` (hence *edge*), `excess > 0`.

Tensor shapes (internal, parallel dims — not a flat vector):
```
absolute_quantiles [B,2,3]  excess_quantiles [B,2,3]
probability_absolute_edge_positive [B,2]  probability_excess_positive [B,2]
```
Multi-task loss: absolute pinball + excess pinball + absolute-edge BCE +
excess-positive BCE + date-grouped ranking loss on the **excess** median only.

## B3 feature separation [impl][test]
`TEMPORAL_FEATURES` (24: returns/range/gap/volume/vol/RSI/ATR/breakout/MA-dist/
benchmark/sector-rel/VIX-complex/HYG/TLT) and `CONTEXT_FEATURES` (20: valuation/
growth/margins/leverage/dilution/accruals/liquidity/event/news + missingness).
Disjoint and complete over FEATURES. Temporal encoder sees only
`x[:,:,temporal_idx]`; context encoder only `x[:,-1,context_idx]`. Proven by:
early temporal feature moves output; early context feature does not; final
context feature does.

## B4A checkpoint + calibration [impl][test]
`MODEL_SCHEMA=6`, `ARCHITECTURE_HASH=v8`. Checkpoints persist schema, arch hash,
feature/temporal/context/target-schema hashes, horizons, round-trip cost,
structured calibration, dataset manifest id, best-effort code commit.
`_load_checked` rejects on schema OR feature OR architecture OR target-schema
mismatch **before** any `state_dict` load — old two-head checkpoints cannot be
coerced in. Calibration is validation-only and now fixes the long-standing q50
median-bias defect (was hard-coded 0); dual-family (absolute threshold =
round-trip cost, excess threshold = 0); ordering restored after offsets;
malformed calibration is a safe no-op, never a silent clamp of forecasts.

## B4B persistence [impl][test][smoke]
Additive `model_forecasts_v2(model_id, as_of, symbol, horizon, absolute_q10/50/90,
excess_q10/50/90, probability_absolute_edge_positive, probability_excess_positive,
resolved_at, realized_absolute, realized_excess, feature_hash, target_schema_hash,
dataset_manifest_id)`. Legacy `model_forecasts` untouched. `record_forecast_v2`
is idempotent and refuses incompatible target hashes; `resolve_forecasts_v2`
writes both realized families from identical start/end sessions.

## B4C inference + consumer semantics [impl][test][smoke]
`predict_today`/`predict_run` emit `{symbol: {horizon: NeuralForecast}}` via one
helper `build_neural_forecast` (no duplicated index math). Node (v2):
- long iff `absolute_q50 − cost > min_edge` AND `probability_absolute_edge_positive
  ≥ min_prob`; `+excess/−absolute` is skipped (not a long, not a misleading avoid).
- `expected_return`/`downside` = **absolute**; `score`/`confidence` = **excess**
  (what the graph ranks via `signed_alpha`). Evidence carries both families + cost.
Shadow metrics report absolute quality separately from excess.

## Tests & coverage
- Full suite: **263 passed** (`.venv/bin/python -m pytest -q`), `git diff --check` clean.
- `tests/test_nn_repair.py`: 59 targeted checks + the end-to-end smoke.
- Coverage (`coverage run --source=specforge -m pytest <neural+graph tests>`):
  `ml/schema.py` 100%, `ml/targets.py` 100%, `ml/__init__.py` 100%,
  `nodes/neural.py` 93%, `neural.py` 71%, `graph.py` 82%.
  (Whole-suite `pytest --cov` is blocked by a coverage-C-tracer × numpy
  "cannot load module more than once" incompatibility in this env; the
  `coverage run` path above works.)

## Smoke test [smoke]
`test_smoke_end_to_end_dual_target`: synthetic point-in-time dataset → bounded
train → checkpoint create → metadata validate → reload → structured batch
inference → typed NeuralForecast (finite, ordered quantiles, bounded probs,
correct absolute/excess mapping) → v2 persistence → resolution (both realized) →
node computation. Fixture tmp DB only; no network, no broker, no live config.

## What remains statistically UNPROVEN
- **No predictive edge is claimed.** The model trains, calibrates, and produces
  economically-correct forecasts; whether it *beats baselines* is unmeasured
  here. Absolute-family shadow evidence must accumulate (5–21 sessions) before
  any real judgement.
- Survivorship bias (analysis #19) unaddressed — a data problem, out of scope.
- Holding adapters still full fine-tunes (analysis #9) — Stage D.

## Stage C readiness / prerequisites
Stage B makes the output economically correct; neural influence remains **0**.
Stage C (NOT started per instruction) can now: add `neural.experimental_blend`
and blend `neural_score` — derived from `NeuralForecast.absolute_edge_after_cost`
— into the engine's final candidate score at a bounded, audited, fail-safe weight
that (a) is independent of the graph, (b) falls back to deterministic when the
model is unavailable/stale/invalid, and (c) can create a bounded neural probe via
the existing `entry_mode="probe"`. Prerequisite satisfied: the node already
distinguishes absolute economics from excess ranking, so the blend has a correct
absolute edge to consume.

---

```
STARTING CHECKPOINT ......... 476731e (tag nn-repair-pre-dual-targets-20260715)
B1 TARGET MIGRATION ......... done — absolute + excess targets (14ac547, bf94a54)
B3 FEATURE SEPARATION ....... done — 24 temporal / 20 context, disjoint (6124eed)
B4 CHECKPOINT/CALIBRATION ... done — schema v6, dual calibration, q50 bias fixed (e37c5dc)
B4 PERSISTENCE .............. done — model_forecasts_v2, dual realized (38987c3)
B4 INFERENCE/NODE SEMANTICS . done — absolute edge trades, excess ranks (5d1fe68)
TEST RESULTS ................ 263 passed; diff --check clean
COVERAGE .................... ml 100%, node 93%, neural.py 71%, graph 82%
SMOKE TEST .................. green end-to-end (train→…→node), synthetic only
LOCAL COMMITS ............... 8 stage commits, none pushed
WHAT REMAINS UNPROVEN ....... neural predictive edge; survivorship; holding adapters
STAGE C READINESS ........... ready — bounded graph-independent blend + probe next
```

---

# Stage C addendum (2026-07-16) — direct blend + exploration probe

**This section documents SOFTWARE validation only. Nothing here is evidence of
predictive performance.** The model has no schema-6 champion yet; measured
edge, calibration quality, and incremental-vs-deterministic results remain
entirely unproven until shadow forecasts mature and the Stage F policy
comparison runs.

## Commits
| Commit | Sprint | Tests |
|--------|--------|------:|
| `d017e00` | C1 direct bounded neural blend (graph-independent) | 270 |
| `39ddd7b` | C2 exploration probe + C1 cached-forecast audit fixes | 284 |

## C1 audit outcome (fixed in `39ddd7b`)
The C1 stash had three real defects, all confirmed and closed:
1. not cleared on entry — an offline cycle or a raised exception preserved the
   previous cycle's forecasts → now cleared before anything can raise, and
   stamped with the cycle `as_of`;
2. no cycle identity — engine now discards a stash whose `as_of` stamp
   mismatches, and `policy._valid_forecast` re-validates every forecast
   (typed, current `as_of`, reported model id, current feature schema) —
   stale/foreign/malformed forecasts are inert and counted in the audit;
3. inference count unproven — now asserted: offline/replay cycles run ZERO
   live inference; online cycles exactly one (engine consumes only the stash;
   it has no `predict_today` call site).

## Probe limits (all enforced, all tested)
one probe max (durable `positions.entry_mode`, additive migration) · vetted
candidates only, never fabricated from a forecast · not held / not normally
selected · absolute edge after 0.16% round-trip cost ≥ 0.75% · P(abs edge)
≥ 0.57 · abs q90−q10 ≤ 0.15 · size = normal position × 0.25, applied exactly
once in `portfolio.construct()` · capped by 20% of equity and remaining cash
headroom · one dedicated slot beyond the deterministic batch but inside the
global position cap · governor retains full authority (resizes or
hard-rejects a probe like any order) · neural failure ⇒ no probe, no blend,
deterministic behaviour, exits unaffected.

---

# Sprint D addendum (2026-07-16) — explicit model lifecycle

Commit: **`ec13199`** — 297 tests pass (284 → 297). Software validation only;
no predictive claim.

## Lifecycle state machine (authoritative field: `lifecycle_state`)
```
training ──▶ validation_candidate ──▶ validation_winner ──▶ sealed_candidate
                                                                  │ offline gates
                                                                  ▼
                     champion ◀── production_candidate ◀── experimental_live
                        │           ▲ forward shadow        (BOUNDED permitted
                        │           │ gates re-confirmed     blend, persisted)
                        ▼           │
                     retired    rejected / incompatible (terminal side states)
```
Legacy `status` is now only a projection (champion/incompatible/retired map
through; everything else reads 'challenger') so existing queries/APIs work.

Key properties, each with a test:
- promotion evaluates ALL eligible finalists, ranked deterministically
  (metric desc → created_at asc → id): a newer weak model cannot hide an
  older qualified one;
- validation-only rows are never finalists; holding models additionally need
  out-of-sample shadow observations at BOTH horizons (the validation-only
  auto-promotion is removed);
- champion swaps validate activation first and retire the predecessor in the
  same transaction — a failed activation leaves it intact; concurrent
  attempts converge to exactly one champion;
- every transition persists prior/new state, reason, evidence, permitted
  blend, parent, and hashes in `model_transitions`;
- the old `promote_stage1` holes are closed for BOTH the TCN and the graph:
  offline validity now earns `experimental_live` (bounded blend for the TCN,
  zero live blend for the graph) — full championship requires forward shadow
  evidence; a graph activates only against the exact TCN it was trained with,
  with fail-closed cohort utility evidence.
