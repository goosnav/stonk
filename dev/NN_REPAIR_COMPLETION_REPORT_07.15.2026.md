# NN Repair — Completion Report (2026-07-17)

Branch: `experimental_NN_repair` (from `experimental_NN`). All commits LOCAL,
nothing pushed, no broker access, no live runtime/DB mutation. Safety tag
`nn-repair-pre-dual-targets-20260715` = `476731e`. Suite: **322 passed**.

Everything here is **software validation. No predictive edge is claimed
anywhere** — no schema-6 champion exists yet, so no learned influence has
actually fired.

## 1. Confirmed original defects (all verified against code, then fixed)
1. Neural events ignored by the evidence ensemble; the TCN's only route to a
   trade was the unproven graph, which all-or-nothing-blocked ("BLOCKED:
   GRAPH"). → Stage C direct blend, graph-independent.
2. Excess return read as absolute expected return in the node (a stock down
   5% vs SPY down 10% scored as "long +5%"). → dual-target contract; node
   keys on absolute-after-cost.
3. "60-session" TCN had a 15-session receptive field (d=1,2,4). → 5 blocks,
   field 63.
4. All 44 features fed both branches (annual fundamentals ×60 rows through
   the conv stack). → 24 temporal / 20 context split, disjoint + tested.
5. q50 calibration offset hard-coded to 0. → median-bias corrected,
   validation-only, dual-family.
6. Walk-forward folds shared a boundary session and touched the sealed
   block. → half-open purged `_fold_windows`.
7. Graph promotion gate compounded overlapping 21-day labels as daily
   returns. → all-21-staggered-offset non-overlapping cohorts, median
   aggregated, fail-closed (`utility_evidence`).
8. Promotion evaluated only the newest challenger; `challenger` conflated
   ~4 lifecycle stages; stage-1 promotion gave 10% live blend from history
   alone (TCN and graph). → explicit lifecycle + all-finalists ranking;
   offline validity earns only bounded `experimental_live`.
9. Holding models auto-promoted from validation-only metrics. → blocked;
   promotion needs OOS shadow evidence at both horizons.
10. Trading cycle guarded by a process-local `threading.Lock`. → SQLite
    cross-process lease + pre-placement fencing.
11. `cfg.data["universe"]["symbols"]` mutated during cycles (4 sites). →
    cycle-local symbols; static test enforces zero mutation sites.
12. Approved orders revalidated but never repriced (stale limit hours old).
    → fresh quote → 3% drift gate → recompute → governor re-review → row
    write-back; live-no-quote defers fail-closed.
13. `advanced_override` turned every dangerous config check into a warning.
    → removed; scoped expiring per-parameter `risk_exceptions`; leverage
    never exceptable; `max_equity` voids buys at the governor.

## 2. Rejected / revised hypotheses from the analysis
- "Graph top-decile pooled across dates": the portfolio series was already
  per-date; only the `net_alpha` diagnostic pooled. Downgraded, aligned.
- `forecast.py` was not a neural contract (it's the analog-bootstrap interval
  module); the typed contract was net-new (`ml/schema.py`).
- The backtester needed a policy layer, not a rebuild — it already ran the
  live `run_cycle` per session in an isolated DB.
- Survivorship bias (#19) confirmed but is a data-acquisition problem;
  labeled, not code-fixed.

## 3. Architecture before → after
Before: excess-only 3-block TCN whose output reached trading only through an
untrained graph; one overloaded `challenger` status; thread-local trading
lock; global config bypass; stale approval placement.

After:
```
point-in-time dataset (absolute + excess targets, cost threshold persisted)
  → dual-branch causal TCN (24 temporal → d=1,2,4,8,16 conv · 20 context →
    MLP · fused → 4 quantile + 4 probability heads)
  → validation-only dual-family calibration → schema-6 checkpoint (full
    provenance hashes) → lifecycle (validation → sealed → experimental_live
    → production_candidate → champion; all-finalists deterministic ranking)
  → typed NeuralForecast {absolute,excess}×{q10,q50,q90} + 2 probabilities
  → BOUNDED direct blend (1−b)·deterministic + b·neural, b∈[0.05,0.40],
    graph-independent, lifecycle-capped, fail-closed validation
  → optional single exploration probe (×0.25, budget-capped, governor-ruled)
  → deterministic governor (unchanged authority) → broker review
```
The graph is demoted to a competing meta-model: it activates only against its
exact training TCN, with fail-closed utility evidence, and owns the learned
pathway only as champion (direct blend stands down — no double count).

## 4. Files changed / DB migrations (all additive)
Code: `specforge/{neural,graph,engine,execution,risk,config,store,data,
portfolio,models,research}.py`, `specforge/nodes/neural.py`,
`specforge/ml/{__init__,schema,targets,policy,lifecycle}.py` (new package),
`specforge/backtest.py`, `configs/{default,live}.yaml`,
`tests/{test_nn_repair,test_neural,test_risk,test_hypothesis,test_pipeline}.py`.

Migrations (additive; no history rewritten/deleted): `model_forecasts_v2`,
`model_transitions`, `positions.entry_mode`, `model_runs.{lifecycle_state,
permitted_blend}`, `graph_versions.{lifecycle_state,temporal_model_id}`,
lifecycle backfill classification, kv-based `lease:*` rows.

## 5. Contracts
- **Model/forecast**: `NeuralForecast` (ml/schema.py) — absolute AND excess
  q10/q50/q90, `probability_absolute_edge_positive` (threshold = 0.16%
  round-trip cost), `probability_excess_positive`, model/dataset/feature
  provenance; fail-loud validation, never clamps. Long eligibility =
  absolute edge after cost, never excess.
- **Point-in-time**: targets strictly forward, embargoed splits, SEC facts
  keyed to filing dates. **The full event_at/known_at contract (news
  ingestion/classification time) is NOT implemented** — see §10.
- **Lifecycle**: diagram + rules in the Sprint D addendum of
  `dev/NN_REPAIR_STAGE_B_REPORT_07.15.2026.md`; every transition persisted
  with evidence in `model_transitions`.
- **Exposure limits**: blend ≤ min(config max 0.40, lifecycle permitted);
  probe ≤ normal×0.25, ≤20% equity, 1 open max, governor-ruled. All audited
  per candidate/cycle.

## 6. Evaluation methodology + policy comparison
Forecast metrics: per-date rank IC, pinball, coverage, directional accuracy,
after-cost per-session top-decile alpha; absolute and excess reported
separately (shadow + tournament). Portfolio utility: non-overlapping
21-session cohorts over all 21 staggered offsets, median-aggregated,
fail-closed below 3 cohorts × 8 offsets.

`backtest.compare_policies(cfg)` runs deterministic / fixed_blend /
neural_only through the SAME engine, same sessions, same costs, same
governor, in isolated per-policy DBs, and reports each policy plus
incremental-vs-deterministic deltas (window divergence hard-errors).
**Current honest result: the policies coincide** — no champion exists, so
offline cycles produce no forecasts (asserted in the test suite as a
documented identity, not hidden). The instrument is ready; the evidence is
not yet.

## 7. Test / coverage / complexity results
- `.venv/bin/python -m pytest -q` → **322 passed** (baseline 204 at branch
  point; ~118 added by this effort; every sprint landed green).
- Coverage (`coverage run --source=specforge -m pytest` over the neural/
  graph/risk suites): ml/schema 100%, ml/targets 100%, ml/policy 92%,
  ml/lifecycle 93%, nodes/neural 96%, risk 92%, graph 87%, neural 73%,
  execution 63%, config 66%. Whole-suite `pytest --cov` remains blocked by a
  coverage-tracer × numpy reimport incompatibility in this env.
- Complexity: radon is not installed in the venv — **UNVERIFIED by tool**.
  Structurally: `_fold_windows`, `_cohort_returns`/`_staggered_portfolio_
  metrics`, `_graph_offline_gate`, `ml/policy.py`, `ml/lifecycle.py` were
  extracted as small pure units from the former F-graded monoliths;
  `engine._run_cycle` and `neural.train_challenger` remain large (§10).

## 8. Reproduction commands
```bash
.venv/bin/python -m pytest -q                                    # full suite
.venv/bin/python -m pytest tests/test_nn_repair.py -q            # repair battery
.venv/bin/python - <<'PY'                                        # policy comparison
from specforge.config import load_config
from specforge.backtest import compare_policies
print(compare_policies(load_config("paper"), years=3)["incremental_vs_deterministic"])
PY
# training (bounded, writes a schema-6 challenger; research plane calls this):
.venv/bin/python -c "from specforge.config import load_config; from specforge.store import Store; from specforge import neural; cfg=load_config('paper'); print(neural.train_challenger(cfg, Store(cfg.get('db_path')), max_seconds=600))"
```
Launch (unchanged): paper `./run.sh` / live `./scripts/restart_live.sh`
(live picks up the migrated `risk_exceptions` config on next restart).

## 9. Remaining blockers before merge to `main`
1. Train a schema-6 champion and let it walk the lifecycle ramp; until then
   neural influence is structurally 0 and the policy comparison is empty.
2. Backtest-time neural replay: historical OOS forecasts are still persisted
   to v1 (excess-only); training must write `model_forecasts_v2` rows so
   `compare_policies` can replay real dual-family forecasts offline.
3. `known_at` point-in-time contract for news/AI features (#4 of the brief).
4. Date-grouped batch sampler + memory-mapped panel (#5/#6) — the 12k-window
   cap still limits temporal density.
5. Seed ensembles (#9) and holding adapters replacing full fine-tunes (#10 —
   currently blocked-from-promotion, not yet replaced).
6. Local API session token; dependency lockfile + fingerprint in checkpoints.
7. Full `neural.py` → `ml/` decomposition and `_run_cycle` phase split
   (partial: schema/targets/policy/lifecycle already live in `ml/`).
8. Branch remains a platform-sized diff vs `main` — split for review per the
   analysis before merging.

## 10. Known limitations
Options approvals keep the revalidate-only path (no fresh chain source),
visibly audited. A live approval whose symbol leaves the quoted universe
defers until TTL expiry (visible each cycle). Survivor-universe history is
labeled, not fixed. `training`/`validation_winner`/`rejected` states exist
but the tournament doesn't yet stamp `validation_winner`.

---
```
WHAT WAS BROKEN ... the model predicted excess but traded it as absolute; saw 15
                    of 60 sessions; could only act through an untrained graph
                    that blocked everything; promotion, folds, and the utility
                    metric were statistically invalid; the cycle lock, config
                    mutation, stale approvals, and one global risk bypass were
                    real-money operational holes.
WHAT WAS CHANGED .. dual absolute/excess targets + typed forecast contract;
                    63-session dual-branch TCN; validation-only dual
                    calibration; schema-6 provenance checkpoints; v2 forecast
                    persistence; explicit lifecycle with all-finalists
                    promotion; bounded graph-independent blend + one governed
                    exploration probe; cross-process lease + fencing; immutable
                    cycle config; approval repricing; scoped risk exceptions;
                    policy-comparison harness. 13 local commits, each green.
WHAT NOW WORKS .... the honest machinery end to end: train → checkpoint →
                    lifecycle → bounded influence → governor, with fail-closed
                    validation at every seam and 322 passing tests.
WHAT REMAINS
UNPROVEN .......... any predictive edge whatsoever. No champion exists; blend
                    and probe have never fired; the policy comparison currently
                    returns identical books by construction.
TEST RESULTS ...... 322 passed, 0 failed, diff-check clean (204 at baseline).
MODEL RESULTS ..... none claimed — by design, until a schema-6 champion earns
                    the ramp on forward evidence.
SAFE NEXT STEP .... let the research plane train schema-6 challengers and
                    accumulate shadow forecasts; wire v2 OOS persistence into
                    training so compare_policies becomes a real experiment
                    before any blend fires live.
```
