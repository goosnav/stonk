# NN Repair — Stage B Report (2026-07-15)

Branch: `experimental_NN_repair`. No push, no live restart, no DB/checkpoint/
broker interaction performed.

This report is deliberately honest about implemented-vs-designed. Stage B was
specified as four sub-stages. **B2 (receptive field) is implemented, tested, and
committed.** B1/B3/B4 (the absolute-return target contract, temporal/context
split, and its inference/node/shadow ripple) are **fully designed here but not
yet landed** — they are one coupled migration of the live model's target
contract that crosses the checkpoint schema, the `shadow_forecasts` table, the
graph's stored neural inputs, and five test modules. Per the brief's own
guardrails ("do not make one enormous unverified rewrite", "keep the branch
runnable after every sub-stage", "do not describe Stage B as successful merely
because the code executes"), that migration is scoped as the next focused unit
rather than crammed in unverified.

---

## 1. Stage A audit findings & corrections

Audited the three Stage-A fixes; all held up, and I hardened two per the
continuation brief:

- **`NeuralForecast` (specforge/ml/schema.py)** — was already fail-loud (raised
  on unordered quantiles / out-of-range probabilities), not silently clamping.
  Hardened to also reject non-finite values (NaN/±inf) explicitly, require a
  supported positive horizon (`SUPPORTED_HORIZONS = (5, 21)`), and require
  non-empty provenance (`model_id`, `dataset_manifest_id`, `feature_schema_hash`).
  Calibration stays outside the contract (applied to raw output before
  construction). No silent clamping anywhere.
- **`_fold_windows` (specforge/neural.py)** — confirmed half-open `[ts, te)`,
  index-based (not calendar), embargo gap `ts - train_pos = embargo + 1` so the
  last training label ends at `ts-1` (no leak), and the last fold stops at
  `sealed` (no sealed-block leak). Added proofs as tests.
- **Graph portfolio metric** — the Stage-A single-phase `dates[::21]` was
  insufficient (one arbitrary alignment). **Corrected**: now evaluates ALL 21
  staggered non-overlapping cohort offsets, aggregates by median, and reports
  `worst_offset_sharpe` / `worst_offset_drawdown` / `n_valid_offsets` /
  `cohorts_per_offset`. **Fails closed** (`portfolio_utility = -1.0`,
  `utility_evidence="insufficient"`) below `_MIN_COHORTS_PER_OFFSET=3` /
  `_MIN_VALID_OFFSETS=8`, so a `> 0` promotion gate cannot pass on thin
  evidence. Computed once over the pooled OOS span (per-fold spans are too short
  for enough independent 21-session cohorts). Cost deducted exactly once per
  cohort. Explicitly labeled an **interim forecast-policy diagnostic**, not a
  production equity simulation (superseded by the Stage-F policy backtester).

**Stage A checkpoint commit:** `3b6c4f62adc244fd3a3ab3e9d6f8cc57f4c69090`

## 2. Stage B2 — implemented

**Before:** 3 causal blocks, dilations 1,2,4, kernel 3 → receptive field
`1 + 2·(1+2+4) = 15` sessions. A "60-session" model that saw only the last 15.

**After:** 5 causal blocks, dilations 1,2,4,8,16 → `1 + 2·(1+2+4+8+16) = 63`
sessions ≥ the 60-session window. Channels 32, dropout 0.1, residual + GELU
retained. `ARCHITECTURE_HASH` bumped `tcn-v5…d1,2,4…` → `tcn-v6…d1,2,4,8,16…`.

**Checkpoint compatibility:** old checkpoints fail the `architecture_hash` check
in `_load_checked`/`refresh_compatibility` and are marked
`incompatibility_reason="architecture mismatch"` — clean rejection, never a
`load_state_dict` crash. Verified by the existing green suite.

**Smoke (end-to-end, already in suite):** `test_neural.py::
test_training_writes_challenger_not_champion` runs `build_dataset →
train_challenger(max_seconds=2) → _save → _load_checked` on the new
architecture; green. So dataset construction, a bounded train, checkpoint
create, and checkpoint reload are all exercised on the 5-block model with no
NaNs and ordered quantiles (asserted by `test_tcn_quantiles_are_ordered` and
`test_validation_calibration_preserves_quantile_order_and_probability_bounds`).

**Stage B2 commit:** `72e0f3a9af29048c13426fc4479fd35f690b7b23`

## 3. Files changed (this session, both stages)
- `specforge/ml/__init__.py`, `specforge/ml/schema.py` — new `NeuralForecast`.
- `specforge/neural.py` — `_fold_windows` (half-open folds); 5-block TCN; hash.
- `specforge/graph.py` — `_cohort_returns` / `_offset_metrics` /
  `_staggered_portfolio_metrics`; pooled OOS metric in `walk_forward_fit`.
- `tests/test_nn_repair.py` — 22 checks (schema edge cases, folds, staggered
  metric, receptive field).
- `dev/NN_REPAIR_IMPLEMENTATION_PLAN_07.15.2026.md`, this report.

## 4. Test results
`.venv/bin/python -m pytest -q` → **226 passed** (204 baseline + 22 new), ~22s.
`git diff --check` clean.

---

## 5. B1 / B3 / B4 — implementation-ready design (NOT yet landed)

The semantic core: the model must emit **absolute** and **excess** distributions
so the node keys long-eligibility off absolute-return-after-cost, never excess.

### Target contract (B1)
```python
# specforge/neural.py
TARGETS = (("absolute", 5), ("absolute", 21), ("excess", 5), ("excess", 21))
COST_THRESHOLD = 0.0016
PROB_THRESHOLDS = (COST_THRESHOLD, COST_THRESHOLD, 0.0, 0.0)  # abs>cost ; excess>0
TARGET_HASH = sha256(repr(TARGETS)+repr(PROB_THRESHOLDS))[:16]
```
`build_dataset`: `Y` becomes 4 columns in `TARGETS` order —
`abs_h = c.shift(-h)/c - 1`; `exc_h = abs_h - (spy.shift(-h)/spy - 1)`.
`target_scale` shape `(4,)`. Return `targets=TARGETS` (replace `horizons`).

### Architecture (B3, cheap via in-model slicing — keeps `X` storage as `(n,60,44)`)
```python
TEMPORAL_FEATURES = [r1,range,gap,volume_z,vol21,rsi14,atr14,breakout60,
  sma50_d,sma200_d,spy_r1,spy_r21,sector_relative_r21,vix,vix9d,vix3m,vix6m,
  vvix,vix_curve_9d_3m,vix_curve_1m_3m,implied_realized_spread,hyg_r21,tlt_r21,
  vol_context_missing]                      # 24 daily-varying
CONTEXT_FEATURES  = [valuation(+missing),event_proximity(+missing),
  revenue_growth(+missing),operating_margin(+missing),fcf_margin(+missing),
  debt_assets(+missing),dilution(+missing),accruals(+missing),
  liquidity(+missing),news_sentiment(+missing)]   # 20 point-in-time
TEMPORAL_IDX / CONTEXT_IDX = [FEATURES.index(f) for f in each]
```
Model forward:
```python
temporal = self.blocks(x[:, :, TEMPORAL_IDX].transpose(1,2))[..., -1]   # (n,32)
context  = self.context_mlp(x[:, -1, CONTEXT_IDX])                      # (n,32)
z        = self.fusion(cat([temporal, context], -1))                    # (n,64)
# 4 quantile heads Linear(64,3) + 4 probability heads Linear(64,1), TARGETS order
```
This makes B3's "snapshot features never enter the temporal encoder" true by
construction (context columns are sliced out of the conv branch and read only
from the last row). Bump `ARCHITECTURE_HASH` → v7, `MODEL_SCHEMA` → 6.

### Ripple (B4) — every consumer, keyed by `TARGETS` not `horizons`
- `_make_model(n_features, n_targets=4)`; callers pass `len(TARGETS)`.
- `_load_checked`: `len(target_scale)==len(TARGETS)`; `_make_model(...,len(TARGETS))`;
  payload stores `targets`, `target_hash`. Old checkpoints already rejected on
  `architecture_hash`/`schema_version` (test #9).
- training BCE label: `(Yraw > PROB_THRESHOLDS).float()` (broadcast).
- `_calibration`: `observed = (truth[:,i] > PROB_THRESHOLDS[i]).mean()`.
- `_metrics`: key by target name (`"absolute_5"`, `"excess_21"`, …).
- `_selection_score`: rank IC from the **excess** targets (ranking), edge from
  the **absolute** targets (economics); rekey.
- `holding_gate_passed`: rekey to absolute targets (fail-closed if missing).
- `_baseline_metrics`: 4-target zero/momentum baselines.
- `predict_today` / `predict_run`: build `{sym: {"5": NeuralForecast,
  "21": NeuralForecast}}` — absolute_q* from absolute slots, excess_q* from
  excess slots, both probabilities from the aligned prob heads.
- `shadow_forecasts` table: add `realized_absolute` + `kind` columns (additive
  migration); `shadow_metrics` reports abs & excess coverage/IC separately.
- graph stored neural inputs (`graph.py:705-709`): store absolute q50 for the
  economic edge, excess q50 for ranking.
- `nodes/neural.py`: `direction="long"` and `expected_return = f.absolute_q50`
  **only if** `f.absolute_edge_after_cost(cost) > min_edge` and
  `f.probability_absolute_positive >= min_p`; excess merely confirms. Evidence
  string carries absolute median, excess median, and modeled cost.
- model card `describe`/`architecture`: report two-branch + `TARGETS`.

### Stage B acceptance tests to add (12)
Absolute & excess separately constructed; no future price in a window; positive
excess + negative absolute ⇒ no long; all four quantile groups ordered; both
prob heads bounded; earliest of 60 sessions moves output (done, B2); snapshot
feature at an earlier timestep does NOT move output (temporal isolation);
calibration uses validation only; old checkpoint rejected via metadata; model
card truthful; model failure ⇒ deterministic fallback (engine); exits
independent of neural availability (engine).

### Bounded smoke (B4)
Reuse the `test_neural.py` fixture store: `build_dataset → train_challenger
(max_seconds≈3) → _load_checked → predict_run` on synthetic data; assert typed
`NeuralForecast` out, finite, ordered, no production-DB mutation, no broker.

## 6. Known limitations / unproven
- **B1/B3/B4 not implemented** — the semantic bug (excess treated as absolute in
  `nodes/neural.py:37-47`) is still live in code; only the *contract* to fix it
  (`NeuralForecast`) exists. Neural influence remains 0 regardless.
- No predictive-value claim is made for any model. B2 fixes an architectural
  defect (receptive field); it does not demonstrate edge.
- Survivorship bias (analysis #19) unaddressed — a data problem, out of scope.

## 7. Exact next step for Stage C
Stage C (direct bounded neural blend, decoupled from the graph) depends on B1
(absolute forecasts). Sequence: finish B1/B3/B4 per §5 → then add
`neural.experimental_blend` and blend `neural_score` (from
`NeuralForecast.absolute_edge_after_cost`) into the engine's final candidate
score at a bounded, audited, fail-safe weight that never gates on the graph and
falls back to deterministic when neural is unavailable.

---

```
STAGE A AUDIT ........ passed; NeuralForecast hardened (finite/horizon/provenance,
                       no clamp); graph metric upgraded to all-21-offsets, fail-closed
STAGE A COMMIT ....... 3b6c4f62adc244fd3a3ab3e9d6f8cc57f4c69090
STAGE B IMPLEMENTED .. B2 only: full 60-session receptive field (5 dilated blocks);
                       B1/B3/B4 fully designed (§5), not landed (coupled schema migration)
TEST RESULTS ......... 226 passed (204 baseline + 22 new); diff --check clean
SMOKE TEST ........... train→checkpoint→reload green on new architecture (test_neural.py)
CHECKPOINT COMPAT .... old checkpoints marked incompatible via architecture_hash, no crash
WHAT REMAINS UNPROVEN  neural predictive edge; B1 semantic fix still absent in node code
STAGE C NEXT STEP .... land B1/B3/B4 (§5), then bounded direct neural blend decoupled
                       from the graph, deterministic fallback preserved
```
