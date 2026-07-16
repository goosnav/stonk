# NN Repair — Implementation Plan (2026-07-15)

Branch: `experimental_NN` (repair work on `experimental_NN_repair`, cut from it).
Baseline at authoring: **204 tests pass** (`.venv/bin/python -m pytest -q`, 21s).
Author: principal-engineer repair pass per `dev/PROBLEM_ANALYSIS_07.15.2026.md`.

Goal: give the neural model **real, bounded, honest, reversible** influence that
does **not** depend on the analog graph, without weakening the deterministic
governor. Every stage keeps the branch runnable and the suite green.

---

## A. Verification of the analysis (against actual code)

Each claim was checked against the implementation. `file:line` is the evidence.
"CONFIRMED" = defect is real as described. "REVISED" = real but the analysis
mis-stated a detail; the corrected statement follows.

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | Neural events ignored by evidence ensemble | **CONFIRMED** | `ensemble.NODE_FAMILY` has no `"neural"` key (`ensemble.py:20-35`); events whose `node_id` isn't in it are skipped (`ensemble.py:113-116`). Neural reaches trades **only** via the graph. |
| 2 | Excess-return prediction treated as absolute expected return | **CONFIRMED** | Target is excess: `(c.shift(-h)/c-1) - benchmark` (`neural.py:369`). Node sets `direction = "long" if pred>0` and `expected_return = pred` from the excess q50 (`nodes/neural.py:37-47`). No absolute head exists; shadow tracks `realized_excess` only (`neural.py:1323`). |
| 3 | TCN receptive field ~15 sessions, not 60 | **CONFIRMED** | Blocks dilations 1,2,4, kernel 3 (`neural.py:449-451`) → R = 1+2·(1+2+4) = **15**. Input window is 60 (`neural.py:369`, config `input_sessions:60`). |
| 4 | Static/annual features repeated through every timestep | **CONFIRMED** | All 44 features go into one tensor; the `context` branch also just re-reads the **last row** (`neural.py:452,461`). No temporal/context split. |
| 5 | News joined by publication time, not known-at | **CONFIRMED (structural)** | News features group by `substr(published_at,1,10)`; no `known_at`/`classified_at` gate. (SEC path is closer — keyed to filing date.) |
| 6 | 12,000-window cap → few windows/symbol | **CONFIRMED** | `SAFE_MAX_TRAINING_WINDOWS=12_000` (`neural.py:53`); `per_symbol_cap = window_limit // len(symbols)` (`neural.py:353`) → **8/symbol at 1500 symbols**. |
| 7 | Random batching weakens cross-sectional rank loss | **CONFIRMED** | Windows built per-symbol then shuffled; `_rank_loss` compares whatever lands in a batch (`neural.py:494-505`), not a date cross-section. |
| 8 | Graph portfolio metric compounds overlapping 21d labels | **CONFIRMED** | `curve = np.cumprod(1 + daily)` where `daily` is per-**every**-date top-decile 21d excess (`graph.py:514-527`). Adjacent points overlap ~20/21. Sharpe std is on autocorrelated samples → inflated. Gates promotion via `portfolio_utility`/`oos_sharpe` (`graph.py:543-546`). |
| 9 | Graph top-decile pooled across dates | **REVISED** | The **portfolio** series is already per-date (`graph.py:515-520`). Only the `net_alpha_{h}d` **diagnostic** pools across the fold (`graph.py:511-513`). Real bug is #8's compounding, not pooling; will still align `net_alpha` to per-date for consistency. |
| 10 | Adjacent walk-forward folds share a boundary date | **CONFIRMED** | Inclusive mask `dates<=unique[test_end_pos]` (`neural.py:705`) and `test_end_pos == next fold's test_start_pos` → boundary session in two folds. |
| 11 | Generic `challenger` conflates lifecycle stages | **CONFIRMED** | `status` is only `champion`/`challenger`/`incompatible` (`neural.py:1037,1217`); no validation/sealed/forward-shadow distinction. |
| 12 | Promotion evaluates only newest challenger | **CONFIRMED** | `...status='challenger' ORDER BY created_at DESC LIMIT 1` (`neural.py:1412`); graph identical (`graph.py:609-611`). |
| 13 | Per-holding models promote from validation-only | **CONFIRMED** | `holding_gate_passed(metrics)` runs on validation metrics and auto-`promote()`s (`neural.py:1059-1063,1112-1120`); no sealed holding trial. |
| 14 | Full per-symbol TCN fine-tuning unjustified | **CONFIRMED** | Holding path clones the whole global net and fine-tunes it (`neural.py:832,997`) on one symbol's ~1250 overlapping bars. |
| 15 | Process-local trading lock | **CONFIRMED** | `_CYCLE_LOCK = threading.Lock()` (`engine.py:29,49,56`). Cannot fence daemon vs CLI vs restart. |
| 16 | Mutable config leaks across cycles | **CONFIRMED** | `cfg.data["universe"]["symbols"] = symbols` (`engine.py:81,93`); `MarketContext.universe` reads the same object. |
| 17 | `advanced_override` too broad | **CONFIRMED** | One flag turns **every** validator error into a warning (`config.py:154-158`); `live.yaml:18` sets it true. |
| 18 | Approved orders revalidated but not repriced | **CONFIRMED** (execution path) | Approval re-checks funds/risk/data-age but keeps original price/qty (per `execution.py` approval flow; to re-cite exact lines when touched). |
| 19 | Survivorship bias in historical training | **CONFIRMED (data)** | Universe is current-survivor membership; no delisted/PIT security master. Label as "current-survivor" — data problem, out of code scope. |
| 20 | Backtester can't evaluate neural/graph via live path | **PARTIAL** | `backtest.py` reuses the live cycle, but there is no policy-swap (deterministic vs neural vs blend) and no per-policy isolated book. |

Extra confirmed facts that shape the fix:
- **The graph is the sole neural pathway AND its gate is all-or-nothing.** Engine
  removes the neural node from the ensemble whenever the graph blend is active
  (`engine.py:273-274`), and zeroes the blend if any required graph node is
  `blocked` (`engine.py:284-291`) — this is the "BLOCKED: GRAPH" deadlock.
- **Checkpoint safety net exists.** `MODEL_SCHEMA` + `FEATURE_HASH` +
  `ARCHITECTURE_HASH` are checked at load and mismatches return a clean
  incompatibility reason, not a crash (`neural.py:1171-1176`). So architecture
  changes land safely **iff** `ARCHITECTURE_HASH` is bumped in the same change.
- **`specforge/forecast.py` is NOT the `NeuralForecast` contract** the analysis
  sketches — it's the analog-bootstrap interval module for `TradeCandidate`.
  The typed neural contract is net-new (goes in `specforge/ml/schema.py`).

## Rejected / revised hypotheses
- #9 (pooled top-decile): the *portfolio* metric is already per-date; only a
  diagnostic pools. Down-graded from "critical" to "consistency cleanup."
- #20: the backtester foundation is sound; needs a policy interface, not a rebuild.
- Survivorship (#19) is a **data-acquisition** problem, not a code defect —
  addressed by labeling, not by this branch's code.

---

## B. Staged implementation (each stage: runnable + green + one runnable check)

Ordering follows the brief's Phase 1→5 but front-loads the three
architecture-independent fixes so the branch improves immediately without a
checkpoint-schema churn.

### Stage A — independent honest-math fixes (no arch/schema change) ← THIS SESSION
- **A1** `specforge/ml/` package + `ml/schema.py`: frozen `NeuralForecast`
  contract (absolute + excess quantiles, both probabilities, provenance ids).
  Additive; nothing consumes it yet. *Check: `test_ml_schema.py` — quantile
  ordering + immutability.*
- **A2** Fold boundary → half-open (`neural.py:701-705`): `< test_end_pos`.
  *Check: adjacent folds share no test date.*
- **A3** Graph metric (`graph.py:514-527`): extract `_cohort_returns` /
  `_portfolio_metrics`; compound **non-overlapping 21-session cohorts** only;
  Sharpe/DD from independent cohorts. Align `net_alpha` to per-date.
  *Check: overlapping labels are not compounded (curve length == #cohorts).*

### Stage B — correct neural semantics + real receptive field (one schema bump)
Bump `ARCHITECTURE_HASH` + `MODEL_SCHEMA` once; old checkpoints cleanly
incompatible.
- Receptive field: 5 blocks, dilations 1,2,4,8,16 → R=63. *Check: earliest
  session influences output.*
- Absolute **and** excess heads (5d/21d each) + `P(abs>0)`, `P(excess>0)`.
- Temporal branch vs snapshot/context branch with independent encoders + fusion.
- Median calibration (validation residual offset) + probability calibration.
- `predict_today` emits `NeuralForecast`; `nodes/neural.py` consumes absolute
  edge after cost, not raw excess. *Check: +excess/−absolute ⇒ no positive edge.*

### Stage C — direct bounded neural influence (decoupled from graph)
- Config: `neural.experimental_blend/min_blend/max_blend`.
- Engine: `final = (1-b)·deterministic + b·neural_score`, `b` gated by a
  **neural-only** integrity/lifecycle check (not the graph), bounded, audited,
  attributable; **0 and deterministic fallback** when neural unavailable/stale/
  invalid. Graph becomes a separate competing meta-model, never a gate on neural.
- Bounded neural exploration sleeve via existing `entry_mode="probe"`.
  *Checks: blend applied exactly once; not double-counted with graph; neural
  failure ⇒ deterministic; neural failure cannot block exits; probe ≤ its cap.*

### Stage D — lifecycle states + promote-all-finalists
- Explicit `lifecycle_state` (training→validation_candidate→validation_winner→
  sealed_candidate→experimental_live→champion/rejected/incompatible/retired) +
  provenance columns (additive migration). Promotion evaluates **all** eligible
  finalists. Holding adapters cannot promote from validation-only.
  *Checks: validation-only can't promote; newest can't hide older finalist.*

### Stage E — operational hardening
Cross-process SQLite trading lease (reuse research lease design); immutable
cycle config (pass explicit `symbols` into `MarketContext`); scoped/expiring
`risk_exceptions` replacing `advanced_override` (hard invariants stay
un-overridable); reprice approved orders before placement; per-launch bearer
token for mutating local API routes; pinned deps + fingerprint in checkpoints.

### Stage F — counterfactual books + policy backtest
Policy interface (deterministic / neural-only / fixed-blend / learned) over the
existing same-engine backtester; isolated per-policy book; incremental-vs-
deterministic reporting in the dashboard.

### Stage G — refactor
Split `neural.py` into `ml/{features,dataset,model,losses,calibration,baselines,
training,evaluation,registry,inference,policy}.py`; split `engine._run_cycle`
into explicit phase functions. Real module boundaries + typed contracts, not a
cosmetic split.

---

## C. Honest scope note
This is a multi-session effort. Stage A lands this session (self-contained,
tested, green). Stages B–G are sequenced above and executed in later passes,
suite green after each. Neural live influence stays **0 until a valid schema-B
champion exists AND the operator enables the blend**, so nothing changes live
behavior by default. No fake progress: only stages actually implemented and
tested are reported as done, with fresh command output as evidence.
