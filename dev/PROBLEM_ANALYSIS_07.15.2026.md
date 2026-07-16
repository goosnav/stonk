



# Verdict

`experimental_NN` is an ambitious, unusually thoughtful solo trading-system prototype. The deterministic governor, broker-review boundary, audit trail, point-in-time data design, and fail-silent learned models are legitimate strengths.

It is **not ready to merge into `main`, distribute, or run with meaningful capital**. The largest risks are not ordinary syntax bugs. They are:

1. questionable model-validation mathematics,
2. promotion-state logic,
3. cross-process concurrency,
4. overly broad safety overrides,
5. insufficient test coverage in the most critical modules,
6. severe architectural concentration in a few functions.

The repository’s own release checklist agrees: it explicitly says **“NOT YET CANDIDATE_FOR_ACCEPTANCE.”** fileciteturn36file0L3-L5

## What the Radon result actually means

An average complexity of **B / 6.75 across 733 blocks is acceptable**. The problem is that the worst complexity is concentrated precisely where mistakes are most consequential:

| Hotspot | Complexity | Why it matters |
|---|---:|---|
| `engine._run_cycle` | F / 103 | Entire live trading lifecycle |
| `neural.train_challenger` | F / 92 | Training, validation, tournament, calibration, persistence |
| `ensemble.score` | F / 80 | Production trade selection |
| `graph.fit_weights` | F / 64 | Learned graph construction |
| `portfolio.rebalance_plan` | F / 62 | Forced sales and reallocations |
| `research.run_next` | F / 54 | Durable model/research orchestration |
| `graph.maybe_promote` | F / 42 | Whether learned models affect live capital |
| `neural.maybe_promote` | E / 39 | Champion promotion and rollback |

So the command is useful, but it does **not** prove the software is bad. It identifies the places where state-space explosion, hidden coupling, and incomplete tests are most likely. fileciteturn0file0

# Critical problems

## 1. The graph’s portfolio-validation metric appears mathematically invalid

The graph walk-forward code takes a sequence of **overlapping 21-day forward returns**, generates one observation for nearly every date, and then compounds those observations as though they were sequential, independent portfolio returns. It subsequently derives drawdown, annualized return, and Sharpe from that constructed curve. fileciteturn24file0L17-L30

That is not a valid backtest equity curve. Adjacent 21-day labels overlap approximately 20/21 of their underlying return period. Compounding them daily massively double-counts the same price movement.

Worse, `portfolio_utility` and `oos_sharpe` are used as graph-promotion gates. fileciteturn24file0L170-L178

**Required fix:** replace this with either:

- an actual point-in-time portfolio simulation with position overlap, turnover, cash, costs, and execution timing, or
- non-overlapping 21-session cohorts,
- plus Newey-West-adjusted statistical inference for overlapping labels if you retain them for predictive metrics.

Do not promote any graph using the current utility calculation.

---

## 2. Graph top-decile alpha is pooled across dates

The graph code calculates the 90th-percentile prediction cutoff across the entire fold before computing after-cost alpha. fileciteturn24file0L13-L16

That allows a model to score well through broad market timing rather than daily cross-sectional stock selection. Your neural implementation correctly calculates top-decile selection separately for each session; the graph should use the same definition.

**Required fix:** calculate daily cross-sectional selections, then aggregate daily cohort returns. Use identical metric definitions across the TCN, graph, and production ranker.

---

## 3. Neural walk-forward folds overlap by one date

The neural fold code uses inclusive test masks:

- `test_end_pos = test_start_pos + width`
- current fold includes `<= test_end`
- next fold starts at that same boundary date.

That means one OOS session can appear in two adjacent folds. fileciteturn18file0L161-L168

This is probably a small numerical distortion, but it is a genuine validation defect.

**Fix:** use half-open intervals or set the end position to `test_start_pos + width - 1`.

---

## 4. “Shadow mode” does not fully mean shadow mode

When no champion exists, an offline-approved TCN can be promoted at stage one without forward-shadow evidence. fileciteturn21file0L117-L120

Likewise, the first offline-approved graph can be promoted without satisfying the later forward-shadow threshold. fileciteturn24file0L183-L188

Once both champions exist, activation stage one gives the learned system a **10% live blend immediately**. fileciteturn25file0L40-L66

That conflicts with the stronger claim elsewhere that the learned stack has zero influence until shadow gates pass. In effect:

> sealed historical validation → immediate 10% live influence

That may be defensible as a controlled rollout, but it is not forward-shadow validation.

**Recommended policy:**

- Stage 0: offline-approved, 0% influence.
- Stage 1: at least 30 distinct forward sessions, 0% influence.
- Stage 2: 5–10% influence after forward metrics pass.
- Stage 3: 25%.
- Stage 4: at most 50%.

For a financial system, I would not grant live influence solely from historical model selection.

---

## 5. Promotion considers only the newest challenger

Both promotion mechanisms query the most recently created challenger rather than evaluating all viable challengers.

The TCN implementation selects:

```sql
ORDER BY created_at DESC LIMIT 1
```

and either promotes or shadows that one run. fileciteturn21file0L104-L128

The graph does the same. fileciteturn24file0L136-L143

This creates a state-machine problem: a newly created weak or validation-only challenger can hide an older finalized challenger that has accumulated better forward-shadow evidence.

**Fix:** explicitly maintain tournament states:

- `training`
- `validation_candidate`
- `sealed_finalist`
- `forward_shadow`
- `champion`
- `rejected`
- `retired`

Then evaluate all `forward_shadow` models and deterministically choose the best eligible model. Do not overload `challenger` to mean several different lifecycle stages.

---

## 6. Trading-cycle locking is process-local

`_CYCLE_LOCK` is a Python `threading.Lock`. It prevents two cycles inside the same process, but it cannot stop:

- the daemon and a CLI invocation,
- two app instances,
- a stale process and a restarted process,
- two separately launched schedulers.

fileciteturn13file0L28-L58

The research system already recognizes this problem and uses SQLite leases. The trading cycle needs the same treatment.

**Fix:** add a transactional SQLite lease keyed by mode/account:

```text
trading_cycle_lease:live:<account>
```

Include owner PID, process start identity, acquired time, heartbeat, and expiration. The database lease—not a thread lock—must be authoritative.

This is a real-money P0 issue.

---

## 7. `_run_cycle` mutates the shared configuration object

The engine replaces:

```python
cfg.data["universe"]["symbols"] = symbols
```

when using active-research membership or a hypothesis watchlist. fileciteturn13file0L75-L96

`MarketContext.universe` subsequently reads directly from that same mutable config. fileciteturn16file0L24-L27

This means cycle-local state can survive into later cycles. For example:

- an old active universe can remain after its snapshot disappears;
- a hypothesis-added ticker can remain after the hypothesis is disabled;
- tests can become order-dependent;
- different services holding the same config object may see mutated state.

**Fix:** make `Config` immutable and pass an explicit `symbols` collection to `MarketContext`. Never mutate the loaded configuration during execution.

---

## 8. `advanced_override` is too broad

Live mode sets `advanced_override: true` to support the current tiny account, then raises:

- cycle risk to 40%,
- single-position exposure to 30%,
- total deployment to 98%.

fileciteturn28file0L17-L29

But `advanced_override` does not approve only those three intentional exceptions. It turns every configured dangerous threshold from an error into a warning. The validator currently checks only six conditions. fileciteturn30file0L101-L109

This is a classic safety-interlock design error: one global bypass defeats unrelated future protections.

**Better design:**

```yaml
risk_exceptions:
  max_single_equity_position:
    value: 0.30
    max_equity: 100
    expires: 2026-08-01
    reason: "$50 live-probation account"
```

Hard invariants—no leverage, no shorting, no naked options, absolute dollar caps—should remain impossible to override.

The current `$50` hard cycle cap limits immediate damage, but this configuration should automatically fail once account equity exceeds the probation range.

---

## 9. The local real-money control plane lacks authentication

The application has useful defenses:

- loopback binding,
- trusted-host checks,
- cross-origin mutation rejection.

But the repository itself identifies the missing control: a **per-launch unguessable token for privileged endpoints**. fileciteturn36file0L61-L68

Origin checking is not authorization. Local malware, another user process, browser extensions, or a non-browser client can generally omit `Origin`.

For live mode, mutating routes should require:

```text
Authorization: Bearer <ephemeral-session-token>
```

For especially dangerous actions—enabling live trading, resetting major kill switches, approving unusually large orders—use a second explicit confirmation nonce.

# Important secondary problems

## 10. Test count looks good, but coverage is inadequate

The branch reports 163 offline tests and zero learned influence at the latest handoff because no compatible TCN or graph champion currently exists. fileciteturn12file0L102-L132

That is good discipline. But the release checklist says aggregate line coverage across broker, execution, research, graph, and neural is only **55%**, versus an internal target of 85%. fileciteturn36file0L56-L60

Those are exactly the modules that can lose money or corrupt model state.

Your CI runs the suite on three operating systems, but it does not enforce coverage. fileciteturn9file0L9-L29

Add:

```bash
pytest \
  --cov=specforge \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=85
```

I would additionally set module-specific minimums:

- `risk.py`: 95%
- `execution.py`: 95%
- broker adapters: 90%
- promotion logic: 95%
- research leases/state transitions: 90%

A global percentage alone lets easy files conceal weak critical-path coverage.

---

## 11. Dependencies are effectively unpinned

Most runtime dependencies, including PyTorch, FastAPI, pandas, NumPy, MCP, and yfinance, have no locked versions. fileciteturn6file0L7-L25

That makes:

- CI nondeterministic,
- model serialization fragile,
- production behavior dependent on installation date,
- rollback difficult,
- cross-platform failures unpredictable.

Generate and commit platform-aware lock files or a constraints file with hashes. Store the exact dependency fingerprint in every model checkpoint.

---

## 12. Dynamic ETF handling is hardcoded

The ensemble identifies ETFs using a fixed 15-symbol set. fileciteturn39file0L18-L20

It then decides between ETF and company evidence families based on membership in that set. fileciteturn39file0L108-L111

Your research universe is intended to contain up to 1,500 instruments. Any ETF outside that hardcoded list may be treated as an operating company and incorrectly subjected to company dossier and fundamental logic.

Use the `instruments.is_etf` metadata already maintained by the universe layer.

---

## 13. Approved orders are revalidated but not repriced

Human-approved intents are checked again for funds, risk, and data age, but retain their original price and quantity. fileciteturn34file0L19-L53

The six-hour timeout reduces the problem but does not solve it. A material move within six hours can leave the order semantically stale.

At placement:

1. fetch a current executable quote;
2. compare it with the original reference price;
3. reject or require reapproval past a deviation threshold;
4. recalculate quantity and notional;
5. rerun the governor.

---

## 14. The branch is too broad to review safely

This branch is only six commits ahead of `main`, but it changes almost the entire system:

- neural architecture,
- learned graph,
- research orchestration,
- evidence system,
- universe selection,
- portfolio rebalancing,
- AI routing,
- runtime worker isolation,
- GUI,
- security,
- broker behavior,
- tests,
- deployment scripts.

That is not an experimental neural branch anymore. It is a platform rewrite. A regression becomes difficult to locate, bisect, or roll back.

Split it into reviewable layers:

1. runtime isolation and security;
2. evidence/universe changes;
3. TCN dataset and training;
4. graph learning and promotion;
5. portfolio/rebalancing;
6. GUI and operator controls;
7. packaging and release infrastructure.

---

## 15. The public repository exposes substantial internal design material

The README directly links the architecture, decisions, progress, plan, and canonical agent specification. fileciteturn5file0L176-L181

The `.gitignore` excludes runtime data and model artifacts, but not `dev/`, `AGENTS.md`, design rationale, release-state files, or roadmaps. fileciteturn37file0L3-L24

Because the repository and branch are public, `experimental_NN` currently exposes much of the internal scaffolding and product strategy. A non-default branch is still public.

If this is intended as proprietary product infrastructure:

- move internal materials to a private repository;
- remove tracked internal documents with `git rm --cached`;
- rewrite Git history if actual secrets or valuable unpublished IP were committed;
- retain only customer-facing architecture and usage documentation publicly.

`.gitignore` does not remove content already committed.

# What is good

The repository is substantially better architected than the average “AI trading bot” project:

- Signal generators do not directly place orders.
- The governor has final deterministic authority.
- Equity notional is treated as full worst-case loss.
- Resting buy orders consume cycle budget.
- Live/paper state is separated.
- Broker review occurs before placement.
- Point-in-time data and embargoes are explicitly considered.
- Checkpoint compatibility and hashes are checked.
- Learned models currently remain at zero influence when no valid champion pair exists.
- The project records known defects rather than pretending it is production-ready.

Those are the correct instincts. The main danger is that the statistical sophistication now exceeds the codebase’s ability to make every validation and state-transition assumption obviously correct.

# Recommended sequence

1. **Keep neural and graph influence hard-disabled at 0%.**
2. Correct graph portfolio metrics and neural fold boundaries.
3. Redesign promotion around explicit lifecycle states and all eligible challengers.
4. Require forward-shadow evidence before the first nonzero live blend.
5. Add a cross-process trading-cycle lease.
6. Remove shared configuration mutation.
7. Replace `advanced_override` with narrow, expiring exceptions.
8. Add local API authentication.
9. Pin dependencies and enforce branch/module coverage.
10. Split the branch before merging.
11. Remove internal development material from the public repository if it is proprietary.

My strongest conclusion: **the deterministic core is promising, but the learned overlay currently has enough validation and lifecycle ambiguity that it should be treated as research telemetry—not an allocator of live capital.**

---
---
---

The previous section was the initial thoughts, the section below details the final path forward.

---
---
---

# Revised conclusion

Your objection is valid. A permanent or indefinite zero-weight neural system defeats the purpose of the branch.

The correct end state is:

> **The neural model should have real, measurable influence from the beginning of the experiment, but through a dedicated bounded pathway that does not depend on the graph being promoted and does not confuse “experimental influence” with “production validation.”**

I would not merely patch the current promotion thresholds. I would partially redesign the neural stack around four principles:

1. **Direct neural influence:** the TCN gets its own explicit allocation in the production score.
2. **Statistically honest training:** point-in-time features, date-balanced sampling, purged validation, and a real portfolio simulator.
3. **Separation of concerns:** the TCN predicts returns; a later meta-model learns how to combine specialists.
4. **Bounded live experimentation:** the neural component starts at a nonzero weight, generates actual decisions, and is evaluated against counterfactual baseline portfolios.

The current code has a sophisticated framework, but several foundational details prevent the neural model from getting a fair test.

---

# 1. The neural model currently does not have the influence the configuration implies

Your configuration says:

```yaml
neural:
  enabled: true
  weight: 0.15
```

But `ensemble.NODE_FAMILY` does not contain `"neural"`, so the evidence ensemble ignores neural `SignalEvent`s.

The neural node therefore only affects trades through the analog graph. The graph then applies its blend only when:

* there is a champion graph;
* there is a compatible champion TCN;
* the graph was trained with that exact TCN;
* the activation stage is nonzero;
* all required per-symbol activations are considered complete.

The graph blend explicitly becomes zero without a champion.

The engine also removes the neural event from the ordinary ensemble once the graph blend becomes active, presumably to prevent double counting.

So the actual architecture is effectively:

```text
TCN
  ↓
neural SignalEvent
  ↓
ignored by evidence ensemble
  ↓
usable only through trained graph
  ↓
graph requires TCN + graph promotion
```

That is the wrong dependency structure. It makes the neural model dependent on a second experimental model before it can contribute.

## Fix

Integrate the TCN directly as a bounded learned component:

```python
production_score = deterministic_evidence_score
neural_score = calibrated_neural_score

final_score = (
    (1.0 - neural_blend) * production_score
    + neural_blend * neural_score
)
```

Initial experimental configuration:

```yaml
neural:
  enabled: true
  experimental_blend: 0.15
  max_blend: 0.40
  min_blend: 0.05
```

The graph should not own the neural model’s right to participate. The graph should eventually compete against this simple fixed blend as a replacement meta-model.

That gives the neural model genuine influence immediately after passing a minimum offline integrity gate.

---

# 2. The neural target has a semantic error

The TCN target is excess return:

```python
stock_future_return - SPY_future_return
```

But the neural node interprets that value as an absolute expected return:

```python
direction = "long" if pred > 0 else "avoid"
expected_return = pred
```

That is incorrect.

Suppose the model predicts:

```text
Stock return: -5%
SPY return:   -10%
Excess return: +5%
```

The current node produces approximately:

```text
direction = long
expected_return = +5%
```

But the actual prediction is a 5% loss that merely outperforms SPY.

The macro-regime gate may reduce deployment, but it cannot repair this semantic mismatch.

## Fix: use separate absolute and relative targets

The neural model should have four return heads:

```text
5-day absolute return distribution
21-day absolute return distribution
5-day excess return distribution
21-day excess return distribution
```

And two probability heads:

```text
P(absolute return > transaction cost), 5-day
P(absolute return > transaction cost), 21-day
```

Optional additional heads:

```text
P(excess return > 0)
expected downside / expected shortfall
```

A forecast object should look like:

```python
@dataclass(frozen=True)
class NeuralForecast:
    symbol: str
    as_of: str
    horizon_sessions: int

    absolute_q10: float
    absolute_q50: float
    absolute_q90: float

    excess_q10: float
    excess_q50: float
    excess_q90: float

    probability_absolute_positive: float
    probability_excess_positive: float

    model_id: str
    feature_snapshot_id: str
```

Long eligibility would be:

```python
absolute_edge = forecast.absolute_q50 - expected_cost

eligible = (
    absolute_edge > min_absolute_edge
    and forecast.probability_absolute_positive >= min_probability
    and forecast.excess_q50 >= min_excess_edge
)
```

The neural score could then be:

```python
def neural_score(f: NeuralForecast, expected_cost: float) -> float:
    absolute_edge = f.absolute_q50 - expected_cost
    downside_penalty = max(0.0, -f.absolute_q10)

    direction_confidence = 2.0 * f.probability_absolute_positive - 1.0
    relative_confirmation = math.tanh(f.excess_q50 / 0.04)

    raw = (
        0.50 * math.tanh(absolute_edge / 0.04)
        + 0.30 * direction_confidence
        + 0.20 * relative_confirmation
    )

    uncertainty = max(0.01, f.absolute_q90 - f.absolute_q10)
    reliability = min(1.0, 0.08 / uncertainty)

    return max(-1.0, min(1.0, raw * reliability))
```

This makes the output economically meaningful.

---

# 3. The stated 60-session TCN only has a 15-session temporal receptive field

The architecture uses:

```text
kernel size = 3
dilations = 1, 2, 4
```

For a causal TCN, the receptive field is approximately:

[
R = 1 + (k-1)\sum_i d_i
]

Therefore:

[
R = 1 + 2(1+2+4)=15
]

So although the input tensor contains 60 sessions, the final temporal representation only directly sees the last 15 sessions.

Longer-history information enters indirectly through engineered features such as `sma200_d`, `breakout60`, and `r21`, but the sequence model itself does not model the full 60-session trajectory.

## Fix

Use five residual blocks:

```text
dilations = 1, 2, 4, 8, 16
kernel = 3
```

Then:

[
R = 1 + 2(1+2+4+8+16)=63
]

That matches the 60-session window.

I would use:

```text
Channels: 32
Blocks: 5
Kernel: 3
Dilations: 1, 2, 4, 8, 16
Dropout: 0.10–0.20
Normalization: LayerNorm or weight normalization
Residual connections: yes
```

The resulting model remains small enough for a Mac mini but genuinely uses the intended sequence.

---

# 4. Split temporal features from static/context features

The current model puts all 44 features into every time step. Those features include:

* price and volume features;
* volatility-curve features;
* SEC fundamentals;
* valuation;
* event proximity;
* AI-derived news;
* missingness indicators.

That is not the best representation.

Annual fundamentals are not really temporal at the daily scale. Forward-filling an operating margin across every row in a 60-day sequence makes the TCN repeatedly process the same value.

## Better architecture

### Temporal branch

Use daily-varying information:

```text
returns
range
gap
volume z-score
realized volatility
RSI
ATR
breakout
moving-average distance
SPY returns
sector-relative returns
VIX complex
HYG/TLT context
```

Shape:

```text
batch × 60 sessions × approximately 25 features
```

### Snapshot branch

Use latest point-in-time context:

```text
valuation
revenue growth
operating margin
free-cash-flow margin
debt/assets
dilution
accruals
liquidity
news sentiment
event proximity
all missingness flags
sector or industry encoding
```

Shape:

```text
batch × approximately 20 features
```

### Fusion

```python
temporal_embedding = tcn(temporal_window)
context_embedding = context_mlp(snapshot_features)

joint = torch.cat(
    [temporal_embedding, context_embedding],
    dim=-1,
)

joint = fusion_mlp(joint)
```

This forces the model to treat:

* price history as a sequence;
* company state as a point-in-time condition.

That is cleaner and easier to debug.

---

# 5. Replace the 12,000-window in-memory design with a streaming dataset

The current process cap is 12,000 windows.

The sampling logic divides the global cap across symbols:

```python
per_symbol_cap = window_limit // len(symbols)
```

At 1,500 symbols:

[
12,000 / 1,500 = 8
]

That means the system may use only eight windows per symbol.

This is a poor tradeoff. It maximizes breadth but destroys temporal density. It also makes the number and composition of dates unstable.

The model has roughly tens of thousands of parameters and is being trained on highly overlapping windows with an effective sample size much smaller than 12,000.

## Fix: store features once, index windows lazily

Do not materialize this:

```text
12,000 × 60 × 44
```

Instead create a panel:

```text
symbol × session × feature
```

Then maintain a lightweight window index:

```python
@dataclass(frozen=True)
class WindowRef:
    symbol_id: int
    end_session_index: int
    target_5_index: int
    target_21_index: int
```

A PyTorch dataset slices the window only when requested:

```python
class MarketWindowDataset(torch.utils.data.Dataset):
    def __init__(self, panel, window_refs, window=60):
        self.panel = panel
        self.refs = window_refs
        self.window = window

    def __getitem__(self, index):
        ref = self.refs[index]

        temporal = self.panel.temporal[
            ref.symbol_id,
            ref.end_session_index - self.window + 1:
            ref.end_session_index + 1,
        ]

        context = self.panel.context[
            ref.symbol_id,
            ref.end_session_index,
        ]

        targets = self.panel.targets[
            ref.symbol_id,
            ref.end_session_index,
        ]

        return temporal, context, targets
```

Use:

* NumPy memory maps;
* Arrow/Parquet;
* or chunked `.npy` arrays.

This allows 100,000–500,000 indexed windows without loading all windows into RAM.

The resource bound should become:

```yaml
neural:
  max_training_steps: 3000
  max_training_seconds: 900
  batch_size: 256
```

Bound computation, not dataset validity.

---

# 6. Sample by date, not primarily by symbol

The model’s live function is cross-sectional ranking: choose the best symbols on a given date.

But the current dataset is built symbol by symbol and then shuffled. The ranking loss groups whatever same-date examples happen to land in the same random batch.

That weakens the rank objective.

## Fix: date-aware batches

A training batch should contain a cross-section from one or several dates:

```text
Date A: 64 symbols
Date B: 64 symbols
Date C: 64 symbols
Date D: 64 symbols
```

Then calculate:

* return-distribution loss per symbol;
* ranking loss within each date;
* no ranking comparisons across dates.

A sampler could be:

```python
class DateGroupedBatchSampler:
    def __iter__(self):
        dates = shuffled_training_dates()

        for date_batch in chunks(dates, dates_per_batch):
            indices = []

            for date in date_batch:
                candidates = self.indices_by_date[date]
                indices.extend(
                    random_sample(candidates, self.symbols_per_date)
                )

            yield indices
```

A more robust ranking loss:

```python
def pairwise_rank_loss(pred, truth, dates):
    losses = []

    for day in torch.unique(dates):
        mask = dates == day
        p = pred[mask]
        y = truth[mask]

        if len(p) < 8:
            continue

        i, j = sampled_pairs(len(p), max_pairs=512)
        sign = torch.sign(y[i] - y[j])

        valid = sign != 0
        losses.append(
            F.softplus(
                -sign[valid] * (p[i][valid] - p[j][valid])
            ).mean()
        )

    return torch.stack(losses).mean()
```

This aligns optimization with the actual portfolio-ranking task.

---

# 7. Fix point-in-time availability as a first-class data contract

The SEC implementation is directionally good because it keys facts to filing dates. But availability semantics are inconsistent across data sources.

The neural news feature groups data by `published_at`:

```sql
SELECT substr(published_at,1,10) ...
```

It does not appear to require:

```text
classified_at <= model decision time
ingested_at <= model decision time
```

That matters because an article may have been:

* published at 10:00;
* ingested at 15:00;
* classified the next day;
* then retrospectively assigned to the publication date.

The historical feature builder could therefore treat the classification as though it existed earlier.

## Fix: every feature needs `event_at` and `known_at`

For every external record:

```text
event_at: when the underlying event happened
known_at: earliest time the trading system actually possessed the usable record
```

Examples:

| Data                    | `event_at`              | `known_at`                                  |
| ----------------------- | ----------------------- | ------------------------------------------- |
| Daily bar               | session date            | official close plus ingestion delay         |
| SEC filing              | filing timestamp        | ingestion timestamp                         |
| News article            | publication timestamp   | max(publication, ingestion, classification) |
| AI dossier              | source publication time | dossier completion time                     |
| Analyst-derived feature | source event time       | computation completion time                 |

Historical joins must use:

```sql
known_at <= decision_timestamp
```

Not merely:

```sql
event_at <= decision_timestamp
```

I would create:

```sql
CREATE TABLE feature_observations (
    feature_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    event_at TEXT NOT NULL,
    known_at TEXT NOT NULL,
    value REAL,
    missing INTEGER NOT NULL,
    source_id TEXT,
    source_hash TEXT,
    PRIMARY KEY(feature_name, symbol, event_at, known_at)
);
```

The training dataset should contain a manifest specifying the exact decision timestamp convention.

For example:

```text
Decision timestamp: 09:35 ET
Permitted price data: previous completed session
Permitted filings/news: known before 09:30 ET
Execution assumption: 09:35–09:40 executable quote
```

Without that contract, “no lookahead” remains ambiguous.

---

# 8. Fix survivorship bias before trusting the long historical run

The research universe is based largely on currently available instruments and current catalog membership. Training current survivors back to 2011 introduces survivorship bias:

* bankrupt companies disappear;
* delisted companies disappear;
* acquired companies disappear;
* ticker histories may be incomplete;
* current successful companies are overrepresented.

For an experimental prototype, you can continue using the available universe. But the result must be labeled:

```text
current-survivor-universe historical evaluation
```

It cannot be treated as a fully point-in-time US-equity simulation.

## Practical progression

### Initial version

Use the current liquid universe but shorten the historical scope:

```text
2018–present or 2020–present
```

That reduces, but does not eliminate, the distortion.

### Serious version

Acquire point-in-time:

* security master;
* delisted securities;
* historical index/universe membership;
* corporate actions;
* adjusted OHLCV;
* historical fundamentals availability.

This is one of the places where better data will improve the system more than a more complicated neural architecture.

---

# 9. Eliminate per-holding full neural networks

The current system can fine-tune a complete TCN for each holding after 1,250 bars.

Even with initialization from the global model, one symbol gives roughly:

```text
1,250 observations
minus window and target requirements
with heavily overlapping samples
```

That is not enough independent data to safely fine-tune an entire neural network.

Worse, holding challengers can automatically promote when `holding_gate_passed()` succeeds.

The auto-promotion check does not visibly require that the result came from the final sealed trial rather than an ordinary validation trial.

## Fix: replace holding TCNs with small adapters

Keep the global TCN frozen.

Per-symbol adaptation should be limited to something like:

```text
2 horizon biases
2 horizon scale factors
2 probability calibration offsets
```

Or at most a low-rank adapter on the final layer.

Example:

```python
class SymbolAdapter(nn.Module):
    def __init__(self, horizons=2):
        super().__init__()
        self.mean_offset = nn.Parameter(torch.zeros(horizons))
        self.log_scale = nn.Parameter(torch.zeros(horizons))
        self.probability_offset = nn.Parameter(torch.zeros(horizons))
```

The adapter modifies the global output:

```python
adjusted_q50 = global_q50 + mean_offset
adjusted_width = global_width * exp(log_scale)
adjusted_probability = sigmoid(logit(global_probability) + probability_offset)
```

Regularize heavily toward zero.

This allows the system to learn that, for example, a particular stock tends to have:

* wider forecast errors;
* persistent underprediction;
* different probability calibration;

without training a new market model from roughly one stock’s history.

---

# 10. Improve the model-selection tournament

The current six hyperparameter trials are sensible in spirit. The validation winner is selected before the final test.

But the final system should distinguish:

1. architecture selection;
2. hyperparameter selection;
3. random-seed variance;
4. final ensemble construction;
5. sealed evaluation.

## Recommended tournament

### Phase A: cheap architecture screening

Compare:

```text
Ridge/elastic net
Gradient-boosted tree baseline
MLP
Current TCN
Expanded-receptive-field TCN
GRU or temporal mixer
```

These are not all production candidates. They are baselines.

### Phase B: neural configuration selection

Select among three or four TCN configurations using validation only.

### Phase C: seed ensemble

For the winning configuration, train three independent seeds:

```text
seed 11
seed 29
seed 47
```

The production neural forecast should be an ensemble:

```python
q50 = median(model_q50s)
q10 = quantile(model_q10s, conservative_level)
q90 = quantile(model_q90s, optimistic_level)
probability = mean(model_probabilities)
```

An ensemble of several modest models will usually be more robust than a single winner chosen from noisy validation differences.

The checkpoint should contain:

```text
architecture hash
feature schema hash
dataset manifest hash
three member checkpoints
calibration object
training commit SHA
dependency lock hash
```

---

# 11. Improve the objective function

The existing loss combines:

* quantile pinball loss;
* binary positive-return loss;
* ranking loss.

That is a good base.

I would change it to:

[
L =
L_{\text{absolute quantile}}

* 0.5 L_{\text{excess quantile}}
* 0.15 L_{\text{absolute sign}}
* 0.10 L_{\text{cross-sectional rank}}
* 0.02 L_{\text{uncertainty regularization}}
  ]

The exact coefficients must be tuned on validation only.

The uncertainty regularization prevents pathological interval collapse:

```python
width = q90 - q10
minimum_width = target_volatility * 0.5

width_penalty = F.relu(minimum_width - width).mean()
```

I would also median-calibrate the q50 output. The current calibration leaves the median offset at zero while correcting the interval bounds.

Use validation residuals:

```python
median_offset = np.median(truth - predicted_q50)
```

Probability calibration should use:

* Platt scaling;
* beta calibration;
* or isotonic regression when sufficient validation samples exist.

Matching only the average positive-return rate corrects the intercept but not the calibration slope.

---

# 12. Rebuild evaluation around a genuine policy backtest

The current graph utility calculation compounds overlapping 21-day forward labels as if they were sequential returns.

That must be removed.

A forecast metric and a trading metric are different things.

## Forecast evaluation

Per horizon:

```text
daily rank IC
pinball loss
coverage
Brier score
cross-sectional top-decile excess return
calibration curve
turnover of top-decile membership
```

## Policy evaluation

Run the actual engine against historical dates:

```text
signal generated using information known at decision time
order placed at next executable price
capital and existing holdings respected
position caps respected
transaction costs included
stops and time exits included
overlapping positions included
```

You already have the correct conceptual foundation in `backtest.py`: it runs the same cycle used by the live engine.

I would extend it to support policies:

```python
class ScoringPolicy(Protocol):
    def score(self, candidate_context) -> PolicyScore:
        ...
```

Then evaluate four policies using the same data and execution assumptions:

```text
P0: deterministic evidence only
P1: neural only
P2: fixed deterministic/neural blend
P3: learned meta-model blend
```

Each policy gets an isolated database and paper account.

The report must compare:

```text
incremental CAGR versus P0
incremental Sharpe versus P0
incremental max drawdown
turnover
cost drag
average exposure
regime performance
sector concentration
forecast calibration
```

The primary question is not:

> Did the neural model make money?

It is:

> Did adding the neural model improve the same system under the same opportunity set, execution model, and capital constraints?

---

# 13. Use purged walk-forward validation correctly

The fold boundary needs to be half-open so adjacent folds do not share a test session.

Current logic uses inclusive date boundaries, which can permit a boundary session to appear in both adjacent folds.

## Correct structure

For each fold:

```text
training dates: <= train_end
embargo: next 21 sessions
test dates: [test_start, test_end)
```

Use explicit session indices:

```python
train_indices = session_index < train_end_index
test_indices = (
    session_index >= test_start_index
    and session_index < test_end_index
)
```

Do not infer these boundaries from string comparisons when exact session indices are available.

## Proposed folds

For a 2018–2026 dataset:

```text
Fold 1: train 2018–2020, test 2021
Fold 2: train 2018–2021, test 2022
Fold 3: train 2018–2022, test 2023
Fold 4: train 2018–2023, test 2024
Fold 5: train 2018–2024, test 2025
Final sealed/prequential: 2026
```

The exact years depend on data availability.

The final test must be treated as consumed. Once you inspect it and modify the model, it is no longer sealed. Record that explicitly.

---

# 14. Replace the current graph with a proper stacked model

The current graph is trying to do two things:

1. provide interpretability through named specialist nodes;
2. learn nonlinear interactions and topology.

The concept is good. The current training data is not sufficient.

`graph_samples()` trains from stored `SignalEvent`s.

But signals are sparse and thresholded. A missing row can mean:

* the node was neutral;
* the node lacked data;
* the node was disabled;
* the node failed;
* the node did not emit because it was below a threshold.

Those are not equivalent.

The graph also uses a per-symbol `activation_complete` requirement that blocks blending when any enabled node is unavailable or blocked.

That produces an all-or-nothing system.

## Step 1: store dense expert activations

For every:

```text
decision date
symbol
node
node version
```

persist:

```sql
CREATE TABLE expert_activations (
    decision_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_version TEXT NOT NULL,

    state TEXT NOT NULL,
    signed_score REAL NOT NULL,
    confidence REAL NOT NULL,
    expected_absolute_return REAL,
    expected_excess_return REAL,

    data_as_of TEXT,
    known_at TEXT,
    feature_hash TEXT,

    PRIMARY KEY(decision_at, symbol, node_id, node_version)
);
```

Possible states:

```text
active
verified_neutral
data_unavailable
human_disabled
failed
stale
```

Do not infer state from the absence of a row.

## Step 2: generate out-of-fold base predictions

The meta-model may only train on specialist predictions that were themselves produced out of sample.

For the TCN:

```text
Fold 1 TCN predicts Fold 1 test
Fold 2 TCN predicts Fold 2 test
...
```

For trainable deterministic weights or AI components, use the same rule.

The graph must never train on predictions generated by a model trained on those same labels.

## Step 3: begin with a fixed-topology regularized stacker

Before topology mutation, use:

```text
ridge regression
elastic net
small constrained MLP
```

Inputs:

```text
specialist scores
specialist confidence
availability masks
regime
volatility
evidence coverage
TCN absolute/excess forecasts
```

Output:

```text
5-day expected absolute return
21-day expected absolute return
```

A simple stacker is easier to validate and much less likely to overfit than random graph topology mutation.

## Step 4: build a mixture-of-experts gate

The mature end state can be:

[
\hat{r} = \sum_i g_i(x, m) \hat{r}_i
]

Where:

* (\hat{r}_i) is expert (i)’s forecast;
* (g_i) is a learned regime-conditioned weight;
* (m) is the availability mask;
* the weights sum to one.

Example:

```python
expert_weights = softmax(
    gating_network(
        regime_features,
        volatility_features,
        availability_masks,
        evidence_coverage,
    )
)

final_return = torch.sum(
    expert_weights * expert_forecasts,
    dim=-1,
)
```

This is the principled version of what the analog graph is trying to achieve.

## Do not force nonzero learned influence

The current graph preserves a minimum path from every enabled specialist.

That conflates two concepts:

```text
node remains enabled and observable
node must have nonzero trading influence
```

A node should remain:

* computed;
* logged;
* visible;
* available for retraining;

while its learned trading weight can fall to zero.

Human control should determine whether the analysis runs. The statistical system should determine whether its forecast currently deserves influence.

---

# 15. Give the neural model real influence through an experimental sleeve

I would implement two simultaneous mechanisms.

## A. Score influence

Start with:

```yaml
neural:
  experimental_blend: 0.15
```

That means 15% of candidate ranking comes from the calibrated TCN.

This is real influence even when it does not independently create an order.

## B. Dedicated neural exploration capacity

Reserve a bounded fraction of new-position capacity for a neural-disagreement candidate.

Example:

```yaml
neural:
  exploration:
    enabled: true
    max_positions: 1
    budget_fraction: 0.20
    minimum_absolute_edge: 0.0075
    minimum_probability_positive: 0.57
```

If the deterministic ensemble and neural model disagree, the system may open one smaller neural-driven probe:

```text
normal target size × 0.25
```

The probe still passes:

* cash limits;
* account deployment;
* single-position cap;
* liquidity checks;
* duplicate protection;
* kill switches;
* broker review.

This is not handwaving. It creates actual experimental exposure and produces real realized outcomes.

The `entry_mode="probe"` mechanism already exists in `TradeCandidate`.

Use that machinery explicitly for neural exploration.

---

# 16. Maintain parallel counterfactual portfolios

A live blended account alone cannot tell you what the neural contribution accomplished.

For every live cycle, maintain four virtual books:

```text
deterministic_book
neural_book
blended_book
actual_broker_book
```

They all receive the same:

* timestamps;
* prices;
* spread/slippage assumptions;
* liquidity filters;
* position caps;
* exit rules.

Only the scores differ.

Persist:

```sql
CREATE TABLE policy_positions (
    policy_id TEXT,
    symbol TEXT,
    opened_at TEXT,
    qty REAL,
    average_cost REAL,
    status TEXT,
    PRIMARY KEY(policy_id, symbol, opened_at)
);

CREATE TABLE policy_marks (
    policy_id TEXT,
    timestamp TEXT,
    equity REAL,
    cash REAL,
    exposure REAL,
    drawdown REAL,
    PRIMARY KEY(policy_id, timestamp)
);

CREATE TABLE policy_decisions (
    policy_id TEXT,
    cycle_id TEXT,
    symbol TEXT,
    score REAL,
    target_notional REAL,
    action TEXT,
    reason TEXT
);
```

The dashboard should show:

```text
Neural incremental return versus deterministic baseline
Neural incremental drawdown
Neural contribution to accepted trades
Neural contribution to rejected trades
Counterfactual trades added
Counterfactual trades avoided
```

This is how the system genuinely learns whether the neural model is useful.

---

# 17. Redesign model lifecycle states

The current generic `challenger` state represents too many things:

* early validation trial;
* final tournament candidate;
* sealed-test candidate;
* forward-shadow candidate.

Promotion functions also focus on the newest challenger rather than the best eligible challenger.

## Replace states with:

```text
training
validation_candidate
validation_winner
sealed_candidate
experimental_live
production_candidate
champion
rejected
incompatible
retired
```

Model transitions:

```text
training
  → validation_candidate
  → validation_winner
  → sealed_candidate
  → experimental_live
  → champion
```

At each transition, persist:

```text
which gate passed
which metrics were used
which dataset was used
who or what authorized transition
permitted live blend
```

Example schema additions:

```sql
ALTER TABLE model_runs ADD COLUMN lifecycle_state TEXT;
ALTER TABLE model_runs ADD COLUMN dataset_manifest_id TEXT;
ALTER TABLE model_runs ADD COLUMN code_commit TEXT;
ALTER TABLE model_runs ADD COLUMN dependency_hash TEXT;
ALTER TABLE model_runs ADD COLUMN trial_group TEXT;
ALTER TABLE model_runs ADD COLUMN random_seed INTEGER;
ALTER TABLE model_runs ADD COLUMN permitted_live_blend REAL DEFAULT 0;
ALTER TABLE model_runs ADD COLUMN gate_evidence TEXT;
ALTER TABLE model_runs ADD COLUMN promoted_at TEXT;
```

Promotion should query all eligible models:

```sql
SELECT *
FROM model_runs
WHERE lifecycle_state = 'experimental_live'
  AND incompatibility_reason IS NULL
ORDER BY validation_selection_score DESC;
```

Then apply deterministic comparison rules.

---

# 18. Use a nonzero but disciplined influence ramp

My preferred ramp is:

| Stage                  | Neural influence | Requirement                                       |
| ---------------------- | ---------------: | ------------------------------------------------- |
| Integrity              |               0% | Checkpoint and dataset integrity only             |
| Experimental           |           10–15% | Purged folds, baselines beaten, sealed test valid |
| Confirmed experimental |           20–25% | Positive forward forecast statistics              |
| Probation              |              30% | Incremental policy performance positive           |
| Mature                 |          40% max | Multiple regimes and sufficient live history      |

The key correction to my previous answer is:

> The system should enter **experimental live influence**, not remain at zero until it has years of live labels.

The first nonzero stage should still require offline validity. Otherwise you are testing code defects and leakage rather than model quality.

For your existing small account, a 10–15% scoring influence plus one quarter-sized neural probe is enough to generate actual evidence without allowing the experimental system to dominate every trade.

---

# 19. Refactor the neural code into modules

`neural.py` currently handles:

* feature construction;
* dataset construction;
* architecture;
* loss functions;
* calibration;
* baselines;
* walk-forward training;
* checkpointing;
* promotion;
* inference;
* shadow evaluation;
* holding models.

That is why `train_challenger` has F-level complexity.

I would split it as follows:

```text
specforge/ml/
    schema.py
    availability.py
    features.py
    panel.py
    dataset.py
    sampling.py
    model.py
    losses.py
    calibration.py
    baselines.py
    training.py
    evaluation.py
    registry.py
    inference.py
    policy.py
```

### Existing functions move to:

| Current                    | Destination                                       |
| -------------------------- | ------------------------------------------------- |
| `_features`                | `ml/features.py`                                  |
| `build_dataset`            | `ml/dataset.py`                                   |
| `_make_model`              | `ml/model.py`                                     |
| `_pinball`, `_rank_loss`   | `ml/losses.py`                                    |
| `_calibration`             | `ml/calibration.py`                               |
| `_baseline_metrics`        | `ml/baselines.py`                                 |
| `_walk_forward_metrics`    | `ml/evaluation.py`                                |
| `train_challenger`         | decomposed across `training.py` and `registry.py` |
| `promote`, `maybe_promote` | `ml/registry.py`                                  |
| `predict_today`            | `ml/inference.py`                                 |

A new training flow:

```python
def run_tournament(request: TournamentRequest) -> TournamentResult:
    manifest = create_dataset_manifest(request)
    dataset = open_dataset(manifest)

    trial_results = [
        train_trial(dataset, spec)
        for spec in request.trial_specs
    ]

    winner = select_validation_winner(trial_results)
    ensemble = retrain_seed_ensemble(dataset, winner.spec)
    sealed = evaluate_sealed(ensemble, dataset)
    run = persist_model_run(ensemble, sealed, manifest)

    return advance_lifecycle(run)
```

Each function becomes independently testable.

---

# 20. Refactor the live engine into explicit phases

`_run_cycle` currently handles almost every subsystem.

Introduce:

```python
@dataclass
class CycleContext:
    cycle_id: str
    config: Config
    store: Store
    broker: BrokerAdapter
    market: MarketContext
    account: AccountState
    regime: RegimeResult
    model_state: ModelState
    budget: CycleState
```

Then:

```python
def run_cycle(...):
    with trading_cycle_lease(...):
        ctx = prepare_cycle(...)
        settle_prior_orders(ctx)
        process_exits(ctx)
        process_approved_orders(ctx)

        activations = compute_specialists(ctx)
        forecasts = compute_neural_forecasts(ctx)
        candidates = build_candidates(ctx, activations, forecasts)
        targets = construct_portfolio(ctx, candidates)
        results = execute_targets(ctx, targets)

        finalize_cycle(ctx, results)
        return results
```

This makes it possible to test:

```text
candidate generation without execution
neural blending without broker state
execution without model code
policy backtests without FastAPI
```

---

# 21. Operational fixes remain necessary

These are not reasons to disable the neural model. They are required to make its experiment trustworthy.

## Cross-process trading lease

The trading lock is currently a process-local `threading.Lock`.

Use the same SQLite lease design already used by research workers.

Key:

```text
trading_cycle_lease:<mode>:<account_hash>
```

A CLI scan and daemon scan must never run concurrently.

## Immutable configuration

The engine and research code mutate:

```python
cfg.data["universe"]["symbols"]
```

The research shadow code restores the old value afterward, but mutation remains fragile.

Pass an explicit universe into `MarketContext` instead.

## Narrow safety exceptions

Replace global `advanced_override` with scoped exceptions:

```yaml
risk_exceptions:
  - parameter: max_single_equity_position
    value: 0.30
    expires_at: 2026-08-15
    maximum_account_equity: 100
    reason: live probation account
```

## Reprice approved orders

At approval execution:

```text
fetch current quote
reject if quote moved beyond tolerance
recompute quantity
rerun governor
place order
```

## Local API authentication

Add a random per-launch bearer token for all mutation routes.

## Dependency locking

Pin the exact versions used to create and load model checkpoints.

---

# 22. Tests I would add before trusting the redesigned neural stack

## Data integrity

```text
News is unavailable before classified_at
SEC fact unavailable before filing known_at
No feature row has known_at after decision_at
No target observation appears in its own input window
No validation/test target crosses the embargo
Universe membership is point-in-time or explicitly survivor-labeled
```

## Model semantics

```text
Positive excess + negative absolute return cannot generate a positive absolute edge
TCN receptive field reaches all 60 sessions
Quantiles remain ordered
Calibration never uses test outcomes
Three-seed ensemble is deterministic given stored seeds
```

## Evaluation

```text
Adjacent folds do not share dates
Top-decile selection is performed within each date
Overlapping horizon labels are never compounded as daily portfolio returns
Policy backtest respects cash, exposure, turnover, and holding overlap
Transaction costs increase monotonically with turnover
```

## Lifecycle

```text
Validation-only holding adapter cannot promote
Newest challenger cannot conceal an older eligible finalist
A model cannot inspect the sealed set twice
A code/feature/dependency hash mismatch blocks inference
A graph trained against TCN A cannot run with TCN B
```

## Live experiment

```text
Neural blend is exactly the configured percentage
Neural-only, deterministic-only, and blended books receive identical prices
A neural probe remains below its dedicated budget
A model failure falls back to deterministic scoring
Model failure cannot block exits
```

## Concurrency

```text
Daemon scan + CLI scan produce one cycle
Expired lease can be recovered
Live and paper leases are isolated
A crashed worker cannot commit after losing ownership
```

---

# Recommended implementation sequence

## Phase 1 — Make the current neural experiment real

1. Add absolute-return targets.
2. Add neural directly to final scoring at 15%.
3. Add a neural probe pathway.
4. Add deterministic, neural-only, and blended shadow books.
5. Fix graph overlapping-return metrics.
6. Fix fold boundaries.
7. Block early holding-network promotion.

This gives the current model genuine influence quickly.

## Phase 2 — Repair the dataset

1. Introduce `known_at`.
2. Separate temporal and context features.
3. Build a memory-mapped feature panel.
4. Implement date-grouped sampling.
5. Expand the TCN receptive field to 63 sessions.
6. Add stronger baselines.
7. Train three-seed ensembles.

## Phase 3 — Repair validation

1. Purged expanding folds.
2. Strict validation winner selection.
3. One-time sealed evaluation.
4. Exact policy backtest.
5. Incremental comparison against deterministic baseline.
6. Bootstrap uncertainty around incremental results.

## Phase 4 — Rebuild the meta-model

1. Persist dense specialist activations.
2. Generate out-of-fold expert forecasts.
3. Train regularized linear stacker.
4. Compare fixed blend against stacker.
5. Introduce constrained mixture-of-experts only after the simple stacker wins.

## Phase 5 — Production hardening

1. Cross-process trading lease.
2. Immutable cycle config.
3. Session authentication.
4. Scoped safety exceptions.
5. Repricing and revalidation.
6. Dependency lock and coverage enforcement.
7. Split the branch into reviewable commits.

# The end state

The strongest realistic architecture is:

```text
Point-in-time feature store
        ↓
Global multi-task TCN ensemble
  ├─ absolute-return quantiles
  ├─ excess-return quantiles
  ├─ probability of positive absolute return
  └─ calibrated uncertainty
        ↓
Direct bounded neural score contribution
        ↓
Dense specialist activation layer
        ↓
Constrained mixture-of-experts meta-model
        ↓
Portfolio constructor
        ↓
Deterministic governor
        ↓
Broker review and execution
```

Alongside:

```text
deterministic counterfactual book
neural-only counterfactual book
fixed-blend counterfactual book
actual live book
```

My revised position is therefore not “turn off the neural model.” It is:

> **Remove the neural model’s dependence on the immature graph, give it a direct 10–15% experimental influence plus bounded probe capacity, and rebuild the data/evaluation infrastructure so the model is being tested on an honest problem.**

The biggest return-to-the-drawing-board item is not abandoning the TCN. It is fixing the target semantics, data availability, sampling, and evaluation contract around it. The TCN itself is a reasonable architecture once its receptive field and input organization are corrected.
