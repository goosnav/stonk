# V4 — Hypothesis Engine, Steering, Model Observatory

> **Audience: any agent (or human) picking this up cold.** Read
> [PROGRESS.md](PROGRESS.md), [DECISIONS.md](DECISIONS.md) (D1–D27), and
> [ARCHITECTURE.md](ARCHITECTURE.md) first. The ROADMAP ground rules apply to
> every sprint here — especially: **never weaken risk.py**, **nodes emit
> forecasts never orders**, **AI output only ever becomes structured input to
> deterministic scoring**, **all data as-of correct**, run
> `.venv/bin/pytest tests/ -q --ignore=tests/test_gui.py` after every change.

## What V4 adds (user directives, 2026-07-09)

1. **Hypothesis layer** — with AI enabled, the system maintains a persistent
   **north-star hypothesis** (rarely changed) and a **short-term hypothesis**
   (rotated every few days). Old hypotheses are dated and archived to
   `data/hypotheses/archive/`. The system trades on the hypothesis via a
   signal node. Feature is config-switchable (`hypothesis.enabled`).
2. **Steering** — strategic choices (hypothesis adoption, north-star changes,
   node promotions, watchlist adds, risk suggestions) surface in the GUI as
   steerable requests with options + an AI recommendation, like Claude Code
   plan mode. Requests expire; trading NEVER blocks on them.
3. **Autonomy invariant** — as long as there is money in the account the
   system trades with zero human input (already true: `approval_mode: auto`,
   intents expire in 6h live). V4 must not introduce any blocking path.
4. **GUI** — proper portfolio-value chart (axes, timeframes, intraday marks);
   "Risk & Budget" tab renamed to "Config"; new **Model** tab visualizing the
   node network and its learned shape.

## Design decisions (ratified by user 2026-07-09)

- **Hypothesis power = ensemble voice + watchlist.** A deterministic
  `hypothesis` node reads the stored ACTIVE short-term hypothesis and emits
  SignalEvents from its stances (symbol, direction, conviction, horizon).
  It is weighted, attribution-measured, and governor-gated like every other
  node. The hypothesis may also add ≤ `hypothesis.max_watchlist` symbols to
  the scan universe. The AI never sizes or places anything.
- **Tiered expiry defaults.** Steering requests that expire un-answered:
  short-term hypothesis adoption → **auto-adopt** the recommendation;
  north-star changes, node promotions, risk/deployment suggestions →
  **keep status quo**. Bootstrap exception: the very first north star
  auto-adopts on expiry (no status quo exists).
- **Steering scope = everything strategic** (user chose the widest option):
  `hypothesis_adopt`, `north_star_change`, `node_promotion`,
  `watchlist_add`, `risk_suggestion`. Risk suggestions apply through
  app.py `_set_override` so Config.validate() still rejects dangerous values.
- **Portfolio chart = proper chart + intraday marks**, recorded (throttled)
  from the live-quote status path plus each scan mark.
- **No build chain** (D2 stands): Model tab is vanilla JS + SVG.

## Data model

```sql
CREATE TABLE hypotheses(
  id TEXT PRIMARY KEY, tier TEXT,          -- north_star | short_term
  status TEXT,                             -- proposed | active | retired
  created_at TEXT, activated_at TEXT, retired_at TEXT,
  thesis TEXT,                             -- markdown, human-readable
  stances TEXT,                            -- JSON [{symbol,direction,conviction,horizon_days,rationale}]
  watchlist TEXT,                          -- JSON [symbols]
  invalidation TEXT,                       -- what would falsify this
  source TEXT,                             -- ai | human
  parent_id TEXT                           -- hypothesis it replaced
);
CREATE TABLE steering(
  id TEXT PRIMARY KEY, kind TEXT, created_at TEXT, expires_at TEXT,
  title TEXT, context TEXT,                -- markdown context for the human
  options TEXT,                            -- JSON [{key,label,detail}]
  recommended TEXT,                        -- option key the AI recommends
  default_on_expiry TEXT,                  -- adopt | status_quo
  status TEXT,                             -- pending | decided | expired
  decided_key TEXT, decided_at TEXT, decided_via TEXT  -- gui | expiry
);
CREATE TABLE equity_intraday(
  ts TEXT, equity REAL, cash REAL, source TEXT     -- throttled live marks
);
```

Files (mirror of DB, for humans + the user's explicit archive requirement):
- `data/hypotheses/north_star.md`, `data/hypotheses/short_term.md` (current)
- `data/hypotheses/archive/YYYY-MM-DD-<tier>-<id8>.md` (on retirement)

## Config additions (configs/default.yaml)

```yaml
hypothesis:
  enabled: false                 # master switch (also requires ai.enabled to GENERATE)
  short_term_max_age_days: 5     # regen when older than this
  regen_on_regime_change: true
  north_star_review_days: 30
  max_watchlist: 8
  steering_ttl_hours: 24         # steering request lifetime
nodes:
  hypothesis: {enabled: false, weight: 0.4, horizon_days: 10, status: experimental}
```

## Sprints

### Sprint 1 — Hypothesis core (store, files, generation, node)
- `store.py`: `hypotheses` table + CRUD (`save_hypothesis`, `active_hypothesis(tier)`,
  `retire_hypothesis`, point-in-time: `active_hypothesis(tier, as_of)` only
  returns rows `activated_at <= as_of` — ROADMAP rule 3).
- `specforge/hypothesis.py`: generation (AIClient JSON call: inputs = regime,
  per-symbol momentum/vol one-liners, portfolio, market headlines, north star,
  prior hypothesis + its measured node scorecard; output = strict JSON thesis/
  stances/watchlist/invalidation), file mirroring + archive, activation and
  retirement lifecycle. Budgeted via reserve-then-commit; parse failure ⇒
  discard, deterministic pipeline unaffected (D14 posture).
- `specforge/nodes/hypothesis.py`: deterministic node; reads active short-term
  hypothesis as-of ctx; emits stance SignalEvents; `[]` when disabled/none.
- `engine.py`: merge active hypothesis watchlist into scan symbols (bounded).
- Tests: lifecycle + archive files; node point-in-time; watchlist merge;
  generation parse-failure degradation. **No orders placed by any of this.**

### Sprint 2 — Steering (queue, expiry, apply, API)
- `store.py`: `steering` table + CRUD; expiry sweep (on read + scheduler tick).
- `specforge/steering.py`: `create_request(kind, …)`, `decide(id, key, via)`,
  `apply(request, key)` per kind (hypothesis_adopt / north_star_change /
  node_promotion / watchlist_add / risk_suggestion — the last through
  app `_set_override` validation). Tiered expiry defaults per design.
- `app.py`: `GET /api/steering`, `POST /api/steering/{id}`; audit everything.
- Hypothesis regen (post-close scheduler + `stonk hypothesis` CLI) creates
  steering requests instead of hard-activating (except bootstrap north star).
- Tests: expiry tiers (adopt vs status_quo), apply paths, dangerous risk
  suggestion rejected by validate, non-blocking guarantee (scan runs fine with
  10 pending requests).

### Sprint 3 — GUI (Config rename, portfolio chart, steering panel)
- Tab label "Risk & Budget" → "Config" (`data-p="risk"` unchanged).
- `equity_intraday` marks: record from `/api/status` quote path (throttle ≥5min,
  market hours) + every scan; `GET /api/portfolio_value?range=1D|1W|1M|ALL`
  merges intraday + daily curve. Chart with time axis, $ gridlines, current
  value readout, range buttons.
- Steering panel on Overview: pending requests with options, recommended
  highlighted, countdown, "if you do nothing: X". POST decision.
- Playwright smoke additions in test_gui.py (tab renders, no JS errors).

### Sprint 4 — Model tab + wiring + deploy
- `GET /api/model`: nodes (role/status/enabled/base weight/learned multiplier/
  effective weight in current regime/scorecard/degraded/AI flag), ensemble
  params, regime, active hypothesis summary, kill switches, budget.
- New "Model" tab: SVG flow diagram data → nodes → ensemble → governor →
  broker; node box size/color = effective weight & measured IR; hypothesis box
  feeds the hypothesis node; degraded nodes dashed. Vanilla JS.
- Scheduler: post-close hypothesis regen + north-star review cadence.
- Docs: DECISIONS.md D34 (hypothesis/steering), PROGRESS.md, ARCHITECTURE.md
  module map update. Full test run, GUI smoke, live restart + verify.

## DO-NOT list (all sprints)
- Do NOT let hypothesis/steering write orders, positions, or touch risk.py
  internals. The ONLY trading influence is SignalEvents through the ensemble
  and config overrides through the validated `_set_override` path.
- Do NOT let generation run inside market-hours scan cycles (cost/latency);
  post-close + CLI only.
- Do NOT block any scan on a pending steering request.
- Do NOT auto-adopt north-star changes, promotions, or risk suggestions on
  expiry (status quo wins; bootstrap north star is the sole exception).
- Do NOT store secrets or hypothesis API responses outside the budget ledger
  posture already in ai.py.
