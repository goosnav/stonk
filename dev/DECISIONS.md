# Decision log

Short ADR-style entries, newest last. Each: context → decision → why.

## D1. Adopt AGENTS.md wholesale as canonical spec (2026-07-06)
It already encodes the right philosophy (deterministic switchboard, risk governor,
AI as enrichment). This repo implements it; deltas are logged here.

## D2. Python 3.13 + venv + pip, SQLite, FastAPI + one vanilla HTML page (2026-07-06)
No uv on machine → plain venv. SQLite (stdlib) for all state — single small-account
engine, no concurrency pressure. No React build chain: one dashboard.html with
fetch polling keeps GUI hackable. No ORM.

## D3. Time-step budget is the primary safety primitive (2026-07-06, user)
Each scan cycle has a hard $ deployment cap = min(equity × time_step_budget_pct,
abs_cap) × regime deployment multiplier. Worst-case loss (full notional for equity,
full premium for options) of all NEW positions in the cycle must fit inside it.
User accepts total-loss risk of deployed capital; the budget bounds the bleed rate.

## D4. Approval policy: threshold mode default (2026-07-06, user)
Autonomous below `approval_notional_threshold_pct` of equity; above → GUI approval
queue. `auto` (full autonomy) and `all` (approve everything) selectable.

## D5. Options gated by account scale (2026-07-06, user)
Start <$1k → options node off. Auto-unlock when
`equity × max_single_option_premium_risk ≥ min_viable_option_premium` ($75 default
→ ~$5k account). `options_enabled: auto|true|false`.

## D6. Broker order path: standalone MCP client first, Claude-session bridge fallback (2026-07-06, user)
Robinhood MCP is OAuth; custom-client support unverified (docs name Claude/ChatGPT/
Cursor/etc). Try MCP-spec OAuth 2.1 + dynamic client registration from Python; if
allowlisted-out, `robinhood_bridge` adapter: engine queues reviewed intents in DB,
a scheduled Claude Code session (which has the RH MCP connected) relays them and
writes back fills. Engine keeps all decision/review authority either way.

## D7. AI via OpenRouter-compatible API, reserve-then-commit budgeting (2026-07-06, user)
Default <$1/day, cheap models (DeepSeek/Qwen), model+budget+price-table in config/GUI.
Ledger pre-estimates task cost, reserves it, and skips the task (deterministic
fallback) if the full task doesn't fit the remaining budget — never half-spends.

## D8. Same MarketContext powers live and backtest (2026-07-06)
Lookahead prevention by construction: context only exposes rows ≤ as_of. Backtest
is "run the live pipeline at historical as_of dates", not a parallel code path.

## D9. Data: Stooq primary, yfinance fallback, daily bars only in MVP (2026-07-06)
Free, no keys. Signals validated at daily horizon; minute data = cost before edge.
^VIX via yfinance (Stooq lacks it reliably).

## D10. Error bars from backtest analog trades (2026-07-06)
forecast.py bootstraps horizon returns of historical trades in the same
(score-bucket × regime) cell — produced by backtest.py into the same schema —
falling back to a wide vol-scaled prior labeled confidence=low when analogs are
scarce. No naked point estimates in the GUI.

## D11. Paper account starts at $1k to mirror real funding (2026-07-06)
Fractional shares assumed (Robinhood supports; paper broker allows) — at <$1k,
whole-share-only would make position sizing impossible.

## D12. RH MCP tool schemas captured from a live session (2026-07-06)
The exact schemas of the connected Robinhood MCP were inspected during
development and encoded in broker/robinhood_mcp.py: all numeric params are
STRINGS; `ref_id` (UUID) is the idempotency key; fractional shares require
type=market + regular_hours (limit orders need whole shares); accounts must be
agentic_allowed=true; get_portfolio (not get_accounts) is authoritative for
buying power. Consequence: fractional entries go as market orders in regular
hours on our liquid-only universe — the "limit by default" rule applies to
whole-share orders. Runtime tool discovery still runs because RH says the
surface will evolve.

## D14. AI parse failures disable AI, not trading (2026-07-06)
AGENTS.md lists an "AI output parsing failure kill switch". Halting TRADING
because an enrichment feed emits garbage is backwards for a deterministic-core
system: 5 unparseable responses/day set kv `ai_disabled_until` (+24h) and the
deterministic pipeline continues untouched. Tested in test_ai_attribution.py.

## D15. Drawdown kill switch: cooldown + auto-resume (2026-07-06)
Backtest v1 finding: the manual-reset drawdown switch tripped in the 2022 bear
(-15.1% from peak) and froze entries for the remaining 3 YEARS of the
simulation (OOS window = flat cash). Live, the same failure mode = system
quietly dead until the human notices. New behavior: trips block entries for
`drawdown_cooldown_days` (default 10 trading-ish days), then auto-clear;
`null` restores manual-only. Exits always continue. GUI reset still available
earlier. This is a risk-policy change, not backtest curve-fitting: the
per-trade edge was positive before and after the halt.

## D17. Drawdown high-water mark resets when the switch clears (2026-07-06)
v2 backtest stayed frozen even WITH the D15 cooldown: the trip condition is
level-based (equity < peak×0.85), so an all-cash account below the old peak
re-trips the moment the cooldown clears — a trip/clear/re-trip livelock. Fix:
clearing the drawdown switch (auto or manual) stamps kv `dd_peak_reset_d`; the
HWM is computed only from equity since that date. Semantics: each drawdown
episode absorbs one kill_switch_drawdown tranche, then the book restarts flat
with a fresh baseline. Test: test_drawdown_trip_clears_and_baseline_resets.

## D18. Backtest v3 = validation gate PASSED; aggressive profile failed OOS (2026-07-06)
v3 (default risk, D15+D17 fixes, dev/reports/backtest_v3.json): CAGR 7.55%,
Sharpe 0.76, maxDD 16.8%, PF 1.39, 1756 cost-included trades — and OOS (last
30%) BETTER than in-sample (8.8% vs 7.1% CAGR): no overfit signature. Momentum
+0.95%/trade (n=1593), sector_rotation +0.66% (n=695), reversal ≈0 (n=64,
both runs) → reversal defaulted OFF. configs/aggressive.yaml: 15.6% in-sample
but −0.8% OOS and 25% maxDD → NOT default; treat as a paper experiment only.
Conclusion: default stack ships as-is; growth path is new nodes measured live
(earnings drift, congress, insider, news, options overlay), not bigger sizing.
v3's 1756 analog trades feed live error bars (restored after the aggressive
run clobbered them — backtests that copy analogs must run LAST or re-copy).

## D19. Governor vetoes ≠ broker rejections (2026-07-06)
The rejected-orders kill switch livelocked: it counted the governor's healthy
"no" decisions, and each trip caused more rejections next cycle. Governor
vetoes now get order status `vetoed`; only broker-level `rejected` counts
toward the storm switch (its actual purpose: runaway order loops at the
broker). Vetoed orders are also excluded from the duplicate-cooldown check.

## D20. Engine↔broker position mismatch guard (2026-07-06)
Each cycle compares engine position metadata to broker truth. Broker wins:
engine-only rows are closed (audited `position_mismatch`, no fake trade
recorded); broker-only holdings are surfaced for the operator. Guarded
against dead feeds (equity 0 + no positions ⇒ skip, don't wipe state).
Found via a dev accident (DB deleted under a live server process — never
`rm data/specforge.db` while anything is running).

## D21. Per-thread SQLite connections (2026-07-06)
The dashboard fires ~13 parallel fetches; FastAPI's threadpool + ONE shared
sqlite3 connection interleaved cursors → random 500s (JSONDecodeError on
empty rows). Store.db is now a threading.local per-thread connection (WAL =
safe concurrent readers, busy_timeout=15s for writers). Verified with a
140-request concurrency hammer: 0 non-200s. Found via headless Playwright
render testing — screenshots in dev/reports/gui_*.png.

## D22. Standalone Robinhood MCP client VERIFIED LIVE (2026-07-06 21:11)
The D6 primary path works: user completed OAuth in-browser, tokens cached at
~/.specforge/rh_tokens.json (0600), and the Python MCP client then read the
real account with NO browser: equity $50 / cash $50 / buying_power $50 on
agentic account 934803396 (agentic_allowed=true, type=cash), plus real-time
quotes (SPY 751.31 incl. bid/ask). The bridge remains a fallback only.
Live response shapes encoded in the adapter: everything wraps in
{"data": ..., "guide": ...} (unwrapped once in _call); portfolio buying_power
is NESTED {"buying_power": "50.0000"}; quotes are data.results[].quote.
.env is now auto-loaded by config.py (stdlib parser, real env wins);
RH_ACCOUNT_WHITELIST=934803396 and LIVE_TRADING_ENABLED=true are set — live
orders still require starting with `--mode live` (config flag is the third gate).

## D16. Backtest v1 result (2026-07-06, dev/reports/backtest_v1.json)
Per-trade engine works: PF 1.34, win 48%, avg win +6.5% vs avg loss -4.5%,
momentum n=988 avg +0.74%/trade AFTER costs. Portfolio-level CAGR only 4%
because (a) the drawdown freeze (fixed, D15) and (b) conservative deployment
(vol-targeted sizing uses a fraction of the account). earnings_drift traded 0×
in backtest — yfinance has no deep earnings history; judge that node on
paper/live only. Aggressiveness knob = sizing/deployment, not new signals.

## D13. Async fills reconciled at cycle start (2026-07-06)
Live/bridge orders don't fill synchronously. Executor.reconcile() polls
resting/relayed orders each cycle via broker.poll_order() and creates
positions/trades through the same bookkeeping path as immediate fills.

## D23. Positions are mode-tagged (2026-07-06)
Paper and live share data/specforge.db. Without tagging, a live scan would
treat leftover paper positions as real holdings (and the D20 mismatch guard
would churn closing them). positions.mode column added (ALTER-migration for
existing DBs, default 'paper'); engine reads filter by the cycle's mode.
Trades were already source-tagged; equity_curve keyed by source.

## D24. Regime-conditioned weight multipliers replace, not stack (2026-07-07)
ROADMAP Sprint D item. attribution.update_weights now computes a per-regime
multiplier for each node using the same shrunk-IR formula, but only for
(node, regime) cells with n >= weight_learning.regime_min_n (default 30).
Stored in kv `regime_multipliers` = {node_id: {regime: mult}}; consumed in
ensemble.s_node_weight INSTEAD OF the global multiplier when the current
regime has a qualifying cell (fallback: global). Chose "replace" over
"stack" (global x regime) so the governor's [min_multiplier, max_multiplier]
bound holds trivially — a product of two in-bound factors could reach 4.0.
Cells that fall below the threshold get their entry deleted so stale
multipliers cannot outlive their evidence. Inert today: no (node, regime)
cell has 30 paper/live trades yet; it activates as Sprint A accumulates data.
Test: tests/test_ai_attribution.py::test_regime_conditioned_multipliers.

## D25. Approval expiry enforced at decision time, not just cycle start (2026-07-07)
Gap found during live probation review: the cycle-start expiry sweep in
execution.process_approval_queue only scans approvals with status='pending'.
If the human clicked Approve on an intent whose expires_at had already
passed (e.g. approving a 12:30 proposal at 15:20), the row flipped to
'approved' before any sweep saw it, and the next cycle would place the
order at a price quoted hours earlier. Fix lives in store.decide_approval —
the single chokepoint both the GUI (/api/approvals) and the CLI (approve
command) route through — which now checks expires_at when status='approved',
marks the approval AND the order 'expired', and raises ValueError. GUI maps
it to HTTP 409; CLI prints REFUSED. String comparison of ISO-with-tz
timestamps matches the existing convention in process_approval_queue.
Test: tests/test_risk.py::test_approve_after_expiry_refused.

## D26. Nightly git commit of dev/reports (2026-07-07)
Post-close job calls `_commit_reports()` (app.py, module level for
testability): if `dev/reports` changed, `git add + commit` best-effort and
audit `reports_committed`. Reports were accumulating uncommitted on a
machine with no backup besides git. Test: tests/test_pipeline.py.

## D27. Missed-scan watchdog (2026-07-07)
Scan/post-close jobs get `misfire_grace_time=1800` (laptop-wake tolerance)
and an APScheduler EVENT_JOB_MISSED listener that audits `scheduler_missed`
and desktop-notifies. A sleeping laptop silently skipping scans was
indistinguishable from a healthy no-trade day.

## D28. /api/health liveness endpoint + named scheduler jobs (2026-07-07)
No-broker-call probe returning mode, scheduler_running, and per-job
next_runs; scan jobs renamed to readable ids (`scan_HH:MM`, `post_close`).
Lets the daily check (and any future uptime monitor) verify the scheduler
without touching Robinhood. Test: tests/test_pipeline.py.

## D29. Notify when a scan queues approval intents (2026-07-08)
Three days running, intents queued at scan time expired unseen (D25 made
that safe but silent). scan_job now counts pending_approval entries in the
cycle summary and desktop-notifies. Two-line count over already-tested
summary data — no new test.

## D30. Live approval TTL cut to 6h (2026-07-08)
process_approval_queue places approved intents at their ORIGINAL limit/qty
from queue time — it re-runs broker review but never re-prices. With
live.yaml's approval_mode: all, every live order rides this path, so the
default 24h approval_timeout_hours allowed a market-ish fractional order to
fire at yesterday's price. Cheapest correct fix: live.yaml sets
approval_timeout_hours: 6, so a 09:45 intent approved by the 15:30 scan
still places same-session and anything older dies via the existing D25
sweep. Re-quoting at placement is the upgrade path if TTL ever needs to
grow. Config-only; default.yaml (paper) keeps 24h.

## D31. Silence MCP session-termination warning noise (2026-07-08)
Robinhood's MCP server returns 400 on the session-DELETE the mcp client
library sends at teardown, so every broker call logged "Session termination
failed: 400" (mcp/client/streamable_http.py warning) — ~2 lines per scan,
pure noise in server.log. Not our bug and harmless (each _call_async opens
a fresh session; nothing depends on clean termination). One-line fix: set
the mcp.client.streamable_http logger to ERROR in robinhood_mcp.py, with a
comment explaining why. Real transport failures still raise from the call
itself, so nothing is masked.

## D32 (2026-07-08): pending_approvals in /api/health; stop tracking data/server.log
Every scheduled-session daily check needed a raw sqlite query to see the
human approval queue; /api/health is the surface those checks already poll.
Added `pending_approvals` (local DB count via store.pending_approvals(), no
broker calls, so the endpoint stays a cheap liveness probe). Alternative
considered: separate /api/approvals summary endpoint — rejected, the GUI
already lists approvals and the health probe just needs the count.
Also: data/server.log was accidentally committed in the D28 commit
(7d3029c) and showed as perpetually modified; removed from tracking and
added data/*.log to .gitignore. Runtime logs are machine-local state like
the DB, not repo history.

## D33 (2026-07-08): guarded restart script (scripts/restart_live.sh)
Every scheduled session that lands a server-affecting commit repeats the
same manual dance: check ET clock, kill the nohup pid, restart, curl
health. That toil is error-prone in the one way that matters — a restart
during market hours would disturb the live broker session and can miss a
scheduled scan. New scripts/restart_live.sh encodes the whole procedure:
refuses on weekdays 09:30-16:30 ET (--force to override), kills the
existing "specforge --mode live serve" process, restarts via nohup into
data/server.log, and polls /api/health up to 15s before declaring success.
Holidays aren't checked — a holiday refusal is merely over-cautious, never
harmful. Alternative considered: fold this into install_service.sh /
launchd (which restarts on crash) — but launchd was a human decision never
taken, and this script is useful either way (launchd restart ≠ deliberate
"pick up new code now" restart).

## D25. Autonomous trading is the DEFAULT (2026-07-06, user directive)
User: "I need this system to run automatically… the default should be that the
agent trades as it wants to." Flipped `approval_mode` default threshold→`auto`
in default.yaml AND live.yaml (was day-one `all`). auto = governor never
returns REQUIRES_HUMAN_APPROVAL; the engine places every risk-approved order
itself. Still fully configurable (auto|threshold|all) via config or GUI Risk
tab — the confirm-trade feature is preserved, just off by default. Also, per
"any money in the account is fair game", live max_account_deployment 0.70→0.98
/ min_cash_reserve 0.30→0.02. The real bound is unchanged: time-step budget
($20/cycle, $50 abs cap) + position caps + kill switches. install_service.sh
live now hard-checks the triple gate before installing the autonomous service.

## D26. Control Center v3: truth layer + terminal UI + FIRST LIVE TRADE (2026-07-09)
User: no fake information, fail gracefully, explain the run model, Bloomberg-
weight UI. Shipped: specforge/health.py system_health() (mode, real broker
probe, heartbeat, market clock, readiness with ALWAYS-populated reasons);
heartbeats from serve/cron/manual-scan; `stonk tui`; scripts/
install_cron.sh (launchd ping run-model); terminal restyle (zero radius, mono
stack, amber/green tags, SIM prefix on every simulated $, per-panel FEED
ERROR guards, no emoji). FIRST AUTONOMOUS LIVE TRADE at 10:50 ET: BUY
0.033319 GE @ $359.86 avg (RH order 6a4fb551…, market order per fractional
rule D12), sized by the governor to the $12 neutral-regime cycle budget.
CLOSED GAP (D26 follow-up): orders table is now mode-tagged, mirroring
positions (D23). `mode` column added + ALTER-migrated; the migration backfills
pre-existing rows to 'live' only when they carry a broker_order_id (paper fills
never do), so the resting live GE order above stays visible to live-mode
reconcile. Executor stamps self.mode at record_order; Governor filters by mode
in recent_order_exists and orders_today (duplicate cooldown + daily caps);
reconcile and the approval-queue placement query filter by mode too. Regression:
tests/test_pipeline.py::test_paper_orders_invisible_to_live_mode.

## D27. GUI API-key + provider management (2026-07-09)
The AI enrichment key was env-only (OPENROUTER_API_KEY in .env), no GUI path.
Added a provider picker (OpenRouter / OpenAI / Anthropic / custom) + key field
to the Risk/Config page. All four providers speak the OpenAI chat-completions
shape (Anthropic via its OpenAI-compatible endpoint), so only base_url + key
change — the httpx call in ai.py is untouched. The key persists to ROOT/.env
(gitignored, chmod 600) via config.set_env_var and is applied to os.environ
live, so the next scan's fresh AIClient picks it up without a restart. ai.py
now reads AI_API_KEY first, falling back to OPENROUTER_API_KEY for back-compat.
Secrets never touch the DB and are never echoed back: GET returns only
{key_set, last-4 hint}; the audit logs provider/base_url but not the key.

## D34. V4: hypothesis layer + steering + model observatory (2026-07-09, user)
Plan: dev/V4_PLAN.md. Two AI-generated hypothesis tiers in the `hypotheses`
table: a rarely-changing NORTH STAR and a SHORT-TERM hypothesis (rotated every
short_term_max_age_days or on regime change). Current ones mirror to
data/hypotheses/*.md; every retired one archives, dated, to
data/hypotheses/archive/ (the user-facing log). The ONLY trading influence is
nodes/hypothesis.py emitting the stored active hypothesis's stances as
SignalEvents into the ensemble — weighted, attribution-measured, governor-
gated like any node — plus a ≤max_watchlist universe merge. AI never sizes or
places anything; generation is post-close/CLI only, strict-JSON validated,
discard-on-garbage (D14 posture). Feature is off by default
(hypothesis.enabled + ai.enabled required).

STEERING: strategic choices (hypothesis adoption, north-star changes, node
promotions, watchlist adds, risk suggestions) queue in the `steering` table
with options + an AI recommendation and a TTL. Trading NEVER blocks on them
(user directive: autonomous as long as there's money). Tiered expiry defaults:
hypothesis_adopt/watchlist_add auto-adopt the recommendation; north_star_change
/node_promotion/risk_suggestion keep the status quo (so nothing top-level or
risk-touching drifts silently; promotions still effectively need a human,
without ever blocking). Bootstrap exception: the FIRST north star auto-adopts
(no status quo exists). Risk suggestions apply through config.apply_override —
the same validated path as the GUI (refactored out of app.py) — so dangerous
values are rejected regardless of who triggers the apply.

GUI: "Risk & Budget" tab relabeled "Config" (data-p unchanged). Portfolio
value chart: `equity_intraday` throttled marks (from /api/status) merged with
daily scan marks via /api/portfolio_value (intraday supersedes the daily row's
synthetic 16:00 stamp for days it covers); time axis, $ gridlines, 1D/1W/1M/ALL.
Steering panel on Overview with countdown + "what happens if I do nothing".
New Model tab: /api/model aggregates every node (base weight × learned
multiplier = effective weight in the current regime, scorecard, 7d signal
count) rendered as an SVG flow network (data → nodes → ensemble → governor →
broker) — the model's learned shape at a glance. Vanilla JS/SVG per D2.

## D35. Dynamism pass: live pricing, hourly cycles, decision observability (2026-07-09, user)
User pain: one trade, zero movement, tokens burned invisibly. ROOT CAUSE was
not timidity — scans priced limit orders off the LAST DAILY CLOSE (ctx.close),
so live orders rested unfilled all day (the GE order). Fixes, in order of
expected APR impact, none of which dilute edge quality (min_final_score and
the flat-edge nodes stay untouched — "dynamic" must not mean RNG):
1. run_cycle now overlays LIVE quotes (QuoteService: stooq/yfinance, ~15min
   delayed — still beats yesterday) onto entry limit pricing, exit stop
   checks (stops fire intraday now), and paper broker marks. Injectable
   (live_quotes param) for tests; backtests stay lookahead-clean
   (refresh_data=False + no injection ⇒ old behavior, verified).
2. live.yaml: hourly scans (7/day, 09:35–15:30) — more decision points and
   faster reconcile of resting orders; max_daily_new_positions 3→6 (it was
   the binding constraint at 7 cycles/day; per-cycle budget, single-position
   cap, deployment cap, kill switches all unchanged).
3. Observability = the product answer to "zero movement": /api/today digest
   (scans, candidates, order outcomes, top veto reasons — from audit/orders,
   real or empty, never invented) rendered as the Today panel on Overview;
   news_sentiment now stores a per-run synopsis (kv news_synopsis, incl.
   already-priced passes — they explain no-trades) shown as "AI READ" with
   age label; active hypothesis one-liner alongside.
4. Indicator tiles on Overview (SPY vs 50/200sma, VIX zone, breadth,
   deployment) — computed from the same numbers regime.classify uses.
5. GUI: --amber #ffb000 → true orange #ff7a1a (user request).
No margin/borrowing anywhere (buys still capped by cash/budget/deployment).

## D36. Net P&L truth, decisions feed, per-purpose AI routing (2026-07-09, user)
1. NET P&L replaces portfolio value as the main chart. Equity deltas lie the
   moment a deposit lands (the user's $50 top-up rendered as "growth" and a
   fake +$49.97 day P&L). P&L is now computed ONLY from trading: realized
   (SUM trades.pnl) + unrealized (open positions vs live marks), stamped onto
   equity_intraday marks (new pnl column, ALTER-migrated). /api/pnl serves the
   series; day P&L in /api/status is pnl-mark-based, deposit-proof. Portfolio
   value stays available via /api/portfolio_value.
2. DECISIONS FEED (/api/decisions + Trading tab panel): every candidate the
   last cycle considered with the governor's verdict + reasons + result, plus
   all working orders (resting/relayed/pending approval). Answers "what is the
   agent deciding and what's queued" from candidates/audit/orders — raw trail,
   nothing invented.
3. PER-PURPOSE AI ROUTING (ai.models): bulk headline classification stays on
   a cheap model (MAX_HEADLINES 8→14 — read more, it's pennies); the
   hypothesis/strategy layer routes to a flagship reasoner
   (anthropic/claude-sonnet-4.5 default; x-ai/grok-4 one config field away)
   with reasoning-effort passthrough. Budgets: daily (existing) + NEW monthly
   ceiling (default $30, user band $10-50) + per-purpose monthly caps
   (hypothesis $20). All enforced in reserve-then-commit BEFORE any call.
   Full agentic tool-loop deliberately deferred: flagship + reasoning +
   curated context first; add tools when measured results demand them.
4. OPERATOR FIX: runtime GUI override had time_step_budget_abs_cap=3 →
   $1.80/cycle → "time-step budget exhausted ×25" → 1 fill from 296
   candidates today. Raised to 50 (+ max_daily_new_positions 6) via the
   validated override path, audit-logged. The Today panel's veto digest is
   what surfaced this — observability paying for itself same-day.
5. Model tab: min node-box height 34px (two text lines no longer overlap).

## D37. Fundamentals research node + engine P&L marks (2026-07-10, user)
FUNDAMENTALS NODE (nodes/fundamentals.py, enabled, experimental, weight 0.15):
company-published financials (yfinance info ratios + quarterly revenue/net-
income trend) → compact brief → ONE LLM call (purpose "fundamentals", routed
via ai.models, cheap default) → strict-JSON valuation verdict {valuation,
direction long|avoid|neutral, conviction, horizon, thesis, red_flags} →
deterministic SignalEvent. Long-only system: a sell/short view = avoid
(suppresses the long case). Analysis kv-cached 3 days/symbol (fundamentals
move quarterly — re-asking each scan is token burn); ETFs/indices skipped;
offline/backtest → [] (no point-in-time fundamentals feed = lies). Synopsis
kv feeds the Today panel's FUNDAMENTALS block. RELIABILITY: proven end-to-end
with a REAL OpenRouter call before enabling (AAPL→avoid PE38, GE→avoid,
NVDA→long; SPY skipped; $0.0005 for 3 names; cache re-run = zero spend), plus
offline unit tests (validation clamps, cache, degradation). Earnings-call
TRANSCRIPT ingestion deferred: no free reliable transcript source; the
published-numbers brief is the honest v1, swap the brief builder when a
transcript feed exists.
ENGINE P&L MARKS: run_cycle now stamps a pnl intraday mark every live/paper
cycle (realized + unrealized at current marks) — the Net P&L chart populates
even with no dashboard open. Root cause of "P&L not rendering": only ONE mark
existed (marks previously came only from GUI status polls), so the chart
showed its single-point placeholder.

## D38. Startup crash fix + double-click Stonk.app (2026-07-10, user)
CRASH: after a reboot, `run.command` (→ run.sh → `serve` with NO --mode →
defaults PAPER) crashed with "single position > 25% rejected". Root cause:
config_overrides is a mode-agnostic kv blob, but the GUI had stored live-mode
risk values (max_single_equity_position 0.30, time_step_budget_pct 0.8) that
are only legal under live.yaml's advanced_override. Loading them in paper mode
fails Config.validate(). (The smoke `scan` passed because cmd_scan loads config
WITHOUT overrides; only serve→current_config applies them.) FIX: current_config
now catches ConfigError, refuses the override, and keeps the SAFE committed
file config (audited config_override_rejected) instead of crashing. This is
STRICTER not weaker — the dangerous value is rejected exactly as validate
intends; validate itself and the set-time apply_override path are unchanged
(still raise). Upgrade path noted in code: per-key pruning if a mode ever has
a mix of safe+unsafe overrides worth partially keeping. Mode-scoped override
blobs are the "proper" fix, deferred (YAGNI until the shared blob bites again).
STONK.APP: double-click launcher at ~/Applications/Stonk.app. Deliberately NOT
py2app/pyinstaller — freezing fastapi/uvicorn/yfinance is fragile; the app is a
thin launcher for the existing .venv (laziest + most robust). Bundle = Info.plist
+ MacOS/Stonk (opens Terminal on scripts/stonk_app.sh) + Resources/stonk.icns.
Launcher is idempotent: attaches to a running server (opens browser, no double-
bind) else boots `--mode live serve` and opens the dashboard when healthy.
Icon: assets/stonk.svg → QuickLook render → iconutil .icns (no image deps).
Rebuild anytime with scripts/build_stonk_app.sh [dest].

## D41. Dual networks, immutable champions, official universe (2026-07-11)

The first MLP overfit unchanged rows in production research (rank-IC 0.043 →
0.0076 by epoch 22,315), proving that nonstop champion training is unsafe.
Specialist modules now form a separate analog-neural DAG whose edge weights are
differentiable while their equations remain explicit. A causal quantile TCN is
one specialist, not the ensemble itself. All training writes challengers;
champion replacement is atomic and validation/shadow-gated.

Stock discovery uses Nasdaq Trader listing files plus SEC CIK enrichment, not
web search. The free-data universe is labeled survivorship-limited and requires
forward shadow evidence before learned live influence. Research work is bounded,
non-AI, closed-market-only, and yields before trading.
## D42 (2026-07-11): truthful autonomy status and durable research actions

Trading heartbeat age is only a failure signal while the market is open.
Closed-market UI reports the scheduler/research worker separately. Manual
discovery, deep research, and holding training enqueue durable jobs; they use
the autonomous worker, never contact execution, and never bypass budgets or
champion gates. Research history breadth reaches 500 names before additional
global TCN trials, preventing repeated optimization of the original 30-name
sample.
