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
