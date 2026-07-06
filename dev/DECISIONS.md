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

## D13. Async fills reconciled at cycle start (2026-07-06)
Live/bridge orders don't fill synchronously. Executor.reconcile() polls
resting/relayed orders each cycle via broker.poll_order() and creates
positions/trades through the same bookkeeping path as immediate fills.
