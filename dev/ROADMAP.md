# ROADMAP — how to bring SpecForge to fruition

> **Audience: any agent (or human) picking this up cold.** Follow the sprints
> in order. Each has exact commands, acceptance criteria, and a DO-NOT list.
> Read [PROGRESS.md](PROGRESS.md) (current state), [DECISIONS.md](DECISIONS.md)
> (D1–D18, why things are the way they are), and [ARCHITECTURE.md](ARCHITECTURE.md)
> (module map) before writing any code. The canonical spec is [../AGENTS.md](../AGENTS.md).

## Ground rules (apply to every sprint — do not violate)

1. **Never weaken `specforge/risk.py`.** The governor, time-step budget, kill
   switches, and option validation are the product. Any change there needs a
   new test in `tests/test_risk.py` proving the constraint still holds.
2. **Nodes emit forecasts, never orders.** If you find yourself importing a
   broker inside `specforge/nodes/`, stop — you're doing it wrong.
3. **All external data must be as-of correct.** New data source ⇒ store what
   date the information became PUBLIC (filing/publication date), and only use
   rows `<= ctx.as_of`. Backtests must be able to replay it or the node must
   return `[]` when `ctx.offline` is true.
4. **Run `.venv/bin/pytest tests/ -q` after every change.** 25 tests, all
   offline, <2s. A red test = you broke an invariant, revert and rethink.
5. **Log decisions.** Any behavioral change gets a numbered entry in
   `dev/DECISIONS.md` and a line in `dev/PROGRESS.md`.
6. **Never commit secrets.** `.env` is gitignored; keep it that way.
7. Use `.venv/bin/python` / `.venv/bin/specforge` (no uv on this machine).

## Current state, one paragraph (2026-07-06)

All five build phases are CODE-COMPLETE and tested (25 offline tests green).
Validation gate passed: 10-year walk-forward backtest (costs included) shows
CAGR 7.6%, Sharpe 0.76, maxDD 16.8%, profit factor 1.39, out-of-sample better
than in-sample (D18). The GUI runs at `http://127.0.0.1:8420` via
`.venv/bin/specforge serve`. 1,756 backtest analog trades feed live error
bars. What remains is OPERATION, not construction: run paper for weeks, then
tiny live, then let attribution learn, then (only if measured) widen.

## Sprint A — Paper campaign (start now, runs ~2–4 weeks wall-clock)

Goal: the closed loop runs unattended on the schedule and accumulates real
paper trades.

1. Start (or confirm running): `nohup .venv/bin/specforge serve --port 8420 &`
   — the scheduler scans Mon–Fri 09:45/12:30/15:30 ET + 16:30 post-close.
   Better: install it as a launchd/systemd service so it survives reboots.
2. Daily check (or automate as a cron/scheduled Claude session):
   - `curl -s localhost:8420/api/status | python3 -m json.tool` — look at
     `kill_switches` (investigate any), `equity`, `regime`.
   - `curl -s "localhost:8420/api/audit?limit=50"` — grep for
     `scheduler_error` and `node_degraded`. Zero scheduler_errors expected.
3. Weekly: check `/api/nodes` — `n_trades` should grow; expectancy columns
   fill in as round-trips close.
4. Enable the free exotic nodes after the first clean week: in the GUI
   switchboard turn on `congress_trades` and `insider` (they carry small
   weights; they can only nudge).
5. (Optional) AI: put `OPENROUTER_API_KEY` in `.env`, enable AI + the
   `news_sentiment` node in the GUI, budget $1/day. Verify the cost meter
   shows spend and that a day of parse failures disables AI, not trading.

**Accept when:** ≥10 paper round-trips closed, zero scheduler_errors in a
week, weight multipliers updating after post-close (see `weights` table or
`/api/nodes` "learned ×").
**Do NOT:** raise risk limits "because paper is going well" — that decision
belongs to Sprint C with data.

## Sprint B — Robinhood live probation (needs the human)

Prereq: Sprint A accept criteria met, and the human has funded the Robinhood
**Agentic** account.

1. Human steps (cannot be done by an agent): fund account; `cp .env.example
   .env`; set `LIVE_TRADING_ENABLED=true`, `RH_ACCOUNT_WHITELIST=<acct#>`.
2. OAuth probe: `.venv/bin/specforge --mode live status`. First call opens a
   browser for Robinhood login. Two outcomes:
   - **Works** → you'll see account JSON. Proceed with `broker: robinhood_mcp`.
   - **BrokerAuthError mentioning registration/allowlist** → Robinhood doesn't
     accept custom MCP clients. Edit `configs/live.yaml`: `broker:
     robinhood_bridge`, then schedule a Claude Code session (with the
     Robinhood connector enabled) to run [../scripts/bridge_prompt.md]
     (../scripts/bridge_prompt.md) at ~09:50/12:35/15:35 ET each weekday.
     The bridge protocol is tested (tests/test_bridge.py) — follow the prompt
     file exactly.
3. Read-only soak for 2–3 days: run `--mode live status` daily; confirm
   account/position numbers match the Robinhood app. No orders yet (the $50
   `time_step_budget_abs_cap` in live.yaml plus approval thresholds keep any
   accident tiny anyway).
4. First live order: temporarily set `risk.approval_mode: all` in
   configs/live.yaml, run `.venv/bin/specforge --mode live scan`, approve the
   single queued intent in the GUI, verify the fill appears in both the
   Robinhood app and `/api/status` next cycle (reconcile step).
5. Restore `approval_mode: threshold`, start `.venv/bin/specforge --mode live
   serve`, and let it run at the $50/cycle cap for ≥2 weeks.

**Accept when:** ≥5 live fills reconciled correctly (engine positions ==
broker positions), no duplicate orders, kill switches behave.
**Do NOT:** raise the $50 cap during probation; skip broker review; trade
options (locked at this account size anyway).

## Sprint C — Learn and scale (data-driven, ongoing)

1. After ~30 live/paper round-trips, read `/api/nodes` and the nightly
   `promotion_proposals` (kv / audit). Apply promotions by editing node
   `status` in configs/default.yaml — a HUMAN approves, per D-rules.
2. Prune what attribution flags (it auto-disables clear losers; you confirm).
3. Scaling decision (human): if live expectancy is positive over a fair
   sample, raise `time_step_budget_abs_cap` gradually (50 → 150 → 400…),
   NOT the percentage limits. The aggressive.yaml profile failed
   out-of-sample (D18) — do not adopt it wholesale; if desired, paper-trade
   it in parallel first (`--mode aggressive`, separate db_path!).
4. When account equity × 1.5% ≥ $75 (~$5k), the options overlay auto-unlocks.
   Before first option trade: implement RH option order tools in
   `broker/robinhood_mcp.py` (`review_option_order`/`place_option_order`,
   mirroring the equity methods — schemas via runtime tool discovery) and add
   a paper test. Until then options only work on the paper broker.

## Sprint E — Control Center v2 (product-ization; spec in [PRODUCT.md](PRODUCT.md))

Goal: the GUI becomes the sellable product — glanceable, live, self-explaining.
Do these in order; each step is small, committable, and independently useful.

1. **QuoteService** (`specforge/quotes.py`): provider chain broker→stooq→
   yfinance from `configs data_sources.quotes`, 30s cache, `{price,
   change_pct, as_of, source}` per symbol. Acceptance: `/api/quotes?symbols=
   SPY,QQQ` returns live-ish prices with source labels; offline → clear error
   fields, no crash. [DONE 2026-07-06]
2. **New endpoints** in app.py: `/api/market`, `/api/quotes`,
   `/api/broker/status`, `/api/broker/connect` (background OAuth probe →
   kv `broker_probe`), `/api/proposals`. Status positions marked with live
   quotes. Acceptance: curl each, 200 + sane JSON. [DONE 2026-07-06]
3. **Dashboard v2** per PRODUCT.md layout (tabs, market strip, countdown,
   alert rail, explainers, node descriptions, Connect Robinhood card).
   Acceptance: every section has an explainer line; positions P&L uses live
   marks; broker card shows connect flow states. [DONE 2026-07-06 — first pass]
4. **Polish pass** [DONE 2026-07-06]: thesis tooltips on candidates, data
   freshness panel on Activity tab (+ /api/freshness), $-equivalents next to
   risk percentages, promotion "apply" button (validated status override),
   node degraded badges (from audit), empty-state copy. Verified: endpoints
   200, invalid status rejected 400, JS node --check clean, DOM id
   cross-check clean, 25 tests green.
   Remaining known-unverified: no human has SEEN the page render (Chrome
   extension offline for the agent). First look may find cosmetic nits only —
   the API contract and JS syntax/id references are machine-verified.
5. **Packaging spike** (later): pipx entry point already exists
   (`specforge serve`); document + screenshot for a landing page.

## Sprint D — Hardening backlog (do opportunistically, lowest priority first)

- launchd/systemd service files for `specforge serve` (survive reboots).
- Nightly `git commit` of dev/reports + a DB backup (sqlite `.backup`).
- Regime-conditioned weight multipliers in attribution.py (data exists in
  scorecards `by_regime`; apply multiplier per regime once n≥30 per cell).
- Earnings drift: replace yfinance earnings with a point-in-time source
  (EDGAR 8-K parsing) so it can be backtested honestly.
- Alpaca adapter (`broker/alpaca.py`, REST) as a second live venue.
- GUI: render `promotion_proposals` and node `degraded_reason`.
- Watchdog: alert (push/email) when a kill switch trips or the scheduler
  misses a scan.

## What success looks like

The machine says "no trade" most days, deploys inside its budget when the
ensemble has conviction, cuts losers mechanically, and its projected-APR
banner converges toward its realized curve with a confidence label that has
earned the word "medium". Realistic expectation from the validated stack:
high-single-digit to low-teens APR at ~10% vol with sub-20% drawdowns —
upside beyond that must come from NEW measured nodes, never from deleting
safety rails.
