# Stonk Terminal tutorial — zero to live, safely

## 0. Requirements

- macOS/Linux with Python 3.12+ (Windows: `run.ps1`)
- Nothing else for paper mode. Live mode needs a funded Robinhood Agentic
  account. AI nodes need an OpenRouter API key (optional).

## 1. Install & smoke test

```bash
./run.sh
```

This creates `.venv`, installs dependencies, runs the offline test suite,
downloads market data if missing, runs one paper scan cycle as a smoke test,
and starts the GUI at **http://127.0.0.1:8420**.

Troubleshooting:
- `python3 not found` → install Python from python.org or `brew install python`.
- Data download partial → rerun `.venv/bin/stonk data --full` (Stooq
  rate-limits occasionally; the risk governor refuses to trade on stale data,
  so nothing unsafe happens either way).
- Port busy → `.venv/bin/stonk serve --port 8421`.

## 2. Understand the dashboard

- **Header**: equity, cash, day P&L, drawdown, current regime and the cycle
  budget (the max the engine may deploy this scan).
- **Projected strategy return**: bootstrapped APR with an 80% interval and a
  confidence label; the basis line tells you how many backtest/paper/live
  trades the number rests on. Low confidence = wide bars = believe accordingly.
- **Candidates**: each shows horizon return + interval + P(>0). The APR column
  is deliberately secondary.
- **Switchboard**: toggle nodes, edit base weights; "learned ×" is the
  self-improvement multiplier (bounded 0.3–2.0).
- **Risk controls**: editable within guardrails — dangerous values are rejected
  unless you set `advanced_override: true` in the config file yourself.
- **Approvals**: orders above the approval threshold (default 10% of equity)
  wait here up to 24h. `approval_mode`: `auto` (full autonomy), `threshold`
  (default), `all` (approve everything).
- **Kill switches**: routine incidents recover automatically. Broker rejection
  storms cool down for 30 minutes, daily/weekly loss limits clear on schedule,
  and drawdown protection resumes after its configured cooldown with a fresh
  baseline. Only a switch explicitly marked `MAJOR` has no recovery time and
  requires `stonk reset-kill <name>` after investigation. Exits remain allowed
  while new entries are paused.

## 3. Validate before believing

```bash
.venv/bin/stonk backtest --years 10 --tag v1
cat dev/reports/backtest_v1.json
```

Read `out_of_sample_30pct` before `overall`. Compare against
`benchmark_buy_hold_return`. The backtest writes analog trades into the live DB
so candidate error bars are grounded from day one.

Then let paper mode run for at least 2–4 weeks:

```bash
.venv/bin/stonk tui          # quiet terminal dashboard + continuous scheduler
# or: .venv/bin/stonk serve  # quiet GUI/headless server on port 8420
```

The scheduler evaluates continuously at the configured market-hours interval.
For automation, `.venv/bin/stonk --mode live tui --once --no-color` prints a
stable account/engine snapshot and exits without stopping an existing daemon.

## 4. AI nodes (optional)

```bash
# .env
OPENROUTER_API_KEY=sk-or-...
```

In the GUI: AI panel → enable, pick model (default deepseek chat, ~$0.25/M
input tokens), set daily budget. The cost meter shows estimated $/day per node
and actual spend. The ledger reserves a task's full cost up front and skips
cleanly when the budget is exhausted; 5 unparseable responses in a day disable
AI until tomorrow (trading continues deterministically). Enable
`news_sentiment` in the switchboard once the key works. `congress_trades` and
`insider` are free (public data) — enable them anytime; they carry small
weights by design.

## 5. Going live

Gate checklist (all must be true — the engine enforces the mechanical ones):

- [ ] Paper ran ≥2 weeks without scheduler exceptions (`/api/audit`, look for
      `scheduler_error`)
- [ ] Backtest OOS results reviewed and accepted by you
- [ ] `pytest` green
- [ ] You funded the Robinhood **Agentic** account with money you can lose
- [ ] `.env`: `LIVE_TRADING_ENABLED=true`, `RH_ACCOUNT_WHITELIST=<acct#>`

Then, to trade **autonomously** (the default):

```bash
.venv/bin/stonk --mode live serve            # runs now, scans on schedule
# or run it as a background service that starts at login + auto-restarts:
./scripts/install_service.sh live
```

- **Autonomous by default**: `approval_mode: auto` in `configs/live.yaml` means
  the engine places every risk-approved order itself — no confirm click. The
  time-step budget ($20/cycle, $50 hard cap), position caps, and kill switches
  are the bound, not a human. To gate trades instead, set `approval_mode` to
  `threshold` (only big orders wait) or `all` (every order waits) in
  `configs/live.yaml` or live from the GUI **Risk & Budget** tab.
- "Any money in the account is fair game": live is configured to deploy up to
  98% of the balance. Kill switches (daily/weekly loss, drawdown) still halt it.
- Live config uses `broker: robinhood_mcp` (your verified connection) and a
  **$50 hard budget per scan cycle**. Raise `time_step_budget_abs_cap` as the
  account grows.
- First run opens Robinhood's OAuth page in your browser (tokens cached in
  `~/.specforge/`, chmod 600).
- **If OAuth fails** with a registration/allowlist error, Robinhood doesn't
  accept custom MCP clients yet. Switch `configs/live.yaml` to
  `broker: robinhood_bridge` and schedule a Claude Code session (which has the
  Robinhood connector) to run [scripts/bridge_prompt.md](scripts/bridge_prompt.md)
  after each scan time. The engine still makes every decision; the session is
  dumb transport with its own safety rules.
- Robinhood constraint worth knowing: fractional-share orders must be market
  orders in regular hours (encoded in the adapter). Whole-share orders go as
  limit orders.

## 6. Monitoring & keeping it alive

[RUNBOOK.md](RUNBOOK.md) is the full operator manual. The short version:

```bash
scripts/check_health.py               # one-line verdict; exit 0 ok / 1 needs-operator
                                      # / 2 app-failure / 3 down (watchdog contract)
scripts/install_service.sh status     # launchd state + the same verdict
curl -s :8420/api/metrics | jq        # monitor contract: uptime, cycles, last error
```

`status: ok` while `trading: no` is normal (market closed, paper mode).
`degraded` means the app is fine but something needs *you* — broker auth,
kill switch, stale data — and restarting won't help. Only `stale`/`down`
justify a restart (`./scripts/restart_live.sh`, market-hours guarded).
Everything above is read-only and safe to run against the live server.

## 7. Watch the self-improvement loop

Nightly post-close job: marks equity, updates node scorecards, moves weight
multipliers (shrunk toward zero edge until ≥20 trades), auto-disables nodes
with clearly negative live expectancy (n≥30), and records promotion
*proposals* — promotions themselves are yours to approve by editing the node's
`status` in config after reviewing the scorecard.

CLI equivalents: `stonk status` (JSON overview), `stonk approve|reject
<intent_id>`, `stonk reset-kill <name>`, `stonk bridge-dump|bridge-report`.

## 8. What to expect (honesty section)

The projected-APR banner is a measurement, not a promise. The strategy stack
(momentum + earnings drift + sector confirmation + regime gate + strict loss
control) is the retail-plausible edge set; the system's real job is to *find
out* which nodes carry weight and to keep position sizes survivable while it
learns. Expect: many "no trade" cycles (feature, not bug), small losses often,
drawdown switches occasionally tripping, and the projection's confidence label
staying "low" until real trade count builds. If live expectancy is negative
after a fair sample, the correct move — which the system will propose — is
smaller size or off, not doubling down.
