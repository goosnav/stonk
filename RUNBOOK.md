# Stonk Terminal RUNBOOK — keep it alive, know when it isn't

Operator manual for the live deployment (2026-07-10 reliability rebase).
Audience: John, Hermes, and any watchdog. Everything here is **read-only
diagnosis first**; the only mutating actions are the explicitly labeled
start/stop/reset commands.

## 1. What "healthy" means

Liveness ≠ health. A running process proves nothing about scans, data, or the
broker. The system separates four questions:

| Question | Where answered | Healthy looks like |
|---|---|---|
| Is the process serving HTTP? | `GET /health` | 200 `{ok, pid, uptime_s}` in <1s |
| Is the app healthy? | `GET /api/metrics` → `status` | `"ok"` |
| Why is it not trading right now? | `GET /api/health` → `readiness.reasons` | benign reasons only (market closed, paper mode) |
| What is it doing this second? | `GET /api/engine` → `state.phase` | `idle`/`sleeping` between cycles, pipeline phases during |

`status` is the **app-health rollup** monitors act on:

- `ok` — nothing needs anyone. *Not trading* because the market is closed or
  mode is paper still counts as `ok`.
- `degraded` — app alive, something needs the **operator**: broker
  disconnected, broker blocking entries (e.g. the investor-profile
  requirement), kill switch active, market data stale, or a fresh cycle
  error. **Restarting does not help degraded states** — every one of them
  persists in the DB/broker across restarts, by design.
- `stale` — no completed scan within the expected interval while the market
  is open. Restart may help.
- `error` — the scheduler is dead inside the serving process. Restart.

Alert text is sanitized (`specforge/health.py:_redact`): token-ish strings,
account-number-length digit runs, and long blobs are `[redacted]` before they
reach HTTP, logs, or reports.

## 2. Check it (read-only)

```bash
scripts/check_health.py            # one human line + alerts; exit code = verdict
scripts/check_health.py --json     # full verdict for Hermes/automation
scripts/install_service.sh status  # launchd state + the same verdict
.venv/bin/stonk tui --once --no-color   # full account/engine snapshot
curl -s http://127.0.0.1:8420/api/metrics | jq .   # raw monitor contract
```

Checker **exit codes** (the watchdog contract):

| Code | Meaning | Correct reaction |
|---|---|---|
| 0 | ok (including benign not-trading) | nothing |
| 1 | degraded | tell the operator; **never** auto-restart |
| 2 | app failure (stale scans / dead scheduler) | restart is reasonable |
| 3 | down (connection refused / timeout) | restart is reasonable |
| 4 | malformed response | investigate; treat as degraded |

The checker prefers `/api/metrics` (schema `stonk.metrics.v1`) and falls back
to `/api/health` on older builds, so it works against any running version.

## 3. Start / stop / restart — safely

The live server owns port **8420** and the shared DB `data/specforge.db`.
**Never run two live engines**; `stonk serve` refuses a taken port before its
scheduler starts, but don't tempt it.

| Action | Command | Notes |
|---|---|---|
| Start (foreground) | `.venv/bin/stonk --mode live serve` | the Stonk Terminal.app window does this |
| Start (login service) | `./scripts/install_service.sh live` | launchd: auto-restart on crash, 60s throttle |
| Restart live | `./scripts/restart_live.sh` | **refuses during market hours**; `--force` overrides |
| Stop (service) | `./scripts/install_service.sh uninstall` | |
| Stop (foreground/nohup) | `pgrep -f "stonk --mode live serve" \| xargs kill` | close the app window if that started it |

Restart safety invariants (why auto-restart cannot create unsafe trading):

- Kill switches, config overrides, and heartbeats persist in the DB — a
  restart re-reads them; it cannot clear a tripped switch.
- The live gate (`live_trading_enabled` + `LIVE_TRADING_ENABLED` env +
  `RH_ACCOUNT_WHITELIST`) is re-checked on every order path.
- Broker-side blocks (investor profile, suitability) live at Robinhood and
  survive anything local.
- Orders require broker review first; unknown warnings are hard blocks.

Crash-loop protection: launchd `ThrottleInterval` 60s bounds restart
frequency; the Hermes monitor additionally requires 3 consecutive bad polls
(≥90s) plus a 15-minute cooldown before it restarts anything, and only for
exit codes 2/3 — never for `degraded`.

## 4. Diagnosis flows

**Scans stale while market open** (`status: stale`):
1. `curl -s :8420/api/engine | jq .state` — wedged mid-phase? Which phase?
2. `tail -50 logs/runtime-live.log` and `tail -20 logs/audit-live.jsonl`
3. `/api/metrics → last_error` — most recent `scheduler_error` with age.
4. Off-hours restart: `./scripts/restart_live.sh`; market hours: `--force`
   only if the engine is truly wedged (a restart mid-cycle loses that cycle,
   nothing else — state is in SQLite).

**App failure vs broker failure vs data failure** — read `alerts`:
- *App*: `stale`/`error` status, `scheduler_error` in `last_error` → restart
  territory.
- *Broker*: `broker disconnected` / `broker blocking entries` → app is fine;
  fix is at Robinhood (auth, investor profile) or wait out a 429 storm. A
  restart changes nothing.
- *Data*: `market data stale (Nd old)` → `.venv/bin/stonk data` refreshes
  bars; the governor already vetoes entries on stale data, so this is
  degraded, not dangerous.
- *Safe-mode*: `kill switch active: <name>` → deliberate protection. See §5.

**Server down** (`exit 3`): check the Stonk.app window wasn't closed, then
`./scripts/restart_live.sh` (or let launchd/Hermes do it). After any restart:
`scripts/check_health.py` must return 0/1 within ~30s, and the GUI Engine tab
should show `sleeping` (closed) or a cycle within the scan interval (open).

## 5. What requires John (never automated)

- Resetting a **major** kill switch: `stonk reset-kill <name>` after reading
  the audit trail — cooldown switches clear themselves.
- The Robinhood **investor-profile questionnaire** (agentic account) — until
  completed in the RH app, the broker blocks every order and health shows
  `broker blocking entries`. No code change can fix this.
- Raising risk limits, cycle budget, approval mode, node promotions.
- Anything that would place, cancel, or modify a live order manually.

**No-live-order-testing boundary**: tests, smoke checks, monitors, and this
runbook's commands are read-only against live. Nothing in the test suite or
the checker can place trades; order paths are exercised only via the paper
broker and mocks (`tests/`). Do not "test" the live order path by sending a
real order — the first live order should come from the engine inside its
budget, reviewed by the broker, during market hours.

## 6. What Hermes can report (and how)

Poll `scripts/check_health.py --json` (or `/api/metrics` directly): app
status + alerts, uptime, pid, cycles today, errors today, last scan, open
positions count, last sanitized error. Portfolio numbers (equity, P&L,
positions) come from `/api/status` — unchanged contract. The Hermes-side
monitor `~/.hermes/scripts/stonk-health-monitor.py` implements the §2 exit-code
policy, snapshots `/api/status` every 5 minutes, and restarts only per the §3
crash-loop rules. Its verdict lands in
`~/.hermes/project-manager/state/stonk-checks.json`.

## 7. Log map

- `logs/audit-<mode>.jsonl` — rotating mirror of the SQLite `audit` table
  (source of truth); every cycle/order/risk decision.
- `logs/runtime-live.log` — server stdout/stderr (app window / restart script).
- `logs/service.log` — stdout/stderr when running under launchd.
- `data/backups/` — nightly SQLite backups, newest 14 kept.
