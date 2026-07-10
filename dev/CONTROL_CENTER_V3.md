# Control Center v3 — truth, run-model, terminal-grade UI

> Execution plan written so a weaker model can follow it step-by-step and still
> get a great result. Do the chunks IN ORDER; run `.venv/bin/pytest tests/ -q
> --ignore=tests/test_gui.py` and commit after EACH chunk. Never skip the
> acceptance check at the end of a chunk.

## Why (user directives, 2026-07-07)

1. **No fake information.** The dashboard showed paper-sim numbers ($999.84)
   styled identically to real money. Paper is a SIMULATION and must scream it;
   live must show real Robinhood state or an explicit red DISCONNECTED with
   the actual error string. "Fail gracefully" = every failure is visible and
   explained, never silent.
2. **It must actually trade the Robinhood account.** Autonomous by default
   (D25), confirm-trade configurable. Money in the account is fair game.
3. **Run model must be explicit and configurable**: persistent daemon AND/OR
   cron-style pings, explained in the GUI, plus a terminal TUI to watch it.
4. **Bloomberg-terminal weight class UI**: angular (zero border-radius), dense,
   monospace-first typography, amber/green-on-black, uppercase micro-labels,
   tabular numerals. Kill the "AI slop" rounded-card look. No emoji as icons —
   text tags ([LIVE], [SIM], [AI]) only.
5. **Broker layer stays pluggable** (paper / robinhood_mcp / robinhood_bridge
   today; more later). Surface the active adapter and its health in the GUI.

## The truth model (chunk 2 implements this)

One endpoint owns reality: **`GET /api/health`** returns:

```json
{
  "mode": "live|paper",                  // what THIS process trades
  "broker": {"adapter": "robinhood_mcp", "connected": true|false,
              "detail": "<error string when false>", "as_of": "..."},
  "engine": {"last_scan_at": "...", "last_scan_cycle": "...",
              "next_scan_at": "HH:MM ET", "scheduler_alive": true|false,
              "heartbeat_age_s": 42},
  "market": {"open": true|false, "session": "regular|closed|weekend"},
  "data": {"newest_bar": "YYYY-MM-DD", "stale_symbols": 0},
  "readiness": {"trading": true|false, "reasons": ["..."]}   // WHY not, always
}
```

Rules:
- `readiness.reasons` is NEVER empty when trading=false. Examples: "mode is
  paper (simulation)", "broker disconnected: <err>", "market closed",
  "kill switch: drawdown", "no scan heartbeat in 8h — is the service running?".
- scan_job writes kv `heartbeat` = {at, cycle_id, mode} after EVERY cycle
  (also written by `stonk scan` so cron mode heartbeats too).
- Broker connectivity for health = cached probe (60s TTL) of a cheap read;
  failure stores the exception string. Never let health throw — degrade to
  `"connected": false, "detail": str(e)`.

## GUI truth rules (chunk 3)

- Full-width **status bar** (top, always visible): `MODE` tag ([LIVE $] green /
  [SIMULATION] amber reverse-video), broker tag ([RH CONNECTED] / [RH DOWN:
  err…] red), market clock tag, heartbeat tag ("ENGINE 42s ago" / "ENGINE
  DEAD 9h" red), readiness line when not trading: "NOT TRADING — reason".
- Paper mode: every $ panel gets a `SIM` prefix tag; header equity reads
  `SIM $1,000.00`. Live mode: real numbers, no tag.
- Any fetch error in a panel renders `-- FEED ERROR: <msg> --` in that panel
  (red, monospace), never a stale silent value. Wrap each refresh fn.
- Empty ≠ error: "no positions" is a normal state with plain copy.

## Run model (chunks 2 + 3)

Two supported ways to run — document BOTH in the GUI "OPS" panel with exact
copy-paste commands and which is active:

| Model | How | When to prefer |
|---|---|---|
| Daemon | `specforge --mode live serve` or `./scripts/install_service.sh live` (launchd: login start + crash restart). Scheduler fires 09:45/12:30/15:30 ET + post-close. GUI always available. | Default. Mac stays on. |
| Cron ping | `./scripts/install_cron.sh live` → launchd StartCalendarInterval jobs run `specforge --mode live scan` at the same times (+ post-close attribution via `scan --post-close`). No resident process; GUI only when you start it. | Minimal footprint; machine that sleeps (launchd runs missed jobs on wake). |

Both write the same heartbeat, so /api/health can tell you which one is alive.
Detection: kv heartbeat.source = "serve"|"cron".

## TUI (chunk 2)

`stonk tui [--mode live]` — plain ANSI, no deps: clears screen every 5s,
renders: status line (mode/broker/heartbeat/readiness), equity + day P&L,
open positions table, last 8 audit events, next scan time. Ctrl-C exits. It
reads the DB + /api/health logic directly (no server required) so it doubles
as the "is this thing alive" probe.

## Bloomberg-grade design tokens (chunk 3)

- Font stack: `"IBM Plex Mono","JetBrains Mono","SF Mono",Menlo,Consolas,
  monospace` everywhere (self-hosted fallbacks only, no CDN); tabular numbers
  `font-variant-numeric: tabular-nums`.
- Palette (semantic tokens, WCAG AA on black): bg `#0a0a0a`, panel `#111214`,
  hairline `#2a2d33`, text `#e6e6e3`, dim `#8a8f98`, amber `#ffb000` (accents,
  SIM), green `#00d964` (up/ok/LIVE), red `#ff4b4b` (down/error), cyan
  `#4bd8ff` (links/actions). NO gradients, NO shadows, NO border-radius (0
  everywhere), 1px hairline borders.
- Density: 12px base data font, 11px uppercase `letter-spacing:.08em` labels,
  4px vertical cell padding, panels packed in a 12-col grid with 1px gaps on a
  hairline background (grid lines like a terminal).
- Tags: inline `[TEXT]` blocks with 1px border, uppercase; reverse-video
  (bg=color, text=black) for the critical ones (LIVE/SIM/DOWN).
- Keep: 5 tabs, all existing panels/data, explainer lines (restyle to 11px dim).
- Interaction: keyboard focus visible (2px amber outline), buttons are square
  uppercase text buttons, hover = reverse video. No emoji anywhere.

## Chunks

1. **This doc.** [done when committed]
2. **Backend truth + ops**: /api/health, heartbeat writes in scan_job AND cli
   scan, `cmd_tui`, `scripts/install_cron.sh`, `scan --post-close` flag.
   Accept: `curl /api/health` shows readiness with honest reasons in paper
   mode; `stonk tui` renders one frame offline; tests green.
3. **dashboard.html v3**: status bar + truth rules + OPS panel + full restyle
   per tokens above. Accept: `node --check` on extracted JS, id cross-check
   script passes, Playwright smoke (tests/test_gui.py) passes, screenshot
   captured to dev/reports/gui_v3.png.
4. **Go live**: restart service in live mode (`pkill -f "specforge serve" &&
   nohup .venv/bin/stonk --mode live serve --port 8420 &`). Accept:
   /api/health shows mode=live, broker connected, readiness reflects market
   clock; first scheduled scan places real orders (watch audit log).
5. **Docs**: TUTORIAL "How it runs" section mirrors the OPS panel; PROGRESS +
   DECISIONS entry (D26). Accept: committed.

## Do-not list (for the weaker model)

- Do NOT touch specforge/risk.py, engine.py order flow, or broker adapters
  beyond adding the cheap health probe helper.
- Do NOT add npm/build steps, external fonts/CDNs, or new Python deps.
- Do NOT remove any existing /api endpoint (the TUI and tests use them).
- Do NOT invent numbers in the GUI: every value comes from an API field; if
  the field is missing render `--` plus the FEED ERROR rule above.
