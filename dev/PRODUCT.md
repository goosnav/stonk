# PRODUCT.md — Stonk Terminal (the sellable product)

> Vision doc for turning Stonk Terminal from a personal engine into shippable
> software. Written 2026-07-06. Owner-priorities: the GUI *is* the product;
> the engine is the differentiator under it.

## What is being sold

A **local-first trading control center**: one screen where a retail operator
understands, tunes, and supervises an autonomous, risk-bounded trading engine.
Not a signals app, not a chatbot: an operations console with a real machine
behind it. Runs on the customer's computer, connects to *their* brokerage
(Robinhood Agentic MCP first), their data, their AI budget. We never touch
their money or keys.

**Tagline candidates:** "A hedge-fund ops desk for one person." / "The
switchboard for your trading machine."

## Target user

Technically comfortable retail trader/tinkerer with $500–$50k, who wants
automation but refuses black boxes. They will pay for: transparency (why every
trade), bounded risk they can *see*, and knobs that actually connect to the
engine.

## Product pillars (design tests every GUI change against these)

1. **Glanceable truth** — 10 seconds on the Overview tab answers: am I up or
   down, what is the machine doing next, is anything wrong (kill switches,
   degraded nodes, stale data), what regime are we in.
2. **Every number has provenance** — projections carry error bars + basis;
   quotes show source + age; node stats show sample size. No naked numbers.
3. **Controls are contracts** — every knob maps to an audited config override;
   dangerous values are rejected by the same governor as everywhere else.
4. **Live but honest** — live/delayed quotes labeled as such; the engine
   trades on its own scan cadence, the GUI never pretends to be an HFT tape.
5. **Local-first, key-safe** — secrets in .env/OS keychain, DB on disk,
   no phone-home. (Licensing, if any, must respect this.)

## Architecture for shippability

- **Connector registry** (`specforge/quotes.py` + `broker/`): ordered provider
  chain per data type, configured in `configs/default.yaml → data_sources`.
  Quotes: broker (real-time when connected) → stooq light CSV (delayed) →
  yfinance (fallback + indices). Each quote is stamped `{source, as_of}`.
  Adding a provider = one class with `get_quotes(symbols) -> {sym: Quote}`.
- **All GUI state via JSON API** (`app.py`) — the dashboard is one static HTML
  file consuming ~14 endpoints. A future React/Tauri shell can reuse the API
  unchanged. Do not couple engine internals to the GUI.
- **Broker connect as a product flow**: `/api/broker/connect` runs the OAuth
  probe in a background thread; `/api/broker/status` reports
  connected/accounts/error; the GUI renders it as a "Connect Robinhood" card
  with copy-pasteable next steps (whitelist env, funding). Bridge fallback
  stays documented for when RH blocks custom clients.
- **Packaging path (later)**: `pipx install stonk-terminal` → `stonk serve`;
  then a Tauri wrapper for a desktop app. Not started; do not build until the
  console is polished.

## GUI v2 spec (Control Center layout)

Single page, 5 tabs. Every section header carries a one-line plain-English
explainer (the "what am I looking at" test).

1. **Overview** — market strip (SPY/QQQ/IWM/DIA/VIX live w/ change%, source+age
   label); account card (equity, cash, day P&L, drawdown vs baseline); regime
   card with evidence; NEXT SCAN countdown + last scan summary; alert rail
   (kill switches, degraded nodes, pending approvals count, stale data);
   projection banner (APR ± interval + confidence + basis); equity chart;
   Monte Carlo fan.
2. **Trading** — open positions w/ LIVE marks and per-position P&L $/%, stop &
   time-stop columns; candidate table (score, horizon return ± interval,
   P(>0), nodes, thesis on hover); approvals queue with one-click decide;
   recent trades with exit reasons.
3. **Switchboard** — node cards: enabled, base weight, learned multiplier,
   status badge, n/expectancy/win-rate, degraded reason, per-node description,
   promotion proposals surfaced here with Approve (edits status override).
4. **Risk & Budget** — risk editor grouped (per-cycle budget / position caps /
   loss limits / approval policy / options) each with explainer + current
   $-equivalent at current equity; AI panel (enable, model, budget, spend,
   est/day); cost meter (friction bps, AI history).
5. **Activity** — audit log tail w/ event-type filter; scheduler health
   (last/next runs, errors); data freshness per symbol group.

## Live data requirements (v2 scope)

- `/api/quotes?symbols=` — batch quotes, 30s in-process cache, provider chain,
  each `{price, change_pct, as_of, source}`.
- `/api/market` — indices + VIX + breadth + regime + next_scan_at.
- Positions in `/api/status` marked with QuoteService (not last daily close).
- GUI polls: market/quotes 30s; status 10s; heavy panels 60s.

## Out of scope for v2 (explicitly, so nobody half-builds them)

Websocket streaming, multi-account, mobile, user auth/multi-tenant, cloud
hosting, React rewrite, payments/licensing. Each needs its own design pass.

## Monetization sketch (not engineering-blocking)

One-time license + optional update subscription; free paper-only tier;
"bring your own keys" for AI/data. Revisit after the console earns daily use.
