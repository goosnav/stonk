# REBASE_PLAN — Stonk Terminal

## Verdict

**The deepest, most defensible codebase in the portfolio** — 6.3k LOC across 40 modules, 26 passing tests, a real risk governor, live Robinhood MCP OAuth verified, backtests with out-of-sample results better than in-sample. Two honest paths:

1. **Use it for your own business (primary):** run it as your own capital allocator. Its value is compounding evidence — every paper/live trade improves the scorecards. This is the plan the repo itself already encodes (dev/ROADMAP.md).
2. **Sell it (secondary, later):** *not* as "AI trader that makes money" (compliance and credibility suicide), but as **"the risk-governed harness for people who want to automate their own strategies"** — a prosumer open-core tool: engine free/source-available, paid tier = hosted dashboard, broker adapters, alerting, and the attribution/scorecard system. Comps: QuantConnect, Composer ($24–32/mo), NinjaTrader. Selling the *harness* avoids selling performance claims.

Also: **`business-suite/trading-algorithm-lab` is a shallow overlap — fold any unique fixture/backtest ideas in here and archive it.** One trading codebase.

## Current flaws

1. **Zero live-trade evidence.** Everything real (node weights, attribution, kill-switch behavior under stress) is untested against reality. The repo is honest about this; the plan must make soak time the top priority, because every other improvement is downstream of real data.
2. **Single-broker coupling risk.** Robinhood's Agentic Trading MCP is brand-new and could change/vanish. The broker adapter seam exists — but there's only one live implementation. An Alpaca adapter (stable, paper-trading-native, free API) is cheap insurance and is also the adapter any future customer would actually want.
3. **Data-source fragility.** Quote chain is broker→stooq→yfinance. yfinance is a scraper that breaks routinely; stooq is EOD-ish. No paid/reliable market-data seam yet. Fine at $50 equity; not fine at real size or for customers.
4. **Local-only, single-machine ops.** launchd service + SQLite + macOS notifications. If the Mac sleeps, the engine sleeps. For serious personal use it should run on an always-on box (cheap VPS or home server) with remote read-only dashboard + push alerts.
5. **No secrets hygiene story for distribution.** Tokens in `~/.specforge/` is fine for you; a shipped product needs keychain/encrypted storage and a proper first-run auth flow.
6. **GUI is operator-grade, not customer-grade.** 5 tabs of dense provenance is perfect for you. If the sell path is taken, it needs an onboarding path ("connect broker → pick risk profile → watch paper mode") and sane defaults.
7. **License is a placeholder.** Blocks both distribution and open-core strategy. Decide: BSL/fair-source (protects a hosted offering) vs MIT engine + closed dashboard.
8. **Node breadth > node depth.** 8 signal families but the scorecards can't yet say which deserve to exist. Resist adding node #9 before attribution has killed or promoted the current 8.

## Improvements / next-level features

- **Alpaca broker adapter** (portability proof + the customer-relevant broker).
- **Reliable data seam:** one paid/free-tier provider (Polygon/Alpaca data) behind the existing quotes chain.
- **Remote ops:** headless deploy on a VPS, read-only web dashboard behind auth, Telegram/push alerts for kill switches and approvals (you already have Hermes/Telegram infra — reuse it).
- **Walk-forward re-validation job:** scheduled re-backtest with fresh data, alert on regime/edge decay. This is the "self-improving" claim made real and it's also the flagship demo feature for buyers.
- **Strategy-as-config:** nodes already emit forecasts behind a uniform interface — expose a documented node API + example custom node. That's the open-core hook ("bring your own signal, keep our governor").
- **Tear-sheet export:** monthly PDF/HTML performance + attribution report. Personal accountability now; marketing artifact later.

## Sprint plan

### Sprint 1 — Evidence engine (1 week + calendar time)
- Start the paper soak formally: engine running 24/7 on an always-on machine, nightly backup verified, uptime monitored.
- First live order at the $50 cap (human-approved, market hours) to close D22.
- Telegram alerting for kill switches, approvals, and daily P&L summary.
- Fix the LICENSE placeholder (decide open-core posture now — it shapes everything later).

### Sprint 2 — De-risk the platform (2 weeks, parallel with soak)
- Alpaca adapter (paper first) passing the same test suite as the RH bridge.
- Data provider seam + one reliable source; provenance UI already supports showing it.
- Secrets: move tokens to encrypted storage with a documented first-run flow.

### Sprint 3 — Learning loop goes live (2 weeks, gated on ~1 month of trade data)
- Sprint C items from dev/ROADMAP.md: scorecard-driven weight updates on real fills, auto-disable of negative nodes, human-gated promotions exercised end-to-end.
- Walk-forward re-validation cron + decay alerts.
- Monthly tear-sheet export.

### Sprint 4 — Optional productization (only if soak results justify it)
- Custom-node API docs + example repo.
- Onboarding flow (connect broker → risk profile → paper mode default).
- Landing page selling the harness (risk governor, attribution, broker adapters) with your own tear sheets as demo data — no performance promises.
- Pricing test: source-available engine + $29–49/mo hosted dashboard/alerts.

## Definition of shippable
For you: it trades a small live allocation unattended for a month with zero governor violations and a readable monthly tear sheet. For sale: a stranger connects Alpaca paper, picks a risk profile, and watches governed paper trades within 15 minutes — without reading AGENTS.md.
