# Changelog

## 0.1.0 — 2026-07-06 (initial build)

Built, validated, and live-connected in one day. Highlights:

- Phase 1: engine spine — models, store, data, paper broker, risk governor, execution, CLI
- Phase 2-4: clock injection, backtester, GUI (FastAPI + dashboard), Monte Carlo, Robinhood MCP adapter, bridge broker + reconcile
- Phase 5 + fixes: attribution, AI layer, exotic nodes, options overlay, drawdown cooldown (D15), docs + launchers
- D17 drawdown baseline reset, aggressive config, backtest mode inheritance, run.sh verified
- Validation gate passed: v3 backtest (CAGR 7.6%, Sharpe 0.76, OOS>IS), reversal defaulted off, ROADMAP + run summary for handoff
- Sprint E backend: QuoteService provider chain, /api/quotes|market|broker/*|proposals, live position marks, tz-safe daily order counts, RH probe()
- Control Center v2: tabbed dashboard w/ live market strip + explainers + broker connect
- Sprint E progress notes
- Sprint E polish: freshness panel + endpoint, promotion apply button w/ validated status override, node degraded badges
- Probe staleness handling (interrupted state), VIX rounding, GUI error detail row
- Per-thread sqlite connections (D21) — fixes random 500s under parallel dashboard load
- LIVE: standalone RH MCP client verified against real account (D22) — data envelope unwrap, nested buying_power, quote shape, .env autoload
- Live review dry-run passed
- launchd service installer, README screenshots, progress sync
- RH adapter read cache (30s TTL, invalidated on placement)
- Mode-tagged positions (paper|live isolation in shared DB) with migration + test
- D23 decision entry
- Untrack session-local ralph-loop state
- Ops hardening: desktop notifications on kill switches/scan failures/proposals
- gitignore backups dir

Full decision history: dev/DECISIONS.md (D1–D23).
