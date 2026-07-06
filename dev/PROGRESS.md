# Progress / handoff state

> Update this file whenever a module lands or a decision changes.
> To pick up: read PLAN.md (phases) → ARCHITECTURE.md (module map) → this file (what exists).

## Status: Phase 1 (spine) — IN PROGRESS

### Done
- [x] Repo scaffold: git init, pyproject.toml, .gitignore, .env.example
- [x] `.venv` created (python3.13 -m venv .venv), deps installed: numpy pandas fastapi
      uvicorn httpx apscheduler pyyaml pytest mcp yfinance
- [x] configs/: default.yaml (full switchboard + risk + AI config), paper.yaml, live.yaml
- [x] specforge/models.py — all boundary dataclasses
- [x] specforge/config.py — layered load, dangerous-value rejection, live triple-gate
- [x] dev/ docs: PLAN.md (approved plan), ARCHITECTURE.md, DECISIONS.md, this file

### Next (Phase 1 remainder, in order)
- [ ] specforge/store.py — sqlite schema + audit + accessors
- [ ] specforge/data.py — Stooq/yfinance ingestion + MarketContext (as_of slicing)
- [ ] specforge/broker/base.py + paper.py
- [ ] specforge/nodes/base.py + registry
- [ ] specforge/risk.py — governor (time-step budget, caps, kill switches, dup orders, staleness)
- [ ] specforge/execution.py
- [ ] specforge/engine.py — the scan-cycle orchestrator (see ARCHITECTURE.md pipeline)
- [ ] specforge/cli.py — scan/paper/status
- [ ] tests/ for governor + config + no-lookahead
- Exit criteria: `specforge paper` completes a full cycle on real downloaded data;
  audit log reconstructs it; risk tests pass.

### Then
Phase 2 (alpha nodes + backtest) → 3 (GUI) → 4 (Robinhood live) → 5 (self-improvement/AI/options).
Full definitions in PLAN.md.

## Environment notes
- Run everything with `.venv/bin/python` / `.venv/bin/pytest` (no uv on machine).
- No API keys needed until Phase 5 (AI) / Phase 4 live (Robinhood OAuth interactive).
- DB will live at data/specforge.db (gitignored).

## Known risks / open questions
- Robinhood MCP custom-client OAuth may be allowlisted → bridge fallback designed (D6).
- yfinance earnings/fundamentals endpoints are flaky → earnings_drift and
  quality_value nodes must degrade gracefully (node reports "degraded", weight 0).
- Stooq occasionally rate-limits burst downloads → data.py sleeps between symbols on 4xx.
