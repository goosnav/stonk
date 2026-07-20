#!/usr/bin/env bash
# Stonk Terminal launcher: setup → verify → serve. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"
# macOS soft FD limit is 256 — a long-running server (yfinance thread pools,
# SQLite WAL, sockets) exhausts it in days (2026-07-19: Errno 24 broke quotes,
# the broker probe, and yfinance's own cache DB). Raise it for this process.
ulimit -n 4096 2>/dev/null || true

PORT=8420
URL="http://127.0.0.1:$PORT"

# Already serving? (e.g. the live server from Stonk Terminal.app, or a prior
# run.) Open the dashboard and get out of the way — never rerun setup/tests or
# fight for the port. This is the double-click contract: the website appears.
existing=$(curl -sf --max-time 3 "$URL/api/version" 2>/dev/null || true)
if [[ "$existing" == *'"version"'* && "$existing" == *'"mode"'* ]]; then
  echo "» Stonk Terminal already running at $URL — opening dashboard"
  open "$URL" 2>/dev/null || true
  exit 0
fi

PY=python3
command -v $PY >/dev/null || { echo "python3 not found — install Python 3.12+"; exit 1; }

if [ ! -x .venv/bin/python ]; then
  echo "» creating .venv and installing dependencies…"
  $PY -m venv .venv
  .venv/bin/pip install -q --upgrade pip
fi
.venv/bin/pip install -q -e ".[dev]"

echo "» running offline test suite…"
.venv/bin/pytest tests/ -q

if [ ! -f data/specforge.db ] || [ -z "$(.venv/bin/python -c "
from specforge.store import Store; print(Store('data/specforge.db').latest_bar_date('SPY') or '')" 2>/dev/null)" ]; then
  echo "» downloading market data (first run, ~2 min)…"
  .venv/bin/stonk data --full
fi

echo "» smoke test: one paper scan cycle…"
.venv/bin/stonk scan --no-refresh > /tmp/stonk_smoke.json
.venv/bin/python - <<'EOF'
import json
s = json.load(open("/tmp/stonk_smoke.json"))
assert "cycle_id" in s and "equity" in s, f"smoke scan malformed: {s}"
from specforge.store import Store
rows = Store("data/specforge.db").audit_rows(cycle_id=s["cycle_id"])
assert rows, "smoke scan wrote no audit rows"
print(f"  smoke OK: cycle {s['cycle_id']} regime={s['regime']} "
      f"signals={s['signals']} equity=${s['equity']}")
EOF

echo "» starting GUI at $URL (Ctrl-C to stop)"
# open the browser once the server answers (background waiter — exec below
# replaces this shell, so the open must not depend on it)
( for _ in $(seq 1 60); do
    v=$(curl -sf --max-time 2 "$URL/api/version" 2>/dev/null || true)
    if [[ "$v" == *'"version"'* ]]; then
      open "$URL" 2>/dev/null || true
      exit 0
    fi
    sleep 1
  done ) &
exec .venv/bin/stonk serve --port 8420
