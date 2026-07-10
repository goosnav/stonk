#!/usr/bin/env bash
# Stonk Terminal launcher: setup → verify → serve. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"

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

echo "» starting GUI at http://127.0.0.1:8420 (Ctrl-C to stop)"
exec .venv/bin/stonk serve --port 8420
