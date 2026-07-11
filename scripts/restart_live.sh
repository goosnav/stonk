#!/usr/bin/env bash
# Restart the live Stonk Terminal server safely: refuses during NYSE market hours
# (a restart mid-session would disturb the live broker session and can miss
# a scheduled scan). Used by the scheduled maintenance sessions to pick up
# new commits. Usage:
#   ./scripts/restart_live.sh          # guarded restart
#   ./scripts/restart_live.sh --force  # skip the market-hours guard
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=8420
mkdir -p logs

if [ "${1:-}" != "--force" ]; then
  # ponytail: coarse guard — weekday 09:30-16:30 ET blocks, holidays not
  # checked (a holiday restart is merely over-cautious, never harmful).
  read -r dow hm < <(TZ=America/New_York date '+%u %H%M')
  if [ "$dow" -le 5 ] && [ "$hm" -ge 0930 ] && [ "$hm" -lt 1630 ]; then
    echo "refusing: NYSE market hours (ET $(TZ=America/New_York date '+%a %H:%M')). Use --force to override." >&2
    exit 1
  fi
fi

PIDS=$(pgrep -f "(stonk|specforge) --mode live serve" || true)
if [ -n "$PIDS" ]; then
  # kill every matching pid (a hung instance plus its replacement can coexist)
  echo "$PIDS" | xargs kill 2>/dev/null || true
  # wait for the port to free up
  for _ in $(seq 1 10); do curl -sf "localhost:$PORT/api/health" >/dev/null 2>&1 || break; sleep 1; done
fi

BIN=.venv/bin/stonk
[ -x "$BIN" ] || BIN=.venv/bin/specforge   # compatibility before editable reinstall
nohup "$BIN" --mode live serve --port "$PORT" >> logs/runtime-live.log 2>&1 &
disown

for _ in $(seq 1 15); do
  sleep 1
  if HEALTH=$(curl -sf "localhost:$PORT/api/health" 2>/dev/null); then
    echo "restarted ok: $HEALTH"
    exit 0
  fi
done
echo "server did not come back within 15s — check logs/runtime-live.log" >&2
exit 1
