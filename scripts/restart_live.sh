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

# The double-click launcher is a foreground wrapper around the server. Leaving
# that wrapper behind after killing only its child can make it wait forever or
# race the replacement for the preferred port.
PIDS=$(pgrep -f "(stonk|specforge).*--mode live serve|scripts/stonk_app.sh" || true)
if [ -n "$PIDS" ]; then
  # kill every matching pid (a hung instance plus its replacement can coexist)
  echo "$PIDS" | xargs kill 2>/dev/null || true
  # A graceful Uvicorn shutdown may answer one final health request while it
  # drains open dashboard connections. Wait for both the old PIDs and the
  # listener to disappear; otherwise the replacement can mistake that final
  # response for its own health check.
  for _ in $(seq 1 30); do
    alive=false
    for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && alive=true; done
    if ! $alive && ! curl -sf --max-time 1 "localhost:$PORT/api/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if curl -sf --max-time 1 "localhost:$PORT/api/health" >/dev/null 2>&1; then
  echo "old server still owns port $PORT after shutdown timeout" >&2
  exit 1
fi

BIN=.venv/bin/stonk
[ -x "$BIN" ] || BIN=.venv/bin/specforge   # compatibility before editable reinstall
PYTHONUNBUFFERED=1 nohup "$BIN" --mode live serve --port "$PORT" \
  </dev/null >> logs/runtime-live.log 2>&1 &
NEW_PID=$!
disown

for _ in $(seq 1 15); do
  sleep 1
  if ! kill -0 "$NEW_PID" 2>/dev/null; then
    echo "replacement server exited before becoming healthy — check logs/runtime-live.log" >&2
    exit 1
  fi
  if HEALTH=$(curl -sf "localhost:$PORT/health/live" 2>/dev/null); then
    SERVING_PID=$(python3 -c \
      'import json,sys; print(json.load(sys.stdin).get("pid", ""))' <<<"$HEALTH" 2>/dev/null || true)
    if [ "$SERVING_PID" = "$NEW_PID" ]; then
      # A bind/startup failure can occur after one optimistic response. Require
      # ten consecutive seconds of process + listener ownership before the
      # helper is allowed to report success.
      stable=true
      STABLE="$HEALTH"
      for _ in $(seq 1 10); do
        sleep 1
        STABLE=$(curl -sf --max-time 2 "localhost:$PORT/health/live" 2>/dev/null || true)
        STABLE_PID=$(python3 -c \
          'import json,sys; print(json.load(sys.stdin).get("pid", ""))' <<<"$STABLE" 2>/dev/null || true)
        if ! kill -0 "$NEW_PID" 2>/dev/null || [ "$STABLE_PID" != "$NEW_PID" ]; then
          stable=false
          break
        fi
      done
      if $stable; then
        echo "restarted ok: $STABLE"
        exit 0
      fi
    fi
  fi
done
echo "server did not come back within 15s — check logs/runtime-live.log" >&2
exit 1
