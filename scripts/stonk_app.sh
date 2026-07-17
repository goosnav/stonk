#!/usr/bin/env bash
# Stonk Terminal.app launcher (D38). Double-click target: start the terminal
# center and open the dashboard. Idempotent — if a server is already serving
# on the port, it just opens the browser instead of fighting for it.
set -uo pipefail
umask 077

REPO="/Users/jbs/Documents/code/stonk"
PORT=8420
PORT_END=8430
URL="http://127.0.0.1:$PORT"
cd "$REPO" || { echo "repo not found at $REPO"; read -r; exit 1; }

echo "STONK TERMINAL — autonomous market control center"
echo "repo: $REPO"

# Already up? Attach even if a heavy research task makes /api/health slow.
# /api/version is dependency-free enough for identity and mode detection.
VERSION=$(curl -sf --max-time 3 "$URL/api/version" 2>/dev/null || true)
if [[ "$VERSION" == *'"version"'* && "$VERSION" == *'"mode"'* ]]; then
  echo "Stonk Terminal already owns port $PORT — opening dashboard."
  open "$URL"; exit 0
fi

# An unrelated listener may own the preferred port. The server atomically
# binds the first free port in the bounded range; the browser waiter discovers
# the effective Stonk URL. It never terminates the unrelated process.
if /usr/sbin/lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $PORT is occupied by another process; trying $((PORT + 1))-$PORT_END."
  /usr/sbin/lsof -nP -iTCP:$PORT -sTCP:LISTEN
fi

# first run / missing venv: create + install (visible, one-time ~2 min).
if [ ! -x .venv/bin/stonk ]; then
  echo "first run: creating .venv and installing…"
  python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && \
    .venv/bin/pip install -q -e ".[dev]" || { echo "install failed"; read -r; exit 1; }
fi

# open the browser once the server answers (background waiter).
( for _ in $(seq 1 40); do
    for candidate in $(seq "$PORT" "$PORT_END"); do
      candidate_url="http://127.0.0.1:$candidate"
      version=$(curl -sf --max-time 2 "$candidate_url/api/version" 2>/dev/null || true)
      if [[ "$version" == *'"version"'* && "$version" == *'"mode":"live"'* ]]; then
        open "$candidate_url"; exit 0
      fi
    done
    sleep 1
  done ) &

# live mode = the account this system actually trades (gates still enforced by
# the triple-check in config.live_trading_allowed). Foreground so this window
# shows logs and Ctrl-C / closing it stops the server.
echo "starting live server at $URL  (close this window or Ctrl-C to stop)"
mkdir -p logs
if [ -f logs/runtime-live.log ] && [ "$(stat -f%z logs/runtime-live.log 2>/dev/null || echo 0)" -gt 10000000 ]; then
  mv -f logs/runtime-live.log logs/runtime-live.log.1
fi
touch logs/runtime-live.log
chmod 600 logs/runtime-live.log
set +e
.venv/bin/stonk --mode live serve --port "$PORT" --port-range-end "$PORT_END" \
  2>&1 | tee -a logs/runtime-live.log
status=${PIPESTATUS[0]}
set -e
if [ "$status" -ne 0 ]; then
  echo "Stonk Terminal exited with status $status. Recent log output:"
  tail -30 logs/runtime-live.log
  read -r -p "Press Return to close…" _
fi
exit "$status"
