#!/usr/bin/env bash
# Stonk.app launcher (D38). Double-click target: start the SpecForge control
# center and open the dashboard. Idempotent — if a server is already serving
# on the port, it just opens the browser instead of fighting for it.
set -uo pipefail

REPO="/Users/jbs/Documents/code/stonk"
PORT=8420
URL="http://127.0.0.1:$PORT"
cd "$REPO" || { echo "repo not found at $REPO"; read -r; exit 1; }

echo "SPECFORGE — stonk control center"
echo "repo: $REPO"

# already up? attach, don't double-bind the port.
if curl -sf "$URL/api/health" >/dev/null 2>&1; then
  echo "server already running — opening dashboard."
  open "$URL"; exit 0
fi

# first run / missing venv: create + install (visible, one-time ~2 min).
if [ ! -x .venv/bin/specforge ]; then
  echo "first run: creating .venv and installing…"
  python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && \
    .venv/bin/pip install -q -e ".[dev]" || { echo "install failed"; read -r; exit 1; }
fi

# open the browser once the server answers (background waiter).
( for _ in $(seq 1 40); do
    curl -sf "$URL/api/health" >/dev/null 2>&1 && { open "$URL"; break; }
    sleep 1
  done ) &

# live mode = the account this system actually trades (gates still enforced by
# the triple-check in config.live_trading_allowed). Foreground so this window
# shows logs and Ctrl-C / closing it stops the server.
echo "starting live server at $URL  (close this window or Ctrl-C to stop)"
exec .venv/bin/specforge --mode live serve --port "$PORT"
