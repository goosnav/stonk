#!/usr/bin/env bash
# Install Stonk Terminal as a macOS launchd user agent: starts at login,
# restarts on crash with a 60s throttle (crash-loop protection), logs to
# logs/service.log. Restart-safety: kill switches, the live gate, and broker
# blocks all live in the DB/env, so a supervisor restart can never skip them.
# Usage:
#   ./scripts/install_service.sh [paper|live]     (default: paper)
#   ./scripts/install_service.sh status           service state + health verdict
#   ./scripts/install_service.sh uninstall
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PLIST="$HOME/Library/LaunchAgents/com.specforge.serve.plist"
LABEL="com.specforge.serve"
# Legacy label retained so reinstalling replaces, rather than duplicates, an
# existing live service.

if [ "${1:-}" = "uninstall" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "uninstalled $LABEL"
  exit 0
fi

if [ "${1:-}" = "status" ]; then
  if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
    launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null \
      | grep -E "state = |pid = |last exit code" | sed 's/^[[:space:]]*/  launchd /'
  else
    echo "  launchd: $LABEL not loaded (the server may still run un-supervised" \
         "via Stonk Terminal.app or scripts/restart_live.sh)"
  fi
  exec python3 "$ROOT/scripts/check_health.py"
fi

MODE="${1:-paper}"
[ -x "$ROOT/.venv/bin/stonk" ] || { echo "run ./run.sh once first (.venv missing)"; exit 1; }

# Refuse to install under a foreign (non-launchd) server: launchd would boot a
# second instance that loses the port race and crash-loops against the running
# one. Two live engines must never race.
if curl -sf --max-time 3 "http://127.0.0.1:8420/api/health" >/dev/null 2>&1; then
  if ! launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -q "state = running"; then
    echo "port 8420 is already served by a non-launchd process (Stonk Terminal.app"
    echo "window or restart_live.sh). Stop that instance first, then re-run."
    exit 1
  fi
fi

# Live installs must pass the triple gate NOW, or the service would boot but
# refuse every order — fail loudly instead of pretending it's trading.
if [ "$MODE" = "live" ]; then
  GATE=$("$ROOT/.venv/bin/python" -c "from specforge.config import load_config; ok,why=load_config('live').live_trading_allowed(); print('OK' if ok else 'BLOCKED: '+why)" 2>&1)
  if [ "$GATE" != "OK" ]; then
    echo "live gate not open — $GATE"
    echo "fix: set LIVE_TRADING_ENABLED=true and RH_ACCOUNT_WHITELIST in .env, then retry"
    exit 1
  fi
  echo "live gate OK — installing AUTONOMOUS live trader (approval_mode from configs/live.yaml)"
fi

mkdir -p "$ROOT/logs"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/stonk</string>
    <string>--mode</string><string>$MODE</string>
    <string>serve</string><string>--port</string><string>8420</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>60</integer>
  <key>StandardOutPath</key><string>$ROOT/logs/service.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/service.log</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
sleep 1
if ! launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -q "state = running"; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "service failed to start — macOS may be blocking background access to $ROOT" >&2
  echo "use Stonk Terminal.app, move the repo outside Documents, or grant privacy access" >&2
  exit 1
fi
echo "installed $LABEL (mode=$MODE) — GUI at http://127.0.0.1:8420"
echo "logs: tail -f $ROOT/logs/service.log · status: $0 status · uninstall: $0 uninstall"
