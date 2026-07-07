#!/usr/bin/env bash
# Install SpecForge as a macOS launchd user agent: starts at login, restarts
# on crash, logs to data/service.log. Usage:
#   ./scripts/install_service.sh [paper|live]     (default: paper)
#   ./scripts/install_service.sh uninstall
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PLIST="$HOME/Library/LaunchAgents/com.specforge.serve.plist"
LABEL="com.specforge.serve"

if [ "${1:-}" = "uninstall" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "uninstalled $LABEL"
  exit 0
fi

MODE="${1:-paper}"
[ -x "$ROOT/.venv/bin/specforge" ] || { echo "run ./run.sh once first (.venv missing)"; exit 1; }

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/specforge</string>
    <string>--mode</string><string>$MODE</string>
    <string>serve</string><string>--port</string><string>8420</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>StandardOutPath</key><string>$ROOT/data/service.log</string>
  <key>StandardErrorPath</key><string>$ROOT/data/service.log</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL (mode=$MODE) — GUI at http://127.0.0.1:8420"
echo "logs: tail -f $ROOT/data/service.log · uninstall: $0 uninstall"
