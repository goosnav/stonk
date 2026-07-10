#!/usr/bin/env bash
# Cron run-model (CONTROL_CENTER_V3): no resident process. launchd fires
# `stonk scan` at each scan time (ET) + a post-close attribution run.
# launchd (unlike cron) runs missed jobs when a sleeping Mac wakes.
# Usage: ./scripts/install_cron.sh [paper|live]   |   uninstall
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BASE="com.specforge.scan"
# Legacy label retained so reinstalling replaces existing scheduled jobs.

if [ "${1:-}" = "uninstall" ]; then
  for f in "$HOME/Library/LaunchAgents/$BASE".*.plist; do
    [ -e "$f" ] || continue
    launchctl bootout "gui/$(id -u)/$(basename "$f" .plist)" 2>/dev/null || true
    rm -f "$f"
  done
  echo "uninstalled all $BASE.* jobs"
  exit 0
fi

MODE="${1:-paper}"
[ -x "$ROOT/.venv/bin/stonk" ] || { echo "run ./run.sh once first"; exit 1; }
if [ "$MODE" = "live" ]; then
  GATE=$("$ROOT/.venv/bin/python" -c "from specforge.config import load_config; ok,why=load_config('live').live_trading_allowed(); print('OK' if ok else 'BLOCKED: '+why)")
  [ "$GATE" = "OK" ] || { echo "live gate not open — $GATE"; exit 1; }
fi

# scan times ET -> local: read from config so the two stay in sync
TIMES=$("$ROOT/.venv/bin/python" - "$MODE" <<'EOF'
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from specforge.config import load_config
cfg = load_config(sys.argv[1])
et = ZoneInfo(cfg.get("schedule", "timezone", default="America/New_York"))
for label, hhmm, extra in (
    [("scan%d" % i, t, "") for i, t in enumerate(cfg.get("schedule", "scans", default=[]))]
    + [("postclose", cfg.get("schedule", "post_close", default="16:30"), "--post-close")]):
    h, m = map(int, hhmm.split(":"))
    local = datetime.now(et).replace(hour=h, minute=m).astimezone()
    print(f"{label} {local.hour} {local.minute} {extra}")
EOF
)

echo "$TIMES" | while read -r LABEL H M EXTRA; do
  PLIST="$HOME/Library/LaunchAgents/$BASE.$LABEL.plist"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$BASE.$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/stonk</string>
    <string>--mode</string><string>$MODE</string>
    <string>scan</string>${EXTRA:+<string>$EXTRA</string>}
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>StartCalendarInterval</key><dict>
    <key>Hour</key><integer>$H</integer><key>Minute</key><integer>$M</integer>
    <key>Weekday</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key><string>$ROOT/data/cron.log</string>
  <key>StandardErrorPath</key><string>$ROOT/data/cron.log</string>
</dict></plist>
EOF
  # Weekday 0 means Sunday in launchd; we want Mon-Fri → install 5 copies? No:
  # launchd lacks ranges in one dict; simplest correct: omit Weekday and let
  # the engine's own market-clock/veto handle weekends (scan is harmless).
  /usr/libexec/PlistBuddy -c "Delete :StartCalendarInterval:Weekday" "$PLIST" 2>/dev/null || true
  launchctl bootout "gui/$(id -u)/$BASE.$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "installed $BASE.$LABEL ($H:$(printf '%02d' "$M") local, mode=$MODE)"
done
echo "logs: tail -f data/cron.log · check: .venv/bin/stonk tui"
