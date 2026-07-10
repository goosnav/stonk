#!/usr/bin/env bash
# Build Stonk.app — a double-clickable macOS launcher for the SpecForge control
# center. Regenerates the .icns from assets/stonk.svg (QuickLook render) if the
# icon is missing, then assembles a plain .app bundle (no py2app/pyinstaller:
# the app is a launcher for the existing .venv, not a frozen binary — that's
# both laziest and most robust with fastapi/uvicorn/yfinance deps). D38.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$HOME/Applications}"          # where the .app lands; override as arg
APP="$OUT/Stonk.app"
cd "$REPO"

# --- icon: build assets/stonk.icns if absent ---
if [ ! -f assets/stonk.icns ]; then
  echo "building icon…"
  tmp="$(mktemp -d)"; qlmanage -t -s 1024 -o "$tmp" assets/stonk.svg >/dev/null 2>&1
  png="$tmp/stonk.svg.png"; iset="$tmp/stonk.iconset"; mkdir -p "$iset"
  for s in 16 32 128 256 512; do
    sips -z $s $s "$png" --out "$iset/icon_${s}x${s}.png" >/dev/null
    sips -z $((s*2)) $((s*2)) "$png" --out "$iset/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$iset" -o assets/stonk.icns
  rm -rf "$tmp"
fi

# --- bundle ---
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp assets/stonk.icns "$APP/Contents/Resources/stonk.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Stonk</string>
  <key>CFBundleDisplayName</key><string>Stonk</string>
  <key>CFBundleIdentifier</key><string>com.jbs.stonk</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>Stonk</string>
  <key>CFBundleIconFile</key><string>stonk</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
</dict></plist>
PLIST

cat > "$APP/Contents/MacOS/Stonk" <<LAUNCH
#!/bin/bash
# open the launcher in Terminal so the server logs are visible + quittable
open -a Terminal "$REPO/scripts/stonk_app.sh"
LAUNCH
chmod +x "$APP/Contents/MacOS/Stonk" "$REPO/scripts/stonk_app.sh"

# refresh Finder's icon cache for this bundle
touch "$APP"
echo "built: $APP"
