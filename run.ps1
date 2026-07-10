# Stonk Terminal launcher (Windows). Mirrors run.sh.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Write-Host "» creating .venv and installing dependencies…"
  python -m venv .venv
  .venv\Scripts\pip install -q --upgrade pip
}
.venv\Scripts\pip install -q -e ".[dev]"

Write-Host "» running offline test suite…"
.venv\Scripts\pytest tests/ -q
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

if (-not (Test-Path "data\specforge.db")) {
  Write-Host "» downloading market data (first run, ~2 min)…"
  .venv\Scripts\stonk data --full
}

Write-Host "» smoke test: one paper scan cycle…"
.venv\Scripts\stonk scan --no-refresh
if ($LASTEXITCODE -ne 0) { throw "smoke scan failed" }

Write-Host "» starting GUI at http://127.0.0.1:8420"
.venv\Scripts\stonk serve --port 8420
