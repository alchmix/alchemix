# One-click backend start — env comes from .env (not $env: each session)
# Usage: right-click → Run with PowerShell, or:  .\start_server.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Backend = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Backend

if (-not (Test-Path ".env")) {
    Write-Host "ERROR: Missing .env in $Backend" -ForegroundColor Red
    Write-Host "Copy .env.example to .env and edit once:" -ForegroundColor Yellow
    Write-Host "  Copy-Item .env.example .env"
    exit 1
}

$venvActivate = Join-Path $Backend "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
}

# Waitress HTTP workers — read from .env so API stays responsive during video downloads
$waitressThreads = 32
$waitressChannelTimeout = 300
if (Test-Path ".env") {
    $tMatch = Select-String -Path .env -Pattern '^\s*WAITRESS_THREADS=(\d+)' | Select-Object -First 1
    if ($tMatch) { $waitressThreads = [int]$tMatch.Matches.Groups[1].Value }
    $cMatch = Select-String -Path .env -Pattern '^\s*WAITRESS_CHANNEL_TIMEOUT=(\d+)' | Select-Object -First 1
    if ($cMatch) { $waitressChannelTimeout = [int]$cMatch.Matches.Groups[1].Value }
}

Write-Host "Starting Alchemy Mix backend (env from .env)..." -ForegroundColor Green
Write-Host "  $Backend"
Write-Host "  Waitress threads=$waitressThreads channel_timeout=${waitressChannelTimeout}s"
waitress-serve --listen=*:5000 --threads=$waitressThreads --channel-timeout=$waitressChannelTimeout wsgi:app
