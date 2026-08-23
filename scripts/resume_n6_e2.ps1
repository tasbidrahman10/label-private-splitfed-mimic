# Resume the n=6 E2 free-rider grid after a power cut or reboot.
#
# Safe to run at any time. The grid is checkpointed to
# results/n6_grid_progress.json after every run with an atomic
# temp-then-replace, so an interruption costs at most the single run that was
# in flight. This script re-establishes the server and picks up where the
# checkpoint left off.
#
#     .\scripts\resume_n6_e2.ps1
#
# Add -StatusOnly to just print progress without starting anything.

param([switch]$StatusOnly)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No .venv found at $py" -ForegroundColor Red
    Write-Host "Rebuild it:  python -m venv .venv --clear; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

# The whole n=6 stack resolves clients through this variable. Without it the
# server loads the n=3 profile and the runner refuses to write n=3 numbers
# into the n=6 checkpoint.
$env:SFL_CLIENT_CONFIG_DIR = "configs/clients_n6"

if ($StatusOnly) {
    & $py scripts\run_n6_grid.py --experiment E2_free_rider --status
    exit $LASTEXITCODE
}

# --- is a healthy n=6 server already up? -------------------------------------
$healthy = $false
try {
    $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5
    if ($h.status -eq "ok") { $healthy = $true }
} catch { }

if ($healthy) {
    Write-Host "Server already responding on port 8000 - reusing it." -ForegroundColor Green
} else {
    # Clear anything stale holding the port, then start fresh.
    $stale = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $stale) {
        Write-Host "Stopping stale listener on 8000 (PID $($c.OwningProcess))"
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }

    if (-not (Test-Path "logs")) { New-Item -ItemType Directory logs | Out-Null }
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $log = "logs\n6_e2_server_$stamp.log"

    Write-Host "Starting n=6 server -> $log"
    Start-Process -FilePath $py `
        -ArgumentList "scripts\start_server.py", "--config", "configs/server_n6.yaml" `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -WindowStyle Hidden

    Write-Host -NoNewline "Waiting for server"
    $up = $false
    foreach ($i in 1..60) {
        Start-Sleep -Seconds 2
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5
            if ($h.status -eq "ok") { $up = $true; break }
        } catch { Write-Host -NoNewline "." }
    }
    Write-Host ""
    if (-not $up) {
        Write-Host "Server did not come up. Check $log" -ForegroundColor Red
        exit 1
    }
    Write-Host "Server up (device: $($h.device))." -ForegroundColor Green
}

Write-Host ""
Write-Host "Resuming E2 grid from checkpoint. Completed runs are skipped." -ForegroundColor Cyan
Write-Host ""
& $py scripts\run_n6_grid.py --experiment E2_free_rider
exit $LASTEXITCODE
