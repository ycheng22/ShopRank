<#
.SYNOPSIS
    Starts both ShopRank FastAPI backend and Angular 21 frontend locally.

.DESCRIPTION
    Launches:
      1. FastAPI API server via `uv run uvicorn app.main:app --reload` on port 8000.
      2. Angular 21 UI dev server via `npm start` in the `web/` directory on port 4200.

.EXAMPLE
    .\start_local.ps1
    .\start_local.ps1 -ApiPort 8080 -UiPort 4200
#>

param (
    [int]$ApiPort = 8000,
    [int]$UiPort = 4200,
    [switch]$NoNewWindow
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "             ShopRank Local Development Runner              " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check prerequisites
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Error "uv is required but not installed or not in PATH. Please install uv first."
    exit 1
}

if (-not (Get-Command "npm" -ErrorAction SilentlyContinue)) {
    Write-Error "npm is required but not installed or not in PATH. Please install Node.js (v20+ or v22) first."
    exit 1
}

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $rootDir

# 2. Check .env file
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Warning "No .env found. Copying .env.example to .env..."
        Copy-Item ".env.example" ".env"
    } else {
        Write-Warning "No .env or .env.example found. Please ensure database and provider credentials are set."
    }
}

# 3. Check web dependencies
if (-not (Test-Path "web\node_modules")) {
    Write-Host "`n[Setup] Installing web dependencies..." -ForegroundColor Yellow
    Push-Location "web"
    npm install
    Pop-Location
}

Write-Host "`nStarting services:" -ForegroundColor Yellow
Write-Host "  - Backend API: http://localhost:$ApiPort" -ForegroundColor White
Write-Host "  - Swagger Doc: http://localhost:$ApiPort/docs" -ForegroundColor White
Write-Host "  - Frontend UI: http://localhost:$UiPort" -ForegroundColor White

if ($NoNewWindow) {
    Write-Host "`nRunning in background jobs (press Ctrl+C to terminate both)..." -ForegroundColor Cyan

    $backendJob = Start-Job -ScriptBlock {
        param($dir, $port)
        Set-Location $dir
        uv run uvicorn app.main:app --host 0.0.0.0 --port $port --reload
    } -ArgumentList $rootDir, $ApiPort

    $frontendJob = Start-Job -ScriptBlock {
        param($dir, $port, $apiPort)
        Set-Location "$dir\web"
        $env:API_BASE_URL = "http://localhost:$apiPort"
        npm start -- --port $port
    } -ArgumentList $rootDir, $UiPort, $ApiPort

    try {
        while ($true) {
            Receive-Job $backendJob | Out-Host
            Receive-Job $frontendJob | Out-Host
            Start-Sleep -Seconds 1
        }
    } finally {
        Write-Host "`nStopping jobs..." -ForegroundColor Yellow
        Stop-Job $backendJob -ErrorAction SilentlyContinue | Remove-Job -Force -ErrorAction SilentlyContinue
        Stop-Job $frontendJob -ErrorAction SilentlyContinue | Remove-Job -Force -ErrorAction SilentlyContinue
        Write-Host "All services stopped." -ForegroundColor Green
    }
} else {
    Write-Host "`n[1/2] Launching Backend API in a dedicated window..." -ForegroundColor Green
    $backendCmd = "Set-Location '$rootDir'; `$Host.UI.RawUI.WindowTitle = 'ShopRank API (Port $ApiPort)'; Write-Host 'Starting ShopRank Backend on http://localhost:$ApiPort...' -ForegroundColor Cyan; uv run uvicorn app.main:app --host 0.0.0.0 --port $ApiPort --reload"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

    Write-Host "[2/2] Launching Frontend UI in a dedicated window..." -ForegroundColor Green
    $frontendCmd = "Set-Location '$rootDir\web'; `$Host.UI.RawUI.WindowTitle = 'ShopRank Frontend (Port $UiPort)'; Write-Host 'Starting ShopRank Frontend on http://localhost:$UiPort...' -ForegroundColor Cyan; `$env:API_BASE_URL = 'http://localhost:$ApiPort'; npm start -- --port $UiPort"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

    Write-Host "`nServices started in separate windows." -ForegroundColor Green
    Write-Host "You can close those windows or press Ctrl+C inside them to stop each service." -ForegroundColor Gray
}
