[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$targets = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.ProcessId -ne $PID -and
    $_.Name -eq "node.exe" -and
    (
        $_.CommandLine -like "*$RepoRoot*" -or
        $_.CommandLine -like "*dev:mock*"
    )
})

if ($targets.Count -eq 0) {
    Write-Host "OpenHands Windows mock UI is not running." -ForegroundColor Yellow
    exit 0
}

$targets | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host "Stopped OpenHands Windows mock UI." -ForegroundColor Green
