param([int]$ExpectedApiPid = 2696, [int]$ExpectedLauncherPid = 5596)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$log = Join-Path $project 'state/restart-recovery-20261111/api-reload.txt'
try {
    $ownerId = (Get-NetTCPConnection -State Listen -LocalPort 8788).OwningProcess
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$ownerId"
    if ($ownerId -ne $ExpectedApiPid -or $processInfo.ParentProcessId -ne $ExpectedLauncherPid -or $processInfo.CommandLine -notmatch 'uvicorn.*moneymore.web') {
        throw 'Verified API identity changed; no process stopped'
    }
    Stop-Process -Id $ownerId -Force -ErrorAction Stop
    "Verified MoneyMore API $ownerId stopped for supervised code reload at $(Get-Date -Format o)" | Out-File -LiteralPath $log
} catch {
    $_.Exception.Message | Out-File -LiteralPath $log
    exit 1
}
