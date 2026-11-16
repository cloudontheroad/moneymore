param([Parameter(Mandatory=$true)][int]$ExpectedPid)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$listener = @(Get-NetTCPConnection -State Listen -LocalPort 3000)
$proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ExpectedPid"
$recorded = [int](Get-Content (Join-Path $root 'state/runtime/web.pid'))
if ($listener.Count -ne 1 -or $listener[0].OwningProcess -ne $ExpectedPid -or
    $recorded -ne $ExpectedPid -or $proc.CommandLine -notmatch 'vinext.*start') {
    throw 'Dashboard process identity changed; no process stopped.'
}
Stop-Process -Id $ExpectedPid -Force
