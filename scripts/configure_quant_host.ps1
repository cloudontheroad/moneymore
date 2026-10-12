param(
    [ValidateSet("Install", "RepairQmt", "Audit", "Rollback")]
    [string]$Action = "Audit"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $ProjectRoot "state\runtime"
$BackupFile = Join-Path $RuntimeDir "windows-availability-before.json"
$ServiceScript = Join-Path $PSScriptRoot "local_service.ps1"
$QmtGuardScript = Join-Path $PSScriptRoot "miniqmt_guard.ps1"
$UpdatePolicy = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU"
$MoneyMoreTask = "MoneyMore Service"
$QmtTask = "MoneyMore MiniQMT"
$QmtProcess = Get-Process -Name "XtMiniQmt" -ErrorAction SilentlyContinue | Select-Object -First 1
$ExistingQmtTask = Get-ScheduledTask -TaskName $QmtTask -ErrorAction SilentlyContinue
$QmtTaskLauncher = if ($ExistingQmtTask -and [string]$ExistingQmtTask.Actions[0].Arguments -match '-Launcher\s+"([^"]+)"') {
    $Matches[1]
} else {
    $null
}
$QmtInternalExecutable = if ($QmtProcess) {
    $QmtProcess.Path
} elseif ($ExistingQmtTask) {
    [string]$ExistingQmtTask.Actions[0].Execute
} else {
    $null
}
$QmtBinDirectory = if ($QmtTaskLauncher) {
    Split-Path -Parent $QmtTaskLauncher
} elseif ($QmtInternalExecutable -and (Split-Path -Leaf $QmtInternalExecutable) -ine "powershell.exe") {
    Split-Path -Parent $QmtInternalExecutable
} else {
    $null
}
$QmtLauncher = if ($QmtBinDirectory) { Join-Path $QmtBinDirectory "XtItClient.exe" } else { $null }

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

function Read-Policy {
    $Policy = Get-ItemProperty -LiteralPath $UpdatePolicy -ErrorAction SilentlyContinue
    return [pscustomobject]@{
        NoAutoRebootWithLoggedOnUsers = $Policy.NoAutoRebootWithLoggedOnUsers
        AUOptions = $Policy.AUOptions
    }
}

function Read-Task([string]$Name) {
    $Task = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if (-not $Task) { return $null }
    $Info = Get-ScheduledTaskInfo -TaskName $Name -ErrorAction SilentlyContinue
    return [pscustomobject]@{
        name = $Name
        state = [string]$Task.State
        user = [string]$Task.Principal.UserId
        logon_type = [string]$Task.Principal.LogonType
        last_run = $Info.LastRunTime
        last_result = $Info.LastTaskResult
        next_run = $Info.NextRunTime
    }
}

function Write-Audit {
    [pscustomobject]@{
        observed_at = [DateTimeOffset]::Now.ToString("o")
        update_policy = Read-Policy
        moneymore_task = Read-Task $MoneyMoreTask
        qmt_task = Read-Task $QmtTask
        qmt_launcher = $QmtLauncher
        qmt_exists = Test-Path -LiteralPath $QmtLauncher
        active_power_scheme = (powercfg /getactivescheme | Out-String).Trim()
    } | ConvertTo-Json -Depth 6
}

if ($Action -in @("Install", "RepairQmt")) {
    if (-not (Test-Path -LiteralPath $ServiceScript)) {
        throw "MoneyMore service script not found: $ServiceScript"
    }
    if (-not (Test-Path -LiteralPath $QmtLauncher)) {
        throw "MiniQMT official launcher not found: $QmtLauncher"
    }
    if (-not (Test-Path -LiteralPath $QmtGuardScript)) {
        throw "MiniQMT guard script not found: $QmtGuardScript"
    }
    if (-not (Test-Path -LiteralPath $BackupFile)) {
        [pscustomobject]@{
            saved_at = [DateTimeOffset]::Now.ToString("o")
            update_policy = Read-Policy
        } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $BackupFile -Encoding UTF8
    }

    if ($Action -eq "Install") {
        New-Item -ItemType Directory -Force -Path $UpdatePolicy | Out-Null
        New-ItemProperty -LiteralPath $UpdatePolicy -Name "AUOptions" -PropertyType DWord -Value 4 -Force | Out-Null
        New-ItemProperty -LiteralPath $UpdatePolicy -Name "NoAutoRebootWithLoggedOnUsers" -PropertyType DWord -Value 1 -Force | Out-Null
    }

    $PowerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    if ($Action -eq "Install") {
        $ServiceArguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}" -Action Start' -f $ServiceScript
        $ServiceAction = New-ScheduledTaskAction -Execute $PowerShell -Argument $ServiceArguments
        $ServiceTrigger = New-ScheduledTaskTrigger -AtStartup
        $ServiceTrigger.Delay = "PT30S"
        $ServicePrincipal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
        $ServiceSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 6 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
        Register-ScheduledTask -TaskName $MoneyMoreTask -Action $ServiceAction -Trigger $ServiceTrigger -Principal $ServicePrincipal -Settings $ServiceSettings -Description "Start and supervise the MoneyMore API and dashboard after Windows boots." -Force | Out-Null
    }

    $CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $QmtArguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}" -Launcher "{1}"' -f $QmtGuardScript, $QmtLauncher
    $QmtAction = New-ScheduledTaskAction -Execute $PowerShell -Argument $QmtArguments
    $QmtTrigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
    $QmtPrincipal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Highest
    $QmtSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 6 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $QmtTask -Action $QmtAction -Trigger $QmtTrigger -Principal $QmtPrincipal -Settings $QmtSettings -Description "Start Huatai MiniQMT after the trading user signs in." -Force | Out-Null

    Write-Audit
    exit 0
}

if ($Action -eq "Rollback") {
    Unregister-ScheduledTask -TaskName $MoneyMoreTask -Confirm:$false -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $QmtTask -Confirm:$false -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $BackupFile) {
        $Backup = Get-Content -LiteralPath $BackupFile -Raw | ConvertFrom-Json
        New-Item -ItemType Directory -Force -Path $UpdatePolicy | Out-Null
        foreach ($Name in @("AUOptions", "NoAutoRebootWithLoggedOnUsers")) {
            $Value = $Backup.update_policy.$Name
            if ($null -eq $Value) {
                Remove-ItemProperty -LiteralPath $UpdatePolicy -Name $Name -ErrorAction SilentlyContinue
            } else {
                New-ItemProperty -LiteralPath $UpdatePolicy -Name $Name -PropertyType DWord -Value ([int]$Value) -Force | Out-Null
            }
        }
    }
    Write-Audit
    exit 0
}

Write-Audit
