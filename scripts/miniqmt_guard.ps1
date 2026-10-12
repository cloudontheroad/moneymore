param(
    [Parameter(Mandatory = $true)]
    [string]$Launcher,
    [int]$PollSeconds = 20,
    [int]$StartupGraceSeconds = 25,
    [int]$ProbeSeconds = 60,
    [int]$FailureThreshold = 3,
    [switch]$RestartOnProbeFailure
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $ProjectRoot "state\runtime"
$LogFile = Join-Path $RuntimeDir "miniqmt-guard.log"
$StatusFile = Join-Path $RuntimeDir "miniqmt-guard-status.json"
$WorkingDirectory = Join-Path (Split-Path -Parent (Split-Path -Parent $Launcher)) "config\tradingtime"
$MiniLauncher = Join-Path (Split-Path -Parent $Launcher) "XtMiniQmt.exe"
if (Test-Path -LiteralPath $MiniLauncher) { $Launcher = $MiniLauncher }
$QmtPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$QmtBridge = Join-Path $ProjectRoot "scripts\qmt_data_bridge.py"
$QmtSdk = Join-Path $ProjectRoot ".runtime\xtquant_250807"

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

function Write-GuardLog([string]$Message) {
    $Line = "{0} {1}" -f [DateTimeOffset]::Now.ToString("o"), $Message
    Add-Content -LiteralPath $LogFile -Value $Line -Encoding UTF8
}

function Write-GuardStatus([string]$Status, [string]$Message, $Process = $null) {
    $Payload = [pscustomobject]@{
        observed_at = [DateTimeOffset]::Now.ToString("o")
        status = $Status
        message = $Message
        guard_pid = $PID
        qmt_pid = if ($Process) { $Process.Id } else { $null }
        consecutive_probe_failures = $ConsecutiveProbeFailures
    }
    $Temporary = "$StatusFile.$PID.tmp"
    $Payload | ConvertTo-Json -Compress | Set-Content -LiteralPath $Temporary -Encoding UTF8
    Move-Item -LiteralPath $Temporary -Destination $StatusFile -Force
}

function Test-RecoveryWindow {
    $Now = Get-Date
    return $Now.DayOfWeek -notin @([DayOfWeek]::Saturday, [DayOfWeek]::Sunday) -and
        $Now.TimeOfDay -ge ([TimeSpan]::FromHours(8.75)) -and
        $Now.TimeOfDay -le ([TimeSpan]::FromHours(15.25))
}

function Get-QmtMainProcess {
    Get-Process -Name "XtMiniQmt" -ErrorAction SilentlyContinue |
        Sort-Object StartTime -Descending |
        Select-Object -First 1
}

function Remove-OrphanQuoteProcesses {
    if (Get-QmtMainProcess) { return }
    $Quotes = @(Get-Process -Name "miniquote" -ErrorAction SilentlyContinue)
    foreach ($Quote in $Quotes) {
        Write-GuardLog "stopping orphan miniquote pid=$($Quote.Id)"
        Stop-Process -Id $Quote.Id -Force -ErrorAction SilentlyContinue
    }
}

function Test-QmtSession {
    if (-not (Test-Path -LiteralPath $QmtPython) -or
        -not (Test-Path -LiteralPath $QmtBridge)) { return $false }
    $StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $QmtPython
    $StartInfo.Arguments = ('"{0}" probe' -f $QmtBridge)
    $StartInfo.WorkingDirectory = $ProjectRoot
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true
    # The broker's current client requires the matching SDK.  The legacy SDK
    # reports every healthy session as disconnected and previously triggered
    # destructive restart loops.
    $StartInfo.Environment["MONEYMORE_XTQUANT_SDK"] = $QmtSdk
    $Probe = [System.Diagnostics.Process]::new()
    $Probe.StartInfo = $StartInfo
    try {
        [void]$Probe.Start()
        if (-not $Probe.WaitForExit(15000)) {
            $Probe.Kill()
            Write-GuardLog "session probe timed out"
            return $false
        }
        return $Probe.ExitCode -eq 0
    } catch {
        Write-GuardLog "session probe failed: $($_.Exception.Message)"
        return $false
    } finally {
        $Probe.Dispose()
    }
}

function Restart-QmtSession {
    Write-GuardLog "session unhealthy; restarting official client processes"
    Get-Process -Name "XtMiniQmt", "XtItClient", "miniquote" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
    Start-Process -FilePath $Launcher -WorkingDirectory $WorkingDirectory -WindowStyle Minimized
    Start-Sleep -Seconds $StartupGraceSeconds
}

if (-not (Test-Path -LiteralPath $Launcher)) {
    throw "MiniQMT official launcher not found: $Launcher"
}
if (-not (Test-Path -LiteralPath $WorkingDirectory)) {
    throw "MiniQMT working directory not found: $WorkingDirectory"
}

Write-GuardLog "guard started launcher=$Launcher"
$LastProbeAt = [DateTimeOffset]::MinValue
$ConsecutiveProbeFailures = 0
$LastSessionRestartAt = [DateTimeOffset]::MinValue
while ($true) {
    try {
        if (-not (Get-QmtMainProcess)) {
            # XtMiniQmt can leave miniquote behind after its main window exits.
            # The orphan holds shared-memory/config files and destabilizes the
            # next login, so remove it only after confirming the main process is gone.
            Remove-OrphanQuoteProcesses
            Write-GuardLog "main process missing; launching official client"
            Start-Process -FilePath $Launcher -WorkingDirectory $WorkingDirectory -WindowStyle Minimized
            Start-Sleep -Seconds $StartupGraceSeconds
            $Started = Get-QmtMainProcess
            if ($Started) {
                Write-GuardLog "main process restored pid=$($Started.Id)"
                Write-GuardStatus "PROCESS_RESTORED" "QMT process restored; session verification pending" $Started
            } else {
                Write-GuardLog "official launcher returned but XtMiniQmt is still absent"
                Write-GuardStatus "PROCESS_MISSING" "Official launcher returned without XtMiniQmt" $null
            }
        } elseif (([DateTimeOffset]::Now - $LastProbeAt).TotalSeconds -ge $ProbeSeconds) {
            $LastProbeAt = [DateTimeOffset]::Now
            if (Test-QmtSession) {
                if ($ConsecutiveProbeFailures -gt 0) {
                    Write-GuardLog "session probe recovered"
                }
                $ConsecutiveProbeFailures = 0
                Write-GuardStatus "HEALTHY" "XtQuant RPC session is reachable" (Get-QmtMainProcess)
            } else {
                $ConsecutiveProbeFailures += 1
                Write-GuardLog "session probe unhealthy count=$ConsecutiveProbeFailures threshold=$FailureThreshold"
                Write-GuardStatus "SESSION_UNHEALTHY" "XtMiniQmt exists but XtQuant RPC is unreachable" (Get-QmtMainProcess)
                # A history download can temporarily monopolize XtQuant's
                # local RPC service.  Process liveness remains authoritative;
                # probing is observational unless explicitly opted into the
                # old destructive recovery behaviour.
                # Recovery is the safe default during the trading window.  The
                # legacy scheduled task did not pass RestartOnProbeFailure,
                # which left a dead RPC session alive indefinitely.
                if ((Test-RecoveryWindow) -and
                    ([DateTimeOffset]::Now - $LastSessionRestartAt).TotalMinutes -ge 30 -and
                    $ConsecutiveProbeFailures -ge $FailureThreshold) {
                    $LastSessionRestartAt = [DateTimeOffset]::Now
                    Restart-QmtSession
                    $ConsecutiveProbeFailures = 0
                    $LastProbeAt = [DateTimeOffset]::MinValue
                    Write-GuardStatus "SESSION_RESTARTED" "Restarted the complete QMT process chain after repeated RPC failures" (Get-QmtMainProcess)
                }
            }
        }
    } catch {
        Write-GuardLog "guard iteration failed: $($_.Exception.GetType().Name): $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $PollSeconds
}
