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
$QmtPython = Join-Path $ProjectRoot ".runtime\qmt-py311\Scripts\python.exe"
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

if (-not (Test-Path -LiteralPath $Launcher)) {
    throw "MiniQMT official launcher not found: $Launcher"
}
if (-not (Test-Path -LiteralPath $WorkingDirectory)) {
    throw "MiniQMT working directory not found: $WorkingDirectory"
}

$GuardMutex = [System.Threading.Mutex]::new($false, "Local\MoneyMoreMiniQmtGuard")
$OwnsMutex = $false
try { $OwnsMutex = $GuardMutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $OwnsMutex = $true }
if (-not $OwnsMutex) { $GuardMutex.Dispose(); exit 0 }
Write-GuardLog "guard started pid=$PID launcher=$Launcher policy=OBSERVE_SESSION_RELAUNCH_MISSING_ONLY"
$LastProbeAt = [DateTimeOffset]::MinValue
$ConsecutiveProbeFailures = 0
$LaunchRecord = Join-Path $RuntimeDir "miniqmt-last-launch.json"
$LastLaunchAt = [DateTimeOffset]::MinValue
if (Test-Path -LiteralPath $LaunchRecord) {
    try { $LastLaunchAt = [DateTimeOffset]::Parse((Get-Content -LiteralPath $LaunchRecord -Raw | ConvertFrom-Json).observed_at) } catch {}
}
try {
while ($true) {
    try {
        if (-not (Get-QmtMainProcess)) {
            # Never kill broker processes: a quote process may belong to the
            # ordinary client or an in-progress login. Persist the launch
            # cooldown so restarting the guard cannot reset retry frequency.
            if (([DateTimeOffset]::Now - $LastLaunchAt).TotalMinutes -lt 5) {
                Write-GuardStatus "WAITING_FOR_LOGIN" "QMT absent; waiting for login or five-minute launch cooldown" $null
                Start-Sleep -Seconds $PollSeconds
                continue
            }
            $LastLaunchAt = [DateTimeOffset]::Now
            @{ observed_at = $LastLaunchAt.ToString("o"); guard_pid = $PID } | ConvertTo-Json -Compress | Set-Content -LiteralPath $LaunchRecord -Encoding UTF8
            Write-GuardLog "main process missing; launching official client"
            $Launched = Start-Process -FilePath $Launcher -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru
            Write-GuardLog "launch requested pid=$($Launched.Id) reason=MAIN_PROCESS_MISSING"
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
                # Missing ticks, SDK errors and slow downloads do not prove
                # the client is dead. Leave the logged-in client untouched;
                # the trading health gate already blocks stale-data execution.
            }
        }
    } catch {
        Write-GuardLog "guard iteration failed: $($_.Exception.GetType().Name): $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $PollSeconds
}
} finally {
    if ($OwnsMutex) { $GuardMutex.ReleaseMutex() }
    $GuardMutex.Dispose()
}
