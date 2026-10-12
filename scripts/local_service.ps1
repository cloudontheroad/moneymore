param(
    [ValidateSet("Run", "Start", "Stop", "Status", "Ensure")]
    [string]$Action = "Status",
    [int]$ApiPort = 8788,
    [int]$WebPort = 3000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $ProjectRoot "state\runtime"
$LogDir = Join-Path $ProjectRoot "logs"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Node = Join-Path $ProjectRoot ".runtime\node\node.exe"
$Vinext = Join-Path $ProjectRoot "apps\dashboard\node_modules\vinext\dist\cli.js"
$SupervisorPidFile = Join-Path $RuntimeDir "local-service.pid"
$StopFile = Join-Path $RuntimeDir "local-service.stop"
$HeartbeatFile = Join-Path $RuntimeDir "supervisor-heartbeat.json"
$SupervisorEventLog = Join-Path $LogDir "supervisor-events.jsonl"

New-Item -ItemType Directory -Force -Path $RuntimeDir, $LogDir | Out-Null

function Get-RecordedProcess([string]$PidFile) {
    if (-not (Test-Path -LiteralPath $PidFile)) { return $null }
    $RecordedPid = [int](Get-Content -LiteralPath $PidFile -Raw)
    return Get-Process -Id $RecordedPid -ErrorAction SilentlyContinue
}

function Test-Endpoint([string]$Uri) {
    try {
        $Response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 3
        return $Response.StatusCode -eq 200
    } catch { return $false }
}

function Test-PortListening([int]$Port) {
    $Client = [System.Net.Sockets.TcpClient]::new()
    try {
        $Result = $Client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $Result.AsyncWaitHandle.WaitOne(1000)) { return $false }
        $Client.EndConnect($Result)
        return $true
    } catch { return $false }
    finally { $Client.Close() }
}

function Stop-RecordedProcess([string]$PidFile) {
    $Process = Get-RecordedProcess $PidFile
    if ($Process) { Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

function Start-ChildProcess(
    [string]$Name,
    [string]$FilePath,
    [string[]]$Arguments,
    [string]$WorkingDirectory
) {
    # Some launchers inject both Path and PATH. Windows PowerShell's
    # Start-Process builds a case-insensitive environment dictionary and
    # crashes on that duplicate, so normalize it before spawning children.
    $ProcessPath = [System.Environment]::GetEnvironmentVariable("Path", "Process")
    [System.Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [System.Environment]::SetEnvironmentVariable("Path", $ProcessPath, "Process")
    $OutLog = Join-Path $LogDir "$Name.out.log"
    $ErrLog = Join-Path $LogDir "$Name.err.log"
    $Process = Start-Process -FilePath $FilePath -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory -WindowStyle Hidden `
        -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
    Set-Content -LiteralPath (Join-Path $RuntimeDir "$Name.pid") -Value $Process.Id
    [pscustomobject]@{
        observed_at = [DateTimeOffset]::Now.ToString("o")
        event = "CHILD_STARTED"
        component = $Name
        pid = $Process.Id
    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $SupervisorEventLog
    return $Process
}

function Write-SupervisorHeartbeat($Api, $Web) {
    $ApiHealthy = Test-Endpoint "http://127.0.0.1:$ApiPort/api/health"
    $WebHealthy = Test-Endpoint "http://127.0.0.1:$WebPort"
    $Payload = [pscustomobject]@{
        observed_at = [DateTimeOffset]::Now.ToString("o")
        supervisor_pid = $PID
        api_pid = if ($Api -and -not $Api.HasExited) { $Api.Id } else { $null }
        web_pid = if ($Web -and -not $Web.HasExited) { $Web.Id } else { $null }
        api_running = $ApiHealthy
        web_running = $WebHealthy
    }
    # A watchdog launch can briefly overlap an exiting supervisor.  A shared
    # ``.tmp`` name lets those processes move/delete each other's file and was
    # the cause of repeated empty/missing heartbeat alarms.  Each writer now
    # owns its staging file; the final rename remains atomic for readers.
    $Temporary = "$HeartbeatFile.$PID.tmp"
    $Payload | ConvertTo-Json -Compress | Set-Content -LiteralPath $Temporary
    Move-Item -LiteralPath $Temporary -Destination $HeartbeatFile -Force
}

function Assert-Runtime {
    foreach ($Required in @($Python, $Node, $Vinext)) {
        if (-not (Test-Path -LiteralPath $Required)) { throw "Missing runtime: $Required" }
    }
}

function Test-SupervisorHealthy {
    $Supervisor = Get-RecordedProcess $SupervisorPidFile
    if (-not $Supervisor -or -not (Test-Path -LiteralPath $HeartbeatFile)) { return $false }
    try {
        $Heartbeat = Get-Content -LiteralPath $HeartbeatFile -Raw | ConvertFrom-Json
        $Age = ([DateTimeOffset]::Now - [DateTimeOffset]::Parse($Heartbeat.observed_at)).TotalSeconds
        return [int]$Heartbeat.supervisor_pid -eq $Supervisor.Id -and $Age -le 45
    } catch { return $false }
}

function Ensure-MiniQmtGuardTask {
    # XtMiniQmt.exe can remain alive after its RPC session has died.  The QMT
    # guard performs the session probe, while this independent one-minute
    # watchdog makes sure the long-running guard itself did not disappear.
    $TaskName = "MoneyMore MiniQMT"
    try {
        $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($Task -and [string]$Task.State -ne "Running") {
            Start-ScheduledTask -TaskName $TaskName
            [pscustomobject]@{
                observed_at = [DateTimeOffset]::Now.ToString("o")
                event = "MINIQMT_GUARD_RECOVERY_REQUESTED"
                component = "watchdog"
                previous_state = [string]$Task.State
            } | ConvertTo-Json -Compress | Add-Content -LiteralPath $SupervisorEventLog
        }
    } catch {
        [pscustomobject]@{
            observed_at = [DateTimeOffset]::Now.ToString("o")
            event = "MINIQMT_GUARD_RECOVERY_FAILED"
            component = "watchdog"
            error = $_.Exception.Message
        } | ConvertTo-Json -Compress | Add-Content -LiteralPath $SupervisorEventLog
    }
}

function Repair-VinextWindowsStaticPaths {
    & $Node (Join-Path $ProjectRoot "apps\dashboard\scripts\patch-vinext-windows.mjs")
    if ($LASTEXITCODE -ne 0) { throw "Could not prepare the dashboard production server." }
}

switch ($Action) {
    "Ensure" {
        Ensure-MiniQmtGuardTask
        if (Test-SupervisorHealthy) {
            Write-Output "MoneyMore supervisor heartbeat is healthy."
            break
        }
        $Stale = Get-RecordedProcess $SupervisorPidFile
        if ($Stale) {
            Stop-Process -Id $Stale.Id -Force -ErrorAction SilentlyContinue
        }
        Remove-Item -LiteralPath $SupervisorPidFile -Force -ErrorAction SilentlyContinue
        [pscustomobject]@{
            observed_at = [DateTimeOffset]::Now.ToString("o")
            event = "SUPERVISOR_RECOVERY_REQUESTED"
            component = "watchdog"
            stale_pid = if ($Stale) { $Stale.Id } else { $null }
        } | ConvertTo-Json -Compress | Add-Content -LiteralPath $SupervisorEventLog
        & $PSCommandPath -Action Start -ApiPort $ApiPort -WebPort $WebPort
    }
    "Start" {
        Assert-Runtime
        Repair-VinextWindowsStaticPaths
        $Existing = Get-RecordedProcess $SupervisorPidFile
        if ($Existing) { Write-Output "MoneyMore local service already running (PID=$($Existing.Id))."; break }
        Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
        $PowerShell = (Get-Command powershell.exe).Source
        $Process = Start-Process -FilePath $PowerShell -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
            "-File", $PSCommandPath, "-Action", "Run", "-ApiPort", $ApiPort, "-WebPort", $WebPort
        ) -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
        Start-Sleep -Seconds 3
        Write-Output "MoneyMore local service started (PID=$($Process.Id))."
        Write-Output "Dashboard: http://127.0.0.1:$WebPort"
    }
    "Stop" {
        New-Item -ItemType File -Force -Path $StopFile | Out-Null
        Stop-RecordedProcess (Join-Path $RuntimeDir "api.pid")
        Stop-RecordedProcess (Join-Path $RuntimeDir "web.pid")
        Stop-RecordedProcess $SupervisorPidFile
        Write-Output "MoneyMore local service stopped."
    }
    "Status" {
        $Supervisor = Get-RecordedProcess $SupervisorPidFile
        [pscustomobject]@{
            Supervisor = if ($Supervisor) { "RUNNING (PID=$($Supervisor.Id))" } else { "STOPPED" }
            Api = if (Test-Endpoint "http://127.0.0.1:$ApiPort/api/health") { "HEALTHY" } else { "DOWN" }
            Dashboard = if (Test-Endpoint "http://127.0.0.1:$WebPort") { "HEALTHY" } else { "DOWN" }
            Url = "http://127.0.0.1:$WebPort"
        } | Format-List
    }
    "Run" {
        Assert-Runtime
        Repair-VinextWindowsStaticPaths
        $SupervisorMutex = [System.Threading.Mutex]::new(
            $false,
            "Local\MoneyMoreLocalServiceSupervisor"
        )
        $OwnsSupervisorMutex = $false
        try {
            try {
                $OwnsSupervisorMutex = $SupervisorMutex.WaitOne(0)
            } catch [System.Threading.AbandonedMutexException] {
                $OwnsSupervisorMutex = $true
            }
            if (-not $OwnsSupervisorMutex) {
                [pscustomobject]@{
                    observed_at = [DateTimeOffset]::Now.ToString("o")
                    event = "DUPLICATE_SUPERVISOR_SKIPPED"
                    component = "supervisor"
                    pid = $PID
                    reason = "PROCESS_MUTEX_HELD"
                } | ConvertTo-Json -Compress | Add-Content -LiteralPath $SupervisorEventLog
                exit 0
            }
        $ExistingSupervisor = Get-RecordedProcess $SupervisorPidFile
        if ($ExistingSupervisor -and $ExistingSupervisor.Id -ne $PID) {
            [pscustomobject]@{
                observed_at = [DateTimeOffset]::Now.ToString("o")
                event = "DUPLICATE_SUPERVISOR_SKIPPED"
                component = "supervisor"
                pid = $PID
                existing_pid = $ExistingSupervisor.Id
            } | ConvertTo-Json -Compress | Add-Content -LiteralPath $SupervisorEventLog
            exit 0
        }
        Set-Content -LiteralPath $SupervisorPidFile -Value $PID
        Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
        $Api = $null
        $Web = $null
        $NextApiLaunch = [DateTimeOffset]::MinValue
        $NextWebLaunch = [DateTimeOffset]::MinValue
        try {
            while (-not (Test-Path -LiteralPath $StopFile)) {
                $Now = [DateTimeOffset]::Now
                if ((-not $Api -or $Api.HasExited) -and $Now -ge $NextApiLaunch) {
                    $NextApiLaunch = $Now.AddSeconds(45)
                    if (-not (Test-Endpoint "http://127.0.0.1:$ApiPort/api/health") -and
                        -not (Test-PortListening $ApiPort)) {
                        $Api = Start-ChildProcess "api" $Python @(
                            "-m", "uvicorn", "moneymore.web:app", "--host", "127.0.0.1",
                            "--port", "$ApiPort", "--workers", "1"
                        ) $ProjectRoot
                    } else { $Api = $null }
                }
                if ((-not $Web -or $Web.HasExited) -and $Now -ge $NextWebLaunch) {
                    $NextWebLaunch = $Now.AddSeconds(45)
                    if (-not (Test-Endpoint "http://127.0.0.1:$WebPort") -and
                        -not (Test-PortListening $WebPort)) {
                        $Web = Start-ChildProcess "web" $Node @(
                            $Vinext, "start", "--hostname", "127.0.0.1", "--port", "$WebPort"
                        ) (Join-Path $ProjectRoot "apps\dashboard")
                    } else { $Web = $null }
                }
                try {
                    Write-SupervisorHeartbeat $Api $Web
                } catch {
                    [pscustomobject]@{
                        observed_at = [DateTimeOffset]::Now.ToString("o")
                        event = "HEARTBEAT_WRITE_FAILED"
                        component = "supervisor"
                        error = $_.Exception.Message
                    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $SupervisorEventLog
                }
                Start-Sleep -Seconds 10
            }
        } catch {
            [pscustomobject]@{
                observed_at = [DateTimeOffset]::Now.ToString("o")
                event = "SUPERVISOR_FATAL"
                component = "supervisor"
                error = $_.Exception.ToString()
            } | ConvertTo-Json -Compress | Add-Content -LiteralPath $SupervisorEventLog
            throw
        } finally {
            # Only the supervisor still registered as owner may clean shared
            # runtime files.  An older, overlapping instance must never erase
            # the replacement supervisor's heartbeat or child PID files.
            $RecordedOwner = Get-RecordedProcess $SupervisorPidFile
            if ($RecordedOwner -and $RecordedOwner.Id -eq $PID) {
                Stop-RecordedProcess (Join-Path $RuntimeDir "api.pid")
                Stop-RecordedProcess (Join-Path $RuntimeDir "web.pid")
                Remove-Item -LiteralPath $SupervisorPidFile, $StopFile, $HeartbeatFile -Force -ErrorAction SilentlyContinue
            }
            Remove-Item -LiteralPath "$HeartbeatFile.$PID.tmp" -Force -ErrorAction SilentlyContinue
        }
        } finally {
            if ($OwnsSupervisorMutex) {
                $SupervisorMutex.ReleaseMutex()
            }
            $SupervisorMutex.Dispose()
        }
    }
}
