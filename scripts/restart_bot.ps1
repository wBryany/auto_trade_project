[CmdletBinding()]
param(
    [string]$ConfigPath = "config.binance.testnet.json",
    [string]$Exchange = "binance",
    [string]$BindAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8787,
    [string]$PythonPath = "",
    [string]$RuntimeWorkingDirectory = "",
    [switch]$DashboardOnly,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$sourceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$runtimeRoot = if ($RuntimeWorkingDirectory) {
    (Resolve-Path -LiteralPath $RuntimeWorkingDirectory).Path
} else {
    $sourceRoot
}
$checksPath = Join-Path $PSScriptRoot "restart_bot_checks.ps1"
if (-not (Test-Path -LiteralPath $checksPath -PathType Leaf)) {
    throw "Restart validation helpers not found: $checksPath"
}
. $checksPath

$resolvedConfigPath = if ([IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath
} else {
    Join-Path $runtimeRoot $ConfigPath
}
if (-not (Test-Path -LiteralPath $resolvedConfigPath -PathType Leaf)) {
    throw "Config file not found: $resolvedConfigPath"
}

$probeAddress = if ($BindAddress -in @("0.0.0.0", "::")) { "127.0.0.1" } else { $BindAddress }
$serviceUrl = "http://${probeAddress}:$Port"
$runningPythonPath = $null

function Get-DashboardListener {
    $listeners = @(
        Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
    if ($listeners.Count -eq 0) {
        # Get-NetTCPConnection can return no rows under a restricted Windows
        # PowerShell token even though netstat can still report the listener.
        $escapedPort = [regex]::Escape([string]$Port)
        $pattern = "^\s*TCP\s+\S+:$escapedPort\s+\S+\s+LISTENING\s+(?<process>\d+)\s*$"
        $listeners = @(
            netstat -ano -p tcp |
                ForEach-Object {
                    if ($_ -match $pattern) {
                        [int]$Matches["process"]
                    }
                } |
                Sort-Object -Unique
        )
    }
    if ($listeners.Count -gt 1) {
        throw "More than one process is listening on port ${Port}: $($listeners -join ', ')"
    }
    return $listeners | Select-Object -First 1
}

function Get-DashboardStatus {
    param([int]$TimeoutSeconds = 8)

    return Invoke-RestMethod "$serviceUrl/api/status" -TimeoutSec $TimeoutSeconds
}

function Wait-DashboardReady {
    param(
        [Parameter(Mandatory = $true)]
        [Diagnostics.Process]$Process,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lastError = ""
    while ((Get-Date) -lt $deadline) {
        $Process.Refresh()
        if ($Process.HasExited) {
            throw "Dashboard process exited with code $($Process.ExitCode)"
        }
        try {
            $version = Invoke-RestMethod "$serviceUrl/api/version" -TimeoutSec 2
            if ($version.version) {
                return
            }
        } catch {
            $lastError = $_.Exception.Message
        }
        Start-Sleep -Milliseconds 500
    }
    $detail = if ($lastError) { ": $lastError" } else { "" }
    throw "Dashboard did not become ready within $TimeoutSeconds seconds$detail"
}

function Wait-EngineStatus {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$ExpectedRunning,
        [int]$TimeoutSeconds = 45,
        [switch]$RequireHealthy,
        [switch]$RequireFirstCycle
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lastError = "No status response"
    while ((Get-Date) -lt $deadline) {
        try {
            $status = Get-DashboardStatus -TimeoutSeconds 5
            $running = ConvertTo-RestartBoolean (Get-RestartValue $status @("running"))
            if ($running -ne $ExpectedRunning) {
                $lastError = "running=$running"
            } else {
                if ($RequireHealthy) {
                    if ($ExpectedRunning) {
                        Assert-RestartStatusHealth `
                            -Status $status `
                            -RequireRunning `
                            -RequireFirstCycle:$RequireFirstCycle
                    } else {
                        # DashboardService.stop() intentionally preserves the
                        # previous cycle error.  Once running=false is proven,
                        # that historical value must not prevent the stopped
                        # exchange snapshot from being checked.
                        Assert-RestartStatusHealth -Status $status -RequireStopped -AllowLastError
                    }
                }
                return $status
            }
        } catch {
            $lastError = $_.Exception.Message
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Dashboard status did not reach running=$ExpectedRunning within $TimeoutSeconds seconds: $lastError"
}

function Stop-EngineFailClosed {
    param([string]$Reason)

    try {
        Invoke-RestMethod `
            "$serviceUrl/api/stop" `
            -Method Post `
            -ContentType "application/json" `
            -Body "{}" `
            -TimeoutSec 15 |
            Out-Null
        Wait-EngineStatus -ExpectedRunning $false -TimeoutSeconds 20 | Out-Null
        return "engine stop confirmed after validation failure"
    } catch {
        return "WARNING: engine stop could not be confirmed after validation failure: $($_.Exception.Message)"
    }
}

$servicePid = Get-DashboardListener
$beforeSnapshot = $null
if ($servicePid) {
    try {
        $version = Invoke-RestMethod "$serviceUrl/api/version" -TimeoutSec 5
        if (-not $version.version) {
            throw "Unexpected response"
        }
    } catch {
        throw "Port $Port is occupied by PID $servicePid, but it is not a healthy bot dashboard"
    }

    $serviceProcess = Get-Process -Id $servicePid -ErrorAction Stop
    if ($serviceProcess.ProcessName -notmatch "^python(?:w)?$") {
        throw "Refusing to stop unexpected process on port ${Port}: $($serviceProcess.ProcessName) (PID $servicePid)"
    }
    $runningPythonPath = $serviceProcess.Path

    try {
        $beforeStatus = Get-DashboardStatus -TimeoutSeconds 10
        Assert-SafeRestartPreflightStatus -Status $beforeStatus
        $beforeSnapshot = Get-SafeRestartSnapshot -Status $beforeStatus
    } catch {
        throw "Safe restart pre-check failed; no process was stopped: $($_.Exception.Message)"
    }
    Write-Host "Pre-restart exchange snapshot: $(Format-SafeRestartSnapshot $beforeSnapshot)"

    if ($CheckOnly) {
        Write-Host "CheckOnly completed: no POST was sent and no process was stopped."
        return
    }

    Write-Host "Stopping trading engine and dashboard (PID $servicePid)..."
    $stopRequestError = ""
    try {
        Invoke-RestMethod `
            "$serviceUrl/api/stop" `
            -Method Post `
            -ContentType "application/json" `
            -Body "{}" `
            -TimeoutSec 15 |
            Out-Null
    } catch {
        $stopRequestError = $_.Exception.Message
    }

    try {
        $stoppedStatus = Wait-EngineStatus `
            -ExpectedRunning $false `
            -TimeoutSeconds 30 `
            -RequireHealthy
        $stoppedSnapshot = Get-SafeRestartSnapshot -Status $stoppedStatus
        Assert-SafeRestartSnapshotUnchanged -Before $beforeSnapshot -After $stoppedSnapshot
    } catch {
        $stopDetail = if ($stopRequestError) { " Stop request error: $stopRequestError" } else { "" }
        throw "Engine stop was not safely confirmed; dashboard process was left running. $($_.Exception.Message)$stopDetail"
    }
    if ($stopRequestError) {
        Write-Warning "The stop request returned an error, but running=false and the exchange snapshot were independently confirmed: $stopRequestError"
    }
    Write-Host "Post-stop exchange snapshot confirmed: $(Format-SafeRestartSnapshot $stoppedSnapshot)"

    Stop-Process -Id $servicePid -ErrorAction Stop
    Wait-Process -Id $servicePid -Timeout 10 -ErrorAction SilentlyContinue

    $portDeadline = (Get-Date).AddSeconds(10)
    while ((Get-DashboardListener) -and (Get-Date) -lt $portDeadline) {
        Start-Sleep -Milliseconds 250
    }
    if (Get-DashboardListener) {
        throw "Port $Port was not released after stopping PID $servicePid"
    }
} else {
    if ($CheckOnly) {
        throw "CheckOnly requires an existing dashboard on $serviceUrl; no POST was sent and no process was started"
    }
    Write-Host "No existing dashboard listener found; entering fail-closed cold-start validation."
}

if ($PythonPath) {
    $resolvedPythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
} elseif ($runningPythonPath -and (Test-Path -LiteralPath $runningPythonPath -PathType Leaf)) {
    $resolvedPythonPath = $runningPythonPath
} elseif (Test-Path -LiteralPath (Join-Path $runtimeRoot ".venv\Scripts\python.exe") -PathType Leaf) {
    $resolvedPythonPath = Join-Path $runtimeRoot ".venv\Scripts\python.exe"
} elseif (Test-Path -LiteralPath (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe") -PathType Leaf) {
    $resolvedPythonPath = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
} else {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $resolvedPythonPath = $pythonCommand.Source
}

$logDirectory = Join-Path $runtimeRoot "logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutPath = Join-Path $logDirectory "dashboard.restart.$stamp.stdout.log"
$stderrPath = Join-Path $logDirectory "dashboard.restart.$stamp.stderr.log"
$previousPythonPath = $env:PYTHONPATH

try {
    $env:PYTHONPATH = Join-Path $sourceRoot "src"
    Write-Host "Starting dashboard with source=$sourceRoot runtime=$runtimeRoot python=$resolvedPythonPath..."
    $dashboardProcess = Start-Process `
        -FilePath $resolvedPythonPath `
        -ArgumentList @(
            "-m", "btc_futures_bot.main",
            "--config", $resolvedConfigPath,
            "--exchange", $Exchange,
            "--web",
            "--host", $BindAddress,
            "--port", "$Port"
        ) `
        -WorkingDirectory $runtimeRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -PassThru
} finally {
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $previousPythonPath
    }
}

try {
    Wait-DashboardReady -Process $dashboardProcess
} catch {
    if (-not $dashboardProcess.HasExited) {
        Stop-Process -Id $dashboardProcess.Id -ErrorAction SilentlyContinue
    }
    throw "$($_.Exception.Message). See $stderrPath"
}

Write-Host "Dashboard is ready: $serviceUrl (PID $($dashboardProcess.Id))"

try {
    # The new dashboard may need time to establish its read-only private
    # snapshot.  Never start the engine until that snapshot is healthy and the
    # exchange exposure still matches the pre-restart baseline.
    $preStartStatus = Wait-EngineStatus `
        -ExpectedRunning $false `
        -TimeoutSeconds 45 `
        -RequireHealthy
    $preStartSnapshot = Get-SafeRestartSnapshot -Status $preStartStatus
    if ($null -eq $beforeSnapshot) {
        # A cold start has no old dashboard snapshot.  The synchronized new
        # dashboard snapshot becomes the immutable baseline for engine start.
        $beforeSnapshot = $preStartSnapshot
    } else {
        Assert-SafeRestartSnapshotUnchanged -Before $beforeSnapshot -After $preStartSnapshot
    }
} catch {
    throw "Dashboard restarted, but the pre-start exchange snapshot failed validation; engine was not started: $($_.Exception.Message)"
}
Write-Host "Pre-start exchange snapshot confirmed: $(Format-SafeRestartSnapshot $preStartSnapshot)"

if ($DashboardOnly) {
    Write-Host "Dashboard-only restart requested; trading engine was not started."
    Write-Host "Logs: $stderrPath"
    return
}

Write-Host "Starting the configured trading engine..."
try {
    $body = @{exchange = $Exchange} | ConvertTo-Json -Compress
    $startResult = Invoke-RestMethod `
        "$serviceUrl/api/start" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 45
    if (-not (ConvertTo-RestartBoolean (Get-RestartValue $startResult @("running")))) {
        throw "Engine start did not report running=true"
    }
} catch {
    $detail = if ($_.ErrorDetails.Message) { $_.ErrorDetails.Message } else { $_.Exception.Message }
    $failClosedResult = Stop-EngineFailClosed -Reason $detail
    throw "The trading engine start was not confirmed; $failClosedResult. Cause: $detail"
}

try {
    $afterStatus = Wait-EngineStatus `
        -ExpectedRunning $true `
        -TimeoutSeconds 60 `
        -RequireHealthy `
        -RequireFirstCycle
    $afterSnapshot = Get-SafeRestartSnapshot -Status $afterStatus
    Assert-SafeRestartSnapshotUnchanged -Before $beforeSnapshot -After $afterSnapshot
} catch {
    $validationError = $_.Exception.Message
    $failClosedResult = Stop-EngineFailClosed -Reason $validationError
    throw (
        "Post-restart validation failed; $failClosedResult. The script submitted no recovery order, but the engine " +
        "may have changed exchange state during its first cycle. Review Binance before any retry. Cause: $validationError"
    )
}

Write-Host "Post-restart exchange snapshot confirmed: $(Format-SafeRestartSnapshot $afterSnapshot)"
Write-Host "Trading engine is running in $($startResult.mode) mode on $($startResult.exchange); first cycle and live/private health confirmed."
Write-Host "Logs: $stderrPath"
