[CmdletBinding()]
param(
    [string]$ConfigPath = "config.binance.testnet.json",
    [string]$Exchange = "binance",
    [string]$BindAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8787,
    [string]$PythonPath = "",
    [switch]$DashboardOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$resolvedConfigPath = if ([IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath
} else {
    Join-Path $repoRoot $ConfigPath
}
if (-not (Test-Path -LiteralPath $resolvedConfigPath -PathType Leaf)) {
    throw "Config file not found: $resolvedConfigPath"
}

$probeAddress = if ($BindAddress -in @("0.0.0.0", "::")) { "127.0.0.1" } else { $BindAddress }
$serviceUrl = "http://${probeAddress}:$Port"
$runningPythonPath = ""

function Get-DashboardListener {
    $listeners = @(
        Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
    if ($listeners.Count -gt 1) {
        throw "More than one process is listening on port ${Port}: $($listeners -join ', ')"
    }
    return $listeners | Select-Object -First 1
}

function Wait-DashboardReady {
    param(
        [Parameter(Mandatory = $true)]
        [Diagnostics.Process]$Process,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
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
            Start-Sleep -Milliseconds 500
        }
    }
    throw "Dashboard did not become ready within $TimeoutSeconds seconds"
}

$servicePid = Get-DashboardListener
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

    Write-Host "Stopping trading engine and dashboard (PID $servicePid)..."
    try {
        Invoke-RestMethod "$serviceUrl/api/stop" -Method Post -ContentType "application/json" -Body "{}" -TimeoutSec 15 | Out-Null
    } catch {
        Write-Warning "Graceful engine stop returned an error; stopping the verified dashboard process. $($_.Exception.Message)"
    }
    Stop-Process -Id $servicePid -ErrorAction Stop
    Wait-Process -Id $servicePid -Timeout 10 -ErrorAction SilentlyContinue

    $portDeadline = (Get-Date).AddSeconds(10)
    while ((Get-DashboardListener) -and (Get-Date) -lt $portDeadline) {
        Start-Sleep -Milliseconds 250
    }
    if (Get-DashboardListener) {
        throw "Port $Port was not released after stopping PID $servicePid"
    }
}

if ($PythonPath) {
    $resolvedPythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
} elseif ($runningPythonPath -and (Test-Path -LiteralPath $runningPythonPath -PathType Leaf)) {
    $resolvedPythonPath = $runningPythonPath
} elseif (Test-Path -LiteralPath (Join-Path $repoRoot ".venv\Scripts\python.exe") -PathType Leaf) {
    $resolvedPythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
} elseif (Test-Path -LiteralPath (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe") -PathType Leaf) {
    $resolvedPythonPath = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
} else {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $resolvedPythonPath = $pythonCommand.Source
}

$logDirectory = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutPath = Join-Path $logDirectory "dashboard.restart.$stamp.stdout.log"
$stderrPath = Join-Path $logDirectory "dashboard.restart.$stamp.stderr.log"
$previousPythonPath = $env:PYTHONPATH

try {
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    Write-Host "Starting dashboard with $resolvedPythonPath..."
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
        -WorkingDirectory $repoRoot `
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

if (-not $DashboardOnly) {
    Write-Host "Starting the configured trading engine..."
    try {
        $body = @{exchange = $Exchange} | ConvertTo-Json -Compress
        $startResult = Invoke-RestMethod "$serviceUrl/api/start" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 45
        if (-not $startResult.running) {
            throw "Engine start did not report running=true"
        }
        Write-Host "Trading engine is running in $($startResult.mode) mode on $($startResult.exchange)."
    } catch {
        $detail = if ($_.ErrorDetails.Message) { $_.ErrorDetails.Message } else { $_.Exception.Message }
        throw "Dashboard is running, but the trading engine could not start: $detail"
    }
} else {
    Write-Host "Dashboard-only restart requested; trading engine was not started."
}

Write-Host "Logs: $stderrPath"
