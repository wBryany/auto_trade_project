[CmdletBinding()]
param(
    [string]$BindAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8788,
    [string]$PythonPath = "",
    [switch]$DashboardOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$sourceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$configPath = Join-Path $sourceRoot "config.binance.model2.json"
$probeAddress = if ($BindAddress -in @("0.0.0.0", "::")) { "127.0.0.1" } else { $BindAddress }
$serviceUrl = "http://${probeAddress}:$Port"

if ($PythonPath) {
    $resolvedPythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
} elseif (Test-Path -LiteralPath (Join-Path $sourceRoot ".venv\Scripts\python.exe") -PathType Leaf) {
    $resolvedPythonPath = Join-Path $sourceRoot ".venv\Scripts\python.exe"
} elseif (Test-Path -LiteralPath (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe") -PathType Leaf) {
    $resolvedPythonPath = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
} else {
    $resolvedPythonPath = (Get-Command python -ErrorAction Stop).Source
}

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $sourceRoot "src"
    $configurationIdentityJson = (
        & $resolvedPythonPath -c `
            "import json, sys; from btc_futures_bot.main import load_config; c = load_config(sys.argv[1]); m = c.get('trade_model') or {}; print(json.dumps({'mode': str(c.get('mode', 'paper')), 'instance_id': str(c.get('instance_id', '')), 'trade_model_mode': str(m.get('mode', 'off'))}))" `
            $configPath
    ).Trim()
} finally {
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $previousPythonPath
    }
}
$configurationIdentity = $configurationIdentityJson | ConvertFrom-Json
$configuredMode = ([string]$configurationIdentity.mode).Trim().ToLowerInvariant()
$configuredInstanceId = ([string]$configurationIdentity.instance_id).Trim()
$configuredTradeModelMode = ([string]$configurationIdentity.trade_model_mode).Trim().ToLowerInvariant()
if ($configuredInstanceId -ne "trade-model-2") {
    throw "Model 2 config must declare instance_id=trade-model-2; got '$configuredInstanceId'"
}

function Get-Model2Listener {
    $listeners = @(
        Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
    if ($listeners.Count -eq 0) {
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

$enforcedModelStatus = $null
if ($configuredTradeModelMode -eq "enforce") {
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = Join-Path $sourceRoot "src"
        $enforcedModelStatusJson = (
            & $resolvedPythonPath -c `
                "import json, sys; from pathlib import Path; from btc_futures_bot.main import load_config; from btc_futures_bot.trade_model import EntryGate, MetaModelConfig; c = load_config(sys.argv[1]); g = EntryGate(MetaModelConfig.from_mapping(c, c.get('_config_dir') or Path.cwd())); s = g.status(); g.close(); print(json.dumps(s))" `
                $configPath
        ).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "Model 2 artifact preflight failed"
        }
    } finally {
        if ($null -eq $previousPythonPath) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $previousPythonPath
        }
    }
    $enforcedModelStatus = $enforcedModelStatusJson | ConvertFrom-Json
    if (-not [bool]$enforcedModelStatus.ready -or [string]::IsNullOrWhiteSpace([string]$enforcedModelStatus.model_version)) {
        throw "Model 2 enforce mode requires a ready, versioned artifact: $($enforcedModelStatus.error)"
    }
}

if ($configuredMode -eq "live") {
    if ($configuredTradeModelMode -ne "enforce") {
        throw "Model 2 live startup requires trade_model.mode=enforce; got $configuredTradeModelMode"
    }
    if (-not [bool]$enforcedModelStatus.approved_for_live) {
        throw "Model 2 artifact is not ready, versioned, and approved for live execution"
    }
    $liveListener = Get-Model2Listener
    if ($liveListener) {
        try {
            $existingLiveStatus = Invoke-RestMethod "$serviceUrl/api/status" -TimeoutSec 10
        } catch {
            throw "Port $Port is occupied by PID $liveListener, but it is not a healthy Model 2 dashboard"
        }
        $existingTradeModel = $existingLiveStatus.PSObject.Properties["trade_model"]
        if (
            [string]$existingLiveStatus.instance_id -ne "trade-model-2" -or
            $null -eq $existingTradeModel -or
            [string]$existingTradeModel.Value.type -ne "lightgbm_meta"
        ) {
            throw "Refusing to restart port $Port because its current listener is not the trade-model-2 instance"
        }
    }
    # Live uses the hardened exchange-snapshot restart checks. The model's own
    # live approval is checked independently by Engine.prepare_live().
    $arguments = @{
        ConfigPath = $configPath
        Exchange = "binance"
        BindAddress = $BindAddress
        Port = $Port
        PythonPath = $resolvedPythonPath
        RuntimeWorkingDirectory = $sourceRoot
        LogPrefix = "dashboard.model2"
    }
    if ($DashboardOnly) {
        $arguments.DashboardOnly = $true
    }
    & (Join-Path $PSScriptRoot "restart_bot.ps1") @arguments
    exit $LASTEXITCODE
}
if ($configuredMode -ne "paper") {
    throw "Model 2 startup supports paper or live mode; got $configuredMode"
}

function Wait-Model2Dashboard {
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
            throw "Model 2 dashboard exited with code $($Process.ExitCode)"
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
    throw "Model 2 dashboard did not become ready: $lastError"
}

$listener = Get-Model2Listener
if ($listener) {
    try {
        $version = Invoke-RestMethod "$serviceUrl/api/version" -TimeoutSec 5
        $status = Invoke-RestMethod "$serviceUrl/api/status" -TimeoutSec 10
    } catch {
        throw "Port $Port is occupied by PID $listener, but it is not a healthy bot dashboard"
    }
    if ([string]$status.mode -ne "paper") {
        throw "Existing dashboard on port $Port is not in paper mode"
    }
    if ([string]$status.instance_id -ne "trade-model-2") {
        throw "Existing dashboard on port $Port is not the trade-model-2 instance"
    }
    $tradeModelProperty = $status.PSObject.Properties["trade_model"]
    if ($null -eq $tradeModelProperty -or [string]$tradeModelProperty.Value.type -ne "lightgbm_meta") {
        throw "Existing dashboard on port $Port is not the Model 2 service"
    }
    if ([string]$tradeModelProperty.Value.mode -ne $configuredTradeModelMode) {
        throw "Existing Model 2 dashboard uses a different trade-model mode"
    }
    Write-Host "Model 2 dashboard is already ready: $serviceUrl (PID $listener)"
} else {
    $logDirectory = Join-Path $sourceRoot "logs"
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdoutPath = Join-Path $logDirectory "dashboard.model2.$stamp.stdout.log"
    $stderrPath = Join-Path $logDirectory "dashboard.model2.$stamp.stderr.log"
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = Join-Path $sourceRoot "src"
        $dashboardProcess = Start-Process `
            -FilePath $resolvedPythonPath `
            -ArgumentList @(
                "-m", "btc_futures_bot.main",
                "--config", $configPath,
                "--exchange", "binance",
                "--web",
                "--host", $BindAddress,
                "--port", "$Port"
            ) `
            -WorkingDirectory $sourceRoot `
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
        Wait-Model2Dashboard -Process $dashboardProcess
    } catch {
        if (-not $dashboardProcess.HasExited) {
            Stop-Process -Id $dashboardProcess.Id -ErrorAction SilentlyContinue
        }
        throw "$($_.Exception.Message). See $stderrPath"
    }
    Write-Host "Model 2 dashboard is ready: $serviceUrl (PID $($dashboardProcess.Id))"
}

if ($DashboardOnly) {
    Write-Host "Dashboard-only startup requested; the paper engine was not started."
    return
}

$status = Invoke-RestMethod "$serviceUrl/api/status" -TimeoutSec 10
if (-not [bool]$status.running) {
    $body = @{exchange = "binance"} | ConvertTo-Json -Compress
    Invoke-RestMethod `
        "$serviceUrl/api/start" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 45 |
        Out-Null
}

$deadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $deadline) {
    $status = Invoke-RestMethod "$serviceUrl/api/status" -TimeoutSec 10
    if ($status.running -and [double]$status.last_cycle_at -ge [double]$status.started_at) {
        if (-not [string]::IsNullOrWhiteSpace([string]$status.last_error)) {
            throw "Model 2 first cycle failed: $($status.last_error)"
        }
        Write-Host "Model 2 paper engine is running on $serviceUrl; first cycle confirmed."
        return
    }
    Start-Sleep -Milliseconds 500
}
throw "Model 2 paper engine did not complete its first cycle within 60 seconds"
