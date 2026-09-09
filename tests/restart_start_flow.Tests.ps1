$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
. (Join-Path $repoRoot "scripts\restart_bot_checks.ps1")
$parseTokens = $null
$parseErrors = $null
$restartAst = [Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $repoRoot "scripts\restart_bot.ps1"), [ref]$parseTokens, [ref]$parseErrors
)
$startTry = $restartAst.FindAll({
    param($node)
    $node -is [Management.Automation.Language.TryStatementAst] -and
        $node.Extent.Text.Contains('$startDeadline =')
}, $true) | Select-Object -First 1
if ($null -eq $startTry) { throw "Deployment start block not found" }
$startBlock = [scriptblock]::Create($startTry.Extent.Text)

# Execute the real deployment start block with every network and safety probe
# replaced by deterministic mocks. No dashboard/process/order is touched.
function Wait-EngineStatus { param($ExpectedRunning, $TimeoutSeconds, [switch]$RequireHealthy) }
function Stop-EngineFailClosed { param($Reason) }

Describe "restart script deferred startup flow" {
    BeforeEach {
        $serviceUrl = "http://127.0.0.1:0"
        $Exchange = "binance"
        $beforeSnapshot = "unchanged"
        $script:startPosts = 0
        $script:events = New-Object 'Collections.Generic.List[string]'
        Mock Wait-EngineStatus {
            $script:events.Add("stopped-health")
            if ($ExpectedRunning -ne $false -or -not $RequireHealthy) {
                throw "Retry must verify stopped and healthy"
            }
            return @{ running = $false }
        }
        Mock Get-SafeRestartSnapshot { return "unchanged" }
        Mock Format-SafeRestartSnapshot { return "mock flat state" }
        Mock Assert-SafeRestartSnapshotUnchanged {
            $script:events.Add("unchanged")
        }
        Mock Stop-EngineFailClosed { $script:events.Add("stop"); return "stop confirmed" }
    }

    It "rechecks stopped health and unchanged exposure before replaying only a rejected start" {
        Mock Invoke-RestMethod {
            $script:startPosts += 1
            $script:events.Add("post")
            if ($script:startPosts -eq 1) {
                $failure = New-Object Management.Automation.ErrorRecord(
                    (New-Object Exception("start rejected")), "HttpError", "InvalidOperation", $null
                )
                $failure.ErrorDetails = New-Object Management.Automation.ErrorDetails(
                    '{"error":"deferred","retry":{"api_code":"LOCAL_REQUEST_BUDGET","retry_at":1}}'
                )
                throw $failure
            }
            return @{ running = $true; exchange = "binance"; mode = "live" }
        }
        & $startBlock
        if (($script:events -join ",") -ne "post,stopped-health,unchanged,post") {
            throw "Wrong retry safety order: $($script:events -join ',')"
        }
        Assert-MockCalled Invoke-RestMethod -Times 2 -Exactly -Scope It -ParameterFilter {
            $Uri -eq "$serviceUrl/api/start" -and $Method -eq "Post"
        }
    }

    It "never retries when the exchange baseline changed during deferral" {
        Mock Invoke-RestMethod {
            $script:startPosts += 1
            $failure = New-Object Management.Automation.ErrorRecord(
                (New-Object Exception("start rejected")), "HttpError", "InvalidOperation", $null
            )
            $failure.ErrorDetails = New-Object Management.Automation.ErrorDetails(
                '{"error":"limited","retry":{"upstream_status":429,"retry_at":1}}'
            )
            throw $failure
        }
        Mock Assert-SafeRestartSnapshotUnchanged { throw "exchange position changed" }
        $threw = $false
        try { & $startBlock } catch { $threw = $true }
        if (-not $threw -or $script:startPosts -ne 1) { throw "Changed exposure was retried" }
        Assert-MockCalled Stop-EngineFailClosed -Times 1 -Exactly -Scope It
    }

    It "does not replay an ambiguous start timeout" {
        Mock Invoke-RestMethod {
            $script:startPosts += 1
            throw "request timed out"
        }
        $threw = $false
        try { & $startBlock } catch { $threw = $true }
        if (-not $threw -or $script:startPosts -ne 1) { throw "Ambiguous timeout was retried" }
        Assert-MockCalled Wait-EngineStatus -Times 0 -Exactly -Scope It
        Assert-MockCalled Stop-EngineFailClosed -Times 1 -Exactly -Scope It
    }
}
