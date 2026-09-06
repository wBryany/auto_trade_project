from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_POWERSHELL = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
_REPOSITORY = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="Model 2 launcher guards require Windows PowerShell",
)

# Parse the real launcher but execute only its existing-listener guard. Every
# external operation in that block is mocked; no port, process or file changes
# are needed to test paper restart protection and mode migration.
_GUARD_HARNESS = r"""
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$auditTokens = $null
$auditParseErrors = $null
$auditAst = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path (Get-Location).Path 'scripts\start_model2.ps1'),
    [ref]$auditTokens,
    [ref]$auditParseErrors
)
if ($auditParseErrors.Count -ne 0) { throw ($auditParseErrors | Out-String) }
$auditGuardNodes = @($auditAst.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.IfStatementAst] -and
    $node.Extent.Text.StartsWith('if ($listener) {') -and
    $node.Extent.Text.Contains('if ($Restart)')
}, $true))
if ($auditGuardNodes.Count -ne 1) { throw 'Expected exactly one existing-listener guard' }
$auditGuardNode = $auditGuardNodes[0]
# Fail before executing if future launcher changes introduce an unmocked
# command. This keeps the test incapable of accidentally performing I/O.
$auditAllowedCommands = @('Invoke-RestMethod', 'Stop-Process', 'Wait-Process', 'Out-Null', 'Write-Host')
foreach ($auditCommand in $auditGuardNode.FindAll({
    param($node) $node -is [System.Management.Automation.Language.CommandAst]
}, $true)) {
    if ($auditCommand.GetCommandName() -notin $auditAllowedCommands) {
        throw "Unmocked command in restart guard: $($auditCommand.GetCommandName())"
    }
}
$auditGuard = [scriptblock]::Create($auditGuardNode.Extent.Text)

function Test-RestartGuard {
    param(
        [string]$Case,
        [bool]$DoRestart,
        [string]$ExpectedInstance = 'trade-model-2',
        [bool]$PrePosition = $false,
        [bool]$PostPosition = $false,
        [bool]$StillRunning = $false
    )
    $script:auditCaseState = @{
        Calls = [System.Collections.Generic.List[string]]::new()
        StatusReads = 0
    }
    $listener = 424242
    $Port = 8788
    $serviceUrl = 'http://non-network.invalid:8788'
    $configuredExchange = 'okx'
    $configuredTradeModelMode = 'shadow'
    $Restart = $DoRestart
    $prePositions = @()
    $postPositions = @()
    if ($PrePosition) { $prePositions = @([pscustomobject]@{quantity=1}) }
    if ($PostPosition) { $postPositions = @([pscustomobject]@{quantity=1}) }
    $initial = [pscustomobject]@{
        mode='paper'
        instance_id=$ExpectedInstance
        exchange='okx'
        running=$true
        trade_model=[pscustomobject]@{type='lightgbm_meta'; mode='enforce'}
        positions=$prePositions
        open_orders=@()
        last_result=[pscustomobject]@{exchange='okx'}
    }
    $after = [pscustomobject]@{
        running=$StillRunning
        positions=$postPositions
        open_orders=@()
    }
    function Invoke-RestMethod {
        param($Uri, $Method, $ContentType, $Body, $TimeoutSec)
        if ($Uri -eq "$serviceUrl/api/version") { return [pscustomobject]@{version='mock'} }
        if ($Uri -eq "$serviceUrl/api/stop") {
            if ($Method -ne 'Post') { throw 'Unexpected stop method' }
            $script:auditCaseState.Calls.Add('stop')
            return [pscustomobject]@{running=$false}
        }
        if ($Uri -eq "$serviceUrl/api/status") {
            $script:auditCaseState.StatusReads++
            if ($script:auditCaseState.StatusReads -eq 1) { return $initial }
            return $after
        }
        throw 'Unexpected mocked URI'
    }
    function Stop-Process {
        param($Id, $ErrorAction)
        if ($Id -ne 424242) { throw 'Unexpected process identifier' }
        $script:auditCaseState.Calls.Add('kill')
    }
    function Wait-Process {
        param($Id, $Timeout, $ErrorAction)
        if ($Id -ne 424242) { throw 'Unexpected process identifier' }
    }
    $errorText = ''
    try { & $auditGuard } catch { $errorText = $_.Exception.Message }
    return [pscustomobject]@{
        case=$Case
        calls=$script:auditCaseState.Calls.ToArray()
        error=$errorText
    }
}
$auditCases = @(
    Test-RestartGuard -Case 'explicit_flat_enforce_to_shadow' -DoRestart $true
    Test-RestartGuard -Case 'mode_change_without_restart' -DoRestart $false
    Test-RestartGuard -Case 'wrong_instance' -DoRestart $true -ExpectedInstance 'main'
    Test-RestartGuard -Case 'position_before_stop' -DoRestart $true -PrePosition $true
    Test-RestartGuard -Case 'position_after_stop' -DoRestart $true -PostPosition $true
    Test-RestartGuard -Case 'thread_still_running' -DoRestart $true -StillRunning $true
)
[pscustomobject]@{parse_error_count=$auditParseErrors.Count; cases=$auditCases} |
    ConvertTo-Json -Depth 5 -Compress
"""


@pytest.fixture(scope="module")
def restart_guard_results() -> dict[str, dict[str, object]]:
    assert _POWERSHELL is not None
    command = base64.b64encode(_GUARD_HARNESS.encode("utf-16-le")).decode("ascii")
    completed = subprocess.run(
        [_POWERSHELL, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", command],
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["parse_error_count"] == 0
    results = {case["case"]: case for case in payload["cases"]}
    assert len(results) == 6
    return results


@pytest.mark.parametrize(
    "case,calls,error_fragment",
    [
        ("explicit_flat_enforce_to_shadow", ["stop", "kill"], ""),
        ("mode_change_without_restart", [], "use -Restart"),
        ("wrong_instance", [], "not the trade-model-2 instance"),
        ("position_before_stop", [], "has positions or orders"),
        ("position_after_stop", ["stop"], "stopped but not flat"),
        ("thread_still_running", ["stop"], "stop was not confirmed"),
    ],
)
def test_model2_restart_guard_is_fail_closed(
    restart_guard_results: dict[str, dict[str, object]],
    case: str,
    calls: list[str],
    error_fragment: str,
) -> None:
    result = restart_guard_results[case]
    assert result["calls"] == calls
    if error_fragment:
        assert error_fragment in result["error"]
    else:
        assert result["error"] == ""
