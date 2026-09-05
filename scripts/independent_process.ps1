# Launch through the Windows WMI service, outside the calling terminal's job.
function Resolve-ServicePython {
    param([string]$PythonPath, [string]$RuntimeRoot)
    if (-not $PythonPath) {
        $localPython = Join-Path $RuntimeRoot '.venv\Scripts\python.exe'
        $PythonPath = if (Test-Path -LiteralPath $localPython) { $localPython } else { (Get-Command python -ErrorAction Stop).Source }
    }
    $PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
    $base = & $PythonPath -c 'import sys; print(sys.base_prefix)'
    if ($LASTEXITCODE -ne 0 -or -not $base) { throw 'Python runtime validation failed' }
    if ("$PythonPath $base" -match '(?i)codex') {
        throw 'Install standalone Python and recreate .venv; service Python must not depend on Codex.'
    }
    return $PythonPath
}

function Start-IndependentDashboard {
    param(
        [string]$PythonPath, [string]$SourceRoot, [string]$RuntimeRoot,
        [string[]]$Arguments, [string]$StdoutPath, [string]$StderrPath
    )
    $specPath = "$StdoutPath.launch.json"
    @{
        source_root = $SourceRoot; runtime_root = $RuntimeRoot
        arguments = $Arguments; stdout = $StdoutPath; stderr = $StderrPath
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $specPath -Encoding UTF8
    $runner = Join-Path $PSScriptRoot 'independent_process.py'
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow = [uint16]0}
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = ('"{0}" "{1}" "{2}"' -f $PythonPath, $runner, $specPath)
        CurrentDirectory = $RuntimeRoot
        ProcessStartupInformation = $startup
    }
    if ($result.ReturnValue -ne 0) {
        Remove-Item -LiteralPath $specPath -ErrorAction SilentlyContinue
        throw "Windows independent launch failed (WMI code $($result.ReturnValue)); no terminal-child fallback was used."
    }
    return Get-Process -Id $result.ProcessId -ErrorAction Stop
}
