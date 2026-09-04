$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $here
. (Join-Path $repoRoot "scripts\restart_bot_checks.ps1")

function Assert-Throws {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [scriptblock]$Action
    )

    $threw = $false
    try {
        & $Action | Out-Null
    } catch {
        $threw = $true
    }
    if (-not $threw) {
        throw "Expected the action to throw, but it completed successfully"
    }
}

function Assert-DoesNotThrow {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [scriptblock]$Action
    )

    try {
        & $Action | Out-Null
    } catch {
        throw "Expected the action to succeed, but it threw: $($_.Exception.Message)"
    }
}

function New-HeldStatus {
    param(
        [decimal]$Quantity = 0.001,
        [decimal]$EntryPrice = 81119.1,
        [decimal]$PositionStopPrice = 80754.06405,
        [decimal]$ExchangeStopPrice = 80754.0,
        [string]$StopOrderId = "2000001408477909",
        [string]$StopClientId = "btcbot-stop-test",
        [string]$StopSide = "SELL",
        [bool]$ReduceOnly = $true,
        [bool]$ClosePosition = $false,
        [string]$StopStatus = "NEW",
        [switch]$WithoutPriceTick
    )

    $orderSizing = [ordered]@{ quantity_step_raw = "0.001" }
    if (-not $WithoutPriceTick) {
        $orderSizing.price_tick_raw = "0.1"
    }
    return [pscustomobject]@{
        running = $true
        started_at = 100
        last_cycle_at = 101
        last_error = ""
        exchange = "binance"
        environment = "production"
        mode = "live"
        symbol = "BTCUSDT"
        connection = [pscustomobject]@{
            market = $true
            private = $true
            private_stale = $false
            private_error = ""
            private_stream = [pscustomobject]@{
                available = $true
                connected = $true
                ready = $true
                healthy = $true
                last_error = ""
            }
        }
        market = [pscustomobject]@{ stale = $false }
        order_sizing = [pscustomobject]$orderSizing
        positions = @(
            [pscustomobject]@{
                symbol = "BTCUSDT"
                side = "long"
                quantity = $Quantity
                entry_price = $EntryPrice
                stop_price = $PositionStopPrice
            }
        )
        open_orders = @(
            [pscustomobject]@{
                order_id = $StopOrderId
                client_order_id = $StopClientId
                side = $StopSide
                type = "STOP_MARKET"
                status = $StopStatus
                quantity = 0
                stop_price = $ExchangeStopPrice
                reduce_only = $ReduceOnly
                close_position = $ClosePosition
            }
        )
    }
}

function New-FlatStatus {
    return [pscustomobject]@{
        running = $true
        started_at = 100
        last_cycle_at = 101
        last_error = ""
        exchange = "binance"
        environment = "production"
        mode = "live"
        symbol = "BTCUSDT"
        connection = [pscustomobject]@{
            market = $true
            private = $true
            private_stale = $false
            private_error = ""
        }
        market = [pscustomobject]@{ stale = $false }
        order_sizing = [pscustomobject]@{
            quantity_step_raw = "0.001"
            price_tick_raw = "0.1"
        }
        positions = @()
        open_orders = @()
    }
}

Describe "restart_bot PowerShell syntax" {
    It "parses the restart script and pure checks without errors" {
        foreach ($relativePath in @(
            "scripts\restart_bot.ps1",
            "scripts\restart_bot_checks.ps1"
        )) {
            $tokens = $null
            $errors = $null
            $path = Join-Path $repoRoot $relativePath
            [Management.Automation.Language.Parser]::ParseFile(
                $path,
                [ref]$tokens,
                [ref]$errors
            ) | Out-Null
            @($errors).Count | Should Be 0
        }
    }

    It "keeps CheckOnly ahead of every POST and supports a separate runtime directory" {
        $content = Get-Content -LiteralPath (Join-Path $repoRoot "scripts\restart_bot.ps1") -Raw
        $checkOnlyIndex = $content.IndexOf('if ($CheckOnly)')
        $stopFlowIndex = $content.IndexOf('Write-Host "Stopping trading engine and dashboard')
        if ($checkOnlyIndex -lt 0 -or $stopFlowIndex -lt 0 -or $checkOnlyIndex -gt $stopFlowIndex) {
            throw "CheckOnly must return before the mutating restart flow"
        }
        $checkOnlyBlock = $content.Substring($checkOnlyIndex, $stopFlowIndex - $checkOnlyIndex)
        if ($checkOnlyBlock -notmatch '\breturn\b') {
            throw "CheckOnly does not return before the mutating restart flow"
        }
        if ($content -notmatch '\[string\]\$RuntimeWorkingDirectory' -or
            $content -notmatch 'Join-Path \$runtimeRoot \$ConfigPath') {
            throw "RuntimeWorkingDirectory is not wired into relative config resolution"
        }
    }

    It "establishes a healthy cold-start baseline before the engine start POST" {
        $content = Get-Content -LiteralPath (Join-Path $repoRoot "scripts\restart_bot.ps1") -Raw
        $pythonPathInitIndex = $content.IndexOf('$runningPythonPath = $null')
        $listenerBranchIndex = $content.IndexOf('if ($servicePid)')
        $pythonResolutionIndex = $content.IndexOf('if ($PythonPath)')
        $preStartIndex = $content.IndexOf('$preStartStatus = Wait-EngineStatus')
        $coldBaselineIndex = $content.IndexOf('$beforeSnapshot = $preStartSnapshot')
        $engineStartIndex = $content.IndexOf('$startResult = Invoke-RestMethod')
        if ($pythonPathInitIndex -lt 0 -or $listenerBranchIndex -lt $pythonPathInitIndex -or
            $pythonResolutionIndex -lt $listenerBranchIndex) {
            throw "Cold start must initialize the optional running Python path before listener branching and Python resolution"
        }
        if ($preStartIndex -lt 0 -or $coldBaselineIndex -lt $preStartIndex -or
            $engineStartIndex -lt $coldBaselineIndex) {
            throw "Cold-start snapshot must become the baseline before /api/start"
        }
        if ($content -notmatch 'if \(\$servicePid\)' -or
            $content -notmatch 'fail-closed cold-start validation') {
            throw "No-listener cold-start branch is missing"
        }
    }
}

Describe "Get-SafeRestartSnapshot" {
    It "accepts the real tick-normalized long stop" {
        $snapshot = Get-SafeRestartSnapshot -Status (New-HeldStatus)

        $snapshot.HadPosition | Should Be $true
        $snapshot.Position.NormalizedQuantity | Should Be ([decimal]0.001)
        $snapshot.Stop.StopPrice | Should Be ([decimal]80754.0)
        $snapshot.Stop.OrderId | Should Be "2000001408477909"
        $snapshot.OpenOrderCount | Should Be 1
    }

    It "uses the BTCUSDT tick fallback when status omits price_tick" {
        $snapshot = Get-SafeRestartSnapshot -Status (New-HeldStatus -WithoutPriceTick)

        $snapshot.PriceTick | Should Be ([decimal]0.1)
        $snapshot.Stop.NormalizedStopPrice | Should Be ([decimal]80754.0)
    }

    It "does not apply the BTCUSDT tick fallback to another exchange" {
        $status = New-HeldStatus -WithoutPriceTick
        $status.exchange = "okx"

        Assert-Throws { Get-SafeRestartSnapshot -Status $status }
    }

    It "accepts closePosition when reduceOnly is absent" {
        $status = New-HeldStatus -ReduceOnly:$false -ClosePosition:$true
        $order = $status.open_orders[0]
        $order.PSObject.Properties.Remove("reduce_only")
        $order.PSObject.Properties.Remove("close_position")
        $order | Add-Member -NotePropertyName "closePosition" -NotePropertyValue $true
        $snapshot = Get-SafeRestartSnapshot -Status $status

        $snapshot.Stop.ClosePosition | Should Be $true
    }

    It "rejects a local stop that normalizes to a different exchange tick" {
        Assert-Throws {
            Get-SafeRestartSnapshot -Status (
                New-HeldStatus -PositionStopPrice 80754.16405 -ExchangeStopPrice 80754.0
            )
        }
    }

    It "rejects a position quantity that is not aligned to the exchange step" {
        Assert-Throws { Get-SafeRestartSnapshot -Status (New-HeldStatus -Quantity 0.0015) }
    }

    It "rejects a missing protective stop" {
        $status = New-HeldStatus
        $status.open_orders = @()

        Assert-Throws { Get-SafeRestartSnapshot -Status $status }
    }

    It "rejects duplicate or unrelated orders" {
        $status = New-HeldStatus
        $status.open_orders += [pscustomobject]@{
            order_id = "extra"
            client_order_id = "manual"
            side = "BUY"
            type = "LIMIT"
            status = "NEW"
        }

        Assert-Throws { Get-SafeRestartSnapshot -Status $status }
    }

    It "rejects a stop on the wrong side" {
        Assert-Throws { Get-SafeRestartSnapshot -Status (New-HeldStatus -StopSide "BUY") }
    }

    It "rejects a non-protective or inactive stop" {
        Assert-Throws { Get-SafeRestartSnapshot -Status (New-HeldStatus -ReduceOnly:$false) }
        Assert-Throws { Get-SafeRestartSnapshot -Status (New-HeldStatus -StopStatus "FINISHED") }
    }

    It "records a genuinely flat account" {
        $snapshot = Get-SafeRestartSnapshot -Status (New-FlatStatus)

        $snapshot.HadPosition | Should Be $false
        $snapshot.PositionCount | Should Be 0
        $snapshot.OpenOrderCount | Should Be 0
    }

    It "rejects an order on a flat account" {
        $status = New-FlatStatus
        $status.open_orders = (New-HeldStatus).open_orders

        Assert-Throws { Get-SafeRestartSnapshot -Status $status }
    }
}

Describe "Assert-SafeRestartSnapshotUnchanged" {
    It "accepts an unchanged protected position using tick and step values" {
        $before = Get-SafeRestartSnapshot -Status (New-HeldStatus)
        $after = Get-SafeRestartSnapshot -Status (
            New-HeldStatus -Quantity 0.0010000 -EntryPrice 81119.1 -PositionStopPrice 80754.099
        )

        Assert-DoesNotThrow { Assert-SafeRestartSnapshotUnchanged -Before $before -After $after }
    }

    It "rejects a changed position quantity" {
        $before = Get-SafeRestartSnapshot -Status (New-HeldStatus)
        $after = Get-SafeRestartSnapshot -Status (New-HeldStatus -Quantity 0.002)

        Assert-Throws { Assert-SafeRestartSnapshotUnchanged -Before $before -After $after }
    }

    It "rejects a changed stop id or trigger" {
        $before = Get-SafeRestartSnapshot -Status (New-HeldStatus)
        $changedId = Get-SafeRestartSnapshot -Status (New-HeldStatus -StopOrderId "different")
        $changedTrigger = Get-SafeRestartSnapshot -Status (
            New-HeldStatus -PositionStopPrice 80753.96405 -ExchangeStopPrice 80753.9
        )

        Assert-Throws { Assert-SafeRestartSnapshotUnchanged -Before $before -After $changedId }
        Assert-Throws { Assert-SafeRestartSnapshotUnchanged -Before $before -After $changedTrigger }
    }

    It "fails closed on held-to-flat and flat-to-held transitions" {
        $held = Get-SafeRestartSnapshot -Status (New-HeldStatus)
        $flat = Get-SafeRestartSnapshot -Status (New-FlatStatus)

        Assert-Throws { Assert-SafeRestartSnapshotUnchanged -Before $held -After $flat }
        Assert-Throws { Assert-SafeRestartSnapshotUnchanged -Before $flat -After $held }
    }
}

Describe "Assert-RestartStatusHealth" {
    It "requires running, healthy private data, and a completed first cycle" {
        $status = New-HeldStatus

        Assert-DoesNotThrow { Assert-RestartStatusHealth -Status $status -RequireRunning -RequireFirstCycle }
    }

    It "allows a historical last_error only after running=false is confirmed" {
        $status = New-HeldStatus
        $status.running = $false
        $status.last_error = "temporary TLS timeout"

        Assert-DoesNotThrow {
            Assert-RestartStatusHealth -Status $status -RequireStopped -AllowLastError
        }
        Assert-Throws { Assert-RestartStatusHealth -Status $status -RequireStopped }
    }

    It "allows an existing stopped dashboard with a historical last_error" {
        $status = New-HeldStatus
        $status.running = $false
        $status.last_error = "temporary TLS timeout"

        Assert-DoesNotThrow { Assert-SafeRestartPreflightStatus -Status $status }
    }

    It "rejects an existing running dashboard with a last_error" {
        $status = New-HeldStatus
        $status.last_error = "current engine failure"

        Assert-Throws { Assert-SafeRestartPreflightStatus -Status $status }
    }

    It "rejects stale private data, last_error, or a missing first cycle" {
        $stale = New-HeldStatus
        $stale.connection.private_stale = $true
        $errored = New-HeldStatus
        $errored.last_error = "boom"
        $noCycle = New-HeldStatus
        $noCycle.last_cycle_at = 0

        Assert-Throws { Assert-RestartStatusHealth -Status $stale -RequireRunning }
        Assert-Throws { Assert-RestartStatusHealth -Status $errored -RequireRunning }
        Assert-Throws { Assert-RestartStatusHealth -Status $noCycle -RequireRunning -RequireFirstCycle }
    }
}
