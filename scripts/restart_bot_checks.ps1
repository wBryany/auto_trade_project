Set-StrictMode -Version Latest

function Get-RestartValue {
    param(
        [AllowNull()]
        [object]$Object,
        [Parameter(Mandatory = $true)]
        [string[]]$Names,
        [AllowNull()]
        [object]$Default = $null
    )

    if ($null -eq $Object) {
        return $Default
    }
    foreach ($name in $Names) {
        if ($Object -is [Collections.IDictionary] -and $Object.Contains($name)) {
            return $Object[$name]
        }
        $property = $Object.PSObject.Properties[$name]
        if ($null -ne $property) {
            return $property.Value
        }
    }
    return $Default
}

function ConvertTo-RestartBoolean {
    param(
        [AllowNull()]
        [object]$Value,
        [bool]$Default = $false
    )

    if ($null -eq $Value) {
        return $Default
    }
    if ($Value -is [bool]) {
        return [bool]$Value
    }
    switch (([string]$Value).Trim().ToLowerInvariant()) {
        { $_ -in @("1", "true", "yes", "on") } { return $true }
        { $_ -in @("0", "false", "no", "off", "") } { return $false }
        default { throw "Expected a boolean status field, got '$Value'" }
    }
}

function ConvertTo-RestartDecimal {
    param(
        [AllowNull()]
        [object]$Value,
        [Parameter(Mandatory = $true)]
        [string]$Field
    )

    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        throw "Missing numeric status field: $Field"
    }
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $text = if ($Value -is [IFormattable]) {
        $Value.ToString($null, $culture)
    } else {
        [string]$Value
    }
    $parsed = [decimal]0
    if (-not [decimal]::TryParse(
        $text,
        [Globalization.NumberStyles]::Float,
        $culture,
        [ref]$parsed
    )) {
        throw "Invalid numeric status field ${Field}: '$text'"
    }
    return $parsed
}

function ConvertTo-RestartStepValue {
    param(
        [Parameter(Mandatory = $true)]
        [decimal]$Value,
        [Parameter(Mandatory = $true)]
        [decimal]$Step,
        [ValidateSet("Down", "Up", "Nearest")]
        [string]$Direction = "Nearest"
    )

    if ($Step -le 0) {
        throw "Normalization step must be positive"
    }
    $units = $Value / $Step
    $normalizedUnits = switch ($Direction) {
        "Down" { [decimal]::Floor($units) }
        "Up" { [decimal]::Ceiling($units) }
        default { [decimal]::Round($units, 0, [MidpointRounding]::AwayFromZero) }
    }
    return $normalizedUnits * $Step
}

function Get-RestartQuantityStep {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Status
    )

    $orderSizing = Get-RestartValue $Status @("order_sizing", "orderSizing")
    $raw = Get-RestartValue $orderSizing @(
        "quantity_step_raw", "quantityStepRaw", "quantity_step", "quantityStep", "step_size", "stepSize"
    )
    $step = ConvertTo-RestartDecimal $raw "order_sizing.quantity_step"
    if ($step -le 0) {
        throw "order_sizing.quantity_step must be positive"
    }
    return $step
}

function Get-RestartPriceTick {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Status,
        [Parameter(Mandatory = $true)]
        [string]$Symbol
    )

    $orderSizing = Get-RestartValue $Status @("order_sizing", "orderSizing")
    $market = Get-RestartValue $Status @("market")
    $raw = Get-RestartValue $orderSizing @(
        "price_tick_raw", "priceTickRaw", "price_tick", "priceTick", "tick_size", "tickSize"
    )
    if ($null -eq $raw) {
        $raw = Get-RestartValue $market @(
            "price_tick_raw", "priceTickRaw", "price_tick", "priceTick", "tick_size", "tickSize"
        )
    }
    if ($null -ne $raw -and -not [string]::IsNullOrWhiteSpace([string]$raw)) {
        $tick = ConvertTo-RestartDecimal $raw "price_tick"
        if ($tick -le 0) {
            throw "price_tick must be positive"
        }
        return $tick
    }

    # The current dashboard does not expose PRICE_FILTER.tickSize.  Fail closed
    # for unknown products; BTCUSDT's production USD-M tick is 0.1.
    $exchange = ([string](Get-RestartValue $Status @("exchange") "")).Trim().ToLowerInvariant()
    if ($exchange -eq "binance" -and $Symbol.Trim().ToUpperInvariant() -eq "BTCUSDT") {
        return [decimal]0.1
    }
    throw "Dashboard status does not expose price_tick for $Symbol; safe comparison is unavailable"
}

function Test-RestartOptionalFalse {
    param(
        [AllowNull()]
        [object]$Object,
        [Parameter(Mandatory = $true)]
        [string[]]$Names
    )

    foreach ($name in $Names) {
        $missing = New-Object object
        $value = Get-RestartValue $Object @($name) $missing
        if (-not [object]::ReferenceEquals($missing, $value)) {
            return -not (ConvertTo-RestartBoolean $value)
        }
    }
    return $true
}

function Assert-RestartStatusHealth {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Status,
        [switch]$RequireRunning,
        [switch]$RequireStopped,
        [switch]$RequireFirstCycle,
        [switch]$AllowLastError
    )

    if ($RequireRunning -and $RequireStopped) {
        throw "Status cannot be required to be both running and stopped"
    }
    $running = ConvertTo-RestartBoolean (Get-RestartValue $Status @("running"))
    if ($RequireRunning -and -not $running) {
        throw "Trading engine is not running"
    }
    if ($RequireStopped -and $running) {
        throw "Trading engine is still running"
    }

    $lastError = [string](Get-RestartValue $Status @("last_error", "lastError") "")
    if (-not $AllowLastError -and -not [string]::IsNullOrWhiteSpace($lastError)) {
        throw "Trading engine reports last_error: $lastError"
    }

    $connection = Get-RestartValue $Status @("connection")
    if ($null -eq $connection) {
        throw "Dashboard status has no connection object"
    }
    $marketHealthy = ConvertTo-RestartBoolean (
        Get-RestartValue $connection @("market", "market_connected", "marketConnected")
    )
    $privateHealthy = ConvertTo-RestartBoolean (
        Get-RestartValue $connection @(
            "private", "private_available", "privateAvailable", "private_connected", "privateConnected"
        )
    )
    if (-not $marketHealthy) {
        throw "Market connection is not healthy"
    }
    if (-not $privateHealthy) {
        throw "Private connection is not healthy"
    }
    if (-not (Test-RestartOptionalFalse $connection @("private_stale", "privateStale", "stale"))) {
        throw "Private account snapshot is stale"
    }
    $market = Get-RestartValue $Status @("market")
    if (-not (Test-RestartOptionalFalse $market @("stale", "market_stale", "marketStale"))) {
        throw "Market snapshot is stale"
    }

    $privateStream = Get-RestartValue $connection @("private_stream", "privateStream")
    if ($null -ne $privateStream) {
        foreach ($field in @("available", "connected", "ready", "healthy")) {
            $missing = New-Object object
            $value = Get-RestartValue $privateStream @($field) $missing
            if (-not [object]::ReferenceEquals($missing, $value) -and -not (ConvertTo-RestartBoolean $value)) {
                throw "Private stream reports $field=false"
            }
        }
        $streamError = [string](Get-RestartValue $privateStream @("last_error", "lastError") "")
        if (-not [string]::IsNullOrWhiteSpace($streamError)) {
            throw "Private stream reports last_error: $streamError"
        }
    }

    $privateError = [string](Get-RestartValue $connection @("private_error", "privateError") "")
    if (-not [string]::IsNullOrWhiteSpace($privateError)) {
        throw "Private connection reports an error: $privateError"
    }

    if ($RequireFirstCycle) {
        $startedAt = ConvertTo-RestartDecimal (
            Get-RestartValue $Status @("started_at", "startedAt")
        ) "started_at"
        $lastCycleAt = ConvertTo-RestartDecimal (
            Get-RestartValue $Status @("last_cycle_at", "lastCycleAt")
        ) "last_cycle_at"
        if ($startedAt -le 0 -or $lastCycleAt -lt $startedAt) {
            throw "Trading engine has not completed its first cycle"
        }
    }
}

function Assert-SafeRestartPreflightStatus {
    param(
        [Parameter(Mandatory = $true)]
        $Status
    )

    $running = ConvertTo-RestartBoolean (Get-RestartValue $Status @("running"))
    if ($running) {
        Assert-RestartStatusHealth -Status $Status -RequireRunning
    } else {
        # DashboardService.stop() preserves the previous cycle error.  A
        # stopped dashboard can still provide a safe, current private snapshot
        # even when last_error describes the cycle that stopped the engine.
        Assert-RestartStatusHealth -Status $Status -RequireStopped -AllowLastError
    }
}

function Get-SafeRestartSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Status
    )

    $positions = @(
        @(Get-RestartValue $Status @("positions") @()) |
            Where-Object { $null -ne $_ }
    )
    $orders = @(
        @(Get-RestartValue $Status @("open_orders", "openOrders") @()) |
            Where-Object { $null -ne $_ }
    )
    if ($positions.Count -gt 1) {
        throw "Safe restart refused: expected at most one position, found $($positions.Count)"
    }

    $symbol = [string](Get-RestartValue $Status @("symbol") "")
    if ([string]::IsNullOrWhiteSpace($symbol)) {
        throw "Safe restart refused: dashboard status has no symbol"
    }
    $symbol = $symbol.Trim().ToUpperInvariant()
    $identity = [ordered]@{
        Exchange = ([string](Get-RestartValue $Status @("exchange") "")).Trim().ToLowerInvariant()
        Environment = ([string](Get-RestartValue $Status @("environment") "")).Trim().ToLowerInvariant()
        Mode = ([string](Get-RestartValue $Status @("mode") "")).Trim().ToLowerInvariant()
        Symbol = $symbol
    }
    if ($identity.Exchange -ne "binance" -or $identity.Mode -ne "live") {
        throw "Safe restart checks require Binance live mode; got exchange=$($identity.Exchange), mode=$($identity.Mode)"
    }

    if ($positions.Count -eq 0) {
        if ($orders.Count -ne 0) {
            throw "Safe restart refused: account is flat but has $($orders.Count) open order(s)"
        }
        return [pscustomobject]@{
            HadPosition = $false
            Identity = [pscustomobject]$identity
            Position = $null
            Stop = $null
            PositionCount = 0
            OpenOrderCount = 0
            QuantityStep = $null
            PriceTick = $null
        }
    }

    if ($orders.Count -ne 1) {
        throw "Safe restart refused: protected position requires exactly one open order, found $($orders.Count)"
    }

    $position = $positions[0]
    $positionSymbol = ([string](Get-RestartValue $position @("symbol") $symbol)).Trim().ToUpperInvariant()
    if ($positionSymbol -ne $symbol) {
        throw "Safe restart refused: position symbol $positionSymbol does not match status symbol $symbol"
    }
    $side = ([string](Get-RestartValue $position @("side") "")).Trim().ToLowerInvariant()
    if ($side -notin @("long", "short")) {
        throw "Safe restart refused: unsupported position side '$side'"
    }
    $quantity = ConvertTo-RestartDecimal (
        Get-RestartValue $position @("quantity", "position_quantity", "positionQuantity")
    ) "position.quantity"
    $entryPrice = ConvertTo-RestartDecimal (
        Get-RestartValue $position @("entry_price", "entryPrice")
    ) "position.entry_price"
    if ($quantity -le 0 -or $entryPrice -le 0) {
        throw "Safe restart refused: position quantity and entry price must be positive"
    }

    $quantityStep = Get-RestartQuantityStep $Status
    $priceTick = Get-RestartPriceTick $Status $symbol
    $normalizedQuantity = ConvertTo-RestartStepValue $quantity $quantityStep "Down"
    if ($normalizedQuantity -ne $quantity) {
        throw "Safe restart refused: position quantity $quantity is not aligned to step $quantityStep"
    }

    $stop = $orders[0]
    $orderType = ([string](Get-RestartValue $stop @("type", "order_type", "orderType") "")).Trim().ToUpperInvariant().Replace("-", "_")
    if ($orderType -ne "STOP_MARKET") {
        throw "Safe restart refused: sole open order is $orderType, not STOP_MARKET"
    }
    $orderStatus = ([string](Get-RestartValue $stop @("status", "algo_status", "algoStatus") "")).Trim().ToUpperInvariant()
    if ($orderStatus -notin @("NEW", "WORKING")) {
        throw "Safe restart refused: STOP_MARKET is not active (status=$orderStatus)"
    }
    $orderSide = ([string](Get-RestartValue $stop @("side") "")).Trim().ToUpperInvariant()
    $expectedOrderSide = if ($side -eq "long") { "SELL" } else { "BUY" }
    if ($orderSide -ne $expectedOrderSide) {
        throw "Safe restart refused: STOP_MARKET side $orderSide does not protect a $side position"
    }

    $stopOrderId = ([string](Get-RestartValue $stop @(
        "order_id", "orderId", "algo_id", "algoId"
    ) "")).Trim()
    $stopClientId = ([string](Get-RestartValue $stop @(
        "client_order_id", "clientOrderId", "client_algo_id", "clientAlgoId"
    ) "")).Trim()
    if ([string]::IsNullOrWhiteSpace($stopOrderId)) {
        throw "Safe restart refused: STOP_MARKET has no order id"
    }
    if (-not $stopClientId.StartsWith("btcbot-stop-", [StringComparison]::Ordinal)) {
        throw "Safe restart refused: STOP_MARKET is not identified as a bot-owned stop"
    }

    $reduceOnly = ConvertTo-RestartBoolean (
        Get-RestartValue $stop @("reduce_only", "reduceOnly") $false
    )
    $closePosition = ConvertTo-RestartBoolean (
        Get-RestartValue $stop @("close_position", "closePosition") $false
    )
    if (-not ($reduceOnly -or $closePosition)) {
        throw "Safe restart refused: STOP_MARKET is neither reduceOnly nor closePosition"
    }

    $stopPrice = ConvertTo-RestartDecimal (
        Get-RestartValue $stop @("stop_price", "stopPrice", "trigger_price", "triggerPrice")
    ) "stop.stop_price"
    if ($stopPrice -le 0) {
        throw "Safe restart refused: STOP_MARKET trigger must be positive"
    }
    if (($side -eq "long" -and $stopPrice -ge $entryPrice) -or
        ($side -eq "short" -and $stopPrice -le $entryPrice)) {
        throw "Safe restart refused: STOP_MARKET trigger is not protective"
    }
    $priceDirection = if ($orderSide -eq "SELL") { "Down" } else { "Up" }
    $normalizedStopPrice = ConvertTo-RestartStepValue $stopPrice $priceTick $priceDirection
    if ($normalizedStopPrice -ne $stopPrice) {
        throw "Safe restart refused: STOP_MARKET trigger $stopPrice is not aligned to tick $priceTick"
    }

    $orderQuantityRaw = Get-RestartValue $stop @("quantity", "orig_qty", "origQty") 0
    $orderQuantity = ConvertTo-RestartDecimal $orderQuantityRaw "stop.quantity"
    if ($orderQuantity -gt 0) {
        $normalizedOrderQuantity = ConvertTo-RestartStepValue $orderQuantity $quantityStep "Down"
        if ($normalizedOrderQuantity -ne $normalizedQuantity) {
            throw "Safe restart refused: STOP_MARKET quantity does not match the position"
        }
    }

    $localStopRaw = Get-RestartValue $position @("stop_price", "stopPrice")
    if ($null -ne $localStopRaw -and -not [string]::IsNullOrWhiteSpace([string]$localStopRaw)) {
        $localStop = ConvertTo-RestartDecimal $localStopRaw "position.stop_price"
        $normalizedLocalStop = ConvertTo-RestartStepValue $localStop $priceTick $priceDirection
        if ($normalizedLocalStop -ne $normalizedStopPrice) {
            throw "Safe restart refused: local stop does not match the exchange STOP_MARKET at tick precision"
        }
    }

    return [pscustomobject]@{
        HadPosition = $true
        Identity = [pscustomobject]$identity
        Position = [pscustomobject]@{
            Side = $side
            Quantity = $quantity
            NormalizedQuantity = $normalizedQuantity
            EntryPrice = $entryPrice
        }
        Stop = [pscustomobject]@{
            OrderId = $stopOrderId
            ClientId = $stopClientId
            Side = $orderSide
            Type = $orderType
            Status = $orderStatus
            StopPrice = $stopPrice
            NormalizedStopPrice = $normalizedStopPrice
            ReduceOnly = $reduceOnly
            ClosePosition = $closePosition
        }
        PositionCount = 1
        OpenOrderCount = 1
        QuantityStep = $quantityStep
        PriceTick = $priceTick
    }
}

function Assert-SafeRestartSnapshotUnchanged {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Before,
        [Parameter(Mandatory = $true)]
        [object]$After
    )

    foreach ($field in @("Exchange", "Environment", "Mode", "Symbol")) {
        $beforeValue = [string](Get-RestartValue $Before.Identity @($field) "")
        $afterValue = [string](Get-RestartValue $After.Identity @($field) "")
        if (-not $beforeValue.Equals($afterValue, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Safe restart validation failed: $field changed from '$beforeValue' to '$afterValue'"
        }
    }

    if ([bool]$Before.HadPosition -ne [bool]$After.HadPosition) {
        $transition = if ($Before.HadPosition) { "position became flat" } else { "unexpected position appeared" }
        throw "Safe restart validation failed: $transition during restart; engine must remain stopped"
    }
    if (-not $Before.HadPosition) {
        if ([int]$After.PositionCount -ne 0 -or [int]$After.OpenOrderCount -ne 0) {
            throw "Safe restart validation failed: flat baseline gained a position or order"
        }
        return
    }

    if ([decimal]$Before.QuantityStep -ne [decimal]$After.QuantityStep) {
        throw "Safe restart validation failed: exchange quantity step changed"
    }
    if ([decimal]$Before.PriceTick -ne [decimal]$After.PriceTick) {
        throw "Safe restart validation failed: exchange price tick changed"
    }
    if (-not ([string]$Before.Position.Side).Equals([string]$After.Position.Side, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Safe restart validation failed: position side changed"
    }
    if ([decimal]$Before.Position.NormalizedQuantity -ne [decimal]$After.Position.NormalizedQuantity) {
        throw "Safe restart validation failed: position quantity changed"
    }
    $entryDifference = [Math]::Abs(
        [decimal]$Before.Position.EntryPrice - [decimal]$After.Position.EntryPrice
    )
    if ($entryDifference -gt ([decimal]$Before.PriceTick / 2)) {
        throw "Safe restart validation failed: position entry price changed"
    }

    foreach ($field in @("OrderId", "ClientId", "Side", "Type")) {
        $beforeValue = [string](Get-RestartValue $Before.Stop @($field) "")
        $afterValue = [string](Get-RestartValue $After.Stop @($field) "")
        if (-not $beforeValue.Equals($afterValue, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Safe restart validation failed: protective stop $field changed"
        }
    }
    if ([decimal]$Before.Stop.NormalizedStopPrice -ne [decimal]$After.Stop.NormalizedStopPrice) {
        throw "Safe restart validation failed: protective stop trigger changed"
    }
    if (-not ([bool]$After.Stop.ReduceOnly -or [bool]$After.Stop.ClosePosition)) {
        throw "Safe restart validation failed: protective stop lost its close-only flag"
    }
    if ([int]$After.PositionCount -ne 1 -or [int]$After.OpenOrderCount -ne 1) {
        throw "Safe restart validation failed: position or protective stop was duplicated"
    }
}

function Format-SafeRestartSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Snapshot
    )

    if (-not $Snapshot.HadPosition) {
        return "flat; positions=0; open_orders=0; symbol=$($Snapshot.Identity.Symbol)"
    }
    return (
        "side={0}; quantity={1}; entry={2}; stop_order_id={3}; stop_client_id={4}; " +
        "stop_price={5}; stop_side={6}; reduce_only={7}; close_position={8}; positions=1; open_orders=1"
    ) -f @(
        $Snapshot.Position.Side,
        $Snapshot.Position.Quantity,
        $Snapshot.Position.EntryPrice,
        $Snapshot.Stop.OrderId,
        $Snapshot.Stop.ClientId,
        $Snapshot.Stop.StopPrice,
        $Snapshot.Stop.Side,
        $Snapshot.Stop.ReduceOnly,
        $Snapshot.Stop.ClosePosition
    )
}
