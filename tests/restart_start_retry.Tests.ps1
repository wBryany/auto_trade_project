$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
. (Join-Path $repoRoot "scripts\restart_bot_checks.ps1")

Describe "Get-RestartStartRetryPlan" {
    It "accepts a typed local deferral and waits beyond its deadline" {
        $payload = @{ retry = @{ api_code = "LOCAL_REQUEST_BUDGET"; retry_at = 1005 } }
        $plan = Get-RestartStartRetryPlan $payload -NowTimestamp 1000 -DeadlineTimestamp 1090
        if ($null -eq $plan -or $plan.wait_seconds -ne 5.25) { throw "Incorrect local wait plan" }
    }

    It "accepts HTTP 429 but never infers it from human readable text" {
        $payload = @{ retry = @{ upstream_status = 429; api_code = -1003; retry_at = 1003 } }
        $plan = Get-RestartStartRetryPlan $payload -NowTimestamp 1000 -DeadlineTimestamp 1090
        if ($null -eq $plan -or $plan.wait_seconds -ne 3.25) { throw "Incorrect venue wait plan" }
        $untyped = @{ error = "HTTP 429 retry in 3 seconds" }
        if ($null -ne (Get-RestartStartRetryPlan $untyped -NowTimestamp 1000 -DeadlineTimestamp 1090)) {
            throw "An untyped failure must not be replayed"
        }
    }

    It "accepts an already elapsed deadline without an immediate tight retry" {
        $payload = @{ retry = @{ upstream_status = 429; retry_at = 999 } }
        $plan = Get-RestartStartRetryPlan $payload -NowTimestamp 1000 -DeadlineTimestamp 1090
        if ($null -eq $plan -or $plan.wait_seconds -ne 0.25) { throw "Elapsed deadline must yield once" }
    }

    It "rejects bans, ambiguous network errors and unrelated server errors" {
        $payloads = @(
            @{ retry = @{ upstream_status = 418; api_code = "LOCAL_REQUEST_BUDGET"; retry_at = 1003 } },
            @{ retry = @{ upstream_status = 500; retry_at = 1003 } },
            @{ error = "network timeout" },
            @{ retry = @{ api_code = "unrelated"; retry_at = 1003 } }
        )
        foreach ($payload in $payloads) {
            if ($null -ne (Get-RestartStartRetryPlan $payload -NowTimestamp 1000 -DeadlineTimestamp 1090)) {
                throw "Unsafe start retry accepted"
            }
        }
    }

    It "rejects missing, invalid or unbounded deadlines" {
        foreach ($deadline in @($null, "NaN", "Infinity", -1, 0, 1090, 99999)) {
            $payload = @{ retry = @{ upstream_status = 429; retry_at = $deadline } }
            if ($null -ne (Get-RestartStartRetryPlan $payload -NowTimestamp 1000 -DeadlineTimestamp 1090)) {
                throw "Invalid deadline accepted: $deadline"
            }
        }
    }

    It "permits no more than two automatic retries" {
        $payload = @{ retry = @{ upstream_status = 429; retry_at = 1003 } }
        $plan = Get-RestartStartRetryPlan $payload -NowTimestamp 1000 -DeadlineTimestamp 1090 -RetryCount 1
        if ($null -eq $plan) { throw "Second retry incorrectly rejected" }
        if ($null -ne (Get-RestartStartRetryPlan $payload -NowTimestamp 1000 -DeadlineTimestamp 1090 -RetryCount 2)) {
            throw "Third retry must not be accepted"
        }
    }
}
