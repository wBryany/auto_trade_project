from __future__ import annotations

import time
from pathlib import Path

import pytest

from btc_futures_bot.http_client import ApiError, is_rate_limit_error
from btc_futures_bot.notifications import EmailNotificationConfig, EmailNotifier


@pytest.mark.parametrize("wrapped", [False, True])
def test_preventive_defer_does_not_send_or_suppress_real_ip_alert(
    tmp_path: Path, wrapped: bool
) -> None:
    messages = []
    notifier = EmailNotifier(
        EmailNotificationConfig(
            enabled=True,
            smtp_host="smtp.example.test",
            sender="sender@example.test",
            recipients=("recipient@example.test",),
            state_path=str(tmp_path / "email_state.json"),
        ),
        send_fn=messages.append,
    )
    local_error = ApiError(
        "Binance request budget deferred; no request sent",
        api_code="LOCAL_REQUEST_BUDGET",
        retry_at=time.time() + 30,
    )
    error: BaseException = local_error
    if wrapped:
        error = RuntimeError("private initialization deferred")
        error.__cause__ = local_error
    context = {
        "category": "engine_runtime",
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "mode": "live",
        "environment": "production",
        "context": "startup preflight",
        "incident": "start",
    }
    try:
        assert local_error.retry_after_seconds > 0
        assert not is_rate_limit_error(error)
        assert notifier.notify_emergency(error, **context) is False
        assert notifier.flush()
        assert messages == []
        assert notifier.status()["emergency_incidents_in_cooldown"] == 0

        real_error = ApiError(
            "HTTP 429: too many requests",
            status_code=429,
            api_code=-1003,
            retry_at=time.time() + 3,
        )
        # A real response takes precedence over older preventive deferrals
        # retained in Python's exception context.
        real_error.__context__ = local_error
        assert notifier.notify_emergency(real_error, **context)
        assert notifier.flush()
        assert len(messages) == 1
        assert "IP 限频/封禁" in str(messages[0]["Subject"])
        assert "HTTP 状态：429" in messages[0].get_content()
        assert "交易所错误码：-1003" in messages[0].get_content()
    finally:
        notifier.close()


def test_string_that_mentions_local_budget_is_not_silenced(tmp_path: Path) -> None:
    messages = []
    notifier = EmailNotifier(
        EmailNotificationConfig(
            enabled=True,
            smtp_host="smtp.example.test",
            sender="sender@example.test",
            recipients=("recipient@example.test",),
            state_path=str(tmp_path / "email_state.json"),
        ),
        send_fn=messages.append,
    )
    try:
        assert notifier.notify_emergency(
            RuntimeError("LOCAL_REQUEST_BUDGET handler unexpectedly failed"),
            category="engine_runtime",
            exchange="binance",
            symbol="BTCUSDT",
            mode="live",
            environment="production",
            context="engine cycle",
        )
        assert notifier.flush()
        assert len(messages) == 1
        assert "交易引擎报错" in str(messages[0]["Subject"])
    finally:
        notifier.close()


@pytest.mark.parametrize(
    ("category", "incident", "position", "subject"),
    [
        ("order_failure", "protective_stop", "无", "下单失败"),
        ("entry_reconciliation", "filled_entry", "有", "成交明细对账异常"),
        ("engine_runtime", "cycle", "有", "交易引擎报错"),
        ("engine_runtime", "cycle", None, "交易引擎报错"),
        ("engine_runtime", "start", "有", "交易引擎报错"),
        ("ip_restricted", "cycle", "有", "交易引擎报错"),
    ],
)
def test_local_budget_keeps_actionable_position_and_order_alerts(
    tmp_path: Path, category: str, incident: str, position: str | None, subject: str
) -> None:
    messages = []
    notifier = EmailNotifier(
        EmailNotificationConfig(
            enabled=True,
            smtp_host="smtp.example.test",
            sender="sender@example.test",
            recipients=("recipient@example.test",),
            state_path=str(tmp_path / "email_state.json"),
        ),
        send_fn=messages.append,
    )
    error = ApiError(
        "Binance REST deferred locally: preventive budget",
        api_code="LOCAL_REQUEST_BUDGET",
        retry_at=time.time() + 30,
    )
    try:
        assert notifier.notify_emergency(
            error, category=category, exchange="binance", symbol="BTCUSDT",
            mode="live", environment="production", context="operational check",
            incident=incident,
            details={"当前本地仓位": position} if position is not None else None,
        )
        assert notifier.flush()
        assert len(messages) == 1
        assert subject in str(messages[0]["Subject"])
        assert "IP 限频/封禁" not in str(messages[0]["Subject"])
        assert "HTTP 状态：429" not in messages[0].get_content()
    finally:
        notifier.close()
