from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from btc_futures_bot.dashboard import DashboardService
from btc_futures_bot.notifications import EmailNotificationConfig, EmailNotifier


CONFIG = {
    "mode": "live",
    "exchanges": {"binance": {
        "environment": "production", "symbol": "BTCUSDT", "base_url": "https://fapi.binance.com",
    }},
}


def _snapshot(*, recovered: bool = False, recovery_id: str = "renewal-one") -> dict:
    return {
        "private_available": True,
        "private_source": "websocket",
        "private_error": "",
        "order_limits": {},
        "private_stream": {
            "healthy": True,
            "ready": True,
            "last_error": "",
            "retry_at": 0.0,
            "rest_recovery_confirmed": recovered,
            "rest_recovery_id": recovery_id if recovered else "",
        },
    }


def _limited_snapshot(now: datetime) -> dict:
    snapshot = _snapshot()
    snapshot["private_stream"]["last_error"] = "HTTP 429: too many requests"
    snapshot["private_stream"]["retry_at"] = now.timestamp() + 5
    return snapshot


@pytest.mark.parametrize("reappear_after_seconds", [1, 31 * 60])
def test_rest_recovery_resolves_real_notifier_incident_and_new_failure_alerts(
    tmp_path: Path, monkeypatch, reappear_after_seconds: int
) -> None:
    clock = {"now": datetime(2026, 9, 9, 22, 38, 53, tzinfo=ZoneInfo("Asia/Shanghai"))}
    messages = []
    notifier = EmailNotifier(
        EmailNotificationConfig(
            enabled=True, smtp_host="smtp.example.test", sender="sender@example.test",
            recipients=("owner@example.test",), state_path=str(tmp_path / "mail_state.json"),
        ),
        send_fn=messages.append,
        now_fn=lambda: clock["now"],
    )
    service = DashboardService.__new__(DashboardService)
    service.notifier = notifier
    monkeypatch.setattr("btc_futures_bot.dashboard.time.time", lambda: clock["now"].timestamp())
    try:
        failure = _limited_snapshot(clock["now"])
        service._handle_snapshot_alerts(failure, CONFIG, "binance")
        assert notifier.flush()
        service._handle_snapshot_alerts(failure, CONFIG, "binance")
        assert notifier.flush()
        assert len(messages) == 1
        assert notifier.status()["emergency_incidents_in_cooldown"] == 1

        clock["now"] += timedelta(seconds=61)
        recovered = _snapshot(recovered=True)
        service._handle_snapshot_alerts(recovered, CONFIG, "binance")
        assert notifier.status()["emergency_incidents_in_cooldown"] == 0

        # One case catches an incorrectly retained cooldown; the other catches
        # the historical-error reminder after the former 30-minute expiry.
        clock["now"] += timedelta(seconds=reappear_after_seconds)
        service._handle_snapshot_alerts(recovered, CONFIG, "binance")
        assert notifier.flush()
        assert len(messages) == 1
        service._handle_snapshot_alerts(_limited_snapshot(clock["now"]), CONFIG, "binance")
        assert notifier.flush()
        assert len(messages) == 2
        assert all("Binance IP 限频/封禁" in str(message["Subject"]) for message in messages)
    finally:
        notifier.close()


@pytest.mark.parametrize("evidence", [None, False, "true"])
def test_healthy_websocket_without_rest_recovery_evidence_does_not_resolve(evidence) -> None:
    service = DashboardService.__new__(DashboardService)
    service.notifier = Mock()
    snapshot = _snapshot()
    snapshot["private_stream"]["rest_recovery_confirmed"] = evidence
    service._handle_snapshot_alerts(snapshot, CONFIG, "binance")
    service.notifier.resolve_emergency.assert_not_called()
    service.notifier.notify_emergency.assert_not_called()


@pytest.mark.parametrize("field", ["snapshot_error", "private_error", "order_limits", "private_stream"])
@pytest.mark.parametrize("error", ["HTTP 429: too many requests", "temporary TLS timeout"])
@pytest.mark.parametrize("source", ["websocket", "rest"])
def test_recovery_evidence_never_masks_current_snapshot_errors(field: str, error: str, source: str) -> None:
    service = DashboardService.__new__(DashboardService)
    service.notifier = Mock()
    snapshot = _snapshot(recovered=True)
    snapshot["private_source"] = source
    if field == "order_limits":
        snapshot[field]["error"] = error
    elif field == "private_stream":
        snapshot[field]["last_error"] = error
    else:
        snapshot[field] = error
    service._handle_snapshot_alerts(snapshot, CONFIG, "binance")
    service.notifier.resolve_emergency.assert_not_called()
    if "429" in error:
        service.notifier.notify_emergency.assert_called_once()
    else:
        service.notifier.notify_emergency.assert_not_called()


@pytest.mark.parametrize("condition", ["stale", "unavailable", "unhealthy", "not_ready", "retry_future", "retry_invalid"])
def test_incomplete_or_stale_recovery_does_not_resolve(condition: str, monkeypatch) -> None:
    service = DashboardService.__new__(DashboardService)
    service.notifier = Mock()
    monkeypatch.setattr("btc_futures_bot.dashboard.time.time", lambda: 1000.0)
    snapshot = _snapshot(recovered=True)
    if condition == "stale":
        snapshot["private_stale"] = True
    elif condition == "unavailable":
        snapshot["private_available"] = False
    elif condition == "unhealthy":
        snapshot["private_stream"]["healthy"] = False
    elif condition == "not_ready":
        snapshot["private_stream"]["ready"] = False
    elif condition == "retry_future":
        snapshot["private_stream"]["retry_at"] = 1001.0
    else:
        snapshot["private_stream"]["retry_at"] = float("inf")
    service._handle_snapshot_alerts(snapshot, CONFIG, "binance")
    service.notifier.resolve_emergency.assert_not_called()


def test_healthy_rest_snapshot_retains_recovery_behavior() -> None:
    service = DashboardService.__new__(DashboardService)
    service.notifier = Mock()
    snapshot = _snapshot()
    snapshot["private_source"] = "rest"
    service._handle_snapshot_alerts(snapshot, CONFIG, "binance")
    service.notifier.resolve_emergency.assert_called_once_with("ip_restricted", "binance", "snapshot")


def test_websocket_recovery_receipt_is_consumed_only_once() -> None:
    service = DashboardService.__new__(DashboardService)
    service.notifier = Mock()
    recovered = _snapshot(recovered=True)
    for _ in range(3):
        service._handle_snapshot_alerts(recovered, CONFIG, "binance")
    service.notifier.resolve_emergency.assert_called_once_with("ip_restricted", "binance", "snapshot")
    service._handle_snapshot_alerts(_snapshot(recovered=True, recovery_id="renewal-two"), CONFIG, "binance")
    assert service.notifier.resolve_emergency.call_count == 2
    assert len(service._snapshot_rest_recovery_ids) == 1


@pytest.mark.parametrize("error", ["HTTP 429: too many requests", "TLS error"])
def test_new_error_consumes_old_receipt_and_requires_new_rest_recovery(error: str) -> None:
    service = DashboardService.__new__(DashboardService)
    service.notifier = Mock()
    failed = _snapshot(recovered=True)
    failed["order_limits"]["error"] = error
    service._handle_snapshot_alerts(failed, CONFIG, "binance")
    service._handle_snapshot_alerts(_snapshot(recovered=True), CONFIG, "binance")
    service.notifier.resolve_emergency.assert_not_called()
    service._handle_snapshot_alerts(_snapshot(recovered=True, recovery_id="renewal-two"), CONFIG, "binance")
    service.notifier.resolve_emergency.assert_called_once()


def test_active_global_http_cooldown_consumes_old_recovery_receipt(monkeypatch) -> None:
    service = DashboardService.__new__(DashboardService)
    service.notifier = Mock()
    blocked = {"seconds": 5.0}
    monkeypatch.setattr("btc_futures_bot.dashboard.rate_limit_remaining", lambda _url: blocked["seconds"])
    service._handle_snapshot_alerts(_snapshot(recovered=True), CONFIG, "binance")
    service.notifier.resolve_emergency.assert_not_called()
    blocked["seconds"] = 0.0
    service._handle_snapshot_alerts(_snapshot(recovered=True), CONFIG, "binance")
    service.notifier.resolve_emergency.assert_not_called()
    service._handle_snapshot_alerts(_snapshot(recovered=True, recovery_id="renewal-two"), CONFIG, "binance")
    service.notifier.resolve_emergency.assert_called_once()


def test_websocket_recovery_requires_nonempty_receipt() -> None:
    service = DashboardService.__new__(DashboardService)
    service.notifier = Mock()
    service._handle_snapshot_alerts(_snapshot(recovered=True, recovery_id=""), CONFIG, "binance")
    service.notifier.resolve_emergency.assert_not_called()
