from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from btc_futures_bot.dashboard import DashboardHandler, DashboardService
from btc_futures_bot.http_client import ApiError


@pytest.mark.parametrize(
    ("error", "status", "code", "local"),
    [
        (ApiError("deferred", api_code="LOCAL_REQUEST_BUDGET", retry_at=2000), None, "LOCAL_REQUEST_BUDGET", True),
        (ApiError("limited", status_code=429, api_code=-1003, retry_at=2000), 429, -1003, False),
        (ApiError("banned", status_code=418, api_code=-1003, retry_at=2000), 418, -1003, False),
    ],
)
def test_start_response_preserves_typed_retry_metadata(error, status, code, local) -> None:
    service = SimpleNamespace(start=Mock(side_effect=error), operation_logger=Mock())
    handler = SimpleNamespace(
        path="/api/start",
        _body=lambda: {"exchange": "binance"},
        _json=Mock(),
        server=SimpleNamespace(service=service),
    )
    DashboardHandler.do_POST(handler)
    body, response_status = handler._json.call_args.args
    assert response_status == 400
    assert body["error"] == str(error)
    assert body["retry"] == {
        "api_code": code,
        "upstream_status": status,
        "retry_at": 2000,
        "retry_after_seconds": 0,
        "local_deferred": local,
    }
    service.start.assert_called_once_with({"exchange": "binance"})


def test_ambiguous_start_timeout_has_no_retry_metadata() -> None:
    service = SimpleNamespace(start=Mock(side_effect=TimeoutError("timed out")), operation_logger=Mock())
    handler = SimpleNamespace(
        path="/api/start",
        _body=lambda: {},
        _json=Mock(),
        server=SimpleNamespace(service=service),
    )
    DashboardHandler.do_POST(handler)
    assert handler._json.call_args.args == ({"error": "timed out"}, 400)


@pytest.mark.parametrize("restoring", [False, True])
def test_dashboard_alert_includes_active_or_saved_exposure(restoring: bool) -> None:
    notifier = SimpleNamespace(notify_emergency=Mock(return_value=True))
    service = SimpleNamespace(
        notifier=notifier,
        engine=SimpleNamespace(
            position=None if restoring else {"side": "long"},
            _managed_live_position_state={"position": {"side": "long"}} if restoring else None,
        ),
    )
    assert DashboardService._notify_emergency(
        service,
        ApiError("deferred", api_code="LOCAL_REQUEST_BUDGET", retry_at=2000),
        category="engine_runtime", context="startup", incident="start",
        config={"mode": "live", "exchanges": {"binance": {"symbol": "BTCUSDT"}}},
        exchange_name="binance",
    )
    assert notifier.notify_emergency.call_args.kwargs["details"]["当前本地仓位"] == "有"
