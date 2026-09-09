from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from btc_futures_bot.dashboard import DashboardService
from btc_futures_bot.http_client import ApiError


def _config(symbol: str = "BTC-USDT-SWAP") -> dict:
    return {"mode": "paper", "active_exchange": "okx", "dashboard_snapshot_seconds": 15,
            "exchanges": {"okx": {"symbol": symbol, "environment": "demo", "base_url": "https://example.invalid"}}}


def _service() -> DashboardService:
    service = DashboardService.__new__(DashboardService)
    service._config = Mock(return_value=_config())
    service._lock = threading.RLock()
    service._snapshot_condition = threading.Condition(service._lock)
    service._snapshot_refreshing = False
    service._snapshot_generation = 0
    service._exchange_snapshot = {}
    service._snapshot_at = service._private_snapshot_at = 0.0
    service._dashboard_adapter = None
    service._dashboard_adapter_key = None
    service._handle_snapshot_alerts = Mock()
    service.engine = service.reporter = service.notifier = None
    service._thread = service._stop_event = None
    service._stopping = False
    service.exchange_name = "okx"
    service.last_result = None
    service.last_error = ""
    service.started_at = service.last_cycle_at = 0.0
    return service


def test_concurrent_status_adapter_creation_is_single_flight() -> None:
    service = _service()
    entered, release = threading.Event(), threading.Event()
    adapter = SimpleNamespace(close=Mock())

    def create(*_args):
        entered.set()
        assert release.wait(2)
        return adapter

    with patch("btc_futures_bot.dashboard.make_adapter", side_effect=create) as factory:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(service._adapter, _config(), "okx") for _ in range(6)]
            assert entered.wait(2)
            release.set()
            assert all(future.result(timeout=2) is adapter for future in futures)
    factory.assert_called_once()
    adapter.close.assert_not_called()


def test_changed_config_closes_owned_adapter_before_replacement() -> None:
    service = _service()
    timeline = []
    old = SimpleNamespace(close=Mock(side_effect=lambda: timeline.append("close")))
    new = SimpleNamespace(close=Mock())
    with patch("btc_futures_bot.dashboard.make_adapter", return_value=old):
        service._adapter(_config(), "okx")
    with patch("btc_futures_bot.dashboard.make_adapter", side_effect=lambda *_args: timeline.append("create") or new):
        assert service._adapter(_config("ETH-USDT-SWAP"), "okx") is new
    assert timeline == ["close", "create"]
    old.close.assert_called_once()


def test_engine_owned_adapter_is_reused_and_never_retired_by_dashboard() -> None:
    service = _service()
    engine_adapter = SimpleNamespace(name="okx", settings=SimpleNamespace(**_config()["exchanges"]["okx"]), close=Mock())
    service.engine = SimpleNamespace(adapter=engine_adapter)
    service._dashboard_adapter = engine_adapter
    with patch("btc_futures_bot.dashboard.make_adapter") as factory:
        assert service._adapter(_config(), "okx") is engine_adapter
        with service._lock:
            service._close_dashboard_adapter()
    factory.assert_not_called()
    engine_adapter.close.assert_not_called()


def test_save_config_closes_old_dashboard_stream_without_writing_real_config() -> None:
    service = _service()
    adapter = SimpleNamespace(close=Mock())
    service._dashboard_adapter = adapter
    service.config_path = "unused.json"
    service.config_view = Mock(return_value={"exchange": "okx"})
    service.notifier = Mock()
    service._build_notifier = Mock(return_value=Mock())
    service.operation_logger = Mock()
    with patch("btc_futures_bot.dashboard.save_dashboard_config", return_value=Path("unused.local.json")):
        service.save_config({"exchange": "okx"})
    adapter.close.assert_called_once()
    assert service._dashboard_adapter is None
    assert service._snapshot_generation == 1


def test_initial_snapshot_failure_is_cached_and_backed_off() -> None:
    service = _service()
    fetch = Mock(side_effect=ApiError("temporary TLS failure"))
    service._adapter = Mock(return_value=SimpleNamespace(fetch_dashboard_snapshot=fetch))
    with patch("btc_futures_bot.dashboard.time.time", return_value=100):
        failed = service._market_snapshot(_config(), "okx")
    with patch("btc_futures_bot.dashboard.time.time", return_value=101):
        assert service._market_snapshot(_config(), "okx") is failed
    assert failed["private_available"] is False
    assert failed["market"]["stale"] is True
    assert failed["positions"] == []
    assert "temporary TLS failure" in failed["snapshot_error"]
    fetch.assert_called_once()


def test_refresh_wait_timeout_cannot_start_a_second_owner() -> None:
    service = _service()
    service._snapshot_refreshing = True
    service._adapter = Mock()
    with patch.object(service._snapshot_condition, "wait", return_value=False):
        unavailable = service._market_snapshot(_config(), "okx")
    assert unavailable["private_available"] is False
    assert unavailable["market"]["stale"] is True
    assert service._snapshot_refreshing is True
    service._adapter.assert_not_called()


def test_old_inflight_config_snapshot_cannot_repopulate_new_config_cache() -> None:
    service = _service()

    def fetch():
        service._snapshot_generation += 1
        return {"private_available": True, "account": {"wallet_balance": 999}, "market": {"mark_price": 100}}

    service._adapter = Mock(return_value=SimpleNamespace(fetch_dashboard_snapshot=fetch))
    result = service._market_snapshot(_config(), "okx")
    assert result["private_available"] is False
    assert service._exchange_snapshot == {}
    assert service._snapshot_refreshing is False


def test_explicit_stream_staleness_is_not_laundered_by_live_overlay() -> None:
    service = _service()
    live = {"market": {"mark_price": 101, "stale": True}, "market_stream": {"connected": False},
            "private_available": True, "private_stale": True, "private_warning": "WS disconnected",
            "private_source": "websocket", "snapshot_error": "disconnected"}
    service._adapter = Mock(return_value=SimpleNamespace(fetch_live_dashboard_snapshot=lambda: live))
    result = service._with_live_market({}, "okx", {"market": {"mark_price": 100}})
    assert result["market"]["stale"] is True
    assert result["private_stale"] is True
    assert result["private_warning"] == "WS disconnected"
    assert result["snapshot_error"] == "disconnected"


def test_market_only_overlay_cannot_clear_private_error_or_staleness() -> None:
    service = _service()
    service._adapter = Mock(return_value=SimpleNamespace(fetch_live_dashboard_snapshot=lambda: {"market": {"mark_price": 101}}))
    cached = {"market": {"mark_price": 100, "stale": True}, "private_available": True,
              "private_stale": True, "private_error": "authentication lost", "private_warning": "cached"}
    result = service._with_live_market({}, "okx", cached)
    assert result["market"]["mark_price"] == 101
    assert result["private_stale"] is True
    assert result["private_error"] == "authentication lost"


def test_fresh_fallback_warning_is_kept_and_input_is_not_mutated() -> None:
    service = _service()
    live = {"market": {"mark_price": 101}, "private_available": True,
            "private_source": "rest", "private_warning": "REST fallback; WS reconnecting",
            "account": {"wallet_balance": 100}, "positions": [], "open_orders": []}
    service._adapter = Mock(return_value=SimpleNamespace(fetch_live_dashboard_snapshot=lambda: live))
    cached = {"market": {"mark_price": 100, "stale": True}, "private_stale": True, "private_error": "old"}
    before = copy.deepcopy(cached)
    result = service._with_live_market({}, "okx", cached)
    assert "private_stale" not in result and "private_error" not in result
    assert result["private_warning"] == live["private_warning"]
    assert cached == before


def test_stale_private_response_does_not_renew_last_healthy_timestamp() -> None:
    service = _service()
    service._private_snapshot_at = 90.0
    live = {"market": {"mark_price": 101}, "private_available": True, "private_stale": True}
    service._adapter = Mock(return_value=SimpleNamespace(fetch_dashboard_snapshot=lambda: live))
    with patch("btc_futures_bot.dashboard.time.monotonic", return_value=100.0):
        service._market_snapshot(_config(), "okx")
    assert service._private_snapshot_at == 90.0


@pytest.mark.parametrize("stale", [True, False])
def test_connection_health_and_transport_metadata_are_visible(stale: bool) -> None:
    service = _service()
    snapshot = {"market": {"mark_price": 100, "stale": stale, "price_source": "OKX WS"},
                "market_stream": {"connected": not stale}, "private_available": False}
    service._market_snapshot = Mock(return_value=snapshot)
    service._with_live_market = lambda _config, _exchange, value: value
    status = service.status()
    assert status["connection"]["market"] is not stale
    assert status["connection"]["market_source"] == "OKX WS"
    assert status["connection"]["market_stream"] == {"connected": not stale}
