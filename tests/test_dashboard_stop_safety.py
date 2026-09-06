from __future__ import annotations

import copy
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from btc_futures_bot.dashboard import DASHBOARD_HTML, DashboardService
from btc_futures_bot.models import Position
from btc_futures_bot.risk import RiskManager


class ControlledThread:
    def __init__(self, *, alive: bool, after_join=None) -> None:
        self.alive = alive
        self.after_join = after_join
        self.join_timeouts: list[float] = []

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float) -> None:
        self.join_timeouts.append(timeout)
        if self.after_join is not None:
            self.after_join()

    def start(self) -> None:
        self.alive = True


def _position() -> Position:
    return Position("long", 1, 100, 99, 102, 1_000_000, initial_stop_price=99)


def _service() -> DashboardService:
    service = DashboardService.__new__(DashboardService)
    config = {
        "mode": "paper", "active_exchange": "okx", "paper_equity": 10000,
        "exchanges": {"okx": {"enabled": True, "environment": "demo", "symbol": "BTC-USDT-SWAP"}},
    }
    service._config = Mock(return_value=config)
    service._engine_config_snapshot = copy.deepcopy(config)
    service.exchange_name = "okx"
    service._lock = threading.RLock()
    service._stopping = False
    service._thread = ControlledThread(alive=False)
    service._stop_event = threading.Event()
    service.engine = SimpleNamespace(
        config=SimpleNamespace(mode="paper"), adapter=SimpleNamespace(name="okx"),
        position=None, close=Mock(), session_pnl=-2.0,
        risk=RiskManager(), consecutive_losses=1, cooldown_until=1234.0,
        macro_risk=None, entry_gate=None,
    )
    service.reporter = Mock()
    service.operation_logger = Mock()
    service.notifier = Mock()
    service.notifier.status.return_value = {}
    service._dashboard_adapter = Mock()
    service._dashboard_adapter_key = None
    service.last_error = ""
    service.last_result = None
    service.last_cycle_at = 0.0
    service.started_at = 1.0
    service._market_snapshot = Mock(return_value={"market": {"mark_price": 100.1}})
    service._with_live_market = lambda _config, _exchange, snapshot: snapshot
    return service


def test_stop_timeout_preserves_running_thread_engine_and_reporter() -> None:
    service = _service()
    service._thread = ControlledThread(alive=True)
    engine, reporter, thread, event = service.engine, service.reporter, service._thread, service._stop_event

    with pytest.raises(RuntimeError, match="停止未完成"):
        service.stop()

    assert service.running
    assert service.engine is engine and service.reporter is reporter
    assert service._thread is thread and service._stop_event is event
    assert event.is_set()
    assert thread.join_timeouts == [3]
    engine.close.assert_not_called()
    reporter.close.assert_not_called()
    assert service.status()["stop_pending"] is True
    with pytest.raises(RuntimeError, match="停止中"):
        service.start({})


def test_stop_retains_position_opened_by_final_inflight_cycle_and_exposes_it() -> None:
    service = _service()
    engine, reporter = service.engine, service.reporter

    def finish_cycle() -> None:
        engine.position = _position()
        service._thread.alive = False

    service._thread = ControlledThread(alive=True, after_join=finish_cycle)
    result = service.stop()

    assert result == {"running": False, "paper_position_retained": True}
    assert service.engine is engine and service.reporter is reporter
    assert not service.running
    engine.close.assert_not_called()
    reporter.close.assert_not_called()
    status = service.status()
    assert status["paper_position_retained"] is True
    assert len(status["positions"]) == 1
    assert status["positions"][0]["entry_price"] == 100


def test_stopped_paper_position_resumes_exact_engine_and_risk_state() -> None:
    service = _service()
    engine, reporter = service.engine, service.reporter
    engine.position = _position()
    original_position = engine.position
    service.stop()
    resumed_thread = ControlledThread(alive=False)

    with patch("btc_futures_bot.dashboard.threading.Thread", return_value=resumed_thread), patch(
        "btc_futures_bot.dashboard.build_engine"
    ) as build:
        result = service.start({"exchange": "okx"})

    assert result["resumed"] is True and result["running"] is True
    assert service.engine is engine and service.reporter is reporter
    assert engine.position is original_position
    assert engine.session_pnl == -2 and engine.consecutive_losses == 1 and engine.cooldown_until == 1234
    assert service._stop_event is not None and not service._stop_event.is_set()
    build.assert_not_called()


def test_retained_paper_position_blocks_config_write_and_changed_config_restart() -> None:
    service = _service()
    service.engine.position = _position()
    service.stop()
    with patch("btc_futures_bot.dashboard.save_dashboard_config") as save:
        with pytest.raises(RuntimeError, match="仍保留模拟仓位"):
            service.save_config({"mode": "live"})
        save.assert_not_called()

    service._config.return_value["mode"] = "live"
    with patch("btc_futures_bot.dashboard.build_engine") as build:
        with pytest.raises(RuntimeError, match="禁止覆盖仓位"):
            service.restart({})
        build.assert_not_called()
    assert service.engine.position is not None


def test_restart_resumes_paper_position_without_rebuilding() -> None:
    service = _service()
    service.engine.position = _position()
    engine = service.engine
    resumed_thread = ControlledThread(alive=False)
    with patch("btc_futures_bot.dashboard.threading.Thread", return_value=resumed_thread), patch(
        "btc_futures_bot.dashboard.build_engine"
    ) as build:
        result = service.restart({})
    assert result["resumed"] is True
    assert service.engine is engine
    build.assert_not_called()


def test_flat_stop_cleans_resources_only_after_join_finishes() -> None:
    service = _service()
    engine, reporter = service.engine, service.reporter

    def finish_cycle() -> None:
        assert service.engine is engine and service.reporter is reporter
        engine.close.assert_not_called()
        reporter.close.assert_not_called()
        service._thread.alive = False

    service._thread = ControlledThread(alive=True, after_join=finish_cycle)
    assert service.stop() == {"running": False}
    engine.close.assert_called_once_with()
    reporter.close.assert_called_once_with()
    assert service.engine is None and service.reporter is None and service._thread is None
    assert service._stop_event is None and service._engine_config_snapshot is None


def test_concurrent_start_is_blocked_while_stop_joins() -> None:
    service = _service()

    def finish_cycle() -> None:
        with pytest.raises(RuntimeError, match="停止中"):
            service.start({})
        service._thread.alive = False

    service._thread = ControlledThread(alive=True, after_join=finish_cycle)
    service.stop()


def test_shutdown_timeout_does_not_close_shared_resources() -> None:
    service = _service()
    service._thread = ControlledThread(alive=True)
    adapter = service._dashboard_adapter
    with pytest.raises(RuntimeError, match="停止未完成"):
        service.shutdown()
    adapter.close.assert_not_called()
    service.notifier.close.assert_not_called()
    service.engine.close.assert_not_called()


def test_live_stop_does_not_submit_position_close_order() -> None:
    service = _service()
    engine = service.engine
    engine.config.mode = "live"
    engine.position = _position()
    engine.adapter.place_market_order = Mock()
    service.stop()
    engine.close.assert_called_once_with()
    engine.adapter.place_market_order.assert_not_called()


def test_dashboard_explains_shadow_scoring_and_preserved_positions() -> None:
    assert "旁路评分，不拦截开仓" in DASHBOARD_HTML
    assert "参与开仓过滤" in DASHBOARD_HTML
    assert "仅评分，主策略仍可开仓" in DASHBOARD_HTML
    assert "评分=${score}" in DASHBOARD_HTML
    assert "模拟仓位保留在当前进程" in DASHBOARD_HTML
