from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from btc_futures_bot.dashboard import (
    DASHBOARD_HTML,
    DashboardService,
    _json_safe,
    _mark_to_market_view,
    _order_sizing_view,
    _position_dict,
)
from btc_futures_bot.models import Position


class _RecordingEmergencyNotifier:
    def __init__(self) -> None:
        self.notifications: list[dict[str, object]] = []
        self.resolutions: list[tuple[str, str, str]] = []

    def notify_emergency(
        self,
        error: object,
        *,
        category: str,
        exchange: str,
        symbol: str,
        mode: str,
        environment: str,
        context: str,
        incident: str = "",
        details: dict[str, object] | None = None,
    ) -> bool:
        self.notifications.append(
            {
                "error": error,
                "category": category,
                "exchange": exchange,
                "symbol": symbol,
                "mode": mode,
                "environment": environment,
                "context": context,
                "incident": incident,
                "details": details,
            }
        )
        return True

    def resolve_emergency(
        self,
        category: str,
        exchange: str,
        incident: str = "",
    ) -> None:
        self.resolutions.append((category, exchange, incident))


def test_json_safe_replaces_non_finite_values_recursively() -> None:
    payload = {
        "finite": 1.5,
        "nested": [float("inf"), float("-inf"), float("nan")],
    }

    cleaned = _json_safe(payload)

    assert cleaned == {"finite": 1.5, "nested": [None, None, None]}
    assert math.isfinite(cleaned["finite"])


def test_order_sizing_view_uses_share_of_exchange_theoretical_capacity() -> None:
    config = {"risk": {"risk_per_trade": 0.02, "max_notional_pct": 0.3}}
    snapshot = {
        "order_limits": {
            "effective_min_quantity_raw": "0.001",
            "effective_min_notional_raw": "77.5000",
            "min_notional_filter_raw": "50",
            "current_leverage_raw": "20",
            "estimated_max_open_quantity_raw": "0.007",
            "estimated_max_open_notional_raw": "542.5000",
        }
    }
    account = {"wallet_balance_raw": "27.39737452"}

    sizing = _order_sizing_view(config, snapshot, account)

    assert sizing["available"] is True
    assert sizing["risk_budget_raw"] == "0.5479474904"
    assert sizing["strategy_cap_basis_raw"] == "542.5000"
    assert sizing["strategy_notional_cap_raw"] == "162.75000"
    assert sizing["effective_strategy_cap_raw"] == "162.75000"
    assert sizing["strategy_cap_meets_minimum"] is True
    assert sizing["minimum_fallback_available"] is True


def test_order_sizing_view_exposes_minimum_fallback_when_thirty_percent_is_too_low() -> None:
    config = {"risk": {"risk_per_trade": 0.02, "max_notional_pct": 0.3}}
    snapshot = {
        "order_limits": {
            "effective_min_quantity_raw": "0.001",
            "effective_min_notional_raw": "78",
            "estimated_max_open_quantity_raw": "0.002",
            "estimated_max_open_notional_raw": "156",
        }
    }

    sizing = _order_sizing_view(
        config,
        snapshot,
        {"wallet_balance_raw": "8"},
    )

    assert sizing["strategy_notional_cap_raw"] == "46.8"
    assert sizing["strategy_cap_meets_minimum"] is False
    assert sizing["minimum_fallback_available"] is True
    assert "最小单" in sizing["formula"]


def test_dashboard_uses_saved_live_configuration_without_duplicate_confirmation() -> None:
    assert "confirmLive" not in DASHBOARD_HTML
    assert "productionPhrase" not in DASHBOARD_HTML
    assert "BINANCE LIVE BTCUSDT" not in DASHBOARD_HTML
    assert 'id="orderSizingBox"' in DASHBOARD_HTML
    assert 'id="tradeModelBox"' in DASHBOARD_HTML
    assert "s.trade_model||{}" in DASHBOARD_HTML
    assert "const exact=" in DASHBOARD_HTML
    assert "启动权限直接采用“配置”中已保存的交易模式和网络" in DASHBOARD_HTML
    assert "理论最大可开金额使用比例 (%)" in DASHBOARD_HTML
    assert "s.market.mark_price_raw" in DASHBOARD_HTML
    assert "最新成交价格" in DASHBOARD_HTML
    assert "setInterval(loadStatus,1000)" in DASHBOARD_HTML
    assert "动态退出" in DASHBOARD_HTML
    assert "pollSecondsInput.min='1'" in DASHBOARD_HTML
    assert "不改变 K 线周期" in DASHBOARD_HTML


def test_dashboard_trade_report_displays_entry_and_exit_times() -> None:
    assert "<th>开仓时间</th><th>平仓时间</th>" in DASHBOARD_HTML
    assert "formatBeijing(r.entry_time)" in DASHBOARD_HTML
    assert "formatBeijing(r.exit_time)" in DASHBOARD_HTML


def test_dashboard_marks_live_take_profit_as_dynamic() -> None:
    position = Position("long", 0.001, 100.0, 99.0, 102.5, 1)

    live = _position_dict(position, 101.0, mode="live")
    paper = _position_dict(position, 101.0, mode="paper")

    assert live is not None and live["take_profit_mode"] == "dynamic"
    assert paper is not None and paper["take_profit_mode"] == "fixed"


def test_dashboard_reprices_position_and_unrealized_pnl_from_live_mark() -> None:
    snapshot = {
        "positions": [
            {
                "symbol": "BTCUSDT",
                "side": "short",
                "quantity": 0.001,
                "entry_price": 77_852.6,
                "mark_price": 77_900.0,
                "unrealized_pnl": -0.0474,
            }
        ],
        "account": {
            "wallet_balance": 25.5,
            "wallet_balance_raw": "25.5",
            "unrealized_pnl": -0.0474,
            "margin_balance": 25.4526,
        },
    }

    positions, account = _mark_to_market_view(snapshot, "BTCUSDT", 77_800.0)

    assert positions[0]["mark_price"] == 77_800.0
    assert positions[0]["unrealized_pnl"] == pytest.approx(0.0526)
    assert positions[0]["unrealized_pnl_raw"] == "0.0526"
    assert account["unrealized_pnl"] == pytest.approx(0.0526)
    assert account["margin_balance"] == pytest.approx(25.5526)
    assert snapshot["positions"][0]["mark_price"] == 77_900.0


def test_dashboard_market_snapshot_is_single_flight_across_tabs() -> None:
    service = DashboardService.__new__(DashboardService)
    service._lock = threading.RLock()
    service._snapshot_condition = threading.Condition(service._lock)
    service._snapshot_refreshing = False
    service._exchange_snapshot = {}
    service._snapshot_at = 0.0

    class Adapter:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_dashboard_snapshot_nonblocking(self):
            self.calls += 1
            time.sleep(0.05)
            return {"market": {"mark_price": 100.0}, "private_available": True}

        def fetch_dashboard_snapshot(self):
            raise AssertionError("dashboard refresh must not wait for private-stream startup")

    adapter = Adapter()
    service._adapter = lambda _config, _exchange: adapter
    barrier = threading.Barrier(8)
    results = []

    def read_snapshot() -> None:
        barrier.wait()
        results.append(
            service._market_snapshot(
                {"dashboard_snapshot_seconds": 15},
                "binance",
            )
        )

    workers = [threading.Thread(target=read_snapshot) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)

    assert adapter.calls == 1
    assert len(results) == 8
    assert all(result["market"]["mark_price"] == 100.0 for result in results)


def test_stopped_dashboard_builds_only_an_adapter_not_a_trade_engine() -> None:
    service = DashboardService.__new__(DashboardService)
    service._lock = threading.RLock()
    service.engine = None
    service._dashboard_adapter = None
    service._dashboard_adapter_key = None
    expected_adapter = object()
    config = {
        "account": {"margin_mode": "isolated", "position_mode": "net"},
        "exchanges": {
            "binance": {
                "environment": "production",
                "base_url": "https://fapi.binance.com",
                "symbol": "BTCUSDT",
            }
        },
    }

    with patch(
        "btc_futures_bot.dashboard.make_adapter",
        return_value=expected_adapter,
    ) as adapter_factory, patch(
        "btc_futures_bot.dashboard.build_engine",
        side_effect=AssertionError("a dashboard snapshot must not build an engine"),
    ):
        result = service._adapter(config, "binance")

    assert result is expected_adapter
    adapter_factory.assert_called_once_with(
        "binance",
        config["exchanges"]["binance"],
        config["account"],
    )


def test_dashboard_run_loop_alerts_on_error_then_resolves_after_success() -> None:
    service = DashboardService.__new__(DashboardService)
    service._lock = threading.RLock()
    service.last_result = None
    service.last_error = ""
    service.last_cycle_at = 0.0
    service._first_cycle_logged = False
    service._last_logged_error = ""
    service._last_macro_block = ""
    timeline: list[str] = []

    recovered_result = SimpleNamespace(
        exchange="binance",
        status="flat",
        position=None,
        signal=SimpleNamespace(side="flat", score=0),
        raw={},
    )

    class Engine:
        def __init__(self) -> None:
            self.config = SimpleNamespace(poll_seconds=1)
            self.macro_risk = None
            self.evaluate_calls = 0
            self.notifications: list[dict[str, object]] = []
            self.resolutions: list[tuple[str, str]] = []

        def evaluate_once(self):
            self.evaluate_calls += 1
            if self.evaluate_calls == 1:
                timeline.append("evaluate_error")
                raise RuntimeError("venue cycle unavailable")
            assert service._last_logged_error == "venue cycle unavailable"
            timeline.append("evaluate_success")
            return recovered_result

        def notify_emergency(
            self,
            error: object,
            *,
            category: str,
            context: str,
            incident: str = "",
            details: dict[str, object] | None = None,
        ) -> bool:
            timeline.append("notify_emergency")
            self.notifications.append(
                {
                    "error": error,
                    "category": category,
                    "context": context,
                    "incident": incident,
                    "details": details,
                }
            )
            return True

        def resolve_emergency(self, category: str, incident: str = "") -> None:
            timeline.append("resolve_emergency")
            self.resolutions.append((category, incident))

    class TwoCycleStopEvent:
        def __init__(self) -> None:
            self.checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            return self.checks > 2

        def wait(self, _timeout: float) -> bool:
            return False

    class OperationLogger:
        def record(self, *_args: object, **kwargs: object) -> None:
            if kwargs.get("status") == "error":
                timeline.append("record_error")

    engine = Engine()
    service.engine = engine
    service._stop_event = TwoCycleStopEvent()
    service.operation_logger = OperationLogger()

    service._run_loop()

    assert engine.evaluate_calls == 2
    assert len(engine.notifications) == 1
    notification = engine.notifications[0]
    assert str(notification["error"]) == "venue cycle unavailable"
    assert notification["category"] == "engine_runtime"
    assert notification["context"] == "行情周期执行失败"
    assert notification["incident"] == "cycle"
    assert engine.resolutions == [("engine_runtime", "cycle")]
    assert service.last_result is recovered_result
    assert service.last_error == ""
    assert service._last_logged_error == ""
    assert timeline.index("notify_emergency") < timeline.index("record_error")
    assert timeline.index("evaluate_success") < timeline.index("resolve_emergency")


@pytest.mark.parametrize(
    ("snapshot", "expected_error"),
    [
        (
            {
                "market": {"mark_price": 100.0},
                "private_available": True,
                "order_limits": {"error": "HTTP 429: rate limit exceeded"},
            },
            "HTTP 429: rate limit exceeded",
        ),
        (
            {
                "market": {"mark_price": 100.0},
                "private_available": False,
                "private_error": "HTTP 418: IP banned after rate limit violation",
            },
            "HTTP 418: IP banned after rate limit violation",
        ),
    ],
    ids=("order-limits", "private-api"),
)
def test_dashboard_market_snapshot_alerts_on_rate_limit_errors(
    snapshot: dict[str, object],
    expected_error: str,
) -> None:
    service = DashboardService.__new__(DashboardService)
    service._lock = threading.RLock()
    service._snapshot_condition = threading.Condition(service._lock)
    service._snapshot_refreshing = False
    service._exchange_snapshot = {}
    service._snapshot_at = 0.0
    service._private_snapshot_at = 0.0
    service.notifier = _RecordingEmergencyNotifier()

    class Adapter:
        def fetch_dashboard_snapshot(self) -> dict[str, object]:
            return snapshot

    service._adapter = lambda _config, _exchange: Adapter()
    config = {
        "mode": "live",
        "dashboard_snapshot_seconds": 5,
        "exchanges": {
            "binance": {
                "environment": "production",
                "symbol": "BTCUSDT",
            }
        },
    }

    with patch("btc_futures_bot.dashboard.time.time", return_value=100.0):
        service._market_snapshot(config, "binance")

    assert len(service.notifier.notifications) == 1
    notification = service.notifier.notifications[0]
    assert str(notification["error"]) == expected_error
    assert notification["category"] == "ip_restricted"
    assert notification["context"] == "页面交易所状态检测"
    assert notification["incident"] == "snapshot"


def test_dashboard_healthy_market_snapshot_resolves_rate_limit_alert() -> None:
    service = DashboardService.__new__(DashboardService)
    service._lock = threading.RLock()
    service._snapshot_condition = threading.Condition(service._lock)
    service._snapshot_refreshing = False
    service._exchange_snapshot = {}
    service._snapshot_at = 0.0
    service._private_snapshot_at = 0.0
    service.notifier = _RecordingEmergencyNotifier()
    snapshot = {
        "market": {"mark_price": 100.0},
        "private_available": True,
        "private_source": "rest",
        "private_error": "",
        "order_limits": {"effective_min_quantity_raw": "0.001"},
    }

    class Adapter:
        def fetch_dashboard_snapshot(self) -> dict[str, object]:
            return snapshot

    service._adapter = lambda _config, _exchange: Adapter()
    config = {
        "mode": "live",
        "dashboard_snapshot_seconds": 5,
        "exchanges": {
            "binance": {
                "environment": "production",
                "symbol": "BTCUSDT",
            }
        },
    }

    with patch("btc_futures_bot.dashboard.time.time", return_value=100.0):
        service._market_snapshot(config, "binance")

    assert service.notifier.notifications == []
    assert service.notifier.resolutions == [("ip_restricted", "binance", "snapshot")]


def test_dashboard_websocket_snapshot_does_not_falsely_resolve_rest_ip_alert() -> None:
    service = DashboardService.__new__(DashboardService)
    service.notifier = _RecordingEmergencyNotifier()
    snapshot = {
        "market": {"mark_price": 100.0},
        "private_available": True,
        "private_source": "websocket",
        "private_error": "",
        "order_limits": {"effective_min_quantity_raw": "0.001"},
    }
    config = {
        "mode": "live",
        "exchanges": {
            "binance": {"environment": "production", "symbol": "BTCUSDT"}
        },
    }

    service._handle_snapshot_alerts(snapshot, config, "binance")

    assert service.notifier.notifications == []
    assert service.notifier.resolutions == []


def test_dashboard_email_test_rejects_delivery_timeout() -> None:
    service = DashboardService.__new__(DashboardService)

    class Notifier:
        config = SimpleNamespace(
            enabled=True,
            timeout_seconds=1.0,
            password_env="BTC_EMAIL_PASSWORD",
            recipients=("owner@example.com",),
        )
        ready = True

        @staticmethod
        def status() -> dict[str, object]:
            return {"sent_count": 0, "last_error": ""}

        @staticmethod
        def send_test() -> bool:
            return True

        @staticmethod
        def flush(_timeout: float) -> bool:
            return False

    service.notifier = Notifier()

    with pytest.raises(RuntimeError, match="发送超时"):
        service.email_test()


def test_dashboard_explicit_restart_clears_old_ip_rate_limit_gate() -> None:
    service = DashboardService.__new__(DashboardService)
    timeline: list[str] = []

    class OperationLogger:
        @staticmethod
        def record(*_args: object, **_kwargs: object) -> None:
            return None

    service.operation_logger = OperationLogger()
    service.stop = lambda: timeline.append("stop") or {"running": False}
    service.start = lambda _payload: timeline.append("start") or {"running": True}

    with patch(
        "btc_futures_bot.dashboard.clear_rate_limits",
        side_effect=lambda: timeline.append("clear_rate_limits"),
    ):
        result = service.restart({"reason": "proxy node changed"})

    assert result == {"running": True}
    assert timeline == ["stop", "clear_rate_limits", "start"]


def test_dashboard_overlays_cached_private_snapshot_with_live_market_tick() -> None:
    service = DashboardService.__new__(DashboardService)

    class Adapter:
        def fetch_live_market_snapshot(self):
            return {
                "mark_price": 101.0,
                "mark_price_raw": "101.0",
                "last_price": 101.25,
                "last_price_raw": "101.25",
                "last_price_timestamp": 1_700_000_000_000,
            }

    service._adapter = lambda _config, _exchange: Adapter()
    cached = {
        "market": {"mark_price": 100.0, "stale": True},
        "account": {"wallet_balance": 27.0},
        "open_orders": [{"order_id": 1}],
    }

    refreshed = service._with_live_market({}, "binance", cached)

    assert refreshed is not cached
    assert refreshed["market"]["mark_price"] == 101.0
    assert refreshed["market"]["last_price"] == 101.25
    assert "stale" not in refreshed["market"]
    assert refreshed["account"] is cached["account"]
    assert refreshed["open_orders"] is cached["open_orders"]


def test_dashboard_overlays_orders_and_positions_from_live_private_stream() -> None:
    service = DashboardService.__new__(DashboardService)

    class Adapter:
        def fetch_live_dashboard_snapshot(self):
            return {
                "market": {"mark_price": 101.0},
                "account": {"wallet_balance": 27.0},
                "positions": [{"mark_price": 101.0}],
                "open_orders": [{"order_id": 2, "status": "NEW"}],
                "private_available": True,
                "private_source": "websocket",
            }

    service._adapter = lambda _config, _exchange: Adapter()
    cached = {
        "market": {"mark_price": 100.0, "stale": True},
        "account": {"wallet_balance": 26.0},
        "positions": [{"mark_price": 100.0}],
        "open_orders": [{"order_id": 1}],
        "order_limits": {"effective_min_quantity_raw": "0.001"},
        "private_available": True,
        "private_stale": True,
        "private_warning": "old",
    }

    refreshed = service._with_live_market({}, "binance", cached)

    assert refreshed["positions"][0]["mark_price"] == 101.0
    assert refreshed["open_orders"][0]["order_id"] == 2
    assert refreshed["order_limits"] is cached["order_limits"]
    assert "private_stale" not in refreshed
    assert "stale" not in refreshed["market"]


def test_dashboard_snapshot_failure_keeps_private_cache_and_backs_off() -> None:
    service = DashboardService.__new__(DashboardService)
    service._lock = threading.RLock()
    service._snapshot_condition = threading.Condition(service._lock)
    service._snapshot_refreshing = False
    service._exchange_snapshot = {
        "market": {"mark_price": 100.0},
        "account": {"wallet_balance": 10.0},
        "private_available": True,
        "private_error": "",
    }
    service._snapshot_at = 0.0
    service._private_snapshot_at = 90.0

    class Adapter:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_dashboard_snapshot(self):
            self.calls += 1
            raise OSError("temporary TLS EOF")

    adapter = Adapter()
    service._adapter = lambda _config, _exchange: adapter
    config = {"dashboard_snapshot_seconds": 15}
    with (
        patch("btc_futures_bot.dashboard.time.time", return_value=100.0),
        patch("btc_futures_bot.dashboard.time.monotonic", return_value=100.0),
    ):
        stale = service._market_snapshot(config, "binance")

    assert stale["private_available"] is True
    assert stale["private_stale"] is True
    assert stale["private_error"] == ""
    assert stale["market"]["stale"] is True
    assert "temporary TLS EOF" in stale["snapshot_error"]
    assert service._snapshot_at == 100.0

    with patch("btc_futures_bot.dashboard.time.time", return_value=101.0):
        cached = service._market_snapshot(config, "binance")

    assert cached is stale
    assert adapter.calls == 1


def test_dashboard_encoded_transient_failure_keeps_recent_private_snapshot() -> None:
    service = DashboardService.__new__(DashboardService)
    service._lock = threading.RLock()
    service._snapshot_condition = threading.Condition(service._lock)
    service._snapshot_refreshing = False
    service._exchange_snapshot = {
        "market": {"mark_price": 100.0},
        "account": {"wallet_balance": 27.0},
        "positions": [{"symbol": "BTCUSDT", "quantity": 0.001}],
        "open_orders": [{"order_id": 1}],
        "private_available": True,
        "private_error": "",
    }
    service._snapshot_at = 0.0
    service._private_snapshot_at = 90.0

    class Adapter:
        def fetch_dashboard_snapshot(self):
            return {
                "market": {"mark_price": 101.0},
                "positions": [],
                "open_orders": [],
                "private_available": False,
                "private_transient": True,
                "private_error": "temporary TLS timeout",
            }

    service._adapter = lambda _config, _exchange: Adapter()
    with (
        patch("btc_futures_bot.dashboard.time.time", return_value=100.0),
        patch("btc_futures_bot.dashboard.time.monotonic", return_value=100.0),
    ):
        stale = service._market_snapshot(
            {
                "dashboard_snapshot_seconds": 15,
                "dashboard_private_stale_seconds": 90,
            },
            "binance",
        )

    assert stale["market"]["mark_price"] == 101.0
    assert stale["account"]["wallet_balance"] == 27.0
    assert stale["positions"][0]["quantity"] == 0.001
    assert stale["private_available"] is True
    assert stale["private_stale"] is True
    assert stale["private_error"] == ""
    assert "temporary TLS timeout" in stale["snapshot_error"]


def test_dashboard_auth_failure_does_not_reuse_private_snapshot() -> None:
    service = DashboardService.__new__(DashboardService)
    service._lock = threading.RLock()
    service._snapshot_condition = threading.Condition(service._lock)
    service._snapshot_refreshing = False
    service._exchange_snapshot = {
        "market": {"mark_price": 100.0},
        "account": {"wallet_balance": 27.0},
        "private_available": True,
    }
    service._snapshot_at = 0.0
    service._private_snapshot_at = 90.0

    class Adapter:
        def fetch_dashboard_snapshot(self):
            return {
                "market": {"mark_price": 101.0},
                "positions": [],
                "open_orders": [],
                "private_available": False,
                "private_transient": False,
                "private_error": "invalid API key",
            }

    service._adapter = lambda _config, _exchange: Adapter()
    with (
        patch("btc_futures_bot.dashboard.time.time", return_value=100.0),
        patch("btc_futures_bot.dashboard.time.monotonic", return_value=100.0),
    ):
        failed = service._market_snapshot(
            {"dashboard_snapshot_seconds": 15},
            "binance",
        )

    assert failed["private_available"] is False
    assert "account" not in failed
    assert failed["private_error"] == "invalid API key"


def test_dashboard_transient_failure_does_not_reuse_expired_private_snapshot() -> None:
    service = DashboardService.__new__(DashboardService)
    service._lock = threading.RLock()
    service._snapshot_condition = threading.Condition(service._lock)
    service._snapshot_refreshing = False
    service._exchange_snapshot = {
        "market": {"mark_price": 100.0},
        "account": {"wallet_balance": 27.0},
        "private_available": True,
    }
    service._snapshot_at = 0.0
    service._private_snapshot_at = 1.0

    class Adapter:
        def fetch_dashboard_snapshot(self):
            return {
                "market": {"mark_price": 101.0},
                "private_available": False,
                "private_transient": True,
                "private_error": "temporary TLS timeout",
            }

    service._adapter = lambda _config, _exchange: Adapter()
    with (
        patch("btc_futures_bot.dashboard.time.time", return_value=100.0),
        patch("btc_futures_bot.dashboard.time.monotonic", return_value=100.0),
    ):
        failed = service._market_snapshot(
            {
                "dashboard_snapshot_seconds": 15,
                "dashboard_private_stale_seconds": 90,
            },
            "binance",
        )

    assert failed["private_available"] is False
    assert "account" not in failed


def test_dashboard_live_private_failure_never_falls_back_to_paper_equity() -> None:
    service = DashboardService.__new__(DashboardService)
    service._thread = None
    service._lock = threading.RLock()
    service.engine = None
    service.last_result = None
    service.last_error = ""
    service.last_cycle_at = 0.0
    service.started_at = 0.0
    service._config = lambda: {
        "instance_id": "trade-model-2",
        "mode": "live",
        "paper_equity": 10_000.0,
        "active_exchange": "binance",
        "exchanges": {
            "binance": {
                "symbol": "BTCUSDT",
                "environment": "production",
                "base_url": "https://fapi.binance.com",
            }
        },
    }
    service._exchange = lambda _config: "binance"
    service._market_snapshot = lambda _config, _exchange: {
        "market": {"mark_price": 100.0},
        "positions": [],
        "open_orders": [],
        "private_available": False,
        "private_error": "temporary TLS timeout",
    }

    status = service.status()

    assert status["instance_id"] == "trade-model-2"
    assert status["account"]["source"] == "unavailable"
    assert status["account"]["wallet_balance_raw"] == ""
    assert status["account"]["wallet_balance_raw"] != "10000"
    assert status["order_sizing"]["available"] is False


def test_dashboard_never_reports_private_connection_when_snapshot_has_error() -> None:
    service = DashboardService.__new__(DashboardService)
    service._thread = None
    service._lock = threading.RLock()
    service.engine = None
    service.last_result = None
    service.last_error = ""
    service.last_cycle_at = 0.0
    service.started_at = 0.0
    service._config = lambda: {
        "instance_id": "trade-model-2",
        "mode": "paper",
        "paper_equity": 10_000.0,
        "active_exchange": "okx",
        "exchanges": {
            "okx": {
                "symbol": "BTC-USDT-SWAP",
                "environment": "demo",
                "base_url": "https://openapi.okx.com",
            }
        },
    }
    service._exchange = lambda _config: "okx"
    service._market_snapshot = lambda _config, _exchange: {
        "market": {"mark_price": 100.0},
        "positions": [],
        "open_orders": [],
        "private_available": True,
        "private_error": "Invalid OK-ACCESS-KEY",
    }
    service._with_live_market = lambda _config, _exchange, snapshot: snapshot

    status = service.status()

    assert status["connection"]["market"] is True
    assert status["connection"]["private"] is False
    assert status["connection"]["private_error"] == "Invalid OK-ACCESS-KEY"
    with pytest.raises(RuntimeError, match="Invalid OK-ACCESS-KEY"):
        service.private_check()


def test_private_check_rejects_a_stale_cached_account() -> None:
    service = DashboardService.__new__(DashboardService)
    service._config = lambda: {"active_exchange": "binance", "exchanges": {"binance": {}}}
    service._exchange = lambda _config: "binance"
    service._market_snapshot = lambda _config, _exchange: {
        "private_available": True,
        "private_stale": True,
        "account": {"wallet_balance": 27.0},
    }

    try:
        service.private_check()
    except RuntimeError as error:
        assert "私有 API" in str(error)
    else:
        raise AssertionError("stale private data must not pass an API check")
