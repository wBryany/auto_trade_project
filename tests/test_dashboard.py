from __future__ import annotations

import math
import threading
import time
from unittest.mock import patch

from btc_futures_bot.dashboard import (
    DASHBOARD_HTML,
    DashboardService,
    _json_safe,
    _order_sizing_view,
    _position_dict,
)
from btc_futures_bot.models import Position


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
    assert "const exact=" in DASHBOARD_HTML
    assert "启动权限直接采用“配置”中已保存的交易模式和网络" in DASHBOARD_HTML
    assert "理论最大可开金额使用比例 (%)" in DASHBOARD_HTML
    assert "s.market.mark_price_raw" in DASHBOARD_HTML
    assert "动态退出" in DASHBOARD_HTML
    assert "pollSecondsInput.min='1'" in DASHBOARD_HTML
    assert "不改变 K 线周期" in DASHBOARD_HTML


def test_dashboard_marks_live_take_profit_as_dynamic() -> None:
    position = Position("long", 0.001, 100.0, 99.0, 102.5, 1)

    live = _position_dict(position, 101.0, mode="live")
    paper = _position_dict(position, 101.0, mode="paper")

    assert live is not None and live["take_profit_mode"] == "dynamic"
    assert paper is not None and paper["take_profit_mode"] == "fixed"


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

        def fetch_dashboard_snapshot(self):
            self.calls += 1
            time.sleep(0.05)
            return {"market": {"mark_price": 100.0}, "private_available": True}

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

    assert status["account"]["source"] == "unavailable"
    assert status["account"]["wallet_balance_raw"] == ""
    assert status["account"]["wallet_balance_raw"] != "10000"
    assert status["order_sizing"]["available"] is False


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
