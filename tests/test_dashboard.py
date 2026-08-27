from __future__ import annotations

import math

from btc_futures_bot.dashboard import DASHBOARD_HTML, _json_safe, _order_sizing_view, _position_dict
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
