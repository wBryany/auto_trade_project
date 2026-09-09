from __future__ import annotations

import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from btc_futures_bot.backtest import run_backtest
from btc_futures_bot.costs import CostConfig
from btc_futures_bot.dashboard import DashboardService, _time_exit_policy
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.main import load_config, save_dashboard_config
from btc_futures_bot.models import Candle, Position, Signal
from btc_futures_bot.reporting import TradeReporter
from btc_futures_bot.risk import RiskConfig, RiskManager
from btc_futures_bot.strategy import StrategyConfig


def _isolated_risk() -> RiskManager:
    return RiskManager(
        RiskConfig(risk_per_trade=0.005, stop_loss_pct=0.005, max_notional_pct=0.2),
        costs=CostConfig(
            execution="taker", maker_fee_pct=0.0002, taker_fee_pct=0.0005,
            slippage_pct=0.0002, funding_rate_pct_per_8h=0,
            expected_holding_hours=0, min_net_edge_pct=0.001,
        ),
    )


def test_dashboard_save_preserves_disabled_time_exit_and_exposes_it(tmp_path: Path) -> None:
    source = tmp_path / "config.json"
    source.write_text(json.dumps({
        "mode": "paper", "active_exchange": "okx",
        "exchanges": {"okx": {"enabled": True}}, "credentials": {"okx": {}},
        "strategy": {"enable_time_exit": False, "max_hold_seconds": 3600, "hard_max_hold_seconds": 5400},
    }), encoding="utf-8")
    # Ordinary dashboard edits omit the read-only time-exit setting.
    saved = save_dashboard_config(source, {"exchange": "okx", "poll_seconds": 2})
    assert load_config(str(saved))["strategy"]["enable_time_exit"] is False
    service = DashboardService(source)
    assert service.config_view()["enable_time_exit"] is False
    assert _time_exit_policy(load_config(str(saved))) == {
        "enabled": False, "soft_seconds": None, "hard_seconds": None,
    }


def test_time_exit_display_uses_running_strategy_until_restart() -> None:
    saved = {"strategy": {"enable_time_exit": False, "max_hold_seconds": 3600, "hard_max_hold_seconds": 5400}}
    active = StrategyConfig(enable_time_exit=True, max_hold_seconds=3600, hard_max_hold_seconds=5400)
    assert _time_exit_policy(saved, active) == {
        "enabled": True, "soft_seconds": 3600, "hard_seconds": 5400,
    }
    assert not _time_exit_policy(saved, replace(active, enable_time_exit=False))["enabled"]


def _live_engine(*, enable_time_exit: bool) -> TradingEngine:
    strategy = SimpleNamespace(
        config=StrategyConfig(
            mode="traditional_kline", enable_time_exit=enable_time_exit,
            max_hold_seconds=3600, hard_max_hold_seconds=5400, time_exit_min_r=0.2,
            break_even_trigger_r=0, trailing_trigger_r=0,
            enable_adverse_dynamic_exit=False, enable_breakout_failure_exit=False,
            enable_profit_trend_exit=True, profit_trend_exit_trigger_r=1.0,
        ),
        position_trend_invalidated=Mock(return_value=False),
    )
    engine = TradingEngine(
        SimpleNamespace(name="binance", settings=SimpleNamespace(symbol="BTCUSDT", environment="production")),
        strategy, _isolated_risk(), EngineConfig(mode="live"),
    )
    engine.position = Position(
        "long", 1.0, 100.0, 99.0, 103.0, int(time.time() * 1000) - 7_200_000,
        initial_stop_price=99.0, best_price=100.0, stop_order_id="exchange-hard-stop",
    )
    engine._close_live_position = Mock(return_value={})
    return engine


@pytest.mark.parametrize("mark_price,enabled_reason", [(99.8, "hard_time_exit"), (100.5, "time_exit")])
def test_live_holds_beyond_both_time_limits_when_disabled(mark_price: float, enabled_reason: str) -> None:
    signal = Signal("flat", 0, 1)
    disabled = _live_engine(enable_time_exit=False)
    assert disabled._manage_live_position(mark_price, {}, signal) is None
    disabled._close_live_position.assert_not_called()
    assert disabled.position is not None
    assert disabled.position.stop_order_id == "exchange-hard-stop"

    enabled = _live_engine(enable_time_exit=True)
    result = enabled._manage_live_position(mark_price, {}, signal)
    assert result.raw["exit_reason"] == enabled_reason
    enabled._close_live_position.assert_called_once_with(mark_price, enabled_reason)


@pytest.mark.parametrize("exit_kind", ["protected_stop", "trend"])
def test_live_stop_and_trend_exits_remain_active_without_time_limit(exit_kind: str) -> None:
    engine = _live_engine(enable_time_exit=False)
    if exit_kind == "protected_stop":
        engine.position = replace(engine.position, stop_price=100.2, best_price=102.0, stop_reason="trailing_stop")
        mark = 100.1
        expected = "trailing_stop"
    else:
        engine.position = replace(engine.position, best_price=103.0)
        engine.strategy.position_trend_invalidated.return_value = True
        mark = 102.0
        expected = "trend_invalidation"
    candles = {"5m": [Candle(1, mark, mark, mark, mark, 1.0)]}
    result = engine._manage_live_position(mark, candles, Signal("flat", 0, 1))
    assert result.raw["exit_reason"] == expected
    engine._close_live_position.assert_called_once_with(mark, expected)


@pytest.mark.parametrize("enable_time_exit,event,expected", [
    (False, "none", None), (True, "none", "hard_time_exit"),
    (False, "stop", "stop_loss"), (False, "trend", "trend_invalidation"),
])
def test_backtest_unlimited_hold_retains_protection(
    tmp_path: Path, enable_time_exit: bool, event: str, expected: str | None,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    base = 1_700_000_000_000
    event_time = base + 135 * 60_000
    with (data / "1m.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timestamp", "open", "high", "low", "close", "volume"))
        writer.writeheader()
        for index in range(150):
            opening = close = 100.0
            high, low = 100.01, 99.99
            if index == 135 and event == "stop":
                low = close = 98.0
            elif index >= 135 and event == "trend":
                high, close = 101.1, 101.0
                if index > 135:
                    opening, low = 101.0, 100.99
            writer.writerow({"timestamp": base + index * 60_000, "open": opening,
                             "high": high, "low": low, "close": close, "volume": 10.0})

    class OneEntry:
        config = StrategyConfig(
            mode="traditional_kline", trigger_timeframe="1m", regime_timeframe="1m",
            enable_time_exit=enable_time_exit, max_hold_seconds=3600, hard_max_hold_seconds=5400,
            min_stop_loss_pct=0.005, max_stop_loss_pct=0.005, take_profit_r=2.5,
            break_even_trigger_r=0, trailing_trigger_r=0,
            enable_adverse_dynamic_exit=False, enable_breakout_failure_exit=False,
            enable_profit_trend_exit=True, profit_trend_exit_trigger_r=1.0,
        )

        def __init__(self):
            self.calls = 0

        def evaluate(self, candles):
            self.calls += 1
            return Signal("long" if self.calls == 1 else "flat", 6, candles["1m"][-1].timestamp)

        def position_trend_invalidated(self, _side, candles):
            return event == "trend" and candles["1m"][-1].timestamp >= event_time

    reports = tmp_path / "reports"
    reporter = TradeReporter(reports)
    try:
        summary = run_backtest(
            data, strategy=OneEntry(), reporter=reporter,
            risk=_isolated_risk(),
        )
    finally:
        reporter.close()
    with (reports / "trade_report.csv").open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if expected is None:
        assert summary.trades == 0
        assert not records
    else:
        assert summary.trades == 1
        assert records[0]["exit_reason"] == expected
        if not enable_time_exit:
            assert float(records[0]["holding_minutes"]) > 120
