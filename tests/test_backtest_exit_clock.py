from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from btc_futures_bot.backtest import run_backtest
from btc_futures_bot.costs import CostConfig
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.models import Candle, Position, Signal
from btc_futures_bot.risk import RiskConfig, RiskManager
from btc_futures_bot.strategy import StrategyConfig


BASE = 1_800_000_000_000


def _replay(side, reason, *, decision_close=101.0, fill_open=98.0, has_next=True, low=99.9):
    config = StrategyConfig(
        trigger_timeframe="1m", regime_timeframe="1m",
        min_stop_loss_pct=0.05, max_stop_loss_pct=0.05,
        break_even_trigger_r=0, trailing_trigger_r=0,
        enable_time_exit=reason in {"time_exit", "hard_time_exit", "trend_and_time"},
        max_hold_seconds=60, hard_max_hold_seconds=60 if reason == "hard_time_exit" else 120,
        time_exit_min_r=0.1,
        enable_profit_trend_exit=reason in {"trend_invalidation", "trend_and_time"},
        profit_trend_exit_trigger_r=0.1,
    )
    bars = [
        Candle(BASE, 100, 100.1, 99.9, 100, 10),
        Candle(BASE + 60_000, 100, 100.1, 99.9, 100, 10),
        Candle(BASE + 120_000, 100, max(100.1, decision_close + 0.1), min(low, decision_close - 0.1), decision_close, 10),
    ]
    if has_next:
        bars.append(Candle(BASE + 180_000, fill_open, max(fill_open, decision_close) + 0.1,
                           min(fill_open, decision_close) - 0.1, decision_close, 10))
    if side == "short":
        bars = [replace(c, open=200-c.open, high=200-c.low, low=200-c.high, close=200-c.close) for c in bars]
    strategy = SimpleNamespace(
        config=config,
        evaluate=lambda data: Signal(side, 7, BASE + 60_000, ("test_entry",))
        if data["1m"][-1].timestamp == BASE + 60_000
        else Signal("flat", 0, data["1m"][-1].timestamp, ()),
        position_trend_invalidated=lambda *_: True,
    )
    risk = RiskManager(RiskConfig(stop_loss_pct=0.05), costs=CostConfig(funding_rate_pct_per_8h=0))
    records = []
    with patch("btc_futures_bot.backtest.load_csv", return_value=bars), patch(
        "btc_futures_bot.backtest.adverse_dynamic_exit_reason",
        return_value="dynamic_stop_loss" if reason == "dynamic_stop_loss" else "",
    ), patch(
        "btc_futures_bot.backtest.breakout_failure_exit_reason",
        return_value="breakout_failure" if reason == "breakout_failure" else "",
    ):
        summary = run_backtest(Path("."), strategy=strategy, risk=risk,
                               reporter=SimpleNamespace(record_trade=records.append))
    return records, summary, strategy, risk


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("reason", ["time_exit", "hard_time_exit", "dynamic_stop_loss", "breakout_failure", "trend_invalidation"])
def test_closed_bar_exits_use_next_open_and_matching_execution_time(side, reason):
    decision_close = 100 if reason == "hard_time_exit" else 101
    records, summary, strategy, risk = _replay(side, reason, decision_close=decision_close)
    assert summary.trades == 1
    record = records[0]
    assert record.exit_reason == reason
    assert record.exit_price == (98 if side == "long" else 102)
    assert record.exit_time == datetime.fromtimestamp((BASE + 180_000) / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
    assert record.holding_minutes == 1
    assert record.net_pnl == pytest.approx(risk.estimate_net_pnl(
        side, record.entry_price, record.exit_price, record.quantity, holding_hours=1 / 60,
    ))
    if reason in {"time_exit", "hard_time_exit"}:
        engine = TradingEngine(
            SimpleNamespace(name="binance", settings=SimpleNamespace(symbol="BTCUSDT", environment="production")),
            strategy, risk, EngineConfig(mode="live"),
        )
        engine.position = Position(side, 1, 100, 95 if side == "long" else 105,
                                   112.5 if side == "long" else 87.5, BASE + 120_000,
                                   initial_stop_price=95 if side == "long" else 105)
        with patch("btc_futures_bot.engine.time.time", return_value=(BASE + 180_000) / 1000):
            price = decision_close if side == "long" else 200-decision_close
            assert engine._live_time_exit_reason(price) == reason


@pytest.mark.parametrize("reason", ["time_exit", "hard_time_exit", "dynamic_stop_loss", "breakout_failure", "trend_invalidation"])
def test_final_closed_bar_has_no_invented_market_fill(reason):
    records, summary, _, _ = _replay("long", reason, has_next=False)
    assert records == []
    assert summary.trades == 0


def test_soft_time_exit_does_not_use_future_gap_profit_as_confirmation():
    records, summary, _, _ = _replay("long", "time_exit", decision_close=100, fill_open=102)
    # At 60 seconds the known close is not profitable after fees. The future
    # gap must not be consulted to manufacture a profitable time-exit signal.
    assert records == []
    assert summary.trades == 0


def test_adverse_gap_can_turn_a_valid_profitable_time_exit_into_a_loss():
    records, _, _, _ = _replay("long", "time_exit", decision_close=101, fill_open=98)
    assert records[0].exit_reason == "time_exit"
    assert records[0].net_pnl < 0


def test_profit_trend_exit_has_same_priority_as_live_when_time_exit_is_due():
    records, _, _, _ = _replay("long", "trend_and_time")
    assert records[0].exit_reason == "trend_invalidation"


@pytest.mark.parametrize("side", ["long", "short"])
def test_intrabar_stop_keeps_existing_price_timestamp_and_priority(side):
    records, _, _, _ = _replay(side, "dynamic_stop_loss", low=94)
    assert records[0].exit_reason == "stop_loss"
    assert records[0].exit_price == (95 if side == "long" else 105)
    assert records[0].exit_time == datetime.fromtimestamp((BASE + 120_000) / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
