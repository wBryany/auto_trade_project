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
ENTRY_TIME = BASE + 120_000
DECISION_TIME = BASE + 180_000


def _replay(
    side, *, decision_close=101.0, fill_open=98.0, score=4,
    minimum_hold=60, execution_time=DECISION_TIME, has_next=True,
    funding_rate=0.0, later_signals=False,
):
    config = StrategyConfig(
        trigger_timeframe="1m", regime_timeframe="1m",
        min_stop_loss_pct=0.05, max_stop_loss_pct=0.05,
        break_even_trigger_r=0, trailing_trigger_r=0,
        enable_time_exit=False, enable_profit_trend_exit=False,
        min_hold_seconds=minimum_hold, reversal_min_score=5,
    )
    bars = [
        Candle(BASE, 100, 100.1, 99.9, 100, 10),
        Candle(BASE + 60_000, 100, 100.1, 99.9, 100, 10),
        Candle(ENTRY_TIME, 100, max(100.1, decision_close + 0.1),
               min(99.9, decision_close - 0.1), decision_close, 10),
    ]
    if has_next:
        bars.append(Candle(execution_time, fill_open, max(fill_open, decision_close) + 0.1,
                           min(fill_open, decision_close) - 0.1, decision_close, 10))
    if later_signals:
        bars.extend(Candle(execution_time + offset, fill_open, fill_open + 0.1,
                           fill_open - 0.1, fill_open, 10)
                    for offset in (60_000, 120_000, 180_000))
    if side == "short":
        bars = [replace(c, open=200-c.open, high=200-c.low,
                        low=200-c.high, close=200-c.close) for c in bars]
    opposite = "short" if side == "long" else "long"

    def evaluate(data):
        timestamp = data["1m"][-1].timestamp
        if timestamp == BASE + 60_000:
            return Signal(side, 7, timestamp, ("initial_entry",))
        if timestamp == ENTRY_TIME:
            return Signal(opposite, score, timestamp, ("opposite",))
        if later_signals and timestamp == execution_time + 60_000:
            return Signal(side, 7, timestamp, ("later_entry",))
        if later_signals and timestamp == execution_time + 120_000:
            return Signal(opposite, 7, timestamp, ("later_opposite",))
        return Signal("flat", 0, timestamp, ())

    strategy = SimpleNamespace(config=config, evaluate=evaluate)
    risk = RiskManager(RiskConfig(stop_loss_pct=0.05),
                       costs=CostConfig(funding_rate_pct_per_8h=funding_rate))
    records = []
    with patch("btc_futures_bot.backtest.load_csv", return_value=bars), patch(
        "btc_futures_bot.backtest.adverse_dynamic_exit_reason", return_value="",
    ), patch("btc_futures_bot.backtest.breakout_failure_exit_reason", return_value=""):
        summary = run_backtest(Path("."), strategy=strategy, risk=risk,
                               reporter=SimpleNamespace(record_trade=records.append))

    engine = TradingEngine(
        SimpleNamespace(name="binance", settings=SimpleNamespace(symbol="BTCUSDT", environment="production")),
        strategy, risk, EngineConfig(mode="live"),
    )
    position = Position(side, 1, 100, 95 if side == "long" else 105,
                        112.5 if side == "long" else 87.5, ENTRY_TIME)
    with patch("btc_futures_bot.engine.time.time", return_value=DECISION_TIME / 1000):
        live_decision = engine._should_reverse(
            position, decision_close if side == "long" else 200-decision_close,
            Signal(opposite, score, ENTRY_TIME, ("opposite",)),
        )
    return records, summary, live_decision, risk


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("decision_close,expected_exit", [(101.0, True), (100.05, False)])
def test_weak_reversal_decision_uses_known_net_profit_not_future_gap(side, decision_close, expected_exit):
    # 100.05 is a gross gain but a loss after costs. Future fills on either
    # side of break-even must not change the already-known decision.
    for fill_open in (98.0, 102.0):
        records, summary, live_decision, _ = _replay(
            side, decision_close=decision_close, fill_open=fill_open,
        )
        assert live_decision is expected_exit
        assert summary.trades == int(expected_exit)
        if expected_exit:
            assert records[0].exit_reason == "opposite_signal"
            assert records[0].exit_price == (fill_open if side == "long" else 200-fill_open)
            assert (records[0].net_pnl > 0) is (fill_open > 100)
        else:
            assert records == []


@pytest.mark.parametrize("side", ["long", "short"])
def test_strong_reversal_can_close_known_loss_independently_of_future_gap(side):
    for fill_open in (98.0, 102.0):
        records, summary, live_decision, _ = _replay(
            side, decision_close=99, fill_open=fill_open, score=5,
        )
        assert live_decision is True
        assert summary.trades == 1
        assert records[0].exit_reason == "opposite_signal"
        assert records[0].exit_price == (fill_open if side == "long" else 200-fill_open)


@pytest.mark.parametrize("side", ["long", "short"])
def test_future_execution_gap_cannot_satisfy_current_reversal_minimum_hold(side):
    for execution_time in (DECISION_TIME, DECISION_TIME + 600_000):
        records, summary, live_decision, _ = _replay(
            side, minimum_hold=120, score=5, execution_time=execution_time,
        )
        assert live_decision is False
        assert records == []
        assert summary.trades == 0


@pytest.mark.parametrize("side", ["long", "short"])
def test_future_holding_cost_only_changes_realized_reversal_pnl(side):
    for execution_time in (DECISION_TIME, ENTRY_TIME + 8 * 3_600_000):
        records, summary, live_decision, risk = _replay(
            side, fill_open=101, execution_time=execution_time, funding_rate=0.02,
        )
        assert live_decision is True
        assert summary.trades == 1
        record = records[0]
        holding_hours = (execution_time - ENTRY_TIME) / 3_600_000
        assert record.exit_reason == "opposite_signal"
        assert record.exit_time == datetime.fromtimestamp(execution_time / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
        assert record.holding_minutes == pytest.approx(holding_hours * 60)
        expected_pnl = risk.estimate_net_pnl(
            side, record.entry_price, record.exit_price, record.quantity,
            holding_hours=holding_hours,
        )
        assert record.net_pnl == pytest.approx(expected_pnl)
        assert summary.final_equity == pytest.approx(summary.initial_equity + expected_pnl)
        assert (expected_pnl > 0) is (execution_time == DECISION_TIME)


@pytest.mark.parametrize("side", ["long", "short"])
def test_final_opposite_signal_has_no_invented_market_fill(side):
    records, summary, live_decision, _ = _replay(side, has_next=False)
    assert live_decision is True
    assert records == []
    assert summary.trades == 0


@pytest.mark.parametrize("side", ["long", "short"])
def test_reversal_loss_cooldown_starts_at_delayed_fill_not_earlier_decision(side):
    # An hour-long data gap delays the modeled fill. Fresh strong signals one
    # minute after that loss must not bypass its 15-minute entry cooldown.
    records, summary, live_decision, _ = _replay(
        side, execution_time=DECISION_TIME + 3_600_000, later_signals=True,
    )
    assert live_decision is True
    assert summary.trades == 1
    assert records[0].exit_reason == "opposite_signal"
    assert records[0].net_pnl < 0
