from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from btc_futures_bot.backtest import run_backtest
from btc_futures_bot.costs import CostConfig
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.main import build_engine
from btc_futures_bot.models import Position, Signal
from btc_futures_bot.risk import RiskManager
from btc_futures_bot.strategy import MultiTimeframeStrategy, StrategyConfig


def _strategy(costs=None, **kwargs):
    return MultiTimeframeStrategy(StrategyConfig(
        mode="scalp_v2", trigger_timeframe="1m", regime_timeframe="5m",
        scalp_cost_filter_enabled=True, hard_max_hold_seconds=600, **kwargs,
    ), costs=costs)


def test_runtime_and_backtest_refuse_mismatched_admission_costs(tmp_path):
    strategy = _strategy()
    risk = RiskManager(costs=replace(strategy.costs, taker_fee_pct=.001))
    with pytest.raises(ValueError, match="identical costs"):
        TradingEngine(SimpleNamespace(), strategy, risk, EngineConfig())
    with pytest.raises(ValueError, match="identical costs"):
        run_backtest(tmp_path, strategy=strategy, risk=risk)


def test_runtime_and_backtest_refuse_impossible_history_window(tmp_path):
    strategy = _strategy()
    risk = RiskManager(costs=strategy.costs)
    with pytest.raises(ValueError, match="enough closed bars"):
        TradingEngine(SimpleNamespace(), strategy, risk, EngineConfig(candle_limit=60))
    with pytest.raises(ValueError, match="enough closed bars"):
        run_backtest(tmp_path, strategy=strategy, risk=risk, candle_limit=60)


def test_build_engine_injects_selected_exchange_costs(tmp_path):
    raw = {
        "mode": "paper", "active_exchange": "okx", "report_dir": str(tmp_path),
        "exchanges": {"okx": {"costs": {"taker_fee_pct": .0008, "slippage_pct": .0003}}},
        "costs": {"taker_fee_pct": .0001},
        "strategy": {"mode": "scalp_v2", "trigger_timeframe": "1m", "regime_timeframe": "5m",
                     "hard_max_hold_seconds": 600, "scalp_cost_filter_enabled": True},
    }
    with patch("btc_futures_bot.main.make_adapter", return_value=SimpleNamespace()), patch(
        "btc_futures_bot.main.EntryGate", return_value=None,
    ):
        engine = build_engine("okx", raw)
    assert engine.strategy.costs == engine.risk.costs
    assert engine.strategy.costs.taker_fee_pct == .0008
    assert engine.strategy.costs.slippage_pct == .0003


@pytest.mark.parametrize("side,opposite,price", [("long", "short", 100.05), ("short", "long", 99.95)])
def test_ordinary_opposite_scalp_does_not_churn_a_net_loser(side, opposite, price):
    strategy = _strategy()
    engine = TradingEngine(SimpleNamespace(), strategy, RiskManager(costs=strategy.costs), EngineConfig())
    position = Position(side, 1, 100, 99 if side == "long" else 101, 102, 1_000_000)
    with patch("btc_futures_bot.engine.time.time", return_value=1300):
        assert not engine._should_reverse(position, price, Signal(opposite, 5, 0))
        assert not engine._should_reverse(position, price, Signal(opposite, 99, 0))
        profitable = 100.3 if side == "long" else 99.7
        assert engine._should_reverse(position, profitable, Signal(opposite, 5, 0))


def test_legacy_strong_reversal_is_unchanged():
    strategy = MultiTimeframeStrategy(StrategyConfig(mode="traditional_kline"))
    engine = TradingEngine(SimpleNamespace(), strategy, RiskManager(), EngineConfig())
    position = Position("long", 1, 100, 99, 102, 1_000_000)
    with patch("btc_futures_bot.engine.time.time", return_value=1300):
        assert engine._should_reverse(position, 100, Signal("short", 5, 0))
