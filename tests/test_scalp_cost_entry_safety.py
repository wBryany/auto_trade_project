from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from btc_futures_bot.costs import CostConfig
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.entry_costs import evaluate_scalp_cost_coverage
from btc_futures_bot.models import Candle, Position
from btc_futures_bot.risk import RiskManager
from btc_futures_bot.strategy import MultiTimeframeStrategy, StrategyConfig


_SIGNAL_TIMESTAMP = 1_788_686_040_000
_NOW_MS = _SIGNAL_TIMESTAMP + 60_000


def _quiet_primary_market(side: str) -> dict[str, list[Candle]]:
    """Real indicators make a valid setup despite insufficient cost coverage."""
    closes = [100.0 + 0.003 * index + (0.012 if index % 2 == 0 else -0.012) for index in range(70)]
    market: dict[str, list[Candle]] = {}
    for timeframe, interval in (("1m", 60_000), ("5m", 300_000)):
        end = _SIGNAL_TIMESTAMP - 60_000 if timeframe == "1m" else (_NOW_MS // interval - 1) * interval
        candles = []
        for index, close in enumerate(closes):
            opening = closes[index - 1] if index else close
            candles.append(Candle(
                end - (len(closes) - index - 1) * interval,
                opening, max(opening, close) + 0.003, min(opening, close) - 0.003,
                close, 10.0,
            ))
        market[timeframe] = candles
    last = closes[-1]
    market["1m"].append(Candle(_SIGNAL_TIMESTAMP, last, last + 0.068, last - 0.003, last + 0.065, 15.0))
    if side == "short":
        market = {
            timeframe: [replace(
                candle, open=200.0 - candle.open, high=200.0 - candle.low,
                low=200.0 - candle.high, close=200.0 - candle.close,
            ) for candle in candles]
            for timeframe, candles in market.items()
        }
    return market


class _PublicCandleOnlyAdapter:
    name = "okx"
    settings = SimpleNamespace(symbol="BTC-USDT-SWAP", environment="demo")

    def __init__(self, closed: dict[str, list[Candle]]) -> None:
        self.calls: list[str] = []
        self.candles = {}
        for timeframe, interval in (("1m", 60_000), ("5m", 300_000)):
            last = closed[timeframe][-1]
            # This unfinished minute contains a huge wick. It must never be
            # visible to primary indicators or the historical cost windows.
            forming = Candle(last.timestamp + interval, last.close, 200.0, 1.0, last.close, 1_000_000.0)
            self.candles[timeframe] = list(closed[timeframe]) + [forming]

    def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
        self.calls.append(interval)
        return self.candles[interval][-limit:]


def _engine_and_market(side: str) -> tuple[TradingEngine, dict[str, list[Candle]], Mock]:
    market = _quiet_primary_market(side)
    costs = CostConfig(min_net_edge_pct=0.0015, expected_holding_hours=600 / 3600)
    config = StrategyConfig(
        mode="scalp_v2", trigger_timeframe="1m", regime_timeframe="5m",
        min_score=4, min_volume_ratio=1.2, scalp_cost_filter_enabled=True,
        scalp_cost_lookback_windows=6, enable_time_exit=True,
        max_hold_seconds=300, hard_max_hold_seconds=600,
        break_even_trigger_r=0.0, trailing_trigger_r=0.0,
        enable_adverse_dynamic_exit=False,
    )
    # Prove the fixture is not merely flat because some unrelated primary
    # condition failed: only the additional cost guard rejects this setup.
    primary = MultiTimeframeStrategy(replace(config, scalp_cost_filter_enabled=False), costs=costs)
    assert primary.evaluate(market).side == side
    strategy = MultiTimeframeStrategy(config, costs=costs)
    assert strategy.evaluate(market).reasons[0] == "scalp_v2_cost_blocked"
    gate_call = Mock(side_effect=AssertionError("cost-blocked primary must not reach entry gate"))
    engine = TradingEngine(
        _PublicCandleOnlyAdapter(market), strategy, RiskManager(costs=costs),
        EngineConfig(mode="paper", candle_limit=100),
        entry_gate=SimpleNamespace(config=SimpleNamespace(mode="off"), evaluate=gate_call),
    )
    return engine, market, gate_call


@pytest.mark.parametrize("side", ["long", "short"])
def test_forming_volatility_cannot_change_closed_history_cost_admission(side: str) -> None:
    engine, market, gate_call = _engine_and_market(side)
    expected = engine.strategy.evaluate(market)
    with patch("btc_futures_bot.engine.time.time", return_value=_NOW_MS / 1000), patch(
        "btc_futures_bot.strategy.evaluate_scalp_cost_coverage", wraps=evaluate_scalp_cost_coverage,
    ) as coverage:
        result = engine.evaluate_once()

    assert result.status == "no_action"
    assert result.signal == expected
    assert result.signal.side == "flat"
    assert result.signal.reasons[0] == "scalp_v2_cost_blocked"
    assert engine.position is None
    assert engine.adapter.calls == ["1m", "5m"]
    coverage.assert_called_once()
    observed_candles = coverage.call_args.args[0]
    assert observed_candles == market["1m"]
    assert observed_candles[-1].timestamp == _SIGNAL_TIMESTAMP
    assert all(candle.timestamp < _NOW_MS for candle in observed_candles)
    gate_call.assert_not_called()


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("exit_reason", ["hard_time_exit", "stop_loss"])
def test_cost_blocked_primary_cannot_suppress_existing_position_exit(side: str, exit_reason: str) -> None:
    engine, market, gate_call = _engine_and_market(side)
    candle = market["1m"][-1]
    entry = candle.close * (1.0004 if side == "long" else 0.9996)
    if exit_reason == "stop_loss":
        stop = (candle.low + candle.close) / 2 if side == "long" else (candle.high + candle.close) / 2
        expected_exit_price = stop
    else:
        stop = entry * (0.99 if side == "long" else 1.01)
        expected_exit_price = candle.close
    target = entry * (1.02 if side == "long" else 0.98)
    engine.position = Position(
        side, 1.0, entry, stop, target, _NOW_MS - 600_000, initial_stop_price=stop,
    )
    engine.last_position_candle_timestamp = candle.timestamp - 60_000
    with patch("btc_futures_bot.engine.time.time", return_value=_NOW_MS / 1000), patch.object(
        engine, "_close_paper_position", wraps=engine._close_paper_position,
    ) as close_position:
        result = engine.evaluate_once()

    close_position.assert_called_once_with(expected_exit_price, exit_reason)
    assert result.signal.side == "flat"
    assert result.signal.reasons[0] == "scalp_v2_cost_blocked"
    assert engine.position is None
    assert result.position is None
    assert engine.session_pnl < 0
    assert engine.consecutive_losses == 1
    gate_call.assert_not_called()
