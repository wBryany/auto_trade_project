from dataclasses import replace
from types import SimpleNamespace

import pytest

from btc_futures_bot.costs import CostConfig
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.models import Candle, Position, Signal, TradeResult
from btc_futures_bot.risk import RiskConfig, RiskManager
from btc_futures_bot.strategy import StrategyConfig


def bars(spread=0.2):
    return [Candle(i * 60_000, 100, 100 + spread / 2, 100 - spread / 2, 100, 10) for i in range(60)]


def manager(minutes=60):
    return RiskManager(RiskConfig(entry_range_lookback_minutes=minutes), costs=CostConfig(
        taker_fee_pct=0.0005, slippage_pct=0.0002,
        min_net_edge_pct=0.0015, expected_holding_hours=0.1,
    ))


def test_costs_and_configured_net_edge_must_fit_actual_range():
    assert not manager().observed_range_allows_entry(bars(.28), 100)
    assert manager().observed_range_allows_entry(bars(.30), 100)
    # A wide historical candle outside the lookback cannot admit a quiet market.
    assert not manager().observed_range_allows_entry([replace(bars()[0], high=200)] + bars(), 100)
    assert manager(0).observed_range_allows_entry([], 0)


@pytest.mark.parametrize('kind', ['short', 'gap', 'nan', 'invalid_price'])
def test_incomplete_or_invalid_observations_fail_closed(kind):
    candles = bars(.5)
    price = 100
    if kind == 'short': candles.pop()
    if kind == 'gap': candles[20] = replace(candles[20], timestamp=1)
    if kind == 'nan': candles[20] = replace(candles[20], high=float('nan'))
    if kind == 'invalid_price': price = 0
    assert not manager().observed_range_allows_entry(candles, price)


def test_live_entry_gate_ignores_forming_candle_and_avoids_private_order_calls():
    class Adapter:
        name = 'binance'
        settings = SimpleNamespace(symbol='BTCUSDT', environment='production')

        def fetch_candles(self, interval, limit):
            return bars() + [Candle(60 * 60_000, 100, 200, 1, 100, 10)]

        def fetch_live_position(self):
            return None

        def fetch_equity(self):
            raise AssertionError('entry filter must run before private entry calls')

    class Strategy:
        config = StrategyConfig(trigger_timeframe='5m', regime_timeframe='1h')

        def evaluate(self, candles):
            return Signal('long', 6, 60 * 60_000, ('fixed',))

    engine = TradingEngine(Adapter(), Strategy(), manager(), EngineConfig(mode='live'))
    result = engine.evaluate_once()
    assert result.status == 'insufficient_market_range'
    assert engine.position is None
    assert engine.evaluate_once().status == 'no_action'


def test_range_gate_does_not_prevent_live_position_exit(monkeypatch):
    adapter = SimpleNamespace(
        name='binance',
        fetch_candles=lambda *args: bars() + [bars()[-1]],
        fetch_mark_price=lambda: 99.0,
    )
    strategy = SimpleNamespace(
        config=StrategyConfig(trigger_timeframe='5m', regime_timeframe='1h'),
        evaluate=lambda _: Signal('long', 6, 1, ('fixed',)),
    )
    engine = TradingEngine(adapter, strategy, manager(), EngineConfig(mode='live'))
    engine.position = Position('long', 1, 100, 99, 103, 0)
    monkeypatch.setattr(engine, '_reconcile_binance_live_position_if_due', lambda _: None)
    monkeypatch.setattr(engine, '_manage_live_position', lambda *args: TradeResult('binance', 'live_active_exit'))
    def unexpected_gate(*args):
        raise AssertionError('position exit must precede any entry filter')
    monkeypatch.setattr(engine.risk, 'observed_range_allows_entry', unexpected_gate)
    assert engine.evaluate_once().status == 'live_active_exit'
