"""Regression coverage for strict, existing 5m reversal momentum confirmation."""
from dataclasses import asdict, replace

import pytest

from btc_futures_bot.models import Candle
from btc_futures_bot.strategy import (
    MultiTimeframeStrategy,
    StrategyConfig,
    _TraditionalFeatures,
    _traditional_ultra_short_reversal_one_minute_trigger,
)


def observed_reversal(side):
    # The 16:15 CST, 2026-09-09 decision's observed closed-bar features and
    # 12 local pivot bars. Earlier flat bars only warm up the pivot ATR;
    # full signal reconstruction is separately checked against frozen data.
    local_bars = [
        (79371.3, 79400.1, 79333.4, 79333.4, 199.265),
        (79333.4, 79400.0, 79293.3, 79399.9, 154.096),
        (79399.9, 79407.9, 79387.2, 79387.2, 51.685),
        (79387.2, 79387.3, 79350.0, 79371.8, 49.986),
        (79371.8, 79391.3, 79366.6, 79374.8, 73.929),
        (79374.9, 79416.0, 79374.8, 79401.6, 73.502),
        (79401.7, 79490.5, 79401.7, 79436.3, 393.421),
        (79436.3, 79488.0, 79436.2, 79483.6, 216.495),
        (79483.7, 79543.3, 79462.7, 79480.8, 383.996),
        (79480.0, 79480.1, 79436.5, 79472.9, 93.693),
        (79472.9, 79473.0, 79446.6, 79446.7, 41.287),
        (79446.6, 79446.7, 79377.9, 79377.9, 167.014),
    ]
    rows = [(79350.0, 79375.0, 79325.0, 79350.0, 100.0)] * 24 + local_bars
    candles = [Candle(i * 60_000, *row) for i, row in enumerate(rows)]
    execution = _TraditionalFeatures(
        open=79446.6, high=79446.7, low=79377.9, close=79377.9,
        previous_close=79446.7, ema_fast=79417.4687881909,
        previous_ema_fast=79427.36098523863, ema_slow=79363.74432776391,
        previous_ema_slow=79362.3287605403, rsi=55.1002440990457,
        macd_histogram=3.3243994669118635, previous_macd_histogram=11.36333864357622,
        atr=45.856449647441934, volume_ratio=1.249574585933857,
    )
    trigger = _TraditionalFeatures(
        open=79436.3, high=79543.3, low=79377.9, close=79377.9,
        previous_close=79436.3, ema_fast=79301.27301405865,
        previous_ema_fast=79282.11626757332, ema_slow=79224.19722727004,
        previous_ema_slow=79208.82694999705, rsi=64.40057860047584,
        macd_histogram=13.52600296609539, previous_macd_histogram=13.11579151693985,
        atr=98.30461341839701, volume_ratio=2.813351870777439,
    )
    if side == "long":
        def mirror(feature):
            values = asdict(feature)
            for name in ("open", "close", "previous_close", "ema_fast", "ema_slow",
                         "previous_ema_fast", "previous_ema_slow"):
                values[name] = 160_000 - values[name]
            values["high"], values["low"] = 160_000 - feature.low, 160_000 - feature.high
            values["rsi"] = 100 - feature.rsi
            values["macd_histogram"] = -feature.macd_histogram
            values["previous_macd_histogram"] = -feature.previous_macd_histogram
            return _TraditionalFeatures(**values)

        candles = [Candle(c.timestamp, 160_000-c.open, 160_000-c.low,
                          160_000-c.high, 160_000-c.close, c.volume) for c in candles]
        execution, trigger = mirror(execution), mirror(trigger)
    return candles, execution, trigger


def strict_config():
    return StrategyConfig(
        traditional_ultra_short_reversal_momentum_guard_enabled=True,
        traditional_ultra_short_reversal_max_adverse_macd_atr=0.0,
        traditional_ultra_short_reversal_min_liquidity_volume_ratio=0.65,
        traditional_ultra_short_reversal_5m_min_volume_ratio=0.8,
    )


@pytest.mark.parametrize("side", ["short", "long"])
def test_strict_guard_rejects_observed_small_5m_acceleration_and_its_mirror(side):
    candles, execution, trigger = observed_reversal(side)
    strict = strict_config()
    previous_policy = replace(strict, traditional_ultra_short_reversal_max_adverse_macd_atr=0.05)
    # Elevated/depressed 5m RSI permits the early reversal context even while
    # old-direction momentum grows. The dedicated trigger must still veto it.
    assert (trigger.rsi >= 58) if side == "short" else (trigger.rsi <= 44)
    assert _traditional_ultra_short_reversal_one_minute_trigger(
        candles, execution, trigger, side, previous_policy,
    )
    assert not _traditional_ultra_short_reversal_one_minute_trigger(
        candles, execution, trigger, side, strict,
    )


@pytest.mark.parametrize("side", ["short", "long"])
@pytest.mark.parametrize("weakening", [0.0, 0.4])
def test_strict_guard_allows_stalled_or_weakening_5m_without_waiting_for_zero_cross(side, weakening):
    candles, execution, trigger = observed_reversal(side)
    direction = 1 if side == "short" else -1
    trigger = replace(
        trigger, macd_histogram=trigger.previous_macd_histogram - direction * weakening,
    )
    assert trigger.macd_histogram * direction > 0
    assert _traditional_ultra_short_reversal_one_minute_trigger(
        candles, execution, trigger, side, strict_config(),
    )


@pytest.mark.parametrize("side", ["short", "long"])
def test_disabled_guard_retains_existing_early_reversal_behavior(side):
    candles, execution, trigger = observed_reversal(side)
    disabled = replace(strict_config(), traditional_ultra_short_reversal_momentum_guard_enabled=False)
    assert _traditional_ultra_short_reversal_one_minute_trigger(
        candles, execution, trigger, side, disabled,
    )


@pytest.mark.parametrize("side", ["short", "long"])
def test_weakening_5m_does_not_bypass_one_minute_confirmation_or_liquidity(side):
    candles, execution, trigger = observed_reversal(side)
    trigger = replace(trigger, macd_histogram=trigger.previous_macd_histogram)
    no_momentum_confirmation = replace(execution, macd_histogram=execution.previous_macd_histogram)
    assert not _traditional_ultra_short_reversal_one_minute_trigger(
        candles, no_momentum_confirmation, trigger, side, strict_config(),
    )
    assert not _traditional_ultra_short_reversal_one_minute_trigger(
        candles, replace(execution, volume_ratio=0.4),
        replace(trigger, volume_ratio=0.5), side, strict_config(),
    )


def test_strict_reversal_guard_leaves_ultra_short_continuation_long_unchanged():
    def candles(values, interval):
        return [Candle(i*interval, value-0.1, value+0.2, value-0.2, value, 10)
                for i, value in enumerate(values)]

    values = [100 + i*i*0.001 for i in range(48)]
    execution = candles([values[-1]-0.4 + i*0.005 for i in range(39)], 60_000)
    execution.append(Candle(39*60_000, execution[-1].close, values[-1]+0.12,
                            execution[-1].close-0.02, values[-1]+0.1, 40))
    market = {"5m": candles(values, 300_000), "1m": execution,
              "1h": candles([100.0]*40, 3_600_000)}
    base = StrategyConfig(
        mode="traditional_kline", trigger_timeframe="5m", regime_timeframe="1h",
        traditional_trend_fast=5, traditional_trend_slow=20,
        traditional_signal_fast=3, traditional_signal_slow=5,
        traditional_rsi_period=3, traditional_macd_fast=3,
        traditional_macd_slow=6, traditional_macd_signal=2,
        traditional_volume_sma_period=3, traditional_min_volume_ratio=1.1,
        traditional_rsi_long_min=0, traditional_rsi_long_max=100,
        traditional_ultra_short_enabled=True,
        traditional_ultra_short_reversal_enabled=True,
        traditional_ultra_short_1m_max_extension_atr=10.0,
        traditional_ultra_short_reversal_momentum_guard_enabled=True,
        traditional_ultra_short_reversal_max_adverse_macd_atr=0.05,
    )
    original = MultiTimeframeStrategy(base).evaluate(market)
    strict = MultiTimeframeStrategy(replace(
        base, traditional_ultra_short_reversal_max_adverse_macd_atr=0.0,
    )).evaluate(market)
    assert original.side == "long"
    assert "1m_ultra_short_trigger_long" in original.reasons
    assert strict == original
