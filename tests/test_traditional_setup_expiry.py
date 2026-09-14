"""Setup expiry must use the selected path's window without losing fresh signals."""
from dataclasses import replace

import pytest

from btc_futures_bot.models import Candle
from btc_futures_bot.strategy import MultiTimeframeStrategy, StrategyConfig


def _market_and_config(side: str) -> tuple[dict[str, list[Candle]], StrategyConfig]:
    # Adapt the existing strategy persistence fixture: a real EMA cross occurs
    # on the penultimate 5m candle, then the latest candle adds no new setup.
    def candles(values: list[float], interval: int) -> list[Candle]:
        return [
            Candle(i * interval, close - 0.1, close + 0.2, close - 0.2, close, 10.0)
            for i, close in enumerate(values)
        ]

    trigger = candles([105 - i * 0.12 for i in range(39)] + [104.0, 104.2], 300_000)
    trigger[-2] = Candle(trigger[-2].timestamp, 103.8, 104.4, 103.7, 104.0, 25.0)
    trigger[-1] = Candle(trigger[-1].timestamp, 104.0, 104.4, 103.9, 104.2, 25.0)
    market = {
        "5m": trigger,
        "1m": candles([100 + i * 0.05 for i in range(41)], 60_000),
        "1h": candles([100 + i * 0.2 for i in range(41)], 3_600_000),
    }
    if side == "short":
        market = {
            frame: [
                replace(c, open=200-c.open, high=200-c.low, low=200-c.high, close=200-c.close)
                for c in series
            ]
            for frame, series in market.items()
        }
    config = StrategyConfig(
        mode="traditional_kline", trigger_timeframe="5m", regime_timeframe="1h",
        traditional_trend_fast=5, traditional_trend_slow=20,
        traditional_signal_fast=3, traditional_signal_slow=5,
        traditional_rsi_period=3, traditional_macd_fast=3,
        traditional_macd_slow=6, traditional_macd_signal=2,
        traditional_volume_sma_period=3, traditional_min_volume_ratio=1.1,
        traditional_rsi_long_min=0, traditional_rsi_long_max=100,
        traditional_rsi_short_min=0, traditional_rsi_short_max=100,
        traditional_allow_pullback=False, traditional_allow_breakout=False,
    )
    return market, config


@pytest.mark.parametrize("side", ["long", "short"])
def test_normal_one_bar_window_expires_old_setup_but_explicit_two_bars_preserve_it(side):
    market, config = _market_and_config(side)
    # The macro-recheck window must not accidentally broaden normal entries.
    one = replace(config, traditional_setup_valid_bars=1, traditional_blocked_setup_valid_bars=2)
    expired = MultiTimeframeStrategy(one).evaluate(market)
    persisted = MultiTimeframeStrategy(replace(one, traditional_setup_valid_bars=2)).evaluate(market)

    assert expired.side == "flat"
    assert f"{side}_missing=5m_entry_setup" in expired.reasons
    assert persisted.side == side
    assert persisted.timestamp == market["5m"][-1].timestamp
    assert "5m_setup_persisted_1_bars" in persisted.reasons
    assert f"5m_{'golden' if side == 'long' else 'death'}_cross" in persisted.reasons
    assert "macro_blocked_signal_revalidated" not in persisted.reasons


@pytest.mark.parametrize("side", ["long", "short"])
def test_macro_recheck_one_bar_window_expires_old_setup_independently_of_normal_window(side):
    market, config = _market_and_config(side)
    one = replace(config, traditional_setup_valid_bars=2, traditional_blocked_setup_valid_bars=1)
    strategy = MultiTimeframeStrategy(one)
    expired = strategy.reevaluate_blocked_signal(side, market)
    compatible = MultiTimeframeStrategy(replace(one, traditional_blocked_setup_valid_bars=2)).reevaluate_blocked_signal(side, market)

    assert strategy.evaluate(market).side == side
    assert expired.side == "flat"
    assert "blocked_signal_expired_or_invalid" in expired.reasons
    assert compatible.side == side
    assert "5m_setup_persisted_1_bars" in compatible.reasons
    assert "macro_blocked_signal_revalidated" in compatible.reasons


@pytest.mark.parametrize("side", ["long", "short"])
def test_latest_valid_setup_is_preserved_in_normal_and_macro_recheck_paths(side):
    market, config = _market_and_config(side)
    # Cut off the final 5m candle: the same cross is now the latest valid setup.
    fresh = {**market, "5m": market["5m"][:-1]}
    one = MultiTimeframeStrategy(replace(config, traditional_setup_valid_bars=1, traditional_blocked_setup_valid_bars=1))
    two = MultiTimeframeStrategy(replace(config, traditional_setup_valid_bars=2, traditional_blocked_setup_valid_bars=2))
    signal = one.evaluate(fresh)
    rechecked = one.reevaluate_blocked_signal(side, fresh)

    assert signal.side == side
    assert signal == two.evaluate(fresh)
    assert rechecked == two.reevaluate_blocked_signal(side, fresh)
    assert rechecked.side == signal.side
    assert rechecked.timestamp == signal.timestamp == fresh["5m"][-1].timestamp
    assert rechecked.reasons == signal.reasons + ("macro_blocked_signal_revalidated",)
    assert not any("setup_persisted" in reason for reason in signal.reasons)
