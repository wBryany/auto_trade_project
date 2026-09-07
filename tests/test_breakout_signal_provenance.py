from dataclasses import replace
from unittest.mock import patch

import pytest

from btc_futures_bot.models import Candle, Position
from btc_futures_bot.strategy import (
    MultiTimeframeStrategy,
    StrategyConfig,
    _TraditionalFeatures,
    _TraditionalSetupState,
    breakout_failure_exit_reason,
)


def _entry_signal(side, path, source, enabled=True):
    """Exercise real entry routing with fixed, independently qualified features."""
    current = _TraditionalFeatures(
        open=101.0, high=101.4, low=100.9, close=101.2,
        previous_close=101.0, ema_fast=100.7, previous_ema_fast=100.6,
        ema_slow=100.2, previous_ema_slow=100.1, rsi=58.0,
        macd_histogram=1.0, previous_macd_histogram=0.5,
        atr=1.0, volume_ratio=2.0,
    )
    previous = replace(
        current, open=100.2, high=101.2, low=100.0, close=101.0,
        ema_fast=100.5, volume_ratio=0.5 if source == "volume" else 2.0,
        macd_histogram=-0.1 if source == "macd" else 0.5,
    )
    regime = replace(
        current, close=100.8 if path == "structural" else 101.2,
        ema_fast=101.0, ema_slow=100.0,
        rsi=42.0 if path == "structural" else 58.0,
    )
    empty = _TraditionalSetupState(False, False, False, False, False, False, False, False)
    breakout = replace(empty, breakout_long_raw=True, breakout_long=True, breakout_long_level=100)
    current_setup = empty if source in {"volume", "macd", "persisted"} else breakout
    previous_setup = replace(breakout, breakout_long=False) if source == "volume" else breakout
    if source == "cross":
        # A rejected raw breakout is not the reason a qualified cross entered.
        current_setup = replace(breakout, golden_cross=True, breakout_long=False)
    if side == "short":
        def mirror(feature):
            return replace(
                feature, open=200-feature.open, high=200-feature.low,
                low=200-feature.high, close=200-feature.close,
                previous_close=200-feature.previous_close,
                ema_fast=200-feature.ema_fast, previous_ema_fast=200-feature.previous_ema_fast,
                ema_slow=200-feature.ema_slow, previous_ema_slow=200-feature.previous_ema_slow,
                rsi=100-feature.rsi, macd_histogram=-feature.macd_histogram,
                previous_macd_histogram=-feature.previous_macd_histogram,
            )
        current, previous, regime = map(mirror, (current, previous, regime))
        def mirror_setup(setup):
            return replace(
                empty, death_cross=setup.golden_cross,
                breakout_short_raw=setup.breakout_long_raw,
                breakout_short=setup.breakout_long,
                breakout_short_level=setup.breakout_long_level,
            )
        current_setup, previous_setup = map(mirror_setup, (current_setup, previous_setup))
    config = StrategyConfig(
        mode="traditional_kline", trigger_timeframe="5m", regime_timeframe="1h",
        traditional_allow_early_regime=False, traditional_allow_pullback=False,
        traditional_setup_valid_bars=1 if source == "macd" else 2,
        traditional_setup_macd_handoff_max_extension_atr=1.5,
        traditional_setup_volume_handoff_max_extension_atr=0.8,
        traditional_structural_scalp_enabled=True,
        traditional_invalidate_failed_breakouts=True,
        enable_breakout_failure_exit=enabled,
    )
    candles = [Candle(i*300_000, current.open, current.high, current.low, current.close, 10) for i in range(240)]
    with patch("btc_futures_bot.strategy._traditional_features", side_effect=[current, current, regime, previous]), patch(
        "btc_futures_bot.strategy._traditional_setup_state",
        side_effect=[current_setup, previous_setup, previous_setup],
    ):
        signal = MultiTimeframeStrategy(config).evaluate({"5m": candles, "1m": candles, "1h": candles})
    assert signal.side == side
    if path == "structural":
        assert f"1h_structural_scalp_recovery_{side}" in signal.reasons
        assert signal.score == 6
    else:
        assert signal.score == 7
    return signal, config


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("path", ["regular", "structural"])
@pytest.mark.parametrize("source", ["breakout", "persisted", "volume", "macd"])
def test_breakout_derived_entries_preserve_original_level_and_enable_failure_exit(side, path, source):
    signal, config = _entry_signal(side, path, source)
    marker = "5m_breakout" if side == "long" else "5m_breakdown"
    assert signal.reasons.count(marker) == 1
    assert signal.reasons.count("breakout_level=100") == 1
    # Feed the emitted signal into the real exit rule, including actual 1m
    # EMA/MACD calculations. No hand-authored signal metadata bypasses routing.
    adverse_price = 99.0 if side == "long" else 101.0
    closes = [102.0]*42 + [101.8-i*0.4 for i in range(8)]
    if side == "short":
        closes = [200-price for price in closes]
    minutes = [
        Candle((i-35)*60_000, price, price+0.1, price-0.1, price, 10)
        for i, price in enumerate(closes)
    ]
    bars = [Candle(t, adverse_price, adverse_price+0.1, adverse_price-0.1, adverse_price, 10) for t in (300_000, 600_000)]
    position = Position(side, 1, 101.2 if side == "long" else 98.8, 90 if side == "long" else 110, 120, 0)
    assert breakout_failure_exit_reason(position, signal, {"5m": bars, "1m": minutes}, config, adverse_price, 900_000) == "breakout_failure"


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("path", ["regular", "structural"])
def test_independent_cross_does_not_inherit_rejected_raw_breakout(side, path):
    signal, _ = _entry_signal(side, path, "cross")
    assert not any(reason.startswith("breakout_level=") for reason in signal.reasons)
    assert "5m_breakout" not in signal.reasons
    assert "5m_breakdown" not in signal.reasons


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("path", ["regular", "structural"])
@pytest.mark.parametrize("source", ["volume", "macd"])
def test_disabled_breakout_exit_keeps_legacy_handoff_signals(side, path, source):
    signal, _ = _entry_signal(side, path, source, enabled=False)
    assert not any(reason.startswith("breakout_level=") for reason in signal.reasons)
    assert "5m_breakout" not in signal.reasons
    assert "5m_breakdown" not in signal.reasons
