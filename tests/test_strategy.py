from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from btc_futures_bot.exchanges.base import ExchangeSettings
from btc_futures_bot.exchanges.okx import OkxAdapter
from btc_futures_bot.backtest import _tighten_position_stop
from btc_futures_bot.costs import CostConfig
from btc_futures_bot.http_client import ApiError
from btc_futures_bot.indicators import ema, rsi
from btc_futures_bot.models import Candle, Position, Signal
from btc_futures_bot.risk import RiskConfig, RiskManager
from btc_futures_bot.strategy import (
    MultiTimeframeStrategy,
    StrategyConfig,
    _TraditionalFeatures,
    _TraditionalSetupState,
    _aggregate_five_minute_candles,
    _traditional_countertrend_cross_regime,
    _traditional_countertrend_pullback_regime,
    _traditional_cross_quality,
    _traditional_execution_quality,
    _traditional_failed_breakout_short_reversal,
    _traditional_neutral_transition_regime,
    _traditional_predictive_reversal_short,
    _traditional_pressure_room,
    _traditional_structural_scalp_regime,
    _traditional_strong_regime_quality,
    _traditional_ultra_short_one_minute_trigger,
    _traditional_setup_macd_handoff,
    _traditional_setup_volume_handoff,
    dynamic_stop_loss_pct,
    signal_position_size_multiplier,
    signal_stop_loss_overrides,
    signal_stop_timeframe,
    signal_trade_management_overrides,
)


def test_ultra_short_one_minute_trigger_requires_fresh_volume_break() -> None:
    candles = [
        Candle(index * 60_000, 99.95, 100.10, 99.90, 100.0, 10.0)
        for index in range(30)
    ]
    candles[-1] = Candle(candles[-1].timestamp, 100.20, 101.10, 100.10, 101.0, 25.0)
    execution = _TraditionalFeatures(
        open=100.20,
        high=101.10,
        low=100.10,
        close=101.0,
        previous_close=100.0,
        ema_fast=100.8,
        previous_ema_fast=100.0,
        ema_slow=100.5,
        previous_ema_slow=100.0,
        rsi=60.0,
        macd_histogram=1.0,
        previous_macd_histogram=0.2,
        atr=0.5,
        volume_ratio=1.5,
    )
    trigger = replace(execution, close=100.8, ema_fast=100.5, atr=1.0)
    config = StrategyConfig(
        traditional_signal_fast=3,
        traditional_signal_slow=5,
        traditional_rsi_period=3,
        traditional_macd_fast=3,
        traditional_macd_slow=6,
        traditional_macd_signal=2,
        traditional_atr_period=3,
        traditional_volume_sma_period=5,
        traditional_ultra_short_1m_lookback=5,
        traditional_ultra_short_1m_min_volume_ratio=1.2,
        traditional_ultra_short_1m_max_extension_atr=1.0,
    )

    assert _traditional_ultra_short_one_minute_trigger(
        candles,
        execution,
        trigger,
        "long",
        config,
    )
    assert not _traditional_ultra_short_one_minute_trigger(
        candles,
        replace(execution, volume_ratio=1.19),
        trigger,
        "long",
        config,
    )
    signal = Signal("long", 6, 1, ("1m_ultra_short_trigger_long",))
    assert signal_position_size_multiplier(signal) == 0.5
    assert signal_trade_management_overrides(signal, config) == {
        "break_even_trigger_r": 0.7,
        "break_even_lock_r": 0.15,
        "trailing_trigger_r": 0.9,
        "trailing_distance_r": 0.35,
    }


def test_structural_scalp_uses_hourly_context_without_waiving_safety_gates() -> None:
    long_regime = _TraditionalFeatures(
        open=100.0,
        high=101.0,
        low=98.0,
        close=99.2,
        previous_close=98.9,
        ema_fast=100.0,
        previous_ema_fast=100.1,
        ema_slow=97.5,
        previous_ema_slow=97.4,
        rsi=42.0,
        macd_histogram=0.2,
        previous_macd_histogram=0.1,
        atr=1.0,
        volume_ratio=1.0,
    )
    config = StrategyConfig(traditional_structural_scalp_enabled=True)

    assert _traditional_structural_scalp_regime(long_regime, "long", config)
    assert not _traditional_structural_scalp_regime(
        replace(long_regime, close=98.9, ema_fast=100.0),
        "long",
        replace(config, traditional_structural_scalp_max_fast_ema_distance_pct=0.01),
    )
    assert not _traditional_structural_scalp_regime(
        replace(long_regime, macd_histogram=-0.1),
        "long",
        config,
    )
    assert not _traditional_structural_scalp_regime(
        long_regime,
        "long",
        replace(config, traditional_structural_scalp_enabled=False),
    )

    short_regime = replace(
        long_regime,
        close=100.8,
        ema_fast=100.0,
        ema_slow=102.5,
        rsi=58.0,
        macd_histogram=-0.2,
    )
    assert _traditional_structural_scalp_regime(short_regime, "short", config)
    assert not _traditional_structural_scalp_regime(
        replace(short_regime, rsi=51.27),
        "short",
        config,
    )

    signal = Signal("long", 6, 1, ("1h_structural_scalp_recovery_long",))
    assert signal_position_size_multiplier(signal) == 0.5
    assert signal_stop_loss_overrides(signal, config) == {
        "structure_lookback_bars": 2,
        "maximum_stop_loss_pct": 0.006,
    }
    assert signal_trade_management_overrides(signal, config) == {
        "break_even_trigger_r": 1.0,
        "break_even_lock_r": 0.15,
        "trailing_trigger_r": 1.5,
        "trailing_distance_r": 0.5,
    }


def test_traditional_strong_regime_quality_rejects_stale_ema_ordering() -> None:
    feature = _TraditionalFeatures(
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        previous_close=100.4,
        ema_fast=100.2,
        previous_ema_fast=100.3,
        ema_slow=99.9,
        previous_ema_slow=99.8,
        rsi=55.0,
        macd_histogram=-0.1,
        previous_macd_histogram=0.1,
        atr=1.0,
        volume_ratio=1.2,
    )

    assert _traditional_strong_regime_quality(feature, "long", StrategyConfig())
    assert not _traditional_strong_regime_quality(
        feature,
        "long",
        StrategyConfig(
            traditional_strong_regime_min_gap_atr=0.5,
            traditional_strong_regime_require_fast_slope=True,
            traditional_strong_regime_require_macd=True,
        ),
    )


def test_traditional_cross_quality_rejects_weak_wick_close() -> None:
    feature = _TraditionalFeatures(
        open=100.0,
        high=102.0,
        low=99.0,
        close=100.4,
        previous_close=100.0,
        ema_fast=100.1,
        previous_ema_fast=99.9,
        ema_slow=100.0,
        previous_ema_slow=100.0,
        rsi=55.0,
        macd_histogram=0.1,
        previous_macd_histogram=-0.1,
        atr=1.0,
        volume_ratio=1.2,
    )

    assert _traditional_cross_quality(feature, "long", StrategyConfig())
    assert not _traditional_cross_quality(
        feature,
        "long",
        StrategyConfig(
            traditional_cross_min_body_ratio=0.5,
            traditional_cross_min_close_location=0.7,
        ),
    )
    assert not _traditional_cross_quality(
        feature,
        "long",
        StrategyConfig(traditional_cross_max_extension_atr=0.2),
    )


def test_one_minute_execution_quality_rejects_overheated_or_extended_entry() -> None:
    feature = _TraditionalFeatures(
        open=100.0,
        high=101.2,
        low=99.9,
        close=101.0,
        previous_close=100.0,
        ema_fast=100.0,
        previous_ema_fast=99.9,
        ema_slow=99.8,
        previous_ema_slow=99.7,
        rsi=79.0,
        macd_histogram=1.0,
        previous_macd_histogram=0.5,
        atr=0.5,
        volume_ratio=2.0,
    )
    config = StrategyConfig(
        traditional_execution_rsi_long_max=72.0,
        traditional_execution_rsi_short_min=28.0,
        traditional_execution_max_extension_atr=1.0,
    )

    assert not _traditional_execution_quality(feature, "long", config)
    assert not _traditional_execution_quality(
        replace(feature, close=99.0, ema_fast=100.0, ema_slow=100.2, rsi=20.0),
        "short",
        config,
    )
    assert _traditional_execution_quality(
        replace(feature, close=100.2, rsi=60.0, atr=1.0),
        "long",
        config,
    )


def test_pressure_filter_aggregates_10m_and_15m_without_extra_market_data() -> None:
    candles = [
        Candle(index * 300_000, 100.1, 100.2, 99.9, 100.1, 10.0)
        for index in range(90)
    ]
    last = candles[-1]
    candles[-1] = Candle(last.timestamp, 100.1, 100.2, 99.9, 100.0, 10.0)
    config = StrategyConfig(
        mode="traditional_kline",
        trigger_timeframe="5m",
        min_stop_loss_pct=0.0045,
        traditional_pressure_filter_enabled=True,
        traditional_pressure_timeframes_minutes=(10, 15),
        traditional_pressure_ema_period=30,
        traditional_pressure_sma_period=30,
        traditional_pressure_min_room_r=0.8,
    )

    ten_minute = _aggregate_five_minute_candles(candles, 10)
    fifteen_minute = _aggregate_five_minute_candles(candles, 15)
    clear, reason = _traditional_pressure_room(candles, "long", config)

    assert len(ten_minute) == 45
    assert len(fifteen_minute) == 30
    assert clear is False
    assert "room=" in reason
    assert "required=" in reason

    far_resistance = [
        Candle(index * 300_000, 101.0, 101.1, 99.9, 101.0, 10.0)
        for index in range(90)
    ]
    far_last = far_resistance[-1]
    far_resistance[-1] = Candle(far_last.timestamp, 101.0, 101.1, 99.9, 100.0, 10.0)
    assert _traditional_pressure_room(far_resistance, "long", config)[0] is True


def test_predictive_reversal_short_uses_closed_one_minute_spike_and_structure_stop() -> None:
    one_minute = []
    for index in range(70):
        close = 100.05 if index % 2 else 99.95
        one_minute.append(
            Candle(index * 60_000, close, 100.15, 99.85, close, 10.0)
        )
    one_minute.extend(
        (
            Candle(70 * 60_000, 100.0, 102.0, 99.9, 101.5, 50.0),
            Candle(71 * 60_000, 101.5, 101.6, 100.4, 100.5, 15.0),
            Candle(72 * 60_000, 100.5, 100.6, 99.6, 99.7, 15.0),
            Candle(73 * 60_000, 99.7, 99.8, 98.9, 99.0, 15.0),
        )
    )
    five_minute = [
        Candle(
            index * 300_000,
            100.0,
            100.2,
            95.0 if index == 85 else 99.8,
            100.0,
            100.0,
        )
        for index in range(90)
    ]
    config = StrategyConfig(
        trigger_timeframe="5m",
        traditional_predictive_reversal_short_enabled=True,
        traditional_predictive_reversal_confirmation_bars=4,
        traditional_predictive_reversal_min_spike_volume_ratio=2.0,
        traditional_predictive_reversal_min_spike_rsi=0.0,
        traditional_predictive_reversal_min_spike_range_atr=0.0,
        traditional_predictive_reversal_min_spike_extension_atr=0.0,
        traditional_predictive_reversal_min_rsi_drop=0.0,
        traditional_predictive_reversal_confirm_rsi_max=100.0,
        traditional_predictive_reversal_max_execution_extension_atr=20.0,
        traditional_predictive_reversal_max_pressure_distance_pct=0.05,
        traditional_predictive_reversal_min_room_r=1.0,
        traditional_predictive_reversal_stop_lookback_bars=8,
        traditional_predictive_reversal_max_stop_loss_pct=0.05,
        traditional_pressure_timeframes_minutes=(10, 15),
        traditional_pressure_ema_period=30,
        traditional_pressure_sma_period=30,
    )

    signal = _traditional_predictive_reversal_short(
        one_minute,
        five_minute,
        config,
    )

    assert signal.side == "short"
    assert "1m_predictive_reversal_short" in signal.reasons
    assert signal_position_size_multiplier(signal) == 0.5
    assert signal_stop_timeframe(signal, config) == "1m"
    assert signal_stop_loss_overrides(signal, config) == {
        "structure_lookback_bars": 8,
        "maximum_stop_loss_pct": 0.05,
    }

    low_volume_confirmation = list(one_minute)
    low_volume_confirmation[-1] = replace(low_volume_confirmation[-1], volume=0.1)
    assert (
        _traditional_predictive_reversal_short(
            low_volume_confirmation,
            five_minute,
            config,
        ).side
        == "flat"
    )


def test_predictive_reversal_short_fails_closed_when_one_minute_data_has_gap() -> None:
    one_minute = [
        Candle(index * 60_000, 100.0, 100.2, 99.8, 100.0, 10.0)
        for index in range(80)
    ]
    one_minute[-1] = replace(one_minute[-1], timestamp=one_minute[-1].timestamp + 60_000)
    five_minute = [
        Candle(index * 300_000, 100.0, 100.2, 99.8, 100.0, 10.0)
        for index in range(90)
    ]

    signal = _traditional_predictive_reversal_short(
        one_minute,
        five_minute,
        StrategyConfig(
            trigger_timeframe="5m",
            traditional_predictive_reversal_short_enabled=True,
            traditional_pressure_timeframes_minutes=(10, 15),
            traditional_pressure_ema_period=30,
            traditional_pressure_sma_period=30,
        ),
    )

    assert signal.side == "flat"
    assert signal.reasons == ("predictive_reversal_short_data_gap",)


def test_ema_and_rsi_have_values_after_warmup() -> None:
    values = [100 + index for index in range(60)]
    assert ema(values, 20)[-1] is not None
    assert rsi(values, 14)[-1] == 100.0


def test_risk_caps_notional_and_sets_long_protection() -> None:
    protection = RiskManager().protection("long", 10_000, 100, 1.5)
    assert protection.stop_price == 95
    assert protection.take_profit_price == 107.5
    assert protection.quantity * 100 <= 2_000


def test_max_notional_percentage_uses_theoretical_leveraged_capacity() -> None:
    risk = RiskManager(
        RiskConfig(risk_per_trade=0.02, stop_loss_pct=0.001, max_notional_pct=0.30),
        quantity_step=0.001,
        max_leverage=20,
    )

    protection = risk.protection("long", 100, 100, 1.5)

    assert protection.quantity == 6.0
    assert protection.quantity * 100 == 100 * 20 * 0.30


def test_risk_protection_accepts_dynamic_stop_distance() -> None:
    protection = RiskManager().protection("long", 10_000, 100, 1.6, stop_loss_pct=0.0035)
    assert protection.stop_price == 99.65
    assert protection.take_profit_price == 100.56


def test_dynamic_stop_uses_recent_structure_for_long_and_short() -> None:
    candles = [
        Candle(1, 100.0, 102.0, 98.0, 100.0, 10.0),
        Candle(2, 100.0, 103.0, 99.0, 102.0, 10.0),
    ]
    config = StrategyConfig(
        atr_period=2,
        atr_stop_multiplier=1.0,
        min_stop_loss_pct=0.001,
        max_stop_loss_pct=0.10,
        structure_stop_lookback_bars=2,
    )

    long_stop = dynamic_stop_loss_pct(candles, config, 0.05, side="long", entry_price=104.0)
    short_stop = dynamic_stop_loss_pct(candles, config, 0.05, side="short", entry_price=96.0)

    assert abs(long_stop - ((104.0 - 98.0) / 104.0)) < 1e-12
    assert abs(short_stop - ((103.0 - 96.0) / 96.0)) < 1e-12


def test_dynamic_structure_stop_honors_atr_buffer_and_maximum() -> None:
    candles = [
        Candle(1, 100.0, 102.0, 98.0, 100.0, 10.0),
        Candle(2, 100.0, 103.0, 99.0, 102.0, 10.0),
    ]
    config = StrategyConfig(
        atr_period=2,
        atr_stop_multiplier=1.0,
        min_stop_loss_pct=0.001,
        max_stop_loss_pct=0.06,
        structure_stop_lookback_bars=2,
        structure_stop_buffer_atr=0.5,
    )

    assert dynamic_stop_loss_pct(candles, config, 0.05, side="long", entry_price=104.0) == 0.06


def test_dynamic_stop_keeps_legacy_atr_result_when_structure_is_disabled() -> None:
    candles = [
        Candle(1, 100.0, 102.0, 98.0, 100.0, 10.0),
        Candle(2, 100.0, 103.0, 99.0, 102.0, 10.0),
    ]
    config = StrategyConfig(
        atr_period=2,
        atr_stop_multiplier=1.0,
        min_stop_loss_pct=0.001,
        max_stop_loss_pct=0.10,
    )

    without_side = dynamic_stop_loss_pct(candles, config, 0.05)
    with_side = dynamic_stop_loss_pct(candles, config, 0.05, side="long", entry_price=104.0)

    assert without_side == with_side


def test_failed_breakout_short_uses_two_bar_structure_and_special_stop_cap() -> None:
    candles = [
        Candle(1, 100.0, 110.0, 99.0, 101.0, 10.0),
        Candle(2, 101.0, 102.0, 95.0, 96.0, 10.0),
    ]
    config = StrategyConfig(
        atr_period=2,
        atr_stop_multiplier=0.1,
        min_stop_loss_pct=0.001,
        max_stop_loss_pct=0.006,
        structure_stop_lookback_bars=1,
        traditional_failed_breakout_short_stop_lookback_bars=2,
        traditional_failed_breakout_short_max_stop_loss_pct=0.01,
    )
    signal = Signal("short", 6, 2, ("failed_breakout_short_reversal",))

    normal = dynamic_stop_loss_pct(candles, config, 0.005, side="short", entry_price=101.0)
    guarded = dynamic_stop_loss_pct(
        candles,
        config,
        0.005,
        side="short",
        entry_price=101.0,
        **signal_stop_loss_overrides(signal, config),
    )

    assert normal == 0.006
    assert guarded == 0.01


def test_risk_protection_scales_countertrend_position_size() -> None:
    risk = RiskManager()
    full = risk.protection("long", 10_000, 100, 1.5)
    reduced = risk.protection("long", 10_000, 100, 1.5, size_multiplier=0.5)

    assert reduced.quantity < full.quantity
    assert reduced.quantity <= full.quantity * 0.51
    assert reduced.risk_amount == full.risk_amount * 0.5


def test_cost_aware_break_even_price_covers_estimated_costs() -> None:
    risk = RiskManager()
    for side in ("long", "short"):
        exit_price = risk.break_even_price(side, 100.0)
        assert abs(risk.estimate_net_pnl(side, 100.0, exit_price, 1.0)) < 1e-9


def test_profit_protection_moves_long_and_short_stops_past_costs() -> None:
    risk = RiskManager()
    strategy = MultiTimeframeStrategy(StrategyConfig(break_even_trigger_r=1.0, break_even_lock_r=0.1))
    long_position = Position("long", 1.0, 100.0, 99.0, 102.0, 1, initial_stop_price=99.0, best_price=100.0)
    short_position = Position("short", 1.0, 100.0, 101.0, 98.0, 1, initial_stop_price=101.0, best_price=100.0)

    protected_long = _tighten_position_stop(long_position, Candle(2, 100, 101.1, 99.9, 100.9, 1), strategy, risk)
    protected_short = _tighten_position_stop(short_position, Candle(2, 100, 100.1, 98.9, 99.1, 1), strategy, risk)

    assert protected_long.stop_price > risk.break_even_price("long", 100.0)
    assert protected_short.stop_price < risk.break_even_price("short", 100.0)
    assert risk.estimate_net_pnl("long", 100.0, protected_long.stop_price, 1.0) > 0
    assert risk.estimate_net_pnl("short", 100.0, protected_short.stop_price, 1.0) > 0


def test_optimized_profit_protection_waits_and_locks_meaningful_profit() -> None:
    risk = RiskManager()
    strategy = MultiTimeframeStrategy(StrategyConfig())
    position = Position("long", 1.0, 100.0, 99.0, 102.5, 1, initial_stop_price=99.0, best_price=100.0)

    before_trigger = _tighten_position_stop(position, Candle(2, 100.0, 101.2, 99.9, 101.1, 1.0), strategy, risk)
    break_even = _tighten_position_stop(position, Candle(3, 100.0, 101.3, 99.9, 101.25, 1.0), strategy, risk)
    trailing = _tighten_position_stop(break_even, Candle(4, 101.5, 102.1, 101.4, 102.0, 1.0), strategy, risk)

    assert before_trigger.stop_price == position.stop_price
    assert break_even.stop_reason == "break_even_stop"
    assert risk.estimate_net_pnl("long", 100.0, break_even.stop_price, 1.0) > 0.49
    assert trailing.stop_reason == "trailing_stop"
    assert trailing.stop_price > break_even.stop_price


def test_optimized_exit_defaults_match_paper_configuration() -> None:
    config = StrategyConfig()
    assert config.take_profit_r == 2.5
    assert config.break_even_trigger_r == 1.25
    assert config.break_even_lock_r == 0.5
    assert config.trailing_trigger_r == 2.0
    assert config.trailing_distance_r == 0.75
    assert config.enable_profit_trend_exit is True
    assert config.profit_trend_exit_trigger_r == 1.0


def test_short_holding_cost_does_not_charge_unearned_funding_interval() -> None:
    assert CostConfig(expected_holding_hours=0.1).funding_intervals == 0


def test_actual_eight_hour_hold_charges_one_estimated_funding_interval() -> None:
    costs = CostConfig(expected_holding_hours=0.1, funding_rate_pct_per_8h=0.0001)
    breakdown = costs.breakdown(100.0, 99.0, 1.0, holding_hours=8.1)

    assert costs.funding_intervals_for(8.1) == 1
    assert breakdown.funding_fee == 0.01


def test_profitable_short_detects_closed_bar_trend_invalidation() -> None:
    strategy = MultiTimeframeStrategy(StrategyConfig(mode="traditional_kline", trigger_timeframe="5m"))
    closes = [110.0 - index * 0.35 for index in range(36)] + [97.4 + index * 0.8 for index in range(10)]
    candles = [
        Candle(index * 300_000, close - 0.2, close + 0.3, close - 0.3, close, 10.0)
        for index, close in enumerate(closes)
    ]

    assert strategy.position_trend_invalidated("short", {"5m": candles}) is True


def test_candle_model_is_available() -> None:
    candle = Candle(1, 1, 2, 0.5, 1.5, 10)
    assert candle.close == 1.5


def test_okx_one_second_candles_aggregate_to_30_seconds() -> None:
    candles = [
        Candle(1_000, 100, 101, 99, 100.5, 2),
        Candle(20_000, 100.5, 102, 100, 101.5, 3),
        Candle(31_000, 101.5, 103, 101, 102.5, 4),
    ]
    result = OkxAdapter._aggregate_30s(candles)
    assert [(c.timestamp, c.volume) for c in result] == [(0, 5), (30_000, 4)]


def test_okx_quote_volume_is_preserved_for_volume_analysis() -> None:
    candle = OkxAdapter._candle_from_row([1, "100", "101", "99", "100.5", "2", "0.02", "201", "1"])
    assert candle.volume == 2
    assert candle.quote_volume == 201


def test_okx_five_minute_failure_falls_back_to_official_one_minute_candles() -> None:
    settings = ExchangeSettings(
        name="okx",
        environment="demo",
        base_url="https://openapi.okx.com",
        symbol="BTC-USDT-SWAP",
    )
    one_minute_rows = [
        ["300000", "105", "107", "104", "106", "6", "0", "636", "1"],
        ["240000", "104", "106", "103", "105", "5", "0", "525", "1"],
        ["180000", "103", "105", "102", "104", "4", "0", "416", "1"],
        ["120000", "102", "104", "101", "103", "3", "0", "309", "1"],
        ["60000", "101", "103", "100", "102", "2", "0", "204", "1"],
        ["0", "100", "102", "99", "101", "1", "0", "101", "1"],
    ]
    fallback_payload = {"code": "0", "data": one_minute_rows}

    with patch(
        "btc_futures_bot.exchanges.okx.request_json",
        side_effect=[ApiError("temporary 5m failure"), fallback_payload],
    ) as mocked_request:
        candles = OkxAdapter(settings).fetch_candles("5m", 300)

    assert [call.kwargs["params"]["bar"] for call in mocked_request.call_args_list] == ["5m", "1m"]
    assert len(candles) == 2
    assert candles[0] == Candle(0, 100, 106, 99, 105, 15, quote_volume=1555)
    assert candles[1] == Candle(300_000, 105, 107, 104, 106, 6, quote_volume=636)


def test_okx_non_five_minute_failure_does_not_change_timeframe_source() -> None:
    settings = ExchangeSettings(
        name="okx",
        environment="demo",
        base_url="https://openapi.okx.com",
        symbol="BTC-USDT-SWAP",
    )
    with patch(
        "btc_futures_bot.exchanges.okx.request_json",
        side_effect=ApiError("temporary 1h failure"),
    ) as mocked_request:
        try:
            OkxAdapter(settings).fetch_candles("1h", 300)
        except ApiError:
            pass
        else:
            raise AssertionError("1h failure must be propagated")

    assert mocked_request.call_count == 1


def test_scalp_strategy_accepts_fast_breakout_without_full_alignment() -> None:
    def series(count: int, interval_ms: int) -> list[Candle]:
        close = 100.0
        rows: list[Candle] = []
        for index in range(count):
            close += -0.04 if index % 3 == 0 else 0.12
            rows.append(Candle(index * interval_ms, close - 0.02, close + 0.03, close - 0.03, close, 10))
        return rows

    trigger = series(60, 30_000)
    last = trigger[-1]
    trigger[-1] = Candle(last.timestamp, last.open, last.open + 0.5, last.low, last.open + 0.45, 30)
    signal = MultiTimeframeStrategy(StrategyConfig()).evaluate(
        {"30s": trigger, "1m": series(60, 60_000), "5m": series(60, 300_000)}
    )
    assert signal.side == "long"
    assert signal.score >= 2


def test_traditional_kline_strategy_accepts_golden_cross_alignment() -> None:
    def candles(values: list[float], interval_ms: int, volume: float = 10.0) -> list[Candle]:
        return [
            Candle(index * interval_ms, value - 0.1, value + 0.2, value - 0.2, value, volume)
            for index, value in enumerate(values)
        ]

    regime = candles([100 + index * 0.2 for index in range(40)], 3_600_000)
    trigger_values = [105 - index * 0.12 for index in range(39)] + [104.0]
    trigger = candles(trigger_values, 300_000)
    trigger[-1] = Candle(trigger[-1].timestamp, 103.8, 104.4, 103.7, 104.0, 25.0)
    execution = candles([100 + index * 0.05 for index in range(40)], 60_000)
    config = StrategyConfig(
        mode="traditional_kline",
        trigger_timeframe="5m",
        regime_timeframe="1h",
        traditional_trend_fast=5,
        traditional_trend_slow=20,
        traditional_signal_fast=3,
        traditional_signal_slow=5,
        traditional_rsi_period=3,
        traditional_macd_fast=3,
        traditional_macd_slow=6,
        traditional_macd_signal=2,
        traditional_volume_sma_period=3,
        traditional_min_volume_ratio=1.1,
        traditional_rsi_long_min=0,
        traditional_rsi_long_max=100,
    )

    signal = MultiTimeframeStrategy(config).evaluate({"5m": trigger, "1m": execution, "1h": regime})

    assert signal.side == "long"
    assert "5m_golden_cross" in signal.reasons
    assert signal.timestamp == trigger[-1].timestamp


def test_traditional_ultra_short_uses_neutral_hourly_context_at_half_size() -> None:
    def candles(values: list[float], interval_ms: int, volume: float = 10.0) -> list[Candle]:
        return [
            Candle(index * interval_ms, value - 0.1, value + 0.2, value - 0.2, value, volume)
            for index, value in enumerate(values)
        ]

    neutral_regime = candles([100.0] * 40, 3_600_000)
    trigger_values = [100.0 + (index**2) * 0.001 for index in range(48)]
    trigger = candles(trigger_values, 300_000)
    execution_base = trigger_values[-1] - 0.4
    execution = candles([execution_base + index * 0.005 for index in range(39)], 60_000)
    execution.append(
        Candle(
            39 * 60_000,
            execution[-1].close,
            trigger_values[-1] + 0.12,
            execution[-1].close - 0.02,
            trigger_values[-1] + 0.1,
            40.0,
        )
    )
    base = StrategyConfig(
        mode="traditional_kline",
        trigger_timeframe="5m",
        regime_timeframe="1h",
        traditional_trend_fast=5,
        traditional_trend_slow=20,
        traditional_signal_fast=3,
        traditional_signal_slow=5,
        traditional_rsi_period=3,
        traditional_macd_fast=3,
        traditional_macd_slow=6,
        traditional_macd_signal=2,
        traditional_volume_sma_period=3,
        traditional_min_volume_ratio=1.1,
        traditional_rsi_long_min=0,
        traditional_rsi_long_max=100,
        traditional_ultra_short_1m_max_extension_atr=10.0,
    )
    market = {"5m": trigger, "1m": execution, "1h": neutral_regime}

    blocked = MultiTimeframeStrategy(base).evaluate(market)
    enabled = MultiTimeframeStrategy(
        replace(base, traditional_ultra_short_enabled=True)
    ).evaluate(market)

    assert blocked.side == "flat"
    assert enabled.side == "long"
    assert "1h_ultra_short_context_long" in enabled.reasons
    assert signal_position_size_multiplier(enabled) == 0.5


def test_failed_breakout_short_requires_confirmed_blow_off_top() -> None:
    history = [
        Candle(index * 300_000, 99.0, 100.0, 98.0, 99.5, 10.0)
        for index in range(6)
    ]
    rejection = Candle(6 * 300_000, 100.0, 110.0, 99.0, 101.0, 40.0)
    confirmation = Candle(7 * 300_000, 101.0, 102.0, 95.0, 96.0, 30.0)
    regime = _TraditionalFeatures(
        open=104.0,
        high=106.0,
        low=103.0,
        close=105.0,
        previous_close=104.0,
        ema_fast=103.0,
        previous_ema_fast=102.8,
        ema_slow=100.0,
        previous_ema_slow=99.8,
        rsi=62.0,
        macd_histogram=1.0,
        previous_macd_histogram=0.8,
        atr=2.0,
        volume_ratio=1.0,
    )
    previous = replace(
        regime,
        open=rejection.open,
        high=rejection.high,
        low=rejection.low,
        close=rejection.close,
        volume_ratio=3.2,
        macd_histogram=1.0,
    )
    current = replace(
        previous,
        open=confirmation.open,
        high=confirmation.high,
        low=confirmation.low,
        close=confirmation.close,
        volume_ratio=2.2,
        macd_histogram=0.5,
        previous_macd_histogram=1.0,
    )
    execution = replace(
        current,
        close=95.0,
        ema_fast=96.0,
        ema_slow=97.0,
    )
    config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

    signal = _traditional_failed_breakout_short_reversal(
        history + [rejection, confirmation],
        regime,
        previous,
        current,
        execution,
        config,
    )

    assert signal.side == "short"
    assert "failed_breakout_short_reversal" in signal.reasons
    assert signal_position_size_multiplier(signal) == 0.5
    assert _traditional_failed_breakout_short_reversal(
        history + [rejection, confirmation],
        regime,
        previous,
        replace(current, volume_ratio=1.99),
        execution,
        config,
    ).side == "flat"


def test_failed_breakout_short_shadow_logs_candidate_without_trading() -> None:
    def flat_series(count: int, interval_ms: int) -> list[Candle]:
        return [
            Candle(index * interval_ms, 100.0, 100.1, 99.9, 100.0, 10.0)
            for index in range(count)
        ]

    market = {
        "5m": flat_series(40, 300_000),
        "1m": flat_series(40, 60_000),
        "1h": flat_series(210, 3_600_000),
    }
    candidate = Signal("short", 6, market["5m"][-1].timestamp, ("failed_breakout_short_reversal",))
    shadow_config = StrategyConfig(
        mode="traditional_kline",
        trigger_timeframe="5m",
        regime_timeframe="1h",
        traditional_failed_breakout_short_shadow=True,
    )

    with patch(
        "btc_futures_bot.strategy._traditional_failed_breakout_short_reversal",
        return_value=candidate,
    ):
        shadow = MultiTimeframeStrategy(shadow_config).evaluate(market)
        enabled = MultiTimeframeStrategy(
            replace(
                shadow_config,
                traditional_failed_breakout_short_enabled=True,
                traditional_failed_breakout_short_shadow=False,
            )
        ).evaluate(market)

    assert shadow.side == "flat"
    assert "shadow_candidate=short" in shadow.reasons
    assert "shadow_failed_breakout_short_reversal" in shadow.reasons
    assert enabled.side == "short"
    assert signal_position_size_multiplier(enabled) == 0.5


def test_traditional_setup_can_persist_until_confirmation_aligns() -> None:
    def candles(values: list[float], interval_ms: int, volume: float = 10.0) -> list[Candle]:
        return [
            Candle(index * interval_ms, value - 0.1, value + 0.2, value - 0.2, value, volume)
            for index, value in enumerate(values)
        ]

    regime = candles([100 + index * 0.2 for index in range(41)], 3_600_000)
    trigger_values = [105 - index * 0.12 for index in range(39)] + [104.0, 104.2]
    trigger = candles(trigger_values, 300_000)
    trigger[-2] = Candle(trigger[-2].timestamp, 103.8, 104.4, 103.7, 104.0, 25.0)
    trigger[-1] = Candle(trigger[-1].timestamp, 104.0, 104.4, 103.9, 104.2, 25.0)
    execution = candles([100 + index * 0.05 for index in range(41)], 60_000)
    base = StrategyConfig(
        mode="traditional_kline",
        trigger_timeframe="5m",
        regime_timeframe="1h",
        traditional_trend_fast=5,
        traditional_trend_slow=20,
        traditional_signal_fast=3,
        traditional_signal_slow=5,
        traditional_rsi_period=3,
        traditional_macd_fast=3,
        traditional_macd_slow=6,
        traditional_macd_signal=2,
        traditional_volume_sma_period=3,
        traditional_min_volume_ratio=1.1,
        traditional_rsi_long_min=0,
        traditional_rsi_long_max=100,
        traditional_allow_pullback=False,
        traditional_allow_breakout=False,
    )
    market = {"5m": trigger, "1m": execution, "1h": regime}

    strategy = MultiTimeframeStrategy(replace(base, traditional_blocked_setup_valid_bars=2))
    expired = strategy.evaluate(market)
    persisted = strategy.reevaluate_blocked_signal("long", market)

    assert expired.side == "flat"
    assert persisted.side == "long"
    assert "5m_golden_cross" in persisted.reasons
    assert "5m_setup_persisted_1_bars" in persisted.reasons
    assert "macro_blocked_signal_revalidated" in persisted.reasons

    main_strategy_persisted = MultiTimeframeStrategy(
        replace(base, traditional_setup_valid_bars=2)
    ).evaluate(market)
    assert main_strategy_persisted.side == "long"
    assert "5m_setup_persisted_1_bars" in main_strategy_persisted.reasons
    assert "macro_blocked_signal_revalidated" not in main_strategy_persisted.reasons


def test_traditional_setup_macd_handoff_is_one_bar_and_extension_capped() -> None:
    previous = _TraditionalFeatures(
        open=100.0,
        high=101.0,
        low=98.0,
        close=99.0,
        previous_close=100.0,
        ema_fast=99.5,
        previous_ema_fast=100.2,
        ema_slow=100.0,
        previous_ema_slow=100.0,
        rsi=42.0,
        macd_histogram=8.0,
        previous_macd_histogram=10.0,
        atr=1.0,
        volume_ratio=3.0,
    )
    current = replace(
        previous,
        close=98.6,
        ema_fast=100.0,
        macd_histogram=-1.0,
        previous_macd_histogram=8.0,
        volume_ratio=1.5,
    )
    setup = _TraditionalSetupState(
        golden_cross=False,
        death_cross=True,
        pullback_long=False,
        pullback_short=True,
        breakout_long_raw=False,
        breakout_short_raw=True,
        breakout_long=False,
        breakout_short=False,
    )
    enabled = StrategyConfig(
        traditional_rsi_short_min=28.0,
        traditional_rsi_short_max=52.0,
        traditional_min_volume_ratio=1.1,
        traditional_setup_macd_handoff_max_extension_atr=1.5,
    )

    assert _traditional_setup_macd_handoff(previous, current, setup, "short", enabled)
    assert not _traditional_setup_macd_handoff(
        previous,
        current,
        setup,
        "short",
        replace(enabled, traditional_setup_macd_handoff_max_extension_atr=1.25),
    )
    assert not _traditional_setup_macd_handoff(
        replace(previous, volume_ratio=1.0),
        current,
        setup,
        "short",
        enabled,
    )


def test_countertrend_cross_requires_fresh_cross_and_strict_transition_limits() -> None:
    regime = _TraditionalFeatures(
        open=99.9,
        high=100.2,
        low=99.6,
        close=99.8,
        previous_close=99.9,
        ema_fast=100.0,
        previous_ema_fast=100.1,
        ema_slow=101.0,
        previous_ema_slow=101.0,
        rsi=48.0,
        macd_histogram=-1.0,
        previous_macd_histogram=-0.8,
        atr=1.0,
        volume_ratio=1.0,
    )
    trigger = replace(
        regime,
        close=100.8,
        ema_fast=100.0,
        ema_slow=99.9,
        rsi=58.0,
        macd_histogram=1.0,
        atr=1.0,
        volume_ratio=2.0,
    )
    cross = _TraditionalSetupState(True, False, False, False, True, False, False, False)
    pullback_only = replace(cross, golden_cross=False, pullback_long=True)
    config = StrategyConfig(
        traditional_countertrend_cross_max_regime_gap_pct=0.01,
        traditional_early_regime_max_gap_pct=0.005,
        traditional_setup_macd_handoff_max_extension_atr=1.5,
    )

    assert _traditional_countertrend_cross_regime(regime, trigger, cross, "long", config)
    assert not _traditional_countertrend_cross_regime(regime, trigger, pullback_only, "long", config)
    assert not _traditional_countertrend_cross_regime(regime, replace(trigger, volume_ratio=1.9), cross, "long", config)
    assert not _traditional_countertrend_cross_regime(regime, replace(trigger, close=101.6), cross, "long", config)
    assert not _traditional_countertrend_cross_regime(
        regime,
        trigger,
        cross,
        "long",
        replace(config, traditional_countertrend_cross_max_regime_gap_pct=0.0),
    )


def test_countertrend_pullback_requires_fresh_reclaim_and_bounded_transition() -> None:
    regime = _TraditionalFeatures(
        open=99.5,
        high=100.0,
        low=99.0,
        close=99.4,
        previous_close=99.5,
        ema_fast=100.0,
        previous_ema_fast=100.1,
        ema_slow=101.0,
        previous_ema_slow=101.0,
        rsi=42.0,
        macd_histogram=-1.0,
        previous_macd_histogram=-0.8,
        atr=1.0,
        volume_ratio=1.0,
    )
    trigger = replace(
        regime,
        close=100.8,
        ema_fast=100.0,
        ema_slow=99.9,
        rsi=58.0,
        macd_histogram=1.0,
        atr=1.0,
        volume_ratio=1.6,
    )
    pullback = _TraditionalSetupState(False, False, True, False, False, False, False, False)
    cross_only = replace(pullback, pullback_long=False, golden_cross=True)
    config = StrategyConfig(traditional_countertrend_pullback_max_regime_gap_pct=0.0125)

    assert _traditional_countertrend_pullback_regime(regime, trigger, pullback, "long", config)
    assert not _traditional_countertrend_pullback_regime(regime, trigger, cross_only, "long", config)
    assert not _traditional_countertrend_pullback_regime(
        regime,
        replace(trigger, volume_ratio=1.49),
        pullback,
        "long",
        config,
    )
    assert not _traditional_countertrend_pullback_regime(
        regime,
        replace(trigger, close=101.1),
        pullback,
        "long",
        config,
    )
    assert not _traditional_countertrend_pullback_regime(
        replace(regime, rsi=46.0),
        trigger,
        pullback,
        "long",
        config,
    )
    assert not _traditional_countertrend_pullback_regime(
        replace(regime, macd_histogram=0.1),
        trigger,
        pullback,
        "long",
        config,
    )
    assert not _traditional_countertrend_pullback_regime(
        regime,
        trigger,
        pullback,
        "long",
        replace(config, traditional_countertrend_pullback_max_regime_gap_pct=0.0),
    )

    tagged = Signal("long", 6, 1, ("1h_countertrend_pullback_up",))
    normal = Signal("long", 6, 1, ("1h_trend_up",))
    assert signal_position_size_multiplier(tagged) == 0.5
    assert signal_position_size_multiplier(normal) == 1.0


def test_neutral_transition_requires_crossed_regime_and_high_volume_setup() -> None:
    regime = _TraditionalFeatures(
        open=100.1,
        high=100.5,
        low=99.8,
        close=100.1,
        previous_close=99.9,
        ema_fast=100.0,
        previous_ema_fast=99.9,
        ema_slow=100.7,
        previous_ema_slow=100.7,
        rsi=52.0,
        macd_histogram=0.2,
        previous_macd_histogram=0.1,
        atr=1.0,
        volume_ratio=1.0,
    )
    trigger = replace(
        regime,
        close=102.0,
        ema_fast=100.0,
        ema_slow=99.8,
        rsi=65.0,
        macd_histogram=1.0,
        atr=1.0,
        volume_ratio=2.1,
    )
    long_setup = _TraditionalSetupState(True, False, False, False, False, False, False, False)
    opposite_setup = replace(long_setup, golden_cross=False, death_cross=True)
    config = StrategyConfig(traditional_neutral_transition_max_regime_gap_pct=0.0075)

    assert _traditional_neutral_transition_regime(regime, trigger, long_setup, "long", config)
    assert not _traditional_neutral_transition_regime(regime, trigger, opposite_setup, "long", config)
    assert not _traditional_neutral_transition_regime(
        regime,
        replace(trigger, volume_ratio=1.99),
        long_setup,
        "long",
        config,
    )
    assert not _traditional_neutral_transition_regime(
        replace(regime, close=99.8),
        trigger,
        long_setup,
        "long",
        config,
    )
    assert not _traditional_neutral_transition_regime(
        replace(regime, close=100.2),
        trigger,
        long_setup,
        "long",
        config,
    )
    assert not _traditional_neutral_transition_regime(
        replace(regime, rsi=61.0),
        trigger,
        long_setup,
        "long",
        config,
    )
    assert not _traditional_neutral_transition_regime(
        regime,
        replace(trigger, close=102.3),
        long_setup,
        "long",
        config,
    )
    assert not _traditional_neutral_transition_regime(
        regime,
        trigger,
        long_setup,
        "long",
        replace(config, traditional_neutral_transition_max_regime_gap_pct=0.0),
    )

    tagged = Signal("long", 6, 1, ("1h_neutral_transition_up",))
    assert signal_position_size_multiplier(tagged) == 0.5


def test_setup_volume_handoff_requires_delayed_volume_and_tight_extension() -> None:
    previous = _TraditionalFeatures(
        open=100.6,
        high=100.8,
        low=99.8,
        close=100.7,
        previous_close=100.5,
        ema_fast=100.0,
        previous_ema_fast=99.9,
        ema_slow=99.8,
        previous_ema_slow=99.9,
        rsi=58.0,
        macd_histogram=1.0,
        previous_macd_histogram=0.5,
        atr=1.0,
        volume_ratio=1.0,
    )
    current = replace(previous, open=100.7, high=101.0, low=100.5, close=100.8, ema_fast=100.1, volume_ratio=1.2)
    raw_breakout = _TraditionalSetupState(False, False, False, False, True, False, False, False)
    config = StrategyConfig(
        traditional_min_volume_ratio=1.1,
        traditional_breakout_max_extension_atr=1.25,
        traditional_breakout_min_close_location=0.7,
        traditional_setup_volume_handoff_max_extension_atr=0.8,
    )

    assert _traditional_setup_volume_handoff(previous, current, raw_breakout, "long", config)
    assert not _traditional_setup_volume_handoff(
        previous,
        replace(current, close=101.0),
        raw_breakout,
        "long",
        config,
    )
    assert not _traditional_setup_volume_handoff(
        replace(previous, volume_ratio=1.1),
        current,
        raw_breakout,
        "long",
        config,
    )
    assert not _traditional_setup_volume_handoff(
        previous,
        current,
        raw_breakout,
        "long",
        replace(config, traditional_setup_volume_handoff_max_extension_atr=0.0),
    )


def test_traditional_kline_strategy_accepts_early_short_breakdown() -> None:
    def candles(values: list[float], interval_ms: int, volumes: list[float] | None = None) -> list[Candle]:
        selected_volumes = volumes or [10.0] * len(values)
        return [
            Candle(index * interval_ms, value + 0.1, value + 0.2, value - 0.2, value, selected_volumes[index])
            for index, value in enumerate(values)
        ]

    regime_values = [100 + index * 0.3 for index in range(37)] + [110.65, 110.5, 110.35]
    trigger_values = [106 - index * 0.08 for index in range(39)] + [102.2]
    execution_values = [104 - index * 0.05 for index in range(40)]
    config = StrategyConfig(
        mode="traditional_kline",
        trigger_timeframe="5m",
        regime_timeframe="1h",
        traditional_trend_fast=5,
        traditional_trend_slow=20,
        traditional_signal_fast=3,
        traditional_signal_slow=5,
        traditional_rsi_period=3,
        traditional_macd_fast=3,
        traditional_macd_slow=6,
        traditional_macd_signal=2,
        traditional_volume_sma_period=3,
        traditional_min_volume_ratio=1.1,
        traditional_rsi_short_min=0,
        traditional_rsi_short_max=100,
        traditional_breakout_lookback=4,
        traditional_breakout_min_volume_ratio=1.1,
        traditional_breakout_min_body_ratio=0.0,
        traditional_breakout_min_close_location=0.0,
        traditional_breakout_max_extension_atr=100.0,
        traditional_early_regime_max_gap_pct=0.1,
        traditional_early_regime_require_macd_acceleration=False,
    )
    market = {
        "5m": candles(trigger_values, 300_000, [10.0] * 39 + [25.0]),
        "1m": candles(execution_values, 60_000),
        "1h": candles(regime_values, 3_600_000),
    }
    signal = MultiTimeframeStrategy(config).evaluate(market)

    assert signal.side == "short"
    assert "5m_breakdown" in signal.reasons
    assert "1h_early_trend_down" in signal.reasons

    wide_gap_rejected = MultiTimeframeStrategy(
        replace(config, traditional_early_regime_max_gap_pct=0.005)
    ).evaluate(market)
    assert wide_gap_rejected.side == "flat"

    weak_breakout_rejected = MultiTimeframeStrategy(
        replace(
            config,
            traditional_breakout_min_body_ratio=0.6,
            traditional_breakout_min_close_location=0.7,
        )
    ).evaluate(market)
    assert weak_breakout_rejected.side == "flat"
    assert "5m_short_breakout_quality_rejected" in weak_breakout_rejected.reasons


def test_traditional_one_minute_impulse_enters_early_once_at_half_size() -> None:
    def candles(values: list[float], interval_ms: int, volume: float = 10.0) -> list[Candle]:
        return [
            Candle(index * interval_ms, value - 0.05, value + 0.10, value - 0.10, value, volume)
            for index, value in enumerate(values)
        ]

    regime = candles([100.0 + index * 0.1 for index in range(40)], 3_600_000)
    trigger_values = [100.0] * 10 + [100.0 + (index**1.3) * 0.02 for index in range(30)]
    trigger = candles(trigger_values, 300_000)
    base = trigger_values[-1] - 0.2
    execution = candles([base + (index % 5) * 0.01 for index in range(45)], 60_000)
    last = execution[-1]
    execution[-1] = Candle(last.timestamp, base + 0.04, base + 0.50, base + 0.03, base + 0.48, 80.0)
    config = StrategyConfig(
        mode="traditional_kline",
        trigger_timeframe="5m",
        regime_timeframe="1h",
        traditional_trend_fast=5,
        traditional_trend_slow=20,
        traditional_signal_fast=3,
        traditional_signal_slow=5,
        traditional_rsi_period=3,
        traditional_macd_fast=3,
        traditional_macd_slow=6,
        traditional_macd_signal=2,
        traditional_atr_period=3,
        traditional_volume_sma_period=5,
        traditional_min_volume_ratio=1.1,
        traditional_rsi_long_min=0.0,
        traditional_rsi_long_max=100.0,
        traditional_allow_pullback=False,
        traditional_allow_breakout=False,
        traditional_allow_1m_impulse=True,
        traditional_1m_impulse_lookback=10,
        traditional_1m_impulse_confirmation_bars=1,
        traditional_1m_impulse_min_volume_ratio=2.0,
        traditional_1m_impulse_min_body_ratio=0.6,
        traditional_1m_impulse_min_close_location=0.8,
        traditional_1m_impulse_min_range_atr=1.0,
        traditional_1m_impulse_max_extension_atr=10.0,
    )
    strategy = MultiTimeframeStrategy(config)

    first = strategy.evaluate({"5m": trigger, "1m": execution, "1h": regime})

    assert first.side == "long"
    assert first.timestamp == execution[-1].timestamp
    assert "1m_impulse_breakout" in first.reasons
    assert signal_position_size_multiplier(first) == 0.5

    previous = execution[-1]
    confirmation_strategy = MultiTimeframeStrategy(
        replace(config, traditional_1m_impulse_confirmation_bars=2)
    )
    awaiting_confirmation = confirmation_strategy.evaluate(
        {"5m": trigger, "1m": execution, "1h": regime}
    )
    assert awaiting_confirmation.side == "flat"

    execution.append(
        Candle(
            previous.timestamp + 60_000,
            previous.close,
            previous.close + 0.50,
            previous.close - 0.01,
            previous.close + 0.48,
            80.0,
        )
    )
    continuation = strategy.evaluate({"5m": trigger, "1m": execution, "1h": regime})
    confirmed = confirmation_strategy.evaluate({"5m": trigger, "1m": execution, "1h": regime})

    assert continuation.side == "flat"
    assert confirmed.side == "long"
    assert confirmed.timestamp == execution[-1].timestamp

    neutral_regime = candles([100.0] * 40, 3_600_000)
    neutral_context = MultiTimeframeStrategy(
        replace(
            config,
            traditional_ultra_short_enabled=True,
            traditional_1m_impulse_confirmation_bars=1,
        )
    ).evaluate({"5m": trigger, "1m": execution[:-1], "1h": neutral_regime})
    assert neutral_context.side == "flat"


def test_traditional_one_minute_impulse_respects_5m_extension_cap() -> None:
    def candles(values: list[float], interval_ms: int, volume: float = 10.0) -> list[Candle]:
        return [
            Candle(index * interval_ms, value - 0.05, value + 0.10, value - 0.10, value, volume)
            for index, value in enumerate(values)
        ]

    regime = candles([100.0 + index * 0.1 for index in range(40)], 3_600_000)
    trigger_values = [100.0] * 10 + [100.0 + (index**1.3) * 0.02 for index in range(30)]
    trigger = candles(trigger_values, 300_000)
    base = trigger_values[-1] - 0.2
    execution = candles([base + (index % 5) * 0.01 for index in range(45)], 60_000)
    last = execution[-1]
    execution[-1] = Candle(last.timestamp, base + 0.04, base + 0.50, base + 0.03, base + 0.48, 80.0)
    config = StrategyConfig(
        mode="traditional_kline",
        trigger_timeframe="5m",
        regime_timeframe="1h",
        traditional_trend_fast=5,
        traditional_trend_slow=20,
        traditional_signal_fast=3,
        traditional_signal_slow=5,
        traditional_rsi_period=3,
        traditional_macd_fast=3,
        traditional_macd_slow=6,
        traditional_macd_signal=2,
        traditional_atr_period=3,
        traditional_volume_sma_period=5,
        traditional_rsi_long_min=0.0,
        traditional_rsi_long_max=100.0,
        traditional_allow_pullback=False,
        traditional_allow_breakout=False,
        traditional_allow_1m_impulse=True,
        traditional_1m_impulse_lookback=10,
        traditional_1m_impulse_confirmation_bars=1,
        traditional_1m_impulse_max_extension_atr=0.01,
    )

    signal = MultiTimeframeStrategy(config).evaluate({"5m": trigger, "1m": execution, "1h": regime})

    assert signal.side == "flat"
