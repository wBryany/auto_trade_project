from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .indicators import atr, ema, macd, rsi, sma
from .models import Candle, Signal


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for the legacy scalp and traditional K-line modes."""

    mode: str = "scalp"
    trigger_timeframe: str = "30s"
    regime_timeframe: str = "5m"
    ema_fast: int = 5
    ema_slow: int = 13
    ema_regime: int = 34
    rsi_period: int = 7
    atr_period: int = 7
    macd_fast: int = 5
    macd_slow: int = 13
    macd_signal: int = 4
    breakout_lookback: int = 6
    volume_sma_period: int = 12
    min_volume_ratio: float = 1.0
    require_full_alignment: bool = False
    require_volume_confirmation: bool = True
    min_score: int = 2
    take_profit_r: float = 2.5
    atr_stop_multiplier: float = 1.4
    min_stop_loss_pct: float = 0.0025
    max_stop_loss_pct: float = 0.006
    structure_stop_lookback_bars: int = 0
    structure_stop_buffer_atr: float = 0.0
    min_hold_seconds: int = 90
    reversal_min_score: int = 5
    max_hold_seconds: int = 420
    hard_max_hold_seconds: int = 900
    traditional_trend_fast: int = 50
    traditional_trend_slow: int = 200
    traditional_signal_fast: int = 9
    traditional_signal_slow: int = 21
    traditional_rsi_period: int = 14
    traditional_rsi_long_min: float = 50.0
    traditional_rsi_long_max: float = 70.0
    traditional_rsi_short_min: float = 30.0
    traditional_rsi_short_max: float = 50.0
    traditional_macd_fast: int = 12
    traditional_macd_slow: int = 26
    traditional_macd_signal: int = 9
    traditional_atr_period: int = 14
    traditional_volume_sma_period: int = 20
    traditional_min_volume_ratio: float = 1.1
    traditional_strong_regime_min_gap_atr: float = 0.0
    traditional_strong_regime_require_fast_slope: bool = False
    traditional_strong_regime_require_macd: bool = False
    traditional_require_1m_confirmation: bool = True
    traditional_blocked_setup_valid_bars: int = 1
    traditional_setup_macd_handoff_max_extension_atr: float = 0.0
    traditional_setup_volume_handoff_max_extension_atr: float = 0.0
    traditional_allow_pullback: bool = True
    traditional_allow_breakout: bool = True
    traditional_breakout_lookback: int = 6
    traditional_breakout_min_volume_ratio: float = 1.3
    traditional_breakout_min_body_ratio: float = 0.5
    traditional_breakout_min_close_location: float = 0.7
    traditional_breakout_max_extension_atr: float = 1.25
    traditional_cross_min_body_ratio: float = 0.0
    traditional_cross_min_close_location: float = 0.0
    traditional_cross_max_extension_atr: float = 0.0
    traditional_execution_rsi_long_max: float = 0.0
    traditional_execution_rsi_short_min: float = 0.0
    traditional_execution_max_extension_atr: float = 0.0
    traditional_pressure_filter_enabled: bool = False
    traditional_pressure_timeframes_minutes: tuple[int, ...] = (10, 15)
    traditional_pressure_ema_period: int = 30
    traditional_pressure_sma_period: int = 30
    traditional_pressure_min_room_r: float = 0.8
    traditional_allow_1m_impulse: bool = False
    traditional_1m_impulse_allow_long: bool = True
    traditional_1m_impulse_allow_short: bool = True
    traditional_1m_impulse_lookback: int = 20
    traditional_1m_impulse_confirmation_bars: int = 2
    traditional_1m_impulse_min_volume_ratio: float = 3.0
    traditional_1m_impulse_min_body_ratio: float = 0.6
    traditional_1m_impulse_min_close_location: float = 0.65
    traditional_1m_impulse_min_range_atr: float = 1.0
    traditional_1m_impulse_max_extension_atr: float = 1.5
    traditional_failed_breakout_short_enabled: bool = False
    traditional_failed_breakout_short_shadow: bool = False
    traditional_failed_breakout_short_lookback: int = 6
    traditional_failed_breakout_short_prior_volume_ratio: float = 3.0
    traditional_failed_breakout_short_prior_wick_ratio: float = 0.6
    traditional_failed_breakout_short_confirm_volume_ratio: float = 2.0
    traditional_failed_breakout_short_confirm_body_ratio: float = 0.5
    traditional_failed_breakout_short_confirm_close_location: float = 0.6
    traditional_failed_breakout_short_stop_lookback_bars: int = 2
    traditional_failed_breakout_short_max_stop_loss_pct: float = 0.01
    traditional_predictive_reversal_short_enabled: bool = False
    traditional_predictive_reversal_lookback: int = 20
    traditional_predictive_reversal_confirmation_bars: int = 4
    traditional_predictive_reversal_min_spike_volume_ratio: float = 3.0
    traditional_predictive_reversal_min_spike_rsi: float = 65.0
    traditional_predictive_reversal_min_spike_range_atr: float = 1.5
    traditional_predictive_reversal_min_spike_extension_atr: float = 1.5
    traditional_predictive_reversal_min_rsi_drop: float = 12.0
    traditional_predictive_reversal_confirm_rsi_max: float = 55.0
    traditional_predictive_reversal_min_confirmation_volume_ratio: float = 0.5
    traditional_predictive_reversal_max_execution_extension_atr: float = 1.0
    traditional_predictive_reversal_max_pressure_distance_pct: float = 0.008
    traditional_predictive_reversal_min_room_r: float = 1.2
    traditional_predictive_reversal_stop_lookback_bars: int = 8
    traditional_predictive_reversal_max_stop_loss_pct: float = 0.008
    traditional_allow_early_regime: bool = True
    traditional_early_regime_max_gap_pct: float = 0.005
    traditional_early_regime_rsi_long_min: float = 50.0
    traditional_early_regime_rsi_short_max: float = 50.0
    traditional_early_regime_require_macd_acceleration: bool = True
    traditional_countertrend_cross_max_regime_gap_pct: float = 0.0
    traditional_countertrend_pullback_max_regime_gap_pct: float = 0.0
    traditional_neutral_transition_max_regime_gap_pct: float = 0.0
    enable_time_exit: bool = False
    time_exit_min_r: float = 0.5
    break_even_trigger_r: float = 1.25
    break_even_lock_r: float = 0.5
    trailing_trigger_r: float = 2.0
    trailing_distance_r: float = 0.75
    enable_profit_trend_exit: bool = True
    profit_trend_exit_trigger_r: float = 1.0
    profit_trend_exit_require_macd_flip: bool = True


@dataclass(frozen=True)
class _Features:
    close: float
    previous_close: float | None
    ema_fast: float | None
    previous_ema_fast: float | None
    ema_slow: float | None
    ema_regime: float | None
    rsi: float | None
    previous_rsi: float | None
    macd_histogram: float | None
    atr: float | None
    previous_high: float | None
    previous_low: float | None
    volume_ratio: float | None


@dataclass(frozen=True)
class _TraditionalFeatures:
    open: float
    high: float
    low: float
    close: float
    previous_close: float | None
    ema_fast: float | None
    previous_ema_fast: float | None
    ema_slow: float | None
    previous_ema_slow: float | None
    rsi: float | None
    macd_histogram: float | None
    previous_macd_histogram: float | None
    atr: float | None
    volume_ratio: float | None


@dataclass(frozen=True)
class _TraditionalSetupState:
    golden_cross: bool
    death_cross: bool
    pullback_long: bool
    pullback_short: bool
    breakout_long_raw: bool
    breakout_short_raw: bool
    breakout_long: bool
    breakout_short: bool

    @property
    def long_ready(self) -> bool:
        return self.golden_cross or self.pullback_long or self.breakout_long

    @property
    def short_ready(self) -> bool:
        return self.death_cross or self.pullback_short or self.breakout_short


def _features(candles: Sequence[Candle], config: StrategyConfig) -> _Features | None:
    if not candles:
        return None
    closes = [candle.close for candle in candles]
    highs = [candle.high for candle in candles]
    lows = [candle.low for candle in candles]
    ema_fast_values = ema(closes, config.ema_fast)
    ema_slow_values = ema(closes, config.ema_slow)
    ema_regime_values = ema(closes, config.ema_regime)
    _, _, histogram_values = macd(closes, config.macd_fast, config.macd_slow, config.macd_signal)
    rsi_values = rsi(closes, config.rsi_period)
    atr_values = atr(highs, lows, closes, config.atr_period)
    index = len(candles) - 1
    breakout_start = max(0, index - config.breakout_lookback)
    previous_high = max(highs[breakout_start:index], default=None)
    previous_low = min(lows[breakout_start:index], default=None)
    volume_start = max(0, index - config.volume_sma_period)
    previous_volumes = [_volume_value(candle) for candle in candles[volume_start:index]]
    average_volume = sum(previous_volumes) / len(previous_volumes) if previous_volumes else 0.0
    volume_ratio = _volume_value(candles[index]) / average_volume if average_volume > 0 else None
    return _Features(
        close=closes[index],
        previous_close=closes[index - 1] if index else None,
        ema_fast=ema_fast_values[index],
        previous_ema_fast=ema_fast_values[index - 1] if index else None,
        ema_slow=ema_slow_values[index],
        ema_regime=ema_regime_values[index],
        rsi=rsi_values[index],
        previous_rsi=rsi_values[index - 1] if index else None,
        macd_histogram=histogram_values[index],
        atr=atr_values[index],
        previous_high=previous_high,
        previous_low=previous_low,
        volume_ratio=volume_ratio,
    )


def dynamic_stop_loss_pct(
    candles: Sequence[Candle],
    config: StrategyConfig,
    fallback: float,
    *,
    side: str | None = None,
    entry_price: float | None = None,
    structure_lookback_bars: int | None = None,
    maximum_stop_loss_pct: float | None = None,
) -> float:
    """Return a volatility stop widened to a nearby market structure level.

    ATR remains the baseline.  When structure stops are enabled, a long stop
    sits below the lowest recent trigger-bar low and a short stop above the
    highest recent high, with an optional ATR buffer.  The configured maximum
    still caps the distance so one large wick cannot create unbounded risk.
    """
    fallback = float(fallback)
    minimum = float(getattr(config, "min_stop_loss_pct", fallback))
    maximum = float(
        maximum_stop_loss_pct
        if maximum_stop_loss_pct is not None
        else getattr(config, "max_stop_loss_pct", fallback)
    )
    if minimum <= 0 or maximum < minimum:
        return fallback

    period = max(1, int(getattr(config, "atr_period", 7)))
    if len(candles) < max(2, period):
        return fallback

    values = atr(
        [float(candle.high) for candle in candles],
        [float(candle.low) for candle in candles],
        [float(candle.close) for candle in candles],
        period,
    )
    current_atr = values[-1]
    close = float(candles[-1].close)
    if current_atr is None or close <= 0:
        return fallback

    atr_pct = (float(current_atr) / close) * float(getattr(config, "atr_stop_multiplier", 1.4))
    selected = max(minimum, min(maximum, atr_pct))

    lookback = max(
        0,
        int(
            structure_lookback_bars
            if structure_lookback_bars is not None
            else getattr(config, "structure_stop_lookback_bars", 0)
        ),
    )
    price = float(entry_price or 0.0)
    if lookback <= 0 or side not in {"long", "short"} or price <= 0:
        return selected

    window = candles[-lookback:]
    buffer_distance = float(current_atr) * max(0.0, float(getattr(config, "structure_stop_buffer_atr", 0.0)))
    if side == "long":
        structure_price = min(float(candle.low) for candle in window) - buffer_distance
        structure_pct = (price - structure_price) / price
    else:
        structure_price = max(float(candle.high) for candle in window) + buffer_distance
        structure_pct = (structure_price - price) / price
    if structure_pct > 0:
        selected = max(selected, min(maximum, structure_pct))
    return selected


class MultiTimeframeStrategy:
    """Multi-timeframe signal generator with an explicit traditional mode."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def evaluate(self, candles_by_timeframe: Mapping[str, Sequence[Candle]]) -> Signal:
        if self.config.mode == "traditional_kline":
            return self._evaluate_traditional(candles_by_timeframe, setup_valid_bars=1)
        trigger_name = self.config.trigger_timeframe
        regime_name = self.config.regime_timeframe
        trigger = _features(candles_by_timeframe.get(trigger_name, ()), self.config)
        one_minute = _features(candles_by_timeframe.get("1m", ()), self.config)
        regime = _features(candles_by_timeframe.get(regime_name, ()), self.config)
        if trigger is None or one_minute is None or regime is None:
            return Signal("flat", 0, 0, ("insufficient_candles",))

        timestamp = candles_by_timeframe[trigger_name][-1].timestamp
        long_score = 0
        short_score = 0
        long_reasons: list[str] = []
        short_reasons: list[str] = []

        regime_long = _bullish_regime(regime)
        regime_short = _bearish_regime(regime)
        if regime_long:
            long_score += 1
            long_reasons.append(f"{regime_name}_bull_regime")
        if regime_short:
            short_score += 1
            short_reasons.append(f"{regime_name}_bear_regime")

        momentum_long = _bullish_momentum(one_minute)
        momentum_short = _bearish_momentum(one_minute)
        if momentum_long:
            long_score += 1
            long_reasons.append("1m_bull_momentum")
        if momentum_short:
            short_score += 1
            short_reasons.append("1m_bear_momentum")

        trigger_long = _bullish_trigger(trigger)
        trigger_short = _bearish_trigger(trigger)
        if trigger_long:
            long_score += 2
            long_reasons.append(f"{trigger_name}_breakout_or_cross")
        if trigger_short:
            short_score += 2
            short_reasons.append(f"{trigger_name}_breakdown_or_cross")

        if trigger_long and _volume_confirmed(trigger, self.config.min_volume_ratio):
            long_score += 1
            long_reasons.append(f"{trigger_name}_volume")
        if trigger_short and _volume_confirmed(trigger, self.config.min_volume_ratio):
            short_score += 1
            short_reasons.append(f"{trigger_name}_volume")

        long_ready = long_score >= self.config.min_score and long_score > short_score
        short_ready = short_score >= self.config.min_score and short_score > long_score
        if self.config.require_full_alignment:
            volume_ready = _volume_confirmed(trigger, self.config.min_volume_ratio)
            long_ready = long_ready and regime_long and momentum_long and trigger_long and (not self.config.require_volume_confirmation or volume_ready)
            short_ready = short_ready and regime_short and momentum_short and trigger_short and (not self.config.require_volume_confirmation or volume_ready)

        if long_ready:
            return Signal("long", long_score, timestamp, tuple(long_reasons))
        if short_ready:
            return Signal("short", short_score, timestamp, tuple(short_reasons))
        return Signal("flat", max(long_score, short_score), timestamp, ("no_scalp_setup",))

    def reevaluate_blocked_signal(
        self,
        side: str,
        candles_by_timeframe: Mapping[str, Sequence[Candle]],
    ) -> Signal:
        """Revalidate a recent traditional setup after a macro blackout ends."""
        if self.config.mode != "traditional_kline" or side not in {"long", "short"}:
            return Signal("flat", 0, 0, ("blocked_signal_recheck_unsupported",))
        signal = self._evaluate_traditional(
            candles_by_timeframe,
            setup_valid_bars=max(1, int(self.config.traditional_blocked_setup_valid_bars)),
        )
        if signal.side != side:
            return Signal(
                "flat",
                signal.score,
                signal.timestamp,
                tuple(signal.reasons) + ("blocked_signal_expired_or_invalid",),
            )
        return Signal(
            signal.side,
            signal.score,
            signal.timestamp,
            tuple(signal.reasons) + ("macro_blocked_signal_revalidated",),
        )

    def position_trend_invalidated(
        self,
        side: str,
        candles_by_timeframe: Mapping[str, Sequence[Candle]],
    ) -> bool:
        """Detect a closed 5-minute price-and-momentum reversal after profit."""
        if self.config.mode != "traditional_kline" or side not in {"long", "short"}:
            return False
        trigger_candles = candles_by_timeframe.get(self.config.trigger_timeframe, ())
        minimum = max(
            self.config.traditional_signal_slow,
            self.config.traditional_macd_slow + self.config.traditional_macd_signal,
            self.config.traditional_atr_period + 1,
            self.config.traditional_volume_sma_period + 1,
        )
        if len(trigger_candles) < minimum:
            return False
        feature = _traditional_features(
            trigger_candles,
            self.config.traditional_signal_fast,
            self.config.traditional_signal_slow,
            self.config.traditional_rsi_period,
            self.config.traditional_macd_fast,
            self.config.traditional_macd_slow,
            self.config.traditional_macd_signal,
            self.config.traditional_atr_period,
            self.config.traditional_volume_sma_period,
        )
        if feature.ema_fast is None or feature.macd_histogram is None:
            return False
        price_reversed = feature.close < feature.ema_fast if side == "long" else feature.close > feature.ema_fast
        momentum_reversed = feature.macd_histogram < 0 if side == "long" else feature.macd_histogram > 0
        if self.config.profit_trend_exit_require_macd_flip:
            return price_reversed and momentum_reversed
        return price_reversed

    def _evaluate_traditional(
        self,
        candles_by_timeframe: Mapping[str, Sequence[Candle]],
        *,
        setup_valid_bars: int = 1,
    ) -> Signal:
        """Require a closed-bar trend, entry setup, momentum, RSI, volume and execution.

        The higher timeframe supplies direction, the signal timeframe supplies
        a cross, pullback reclaim or breakout, and the 1-minute chart supplies
        execution confirmation. Every input has already had its newest forming
        candle removed by the engine.
        """
        trigger_name = self.config.trigger_timeframe
        regime_name = self.config.regime_timeframe
        trigger_candles = candles_by_timeframe.get(trigger_name, ())
        regime_candles = candles_by_timeframe.get(regime_name, ())
        one_minute_candles = candles_by_timeframe.get("1m", ())
        minimum_trigger = max(
            self.config.traditional_signal_slow,
            self.config.traditional_macd_slow + self.config.traditional_macd_signal,
            self.config.traditional_rsi_period + 1,
            self.config.traditional_atr_period + 1,
            self.config.traditional_volume_sma_period + 1,
        )
        if (
            len(trigger_candles) < minimum_trigger
            or len(regime_candles) < self.config.traditional_trend_slow
            or len(one_minute_candles) < self.config.traditional_signal_slow
        ):
            return Signal("flat", 0, 0, ("insufficient_candles",))

        trigger = _traditional_features(
            trigger_candles,
            self.config.traditional_signal_fast,
            self.config.traditional_signal_slow,
            self.config.traditional_rsi_period,
            self.config.traditional_macd_fast,
            self.config.traditional_macd_slow,
            self.config.traditional_macd_signal,
            self.config.traditional_atr_period,
            self.config.traditional_volume_sma_period,
        )
        execution = _traditional_features(
            one_minute_candles,
            self.config.traditional_signal_fast,
            self.config.traditional_signal_slow,
            self.config.traditional_rsi_period,
            self.config.traditional_macd_fast,
            self.config.traditional_macd_slow,
            self.config.traditional_macd_signal,
            self.config.traditional_atr_period,
            self.config.traditional_volume_sma_period,
        )
        regime = _traditional_features(
            regime_candles,
            self.config.traditional_trend_fast,
            self.config.traditional_trend_slow,
            self.config.traditional_rsi_period,
            self.config.traditional_macd_fast,
            self.config.traditional_macd_slow,
            self.config.traditional_macd_signal,
            self.config.traditional_atr_period,
            self.config.traditional_volume_sma_period,
        )
        if regime.ema_fast is None or regime.ema_slow is None:
            return Signal("flat", 0, 0, ("insufficient_trend_warmup",))

        timestamp = trigger_candles[-1].timestamp
        base_strong_trend_long = regime.close > regime.ema_fast > regime.ema_slow
        base_strong_trend_short = regime.close < regime.ema_fast < regime.ema_slow
        strong_trend_long = base_strong_trend_long and _traditional_strong_regime_quality(
            regime,
            "long",
            self.config,
        )
        strong_trend_short = base_strong_trend_short and _traditional_strong_regime_quality(
            regime,
            "short",
            self.config,
        )
        early_trend_long = (
            self.config.traditional_allow_early_regime
            and not base_strong_trend_long
            and _traditional_early_regime(regime, "long", self.config)
        )
        early_trend_short = (
            self.config.traditional_allow_early_regime
            and not base_strong_trend_short
            and _traditional_early_regime(regime, "short", self.config)
        )
        current_setup = _traditional_setup_state(trigger_candles, self.config, feature=trigger)
        previous_trigger = _traditional_features(
            trigger_candles[:-1],
            self.config.traditional_signal_fast,
            self.config.traditional_signal_slow,
            self.config.traditional_rsi_period,
            self.config.traditional_macd_fast,
            self.config.traditional_macd_slow,
            self.config.traditional_macd_signal,
            self.config.traditional_atr_period,
            self.config.traditional_volume_sma_period,
        )
        previous_setup = _traditional_setup_state(
            trigger_candles[:-1],
            self.config,
            feature=previous_trigger,
        )
        long_macd_handoff = _traditional_setup_macd_handoff(
            previous_trigger,
            trigger,
            previous_setup,
            "long",
            self.config,
        ) and not current_setup.short_ready
        short_macd_handoff = _traditional_setup_macd_handoff(
            previous_trigger,
            trigger,
            previous_setup,
            "short",
            self.config,
        ) and not current_setup.long_ready
        long_volume_handoff = _traditional_setup_volume_handoff(
            previous_trigger,
            trigger,
            previous_setup,
            "long",
            self.config,
        ) and not current_setup.short_ready
        short_volume_handoff = _traditional_setup_volume_handoff(
            previous_trigger,
            trigger,
            previous_setup,
            "short",
            self.config,
        ) and not current_setup.long_ready
        setup_valid_bars = max(1, int(setup_valid_bars))
        long_setup_age: int | None = None
        short_setup_age: int | None = None
        long_setup_state: _TraditionalSetupState | None = None
        short_setup_state: _TraditionalSetupState | None = None
        for age in range(setup_valid_bars):
            history = trigger_candles if age == 0 else trigger_candles[:-age]
            if len(history) < minimum_trigger:
                break
            setup = current_setup if age == 0 else _traditional_setup_state(history, self.config)
            if long_setup_age is None and setup.long_ready:
                long_setup_age = age
                long_setup_state = setup
            if short_setup_age is None and setup.short_ready:
                short_setup_age = age
                short_setup_state = setup

        # An opposite setup inside the validity window invalidates the older one.
        if long_setup_age is not None and short_setup_age is not None:
            if long_setup_age < short_setup_age:
                short_setup_age = None
                short_setup_state = None
            elif short_setup_age < long_setup_age:
                long_setup_age = None
                long_setup_state = None

        trigger_long_setup = long_setup_age is not None or long_macd_handoff or long_volume_handoff
        trigger_short_setup = short_setup_age is not None or short_macd_handoff or short_volume_handoff
        macd_long = trigger.macd_histogram is not None and trigger.macd_histogram > 0
        macd_short = trigger.macd_histogram is not None and trigger.macd_histogram < 0
        rsi_long = trigger.rsi is not None and self.config.traditional_rsi_long_min <= trigger.rsi <= self.config.traditional_rsi_long_max
        rsi_short = trigger.rsi is not None and self.config.traditional_rsi_short_min <= trigger.rsi <= self.config.traditional_rsi_short_max
        volume_ready = trigger.volume_ratio is not None and trigger.volume_ratio >= self.config.traditional_min_volume_ratio
        execution_long_alignment = _traditional_execution(execution, "long")
        execution_short_alignment = _traditional_execution(execution, "short")
        if self.config.traditional_allow_pullback:
            execution_long_alignment = execution_long_alignment or _traditional_reclaim(execution, "long")
            execution_short_alignment = execution_short_alignment or _traditional_reclaim(execution, "short")
        execution_long_quality = _traditional_execution_quality(execution, "long", self.config)
        execution_short_quality = _traditional_execution_quality(execution, "short", self.config)
        execution_long = execution_long_alignment and execution_long_quality
        execution_short = execution_short_alignment and execution_short_quality
        if not self.config.traditional_require_1m_confirmation:
            execution_long = execution_short = True

        pressure_long, pressure_long_reason = _traditional_pressure_room(
            trigger_candles,
            "long",
            self.config,
        )
        pressure_short, pressure_short_reason = _traditional_pressure_room(
            trigger_candles,
            "short",
            self.config,
        )

        countertrend_cross_long = _traditional_countertrend_cross_regime(
            regime,
            trigger,
            current_setup,
            "long",
            self.config,
        )
        countertrend_cross_short = _traditional_countertrend_cross_regime(
            regime,
            trigger,
            current_setup,
            "short",
            self.config,
        )
        countertrend_pullback_long = _traditional_countertrend_pullback_regime(
            regime,
            trigger,
            current_setup,
            "long",
            self.config,
        )
        countertrend_pullback_short = _traditional_countertrend_pullback_regime(
            regime,
            trigger,
            current_setup,
            "short",
            self.config,
        )
        neutral_transition_long = _traditional_neutral_transition_regime(
            regime,
            trigger,
            current_setup,
            "long",
            self.config,
        )
        neutral_transition_short = _traditional_neutral_transition_regime(
            regime,
            trigger,
            current_setup,
            "short",
            self.config,
        )
        trend_long = (
            strong_trend_long
            or early_trend_long
            or countertrend_cross_long
            or countertrend_pullback_long
            or neutral_transition_long
        )
        trend_short = (
            strong_trend_short
            or early_trend_short
            or countertrend_cross_short
            or countertrend_pullback_short
            or neutral_transition_short
        )

        long_checks = {
            f"{regime_name}_trend_up": trend_long,
            f"{trigger_name}_entry_setup": trigger_long_setup,
            f"{trigger_name}_macd_positive": macd_long,
            f"{trigger_name}_rsi_long_zone": rsi_long,
            f"{trigger_name}_volume": volume_ready,
            "1m_execution_up": execution_long,
            "10m_15m_pressure_clear": pressure_long,
        }
        short_checks = {
            f"{regime_name}_trend_down": trend_short,
            f"{trigger_name}_entry_setup": trigger_short_setup,
            f"{trigger_name}_macd_negative": macd_short,
            f"{trigger_name}_rsi_short_zone": rsi_short,
            f"{trigger_name}_volume": volume_ready,
            "1m_execution_down": execution_short,
            "10m_15m_pressure_clear": pressure_short,
        }
        long_score = sum(long_checks.values())
        short_score = sum(short_checks.values())
        if long_score == len(long_checks) and long_score > short_score:
            reasons = tuple(long_checks)
            selected_setup = long_setup_state or current_setup
            setup_reasons = tuple(
                name
                for name, ready in (
                    (f"{trigger_name}_golden_cross", selected_setup.golden_cross),
                    (f"{trigger_name}_pullback_reclaim", selected_setup.pullback_long),
                    (f"{trigger_name}_breakout", selected_setup.breakout_long),
                )
                if ready
            )
            reasons += setup_reasons
            if long_macd_handoff and long_setup_age is None:
                reasons += (f"{trigger_name}_setup_macd_handoff",)
            if long_volume_handoff and long_setup_age is None:
                reasons += (f"{trigger_name}_setup_volume_handoff",)
            if long_setup_age:
                reasons += (f"{trigger_name}_setup_persisted_{long_setup_age}_bars",)
            if early_trend_long and not strong_trend_long:
                reasons += (f"{regime_name}_early_trend_up",)
            if countertrend_cross_long:
                reasons += (f"{regime_name}_countertrend_cross_up",)
            if countertrend_pullback_long:
                reasons += (f"{regime_name}_countertrend_pullback_up",)
            if neutral_transition_long:
                reasons += (f"{regime_name}_neutral_transition_up",)
            return Signal("long", long_score, timestamp, reasons)
        if short_score == len(short_checks) and short_score > long_score:
            reasons = tuple(short_checks)
            selected_setup = short_setup_state or current_setup
            setup_reasons = tuple(
                name
                for name, ready in (
                    (f"{trigger_name}_death_cross", selected_setup.death_cross),
                    (f"{trigger_name}_pullback_reclaim", selected_setup.pullback_short),
                    (f"{trigger_name}_breakdown", selected_setup.breakout_short),
                )
                if ready
            )
            reasons += setup_reasons
            if short_macd_handoff and short_setup_age is None:
                reasons += (f"{trigger_name}_setup_macd_handoff",)
            if short_volume_handoff and short_setup_age is None:
                reasons += (f"{trigger_name}_setup_volume_handoff",)
            if short_setup_age:
                reasons += (f"{trigger_name}_setup_persisted_{short_setup_age}_bars",)
            if early_trend_short and not strong_trend_short:
                reasons += (f"{regime_name}_early_trend_down",)
            if countertrend_cross_short:
                reasons += (f"{regime_name}_countertrend_cross_down",)
            if countertrend_pullback_short:
                reasons += (f"{regime_name}_countertrend_pullback_down",)
            if neutral_transition_short:
                reasons += (f"{regime_name}_neutral_transition_down",)
            return Signal("short", short_score, timestamp, reasons)

        failed_breakout_short = _traditional_failed_breakout_short_reversal(
            trigger_candles,
            regime,
            previous_trigger,
            trigger,
            execution,
            self.config,
        )
        if (
            self.config.traditional_failed_breakout_short_enabled
            and failed_breakout_short.side == "short"
            and pressure_short
        ):
            return failed_breakout_short

        predictive_reversal_short = _traditional_predictive_reversal_short(
            one_minute_candles,
            trigger_candles,
            self.config,
        )
        if predictive_reversal_short.side == "short":
            return predictive_reversal_short

        # A fast impulse branch closes the timing gap between a quiet completed
        # 5-minute context bar and the next 5-minute bar becoming overextended.
        # It still uses only closed candles: the higher timeframes establish a
        # strong directional context, while a fresh 1-minute range break supplies
        # the entry.  Normal 5-minute signals retain priority above this branch.
        impulse_long = (
            self.config.traditional_allow_1m_impulse
            and self.config.traditional_1m_impulse_allow_long
            and strong_trend_long
            and macd_long
            and rsi_long
            and pressure_long
            and _traditional_one_minute_impulse(
                one_minute_candles,
                execution,
                trigger,
                "long",
                self.config,
            )
        )
        impulse_short = (
            self.config.traditional_allow_1m_impulse
            and self.config.traditional_1m_impulse_allow_short
            and strong_trend_short
            and macd_short
            and rsi_short
            and pressure_short
            and _traditional_one_minute_impulse(
                one_minute_candles,
                execution,
                trigger,
                "short",
                self.config,
            )
        )
        if impulse_long and not impulse_short:
            return Signal(
                "long",
                6,
                one_minute_candles[-1].timestamp,
                (
                    f"{regime_name}_strong_trend_up",
                    f"{trigger_name}_macd_positive",
                    f"{trigger_name}_rsi_long_zone",
                    "1m_impulse_breakout",
                    "1m_impulse_quality",
                    "1m_execution_up",
                ),
            )
        if impulse_short and not impulse_long:
            return Signal(
                "short",
                6,
                one_minute_candles[-1].timestamp,
                (
                    f"{regime_name}_strong_trend_down",
                    f"{trigger_name}_macd_negative",
                    f"{trigger_name}_rsi_short_zone",
                    "1m_impulse_breakdown",
                    "1m_impulse_quality",
                    "1m_execution_down",
                ),
            )
        long_missing = ",".join(name for name, ready in long_checks.items() if not ready)
        short_missing = ",".join(name for name, ready in short_checks.items() if not ready)
        quality_rejections: list[str] = []
        if current_setup.breakout_long_raw and not current_setup.breakout_long:
            quality_rejections.append(f"{trigger_name}_long_breakout_quality_rejected")
        if current_setup.breakout_short_raw and not current_setup.breakout_short:
            quality_rejections.append(f"{trigger_name}_short_breakout_quality_rejected")
        if execution_long_alignment and not execution_long_quality:
            quality_rejections.append("1m_long_execution_quality_rejected")
        if execution_short_alignment and not execution_short_quality:
            quality_rejections.append("1m_short_execution_quality_rejected")
        if not pressure_long:
            quality_rejections.append(f"long_pressure_blocked={pressure_long_reason}")
        if not pressure_short:
            quality_rejections.append(f"short_pressure_blocked={pressure_short_reason}")
        if (
            self.config.traditional_failed_breakout_short_enabled
            and failed_breakout_short.side == "short"
            and not pressure_short
        ):
            quality_rejections.append("failed_breakout_short_pressure_rejected")
        if (
            self.config.traditional_failed_breakout_short_shadow
            and not self.config.traditional_failed_breakout_short_enabled
            and failed_breakout_short.side == "short"
        ):
            quality_rejections.append("shadow_candidate=short")
            quality_rejections.extend(
                f"shadow_{reason}" for reason in failed_breakout_short.reasons
            )
        return Signal(
            "flat",
            max(long_score, short_score),
            timestamp,
            (
                "no_traditional_setup",
                f"long_missing={long_missing}",
                f"short_missing={short_missing}",
            )
            + tuple(quality_rejections),
        )


def _bullish_regime(feature: _Features) -> bool:
    return (
        feature.ema_fast is not None
        and feature.ema_slow is not None
        and feature.close >= feature.ema_slow
        and feature.ema_fast >= feature.ema_slow
    )


def _bearish_regime(feature: _Features) -> bool:
    return (
        feature.ema_fast is not None
        and feature.ema_slow is not None
        and feature.close <= feature.ema_slow
        and feature.ema_fast <= feature.ema_slow
    )


def _bullish_momentum(feature: _Features) -> bool:
    return (
        feature.ema_fast is not None
        and feature.ema_slow is not None
        and feature.close >= feature.ema_fast
        and feature.ema_fast >= feature.ema_slow
        and feature.rsi is not None
        and 45 <= feature.rsi <= 88
    )


def _bearish_momentum(feature: _Features) -> bool:
    return (
        feature.ema_fast is not None
        and feature.ema_slow is not None
        and feature.close <= feature.ema_fast
        and feature.ema_fast <= feature.ema_slow
        and feature.rsi is not None
        and 12 <= feature.rsi <= 55
    )


def _bullish_trigger(feature: _Features) -> bool:
    breakout = feature.previous_high is not None and feature.close > feature.previous_high
    cross = (
        feature.ema_fast is not None
        and feature.previous_ema_fast is not None
        and feature.previous_close is not None
        and feature.previous_close <= feature.previous_ema_fast
        and feature.close > feature.ema_fast
    )
    return breakout or cross


def _bearish_trigger(feature: _Features) -> bool:
    breakdown = feature.previous_low is not None and feature.close < feature.previous_low
    cross = (
        feature.ema_fast is not None
        and feature.previous_ema_fast is not None
        and feature.previous_close is not None
        and feature.previous_close >= feature.previous_ema_fast
        and feature.close < feature.ema_fast
    )
    return breakdown or cross


def _volume_confirmed(feature: _Features, minimum_ratio: float) -> bool:
    return feature.volume_ratio is not None and feature.volume_ratio >= minimum_ratio


def _volume_value(candle: Candle) -> float:
    """Use quote turnover when available, otherwise the adapter's volume."""
    if candle.quote_volume is not None and candle.quote_volume > 0:
        return candle.quote_volume
    return candle.volume


def _traditional_features(
    candles: Sequence[Candle],
    fast_period: int,
    slow_period: int,
    rsi_period: int,
    macd_fast_period: int,
    macd_slow_period: int,
    macd_signal_period: int,
    atr_period: int,
    volume_period: int,
) -> _TraditionalFeatures:
    closes = [candle.close for candle in candles]
    highs = [candle.high for candle in candles]
    lows = [candle.low for candle in candles]
    fast_values = ema(closes, fast_period)
    slow_values = ema(closes, slow_period)
    _, _, histogram_values = macd(closes, macd_fast_period, macd_slow_period, macd_signal_period)
    rsi_values = rsi(closes, rsi_period)
    atr_values = atr(highs, lows, closes, atr_period)
    index = len(candles) - 1
    volume_start = max(0, index - volume_period)
    previous_volumes = [_volume_value(candle) for candle in candles[volume_start:index]]
    average_volume = sum(previous_volumes) / len(previous_volumes) if previous_volumes else 0.0
    volume_ratio = _volume_value(candles[index]) / average_volume if average_volume > 0 else None
    return _TraditionalFeatures(
        open=candles[index].open,
        high=highs[index],
        low=lows[index],
        close=closes[index],
        previous_close=closes[index - 1] if index else None,
        ema_fast=fast_values[index],
        previous_ema_fast=fast_values[index - 1] if index else None,
        ema_slow=slow_values[index],
        previous_ema_slow=slow_values[index - 1] if index else None,
        rsi=rsi_values[index],
        macd_histogram=histogram_values[index],
        previous_macd_histogram=histogram_values[index - 1] if index else None,
        atr=atr_values[index],
        volume_ratio=volume_ratio,
    )


def _traditional_setup_state(
    candles: Sequence[Candle],
    config: StrategyConfig,
    *,
    feature: _TraditionalFeatures | None = None,
) -> _TraditionalSetupState:
    selected = feature or _traditional_features(
        candles,
        config.traditional_signal_fast,
        config.traditional_signal_slow,
        config.traditional_rsi_period,
        config.traditional_macd_fast,
        config.traditional_macd_slow,
        config.traditional_macd_signal,
        config.traditional_atr_period,
        config.traditional_volume_sma_period,
    )
    golden_cross_raw = (
        selected.previous_ema_fast is not None
        and selected.previous_ema_slow is not None
        and selected.ema_fast is not None
        and selected.ema_slow is not None
        and selected.previous_ema_fast <= selected.previous_ema_slow
        and selected.ema_fast > selected.ema_slow
    )
    death_cross_raw = (
        selected.previous_ema_fast is not None
        and selected.previous_ema_slow is not None
        and selected.ema_fast is not None
        and selected.ema_slow is not None
        and selected.previous_ema_fast >= selected.previous_ema_slow
        and selected.ema_fast < selected.ema_slow
    )
    golden_cross = golden_cross_raw and _traditional_cross_quality(selected, "long", config)
    death_cross = death_cross_raw and _traditional_cross_quality(selected, "short", config)
    pullback_long = config.traditional_allow_pullback and _traditional_reclaim(selected, "long")
    pullback_short = config.traditional_allow_pullback and _traditional_reclaim(selected, "short")
    breakout_lookback = max(2, int(config.traditional_breakout_lookback))
    breakout_history = candles[max(0, len(candles) - breakout_lookback - 1) : -1]
    breakout_long_raw = (
        config.traditional_allow_breakout
        and bool(breakout_history)
        and selected.ema_fast is not None
        and selected.ema_slow is not None
        and selected.ema_fast >= selected.ema_slow
        and selected.close > max(candle.high for candle in breakout_history)
    )
    breakout_short_raw = (
        config.traditional_allow_breakout
        and bool(breakout_history)
        and selected.ema_fast is not None
        and selected.ema_slow is not None
        and selected.ema_fast <= selected.ema_slow
        and selected.close < min(candle.low for candle in breakout_history)
    )
    return _TraditionalSetupState(
        golden_cross=golden_cross,
        death_cross=death_cross,
        pullback_long=pullback_long,
        pullback_short=pullback_short,
        breakout_long_raw=breakout_long_raw,
        breakout_short_raw=breakout_short_raw,
        breakout_long=breakout_long_raw and _traditional_breakout_quality(selected, "long", config),
        breakout_short=breakout_short_raw and _traditional_breakout_quality(selected, "short", config),
    )


def _traditional_setup_macd_handoff(
    previous: _TraditionalFeatures,
    current: _TraditionalFeatures,
    previous_setup: _TraditionalSetupState,
    side: str,
    config: StrategyConfig,
) -> bool:
    """Allow one bar for MACD to confirm a fresh setup without enabling stale setups."""
    maximum_extension = max(
        0.0,
        float(config.traditional_setup_macd_handoff_max_extension_atr),
    )
    if (
        maximum_extension <= 0
        or previous.macd_histogram is None
        or previous.rsi is None
        or previous.volume_ratio is None
        or current.macd_histogram is None
        or current.ema_fast is None
        or current.atr is None
        or current.atr <= 0
    ):
        return False
    extension_atr = abs(current.close - current.ema_fast) / current.atr
    if extension_atr > maximum_extension:
        return False
    previous_volume_ready = previous.volume_ratio >= config.traditional_min_volume_ratio
    if side == "long":
        previous_rsi_ready = (
            config.traditional_rsi_long_min
            <= previous.rsi
            <= config.traditional_rsi_long_max
        )
        return (
            previous_setup.long_ready
            and previous.macd_histogram <= 0
            and current.macd_histogram > 0
            and previous_rsi_ready
            and previous_volume_ready
        )
    previous_rsi_ready = (
        config.traditional_rsi_short_min
        <= previous.rsi
        <= config.traditional_rsi_short_max
    )
    return (
        previous_setup.short_ready
        and previous.macd_histogram >= 0
        and current.macd_histogram < 0
        and previous_rsi_ready
        and previous_volume_ready
    )


def _traditional_setup_volume_handoff(
    previous: _TraditionalFeatures,
    current: _TraditionalFeatures,
    previous_setup: _TraditionalSetupState,
    side: str,
    config: StrategyConfig,
) -> bool:
    """Carry a raw breakout forward one bar only when volume arrives safely.

    This branch is disabled at zero. It does not persist crosses or pullbacks,
    and it requires the original breakout bar to have failed the normal volume
    floor. The current bar must continue in the same direction while every
    normal trend, MACD, RSI, volume and execution gate is still checked by the
    caller.
    """
    maximum_extension = max(
        0.0,
        float(config.traditional_setup_volume_handoff_max_extension_atr),
    )
    if (
        maximum_extension <= 0
        or previous.ema_fast is None
        or previous.atr is None
        or previous.atr <= 0
        or previous.volume_ratio is None
        or current.ema_fast is None
        or current.atr is None
        or current.atr <= 0
        or current.volume_ratio is None
    ):
        return False
    previous_raw = previous_setup.breakout_long_raw if side == "long" else previous_setup.breakout_short_raw
    previous_qualified = previous_setup.breakout_long if side == "long" else previous_setup.breakout_short
    if not previous_raw or previous_qualified:
        return False
    if previous.volume_ratio >= config.traditional_min_volume_ratio:
        return False
    if current.volume_ratio < config.traditional_min_volume_ratio:
        return False
    previous_extension = abs(previous.close - previous.ema_fast) / previous.atr
    current_extension = abs(current.close - current.ema_fast) / current.atr
    if (
        previous_extension > config.traditional_breakout_max_extension_atr
        or current_extension > maximum_extension
    ):
        return False
    candle_range = previous.high - previous.low
    if candle_range <= 0:
        return False
    close_location = (
        (previous.close - previous.low) / candle_range
        if side == "long"
        else (previous.high - previous.close) / candle_range
    )
    continued = current.close > previous.close if side == "long" else current.close < previous.close
    return close_location >= config.traditional_breakout_min_close_location and continued


def _traditional_strong_regime_quality(
    feature: _TraditionalFeatures,
    side: str,
    config: StrategyConfig,
) -> bool:
    """Require a directional 1h regime to have measurable strength.

    The legacy strong-regime check only verified price/EMA ordering. That
    ordering can remain bullish or bearish for hours after momentum has
    flattened, which lets repeated lower-timeframe continuation signals enter
    during a range. All thresholds default to disabled for backward
    compatibility and can be enabled independently after validation.
    """
    if (
        side not in {"long", "short"}
        or feature.ema_fast is None
        or feature.ema_slow is None
        or feature.atr is None
        or feature.atr <= 0
    ):
        return False

    gap_atr = abs(feature.ema_fast - feature.ema_slow) / feature.atr
    if gap_atr < max(0.0, float(config.traditional_strong_regime_min_gap_atr)):
        return False

    if config.traditional_strong_regime_require_fast_slope:
        if feature.previous_ema_fast is None:
            return False
        if side == "long" and feature.ema_fast <= feature.previous_ema_fast:
            return False
        if side == "short" and feature.ema_fast >= feature.previous_ema_fast:
            return False

    if config.traditional_strong_regime_require_macd:
        if feature.macd_histogram is None:
            return False
        if side == "long" and feature.macd_histogram <= 0:
            return False
        if side == "short" and feature.macd_histogram >= 0:
            return False
    return True


def _traditional_cross_quality(
    feature: _TraditionalFeatures,
    side: str,
    config: StrategyConfig,
) -> bool:
    """Reject weak, wick-heavy or already extended EMA crosses."""
    if side not in {"long", "short"}:
        return False
    candle_range = feature.high - feature.low
    if candle_range <= 0:
        return False
    body_ratio = abs(feature.close - feature.open) / candle_range
    close_location = (
        (feature.close - feature.low) / candle_range
        if side == "long"
        else (feature.high - feature.close) / candle_range
    )
    if body_ratio < max(0.0, float(config.traditional_cross_min_body_ratio)):
        return False
    if close_location < max(0.0, float(config.traditional_cross_min_close_location)):
        return False

    extension_cap = max(0.0, float(config.traditional_cross_max_extension_atr))
    if extension_cap:
        if feature.ema_fast is None or feature.atr is None or feature.atr <= 0:
            return False
        extension_atr = abs(feature.close - feature.ema_fast) / feature.atr
        if extension_atr > extension_cap:
            return False
    return True


def _traditional_early_regime(feature: _TraditionalFeatures, side: str, config: StrategyConfig) -> bool:
    """Allow an early trend only near the higher-timeframe EMA transition.

    This rejects counter-trend entries while the fast and slow regime EMAs are
    still widely separated, and requires momentum to strengthen rather than
    merely remain on the correct side of zero.
    """
    if (
        feature.ema_fast is None
        or feature.ema_slow is None
        or feature.previous_ema_fast is None
        or feature.macd_histogram is None
        or feature.rsi is None
    ):
        return False
    gap_limit = max(0.0, float(config.traditional_early_regime_max_gap_pct))
    require_acceleration = bool(config.traditional_early_regime_require_macd_acceleration)
    if side == "long":
        gap_ready = feature.ema_fast >= feature.ema_slow * (1 - gap_limit)
        momentum_ready = feature.macd_histogram > 0
        if require_acceleration:
            momentum_ready = (
                feature.previous_macd_histogram is not None
                and feature.macd_histogram >= feature.previous_macd_histogram
            )
        return (
            feature.close > feature.ema_fast
            and feature.ema_fast > feature.previous_ema_fast
            and feature.rsi >= config.traditional_early_regime_rsi_long_min
            and gap_ready
            and momentum_ready
        )
    gap_ready = feature.ema_fast <= feature.ema_slow * (1 + gap_limit)
    momentum_ready = feature.macd_histogram < 0
    if require_acceleration:
        momentum_ready = (
            feature.previous_macd_histogram is not None
            and feature.macd_histogram <= feature.previous_macd_histogram
        )
    return (
        feature.close < feature.ema_fast
        and feature.ema_fast < feature.previous_ema_fast
        and feature.rsi <= config.traditional_early_regime_rsi_short_max
        and gap_ready
        and momentum_ready
    )


def _traditional_countertrend_cross_regime(
    regime: _TraditionalFeatures,
    trigger: _TraditionalFeatures,
    setup: _TraditionalSetupState,
    side: str,
    config: StrategyConfig,
) -> bool:
    """Allow only a fresh, high-volume cross near a higher-timeframe transition.

    The branch is disabled at zero. It deliberately excludes pullbacks and
    breakouts so widening the transition allowance cannot admit late trend
    chases such as a high-volume pullback at the end of an impulse.
    """
    regime_gap_cap = max(0.0, float(config.traditional_countertrend_cross_max_regime_gap_pct))
    if (
        regime_gap_cap <= 0
        or regime.ema_fast is None
        or regime.ema_slow is None
        or regime.rsi is None
        or trigger.ema_fast is None
        or trigger.atr is None
        or trigger.atr <= 0
        or trigger.volume_ratio is None
    ):
        return False
    regime_gap = abs(regime.ema_fast - regime.ema_slow) / regime.ema_slow
    price_gap = abs(regime.close - regime.ema_fast) / regime.ema_fast
    price_gap_cap = max(0.0, float(config.traditional_early_regime_max_gap_pct))
    extension_cap = max(
        0.1,
        float(config.traditional_setup_macd_handoff_max_extension_atr)
        or float(config.traditional_breakout_max_extension_atr),
    )
    trigger_extension = abs(trigger.close - trigger.ema_fast) / trigger.atr
    volume_floor = max(float(config.traditional_min_volume_ratio), 2.0)
    shared_ready = (
        regime_gap <= regime_gap_cap
        and price_gap <= price_gap_cap
        and trigger_extension <= extension_cap
        and trigger.volume_ratio >= volume_floor
    )
    if not shared_ready:
        return False
    if side == "long":
        return (
            setup.golden_cross
            and regime.ema_fast < regime.ema_slow
            and regime.close < regime.ema_fast
            and regime.rsi >= 45.0
        )
    return (
        setup.death_cross
        and regime.ema_fast > regime.ema_slow
        and regime.close > regime.ema_fast
        and regime.rsi <= 55.0
    )


def _traditional_countertrend_pullback_regime(
    regime: _TraditionalFeatures,
    trigger: _TraditionalFeatures,
    setup: _TraditionalSetupState,
    side: str,
    config: StrategyConfig,
) -> bool:
    """Allow a fresh, bounded 5m pullback reclaim inside a 1h transition.

    This is intentionally narrower than relaxing the higher-timeframe trend
    gate: the branch is disabled at zero, requires elevated volume, rejects
    entries more than one ATR from the 5m fast EMA, and only accepts a closed
    1h price close to its fast EMA while the regime EMAs remain near transition.
    Position sizing is reduced separately for signals tagged by this branch.
    """
    regime_gap_cap = max(
        0.0,
        float(config.traditional_countertrend_pullback_max_regime_gap_pct),
    )
    if (
        regime_gap_cap <= 0
        or regime.ema_fast is None
        or regime.ema_slow is None
        or regime.rsi is None
        or trigger.ema_fast is None
        or trigger.atr is None
        or trigger.atr <= 0
        or trigger.volume_ratio is None
    ):
        return False
    regime_gap = abs(regime.ema_fast - regime.ema_slow) / regime.ema_slow
    price_gap = abs(regime.close - regime.ema_fast) / regime.ema_fast
    price_gap_cap = min(regime_gap_cap, 0.0075)
    trigger_extension = abs(trigger.close - trigger.ema_fast) / trigger.atr
    shared_ready = (
        regime_gap <= regime_gap_cap
        and price_gap <= price_gap_cap
        and trigger_extension <= 1.0
        and trigger.volume_ratio >= max(float(config.traditional_min_volume_ratio), 1.5)
    )
    if not shared_ready:
        return False
    if side == "long":
        return (
            setup.pullback_long
            and not setup.pullback_short
            and regime.ema_fast < regime.ema_slow
            and regime.close < regime.ema_fast
            and 40.0 <= regime.rsi <= 45.0
            and regime.macd_histogram is not None
            and regime.macd_histogram < 0
        )
    return (
        setup.pullback_short
        and not setup.pullback_long
        and regime.ema_fast > regime.ema_slow
        and regime.close > regime.ema_fast
        and 55.0 <= regime.rsi <= 60.0
        and regime.macd_histogram is not None
        and regime.macd_histogram > 0
    )


def _traditional_neutral_transition_regime(
    regime: _TraditionalFeatures,
    trigger: _TraditionalFeatures,
    setup: _TraditionalSetupState,
    side: str,
    config: StrategyConfig,
) -> bool:
    """Allow a fresh high-volume setup while the closed 1h regime is crossing.

    The transition state exists when price has crossed the 1h EMA50 but the
    EMA50/EMA200 ordering still points the other way. This branch is disabled
    at zero and deliberately requires neutral 1h RSI, proximity to EMA50,
    elevated 5m volume and a bounded 5m extension. The caller still enforces
    direction-specific 5m MACD/RSI and 1m execution confirmation.
    """
    regime_gap_cap = max(
        0.0,
        float(config.traditional_neutral_transition_max_regime_gap_pct),
    )
    if (
        regime_gap_cap <= 0
        or regime.ema_fast is None
        or regime.ema_slow is None
        or regime.rsi is None
        or trigger.ema_fast is None
        or trigger.atr is None
        or trigger.atr <= 0
        or trigger.volume_ratio is None
    ):
        return False
    regime_gap = abs(regime.ema_fast - regime.ema_slow) / regime.ema_slow
    price_gap = abs(regime.close - regime.ema_fast) / regime.ema_fast
    trigger_extension = abs(trigger.close - trigger.ema_fast) / trigger.atr
    crossed_slow_ordering = (
        regime.ema_fast < regime.ema_slow and regime.close >= regime.ema_fast
    ) or (
        regime.ema_fast > regime.ema_slow and regime.close <= regime.ema_fast
    )
    slow_trend_continuation = (
        side == "long" and regime.ema_fast > regime.ema_slow
    ) or (
        side == "short" and regime.ema_fast < regime.ema_slow
    )
    price_gap_cap = min(
        regime_gap_cap,
        0.005 if slow_trend_continuation else 0.0015,
    )
    shared_ready = (
        crossed_slow_ordering
        and regime_gap <= regime_gap_cap
        and price_gap <= price_gap_cap
        and 45.0 <= regime.rsi <= 60.0
        and trigger_extension <= 2.25
        and trigger.volume_ratio >= max(float(config.traditional_min_volume_ratio), 2.0)
    )
    if not shared_ready:
        return False
    if side == "long":
        return setup.long_ready and not setup.short_ready
    return setup.short_ready and not setup.long_ready


def _aggregate_five_minute_candles(
    candles: Sequence[Candle],
    target_minutes: int,
) -> list[Candle]:
    """Build complete higher-timeframe bars from already closed 5m candles."""

    target_minutes = int(target_minutes)
    if target_minutes < 10 or target_minutes % 5:
        return []
    source_ms = 5 * 60_000
    target_ms = target_minutes * 60_000
    expected = target_minutes // 5
    buckets: dict[int, list[Candle]] = {}
    for candle in candles:
        timestamp = int(candle.timestamp)
        bucket = timestamp // target_ms * target_ms
        buckets.setdefault(bucket, []).append(candle)

    aggregated: list[Candle] = []
    for bucket, rows in sorted(buckets.items()):
        ordered = sorted(rows, key=lambda item: item.timestamp)
        expected_timestamps = [bucket + index * source_ms for index in range(expected)]
        if [int(item.timestamp) for item in ordered] != expected_timestamps:
            continue
        quote_values = [item.quote_volume for item in ordered]
        quote_volume = (
            sum(float(value) for value in quote_values if value is not None)
            if all(value is not None for value in quote_values)
            else None
        )
        aggregated.append(
            Candle(
                bucket,
                float(ordered[0].open),
                max(float(item.high) for item in ordered),
                min(float(item.low) for item in ordered),
                float(ordered[-1].close),
                sum(float(item.volume) for item in ordered),
                quote_volume=quote_volume,
            )
        )
    return aggregated


def _traditional_pressure_room(
    trigger_candles: Sequence[Candle],
    side: str,
    config: StrategyConfig,
) -> tuple[bool, str]:
    """Reject entries with insufficient room to a 10m/15m moving average.

    The higher timeframes are synthesized locally from the closed 5m stream,
    so this quality gate adds no REST requests. Required room is expressed in
    R using the strategy's minimum configured stop distance.
    """

    if not config.traditional_pressure_filter_enabled:
        return True, "disabled"
    if config.trigger_timeframe != "5m" or side not in {"long", "short"}:
        return False, "unsupported_trigger_timeframe"
    if not trigger_candles:
        return False, "insufficient_5m_history"

    price = float(trigger_candles[-1].close)
    levels, error = _traditional_higher_timeframe_average_levels(trigger_candles, config)
    if price <= 0:
        return False, "invalid_pressure_price"
    if error:
        return False, error

    barriers: list[tuple[float, str]] = []
    for name, selected in levels:
        if side == "long" and selected > price:
            barriers.append(((selected - price) / price, name))
        if side == "short" and selected < price:
            barriers.append(((price - selected) / price, name))

    if not barriers:
        return True, "no_overhead_or_underfoot_average"
    room_pct, barrier_name = min(barriers, key=lambda item: item[0])
    required_room_pct = max(0.0, float(config.traditional_pressure_min_room_r)) * max(
        0.0,
        float(config.min_stop_loss_pct),
    )
    detail = f"{barrier_name}_room={room_pct:.6f}_required={required_room_pct:.6f}"
    return room_pct >= required_room_pct, detail


def _traditional_higher_timeframe_average_levels(
    trigger_candles: Sequence[Candle],
    config: StrategyConfig,
) -> tuple[list[tuple[str, float]], str]:
    """Return locally synthesized 10m/15m EMA and SMA levels."""

    configured_timeframes = config.traditional_pressure_timeframes_minutes
    if isinstance(configured_timeframes, (int, float, str)):
        configured_timeframes = (int(configured_timeframes),)
    timeframes = tuple(sorted({int(value) for value in configured_timeframes}))
    if not timeframes:
        return [], "invalid_pressure_configuration"
    ema_period = max(1, int(config.traditional_pressure_ema_period))
    sma_period = max(1, int(config.traditional_pressure_sma_period))
    levels: list[tuple[str, float]] = []
    for minutes in timeframes:
        aggregated = _aggregate_five_minute_candles(trigger_candles, minutes)
        closes = [float(candle.close) for candle in aggregated]
        if len(closes) < max(ema_period, sma_period):
            return [], f"{minutes}m_insufficient_history"
        ema_value = ema(closes, ema_period)[-1]
        sma_value = sma(closes, sma_period)[-1]
        if ema_value is None or sma_value is None:
            return [], f"{minutes}m_average_unavailable"
        levels.extend(
            (
                (f"{minutes}m_ema{ema_period}", float(ema_value)),
                (f"{minutes}m_sma{sma_period}", float(sma_value)),
            )
        )
    return levels, ""


def _traditional_predictive_reversal_short(
    one_minute_candles: Sequence[Candle],
    trigger_candles: Sequence[Candle],
    config: StrategyConfig,
) -> Signal:
    """Anticipate a 5m reversal using only already closed 1m candles.

    A fresh, overextended volume spike arms the setup. The first later 1m bar
    that closes below both execution EMAs and the nearby 10m/15m average
    cluster triggers a half-risk short, provided the spike high can be used as
    a bounded stop and recent 5m structure leaves enough downside room.
    """

    timestamp = one_minute_candles[-1].timestamp if one_minute_candles else 0
    if not config.traditional_predictive_reversal_short_enabled:
        return Signal("flat", 0, timestamp, ("predictive_reversal_short_disabled",))
    lookback = max(5, int(config.traditional_predictive_reversal_lookback))
    confirmation_bars = min(
        15,
        max(1, int(config.traditional_predictive_reversal_confirmation_bars)),
    )
    minimum_history = max(
        lookback + confirmation_bars + 2,
        int(config.traditional_volume_sma_period) + confirmation_bars + 2,
        int(config.traditional_macd_slow) + int(config.traditional_macd_signal) + 2,
    )
    if len(one_minute_candles) < minimum_history or len(trigger_candles) < 12:
        return Signal("flat", 0, timestamp, ("predictive_reversal_short_insufficient_history",))
    continuity_window = one_minute_candles[-minimum_history:]
    if not _candles_are_contiguous(continuity_window, 60_000):
        return Signal("flat", 0, timestamp, ("predictive_reversal_short_data_gap",))

    current = _traditional_features(
        one_minute_candles,
        config.traditional_signal_fast,
        config.traditional_signal_slow,
        config.traditional_rsi_period,
        config.traditional_macd_fast,
        config.traditional_macd_slow,
        config.traditional_macd_signal,
        config.traditional_atr_period,
        config.traditional_volume_sma_period,
    )
    previous = _traditional_features(
        one_minute_candles[:-1],
        config.traditional_signal_fast,
        config.traditional_signal_slow,
        config.traditional_rsi_period,
        config.traditional_macd_fast,
        config.traditional_macd_slow,
        config.traditional_macd_signal,
        config.traditional_atr_period,
        config.traditional_volume_sma_period,
    )
    if (
        current.ema_fast is None
        or current.ema_slow is None
        or current.rsi is None
        or current.macd_histogram is None
        or current.volume_ratio is None
        or current.atr is None
        or current.atr <= 0
        or previous.ema_slow is None
        or previous.macd_histogram is None
    ):
        return Signal("flat", 0, timestamp, ("predictive_reversal_short_indicators_unavailable",))

    levels, level_error = _traditional_higher_timeframe_average_levels(
        trigger_candles,
        config,
    )
    if level_error or not levels:
        return Signal(
            "flat",
            0,
            timestamp,
            (f"predictive_reversal_short_{level_error or 'pressure_unavailable'}",),
        )
    pressure_ceiling = max(value for _, value in levels)
    execution_extension = abs(current.close - current.ema_fast) / current.atr
    current_confirmed = (
        current.close < current.ema_fast
        and current.close < current.ema_slow
        and current.close < pressure_ceiling
        and current.macd_histogram < 0
        and current.volume_ratio
        >= max(
            0.0,
            float(config.traditional_predictive_reversal_min_confirmation_volume_ratio),
        )
        and float(config.traditional_execution_rsi_short_min)
        <= current.rsi
        <= float(config.traditional_predictive_reversal_confirm_rsi_max)
        and execution_extension
        <= max(0.0, float(config.traditional_predictive_reversal_max_execution_extension_atr))
    )
    previous_confirmed = (
        previous.close < (previous.ema_fast or previous.close)
        and previous.close < previous.ema_slow
        and previous.close < pressure_ceiling
        and previous.macd_histogram < 0
    )
    if not current_confirmed or previous_confirmed:
        return Signal("flat", 0, timestamp, ("predictive_reversal_short_not_confirmed",))

    current_index = len(one_minute_candles) - 1
    for age in range(1, confirmation_bars + 1):
        spike_index = current_index - age
        spike = one_minute_candles[spike_index]
        spike_feature = _traditional_features(
            one_minute_candles[: spike_index + 1],
            config.traditional_signal_fast,
            config.traditional_signal_slow,
            config.traditional_rsi_period,
            config.traditional_macd_fast,
            config.traditional_macd_slow,
            config.traditional_macd_signal,
            config.traditional_atr_period,
            config.traditional_volume_sma_period,
        )
        if (
            spike_feature.atr is None
            or spike_feature.atr <= 0
            or spike_feature.ema_fast is None
            or spike_feature.rsi is None
            or spike_feature.macd_histogram is None
            or spike_feature.volume_ratio is None
        ):
            continue
        prior = one_minute_candles[max(0, spike_index - lookback) : spike_index]
        if not prior:
            continue
        spike_range = float(spike.high) - float(spike.low)
        spike_extension = abs(float(spike.close) - spike_feature.ema_fast) / spike_feature.atr
        pressure_distance = abs(float(spike.high) - pressure_ceiling) / pressure_ceiling
        risk_distance = float(spike.high) - current.close
        if risk_distance <= 0:
            continue
        recent_support = min(float(candle.low) for candle in trigger_candles[-12:])
        downside_room = current.close - recent_support
        risk_pct = risk_distance / current.close
        ready = (
            float(spike.high) >= max(float(candle.high) for candle in prior)
            and float(spike.close) > float(spike.open)
            and spike_feature.volume_ratio
            >= max(0.0, float(config.traditional_predictive_reversal_min_spike_volume_ratio))
            and spike_feature.rsi
            >= max(0.0, float(config.traditional_predictive_reversal_min_spike_rsi))
            and spike_range / spike_feature.atr
            >= max(0.0, float(config.traditional_predictive_reversal_min_spike_range_atr))
            and spike_extension
            >= max(0.0, float(config.traditional_predictive_reversal_min_spike_extension_atr))
            and pressure_distance
            <= max(0.0, float(config.traditional_predictive_reversal_max_pressure_distance_pct))
            and current.close < float(spike.low)
            and spike_feature.rsi - current.rsi
            >= max(0.0, float(config.traditional_predictive_reversal_min_rsi_drop))
            and current.macd_histogram < spike_feature.macd_histogram
            and risk_pct
            <= max(0.0, float(config.traditional_predictive_reversal_max_stop_loss_pct))
            and downside_room
            >= risk_distance * max(0.0, float(config.traditional_predictive_reversal_min_room_r))
        )
        if ready:
            return Signal(
                "short",
                6,
                timestamp,
                (
                    "1m_predictive_reversal_short",
                    "1m_volume_spike_rejected",
                    "1m_rsi_momentum_rollover",
                    "1m_sell_volume_confirmed",
                    "1m_below_execution_emas",
                    "10m_15m_pressure_rejected",
                    "5m_downside_room_confirmed",
                ),
            )
    return Signal("flat", 0, timestamp, ("predictive_reversal_short_no_fresh_spike",))


def _candles_are_contiguous(candles: Sequence[Candle], interval_ms: int) -> bool:
    if interval_ms <= 0 or len(candles) < 2:
        return False
    return all(
        int(current.timestamp) - int(previous.timestamp) == interval_ms
        for previous, current in zip(candles, candles[1:])
    )


def _traditional_failed_breakout_short_reversal(
    trigger_candles: Sequence[Candle],
    regime: _TraditionalFeatures,
    previous: _TraditionalFeatures,
    current: _TraditionalFeatures,
    execution: _TraditionalFeatures,
    config: StrategyConfig,
) -> Signal:
    """Detect a confirmed blow-off top without relaxing normal short entries.

    A high-volume upper-wick rejection must set a fresh local high, then the
    next closed trigger bar must break its low with strong bearish body and
    volume quality. MACD must be decelerating and the closed 1-minute stream
    must confirm execution down. This separate branch handles fast intrahour
    reversals while the normal 1-hour regime is still bullish.
    """

    lookback = max(2, int(config.traditional_failed_breakout_short_lookback))
    timestamp = trigger_candles[-1].timestamp if trigger_candles else 0
    if (
        len(trigger_candles) < lookback + 2
        or regime.close <= 0
        or regime.ema_fast is None
        or regime.ema_slow is None
        or not (regime.close > regime.ema_fast > regime.ema_slow)
        or previous.volume_ratio is None
        or previous.macd_histogram is None
        or current.volume_ratio is None
        or current.macd_histogram is None
    ):
        return Signal("flat", 0, timestamp, ("no_failed_breakout_short_reversal",))

    rejection = trigger_candles[-2]
    confirmation = trigger_candles[-1]
    history = trigger_candles[-lookback - 2 : -2]
    rejection_range = float(rejection.high) - float(rejection.low)
    confirmation_range = float(confirmation.high) - float(confirmation.low)
    if rejection_range <= 0 or confirmation_range <= 0 or not history:
        return Signal("flat", 0, timestamp, ("no_failed_breakout_short_reversal",))

    rejection_upper_wick_ratio = (
        float(rejection.high) - max(float(rejection.open), float(rejection.close))
    ) / rejection_range
    confirmation_body_ratio = (
        abs(float(confirmation.close) - float(confirmation.open)) / confirmation_range
    )
    confirmation_close_location = (
        float(confirmation.high) - float(confirmation.close)
    ) / confirmation_range
    execution_down = _traditional_execution(execution, "short")
    if config.traditional_allow_pullback:
        execution_down = execution_down or _traditional_reclaim(execution, "short")
    execution_down = execution_down and _traditional_execution_quality(
        execution,
        "short",
        config,
    )

    ready = (
        float(rejection.high) >= max(float(candle.high) for candle in history)
        and previous.volume_ratio
        >= max(0.0, float(config.traditional_failed_breakout_short_prior_volume_ratio))
        and rejection_upper_wick_ratio
        >= max(0.0, float(config.traditional_failed_breakout_short_prior_wick_ratio))
        and float(confirmation.close) < float(confirmation.open)
        and float(confirmation.close) < float(rejection.low)
        and current.volume_ratio
        >= max(0.0, float(config.traditional_failed_breakout_short_confirm_volume_ratio))
        and confirmation_body_ratio
        >= max(0.0, float(config.traditional_failed_breakout_short_confirm_body_ratio))
        and confirmation_close_location
        >= max(0.0, float(config.traditional_failed_breakout_short_confirm_close_location))
        and current.macd_histogram < previous.macd_histogram
        and execution_down
    )
    if not ready:
        return Signal("flat", 0, timestamp, ("no_failed_breakout_short_reversal",))

    return Signal(
        "short",
        6,
        timestamp,
        (
            f"{config.regime_timeframe}_bull_regime_reversal",
            f"{config.trigger_timeframe}_volume_spike_upper_wick",
            f"{config.trigger_timeframe}_failed_breakout_confirmed",
            f"{config.trigger_timeframe}_macd_decelerating",
            "1m_execution_down",
            "failed_breakout_short_reversal",
        ),
    )


def signal_stop_loss_overrides(
    signal: Signal,
    config: StrategyConfig,
) -> dict[str, int | float]:
    """Return branch-specific stop settings without weakening normal entries."""

    if "1m_predictive_reversal_short" in signal.reasons:
        return {
            "structure_lookback_bars": max(
                2,
                int(config.traditional_predictive_reversal_stop_lookback_bars),
            ),
            "maximum_stop_loss_pct": max(
                float(config.max_stop_loss_pct),
                float(config.traditional_predictive_reversal_max_stop_loss_pct),
            ),
        }
    if "failed_breakout_short_reversal" not in signal.reasons:
        return {}
    return {
        "structure_lookback_bars": max(
            2,
            int(config.traditional_failed_breakout_short_stop_lookback_bars),
        ),
        "maximum_stop_loss_pct": max(
            float(config.max_stop_loss_pct),
            float(config.traditional_failed_breakout_short_max_stop_loss_pct),
        ),
    }


def signal_stop_timeframe(signal: Signal, config: StrategyConfig) -> str:
    """Use the timeframe that contains the structure which armed the signal."""

    if "1m_predictive_reversal_short" in signal.reasons:
        return "1m"
    return config.trigger_timeframe


def signal_position_size_multiplier(signal: Signal) -> float:
    """Use half size for guarded countertrend, transition and fast signals."""
    if any(
        "_countertrend_pullback_" in reason
        or "_neutral_transition_" in reason
        or reason.startswith("1m_impulse_")
        or reason == "1m_predictive_reversal_short"
        or reason == "failed_breakout_short_reversal"
        for reason in signal.reasons
    ):
        return 0.5
    return 1.0


def _traditional_one_minute_impulse(
    candles: Sequence[Candle],
    execution: _TraditionalFeatures,
    trigger: _TraditionalFeatures,
    side: str,
    config: StrategyConfig,
) -> bool:
    """Accept only the first closed 1-minute breakout inside a safe 5m envelope.

    The freshness check rejects continuation candles after the range has already
    broken.  Extension is measured against the latest closed 5-minute EMA/ATR,
    which prevents a fast signal from bypassing the existing anti-chase policy.
    """
    lookback = max(2, int(config.traditional_1m_impulse_lookback))
    confirmation_bars = max(1, min(2, int(config.traditional_1m_impulse_confirmation_bars)))
    if (
        side not in {"long", "short"}
        or len(candles) < lookback + confirmation_bars + 1
        or execution.atr is None
        or execution.atr <= 0
        or execution.ema_fast is None
        or execution.ema_slow is None
        or execution.rsi is None
        or execution.macd_histogram is None
        or execution.volume_ratio is None
        or trigger.ema_fast is None
        or trigger.atr is None
        or trigger.atr <= 0
    ):
        return False

    def quality(feature: _TraditionalFeatures) -> bool:
        if (
            feature.atr is None
            or feature.atr <= 0
            or feature.ema_fast is None
            or feature.ema_slow is None
            or feature.rsi is None
            or feature.macd_histogram is None
            or feature.volume_ratio is None
        ):
            return False
        candle_range = feature.high - feature.low
        if candle_range <= 0:
            return False
        body_ratio = abs(feature.close - feature.open) / candle_range
        close_location = (
            (feature.close - feature.low) / candle_range
            if side == "long"
            else (feature.high - feature.close) / candle_range
        )
        range_atr = candle_range / feature.atr
        extension_atr = abs(feature.close - trigger.ema_fast) / trigger.atr
        shared = (
            feature.volume_ratio >= max(0.0, float(config.traditional_1m_impulse_min_volume_ratio))
            and body_ratio >= max(0.0, float(config.traditional_1m_impulse_min_body_ratio))
            and close_location >= max(0.0, float(config.traditional_1m_impulse_min_close_location))
            and range_atr >= max(0.0, float(config.traditional_1m_impulse_min_range_atr))
            and extension_atr <= max(0.0, float(config.traditional_1m_impulse_max_extension_atr))
        )
        if side == "long":
            return (
                shared
                and feature.close >= feature.ema_fast >= feature.ema_slow
                and feature.macd_histogram > 0
                and config.traditional_rsi_long_min <= feature.rsi <= config.traditional_rsi_long_max
            )
        return (
            shared
            and feature.close <= feature.ema_fast <= feature.ema_slow
            and feature.macd_histogram < 0
            and config.traditional_rsi_short_min <= feature.rsi <= config.traditional_rsi_short_max
        )

    if not quality(execution):
        return False

    current = candles[-1]
    if confirmation_bars == 1:
        previous = candles[-2]
        previous_window = candles[-lookback - 1 : -1]
        prior_window = candles[-lookback - 2 : -2]
        if side == "long":
            return (
                current.close > max(candle.high for candle in previous_window)
                and previous.close <= max(candle.high for candle in prior_window)
            )
        return (
            current.close < min(candle.low for candle in previous_window)
            and previous.close >= min(candle.low for candle in prior_window)
        )

    first = candles[-2]
    before_first = candles[-3]
    base_window = candles[-lookback - 2 : -2]
    prior_window = candles[-lookback - 3 : -3]
    first_feature = _traditional_features(
        candles[:-1],
        config.traditional_signal_fast,
        config.traditional_signal_slow,
        config.traditional_rsi_period,
        config.traditional_macd_fast,
        config.traditional_macd_slow,
        config.traditional_macd_signal,
        config.traditional_atr_period,
        config.traditional_volume_sma_period,
    )
    if not quality(first_feature):
        return False
    if side == "long":
        return (
            first.close > max(candle.high for candle in base_window)
            and before_first.close <= max(candle.high for candle in prior_window)
            and current.close > first.high
        )
    return (
        first.close < min(candle.low for candle in base_window)
        and before_first.close >= min(candle.low for candle in prior_window)
        and current.close < first.low
    )


def _traditional_breakout_quality(feature: _TraditionalFeatures, side: str, config: StrategyConfig) -> bool:
    """Reject marginal, wick-heavy, low-volume and overextended breakouts."""
    if feature.ema_fast is None or feature.atr is None or feature.atr <= 0 or feature.volume_ratio is None:
        return False
    candle_range = feature.high - feature.low
    if candle_range <= 0:
        return False
    body_ratio = abs(feature.close - feature.open) / candle_range
    close_location = (
        (feature.close - feature.low) / candle_range
        if side == "long"
        else (feature.high - feature.close) / candle_range
    )
    extension_atr = abs(feature.close - feature.ema_fast) / feature.atr
    return (
        feature.volume_ratio >= max(
            config.traditional_min_volume_ratio,
            config.traditional_breakout_min_volume_ratio,
        )
        and body_ratio >= config.traditional_breakout_min_body_ratio
        and close_location >= config.traditional_breakout_min_close_location
        and extension_atr <= config.traditional_breakout_max_extension_atr
    )


def _traditional_execution(feature: _TraditionalFeatures, side: str) -> bool:
    if feature.ema_fast is None or feature.ema_slow is None:
        return False
    if side == "long":
        return feature.close >= feature.ema_fast and feature.ema_fast >= feature.ema_slow
    return feature.close <= feature.ema_fast and feature.ema_fast <= feature.ema_slow


def _traditional_execution_quality(
    feature: _TraditionalFeatures,
    side: str,
    config: StrategyConfig,
) -> bool:
    """Prevent a valid 1m alignment from becoming an overextended chase."""

    if side not in {"long", "short"}:
        return False
    extension_cap = max(0.0, float(config.traditional_execution_max_extension_atr))
    if extension_cap:
        if feature.ema_fast is None or feature.atr is None or feature.atr <= 0:
            return False
        if abs(feature.close - feature.ema_fast) / feature.atr > extension_cap:
            return False
    if side == "long":
        rsi_cap = max(0.0, float(config.traditional_execution_rsi_long_max))
        return not rsi_cap or (feature.rsi is not None and feature.rsi <= rsi_cap)
    rsi_floor = max(0.0, float(config.traditional_execution_rsi_short_min))
    return not rsi_floor or (feature.rsi is not None and feature.rsi >= rsi_floor)


def _traditional_reclaim(feature: _TraditionalFeatures, side: str) -> bool:
    """Detect a closed-bar reclaim of the fast EMA inside an existing trend."""
    if (
        feature.previous_close is None
        or feature.previous_ema_fast is None
        or feature.ema_fast is None
        or feature.ema_slow is None
    ):
        return False
    if side == "long":
        return (
            feature.previous_close <= feature.previous_ema_fast
            and feature.close > feature.ema_fast
            and feature.ema_fast >= feature.ema_slow
        )
    return (
        feature.previous_close >= feature.previous_ema_fast
        and feature.close < feature.ema_fast
        and feature.ema_fast <= feature.ema_slow
    )
