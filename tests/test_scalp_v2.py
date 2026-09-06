from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from btc_futures_bot.indicators import ema
from btc_futures_bot.models import Candle
from btc_futures_bot.strategy import (
    MultiTimeframeStrategy,
    StrategyConfig,
    _features,
    strategy_replay_cache,
)


def _config(**overrides: object) -> StrategyConfig:
    values = {
        "mode": "scalp_v2",
        "trigger_timeframe": "1m",
        "regime_timeframe": "5m",
        "min_score": 4,
        "min_volume_ratio": 1.1,
    }
    values.update(overrides)
    return StrategyConfig(**values)


def _bars(closes: list[float], interval: int, final_timestamp: int) -> list[Candle]:
    result = []
    for index, close in enumerate(closes):
        open_price = closes[index - 1] if index else close
        result.append(
            Candle(
                final_timestamp - (len(closes) - index - 1) * interval,
                open_price,
                max(open_price, close) + 0.03,
                min(open_price, close) - 0.03,
                close,
                10.0,
            )
        )
    return result


def _market(side: str = "long", *, reclaim: bool = False) -> dict[str, list[Candle]]:
    timestamp = 1_780_000_000_000
    closes = [100.0 + 0.03 * index + (0.12 if index % 2 == 0 else -0.12) for index in range(40)]
    one_minute = _bars(closes, 60_000, timestamp - 60_000)
    close = 101.22 if reclaim else 101.7
    one_minute.append(Candle(timestamp, closes[-1], close + 0.03, closes[-1] - 0.03, close, 15.0))
    five_minute = _bars(closes, 300_000, timestamp - 300_000)
    market = {"1m": one_minute, "5m": five_minute}
    if side == "short":
        market = {
            timeframe: [
                replace(candle, open=200.0 - candle.open, high=200.0 - candle.low,
                        low=200.0 - candle.high, close=200.0 - candle.close)
                for candle in candles
            ]
            for timeframe, candles in market.items()
        }
    return market


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("reclaim", [False, True])
def test_scalp_v2_accepts_fresh_closed_minute_setup_without_hour_history(side: str, reclaim: bool) -> None:
    market = _market(side, reclaim=reclaim)
    signal = MultiTimeframeStrategy(_config()).evaluate(market)

    assert signal.side == side
    assert signal.score == 5
    assert signal.timestamp == market["1m"][-1].timestamp
    assert "scalp_v2" in signal.reasons
    assert ("scalp_v2_1m_ema_reclaim" if reclaim else "scalp_v2_1m_breakout") in signal.reasons


@pytest.mark.parametrize("market", [{}, {"1m": []}, {"5m": []}])
def test_scalp_v2_missing_data_is_flat(market: dict[str, list[Candle]]) -> None:
    signal = MultiTimeframeStrategy(_config()).evaluate(market)
    assert signal.side == "flat"
    assert signal.reasons == ("scalp_v2_insufficient_candles",)


@pytest.mark.parametrize("timeframe", ["1m", "5m"])
def test_scalp_v2_requires_warmed_up_history_on_both_timeframes(timeframe: str) -> None:
    market = _market()
    market[timeframe] = market[timeframe][-10:]
    assert MultiTimeframeStrategy(_config()).evaluate(market).reasons == ("scalp_v2_insufficient_candles",)


@pytest.mark.parametrize("overrides", [{"trigger_timeframe": "5m"}, {"regime_timeframe": "1h"}])
def test_scalp_v2_wrong_timeframes_fail_closed(overrides: dict[str, str]) -> None:
    signal = MultiTimeframeStrategy(_config(**overrides)).evaluate(_market())
    assert signal.side == "flat"
    assert signal.reasons == ("scalp_v2_invalid_timeframes",)


@pytest.mark.parametrize("side", ["long", "short"])
def test_scalp_v2_momentum_alone_cannot_retrigger_even_when_alignment_switch_is_off(side: str) -> None:
    market = _market(side)
    previous = market["1m"][-1]
    change = 0.01 if side == "long" else -0.01
    market["1m"].append(
        Candle(previous.timestamp + 60_000, previous.close, previous.close + 0.02,
               previous.close - 0.02, previous.close + change, 15.0)
    )
    config = _config(min_score=2, require_full_alignment=False, require_volume_confirmation=False)
    signal = MultiTimeframeStrategy(config).evaluate(market)
    assert signal.side == "flat"
    assert "scalp_v2_no_fresh_trigger" in signal.reasons
    # The legacy score-only behavior remains available and unchanged.
    assert MultiTimeframeStrategy(replace(config, mode="scalp")).evaluate(market).side == side


@pytest.mark.parametrize("side", ["long", "short"])
def test_scalp_v2_volume_is_mandatory_even_if_legacy_confirmation_switch_is_off(side: str) -> None:
    market = _market(side)
    market["1m"][-1] = replace(market["1m"][-1], volume=2.0)
    signal = MultiTimeframeStrategy(
        _config(require_full_alignment=False, require_volume_confirmation=False)
    ).evaluate(market)
    assert signal.side == "flat"
    assert "scalp_v2_low_volume" in signal.reasons


def test_scalp_v2_uses_quote_turnover_for_volume_confirmation() -> None:
    market = _market()
    market["1m"] = [replace(candle, quote_volume=1000.0) for candle in market["1m"]]
    market["1m"][-1] = replace(market["1m"][-1], volume=100.0, quote_volume=100.0)
    assert "scalp_v2_low_volume" in MultiTimeframeStrategy(_config()).evaluate(market).reasons


@pytest.mark.parametrize("side", ["long", "short"])
def test_scalp_v2_rejects_execution_too_far_from_fast_ema(side: str) -> None:
    signal = MultiTimeframeStrategy(_config(scalp_max_extension_atr=0.5)).evaluate(_market(side))
    assert signal.side == "flat"
    assert "scalp_v2_overextended" in signal.reasons


def test_scalp_v2_rejects_weak_candle_body_and_close_location() -> None:
    market = _market()
    market["1m"][-1] = replace(market["1m"][-1], open=101.65, high=102.5)
    signal = MultiTimeframeStrategy(_config()).evaluate(market)
    assert signal.side == "flat"
    assert "scalp_v2_weak_candle_body" in signal.reasons
    assert "scalp_v2_weak_close_location" in signal.reasons


def test_scalp_v2_requires_five_minute_direction_agreement() -> None:
    market = _market()
    market["5m"] = _market("short")["5m"]
    signal = MultiTimeframeStrategy(_config()).evaluate(market)
    assert signal.side == "flat"


def test_scalp_v2_flat_five_minute_emas_are_not_directional() -> None:
    market = _market()
    market["5m"] = _bars([100.0] * 40, 300_000, market["5m"][-1].timestamp)
    signal = MultiTimeframeStrategy(_config()).evaluate(market)
    assert signal.side == "flat"
    assert signal.reasons == ("scalp_v2_no_5m_direction",)


def test_scalp_feature_cache_is_opt_in_and_includes_complete_history_and_config() -> None:
    config = _config()
    candles = _market()["1m"]
    revised_history = list(candles)
    revised_history[2] = replace(revised_history[2], close=revised_history[2].close + 0.01)
    with patch("btc_futures_bot.strategy.ema", wraps=ema) as mocked_ema:
        baseline = _features(candles, config)
        assert _features(candles, config) == baseline
        assert mocked_ema.call_count == 6
        with strategy_replay_cache(max_entries_per_helper=2):
            assert _features(candles, config) == baseline
            assert _features(list(candles), config) == baseline
            assert mocked_ema.call_count == 9
            revised = _features(revised_history, config)
            assert mocked_ema.call_count == 12
            assert revised != baseline
            _features(candles, replace(config, ema_fast=6))
            assert mocked_ema.call_count == 15
            # A third distinct result evicts the oldest complete key.
            _features(candles, config)
            assert mocked_ema.call_count == 18
        assert _features(candles, config) == baseline
        assert mocked_ema.call_count == 21


def test_scalp_v2_replay_and_uncached_signals_are_identical() -> None:
    strategy = MultiTimeframeStrategy(_config())
    market = _market()
    uncached = strategy.evaluate(market)
    with strategy_replay_cache():
        assert strategy.evaluate(market) == uncached
        assert strategy.evaluate(market) == uncached


@pytest.mark.parametrize("field_name", ["scalp_min_body_ratio", "scalp_min_close_location"])
@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_scalp_v2_rejects_invalid_quality_ratios(field_name: str, value: float) -> None:
    with pytest.raises(ValueError, match=field_name):
        MultiTimeframeStrategy(_config(**{field_name: value}))


@pytest.mark.parametrize("value", [0.0, -0.01, float("nan"), float("inf"), -float("inf")])
def test_scalp_v2_rejects_invalid_execution_extension(value: float) -> None:
    with pytest.raises(ValueError, match="scalp_max_extension_atr"):
        MultiTimeframeStrategy(_config(scalp_max_extension_atr=value))


@pytest.mark.parametrize("mode", ["scalp", "traditional_kline"])
def test_new_quality_validation_does_not_change_legacy_construction(mode: str) -> None:
    MultiTimeframeStrategy(_config(mode=mode, scalp_min_body_ratio=float("nan"),
                                  scalp_min_close_location=-1.0, scalp_max_extension_atr=0.0))
