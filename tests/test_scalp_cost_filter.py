from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from btc_futures_bot.costs import CostConfig
from btc_futures_bot.entry_costs import evaluate_scalp_cost_coverage
from btc_futures_bot.models import Candle
from btc_futures_bot.strategy import MultiTimeframeStrategy, StrategyConfig


def _costs(**overrides: object) -> CostConfig:
    values = {"min_net_edge_pct": 0.0015, "expected_holding_hours": 1.0 / 6.0}
    values.update(overrides)
    return CostConfig(**values)


def _history(move_pct: float, count: int = 60) -> list[Candle]:
    return [
        Candle(1_780_000_000_000 + index * 60_000, 100.0, 100.0 * (1.0 + move_pct),
               100.0 * (1.0 - move_pct), 100.0, 10.0)
        for index in range(count)
    ]


@pytest.mark.parametrize("side", ["long", "short"])
def test_quiet_windows_cannot_cover_round_trip_costs_and_edge(side: str) -> None:
    result = evaluate_scalp_cost_coverage(_history(0.0005), side, 600, _costs())
    assert not result.allowed
    assert result.reason == "median_below_cost_hurdle"
    assert result.observed_move_pct == pytest.approx(0.0005)
    assert result.latest_move_pct == pytest.approx(0.0005)
    assert result.required_move_pct == pytest.approx(0.0029, rel=0.001)
    assert result.horizon_bars == 10
    assert result.lookback_windows == 6
    assert len(result.window_move_pcts) == 6


@pytest.mark.parametrize("side", ["long", "short"])
def test_active_typical_and_latest_windows_pass(side: str) -> None:
    result = evaluate_scalp_cost_coverage(_history(0.0035), side, 600, _costs())
    assert result.allowed
    assert result.reason == "cost_coverage"
    assert result.observed_move_pct == pytest.approx(0.0035)
    assert result.latest_move_pct == pytest.approx(0.0035)


@pytest.mark.parametrize("side", ["long", "short"])
def test_one_recent_spike_cannot_make_typical_window_activity_pass(side: str) -> None:
    candles = _history(0.0005)
    candles[-1] = replace(candles[-1], high=110.0, low=90.0)
    result = evaluate_scalp_cost_coverage(candles, side, 600, _costs())
    assert not result.allowed
    assert result.reason == "median_below_cost_hurdle"
    assert result.latest_move_pct == pytest.approx(0.1)
    assert result.observed_move_pct == pytest.approx(0.0005)


@pytest.mark.parametrize("side", ["long", "short"])
def test_newest_window_must_pass_even_when_older_windows_are_active(side: str) -> None:
    candles = _history(0.004)
    candles[-10:] = [replace(candle, high=100.05, low=99.95) for candle in candles[-10:]]
    result = evaluate_scalp_cost_coverage(candles, side, 600, _costs())
    assert not result.allowed
    assert result.reason == "latest_below_cost_hurdle"
    assert result.observed_move_pct == pytest.approx(0.004)
    assert result.latest_move_pct == pytest.approx(0.0005)


def test_excursions_use_side_favorable_prices_and_first_open_not_hindsight_extremes() -> None:
    candles = [replace(candle, high=100.35, low=99.95) for candle in _history(0.0035)]
    assert evaluate_scalp_cost_coverage(candles, "long", 600, _costs()).allowed
    short = evaluate_scalp_cost_coverage(candles, "short", 600, _costs())
    assert not short.allowed
    assert short.observed_move_pct == pytest.approx(0.0005)
    # A dip below the opening price cannot be used as a hindsight long entry.
    dipped = [replace(candle, high=100.05, low=99.5) for candle in candles]
    long = evaluate_scalp_cost_coverage(dipped, "long", 600, _costs())
    assert not long.allowed
    assert long.observed_move_pct == pytest.approx(0.0005)


@pytest.mark.parametrize("side", ["long", "short"])
def test_exact_side_hurdle_matches_selected_cost_net_pnl_equation(side: str) -> None:
    costs = _costs(expected_holding_hours=16.0)
    result = evaluate_scalp_cost_coverage(_history(0.004), side, 600, costs)
    direction = 1.0 if side == "long" else -1.0
    exit_price = 100.0 * (1.0 + direction * result.required_move_pct)
    assert costs.estimate_net_pnl(side, 100.0, exit_price, 1.0, holding_hours=600 / 3600) / 100.0 == pytest.approx(costs.min_net_edge_pct)
    # Explicit scalp horizon, not the unrelated default 16-hour estimate.
    assert result.required_move_pct < 0.003


@pytest.mark.parametrize("side", ["long", "short"])
def test_increased_actual_selected_fees_can_block_an_existing_activity_level(side: str) -> None:
    candles = _history(0.0035)
    assert evaluate_scalp_cost_coverage(candles, side, 600, _costs()).allowed
    higher_fees = evaluate_scalp_cost_coverage(candles, side, 600, _costs(taker_fee_pct=0.001))
    assert not higher_fees.allowed
    assert higher_fees.required_move_pct > 0.0035


def test_maker_and_taker_profiles_use_the_selected_execution_rate() -> None:
    candles = _history(0.0025)
    assert not evaluate_scalp_cost_coverage(candles, "long", 600, _costs()).allowed
    assert evaluate_scalp_cost_coverage(candles, "long", 600, _costs(execution="maker")).allowed


def test_shorter_horizon_rebuilds_completed_nonoverlapping_windows() -> None:
    candles = []
    for index in range(60):
        open_price = 100.0 + 0.05 * index
        candles.append(Candle(1_780_000_000_000 + index * 60_000, open_price,
                              open_price + 0.051, open_price - 0.001, open_price + 0.05, 10.0))
    ten_minutes = evaluate_scalp_cost_coverage(candles, "long", 600, _costs())
    five_minutes = evaluate_scalp_cost_coverage(candles, "long", 300, _costs())
    assert ten_minutes.allowed
    assert not five_minutes.allowed
    assert ten_minutes.horizon_bars == 10
    assert five_minutes.horizon_bars == 5
    assert ten_minutes.observed_move_pct > five_minutes.observed_move_pct


def test_horizon_rounds_up_to_complete_minutes_and_needs_corresponding_history() -> None:
    result = evaluate_scalp_cost_coverage(_history(0.004), "long", 601, _costs())
    assert not result.allowed
    assert result.reason == "insufficient_history"
    assert result.horizon_bars == 11


@pytest.mark.parametrize("count", [0, 1, 59])
def test_missing_history_fails_closed(count: int) -> None:
    result = evaluate_scalp_cost_coverage(_history(0.004, count), "long", 600, _costs())
    assert not result.allowed
    assert result.reason == "insufficient_history"


@pytest.mark.parametrize("mutation", ["gap", "duplicate", "reverse", "nan_timestamp", "negative_timestamp"])
def test_discontinuous_or_bad_timestamps_fail_closed(mutation: str) -> None:
    candles = _history(0.004)
    if mutation == "gap":
        candles[10] = replace(candles[10], timestamp=candles[10].timestamp + 60_000)
    elif mutation == "duplicate":
        candles[10] = replace(candles[10], timestamp=candles[9].timestamp)
    elif mutation == "reverse":
        candles.reverse()
    elif mutation == "nan_timestamp":
        candles[10] = replace(candles[10], timestamp=float("nan"))
    else:
        candles[10] = replace(candles[10], timestamp=-1)
    result = evaluate_scalp_cost_coverage(candles, "long", 600, _costs())
    assert not result.allowed
    assert result.reason in {"invalid_timestamp", "discontinuous_history"}


@pytest.mark.parametrize("overrides", [
    {"open": 0.0}, {"open": float("nan")}, {"close": float("inf")}, {"high": 99.0},
    {"low": 101.0}, {"low": -1.0}, {"volume": float("nan")}, {"quote_volume": -1.0},
])
def test_invalid_candle_values_fail_closed(overrides: dict[str, float]) -> None:
    candles = _history(0.004)
    candles[20] = replace(candles[20], **overrides)
    result = evaluate_scalp_cost_coverage(candles, "long", 600, _costs())
    assert not result.allowed
    assert result.reason in {"invalid_ohlc", "invalid_volume"}


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_horizon_fails_closed(value: float) -> None:
    assert not evaluate_scalp_cost_coverage(_history(0.004), "long", value, _costs()).allowed


@pytest.mark.parametrize("value", [True, 2, 13, 6.5])
def test_invalid_window_count_fails_closed(value: object) -> None:
    result = evaluate_scalp_cost_coverage(_history(0.004), "long", 600, _costs(), value)
    assert not result.allowed
    assert result.reason == "invalid_lookback_windows"
    with pytest.raises(ValueError, match="scalp_cost_lookback_windows"):
        MultiTimeframeStrategy(StrategyConfig(mode="scalp_v2", scalp_cost_filter_enabled=True,
                                             scalp_cost_lookback_windows=value))


@pytest.mark.parametrize("cost_overrides", [
    {"taker_fee_pct": float("nan")}, {"slippage_pct": float("inf")},
    {"min_net_edge_pct": float("nan")}, {"funding_rate_pct_per_8h": float("inf")},
    {"taker_fee_pct": 0.999, "slippage_pct": 0.001},
])
def test_nonfinite_or_impossible_cost_inputs_fail_closed(cost_overrides: dict[str, float]) -> None:
    result = evaluate_scalp_cost_coverage(_history(0.004), "long", 600, _costs(**cost_overrides))
    assert not result.allowed
    assert result.reason == "invalid_costs"


def test_only_latest_complete_history_is_used() -> None:
    candles = _history(0.004, 61)
    candles[0] = replace(candles[0], open=float("nan"))
    assert evaluate_scalp_cost_coverage(candles, "long", 600, _costs()).allowed


def _signal_market() -> dict[str, list[Candle]]:
    # Real indicator input supplies a fresh EMA reclaim and bullish context.
    timestamp = 1_780_000_000_000
    closes = [100.0 + 0.03 * index + (0.12 if index % 2 == 0 else -0.12) for index in range(70)]
    market: dict[str, list[Candle]] = {}
    for timeframe, interval in (("1m", 60_000), ("5m", 300_000)):
        candles = []
        for index, close in enumerate(closes):
            open_price = closes[index - 1] if index else close
            candles.append(Candle(timestamp - (len(closes) - index) * interval, open_price,
                                  max(open_price, close) + 0.03, min(open_price, close) - 0.03,
                                  close, 10.0))
        market[timeframe] = candles
    last_close = closes[-1]
    market["1m"].append(Candle(timestamp, last_close, last_close + 0.68,
                               last_close - 0.03, last_close + 0.65, 15.0))
    return market


def test_strategy_guard_runs_after_primary_setup_and_preserves_score_timestamp() -> None:
    config = StrategyConfig(mode="scalp_v2", trigger_timeframe="1m", regime_timeframe="5m", min_score=4,
                            hard_max_hold_seconds=600, scalp_cost_filter_enabled=True)
    market = _signal_market()
    high_costs = _costs(taker_fee_pct=0.002)
    strategy = MultiTimeframeStrategy(config, costs=high_costs)
    unguarded = MultiTimeframeStrategy(replace(config, scalp_cost_filter_enabled=False), costs=high_costs).evaluate(market)
    assert unguarded.side == "long"
    blocked = strategy.evaluate(market)
    assert blocked.side == "flat"
    assert blocked.score == unguarded.score
    assert blocked.timestamp == unguarded.timestamp
    assert blocked.reasons[0] == "scalp_v2_cost_blocked"
    assert "observed=" in blocked.reasons[1]
    assert "latest=" in blocked.reasons[1]
    assert "required=" in blocked.reasons[1]
    assert "horizon_bars=10" in blocked.reasons[1]


def test_passing_strategy_cost_reason_is_stable_and_has_no_numeric_hash_noise() -> None:
    config = StrategyConfig(mode="scalp_v2", trigger_timeframe="1m", regime_timeframe="5m", min_score=4,
                            hard_max_hold_seconds=600, scalp_cost_filter_enabled=True)
    signal = MultiTimeframeStrategy(config, costs=_costs()).evaluate(_signal_market())
    assert signal.side == "long"
    assert signal.reasons[-1] == "scalp_v2_cost_coverage"


def test_no_primary_signal_never_calls_cost_admission() -> None:
    config = StrategyConfig(mode="scalp_v2", trigger_timeframe="1m", regime_timeframe="5m",
                            hard_max_hold_seconds=600, scalp_cost_filter_enabled=True)
    market = _signal_market()
    market["1m"][-1] = replace(market["1m"][-1], volume=0.01)
    with patch("btc_futures_bot.strategy.evaluate_scalp_cost_coverage") as admission:
        assert MultiTimeframeStrategy(config, costs=_costs()).evaluate(market).side == "flat"
        admission.assert_not_called()


@pytest.mark.parametrize("mode", ["scalp", "traditional_kline"])
def test_legacy_modes_never_apply_or_validate_scalp_cost_admission(mode: str) -> None:
    config = StrategyConfig(mode=mode, trigger_timeframe="1m", regime_timeframe="5m",
                            scalp_cost_filter_enabled=True, scalp_cost_lookback_windows=-1)
    with patch("btc_futures_bot.strategy.evaluate_scalp_cost_coverage") as admission:
        MultiTimeframeStrategy(config, costs=_costs(taker_fee_pct=0.5)).evaluate(_signal_market())
        admission.assert_not_called()
