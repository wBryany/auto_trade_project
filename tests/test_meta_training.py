from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from btc_futures_bot.models import Candle, Signal
from btc_futures_bot.main import load_config
from btc_futures_bot.strategy import MultiTimeframeStrategy, StrategyConfig
from btc_futures_bot.trade_model.features import FEATURE_NAMES
from btc_futures_bot.trade_model.training import (
    ApprovalPolicy,
    BarrierConfig,
    FundingRate,
    SplitConfig,
    _iter_replay_signals,
    assess_live_approval,
    choose_validation_threshold,
    chronological_purged_split,
    funding_cost_for_position,
    fixed_triple_barrier_live_approval,
    load_candles,
    replay_primary_candidates,
    resolve_replay_history_window,
    threshold_metrics,
    train_lightgbm_native,
    triple_barrier_label,
)


_REAL_REPLAY_DATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "binance_production"
    / "review_20260904"
)


def _candle(timestamp: int, *, open_: float = 100.0, high: float = 100.2,
            low: float = 99.8, close: float = 100.0) -> Candle:
    return Candle(timestamp, open_, high, low, close, 10.0, quote_volume=1000.0)


def test_triple_barrier_enters_next_open_and_same_bar_double_touch_is_stop() -> None:
    candles = [
        _candle(60_000, high=102.0, low=98.0),
        _candle(120_000),
    ]
    result = triple_barrier_label(
        "long",
        candles,
        BarrierConfig(
            take_profit_pct=0.01,
            stop_loss_pct=0.01,
            horizon_bars=2,
            fee_pct_per_side=0.001,
            slippage_pct_per_side=0.0005,
            fallback_funding_rate_per_8h=0.0,
        ),
    )

    assert result.entry_timestamp == 60_000
    assert result.entry_price == 100.0
    assert result.barrier == "stop_loss"
    assert result.exit_price == 99.0
    assert result.label == 0
    assert result.gross_return == pytest.approx(-0.01)
    assert result.fee_cost == pytest.approx(0.00199)
    assert result.slippage_cost == pytest.approx(0.000995)
    assert result.net_return == pytest.approx(-0.012985)


def test_horizon_label_and_actual_funding_are_net_of_all_costs() -> None:
    start = 7 * 60 * 60 * 1000 + 59 * 60 * 1000
    candles = [
        _candle(start, close=100.08),
        _candle(start + 60_000, open_=100.08, close=100.10),
    ]
    config = BarrierConfig(
        take_profit_pct=0.02,
        stop_loss_pct=0.02,
        horizon_bars=2,
        fee_pct_per_side=0.0002,
        slippage_pct_per_side=0.0001,
        fallback_funding_rate_per_8h=0.0009,
    )
    funding = [FundingRate(8 * 60 * 60 * 1000, 0.0004)]
    result = triple_barrier_label("long", candles, config, funding)

    assert result.barrier == "horizon"
    assert result.funding_cost == pytest.approx(0.0004)
    assert result.net_return < result.gross_return
    assert result.label == 0  # the tiny move is consumed by fee/slippage/funding
    assert funding_cost_for_position(
        "short", start, start + 120_000, funding, fallback_rate=0.1
    ) == pytest.approx(-0.0004)


@dataclass(frozen=True)
class _Record:
    decision_timestamp: int
    exit_timestamp: int


def test_chronological_split_purges_label_overlap_and_embargoes_later_samples() -> None:
    samples = [
        _Record(0, 20),
        _Record(100, 120),
        _Record(200, 350),  # overlaps the 300 boundary after purge gap
        _Record(300, 320),  # validation embargo
        _Record(400, 420),
        _Record(500, 650),  # overlaps the 600 boundary after purge gap
        _Record(600, 620),  # holdout embargo
        _Record(700, 720),
        _Record(800, 820),
    ]
    result = chronological_purged_split(
        samples,
        SplitConfig(
            train_fraction=1 / 3,
            validation_fraction=1 / 3,
            purge_ms=20,
            embargo_ms=50,
        ),
    )

    assert [item.decision_timestamp for item in result.train] == [0, 100]
    assert [item.decision_timestamp for item in result.validation] == [400]
    assert [item.decision_timestamp for item in result.holdout] == [700, 800]
    assert result.purged_train == 1
    assert result.purged_validation == 1
    assert result.embargoed_validation == 1
    assert result.embargoed_holdout == 1


def test_threshold_is_ranked_on_validation_expectancy_pf_then_coverage() -> None:
    selection = choose_validation_threshold(
        [0.90, 0.75, 0.70, 0.20],
        [0.10, -0.02, 0.03, -0.50],
        thresholds=[0.50, 0.80],
        min_selected=1,
        min_coverage=0.20,
    )

    assert selection.eligible is True
    assert selection.threshold == 0.80
    assert selection.metrics.net_expectancy == pytest.approx(0.10)
    assert selection.metrics.coverage == pytest.approx(0.25)


def test_insufficient_samples_can_never_be_approved_for_live() -> None:
    records = [_Record(index * 100, index * 100 + 1) for index in range(12)]
    split = chronological_purged_split(
        records,
        SplitConfig(train_fraction=0.5, validation_fraction=0.25),
    )
    selection = choose_validation_threshold(
        [0.9] * len(split.validation),
        [0.01] * len(split.validation),
        thresholds=[0.5],
        min_selected=1,
        min_coverage=0,
    )
    holdout = threshold_metrics(
        [0.9] * len(split.holdout),
        [0.01] * len(split.holdout),
        selection.threshold,
    )
    approved, reasons = assess_live_approval(
        split,
        selection,
        holdout,
        ApprovalPolicy(
            min_train_samples=100,
            min_validation_samples=30,
            min_holdout_samples=30,
            min_selected_validation=1,
            min_selected_holdout=1,
            min_coverage=0,
        ),
    )

    assert approved is False
    assert any(reason.startswith("insufficient_train_samples") for reason in reasons)
    assert any(reason.startswith("insufficient_validation_samples") for reason in reasons)
    assert any(reason.startswith("insufficient_holdout_samples") for reason in reasons)


def test_fixed_triple_barrier_artifact_is_never_live_approved_even_if_metrics_pass() -> None:
    approved, equivalent, blockers, reasons = fixed_triple_barrier_live_approval(())

    assert approved is False
    assert equivalent is False
    assert blockers
    assert "dynamic live exit" in blockers[0]
    assert reasons == blockers


class _FrozenStrategy:
    def __init__(self, first_decision: int) -> None:
        self.first_decision = first_decision

    def evaluate(self, histories: dict[str, list[Candle]]) -> Signal:
        if not all(histories.values()):
            return Signal("flat", 0, 0)
        decision = histories["1m"][-1].timestamp + 60_000
        intervals = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}
        for timeframe, candles in histories.items():
            assert all(candle.timestamp + intervals[timeframe] <= decision for candle in candles)
        if decision in {self.first_decision, self.first_decision + 60_000}:
            return Signal("long", 7, 777_000, ("frozen_primary_candidate",))
        return Signal("flat", 0, histories["1m"][-1].timestamp)


def test_replay_uses_closed_candles_next_open_and_one_unique_primary_candidate() -> None:
    hours = 24
    one_minute = [_candle(index * 60_000) for index in range(hours * 60 + 1)]
    five_minute = [_candle(index * 300_000) for index in range(hours * 12 + 1)]
    one_hour = [_candle(index * 3_600_000) for index in range(hours + 1)]
    decision = 22 * 3_600_000
    samples, stats = replay_primary_candidates(
        {"1m": one_minute, "5m": five_minute, "1h": one_hour},
        _FrozenStrategy(decision),  # type: ignore[arg-type]
        BarrierConfig(
            take_profit_pct=0.01,
            stop_loss_pct=0.01,
            horizon_bars=2,
            fee_pct_per_side=0,
            slippage_pct_per_side=0,
            fallback_funding_rate_per_8h=0,
        ),
    )

    assert len(samples) == 1
    assert samples[0].decision_timestamp == decision
    assert samples[0].entry_timestamp == decision
    assert len(samples[0].features) == len(FEATURE_NAMES)
    assert stats.primary_candidates == 2
    assert stats.duplicate_candidates == 1


def test_replay_history_window_matches_live_forming_bar_removal() -> None:
    inherited = resolve_replay_history_window(300)
    explicit = resolve_replay_history_window(300, 123)

    assert inherited.configured_candle_limit == 300
    assert inherited.closed_history_limit == 299
    assert inherited.source == "config_candle_limit_minus_forming_bar"
    assert inherited.forming_bar_excluded is True
    assert explicit.configured_candle_limit == 300
    assert explicit.closed_history_limit == 123
    assert explicit.source == "cli_closed_bars_override"
    with pytest.raises(ValueError, match="at least 21"):
        resolve_replay_history_window(21)


def _oscillating_candles(count: int, interval_ms: int, slope: float) -> list[Candle]:
    candles: list[Candle] = []
    previous_close = 100.0
    for index in range(count):
        close = 100.0 + slope * index + math.sin(index * 0.55) * 0.8
        volume = 10.0 + index % 7
        candles.append(
            Candle(
                index * interval_ms,
                previous_close,
                max(previous_close, close) + 0.25,
                min(previous_close, close) - 0.25,
                close,
                volume,
                quote_volume=volume * close,
            )
        )
        previous_close = close
    return candles


def _small_traditional_config() -> StrategyConfig:
    return StrategyConfig(
        mode="traditional_kline",
        trigger_timeframe="5m",
        regime_timeframe="1h",
        traditional_trend_fast=3,
        traditional_trend_slow=5,
        traditional_signal_fast=2,
        traditional_signal_slow=3,
        traditional_rsi_period=3,
        traditional_macd_fast=2,
        traditional_macd_slow=4,
        traditional_macd_signal=2,
        traditional_atr_period=3,
        traditional_volume_sma_period=3,
        traditional_min_volume_ratio=0.5,
        traditional_pressure_filter_enabled=False,
        traditional_ultra_short_enabled=False,
        traditional_ultra_short_countertrend_enabled=False,
        traditional_ultra_short_reversal_enabled=False,
        traditional_ultra_short_pullback_reversal_enabled=False,
        traditional_ultra_short_pullback_resumption_enabled=False,
        traditional_structural_scalp_enabled=False,
        traditional_predictive_reversal_short_enabled=False,
        traditional_failed_breakout_short_enabled=False,
        traditional_v_recovery_long_shadow=False,
        traditional_allow_1m_impulse=False,
    )


def test_cached_and_uncached_replay_emit_identical_full_signals_on_synthetic_data() -> None:
    series = {
        "1m": _oscillating_candles(1_601, 60_000, 0.002),
        "5m": _oscillating_candles(321, 300_000, 0.015),
        "1h": _oscillating_candles(28, 3_600_000, 0.15),
    }
    config = _small_traditional_config()
    uncached = [
        signal
        for _index, _decision, _histories, signal in _iter_replay_signals(
            series,
            MultiTimeframeStrategy(config),
            29,
            use_replay_cache=False,
        )
    ]
    cached = [
        signal
        for _index, _decision, _histories, signal in _iter_replay_signals(
            series,
            MultiTimeframeStrategy(config),
            29,
            use_replay_cache=True,
        )
    ]

    # Dataclass equality covers side, score, timestamp, reasons and every model
    # audit field. The fixture deliberately produces repeated 5m candidates.
    assert cached == uncached
    assert len(cached) == 1_600
    assert any(signal.side in {"long", "short"} for signal in cached)

    barrier = BarrierConfig(
        take_profit_pct=0.01,
        stop_loss_pct=0.01,
        horizon_bars=3,
        fee_pct_per_side=0,
        slippage_pct_per_side=0,
        fallback_funding_rate_per_8h=0,
    )
    uncached_samples, uncached_stats = replay_primary_candidates(
        series,
        MultiTimeframeStrategy(config),
        barrier,
        history_limit=29,
        use_replay_cache=False,
    )
    cached_samples, cached_stats = replay_primary_candidates(
        series,
        MultiTimeframeStrategy(config),
        barrier,
        history_limit=29,
        use_replay_cache=True,
    )
    assert cached_samples == uncached_samples
    assert cached_stats == uncached_stats
    assert cached_stats.emitted_samples > 0


@pytest.mark.skipif(
    not all((_REAL_REPLAY_DATA / f"{timeframe}.csv").is_file() for timeframe in ("1m", "5m", "1h")),
    reason="optional production-review candle fixture is not present",
)
def test_cached_and_uncached_replay_emit_identical_full_signals_on_real_slice() -> None:
    all_candles = {
        timeframe: load_candles(_REAL_REPLAY_DATA / f"{timeframe}.csv")
        for timeframe in ("1m", "5m", "1h")
    }
    history_limit = 299
    start_index = 25_000
    one_minute = all_candles["1m"][start_index - history_limit : start_index + 61]
    earliest = int(one_minute[0].timestamp)
    latest = int(one_minute[-1].timestamp)
    series = {"1m": one_minute}
    for timeframe, interval in (("5m", 300_000), ("1h", 3_600_000)):
        history_start = earliest - history_limit * interval
        series[timeframe] = [
            candle
            for candle in all_candles[timeframe]
            if history_start <= int(candle.timestamp) <= latest
        ]

    raw_config = load_config(Path(__file__).resolve().parents[1] / "config.binance.model2.json")
    strategy_config = StrategyConfig(**dict(raw_config["strategy"]))
    uncached = [
        signal
        for _index, _decision, _histories, signal in _iter_replay_signals(
            series,
            MultiTimeframeStrategy(strategy_config),
            history_limit,
            use_replay_cache=False,
        )
    ]
    cached = [
        signal
        for _index, _decision, _histories, signal in _iter_replay_signals(
            series,
            MultiTimeframeStrategy(strategy_config),
            history_limit,
            use_replay_cache=True,
        )
    ]

    assert cached == uncached
    assert len(cached) == len(one_minute) - 1


class _FakeDataset:
    def __init__(self, data: list[list[float]], **kwargs: object) -> None:
        self.data = data
        self.kwargs = kwargs


class _FakeBooster:
    best_iteration = 3

    def save_model(self, path: str, num_iteration: int) -> None:
        Path(path).write_text(
            f"tree\nnum_iteration={num_iteration}\n", encoding="utf-8"
        )

    def predict(self, matrix: list[list[float]], num_iteration: int) -> list[float]:
        return [0.75 for _ in matrix]


class _FakeLightGBM:
    Dataset = _FakeDataset

    @staticmethod
    def early_stopping(rounds: int, verbose: bool) -> tuple[int, bool]:
        return rounds, verbose

    @staticmethod
    def train(*args: object, **kwargs: object) -> _FakeBooster:
        return _FakeBooster()


def _model_sample(identifier: str, label: int) -> object:
    @dataclass(frozen=True)
    class Sample:
        candidate_id: str
        features: tuple[float, ...]
        label: int

    return Sample(identifier, tuple(0.0 for _ in FEATURE_NAMES), label)


def test_native_lightgbm_writer_can_be_tested_without_lightgbm_dependency(tmp_path: Path) -> None:
    output = tmp_path / "model.txt"
    booster, digest = train_lightgbm_native(
        [_model_sample("loss", 0), _model_sample("win", 1)],  # type: ignore[arg-type]
        [_model_sample("validation", 1)],  # type: ignore[arg-type]
        output,
        lightgbm_module=_FakeLightGBM,
        num_boost_round=5,
    )

    assert isinstance(booster, _FakeBooster)
    assert output.read_text(encoding="utf-8").startswith("tree\n")
    assert len(digest) == 64
