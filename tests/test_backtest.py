from __future__ import annotations

import csv
from pathlib import Path

from btc_futures_bot.backtest import run_backtest
from btc_futures_bot.models import Signal
from btc_futures_bot.reporting import TradeReporter
from btc_futures_bot.strategy import StrategyConfig


def _write_candles(
    path: Path,
    *,
    count: int,
    interval_ms: int,
    high_offset: float = 0.1,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("timestamp", "open", "high", "low", "close", "volume", "volume_quote"),
        )
        writer.writeheader()
        for index in range(count):
            price = 100.0 + index * 0.01
            writer.writerow(
                {
                    "timestamp": 1_700_000_000_000 + index * interval_ms,
                    "open": price,
                    "high": price + high_offset,
                    "low": price - 0.1,
                    "close": price,
                    "volume": 10.0,
                    "volume_quote": 1000.0,
                }
            )


def test_backtest_matches_production_closed_candle_window(tmp_path: Path) -> None:
    _write_candles(tmp_path / "1m.csv", count=360, interval_ms=60_000)
    _write_candles(tmp_path / "5m.csv", count=72, interval_ms=300_000)
    _write_candles(tmp_path / "1h.csv", count=8, interval_ms=3_600_000)

    class RecordingStrategy:
        config = StrategyConfig(
            mode="traditional_kline",
            trigger_timeframe="5m",
            regime_timeframe="1h",
        )

        def __init__(self) -> None:
            self.observed: list[dict[str, int]] = []

        def evaluate(self, candles_by_timeframe: dict[str, list[object]]) -> Signal:
            self.observed.append(
                {name: len(candles) for name, candles in candles_by_timeframe.items()}
            )
            timestamp = candles_by_timeframe["5m"][-1].timestamp
            return Signal("flat", 0, timestamp, ("recording",))

    strategy = RecordingStrategy()
    run_backtest(tmp_path, strategy=strategy, candle_limit=6)

    assert strategy.observed
    assert any(observation["1m"] == 5 for observation in strategy.observed)
    assert all(
        length <= 5
        for observation in strategy.observed
        for length in observation.values()
    )


def test_backtest_closes_and_reverses_on_strong_opposite_signal(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"
    data_dir.mkdir()
    _write_candles(data_dir / "1m.csv", count=5, interval_ms=60_000)

    class ReversalStrategy:
        config = StrategyConfig(
            mode="scalp",
            trigger_timeframe="1m",
            regime_timeframe="1m",
            min_hold_seconds=0,
            reversal_min_score=5,
        )

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, candles_by_timeframe: dict[str, list[object]]) -> Signal:
            self.calls += 1
            timestamp = candles_by_timeframe["1m"][-1].timestamp
            if self.calls % 2 == 1:
                return Signal("long", 6, timestamp, ("initial_long",))
            return Signal("short", 6, timestamp, ("strong_reversal",))

    reporter = TradeReporter(report_dir)
    try:
        summary = run_backtest(data_dir, strategy=ReversalStrategy(), reporter=reporter)
    finally:
        reporter.close()

    with (report_dir / "trade_report.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary.trades == 1
    assert len(rows) == 1
    assert rows[0]["exit_reason"] == "opposite_signal"
    assert rows[0]["signal_reasons"] == "initial_long"


def test_backtest_keeps_losing_position_for_weak_opposite_signal(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_candles(data_dir / "1m.csv", count=5, interval_ms=60_000)

    class WeakReversalStrategy:
        config = StrategyConfig(
            mode="scalp",
            trigger_timeframe="1m",
            regime_timeframe="1m",
            min_hold_seconds=0,
            reversal_min_score=5,
        )

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, candles_by_timeframe: dict[str, list[object]]) -> Signal:
            self.calls += 1
            timestamp = candles_by_timeframe["1m"][-1].timestamp
            if self.calls == 1:
                return Signal("long", 6, timestamp, ("initial_long",))
            return Signal("short", 4, timestamp, ("weak_reversal",))

    summary = run_backtest(data_dir, strategy=WeakReversalStrategy())

    assert summary.trades == 0


def test_backtest_dynamic_exit_default_does_not_force_fixed_take_profit(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_candles(data_dir / "1m.csv", count=5, interval_ms=60_000, high_offset=10.0)

    class OneLongSignal:
        config = StrategyConfig(
            mode="scalp",
            trigger_timeframe="1m",
            regime_timeframe="1m",
            take_profit_r=0.2,
            enable_profit_trend_exit=False,
            break_even_trigger_r=0.0,
            trailing_trigger_r=0.0,
        )

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, candles_by_timeframe: dict[str, list[object]]) -> Signal:
            self.calls += 1
            timestamp = candles_by_timeframe["1m"][-1].timestamp
            if self.calls == 1:
                return Signal("long", 6, timestamp, ("initial_long",))
            return Signal("flat", 0, timestamp, ("hold",))

    dynamic = run_backtest(data_dir, strategy=OneLongSignal())
    fixed = run_backtest(data_dir, strategy=OneLongSignal(), use_fixed_take_profit=True)

    assert dynamic.trades == 0
    assert fixed.trades == 1
