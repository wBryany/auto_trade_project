from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from ..models import Candle, Signal
from ..strategy import MultiTimeframeStrategy, strategy_replay_cache
from .features import FEATURE_NAMES, FeatureValidationError, extract_features
from .policy import LIVE_EXECUTION_BLOCKER, TRAINING_PIPELINE_VERSION


KLINE_DATASET_VERSION = "binance-usdm-klines-v1"
DEFAULT_LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 17,
    "feature_fraction_seed": 17,
    "bagging_seed": 17,
    "data_random_seed": 17,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 1,
}
TIMEFRAME_MILLISECONDS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

@dataclass(frozen=True)
class FundingRate:
    """One Binance funding settlement, expressed as a decimal rate."""

    timestamp: int
    rate: float


@dataclass(frozen=True)
class BarrierConfig:
    """Fixed, auditable outcome definition for every primary-strategy candidate."""

    take_profit_pct: float = 0.0125
    stop_loss_pct: float = 0.005
    horizon_bars: int = 90
    fee_pct_per_side: float = 0.0005
    slippage_pct_per_side: float = 0.0002
    fallback_funding_rate_per_8h: float = 0.0001

    def __post_init__(self) -> None:
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if not 0 < self.stop_loss_pct < 1:
            raise ValueError("stop_loss_pct must be between 0 and 1")
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        for name in (
            "fee_pct_per_side",
            "slippage_pct_per_side",
            "fallback_funding_rate_per_8h",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class BarrierOutcome:
    label: int
    barrier: str
    side: str
    entry_timestamp: int
    exit_timestamp: int
    entry_price: float
    exit_price: float
    bars_held: int
    gross_return: float
    fee_cost: float
    slippage_cost: float
    funding_cost: float
    net_return: float


@dataclass(frozen=True)
class TrainingSample:
    candidate_id: str
    decision_timestamp: int
    entry_timestamp: int
    exit_timestamp: int
    signal_timestamp: int
    side: str
    score: int
    reasons: tuple[str, ...]
    features: tuple[float, ...]
    label: int
    barrier: str
    gross_return: float
    fee_cost: float
    slippage_cost: float
    funding_cost: float
    net_return: float


@dataclass(frozen=True)
class ReplayStats:
    evaluated_decisions: int
    primary_candidates: int
    emitted_samples: int
    duplicate_candidates: int
    incomplete_horizons: int
    invalid_features: int


@dataclass(frozen=True)
class ReplayHistoryWindow:
    """The configured fetch depth and exact closed-bar depth used in replay."""

    configured_candle_limit: int
    closed_history_limit: int
    source: str
    forming_bar_excluded: bool = True


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    purge_ms: int = 0
    embargo_ms: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be below 1")
        if self.purge_ms < 0 or self.embargo_ms < 0:
            raise ValueError("purge_ms and embargo_ms cannot be negative")


T = TypeVar("T")


@dataclass(frozen=True)
class PurgedSplit:
    train: tuple[Any, ...]
    validation: tuple[Any, ...]
    holdout: tuple[Any, ...]
    train_validation_boundary: int
    validation_holdout_boundary: int
    purged_train: int
    purged_validation: int
    embargoed_validation: int
    embargoed_holdout: int


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    total_candidates: int
    selected_trades: int
    coverage: float
    wins: int
    win_rate: float | None
    net_return: float
    net_expectancy: float | None
    profit_factor: float | None


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    metrics: ThresholdMetrics
    eligible: bool
    reason: str
    evaluated_thresholds: int


@dataclass(frozen=True)
class ApprovalPolicy:
    min_train_samples: int = 100
    min_validation_samples: int = 30
    min_holdout_samples: int = 30
    min_selected_validation: int = 10
    min_selected_holdout: int = 10
    min_coverage: float = 0.05
    min_validation_expectancy: float = 0.0
    min_validation_profit_factor: float = 1.0
    min_holdout_expectancy: float = 0.0
    min_holdout_profit_factor: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "min_train_samples",
            "min_validation_samples",
            "min_holdout_samples",
            "min_selected_validation",
            "min_selected_holdout",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0 <= self.min_coverage <= 1:
            raise ValueError("min_coverage must be between 0 and 1")


def fixed_triple_barrier_live_approval(
    statistical_reasons: Sequence[str],
) -> tuple[bool, bool, tuple[str, ...], tuple[str, ...]]:
    """Apply the non-overridable live blocker for this training pipeline.

    The fixed label is intentionally simpler than production position
    management. It may be enforced in paper mode, but its metrics can never
    turn a generated artifact into a live-approved artifact.
    """

    live_blockers = (LIVE_EXECUTION_BLOCKER,)
    all_reasons = (*tuple(str(reason) for reason in statistical_reasons), *live_blockers)
    return False, False, live_blockers, all_reasons


class _TemporalRecord(Protocol):
    decision_timestamp: int
    exit_timestamp: int


def resolve_replay_history_window(
    configured_candle_limit: int,
    closed_bars_override: int | None = None,
) -> ReplayHistoryWindow:
    """Resolve replay depth to the same closed history seen by the live engine.

    Exchanges return up to ``candle_limit`` rows including the newest forming
    candle. ``TradingEngine.evaluate_once`` removes that row before invoking
    the strategy, so a configured value of 300 means 299 closed history bars.
    An explicit CLI override is already expressed in closed bars.
    """

    configured = int(configured_candle_limit)
    if configured < 2:
        raise ValueError("configured candle_limit must be at least 2")
    if closed_bars_override is None:
        selected = configured - 1
        source = "config_candle_limit_minus_forming_bar"
    else:
        selected = int(closed_bars_override)
        source = "cli_closed_bars_override"
    if selected < 21:
        raise ValueError("closed replay history limit must be at least 21")
    return ReplayHistoryWindow(
        configured_candle_limit=configured,
        closed_history_limit=selected,
        source=source,
    )


def triple_barrier_label(
    side: str,
    future_one_minute_candles: Sequence[Candle],
    config: BarrierConfig,
    funding_rates: Sequence[FundingRate] = (),
) -> BarrierOutcome:
    """Label a next-open fill with fixed TP/SL/horizon barriers.

    The first supplied candle is the entry candle and its open is the reference
    fill.  A candle touching TP and SL is deliberately labelled as a stop; 1m
    OHLC cannot establish the favourable intrabar ordering.  Fees, two-sided
    slippage and signed funding are deducted before deciding the binary label.
    """

    normalised_side = str(side).strip().lower()
    if normalised_side not in {"long", "short"}:
        raise ValueError("side must be long or short")
    if len(future_one_minute_candles) < config.horizon_bars:
        raise ValueError(
            f"triple barrier needs {config.horizon_bars} complete future candles; "
            f"got {len(future_one_minute_candles)}"
        )
    candles = list(future_one_minute_candles[: config.horizon_bars])
    _validate_future_candles(candles)
    entry = float(candles[0].open)
    direction = 1.0 if normalised_side == "long" else -1.0
    target = entry * (1.0 + direction * config.take_profit_pct)
    stop = entry * (1.0 - direction * config.stop_loss_pct)

    barrier = "horizon"
    exit_price = float(candles[-1].close)
    exit_index = len(candles) - 1
    for index, candle in enumerate(candles):
        if normalised_side == "long":
            stop_hit = float(candle.low) <= stop
            target_hit = float(candle.high) >= target
        else:
            stop_hit = float(candle.high) >= stop
            target_hit = float(candle.low) <= target
        # Pessimistic ordering is mandatory when both barriers occur in one K.
        if stop_hit:
            barrier = "stop_loss"
            exit_price = stop
            exit_index = index
            break
        if target_hit:
            barrier = "take_profit"
            exit_price = target
            exit_index = index
            break

    exit_candle = candles[exit_index]
    # Candle timestamps are open times. A barrier can settle during the bar;
    # using its close boundary is conservative for funding and overlap purge.
    exit_timestamp = int(exit_candle.timestamp) + TIMEFRAME_MILLISECONDS["1m"]
    gross_return = direction * (exit_price / entry - 1.0)
    fee_cost = config.fee_pct_per_side * (1.0 + exit_price / entry)
    slippage_cost = config.slippage_pct_per_side * (1.0 + exit_price / entry)
    funding_cost = funding_cost_for_position(
        normalised_side,
        int(candles[0].timestamp),
        exit_timestamp,
        funding_rates,
        fallback_rate=config.fallback_funding_rate_per_8h,
    )
    net_return = gross_return - fee_cost - slippage_cost - funding_cost
    return BarrierOutcome(
        label=int(net_return > 0.0),
        barrier=barrier,
        side=normalised_side,
        entry_timestamp=int(candles[0].timestamp),
        exit_timestamp=exit_timestamp,
        entry_price=entry,
        exit_price=exit_price,
        bars_held=exit_index + 1,
        gross_return=gross_return,
        fee_cost=fee_cost,
        slippage_cost=slippage_cost,
        funding_cost=funding_cost,
        net_return=net_return,
    )


# A concise alias makes the primitive easy to use in audit notebooks.
label_triple_barrier = triple_barrier_label


def funding_cost_for_position(
    side: str,
    entry_timestamp: int,
    exit_timestamp: int,
    funding_rates: Sequence[FundingRate] = (),
    *,
    fallback_rate: float = 0.0,
) -> float:
    """Return funding as entry-notional return (negative means a credit)."""

    if exit_timestamp <= entry_timestamp:
        return 0.0
    direction = 1.0 if side == "long" else -1.0
    selected = [
        float(item.rate)
        for item in funding_rates
        if entry_timestamp < int(item.timestamp) <= exit_timestamp
    ]
    if selected:
        return direction * sum(selected)
    if fallback_rate <= 0:
        return 0.0
    interval = 8 * 60 * 60 * 1000
    first_settlement = ((int(entry_timestamp) // interval) + 1) * interval
    settlements = (
        0
        if first_settlement > exit_timestamp
        else 1 + (int(exit_timestamp) - first_settlement) // interval
    )
    # The fallback is conservative and has no sign forecast; charge both sides.
    return float(fallback_rate) * settlements


def _iter_replay_signals(
    series: Mapping[str, Sequence[Candle]],
    strategy: MultiTimeframeStrategy,
    history_limit: int,
    *,
    use_replay_cache: bool = True,
) -> Iterator[tuple[int, int, dict[str, Sequence[Candle]], Signal]]:
    """Yield every eligible minute's signal with its exact as-of histories.

    ``use_replay_cache=False`` is the deterministic reference path used by
    equivalence tests and benchmarks. Both paths share the same close cursors
    and immutable views, so the only difference is reuse of pure strategy
    helper results.
    """

    required = ("1m", "5m", "1h")
    one_minute = series["1m"]
    open_times = {
        name: [int(candle.timestamp) for candle in series[name]]
        for name in required
    }
    closed_counts = {name: 0 for name in required}
    cache_scope = strategy_replay_cache() if use_replay_cache else nullcontext()
    with cache_scope:
        for one_index in range(len(one_minute) - 1):
            decision_timestamp = (
                int(one_minute[one_index].timestamp)
                + TIMEFRAME_MILLISECONDS["1m"]
            )
            # A gap must never be interpreted as the next tradable 1m open.
            if int(one_minute[one_index + 1].timestamp) != decision_timestamp:
                continue

            histories: dict[str, Sequence[Candle]] = {}
            for timeframe in required:
                close_cutoff = decision_timestamp - TIMEFRAME_MILLISECONDS[timeframe]
                timestamps = open_times[timeframe]
                last_closed = closed_counts[timeframe]
                while (
                    last_closed < len(timestamps)
                    and timestamps[last_closed] <= close_cutoff
                ):
                    last_closed += 1
                closed_counts[timeframe] = last_closed
                start = max(0, last_closed - history_limit)
                histories[timeframe] = _CandleWindow(
                    series[timeframe],
                    start,
                    last_closed,
                )
            yield (
                one_index,
                decision_timestamp,
                histories,
                strategy.evaluate(histories),
            )


def replay_primary_candidates(
    candles_by_timeframe: Mapping[str, Sequence[Candle]],
    strategy: MultiTimeframeStrategy,
    barrier_config: BarrierConfig,
    funding_rates: Sequence[FundingRate] = (),
    *,
    history_limit: int = 500,
    use_replay_cache: bool = True,
) -> tuple[list[TrainingSample], ReplayStats]:
    """Replay the frozen primary strategy and emit each candidate exactly once.

    At decision time every supplied timeframe is joined *as of its close*. The
    fill is the following 1m candle's open, exactly one minute after the final
    closed 1m feature candle. A non-flat signal timestamp is consumed once,
    mirroring ``TradingEngine.last_signal_timestamp``; no model-generated or
    parameter-swept candidates are added to the training population. The
    ``use_replay_cache=False`` reference mode exists for equivalence audits.
    """

    required = ("1m", "5m", "1h")
    missing = [name for name in required if name not in candles_by_timeframe]
    if missing:
        raise ValueError(f"missing replay timeframes: {', '.join(missing)}")
    if history_limit < 21:
        raise ValueError("history_limit must be at least 21")
    series = {name: _normalise_candles(candles_by_timeframe[name]) for name in required}
    one_minute = series["1m"]

    emitted: list[TrainingSample] = []
    # TradingEngine starts at zero and therefore also treats a zero-timestamp
    # signal as already consumed.
    consumed_signal_timestamps: set[int] = {0}
    evaluated = primary = duplicates = incomplete = invalid = 0
    for one_index, decision_timestamp, histories, signal in _iter_replay_signals(
        series,
        strategy,
        history_limit,
        use_replay_cache=use_replay_cache,
    ):
        evaluated += 1
        if signal.side not in {"long", "short"}:
            continue
        primary += 1
        signal_timestamp = int(signal.timestamp)
        if signal_timestamp in consumed_signal_timestamps:
            duplicates += 1
            continue
        consumed_signal_timestamps.add(signal_timestamp)

        future = one_minute[one_index + 1 : one_index + 1 + barrier_config.horizon_bars]
        if len(future) < barrier_config.horizon_bars or any(
            int(future[index].timestamp) - int(future[index - 1].timestamp)
            != TIMEFRAME_MILLISECONDS["1m"]
            for index in range(1, len(future))
        ):
            incomplete += 1
            continue
        try:
            feature_mapping = extract_features(
                signal=signal,
                candles_by_timeframe=histories,
            )
        except FeatureValidationError:
            invalid += 1
            continue
        outcome = triple_barrier_label(
            signal.side,
            future,
            barrier_config,
            funding_rates,
        )
        features = tuple(float(feature_mapping[name]) for name in FEATURE_NAMES)
        identity = hashlib.sha256(
            (
                f"{decision_timestamp}|{signal_timestamp}|{signal.side}|"
                f"{','.join(signal.reasons)}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        emitted.append(
            TrainingSample(
                candidate_id=identity,
                decision_timestamp=decision_timestamp,
                entry_timestamp=outcome.entry_timestamp,
                exit_timestamp=outcome.exit_timestamp,
                signal_timestamp=signal_timestamp,
                side=signal.side,
                score=int(signal.score),
                reasons=tuple(signal.reasons),
                features=features,
                label=outcome.label,
                barrier=outcome.barrier,
                gross_return=outcome.gross_return,
                fee_cost=outcome.fee_cost,
                slippage_cost=outcome.slippage_cost,
                funding_cost=outcome.funding_cost,
                net_return=outcome.net_return,
            )
        )
    return emitted, ReplayStats(
        evaluated_decisions=evaluated,
        primary_candidates=primary,
        emitted_samples=len(emitted),
        duplicate_candidates=duplicates,
        incomplete_horizons=incomplete,
        invalid_features=invalid,
    )


# Backwards-friendly, descriptive alias for callers building a dataset.
build_training_samples = replay_primary_candidates


def chronological_purged_split(
    samples: Sequence[T],
    config: SplitConfig | None = None,
) -> PurgedSplit:
    """Make train/validation/holdout partitions without label-window leakage."""

    selected_config = config or SplitConfig()
    if len(samples) < 3:
        raise ValueError("at least three samples are required for a chronological split")
    ordered = sorted(samples, key=lambda sample: _record_value(sample, "decision_timestamp"))
    train_end = max(1, int(len(ordered) * selected_config.train_fraction))
    validation_end = max(train_end + 1, int(
        len(ordered) * (selected_config.train_fraction + selected_config.validation_fraction)
    ))
    validation_end = min(validation_end, len(ordered) - 1)
    train_validation_boundary = _record_value(ordered[train_end], "decision_timestamp")
    validation_holdout_boundary = _record_value(
        ordered[validation_end], "decision_timestamp"
    )

    raw_train = ordered[:train_end]
    raw_validation = ordered[train_end:validation_end]
    raw_holdout = ordered[validation_end:]
    train = tuple(
        sample
        for sample in raw_train
        if _record_value(sample, "exit_timestamp")
        < train_validation_boundary - selected_config.purge_ms
    )
    validation_after_embargo = [
        sample
        for sample in raw_validation
        if _record_value(sample, "decision_timestamp")
        >= train_validation_boundary + selected_config.embargo_ms
    ]
    validation = tuple(
        sample
        for sample in validation_after_embargo
        if _record_value(sample, "exit_timestamp")
        < validation_holdout_boundary - selected_config.purge_ms
    )
    holdout = tuple(
        sample
        for sample in raw_holdout
        if _record_value(sample, "decision_timestamp")
        >= validation_holdout_boundary + selected_config.embargo_ms
    )
    return PurgedSplit(
        train=train,
        validation=validation,
        holdout=holdout,
        train_validation_boundary=train_validation_boundary,
        validation_holdout_boundary=validation_holdout_boundary,
        purged_train=len(raw_train) - len(train),
        purged_validation=len(validation_after_embargo) - len(validation),
        embargoed_validation=len(raw_validation) - len(validation_after_embargo),
        embargoed_holdout=len(raw_holdout) - len(holdout),
    )


purged_chronological_split = chronological_purged_split


def threshold_metrics(
    probabilities: Sequence[float],
    net_returns: Sequence[float],
    threshold: float,
) -> ThresholdMetrics:
    if len(probabilities) != len(net_returns):
        raise ValueError("probabilities and net_returns must have equal length")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    selected = [
        float(value)
        for probability, value in zip(probabilities, net_returns, strict=True)
        if float(probability) >= threshold
    ]
    total = len(probabilities)
    positives = [value for value in selected if value > 0]
    negatives = [value for value in selected if value < 0]
    gross_profit = sum(positives)
    gross_loss = abs(sum(negatives))
    profit_factor: float | None
    if gross_loss:
        profit_factor = gross_profit / gross_loss
    elif gross_profit:
        profit_factor = None  # JSON-safe representation of positive infinity.
    else:
        profit_factor = 0.0
    return ThresholdMetrics(
        threshold=float(threshold),
        total_candidates=total,
        selected_trades=len(selected),
        coverage=(len(selected) / total) if total else 0.0,
        wins=len(positives),
        win_rate=(len(positives) / len(selected)) if selected else None,
        net_return=sum(selected),
        net_expectancy=(sum(selected) / len(selected)) if selected else None,
        profit_factor=profit_factor,
    )


def choose_validation_threshold(
    probabilities: Sequence[float],
    net_returns: Sequence[float],
    *,
    thresholds: Sequence[float] | None = None,
    min_selected: int = 10,
    min_coverage: float = 0.05,
) -> ThresholdSelection:
    """Choose a gate on validation only by expectancy, PF, then coverage."""

    if min_selected < 1:
        raise ValueError("min_selected must be positive")
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between 0 and 1")
    if thresholds is None:
        thresholds = tuple(index / 100 for index in range(50, 96))
    candidates = sorted({float(value) for value in thresholds})
    if not candidates or any(value < 0 or value > 1 for value in candidates):
        raise ValueError("thresholds must contain probabilities between 0 and 1")
    metrics = [threshold_metrics(probabilities, net_returns, value) for value in candidates]
    eligible = [
        item
        for item in metrics
        if item.selected_trades >= min_selected and item.coverage >= min_coverage
    ]
    pool = eligible or [item for item in metrics if item.selected_trades] or metrics
    best = max(pool, key=_threshold_rank)
    reason = (
        "selected_on_validation_net_expectancy_profit_factor_coverage"
        if eligible
        else "no_threshold_met_validation_sample_or_coverage_floor"
    )
    return ThresholdSelection(
        threshold=best.threshold,
        metrics=best,
        eligible=bool(eligible),
        reason=reason,
        evaluated_thresholds=len(metrics),
    )


select_validation_threshold = choose_validation_threshold


def assess_live_approval(
    split: PurgedSplit,
    validation_selection: ThresholdSelection,
    holdout_metrics: ThresholdMetrics,
    policy: ApprovalPolicy | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Apply fixed sufficiency/quality gates after holdout is evaluated once."""

    selected_policy = policy or ApprovalPolicy()
    reasons: list[str] = []
    for name, actual, minimum in (
        ("train", len(split.train), selected_policy.min_train_samples),
        ("validation", len(split.validation), selected_policy.min_validation_samples),
        ("holdout", len(split.holdout), selected_policy.min_holdout_samples),
    ):
        if actual < minimum:
            reasons.append(f"insufficient_{name}_samples:{actual}<{minimum}")
    validation = validation_selection.metrics
    if not validation_selection.eligible:
        reasons.append("validation_threshold_ineligible")
    if validation.selected_trades < selected_policy.min_selected_validation:
        reasons.append(
            "insufficient_selected_validation:"
            f"{validation.selected_trades}<{selected_policy.min_selected_validation}"
        )
    if holdout_metrics.selected_trades < selected_policy.min_selected_holdout:
        reasons.append(
            "insufficient_selected_holdout:"
            f"{holdout_metrics.selected_trades}<{selected_policy.min_selected_holdout}"
        )
    if validation.coverage < selected_policy.min_coverage:
        reasons.append("validation_coverage_below_floor")
    if holdout_metrics.coverage < selected_policy.min_coverage:
        reasons.append("holdout_coverage_below_floor")
    _append_quality_reasons(
        reasons,
        "validation",
        validation,
        selected_policy.min_validation_expectancy,
        selected_policy.min_validation_profit_factor,
    )
    _append_quality_reasons(
        reasons,
        "holdout",
        holdout_metrics,
        selected_policy.min_holdout_expectancy,
        selected_policy.min_holdout_profit_factor,
    )
    return not reasons, tuple(reasons)


def split_ranges(split: PurgedSplit) -> dict[str, dict[str, int | None]]:
    return {
        name: _range_for(getattr(split, name))
        for name in ("train", "validation", "holdout")
    }


def load_candles(path: str | Path) -> list[Candle]:
    """Load downloader JSONL (or an equivalent CSV) without third-party deps."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"candle file does not exist: {source}")
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows: Iterable[Any] = list(csv.DictReader(handle))
    else:
        text = source.read_text(encoding="utf-8-sig")
        stripped = text.lstrip()
        if stripped.startswith("["):
            rows = json.loads(text)
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    candles = [_candle_from_row(row) for row in rows]
    return _normalise_candles(candles)


def load_funding_rates(path: str | Path | None) -> list[FundingRate]:
    if path is None or not Path(path).is_file():
        return []
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    result = [
        FundingRate(
            timestamp=int(row.get("funding_time", row.get("fundingTime"))),
            rate=float(row.get("funding_rate", row.get("fundingRate"))),
        )
        for row in rows
    ]
    return sorted({(item.timestamp, item.rate): item for item in result}.values(), key=lambda x: x.timestamp)


def samples_to_matrix(samples: Sequence[TrainingSample]) -> tuple[list[list[float]], list[int]]:
    width = len(FEATURE_NAMES)
    matrix: list[list[float]] = []
    labels: list[int] = []
    for sample in samples:
        if len(sample.features) != width:
            raise ValueError(
                f"sample {sample.candidate_id} has {len(sample.features)} features; expected {width}"
            )
        matrix.append([float(value) for value in sample.features])
        labels.append(int(sample.label))
    return matrix, labels


def _samples_to_numpy(samples: Sequence[TrainingSample]) -> tuple[Any, Any]:
    """Return the dense arrays accepted by LightGBM's native API."""

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy is optional; install with `pip install -e .[model2]` before training"
        ) from error
    matrix, labels = samples_to_matrix(samples)
    feature_array = np.asarray(matrix, dtype=np.float64, order="C").reshape(
        len(matrix), len(FEATURE_NAMES)
    )
    label_array = np.asarray(labels, dtype=np.int32)
    return feature_array, label_array


def train_lightgbm_native(
    train_samples: Sequence[TrainingSample],
    validation_samples: Sequence[TrainingSample],
    output_path: str | Path,
    *,
    params: Mapping[str, Any] | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    lightgbm_module: Any | None = None,
) -> tuple[Any, str]:
    """Train through LightGBM's native API and atomically save text format."""

    if not train_samples or not validation_samples:
        raise ValueError("non-empty train and validation samples are required")
    labels = {int(sample.label) for sample in train_samples}
    if labels != {0, 1}:
        raise ValueError("training samples must contain both binary classes")
    if lightgbm_module is None:
        try:
            import lightgbm as lightgbm_module  # type: ignore[no-redef]
        except ImportError as error:
            raise RuntimeError(
                "LightGBM is optional; install with `pip install -e .[model2]` before training"
            ) from error
    train_x, train_y = _samples_to_numpy(train_samples)
    validation_x, validation_y = _samples_to_numpy(validation_samples)
    train_set = lightgbm_module.Dataset(
        train_x,
        label=train_y,
        feature_name=list(FEATURE_NAMES),
        free_raw_data=False,
    )
    validation_set = lightgbm_module.Dataset(
        validation_x,
        label=validation_y,
        reference=train_set,
        feature_name=list(FEATURE_NAMES),
        free_raw_data=False,
    )
    selected_params = dict(DEFAULT_LIGHTGBM_PARAMS)
    selected_params.update(dict(params or {}))
    callbacks = []
    if early_stopping_rounds > 0 and hasattr(lightgbm_module, "early_stopping"):
        callbacks.append(lightgbm_module.early_stopping(early_stopping_rounds, verbose=False))
    booster = lightgbm_module.train(
        selected_params,
        train_set,
        num_boost_round=max(1, int(num_boost_round)),
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=callbacks,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        best_iteration = int(getattr(booster, "best_iteration", 0) or 0)
        booster.save_model(temporary_name, num_iteration=best_iteration or -1)
        with open(temporary_name, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return booster, sha256_file(destination)


def predict_samples(booster: Any, samples: Sequence[TrainingSample]) -> list[float]:
    if not samples:
        return []
    matrix, _labels = _samples_to_numpy(samples)
    best_iteration = int(getattr(booster, "best_iteration", 0) or 0)
    values = booster.predict(matrix, num_iteration=best_iteration or -1)
    result = [float(value) for value in values]
    if len(result) != len(samples) or any(not 0 <= value <= 1 for value in result):
        raise ValueError("LightGBM returned invalid prediction probabilities")
    return result


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataclass_json(value: Any) -> Any:
    """Convert nested training dataclasses to manifest-ready JSON values."""

    if hasattr(value, "__dataclass_fields__"):
        return {key: dataclass_json(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): dataclass_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_json(item) for item in value]
    return value


class _CandleWindow(Sequence[Candle]):
    """Zero-copy immutable view over one normalized candle series.

    The strategy API accepts ``Sequence[Candle]`` and never mutates its input.
    Retaining the source and exact half-open indices also gives replay caches a
    collision-free identity without hashing hundreds of candle values on every
    minute.
    """

    __slots__ = ("_source", "_start", "_stop")

    def __init__(self, source: Sequence[Candle], start: int, stop: int) -> None:
        size = len(source)
        if start < 0 or stop < start or stop > size:
            raise IndexError("invalid candle window bounds")
        self._source = source
        self._start = int(start)
        self._stop = int(stop)

    def __len__(self) -> int:
        return self._stop - self._start

    def __iter__(self) -> Iterator[Candle]:
        source = self._source
        for position in range(self._start, self._stop):
            yield source[position]

    def __getitem__(self, index: int | slice) -> Candle | Sequence[Candle]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return _CandleWindow(
                    self._source,
                    self._start + start,
                    self._start + stop,
                )
            return [self[position] for position in range(start, stop, step)]
        selected = int(index)
        if selected < 0:
            selected += len(self)
        if selected < 0 or selected >= len(self):
            raise IndexError("candle window index out of range")
        return self._source[self._start + selected]

    @property
    def strategy_replay_cache_key(self) -> tuple[int, int, int]:
        return id(self._source), self._start, self._stop


def _validate_future_candles(candles: Sequence[Candle]) -> None:
    previous: int | None = None
    for candle in candles:
        timestamp = int(candle.timestamp)
        numeric = tuple(float(value) for value in (
            candle.open, candle.high, candle.low, candle.close, candle.volume
        ))
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("future candle contains a non-finite value")
        if min(numeric[:4]) <= 0 or float(candle.high) < max(
            float(candle.open), float(candle.close), float(candle.low)
        ) or float(candle.low) > min(
            float(candle.open), float(candle.close), float(candle.high)
        ):
            raise ValueError(f"invalid future candle at {timestamp}")
        if previous is not None and timestamp - previous != TIMEFRAME_MILLISECONDS["1m"]:
            raise ValueError("future 1m candles must be contiguous")
        previous = timestamp


def _normalise_candles(candles: Sequence[Candle]) -> list[Candle]:
    by_timestamp = {int(candle.timestamp): candle for candle in candles}
    return [by_timestamp[key] for key in sorted(by_timestamp)]


def _candle_from_row(row: Any) -> Candle:
    if isinstance(row, Mapping):
        def value(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in row and row[name] not in {None, ""}:
                    return row[name]
            return default

        return Candle(
            int(value("open_time", "openTime", "timestamp")),
            float(value("open")),
            float(value("high")),
            float(value("low")),
            float(value("close")),
            float(value("volume")),
            quote_volume=float(value("quote_volume", "quoteVolume", default=0.0)),
        )
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) >= 6:
        return Candle(
            int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]),
            float(row[5]), quote_volume=float(row[7]) if len(row) > 7 else None,
        )
    raise ValueError("unsupported candle row")


def _record_value(record: Any, name: str) -> int:
    value = record[name] if isinstance(record, Mapping) else getattr(record, name)
    return int(value)


def _threshold_rank(item: ThresholdMetrics) -> tuple[float, float, float, int, float]:
    expectancy = item.net_expectancy if item.net_expectancy is not None else -math.inf
    profit_factor = math.inf if item.profit_factor is None and item.net_return > 0 else float(
        item.profit_factor or 0.0
    )
    return expectancy, profit_factor, item.coverage, item.selected_trades, item.threshold


def _append_quality_reasons(
    reasons: list[str],
    prefix: str,
    metrics: ThresholdMetrics,
    minimum_expectancy: float,
    minimum_profit_factor: float,
) -> None:
    expectancy = metrics.net_expectancy
    if expectancy is None or expectancy <= minimum_expectancy:
        reasons.append(f"{prefix}_net_expectancy_not_positive")
    profit_factor = math.inf if metrics.profit_factor is None and metrics.net_return > 0 else float(
        metrics.profit_factor or 0.0
    )
    if profit_factor < minimum_profit_factor:
        reasons.append(f"{prefix}_profit_factor_below_floor")


def _range_for(samples: Sequence[Any]) -> dict[str, int | None]:
    if not samples:
        return {"samples": 0, "decision_start": None, "decision_end": None, "label_end": None}
    return {
        "samples": len(samples),
        "decision_start": _record_value(samples[0], "decision_timestamp"),
        "decision_end": _record_value(samples[-1], "decision_timestamp"),
        "label_end": max(_record_value(item, "exit_timestamp") for item in samples),
    }
