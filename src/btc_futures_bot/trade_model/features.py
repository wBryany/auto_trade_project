from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from ..models import Candle, Signal


FEATURE_SCHEMA_VERSION = "ohlcv-meta-v1"
REQUIRED_TIMEFRAMES = ("1m", "5m", "1h")
MINIMUM_CANDLES = 21
_TIMEFRAME_MS = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}
_REASON_BUCKETS = 4


def _feature_names() -> tuple[str, ...]:
    names = ["signal_side", "signal_score", "signal_reason_count"]
    names.extend(f"signal_reason_bucket_{index}" for index in range(_REASON_BUCKETS))
    for timeframe in REQUIRED_TIMEFRAMES:
        prefix = f"tf_{timeframe}"
        names.extend(
            (
                f"{prefix}_side_return_1",
                f"{prefix}_side_return_5",
                f"{prefix}_realized_vol_20",
                f"{prefix}_atr_pct_14",
                f"{prefix}_side_sma_gap_20",
                f"{prefix}_range_pct",
                f"{prefix}_side_body_pct",
                f"{prefix}_side_close_location",
                f"{prefix}_volume_ratio_20",
                f"{prefix}_quote_volume_ratio_20",
            )
        )
    return tuple(names)


FEATURE_NAMES = _feature_names()


def feature_schema_hash() -> str:
    """Return the stable fingerprint stored in every model manifest.

    The version is deliberately part of the digest.  A transformation change
    must bump ``FEATURE_SCHEMA_VERSION`` even when its output column name stays
    the same.
    """

    payload = {
        "version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


FEATURE_SCHEMA_HASH = feature_schema_hash()


class FeatureValidationError(ValueError):
    """Raised when a candidate cannot be represented without unsafe data."""


def strategy_config_hash(config: Any) -> str:
    """Hash the primary-strategy configuration in a training/runtime-safe way."""

    if isinstance(config, Mapping):
        # Materialise every StrategyConfig default before hashing.  Otherwise
        # a code-default change could alter live behaviour without changing a
        # sparse JSON mapping (and therefore without invalidating the model).
        from ..strategy import StrategyConfig

        try:
            config = asdict(StrategyConfig(**dict(config)))
        except TypeError as error:
            raise ValueError(f"invalid primary strategy configuration: {error}") from error
    elif is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    elif not isinstance(config, Mapping) and hasattr(config, "__dict__"):
        config = vars(config)
    if not isinstance(config, Mapping):
        raise TypeError("strategy configuration must be a mapping or dataclass")

    try:
        encoded = json.dumps(
            _normalise_json(config),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"strategy configuration is not JSON-serializable: {error}") from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalise_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def extract_features(
    signal: Signal,
    candles_by_timeframe: Mapping[str, Sequence[Candle]],
) -> dict[str, float]:
    """Build the sole feature representation used by training and inference.

    ``candles_by_timeframe`` must contain *closed* bars.  The live engine and
    backtester are responsible for the close-time/as-of join; this function
    validates ordering, continuity and values, then uses only the latest 21
    closed bars from 1m, 5m and 1h.
    """

    side = str(signal.side).lower()
    if side not in {"long", "short"}:
        raise FeatureValidationError("entry features require a long or short signal")
    direction = 1.0 if side == "long" else -1.0
    try:
        score = float(signal.score)
    except (TypeError, ValueError) as error:
        raise FeatureValidationError("signal score must be numeric") from error
    if not math.isfinite(score):
        raise FeatureValidationError("signal score must be finite")

    reasons = tuple(str(reason).strip() for reason in (signal.reasons or ()))
    buckets = [0.0] * _REASON_BUCKETS
    for reason in reasons:
        digest = hashlib.sha256(reason.lower().encode("utf-8")).digest()
        buckets[int.from_bytes(digest[:2], "big") % _REASON_BUCKETS] += 1.0

    features: dict[str, float] = {
        "signal_side": direction,
        "signal_score": score,
        "signal_reason_count": float(len(reasons)),
    }
    features.update(
        {f"signal_reason_bucket_{index}": value for index, value in enumerate(buckets)}
    )

    for timeframe in REQUIRED_TIMEFRAMES:
        raw_candles = candles_by_timeframe.get(timeframe)
        if raw_candles is None:
            raise FeatureValidationError(f"missing required timeframe: {timeframe}")
        candles = _validated_window(timeframe, raw_candles)
        features.update(_timeframe_features(timeframe, candles, direction))

    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("internal feature order no longer matches FEATURE_NAMES")
    for name, value in features.items():
        if not math.isfinite(value):
            raise FeatureValidationError(f"feature {name} is not finite")
    return features


def feature_values(features: Mapping[str, float]) -> list[float]:
    """Order a feature mapping exactly as the native LightGBM model expects."""

    missing = [name for name in FEATURE_NAMES if name not in features]
    extras = [name for name in features if name not in FEATURE_NAMES]
    if missing or extras:
        raise FeatureValidationError(
            f"feature mapping does not match schema (missing={missing}, extras={extras})"
        )
    values = [float(features[name]) for name in FEATURE_NAMES]
    if not all(math.isfinite(value) for value in values):
        raise FeatureValidationError("feature mapping contains a non-finite value")
    return values


def _validated_window(timeframe: str, candles: Sequence[Candle]) -> list[Candle]:
    if len(candles) < MINIMUM_CANDLES:
        raise FeatureValidationError(
            f"{timeframe} requires at least {MINIMUM_CANDLES} closed candles; got {len(candles)}"
        )
    window = list(candles[-MINIMUM_CANDLES:])
    interval_ms = _TIMEFRAME_MS[timeframe]
    previous_timestamp: int | None = None
    for index, candle in enumerate(window):
        try:
            timestamp = int(candle.timestamp)
            open_price = float(candle.open)
            high = float(candle.high)
            low = float(candle.low)
            close = float(candle.close)
            volume = float(candle.volume)
            quote_volume = (
                None if candle.quote_volume is None else float(candle.quote_volume)
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise FeatureValidationError(f"invalid {timeframe} candle at index {index}") from error
        numeric = (open_price, high, low, close, volume)
        if not all(math.isfinite(value) for value in numeric):
            raise FeatureValidationError(f"non-finite {timeframe} candle at {timestamp}")
        if min(open_price, high, low, close) <= 0:
            raise FeatureValidationError(f"non-positive {timeframe} OHLC at {timestamp}")
        if high < max(open_price, close, low) or low > min(open_price, close, high):
            raise FeatureValidationError(f"invalid {timeframe} OHLC range at {timestamp}")
        if volume < 0 or (
            quote_volume is not None
            and (not math.isfinite(quote_volume) or quote_volume < 0)
        ):
            raise FeatureValidationError(f"invalid {timeframe} volume at {timestamp}")
        if previous_timestamp is not None and timestamp - previous_timestamp != interval_ms:
            raise FeatureValidationError(
                f"{timeframe} closed-candle history has a gap at {timestamp}"
            )
        previous_timestamp = timestamp
    return window


def _timeframe_features(
    timeframe: str,
    candles: Sequence[Candle],
    direction: float,
) -> dict[str, float]:
    prefix = f"tf_{timeframe}"
    current = candles[-1]
    closes = [float(candle.close) for candle in candles]
    volumes = [float(candle.volume) for candle in candles]
    quote_volumes = [
        float(candle.quote_volume)
        if candle.quote_volume is not None
        else float(candle.close) * float(candle.volume)
        for candle in candles
    ]

    log_returns = [math.log(closes[index] / closes[index - 1]) for index in range(1, len(closes))]
    realised_volatility = math.sqrt(
        sum(value * value for value in log_returns[-20:]) / 20.0
    )

    true_ranges: list[float] = []
    for index in range(len(candles) - 14, len(candles)):
        candle = candles[index]
        previous_close = float(candles[index - 1].close)
        true_ranges.append(
            max(
                float(candle.high) - float(candle.low),
                abs(float(candle.high) - previous_close),
                abs(float(candle.low) - previous_close),
            )
        )
    atr_pct = _mean(true_ranges) / closes[-1]

    prior_closes = closes[-21:-1]
    prior_volume = volumes[-21:-1]
    prior_quote_volume = quote_volumes[-21:-1]
    high_low_range = float(current.high) - float(current.low)
    close_location = (
        (float(current.close) - float(current.low)) / high_low_range
        if high_low_range > 0
        else 0.5
    )
    side_close_location = close_location if direction > 0 else 1.0 - close_location

    return {
        f"{prefix}_side_return_1": direction * (closes[-1] / closes[-2] - 1.0),
        f"{prefix}_side_return_5": direction * (closes[-1] / closes[-6] - 1.0),
        f"{prefix}_realized_vol_20": realised_volatility,
        f"{prefix}_atr_pct_14": atr_pct,
        f"{prefix}_side_sma_gap_20": direction * (closes[-1] / _mean(prior_closes) - 1.0),
        f"{prefix}_range_pct": high_low_range / closes[-1],
        f"{prefix}_side_body_pct": direction
        * ((float(current.close) - float(current.open)) / float(current.open)),
        f"{prefix}_side_close_location": side_close_location,
        f"{prefix}_volume_ratio_20": _safe_ratio(volumes[-1], _mean(prior_volume)),
        f"{prefix}_quote_volume_ratio_20": _safe_ratio(
            quote_volumes[-1], _mean(prior_quote_volume)
        ),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 1.0 if numerator <= 0 else 2.0
    return numerator / denominator
