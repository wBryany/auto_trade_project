from __future__ import annotations

from math import sqrt
from typing import Sequence


def _empty(length: int) -> list[float | None]:
    return [None] * length


def ema(values: Sequence[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")
    result = _empty(len(values))
    if len(values) < period:
        return result
    current = sum(values[:period]) / period
    result[period - 1] = current
    alpha = 2.0 / (period + 1)
    for index in range(period, len(values)):
        current = (values[index] - current) * alpha + current
        result[index] = current
    return result


def sma(values: Sequence[float], period: int) -> list[float | None]:
    result = _empty(len(values))
    if period <= 0 or len(values) < period:
        return result
    rolling = sum(values[:period])
    result[period - 1] = rolling / period
    for index in range(period, len(values)):
        rolling += values[index] - values[index - period]
        result[index] = rolling / period
    return result


def rsi(values: Sequence[float], period: int = 14) -> list[float | None]:
    result = _empty(len(values))
    if period <= 0 or len(values) <= period:
        return result
    gains = []
    losses = []
    for index in range(1, len(values)):
        delta = values[index] - values[index - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period

    def to_rsi(gain: float, loss: float) -> float:
        if loss == 0:
            return 100.0 if gain > 0 else 50.0
        relative_strength = gain / loss
        return 100.0 - (100.0 / (1.0 + relative_strength))

    result[period] = to_rsi(average_gain, average_loss)
    for index in range(period + 1, len(values)):
        gain = gains[index - 1]
        loss = losses[index - 1]
        average_gain = ((average_gain * (period - 1)) + gain) / period
        average_loss = ((average_loss * (period - 1)) + loss) / period
        result[index] = to_rsi(average_gain, average_loss)
    return result


def true_range(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> list[float]:
    result: list[float] = []
    for index, (high, low) in enumerate(zip(highs, lows)):
        if index == 0:
            result.append(high - low)
        else:
            result.append(max(high - low, abs(high - closes[index - 1]), abs(low - closes[index - 1])))
    return result


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> list[float | None]:
    tr = true_range(highs, lows, closes)
    result = _empty(len(tr))
    if period <= 0 or len(tr) < period:
        return result
    current = sum(tr[:period]) / period
    result[period - 1] = current
    for index in range(period, len(tr)):
        current = ((current * (period - 1)) + tr[index]) / period
        result[index] = current
    return result


def macd(
    values: Sequence[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    fast = ema(values, fast_period)
    slow = ema(values, slow_period)
    line: list[float | None] = [
        (fast_value - slow_value) if fast_value is not None and slow_value is not None else None
        for fast_value, slow_value in zip(fast, slow)
    ]
    compact = [value for value in line if value is not None]
    compact_signal = ema(compact, signal_period)
    signal: list[float | None] = [None] * len(line)
    compact_index = 0
    for index, value in enumerate(line):
        if value is not None:
            signal[index] = compact_signal[compact_index]
            compact_index += 1
    histogram = [
        (line_value - signal_value) if line_value is not None and signal_value is not None else None
        for line_value, signal_value in zip(line, signal)
    ]
    return line, signal, histogram


def bollinger(values: Sequence[float], period: int = 20, deviations: float = 2.0) -> tuple[list[float | None], list[float | None], list[float | None]]:
    middle = sma(values, period)
    upper: list[float | None] = [None] * len(values)
    lower: list[float | None] = [None] * len(values)
    for index in range(period - 1, len(values)):
        if middle[index] is None:
            continue
        window = values[index - period + 1 : index + 1]
        mean = middle[index]
        standard_deviation = sqrt(sum((value - mean) ** 2 for value in window) / period)
        upper[index] = mean + deviations * standard_deviation
        lower[index] = mean - deviations * standard_deviation
    return middle, upper, lower
