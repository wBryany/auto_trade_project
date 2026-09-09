"""Closed-history activity checks for minute-scalp cost admission.

Historical excursions are an activity proxy, not a forecast of a
trade's profit, a take-profit hit probability, or a statistical edge claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from statistics import median
from typing import Sequence

from .costs import CostConfig
from .models import Candle


@dataclass(frozen=True)
class ScalpCostCoverage:
    allowed: bool
    reason: str
    side: str
    horizon_bars: int
    lookback_windows: int
    observed_move_pct: float | None = None
    latest_move_pct: float | None = None
    required_move_pct: float | None = None
    window_move_pcts: tuple[float, ...] = ()


def evaluate_scalp_cost_coverage(
    candles: Sequence[Candle],
    side: str,
    horizon_seconds: float,
    costs: CostConfig,
    lookback_windows: int = 6,
) -> ScalpCostCoverage:
    """Require typical AND recent favorable activity to cover net costs.

    The caller must supply closed one-minute bars only. The last ``windows *
    ceil(horizon_seconds / 60)`` candles are partitioned without overlap. Each
    excursion starts at that window's first OPEN, rather than at a hindsight
    low/high, and is divided by that open. Only already completed windows are
    used; the median and the newest window must both cover the price change
    needed to earn ``min_net_edge_pct`` after the configured fees/slippage.

    Funding follows CostConfig's existing whole-eight-hour interval estimate
    using this holding horizon. That approximation does not model proximity
    to the next exchange funding settlement, even for a ten-minute hold.
    """
    horizon_bars = 0
    required_move: float | None = None

    def blocked(reason: str) -> ScalpCostCoverage:
        return ScalpCostCoverage(
            False, reason, side, horizon_bars, lookback_windows,
            required_move_pct=required_move,
        )

    if side not in {"long", "short"}:
        return blocked("invalid_side")
    if isinstance(lookback_windows, bool) or not isinstance(lookback_windows, int) or not 3 <= lookback_windows <= 12:
        return blocked("invalid_lookback_windows")
    try:
        if not isfinite(horizon_seconds) or horizon_seconds <= 0:
            return blocked("invalid_horizon")
        horizon_bars = ceil(horizon_seconds / 60.0)
        rates = (costs.fee_pct, costs.slippage_pct, costs.funding_rate_pct_per_8h, costs.min_net_edge_pct)
        if any(not isfinite(rate) or rate < 0 for rate in rates):
            return blocked("invalid_costs")
        rate = costs.fee_pct + costs.slippage_pct
        if not isfinite(rate) or rate >= 1.0:
            return blocked("invalid_costs")
        funding = costs.funding_rate_pct_per_8h * costs.funding_intervals_for(horizon_seconds / 3600.0)
        # Exit fees/slippage depend on the EXIT notional. Inverting the same
        # CostConfig net-PnL equation gives a slightly different hurdle for
        # each side and includes the minimum net edge in the inversion.
        denominator = 1.0 - rate if side == "long" else 1.0 + rate
        required_move = (2.0 * rate + funding + costs.min_net_edge_pct) / denominator
        if not isfinite(required_move) or required_move < 0 or (side == "short" and required_move >= 1.0):
            return blocked("invalid_costs")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return blocked("invalid_costs_or_horizon")

    required_bars = horizon_bars * lookback_windows
    if len(candles) < required_bars:
        return blocked("insufficient_history")
    history = candles[-required_bars:]
    previous_timestamp: int | float | None = None
    for candle in history:
        try:
            timestamp = candle.timestamp
            if isinstance(timestamp, bool) or not isfinite(timestamp) or timestamp < 0 or int(timestamp) != timestamp:
                return blocked("invalid_timestamp")
            if previous_timestamp is not None and timestamp - previous_timestamp != 60_000:
                return blocked("discontinuous_history")
            prices = (candle.open, candle.high, candle.low, candle.close)
            if any(not isfinite(price) or price <= 0 for price in prices):
                return blocked("invalid_ohlc")
            if not candle.low <= min(candle.open, candle.close) <= max(candle.open, candle.close) <= candle.high:
                return blocked("invalid_ohlc")
            if not isfinite(candle.volume) or candle.volume < 0:
                return blocked("invalid_volume")
            if candle.quote_volume is not None and (not isfinite(candle.quote_volume) or candle.quote_volume < 0):
                return blocked("invalid_volume")
        except (AttributeError, TypeError, ValueError, OverflowError):
            return blocked("invalid_candle")
        previous_timestamp = timestamp

    excursions: list[float] = []
    for start in range(0, required_bars, horizon_bars):
        window = history[start:start + horizon_bars]
        entry_open = window[0].open
        favorable_move = (
            max(candle.high for candle in window) - entry_open
            if side == "long"
            else entry_open - min(candle.low for candle in window)
        ) / entry_open
        if not isfinite(favorable_move) or favorable_move < 0:
            return blocked("invalid_excursion")
        excursions.append(favorable_move)
    observed_move = float(median(excursions))
    latest_move = excursions[-1]
    allowed = observed_move >= required_move and latest_move >= required_move
    reason = "cost_coverage" if allowed else "median_below_cost_hurdle" if observed_move < required_move else "latest_below_cost_hurdle"
    return ScalpCostCoverage(
        allowed, reason, side, horizon_bars, lookback_windows,
        observed_move, latest_move, required_move, tuple(excursions),
    )
