from __future__ import annotations

import argparse
import csv
from bisect import bisect_right
from dataclasses import dataclass, replace
from pathlib import Path

from .costs import CostConfig
from .models import Candle, Position, Signal
from .reporting import TradeRecord, TradeReporter
from .risk import RiskConfig, RiskManager
from .strategy import (
    MultiTimeframeStrategy,
    StrategyConfig,
    dynamic_stop_loss_pct,
    signal_position_size_multiplier,
    signal_stop_loss_overrides,
    signal_stop_timeframe,
    signal_trade_management_overrides,
)


_TIMEFRAME_MS = {
    "30s": 30_000,
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}


@dataclass(frozen=True)
class BacktestSummary:
    initial_equity: float
    final_equity: float
    trades: int
    wins: int
    max_drawdown_pct: float


def load_csv(path: Path) -> list[Candle]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        candles = []
        for row in rows:
            timestamp = int(float(row["timestamp"]))
            if timestamp < 10_000_000_000:
                timestamp *= 1000
            raw_quote_volume = row.get("quote_volume") or row.get("volume_quote")
            quote_volume = float(raw_quote_volume) if raw_quote_volume not in (None, "") else None
            candles.append(Candle(timestamp, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row.get("volume", 0)), quote_volume=quote_volume))
    return sorted(candles, key=lambda candle: candle.timestamp)


def run_backtest(
    data_dir: Path,
    *,
    initial_equity: float = 10_000,
    fee_pct: float = 0.0004,
    slippage_pct: float = 0.0002,
    candle_limit: int = 300,
    strategy: MultiTimeframeStrategy | None = None,
    risk: RiskManager | None = None,
    reporter: TradeReporter | None = None,
    use_fixed_take_profit: bool = False,
) -> BacktestSummary:
    strategy = strategy or MultiTimeframeStrategy(StrategyConfig())
    trigger_timeframe = strategy.config.trigger_timeframe
    required_timeframes = list(dict.fromkeys((trigger_timeframe, "1m", strategy.config.regime_timeframe)))
    bars: dict[str, list[Candle]] = {}
    for timeframe in required_timeframes:
        path = data_dir / f"{timeframe}.csv"
        if not path.exists() and timeframe == "30s":
            path = data_dir / "1m.csv"
        bars[timeframe] = load_csv(path)
    # Live evaluation always has a closed 1m execution stream.  Driving a
    # replay from a slower trigger timeframe skips the intervening 1m bars and
    # materially delays stop/trailing management, so 1m is the universal
    # replay clock even when signals come from 5m or 15m candles.
    driver_timeframe = "1m"
    driver = bars[driver_timeframe]
    if not driver:
        raise ValueError(f"{driver_timeframe}.csv contains no candles")
    timestamps_by_timeframe = {
        timeframe: [item.timestamp for item in series]
        for timeframe, series in bars.items()
    }
    risk = risk or RiskManager(RiskConfig(), costs=CostConfig(taker_fee_pct=fee_pct, slippage_pct=slippage_pct))
    equity = initial_equity
    peak_equity = equity
    max_drawdown = 0.0
    position: Position | None = None
    trades = 0
    wins = 0
    last_signal_timestamp = 0
    position_signal = None
    position_equity_before = 0.0
    # A loss cooldown is deterministic from trade timestamps.  Session-wide
    # loss halts depend on when the live process was restarted, so applying
    # them across an arbitrary multi-day history would invent a session
    # boundary and can permanently suppress the rest of the replay.
    cooldown_until_ms = 0
    consecutive_losses = 0
    # Production requests ``candle_limit`` rows and removes the newest forming
    # candle before evaluation.  Restrict every historical decision to that
    # same number of closed rows so long-period EMA initialization, transition
    # signals and position invalidation cannot diverge from the paper engine.
    closed_candle_limit = max(1, int(candle_limit) - 1)

    for index, candle in enumerate(driver):
        if index < 1:
            continue
        decision_timestamp = candle.timestamp + _TIMEFRAME_MS[driver_timeframe]
        next_execution_candle = driver[index + 1] if index + 1 < len(driver) else None
        next_execution_open = next_execution_candle.open if next_execution_candle is not None else None
        next_execution_timestamp = (
            next_execution_candle.timestamp if next_execution_candle is not None else None
        )
        candles_by_timeframe: dict[str, list[Candle]] = {}
        for timeframe, series in bars.items():
            latest_closed_open = decision_timestamp - _TIMEFRAME_MS[timeframe]
            end = bisect_right(timestamps_by_timeframe[timeframe], latest_closed_open)
            start = max(0, end - closed_candle_limit)
            candles_by_timeframe[timeframe] = series[start:end]
        if not candles_by_timeframe.get(trigger_timeframe) or not candles_by_timeframe.get("1m"):
            continue
        signal = strategy.evaluate(candles_by_timeframe)

        if position is not None:
            exit_candle = (candles_by_timeframe.get("1m") or [candle])[-1]
            if position.side == "long":
                position = replace(
                    position,
                    best_price=max(position.best_price or position.entry_price, exit_candle.high),
                    worst_price=min(position.worst_price or position.entry_price, exit_candle.low),
                )
            else:
                position = replace(
                    position,
                    best_price=min(position.best_price or position.entry_price, exit_candle.low),
                    worst_price=max(position.worst_price or position.entry_price, exit_candle.high),
                )
            exit_price: float | None = None
            if position.side == "long":
                if exit_candle.low <= position.stop_price:
                    # A market stop cannot fill at the stop price after the
                    # candle has already opened below it.
                    exit_price = min(position.stop_price, exit_candle.open)
                    exit_reason = _stop_exit_reason(position)
                elif use_fixed_take_profit and exit_candle.high >= position.take_profit_price:
                    exit_price = position.take_profit_price
                    exit_reason = "take_profit"
            else:
                if exit_candle.high >= position.stop_price:
                    exit_price = max(position.stop_price, exit_candle.open)
                    exit_reason = _stop_exit_reason(position)
                elif use_fixed_take_profit and exit_candle.low <= position.take_profit_price:
                    exit_price = position.take_profit_price
                    exit_reason = "take_profit"
            time_exit_enabled = bool(getattr(strategy.config, "enable_time_exit", False))
            soft_max_hold_seconds = max(0, int(strategy.config.max_hold_seconds))
            if exit_price is None and time_exit_enabled and soft_max_hold_seconds:
                hard_max_hold_seconds = max(
                    soft_max_hold_seconds,
                    int(getattr(strategy.config, "hard_max_hold_seconds", soft_max_hold_seconds)),
                )
                held_seconds = (exit_candle.timestamp - position.opened_at) / 1000
                net_at_close = risk.estimate_net_pnl(
                    position.side,
                    position.entry_price,
                    exit_candle.close,
                    position.quantity,
                    holding_hours=max(0.0, held_seconds / 3600),
                )
                initial_stop = position.initial_stop_price or position.stop_price
                stop_distance = abs(position.entry_price - initial_stop)
                favorable_distance = (
                    exit_candle.close - position.entry_price
                    if position.side == "long"
                    else position.entry_price - exit_candle.close
                )
                current_r = favorable_distance / stop_distance if stop_distance > 0 else 0.0
                min_time_exit_r = max(0.0, float(getattr(strategy.config, "time_exit_min_r", 0.5)))
                profitable_time_exit = net_at_close > 0 and current_r >= min_time_exit_r
                if held_seconds >= soft_max_hold_seconds and (profitable_time_exit or held_seconds >= hard_max_hold_seconds):
                    exit_price = exit_candle.close
                    exit_reason = "time_exit" if profitable_time_exit else "hard_time_exit"
            if exit_price is None and _profit_trend_exit_ready(position, strategy, candles_by_timeframe):
                exit_price = exit_candle.close
                exit_reason = "trend_invalidation"
            if exit_price is not None:
                holding_hours = max(0.0, (exit_candle.timestamp - position.opened_at) / 3_600_000)
                pnl = risk.estimate_net_pnl(
                    position.side,
                    position.entry_price,
                    exit_price,
                    position.quantity,
                    holding_hours=holding_hours,
                )
                if reporter is not None:
                    reporter.record_trade(
                        TradeRecord.from_position(
                            exchange="backtest",
                            symbol="BTC-USDT",
                            mode="backtest",
                            position=position,
                            exit_price=exit_price,
                            exit_time_ms=exit_candle.timestamp,
                            exit_reason=exit_reason,
                            costs=risk.costs,
                            equity_before=position_equity_before,
                            signal=position_signal,
                        )
                    )
                equity += pnl
                trades += 1
                wins += int(pnl > 0)
                if pnl < 0:
                    consecutive_losses += 1
                    cooldown_minutes = max(0, int(risk.config.cooldown_minutes))
                    threshold = int(risk.config.max_consecutive_losses)
                    if threshold > 0 and consecutive_losses >= threshold:
                        cooldown_minutes = max(
                            cooldown_minutes,
                            max(0, int(getattr(risk.config, "loss_streak_pause_minutes", 0))),
                        )
                    cooldown_until_ms = decision_timestamp + cooldown_minutes * 60_000
                else:
                    consecutive_losses = 0
                position = None
                peak_equity = max(peak_equity, equity)
                max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
            else:
                position = _tighten_position_stop(
                    position,
                    exit_candle,
                    strategy,
                    risk,
                    position_signal,
                )

        # Production evaluates the newly closed candles after position
        # management and may close (then reverse) when a fresh, sufficiently
        # strong opposite signal appears.  Keep that path in the replay too;
        # otherwise backtests can hold through a live ``opposite_signal`` exit
        # and materially misstate both the trade result and later entries.
        if (
            position is not None
            and signal.side != "flat"
            and signal.side != position.side
            and signal.timestamp != last_signal_timestamp
        ):
            execution_timestamp = next_execution_timestamp or decision_timestamp
            held_seconds = max(0.0, (execution_timestamp - position.opened_at) / 1000)
            minimum_hold = max(0, int(getattr(strategy.config, "min_hold_seconds", 60)))
            if held_seconds >= minimum_hold and next_execution_open is not None:
                net_exit = risk.estimate_net_pnl(
                    position.side,
                    position.entry_price,
                    next_execution_open,
                    position.quantity,
                    holding_hours=held_seconds / 3600,
                )
                required_score = max(1, int(getattr(strategy.config, "reversal_min_score", 5)))
                should_reverse = net_exit > 0 or signal.score >= required_score
                if should_reverse:
                    pnl = risk.estimate_net_pnl(
                        position.side,
                        position.entry_price,
                        next_execution_open,
                        position.quantity,
                        holding_hours=held_seconds / 3600,
                    )
                    if reporter is not None:
                        reporter.record_trade(
                            TradeRecord.from_position(
                                exchange="backtest",
                                symbol="BTC-USDT",
                                mode="backtest",
                                position=position,
                                exit_price=next_execution_open,
                                exit_time_ms=execution_timestamp,
                                exit_reason="opposite_signal",
                                costs=risk.costs,
                                equity_before=position_equity_before,
                                signal=position_signal,
                            )
                        )
                    equity += pnl
                    trades += 1
                    wins += int(pnl > 0)
                    if pnl < 0:
                        consecutive_losses += 1
                        cooldown_minutes = max(0, int(risk.config.cooldown_minutes))
                        threshold = int(risk.config.max_consecutive_losses)
                        if threshold > 0 and consecutive_losses >= threshold:
                            cooldown_minutes = max(
                                cooldown_minutes,
                                max(0, int(getattr(risk.config, "loss_streak_pause_minutes", 0))),
                            )
                        cooldown_until_ms = decision_timestamp + cooldown_minutes * 60_000
                    else:
                        consecutive_losses = 0
                    position = None
                    peak_equity = max(peak_equity, equity)
                    max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)

        if position is None and signal.side != "flat" and signal.timestamp != last_signal_timestamp:
            last_signal_timestamp = signal.timestamp
            if next_execution_open is None:
                continue
            loss_streak_pause = max(0, int(getattr(risk.config, "loss_streak_pause_minutes", 0)))
            threshold = int(risk.config.max_consecutive_losses)
            if threshold > 0 and consecutive_losses >= threshold and loss_streak_pause:
                if decision_timestamp < cooldown_until_ms:
                    continue
                consecutive_losses = 0
            if decision_timestamp < cooldown_until_ms:
                continue
            trigger_candles = candles_by_timeframe[trigger_timeframe]
            stop_candles = candles_by_timeframe.get(
                signal_stop_timeframe(signal, strategy.config),
                trigger_candles,
            )
            # The signal is known only after the latest candle closes.  A
            # market order is therefore modeled at the next 1m open rather
            # than retroactively at the signal candle's close.
            entry_price = next_execution_open
            stop_loss_pct = dynamic_stop_loss_pct(
                stop_candles,
                strategy.config,
                risk.config.stop_loss_pct,
                side=signal.side,
                entry_price=entry_price,
                **signal_stop_loss_overrides(signal, strategy.config),
            )
            protection = risk.protection(
                signal.side,
                equity,
                entry_price,
                strategy.config.take_profit_r,
                stop_loss_pct=stop_loss_pct,
                size_multiplier=signal_position_size_multiplier(signal),
            )
            if not risk.is_cost_effective(signal.side, entry_price, protection.take_profit_price, protection.quantity):
                continue
            position = Position(
                signal.side,
                protection.quantity,
                entry_price,
                protection.stop_price,
                protection.take_profit_price,
                next_execution_timestamp or decision_timestamp,
                initial_stop_price=protection.stop_price,
                best_price=entry_price,
                worst_price=entry_price,
            )
            position_signal = signal
            position_equity_before = equity

    return BacktestSummary(initial_equity, equity, trades, wins, max_drawdown * 100)


def _stop_is_protected(position: Position) -> bool:
    if position.initial_stop_price is None:
        return False
    if position.side == "long":
        return position.stop_price > position.initial_stop_price
    return position.stop_price < position.initial_stop_price


def _stop_exit_reason(position: Position) -> str:
    if not _stop_is_protected(position):
        return "stop_loss"
    if position.stop_reason in {"break_even_stop", "trailing_stop"}:
        return position.stop_reason
    return "trailing_stop"


def _profit_trend_exit_ready(
    position: Position,
    strategy: MultiTimeframeStrategy,
    candles_by_timeframe: dict[str, list[Candle]],
) -> bool:
    if not bool(getattr(strategy.config, "enable_profit_trend_exit", False)):
        return False
    initial_stop = position.initial_stop_price or position.stop_price
    risk_distance = abs(position.entry_price - initial_stop)
    if risk_distance <= 0:
        return False
    if position.side == "long":
        favorable_r = ((position.best_price or position.entry_price) - position.entry_price) / risk_distance
    else:
        favorable_r = (position.entry_price - (position.best_price or position.entry_price)) / risk_distance
    trigger_r = max(0.0, float(getattr(strategy.config, "profit_trend_exit_trigger_r", 1.0)))
    return favorable_r >= trigger_r and strategy.position_trend_invalidated(position.side, candles_by_timeframe)


def _tighten_position_stop(
    position: Position,
    candle: Candle,
    strategy: MultiTimeframeStrategy,
    risk: RiskManager,
    signal: Signal | None = None,
) -> Position:
    initial_stop = position.initial_stop_price or position.stop_price
    risk_distance = abs(position.entry_price - initial_stop)
    if risk_distance <= 0:
        return position
    if position.side == "long":
        best_price = max(position.best_price or position.entry_price, candle.high)
        worst_price = min(position.worst_price or position.entry_price, candle.low)
        favorable_r = (best_price - position.entry_price) / risk_distance
    else:
        best_price = min(position.best_price or position.entry_price, candle.low)
        worst_price = max(position.worst_price or position.entry_price, candle.high)
        favorable_r = (position.entry_price - best_price) / risk_distance
    candidate = position.stop_price
    candidate_reason = position.stop_reason
    management = signal_trade_management_overrides(signal, strategy.config)
    break_even_trigger = max(
        0.0,
        float(
            management.get(
                "break_even_trigger_r",
                getattr(strategy.config, "break_even_trigger_r", 1.25),
            )
        ),
    )
    if break_even_trigger and favorable_r >= break_even_trigger:
        holding_hours = max(0.0, (candle.timestamp - position.opened_at) / 3_600_000)
        cost_break_even = risk.break_even_price(
            position.side,
            position.entry_price,
            holding_hours=holding_hours,
        )
        lock_distance = risk_distance * max(
            0.0,
            float(
                management.get(
                    "break_even_lock_r",
                    getattr(strategy.config, "break_even_lock_r", 0.5),
                )
            ),
        )
        protected_stop = cost_break_even + lock_distance if position.side == "long" else cost_break_even - lock_distance
        protection_improved = protected_stop > candidate if position.side == "long" else protected_stop < candidate
        if protection_improved:
            candidate = protected_stop
            candidate_reason = "break_even_stop"
    trailing_trigger = max(
        0.0,
        float(
            management.get(
                "trailing_trigger_r",
                getattr(strategy.config, "trailing_trigger_r", 2.0),
            )
        ),
    )
    if trailing_trigger and favorable_r >= trailing_trigger:
        trailing_distance = risk_distance * max(
            0.1,
            float(
                management.get(
                    "trailing_distance_r",
                    getattr(strategy.config, "trailing_distance_r", 0.75),
                )
            ),
        )
        trailing_stop = best_price - trailing_distance if position.side == "long" else best_price + trailing_distance
        trailing_improved = trailing_stop > candidate if position.side == "long" else trailing_stop < candidate
        if trailing_improved:
            candidate = trailing_stop
            candidate_reason = "trailing_stop"
    candidate = min(candidate, candle.close) if position.side == "long" else max(candidate, candle.close)
    stop_improved = candidate > position.stop_price if position.side == "long" else candidate < position.stop_price
    return replace(
        position,
        stop_price=candidate if stop_improved else position.stop_price,
        initial_stop_price=initial_stop,
        best_price=best_price,
        stop_reason=candidate_reason if stop_improved else position.stop_reason,
        worst_price=worst_price,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the BTC multi-timeframe strategy")
    parser.add_argument("data_dir", type=Path, help="directory containing 1m.csv, 5m.csv, 1h.csv and 4h.csv")
    parser.add_argument("--report-dir", type=Path, help="also write trade_report.csv and daily/monthly summaries")
    parser.add_argument("--candle-limit", type=int, default=300, help="match the production fetch limit (default: 300)")
    parser.add_argument(
        "--fixed-take-profit",
        action="store_true",
        help="model the legacy paper fixed take-profit; default mirrors live dynamic exits",
    )
    args = parser.parse_args()
    reporter = TradeReporter(args.report_dir) if args.report_dir else None
    try:
        print(
            run_backtest(
                args.data_dir,
                candle_limit=args.candle_limit,
                reporter=reporter,
                use_fixed_take_profit=args.fixed_take_profit,
            )
        )
    finally:
        if reporter is not None:
            reporter.close()


if __name__ == "__main__":
    main()
