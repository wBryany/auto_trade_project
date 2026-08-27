from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .exchanges.base import ExchangeAdapter
from .indicators import atr
from .macro_risk import MacroRiskController, MacroRiskDecision
from .models import Candle, OrderRequest, Position, Signal, TradeResult
from .notifications import EmailNotifier
from .reporting import TradeRecord, TradeReporter
from .risk import RiskManager
from .strategy import MultiTimeframeStrategy, signal_position_size_multiplier

LOG = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    mode: str = "paper"
    poll_seconds: int = 15
    paper_equity: float = 10000.0
    candle_limit: int = 300
    take_profit_r: float = 2.5
    reconciliation_state_path: str = ""


class TradingEngine:
    def __init__(
        self,
        adapter: ExchangeAdapter,
        strategy: MultiTimeframeStrategy,
        risk: RiskManager,
        config: EngineConfig,
        reporter: TradeReporter | None = None,
        macro_risk: MacroRiskController | None = None,
        notifier: EmailNotifier | None = None,
    ) -> None:
        self.adapter = adapter
        self.strategy = strategy
        self.risk = risk
        self.config = config
        self.position: Position | None = None
        self.last_signal_timestamp = 0
        self.session_pnl = 0.0
        self.consecutive_losses = 0
        self.cooldown_until = 0.0
        self.reporter = reporter
        self._restore_recent_loss_streak()
        self.position_signal: Signal | None = None
        self.position_equity_before = 0.0
        self.last_position_candle_timestamp = 0
        self.macro_risk = macro_risk
        self.pending_macro_signal: Signal | None = None
        self.notifier = notifier
        self.live_preflight: dict[str, Any] = {}
        self.unmanaged_live_position = self._load_unmanaged_live_position()

    def prepare_live(self) -> dict[str, Any]:
        if self.config.mode != "live":
            self.live_preflight = {"prepared": False, "mode": self.config.mode}
            return self.live_preflight
        self.live_preflight = self.adapter.prepare_live(max_leverage=self.risk.max_leverage)
        return self.live_preflight

    def _restore_recent_loss_streak(self) -> None:
        """Restore a timed loss-streak guard from completed trade history.

        A process restart must not silently clear a configured multi-loss
        circuit breaker. Restoration is opt-in through a positive loss
        threshold and ``loss_streak_pause_minutes``. A threshold of zero
        disables the loss-streak guard entirely.
        """
        threshold = int(self.risk.config.max_consecutive_losses)
        pause_minutes = max(0, int(getattr(self.risk.config, "loss_streak_pause_minutes", 0)))
        if self.reporter is None or threshold <= 0 or pause_minutes <= 0:
            return
        environment = str(getattr(getattr(self.adapter, "settings", None), "environment", "testnet"))
        scope = "production" if self.config.mode == "live" and environment == "production" else "testnet"
        try:
            rows = self.reporter.query_trades(
                exchange=self.adapter.name,
                scope=scope,
                limit=max(100, threshold + 1),
            )
        except Exception:
            LOG.exception("failed to restore recent loss streak")
            return
        if not rows:
            return

        consecutive = 0
        for row in rows:
            if float(row.get("net_pnl") or 0.0) >= 0:
                break
            consecutive += 1
        if consecutive <= 0:
            return

        latest_exit = str(rows[0].get("exit_time") or "")
        try:
            latest_exit_at = datetime.fromisoformat(latest_exit.replace("Z", "+00:00"))
            if latest_exit_at.tzinfo is None:
                latest_exit_at = latest_exit_at.replace(tzinfo=timezone.utc)
            latest_exit_epoch = latest_exit_at.timestamp()
        except ValueError:
            LOG.warning("cannot parse latest trade exit time while restoring risk state: %s", latest_exit)
            return

        self.consecutive_losses = consecutive
        normal_cooldown = max(0, int(self.risk.config.cooldown_minutes)) * 60
        selected_cooldown = normal_cooldown
        if consecutive >= threshold:
            selected_cooldown = max(selected_cooldown, pause_minutes * 60)
        self.cooldown_until = latest_exit_epoch + selected_cooldown
        LOG.warning(
            "restored loss streak=%s cooldown_until=%s scope=%s",
            self.consecutive_losses,
            datetime.fromtimestamp(self.cooldown_until, tz=timezone.utc).isoformat(),
            scope,
        )

    def evaluate_once(self) -> TradeResult:
        if self.notifier is not None and self.reporter is not None:
            try:
                self.notifier.maybe_send_daily_report(
                    self.reporter,
                    exchange=self.adapter.name,
                    symbol=self.adapter.settings.symbol,
                    mode=self.config.mode,
                )
            except Exception:  # Reporting must not interrupt market evaluation.
                LOG.exception("daily email report scheduling failed")
        candles_by_timeframe: dict[str, list[Any]] = {}
        trigger_timeframe = self.strategy.config.trigger_timeframe
        requested_timeframes = list(dict.fromkeys((trigger_timeframe, "1m", self.strategy.config.regime_timeframe)))
        for timeframe in requested_timeframes:
            exchange_timeframe = "1m" if timeframe == "30s" and self.adapter.name != "okx" else timeframe
            candles = self.adapter.fetch_candles(exchange_timeframe, self.config.candle_limit)
            # The newest candle may still be forming. Exclude it so a signal is based on closed bars.
            candles_by_timeframe[timeframe] = candles[:-1] if len(candles) > 1 else candles
        if self.config.mode == "live" and self.adapter.name == "binance":
            reconciliation_candles = candles_by_timeframe.get("1m") or candles_by_timeframe.get(trigger_timeframe, [])
            if reconciliation_candles:
                self._reconcile_binance_live_position(reconciliation_candles[-1])
        macro_decision = (
            self.macro_risk.decision(candles_by_timeframe)
            if self.macro_risk is not None
            else MacroRiskDecision()
        )
        if self.config.mode == "paper":
            exit_candles = candles_by_timeframe.get("1m") or candles_by_timeframe.get(trigger_timeframe, [])
            if exit_candles and self.position is not None:
                exit_candle = exit_candles[-1]
                # Polling is faster than the execution timeframe. Process each
                # closed candle's OHLC exactly once, otherwise a stop tightened
                # from that candle's high can be falsely hit by the same
                # candle's earlier low on the next poll.
                if exit_candle.timestamp > self.last_position_candle_timestamp:
                    self.last_position_candle_timestamp = exit_candle.timestamp
                    self._manage_paper_position(exit_candle, candles_by_timeframe)
                # Scheduled-event management is time-sensitive, so it may run
                # between new candles. It runs after OHLC processing to ensure
                # a newly tightened macro stop cannot look back into this bar.
                if self.position is not None:
                    self._manage_macro_position(exit_candle, macro_decision)
        signal = self.strategy.evaluate(candles_by_timeframe)
        if not macro_decision.blocked and signal.side == "flat" and self.pending_macro_signal is not None:
            if self._pending_macro_signal_expired(candles_by_timeframe):
                self.pending_macro_signal = None
            else:
                reevaluate = getattr(self.strategy, "reevaluate_blocked_signal", None)
                if callable(reevaluate):
                    signal = reevaluate(self.pending_macro_signal.side, candles_by_timeframe)
                else:
                    self.pending_macro_signal = None
        elif (
            not macro_decision.blocked
            and signal.side != "flat"
            and self.pending_macro_signal is not None
            and signal.side != self.pending_macro_signal.side
        ):
            self.pending_macro_signal = None
        LOG.info("%s signal=%s score=%s reasons=%s", self.adapter.name, signal.side, signal.score, ",".join(signal.reasons))
        live_mark_price: float | None = None
        if self.config.mode == "live" and self.position is not None:
            # The exchange-side stop and take-profit remain the outage-safe
            # fallback.  While the engine is online, every polling cycle also
            # evaluates the current venue mark price for a protected trailing
            # stop, profitable trend invalidation, or configured time exit.
            live_mark_price = self.adapter.fetch_mark_price()
            active_exit = self._manage_live_position(
                live_mark_price,
                candles_by_timeframe,
                signal,
            )
            if active_exit is not None:
                return active_exit
        if signal.side == "flat" or signal.timestamp == self.last_signal_timestamp:
            return TradeResult(self.adapter.name, "no_action", signal=signal, position=self.position)

        execution_candles = candles_by_timeframe.get("1m") or candles_by_timeframe[trigger_timeframe]
        current_price = live_mark_price or execution_candles[-1].close
        if self.position is not None:
            if self.position.side == signal.side:
                return TradeResult(self.adapter.name, "position_held", signal=signal, position=self.position)
            if not self._should_reverse(self.position, current_price, signal):
                return TradeResult(self.adapter.name, "position_held_for_costs", signal=signal, position=self.position)
            if self.config.mode == "live":
                self._close_live_position(current_price, "opposite_signal")
            else:
                self._close_paper_position(current_price, "opposite_signal")
            self.position = None

        if macro_decision.blocked:
            self.pending_macro_signal = signal
            blocked_signal = Signal(
                signal.side,
                signal.score,
                signal.timestamp,
                tuple(signal.reasons) + (f"entry_blocked={macro_decision.reason}",),
            )
            return TradeResult(
                self.adapter.name,
                "macro_event_blocked",
                signal=blocked_signal,
                position=self.position,
                raw={"macro_risk": self.macro_risk.status() if self.macro_risk else {}},
            )

        # A macro-blocked closed-bar signal remains eligible for re-evaluation
        # when the blackout ends. Other entry gates still consume it once.
        self.pending_macro_signal = None
        self.last_signal_timestamp = signal.timestamp
        if not self._entry_allowed():
            return TradeResult(self.adapter.name, "risk_blocked", signal=signal, position=self.position)

        equity = self.config.paper_equity if self.config.mode != "live" else self.adapter.fetch_equity()
        dynamic_stop_pct = self._dynamic_stop_loss_pct(candles_by_timeframe[trigger_timeframe])
        protection = self.risk.protection(
            signal.side,
            equity,
            current_price,
            self.config.take_profit_r,
            stop_loss_pct=dynamic_stop_pct,
            size_multiplier=signal_position_size_multiplier(signal),
        )
        if not self.risk.is_cost_effective(signal.side, current_price, protection.take_profit_price, protection.quantity):
            return TradeResult(self.adapter.name, "cost_blocked", signal=signal, position=self.position)
        if self.config.mode == "live":
            requested_quantity, sizing_raw = self._select_live_entry_quantity(
                protection.quantity,
                current_price,
                signal,
            )
            if requested_quantity is None:
                return TradeResult(
                    self.adapter.name,
                    "minimum_order_unavailable",
                    signal=signal,
                    position=self.position,
                    raw=sizing_raw,
                )
            if not self.risk.is_cost_effective(
                signal.side,
                current_price,
                protection.take_profit_price,
                requested_quantity,
            ):
                return TradeResult(self.adapter.name, "cost_blocked", signal=signal, position=self.position)
            entry_client_id = self._client_id("entry")
            try:
                entry_payload = self.adapter.place_market_order(
                    OrderRequest(
                        "buy" if signal.side == "long" else "sell",
                        requested_quantity,
                        False,
                        entry_client_id,
                    )
                )
            except Exception as entry_error:
                if self._is_order_capacity_error(entry_error):
                    self._notify_order_unavailable(
                        signal,
                        current_price,
                        sizing_raw,
                        f"交易所拒绝最小订单：{entry_error}",
                    )
                    return TradeResult(
                        self.adapter.name,
                        "minimum_order_unavailable",
                        signal=signal,
                        position=self.position,
                        raw={**sizing_raw, "error": str(entry_error)},
                    )
                raise
            filled_quantity, filled_price = self.adapter.market_fill(
                entry_payload,
                fallback_price=current_price,
            )
            stop_distance = filled_price * dynamic_stop_pct
            stop_price = filled_price - stop_distance if signal.side == "long" else filled_price + stop_distance
            take_profit_price = (
                filled_price + stop_distance * self.config.take_profit_r
                if signal.side == "long"
                else filled_price - stop_distance * self.config.take_profit_r
            )
            stop_client_id = self._client_id("stop")
            # Binance keeps only the hard stop on the exchange. Profit exits
            # are decided online from trend invalidation, break-even/trailing
            # protection and the configured time-exit rules.
            take_profit_client_id = ""
            position = Position(
                signal.side,
                filled_quantity,
                filled_price,
                stop_price,
                take_profit_price,
                int(time.time() * 1000),
                initial_stop_price=stop_price,
                best_price=filled_price,
                worst_price=filled_price,
                entry_order_id=str(entry_payload.get("orderId") or ""),
                entry_client_id=entry_client_id,
                stop_client_id=stop_client_id,
                take_profit_client_id=take_profit_client_id,
            )
            # Persist local exposure immediately after a confirmed fill. If
            # protection fails, the emergency-close path can never lose track
            # of a real position.
            self.position = position
            self.position_signal = signal
            self.position_equity_before = equity
            try:
                protection_payload = self.adapter.place_protection_orders(
                    position,
                    stop_client_id=stop_client_id,
                    take_profit_client_id=take_profit_client_id,
                )
                stop_payload = protection_payload.get("stop") or {}
                if not stop_payload:
                    raise RuntimeError("exchange did not confirm a stop order")
                position = replace(
                    position,
                    stop_order_id=str(stop_payload.get("algoId") or stop_payload.get("orderId") or ""),
                    take_profit_order_id="",
                )
                self.position = position
                status = "live_order_protected"
                live_raw = {
                    "entry_order_id": position.entry_order_id,
                    "stop_order_id": position.stop_order_id,
                    "take_profit_order_id": "",
                    "server_side_stop_confirmed": bool(protection_payload.get("confirmed")),
                    "server_side_take_profit_enabled": False,
                    "profit_exit_mode": "dynamic_trend_and_trailing",
                    "sizing": sizing_raw,
                }
            except Exception as protection_error:
                rollback_error = ""
                try:
                    rollback_payload = self.adapter.emergency_close(
                        position,
                        client_id=self._client_id("emergency-close"),
                    )
                    rollback_quantity, rollback_price = self.adapter.market_fill(
                        rollback_payload,
                        fallback_price=filled_price,
                    )
                    if rollback_quantity + 1e-12 < position.quantity:
                        raise RuntimeError("emergency close was only partially filled")
                    self._confirm_binance_flat("protection-failure emergency close")
                    self.adapter.cancel_protection_orders(position)
                    self._close_paper_position(rollback_price, "protection_failed_emergency_close")
                    return TradeResult(
                        self.adapter.name,
                        "live_entry_rolled_back",
                        signal=signal,
                        position=None,
                        raw={"error": str(protection_error), "emergency_close": "confirmed"},
                    )
                except Exception as rollback_failure:
                    rollback_error = str(rollback_failure)
                return TradeResult(
                    self.adapter.name,
                    "live_unprotected_close_pending",
                    signal=signal,
                    position=self.position,
                    raw={"protection_error": str(protection_error), "emergency_close_error": rollback_error},
                )
        else:
            position = Position(
                signal.side,
                protection.quantity,
                current_price,
                protection.stop_price,
                protection.take_profit_price,
                int(time.time() * 1000),
                initial_stop_price=protection.stop_price,
                best_price=current_price,
                worst_price=current_price,
            )
            status = "paper_signal"
        self.position = position
        self.position_signal = signal
        self.position_equity_before = equity
        # Entry occurs at the latest closed execution candle's close. Its high
        # and low happened before entry and must never manage the new position.
        self.last_position_candle_timestamp = execution_candles[-1].timestamp
        if self.notifier is not None:
            try:
                self.notifier.notify_open(
                    position,
                    signal,
                    exchange=self.adapter.name,
                    symbol=self.adapter.settings.symbol,
                    mode=self.config.mode,
                )
            except Exception:  # Notification failures must never affect an entry.
                LOG.exception("open email notification failed; position remains open")
        return TradeResult(
            self.adapter.name,
            status,
            signal=signal,
            position=position,
            raw=live_raw if self.config.mode == "live" else None,
        )

    def _reconcile_binance_live_position(self, candle: Any) -> None:
        remote = self.adapter.fetch_live_position()
        if self.position is None:
            if remote is not None:
                self._observe_unmanaged_live_position(remote)
                raise RuntimeError(
                    "unmanaged Binance position detected; engine blocked until it is resolved on the exchange"
                )
            if self.unmanaged_live_position is not None:
                self._resolve_unmanaged_live_position(candle)
            return
        position = self.position
        if remote is not None:
            if remote.get("side") != position.side:
                raise RuntimeError("Binance position side differs from local engine state")
            remote_quantity = float(remote.get("quantity") or 0)
            if abs(remote_quantity - position.quantity) > max(1e-12, position.quantity * 0.001):
                raise RuntimeError("Binance position quantity differs from local engine state")
            if not position.stop_order_id:
                try:
                    rollback_payload = self.adapter.emergency_close(
                        position,
                        client_id=self._client_id("emergency-retry"),
                    )
                    closed_quantity, close_price = self.adapter.market_fill(
                        rollback_payload,
                        fallback_price=float(remote.get("mark_price") or candle.close),
                    )
                    if closed_quantity + 1e-12 < position.quantity:
                        raise RuntimeError("emergency close retry was only partially filled")
                    self._confirm_binance_flat("unprotected-position emergency close retry")
                    self.adapter.cancel_protection_orders(position)
                    self._close_paper_position(close_price, "protection_failed_emergency_close")
                except Exception as error:
                    raise RuntimeError(f"unprotected Binance position; emergency close retry failed: {error}") from error
            return

        # The venue is flat while the engine still has a position. The hard
        # stop, an online dynamic exit, or an external close most likely won
        # the race between polls. Inspect known bot ids (including legacy
        # take-profit ids), cancel anything left, and record the best price.
        statuses = self.adapter.fetch_protection_status(position)
        exit_reason = "exchange_position_closed"
        exit_price = float(candle.close)
        for name, reason, configured_price in (
            ("take_profit", "take_profit", position.take_profit_price),
            (
                "stop",
                "stop_loss",
                position.initial_stop_price or position.stop_price,
            ),
        ):
            payload = statuses.get(name) or {}
            status = str(payload.get("algoStatus") or payload.get("status") or "").upper()
            if status in {"TRIGGERED", "FINISHED", "FILLED"}:
                exit_reason = reason
                for candidate in (
                    payload.get("actualPrice"),
                    payload.get("avgPrice"),
                    payload.get("triggerPrice"),
                    configured_price,
                ):
                    try:
                        selected_price = float(candidate or 0)
                    except (TypeError, ValueError):
                        continue
                    if selected_price > 0:
                        exit_price = selected_price
                        break
                break
        self.adapter.cancel_protection_orders(position)
        self._close_paper_position(exit_price, exit_reason)

    def _observe_unmanaged_live_position(self, remote: dict[str, Any]) -> None:
        side = str(remote.get("side") or "")
        quantity = float(remote.get("quantity") or 0)
        entry_price = float(remote.get("entry_price") or 0)
        if side not in {"long", "short"} or quantity <= 0 or entry_price <= 0:
            raise RuntimeError("unmanaged Binance position details are invalid")
        current = self.unmanaged_live_position
        if current is not None:
            same_position = (
                current.get("side") == side
                and abs(float(current.get("entry_price") or 0) - entry_price)
                <= max(1e-8, entry_price * 0.00001)
            )
            if same_position:
                tracked_quantity = float(current.get("quantity") or 0)
                if quantity > tracked_quantity + max(1e-12, tracked_quantity * 0.001):
                    # Preserve the largest observed exposure so several partial
                    # close fills can be aggregated into one final email.
                    current["quantity"] = quantity
                    self._save_unmanaged_live_position()
                if not current.get("open_notified"):
                    self._notify_unmanaged_live_position_open()
                return
        observed_at_ms = int(time.time() * 1000)
        self.unmanaged_live_position = {
            "exchange": self.adapter.name,
            "symbol": self.adapter.settings.symbol,
            "environment": getattr(self.adapter.settings, "environment", "production"),
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "observed_at_ms": observed_at_ms,
            "open_notified": False,
        }
        self._save_unmanaged_live_position()
        self._notify_unmanaged_live_position_open()

    def _notify_unmanaged_live_position_open(self) -> None:
        snapshot = self.unmanaged_live_position
        if snapshot is None:
            return
        if self.notifier is not None:
            try:
                accepted = self.notifier.notify_reconciled_open(
                    exchange=self.adapter.name,
                    symbol=self.adapter.settings.symbol,
                    mode=self.config.mode,
                    environment=getattr(self.adapter.settings, "environment", "production"),
                    side=str(snapshot["side"]),
                    quantity=float(snapshot["quantity"]),
                    entry_price=float(snapshot["entry_price"]),
                    observed_at_ms=int(snapshot["observed_at_ms"]),
                )
                snapshot["open_notified"] = bool(accepted)
                self._save_unmanaged_live_position()
            except Exception:
                LOG.exception("reconciled open email notification failed; unmanaged position remains blocked")

    def _resolve_unmanaged_live_position(self, candle: Any) -> None:
        snapshot = self.unmanaged_live_position
        if snapshot is None:
            return
        position = Position(
            str(snapshot["side"]),
            float(snapshot["quantity"]),
            float(snapshot["entry_price"]),
            float(snapshot["entry_price"]),
            float(snapshot["entry_price"]),
            int(snapshot["observed_at_ms"]),
        )
        fill: dict[str, Any] | None = None
        try:
            fill = self.adapter.fetch_latest_close_fill(
                position,
                after_ms=int(snapshot["observed_at_ms"]),
            )
        except Exception:
            LOG.exception("Binance close-fill lookup failed; fallback email will use the latest closed candle")
        exit_price = float((fill or {}).get("price") or candle.close)
        exit_time_ms = int((fill or {}).get("timestamp") or time.time() * 1000)
        if self.notifier is not None:
            try:
                self.notifier.notify_reconciled_close(
                    exchange=self.adapter.name,
                    symbol=self.adapter.settings.symbol,
                    mode=self.config.mode,
                    environment=getattr(self.adapter.settings, "environment", "production"),
                    side=position.side,
                    quantity=float((fill or {}).get("quantity") or position.quantity),
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    exit_time_ms=exit_time_ms,
                    realized_pnl=(float(fill["realized_pnl"]) if fill and "realized_pnl" in fill else None),
                    commission=(float(fill["commission"]) if fill and "commission" in fill else None),
                    commission_assets=tuple((fill or {}).get("commission_assets") or ()),
                    source=str((fill or {}).get("source") or "Binance 仓位对账"),
                )
            except Exception:
                LOG.exception("reconciled close email notification failed; venue position is already flat")
        self.unmanaged_live_position = None
        self._save_unmanaged_live_position()

    def _load_unmanaged_live_position(self) -> dict[str, Any] | None:
        if self.config.mode != "live" or self.adapter.name != "binance" or not self.config.reconciliation_state_path:
            return None
        path = Path(self.config.reconciliation_state_path)
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            snapshot = state.get("unmanaged_position")
            if not isinstance(snapshot, dict):
                return None
            if (
                snapshot.get("exchange") != self.adapter.name
                or snapshot.get("symbol") != self.adapter.settings.symbol
                or snapshot.get("environment") != getattr(self.adapter.settings, "environment", "production")
            ):
                return None
            return snapshot
        except Exception as error:
            LOG.warning("live reconciliation state ignored: %s", error)
            return None

    def _save_unmanaged_live_position(self) -> None:
        if not self.config.reconciliation_state_path:
            return
        path = Path(self.config.reconciliation_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"unmanaged_position": self.unmanaged_live_position},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _manage_live_position(
        self,
        mark_price: float,
        candles_by_timeframe: dict[str, list[Any]],
        signal: Signal,
    ) -> TradeResult | None:
        """Apply online strategy exits while exchange hard protection remains open."""
        if self.position is None:
            return None
        if mark_price <= 0:
            raise ValueError("live mark price must be positive")
        position = self.position
        if position.side == "long":
            position = replace(
                position,
                best_price=max(position.best_price or position.entry_price, mark_price),
                worst_price=min(position.worst_price or position.entry_price, mark_price),
            )
        else:
            position = replace(
                position,
                best_price=min(position.best_price or position.entry_price, mark_price),
                worst_price=max(position.worst_price or position.entry_price, mark_price),
            )
        self.position = position

        exit_reason = ""
        if self._stop_is_protected(position):
            protected_stop_hit = (
                mark_price <= position.stop_price
                if position.side == "long"
                else mark_price >= position.stop_price
            )
            if protected_stop_hit:
                exit_reason = self._stop_exit_reason(position)
        if not exit_reason and self._profit_trend_exit_ready(candles_by_timeframe):
            exit_reason = "trend_invalidation"
        if not exit_reason:
            exit_reason = self._live_time_exit_reason(mark_price)

        if exit_reason:
            close_result = self._close_live_position(mark_price, exit_reason)
            return TradeResult(
                self.adapter.name,
                "live_active_exit",
                signal=signal,
                position=self.position,
                raw={
                    "exit_reason": exit_reason,
                    "decision_mark_price": mark_price,
                    **close_result,
                },
            )

        # Tightening is deliberately local: Binance's original hard stop
        # remains untouched as the network-outage fallback. While online, each
        # poll can exit from trend invalidation or an improved local
        # break-even/trailing level; no fixed exchange take-profit is placed.
        now_ms = int(time.time() * 1000)
        self._tighten_paper_stop(
            Candle(now_ms, mark_price, mark_price, mark_price, mark_price, 0.0)
        )
        return None

    def _live_time_exit_reason(self, mark_price: float) -> str:
        position = self.position
        if position is None or not bool(getattr(self.strategy.config, "enable_time_exit", False)):
            return ""
        soft_limit = max(0, int(getattr(self.strategy.config, "max_hold_seconds", 0)))
        if not soft_limit:
            return ""
        hard_limit = max(
            soft_limit,
            int(getattr(self.strategy.config, "hard_max_hold_seconds", soft_limit)),
        )
        held_seconds = max(0.0, (time.time() * 1000 - position.opened_at) / 1000)
        if held_seconds < soft_limit:
            return ""
        initial_stop = position.initial_stop_price or position.stop_price
        risk_distance = abs(position.entry_price - initial_stop)
        favorable_distance = (
            mark_price - position.entry_price
            if position.side == "long"
            else position.entry_price - mark_price
        )
        current_r = favorable_distance / risk_distance if risk_distance > 0 else 0.0
        net_at_mark = self.risk.estimate_net_pnl(
            position.side,
            position.entry_price,
            mark_price,
            position.quantity,
            holding_hours=held_seconds / 3600,
        )
        minimum_r = max(0.0, float(getattr(self.strategy.config, "time_exit_min_r", 0.5)))
        if net_at_mark > 0 and current_r >= minimum_r:
            return "time_exit"
        return "hard_time_exit" if held_seconds >= hard_limit else ""

    def _close_live_position(self, reference_price: float, exit_reason: str) -> dict[str, Any]:
        if self.position is None:
            return {"already_flat": True, "exit_price": reference_price}
        position = self.position
        try:
            close_payload = self.adapter.place_market_order(
                OrderRequest(
                    self._opposite_side(position.side),
                    position.quantity,
                    True,
                    self._client_id("close"),
                )
            )
            closed_quantity, close_price = self.adapter.market_fill(
                close_payload,
                fallback_price=reference_price,
            )
            if closed_quantity + 1e-12 < position.quantity:
                raise RuntimeError("live close was only partially filled")
            self._confirm_binance_flat(f"{exit_reason} active close")
        except Exception as close_error:
            # A server-side protection order can win the race against the
            # reduce-only market close.  If Binance is already flat, reconcile
            # the actual exchange exit instead of attempting another close.
            if self.adapter.name == "binance":
                try:
                    remote = self.adapter.fetch_live_position()
                except Exception:
                    remote = {"unknown": True}
                if remote is None:
                    self._reconcile_binance_live_position(
                        Candle(
                            int(time.time() * 1000),
                            reference_price,
                            reference_price,
                            reference_price,
                            reference_price,
                            0.0,
                        )
                    )
                    return {
                        "exchange_race_reconciled": True,
                        "exit_price": reference_price,
                        "close_error": str(close_error),
                    }
            raise

        cancellation = self.adapter.cancel_protection_orders(position)
        if cancellation.get("errors"):
            LOG.warning("%s protection cancellation warnings=%s", self.adapter.name, cancellation["errors"])
        self._close_paper_position(close_price, exit_reason)
        return {
            "closed_quantity": closed_quantity,
            "exit_price": close_price,
            "protection_cancelled": cancellation,
        }

    def _select_live_entry_quantity(
        self,
        desired_quantity: float,
        reference_price: float,
        signal: Signal,
    ) -> tuple[float | None, dict[str, Any]]:
        sizing: dict[str, Any] = {
            "desired_quantity": desired_quantity,
            "minimum_fallback_used": False,
        }
        try:
            selected = self.adapter.normalize_order_quantity(desired_quantity, reference_price)
            sizing["selected_quantity"] = selected
            return selected, sizing
        except ValueError as normalization_error:
            if self.adapter.name != "binance":
                raise
            original_normalization_error = normalization_error

        try:
            snapshot = self.adapter.fetch_dashboard_snapshot()
        except Exception as snapshot_error:
            sizing.update({"blocked": True, "capacity_error": str(snapshot_error)})
            self._notify_order_unavailable(
                signal,
                reference_price,
                sizing,
                f"无法确认账户是否足以开交易所最小订单：{snapshot_error}",
            )
            return None, sizing

        limits = dict(snapshot.get("order_limits") or {})
        minimum_quantity = float(limits.get("effective_min_quantity_raw") or 0)
        minimum_notional = float(limits.get("effective_min_notional_raw") or 0)
        available_quantity = float(limits.get("estimated_max_open_quantity_raw") or 0)
        available_notional = float(limits.get("estimated_max_open_notional_raw") or 0)
        sizing.update(
            {
                "minimum_quantity": minimum_quantity,
                "minimum_notional": minimum_notional,
                "available_quantity": available_quantity,
                "available_notional": available_notional,
            }
        )
        if minimum_quantity <= 0 or desired_quantity + 1e-12 >= minimum_quantity:
            # The original normalization error was not caused by the minimum
            # order threshold, so it must not be silently converted into a
            # different quantity.
            raise original_normalization_error

        private_available = bool(snapshot.get("private_available"))
        capacity_sufficient = (
            private_available
            and available_quantity + 1e-12 >= minimum_quantity
            and available_notional + 1e-9 >= minimum_notional
        )
        if not capacity_sufficient:
            reason = str(
                snapshot.get("private_error")
                or "账户当前理论可开金额低于交易所最小订单"
            )
            sizing.update({"blocked": True, "capacity_reason": reason})
            self._notify_order_unavailable(signal, reference_price, sizing, reason)
            return None, sizing

        try:
            selected = self.adapter.normalize_order_quantity(minimum_quantity, reference_price)
        except ValueError as minimum_error:
            sizing.update({"blocked": True, "capacity_reason": str(minimum_error)})
            self._notify_order_unavailable(
                signal,
                reference_price,
                sizing,
                f"交易所最小订单校验仍未通过：{minimum_error}",
            )
            return None, sizing
        sizing.update(
            {
                "selected_quantity": selected,
                "minimum_fallback_used": True,
                "override_reason": "30%容量/风险计算结果低于交易所最小订单，按交易所最小数量下单",
            }
        )
        LOG.warning(
            "%s minimum-order fallback side=%s desired=%.8f selected=%.8f min_notional=%.4f",
            self.adapter.name,
            signal.side,
            desired_quantity,
            selected,
            minimum_notional,
        )
        return selected, sizing

    def _notify_order_unavailable(
        self,
        signal: Signal,
        current_price: float,
        sizing: dict[str, Any],
        reason: str,
    ) -> None:
        if self.notifier is None:
            return
        notify = getattr(self.notifier, "notify_order_unavailable", None)
        if not callable(notify):
            return
        try:
            notify(
                exchange=self.adapter.name,
                symbol=self.adapter.settings.symbol,
                mode=self.config.mode,
                side=signal.side,
                current_price=current_price,
                minimum_quantity=float(sizing.get("minimum_quantity") or 0),
                minimum_notional=float(sizing.get("minimum_notional") or 0),
                available_quantity=float(sizing.get("available_quantity") or 0),
                available_notional=float(sizing.get("available_notional") or 0),
                reason=reason,
            )
        except Exception:  # Notification failures must never affect order safety.
            LOG.exception("minimum-order email notification failed; entry remains blocked")

    @staticmethod
    def _is_order_capacity_error(error: Exception) -> bool:
        message = str(error).lower().replace(" ", "")
        return any(
            marker in message
            for marker in (
                '"code":-2019',
                '"code":-4164',
                '"code":-4004',
                "marginisinsufficient",
                "insufficientbalance",
                "minimumnotional",
                "belownotionalminimum",
            )
        )

    def _confirm_binance_flat(self, context: str) -> None:
        if self.adapter.name != "binance":
            return
        remote = self.adapter.fetch_live_position()
        if remote is not None:
            raise RuntimeError(f"{context} returned a fill but Binance still reports an open position")

    def _pending_macro_signal_expired(self, candles_by_timeframe: dict[str, list[Any]]) -> bool:
        pending = self.pending_macro_signal
        if pending is None:
            return True
        trigger_timeframe = self.strategy.config.trigger_timeframe
        trigger_candles = candles_by_timeframe.get(trigger_timeframe, [])
        if not trigger_candles:
            return True
        timeframe_ms = {
            "30s": 30_000,
            "1m": 60_000,
            "5m": 300_000,
            "15m": 900_000,
            "1h": 3_600_000,
            "4h": 14_400_000,
        }.get(trigger_timeframe, 0)
        valid_bars = max(
            1,
            int(getattr(self.strategy.config, "traditional_blocked_setup_valid_bars", 1)),
        )
        maximum_age_ms = max(0, valid_bars - 1) * timeframe_ms
        return trigger_candles[-1].timestamp - pending.timestamp > maximum_age_ms

    def _should_reverse(self, position: Position, current_price: float, signal: Signal) -> bool:
        """Avoid paying a round-trip fee for weak, losing reversals."""
        held_seconds = max(0.0, (time.time() * 1000 - position.opened_at) / 1000)
        minimum_hold = max(0, int(getattr(self.strategy.config, "min_hold_seconds", 60)))
        if held_seconds < minimum_hold:
            return False
        net_exit = self.risk.estimate_net_pnl(
            position.side,
            position.entry_price,
            current_price,
            position.quantity,
            holding_hours=held_seconds / 3600,
        )
        if net_exit > 0:
            return True
        required_score = max(1, int(getattr(self.strategy.config, "reversal_min_score", 5)))
        return signal.score >= required_score

    def _dynamic_stop_loss_pct(self, candles: list[Any]) -> float:
        """Set the stop distance from recent 30-second volatility.

        The configured risk stop remains the fallback when ATR is unavailable.
        Clamping prevents a single noisy candle from creating an unusably wide
        stop or a stop that is too close to survive normal market noise.
        """
        config = self.strategy.config
        fallback = float(self.risk.config.stop_loss_pct)
        if len(candles) < max(2, int(getattr(config, "atr_period", 7))):
            return fallback
        values = atr(
            [float(candle.high) for candle in candles],
            [float(candle.low) for candle in candles],
            [float(candle.close) for candle in candles],
            int(getattr(config, "atr_period", 7)),
        )
        current_atr = values[-1]
        close = float(candles[-1].close)
        if current_atr is None or close <= 0:
            return fallback
        raw_pct = (float(current_atr) / close) * float(getattr(config, "atr_stop_multiplier", 1.4))
        minimum = float(getattr(config, "min_stop_loss_pct", fallback))
        maximum = float(getattr(config, "max_stop_loss_pct", fallback))
        if minimum <= 0 or maximum < minimum:
            return fallback
        return max(minimum, min(maximum, raw_pct))

    def _entry_allowed(self) -> bool:
        threshold = int(self.risk.config.max_consecutive_losses)
        if threshold > 0 and self.consecutive_losses >= threshold:
            pause_minutes = max(0, int(getattr(self.risk.config, "loss_streak_pause_minutes", 0)))
            if pause_minutes <= 0 or time.time() < self.cooldown_until:
                return False
            LOG.info("loss-streak pause elapsed; resetting entry circuit breaker")
            self.consecutive_losses = 0
        if self.session_pnl <= -(self.config.paper_equity * self.risk.config.max_daily_loss_pct):
            return False
        return time.time() >= self.cooldown_until

    def _manage_paper_position(
        self,
        candle: Any,
        candles_by_timeframe: dict[str, list[Any]] | None = None,
    ) -> None:
        if self.position is None:
            return
        if self.position.side == "long":
            self.position = replace(
                self.position,
                best_price=max(self.position.best_price or self.position.entry_price, float(candle.high)),
                worst_price=min(self.position.worst_price or self.position.entry_price, float(candle.low)),
            )
        else:
            self.position = replace(
                self.position,
                best_price=min(self.position.best_price or self.position.entry_price, float(candle.low)),
                worst_price=max(self.position.worst_price or self.position.entry_price, float(candle.high)),
            )
        exit_price: float | None = None
        exit_reason = ""
        if self.position.side == "long":
            if candle.low <= self.position.stop_price:
                exit_price = self.position.stop_price
                exit_reason = self._stop_exit_reason(self.position)
            elif candle.high >= self.position.take_profit_price:
                exit_price = self.position.take_profit_price
                exit_reason = "take_profit"
        else:
            if candle.high >= self.position.stop_price:
                exit_price = self.position.stop_price
                exit_reason = self._stop_exit_reason(self.position)
            elif candle.low <= self.position.take_profit_price:
                exit_price = self.position.take_profit_price
                exit_reason = "take_profit"
        time_exit_enabled = bool(getattr(self.strategy.config, "enable_time_exit", False))
        soft_max_hold_seconds = max(0, int(getattr(self.strategy.config, "max_hold_seconds", 0)))
        if exit_price is None and time_exit_enabled and soft_max_hold_seconds:
            hard_max_hold_seconds = max(
                soft_max_hold_seconds,
                int(getattr(self.strategy.config, "hard_max_hold_seconds", soft_max_hold_seconds)),
            )
            held_seconds = max(0.0, (int(time.time() * 1000) - self.position.opened_at) / 1000)
            if held_seconds >= soft_max_hold_seconds:
                net_at_close = self.risk.estimate_net_pnl(
                    self.position.side,
                    self.position.entry_price,
                    candle.close,
                    self.position.quantity,
                    holding_hours=held_seconds / 3600,
                )
                initial_stop = self.position.initial_stop_price or self.position.stop_price
                stop_distance = abs(self.position.entry_price - initial_stop)
                favorable_distance = (
                    candle.close - self.position.entry_price
                    if self.position.side == "long"
                    else self.position.entry_price - candle.close
                )
                current_r = favorable_distance / stop_distance if stop_distance > 0 else 0.0
                min_time_exit_r = max(0.0, float(getattr(self.strategy.config, "time_exit_min_r", 0.5)))
                profitable_time_exit = net_at_close > 0 and current_r >= min_time_exit_r
                if profitable_time_exit or held_seconds >= hard_max_hold_seconds:
                    exit_price = candle.close
                    exit_reason = "time_exit" if profitable_time_exit else "hard_time_exit"
        if exit_price is None and self._profit_trend_exit_ready(candles_by_timeframe):
            exit_price = float(candle.close)
            exit_reason = "trend_invalidation"
        if exit_price is not None:
            self._close_paper_position(exit_price, exit_reason)
        else:
            self._tighten_paper_stop(candle)

    def _manage_macro_position(self, candle: Any, decision: MacroRiskDecision) -> None:
        """Exit weak positions or protect profitable ones before a major release."""
        if self.position is None or self.macro_risk is None or decision.event is None:
            return
        seconds_to_event = decision.seconds_to_event
        if seconds_to_event is None or seconds_to_event < 0:
            return
        protection_window = self.macro_risk.config.position_management_before_minutes * 60
        if seconds_to_event > protection_window:
            return
        position = self.position
        initial_stop = position.initial_stop_price or position.stop_price
        risk_distance = abs(position.entry_price - initial_stop)
        if risk_distance <= 0:
            return
        current_r = (
            (float(candle.close) - position.entry_price) / risk_distance
            if position.side == "long"
            else (position.entry_price - float(candle.close)) / risk_distance
        )
        if current_r < self.macro_risk.config.position_min_r:
            LOG.warning(
                "%s macro event exit event=%s side=%s current_r=%.2f",
                self.adapter.name,
                decision.event.name,
                position.side,
                current_r,
            )
            self._close_paper_position(float(candle.close), "macro_event_exit")
            return

        holding_hours = max(0.0, (time.time() * 1000 - position.opened_at) / 3_600_000)
        cost_break_even = self.risk.break_even_price(
            position.side,
            position.entry_price,
            holding_hours=holding_hours,
        )
        candidate = min(cost_break_even, float(candle.close)) if position.side == "long" else max(cost_break_even, float(candle.close))
        improved = candidate > position.stop_price if position.side == "long" else candidate < position.stop_price
        if improved:
            self.position = replace(position, stop_price=candidate, stop_reason="macro_event_stop")
            LOG.warning(
                "%s macro event protected stop event=%s side=%s stop=%.4f current_r=%.2f",
                self.adapter.name,
                decision.event.name,
                position.side,
                candidate,
                current_r,
            )

    def _profit_trend_exit_ready(self, candles_by_timeframe: dict[str, list[Any]] | None) -> bool:
        if self.position is None or not candles_by_timeframe:
            return False
        if not bool(getattr(self.strategy.config, "enable_profit_trend_exit", False)):
            return False
        initial_stop = self.position.initial_stop_price or self.position.stop_price
        risk_distance = abs(self.position.entry_price - initial_stop)
        if risk_distance <= 0:
            return False
        if self.position.side == "long":
            favorable_r = ((self.position.best_price or self.position.entry_price) - self.position.entry_price) / risk_distance
        else:
            favorable_r = (self.position.entry_price - (self.position.best_price or self.position.entry_price)) / risk_distance
        trigger_r = max(0.0, float(getattr(self.strategy.config, "profit_trend_exit_trigger_r", 1.0)))
        if favorable_r < trigger_r:
            return False
        invalidated = getattr(self.strategy, "position_trend_invalidated", None)
        return bool(invalidated and invalidated(self.position.side, candles_by_timeframe))

    def _tighten_paper_stop(self, candle: Any) -> None:
        """Move a profitable paper position to cost-aware break-even, then trail it."""
        if self.position is None:
            return
        position = self.position
        initial_stop = position.initial_stop_price or position.stop_price
        risk_distance = abs(position.entry_price - initial_stop)
        if risk_distance <= 0:
            return
        if position.side == "long":
            best_price = max(position.best_price or position.entry_price, float(candle.high))
            worst_price = min(position.worst_price or position.entry_price, float(candle.low))
            favorable_r = (best_price - position.entry_price) / risk_distance
        else:
            best_price = min(position.best_price or position.entry_price, float(candle.low))
            worst_price = max(position.worst_price or position.entry_price, float(candle.high))
            favorable_r = (position.entry_price - best_price) / risk_distance

        candidate = position.stop_price
        candidate_reason = position.stop_reason
        break_even_trigger = max(0.0, float(getattr(self.strategy.config, "break_even_trigger_r", 1.25)))
        if break_even_trigger and favorable_r >= break_even_trigger:
            holding_hours = max(0.0, (time.time() * 1000 - position.opened_at) / 3_600_000)
            cost_break_even = self.risk.break_even_price(
                position.side,
                position.entry_price,
                holding_hours=holding_hours,
            )
            lock_distance = risk_distance * max(0.0, float(getattr(self.strategy.config, "break_even_lock_r", 0.5)))
            protected_stop = cost_break_even + lock_distance if position.side == "long" else cost_break_even - lock_distance
            protection_improved = protected_stop > candidate if position.side == "long" else protected_stop < candidate
            if protection_improved:
                candidate = protected_stop
                candidate_reason = "break_even_stop"

        trailing_trigger = max(0.0, float(getattr(self.strategy.config, "trailing_trigger_r", 2.0)))
        if trailing_trigger and favorable_r >= trailing_trigger:
            trailing_distance = risk_distance * max(0.1, float(getattr(self.strategy.config, "trailing_distance_r", 0.75)))
            trailing_stop = best_price - trailing_distance if position.side == "long" else best_price + trailing_distance
            trailing_improved = trailing_stop > candidate if position.side == "long" else trailing_stop < candidate
            if trailing_improved:
                candidate = trailing_stop
                candidate_reason = "trailing_stop"

        candidate = min(candidate, float(candle.close)) if position.side == "long" else max(candidate, float(candle.close))
        stop_improved = candidate > position.stop_price if position.side == "long" else candidate < position.stop_price
        self.position = replace(
            position,
            stop_price=candidate if stop_improved else position.stop_price,
            initial_stop_price=initial_stop,
            best_price=best_price,
            stop_reason=candidate_reason if stop_improved else position.stop_reason,
            worst_price=worst_price,
        )
        if stop_improved:
            LOG.info(
                "%s paper protected stop side=%s stop=%.4f best=%.4f r=%.2f",
                self.adapter.name,
                position.side,
                candidate,
                best_price,
                favorable_r,
            )

    @staticmethod
    def _stop_is_protected(position: Position) -> bool:
        initial_stop = position.initial_stop_price
        if initial_stop is None:
            return False
        if position.side == "long":
            return position.stop_price > initial_stop
        return position.stop_price < initial_stop

    @classmethod
    def _stop_exit_reason(cls, position: Position) -> str:
        if not cls._stop_is_protected(position):
            return "stop_loss"
        if position.stop_reason in {"break_even_stop", "trailing_stop", "macro_event_stop"}:
            return position.stop_reason
        return "trailing_stop"

    def _close_paper_position(self, exit_price: float, exit_reason: str) -> None:
        if self.position is None:
            return
        position = self.position
        exit_time_ms = int(time.time() * 1000)
        holding_hours = max(0.0, (exit_time_ms - position.opened_at) / 3_600_000)
        pnl = self.risk.estimate_net_pnl(
            position.side,
            position.entry_price,
            exit_price,
            position.quantity,
            holding_hours=holding_hours,
        )
        record: TradeRecord | None = None
        if self.reporter is not None or self.notifier is not None:
            record = TradeRecord.from_position(
                exchange=self.adapter.name,
                symbol=getattr(self.adapter.settings, "symbol", ""),
                mode=self.config.mode,
                environment=getattr(self.adapter.settings, "environment", "testnet"),
                position=position,
                exit_price=exit_price,
                exit_time_ms=exit_time_ms,
                exit_reason=exit_reason,
                costs=self.risk.costs,
                equity_before=self.position_equity_before or self.config.paper_equity,
                signal=self.position_signal,
            )
        if self.reporter is not None and record is not None:
            self.reporter.record_trade(record)
        self.session_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
            cooldown_minutes = max(0, int(self.risk.config.cooldown_minutes))
            threshold = int(self.risk.config.max_consecutive_losses)
            if threshold > 0 and self.consecutive_losses >= threshold:
                cooldown_minutes = max(
                    cooldown_minutes,
                    max(0, int(getattr(self.risk.config, "loss_streak_pause_minutes", 0))),
                )
            self.cooldown_until = time.time() + cooldown_minutes * 60
        else:
            self.consecutive_losses = 0
        LOG.info(
            "%s %s exit side=%s price=%.4f pnl=%.4f",
            self.adapter.name,
            self.config.mode,
            position.side,
            exit_price,
            pnl,
        )
        self.position = None
        self.position_signal = None
        self.position_equity_before = 0.0
        self.last_position_candle_timestamp = 0
        if self.notifier is not None and record is not None:
            try:
                report_date = (
                    datetime.fromtimestamp(exit_time_ms / 1000, tz=self.reporter.timezone).date()
                    if self.reporter is not None
                    else None
                )
                self.notifier.notify_close(
                    record,
                    reporter=self.reporter,
                    report_date=report_date,
                )
            except Exception:  # Position cleanup is already complete and cannot be rolled back by email.
                LOG.exception("close email notification failed; position remains closed")

    def close(self) -> None:
        if self.notifier is not None:
            self.notifier.close()

    def run_forever(self) -> None:
        LOG.info("starting %s mode for %s; poll=%ss", self.config.mode, self.adapter.name, self.config.poll_seconds)
        while True:
            try:
                result = self.evaluate_once()
                if result.status not in {"no_action", "position_held"}:
                    LOG.warning("%s", result)
            except Exception:
                LOG.exception("cycle failed for %s; no new order submitted", self.adapter.name)
            time.sleep(max(5, self.config.poll_seconds))

    @staticmethod
    def _opposite_side(side: str) -> str:
        return "sell" if side == "long" else "buy"

    @staticmethod
    def _client_id(prefix: str) -> str:
        stem = f"btcbot-{prefix}-"
        suffix_length = 36 - len(stem)
        if suffix_length < 8:
            raise ValueError("client order id prefix is too long for Binance")
        return f"{stem}{uuid.uuid4().hex[:suffix_length]}"
