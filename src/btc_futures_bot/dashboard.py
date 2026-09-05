from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .engine import cycle_wait_seconds, normalized_poll_seconds
from .http_client import ApiError, clear_rate_limits, is_rate_limit_error
from .main import (
    build_engine,
    credential_values,
    load_config,
    local_config_path,
    report_directory,
    save_dashboard_config,
)
from .notifications import EmailNotificationConfig, EmailNotifier
from .operation_log import OperationLogger
from .reporting import TradeReporter

LOG = logging.getLogger(__name__)


def _masked(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "已保存"
    return f"{value[:4]}…{value[-4:]}"


def _json_safe(value: Any) -> Any:
    """Return standards-compliant JSON values for browser responses."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value if value not in (None, "") else "0"))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _order_sizing_view(
    config: dict[str, Any],
    snapshot: dict[str, Any],
    account: dict[str, Any],
) -> dict[str, Any]:
    limits = dict(snapshot.get("order_limits") or {})
    account_unavailable = account.get("source") == "unavailable"
    result: dict[str, Any] = {
        "available": bool(limits)
        and not bool(limits.get("error"))
        and not account_unavailable,
        "error": (
            "私有账户快照暂不可用"
            if account_unavailable
            else str(limits.get("error") or "")
        ),
        **limits,
    }
    if not result["available"]:
        return result
    equity = _decimal(account.get("wallet_balance_raw") or account.get("wallet_balance"))
    risk_per_trade = _decimal(config.get("risk", {}).get("risk_per_trade", 0))
    max_notional_pct = _decimal(config.get("risk", {}).get("max_notional_pct", 0))
    minimum = _decimal(limits.get("effective_min_notional_raw"))
    exchange_capacity = _decimal(limits.get("estimated_max_open_notional_raw"))
    configured_leverage = _decimal(config.get("account", {}).get("max_leverage", 0))
    fallback_capacity = equity * configured_leverage
    theoretical_capacity = exchange_capacity if exchange_capacity > 0 else fallback_capacity
    strategy_cap = theoretical_capacity * max_notional_pct
    effective_cap = min(strategy_cap, theoretical_capacity) if theoretical_capacity > 0 else strategy_cap
    result.update(
        {
            "risk_budget_raw": _decimal_text(equity * risk_per_trade),
            "risk_per_trade_raw": _decimal_text(risk_per_trade),
            "max_notional_pct_raw": _decimal_text(max_notional_pct),
            "strategy_cap_basis_raw": _decimal_text(theoretical_capacity),
            "strategy_cap_basis": "exchange_estimate" if exchange_capacity > 0 else "equity_times_leverage",
            "strategy_notional_cap_raw": _decimal_text(strategy_cap),
            "effective_strategy_cap_raw": _decimal_text(effective_cap),
            "strategy_cap_meets_minimum": bool(minimum > 0 and effective_cap >= minimum),
            "minimum_fallback_available": bool(
                minimum > 0 and theoretical_capacity >= minimum
            ),
            "formula": "先取 min(风险预算÷止损含成本, 理论最大可开金额×30%)；低于交易所最小单时，在账户容量足够的前提下改用最小单，否则不下单并邮件通知",
        }
    )
    return result


def _position_dict(
    position: Any,
    mark_price: float | None = None,
    *,
    mode: str = "paper",
) -> dict[str, Any] | None:
    if position is None:
        return None
    mark = float(mark_price or position.entry_price)
    direction = 1 if position.side == "long" else -1
    unrealized = (mark - position.entry_price) * position.quantity * direction
    notional = abs(position.entry_price * position.quantity)
    return {
        "source": "paper",
        "side": position.side,
        "quantity": position.quantity,
        "entry_price": position.entry_price,
        "mark_price": mark,
        "stop_price": position.stop_price,
        "take_profit_price": position.take_profit_price,
        "take_profit_mode": "dynamic" if mode == "live" else "fixed",
        "unrealized_pnl": unrealized,
        "unrealized_pnl_pct": unrealized / notional if notional else 0.0,
        "opened_at": position.opened_at,
    }


def _mark_to_market_view(
    snapshot: dict[str, Any],
    symbol: str,
    mark_price: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refresh price-derived position/account values without a private API read."""

    positions = [dict(position) for position in snapshot.get("positions") or []]
    account = dict(snapshot.get("account") or {})
    selected_symbol = str(symbol or "").upper()
    refreshed_pnl = Decimal("0")
    all_positions_refreshed = bool(positions)
    for position in positions:
        position_symbol = str(position.get("symbol") or selected_symbol).upper()
        if mark_price <= 0 or position_symbol != selected_symbol:
            all_positions_refreshed = False
            continue
        entry = _decimal(position.get("entry_price"))
        quantity = abs(_decimal(position.get("quantity")))
        direction = Decimal("1") if str(position.get("side")) == "long" else Decimal("-1")
        pnl = (Decimal(str(mark_price)) - entry) * quantity * direction
        notional = abs(entry * quantity)
        position["mark_price"] = mark_price
        position["unrealized_pnl"] = float(pnl)
        position["unrealized_pnl_raw"] = _decimal_text(pnl)
        position["unrealized_pnl_pct"] = float(pnl / notional) if notional else 0.0
        refreshed_pnl += pnl
    if account and all_positions_refreshed:
        wallet = _decimal(account.get("wallet_balance_raw") or account.get("wallet_balance"))
        margin = wallet + refreshed_pnl
        account["unrealized_pnl"] = float(refreshed_pnl)
        account["unrealized_pnl_raw"] = _decimal_text(refreshed_pnl)
        account["margin_balance"] = float(margin)
        account["margin_balance_raw"] = _decimal_text(margin)
    return positions, account


class DashboardService:
    def __init__(self, config_path: str) -> None:
        self.config_path = str(Path(config_path))
        initial_config = load_config(self.config_path)
        self.exchange_name = str(initial_config.get("active_exchange", "binance"))
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self.engine: Any = None
        self.reporter: TradeReporter | None = None
        self.last_result: Any = None
        self.last_error = ""
        self.last_cycle_at = 0.0
        self.started_at = 0.0
        self._exchange_snapshot: dict[str, Any] = {}
        self._snapshot_at = 0.0
        self._private_snapshot_at = 0.0
        self._snapshot_refreshing = False
        self._snapshot_condition = threading.Condition(self._lock)
        self._dashboard_adapter: Any = None
        self._dashboard_adapter_key: tuple[str, str, str, str] | None = None
        self.operation_logger = OperationLogger(Path(self.config_path).parent / "logs" / "operation_log.jsonl")
        self.notifier = self._build_notifier(initial_config)
        self._first_cycle_logged = False
        self._last_logged_error = ""
        self._last_macro_block = ""
        self.server_started_at = time.time()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _config(self) -> dict[str, Any]:
        return load_config(self.config_path)

    def _build_notifier(self, config: dict[str, Any]) -> EmailNotifier:
        exchange_name = self._exchange(config)
        try:
            email_config = EmailNotificationConfig.from_mapping(
                config.get("email_notifications", {}),
                default_timezone=str(config.get("report_timezone", "Asia/Shanghai")),
                default_state_path=str(
                    report_directory(config, exchange_name) / "email_notification_state.json"
                ),
            )
            return EmailNotifier(email_config)
        except Exception as error:
            LOG.exception("email notifications disabled because configuration is invalid")
            return EmailNotifier(
                EmailNotificationConfig(enabled=False),
                initial_error=f"邮件配置无效，通知已隔离禁用：{error}",
            )

    def _exchange(self, config: dict[str, Any], requested: str = "") -> str:
        if requested and requested in config.get("exchanges", {}):
            return requested
        active = str(config.get("active_exchange", ""))
        if active in config.get("exchanges", {}):
            return active
        if self.exchange_name in config.get("exchanges", {}):
            return self.exchange_name
        enabled = [name for name, value in config.get("exchanges", {}).items() if value.get("enabled")]
        return enabled[0] if enabled else "binance"

    def _notify_emergency(
        self,
        error: BaseException | str,
        *,
        category: str,
        context: str,
        incident: str,
        config: dict[str, Any],
        exchange_name: str,
        details: dict[str, Any] | None = None,
    ) -> bool:
        notifier = getattr(self, "notifier", None)
        if notifier is None:
            return False
        notify = getattr(notifier, "notify_emergency", None)
        if not callable(notify):
            return False
        exchange = config.get("exchanges", {}).get(exchange_name, {})
        try:
            return bool(
                notify(
                    error,
                    category=category,
                    exchange=exchange_name,
                    symbol=str(exchange.get("symbol") or ""),
                    mode=str(config.get("mode") or "paper"),
                    environment=str(exchange.get("environment") or ""),
                    context=context,
                    incident=incident,
                    details=details,
                )
            )
        except Exception:
            LOG.exception("dashboard emergency email notification enqueue failed")
            return False

    def _handle_snapshot_alerts(
        self,
        snapshot: dict[str, Any],
        config: dict[str, Any],
        exchange_name: str,
    ) -> None:
        notifier = getattr(self, "notifier", None)
        if notifier is None:
            return
        order_limits = snapshot.get("order_limits") or {}
        private_stream = snapshot.get("private_stream") or {}
        candidates = (
            order_limits.get("error") if isinstance(order_limits, dict) else "",
            snapshot.get("private_error"),
            snapshot.get("snapshot_error"),
            private_stream.get("last_error") if isinstance(private_stream, dict) else "",
        )
        rate_limit_error = next(
            (item for item in candidates if item and is_rate_limit_error(item)),
            None,
        )
        if rate_limit_error is not None:
            self._notify_emergency(
                str(rate_limit_error),
                category="ip_restricted",
                context="页面交易所状态检测",
                incident="snapshot",
                config=config,
                exchange_name=exchange_name,
            )
            return
        if (
            snapshot.get("private_available")
            and str(snapshot.get("private_source") or "").lower() == "rest"
            and not (
            isinstance(order_limits, dict) and order_limits.get("error")
            )
        ):
            resolve = getattr(notifier, "resolve_emergency", None)
            if callable(resolve):
                try:
                    resolve("ip_restricted", exchange_name, "snapshot")
                except Exception:
                    LOG.exception("dashboard IP alert recovery reset failed")

    def config_view(self) -> dict[str, Any]:
        config = self._config()
        exchange_name = self._exchange(config)
        exchange = config.get("exchanges", {}).get(exchange_name, {})
        saved = credential_values(config, exchange_name)
        key = str(saved.get("api_key", "")) or (os.getenv(str(exchange.get("api_key_env", ""))) if exchange.get("api_key_env") else "")
        secret = str(saved.get("api_secret", "")) or (os.getenv(str(exchange.get("api_secret_env", ""))) if exchange.get("api_secret_env") else "")
        passphrase = str(saved.get("passphrase", ""))
        email_config = EmailNotificationConfig.from_mapping(
            config.get("email_notifications", {}),
            default_timezone=str(config.get("report_timezone", "Asia/Shanghai")),
            default_state_path=str(report_directory(config, exchange_name) / "email_notification_state.json"),
        )
        return {
            "config_file": str(local_config_path(self.config_path)),
            "exchange": exchange_name,
            "mode": config.get("mode", "paper"),
            "symbol": exchange.get("symbol", "BTCUSDT"),
            "environment": exchange.get("environment", "testnet"),
            "base_url": exchange.get("base_url", ""),
            "credentials": {
                "api_key_masked": _masked(key),
                "secret_saved": bool(secret),
                "passphrase_saved": bool(passphrase),
            },
            "poll_seconds": config.get("poll_seconds", 15),
            "paper_equity": config.get("paper_equity", 10000),
            "max_leverage": config.get("account", {}).get("max_leverage", 3),
            "stop_loss_pct": config.get("risk", {}).get("stop_loss_pct", 0.05),
            "risk_per_trade": config.get("risk", {}).get("risk_per_trade", 0.005),
            "max_notional_pct": config.get("risk", {}).get("max_notional_pct", 0.2),
            "taker_fee_pct": config.get("costs", {}).get("taker_fee_pct", 0.0005),
            "maker_fee_pct": config.get("costs", {}).get("maker_fee_pct", 0.0002),
            "slippage_pct": config.get("costs", {}).get("slippage_pct", 0.0002),
            "min_net_edge_pct": config.get("costs", {}).get("min_net_edge_pct", 0.0015),
            "min_score": config.get("strategy", {}).get("min_score", 8),
            "take_profit_r": config.get("strategy", {}).get("take_profit_r", 1.6),
            "atr_stop_multiplier": config.get("strategy", {}).get("atr_stop_multiplier", 1.4),
            "min_stop_loss_pct": config.get("strategy", {}).get("min_stop_loss_pct", 0.0025),
            "max_stop_loss_pct": config.get("strategy", {}).get("max_stop_loss_pct", 0.006),
            "strategy_mode": config.get("strategy", {}).get("mode", "scalp"),
            "trigger_timeframe": config.get("strategy", {}).get("trigger_timeframe", "30s"),
            "regime_timeframe": config.get("strategy", {}).get("regime_timeframe", "5m"),
            "max_hold_seconds": config.get("strategy", {}).get("max_hold_seconds", 300),
            "hard_max_hold_seconds": config.get("strategy", {}).get("hard_max_hold_seconds", 900),
            "min_hold_seconds": config.get("strategy", {}).get("min_hold_seconds", 60),
            "reversal_min_score": config.get("strategy", {}).get("reversal_min_score", 5),
            "volume_sma_period": config.get("strategy", {}).get("volume_sma_period", 20),
            "min_volume_ratio": config.get("strategy", {}).get("min_volume_ratio", 1.3),
            "require_full_alignment": config.get("strategy", {}).get("require_full_alignment", True),
            "email_notifications": {
                "enabled": email_config.enabled,
                "smtp_host": email_config.smtp_host,
                "smtp_port": email_config.smtp_port,
                "security": email_config.security,
                "sender": email_config.sender,
                "recipients": list(email_config.recipients),
                "username_env": email_config.username_env,
                "password_env": email_config.password_env,
                "password_configured": bool(os.getenv(email_config.password_env, "").strip()),
                "daily_report_enabled": email_config.daily_report_enabled,
                "daily_report_hour": email_config.daily_report_hour,
                "timezone": email_config.timezone,
            },
        }

    def save_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.running:
                raise RuntimeError("请先停止交易引擎，再修改配置")
            before = self.config_view()
            self.exchange_name = str(payload.get("exchange", "binance"))
            target = save_dashboard_config(self.config_path, payload)
            previous_notifier = self.notifier
            if not previous_notifier.close():
                LOG.warning(
                    "previous email notifier is still draining after its SMTP timeout"
                )
            replacement_notifier = self._build_notifier(self._config())
            self.notifier = replacement_notifier
            self._exchange_snapshot = {}
            self._snapshot_at = 0
            self._private_snapshot_at = 0
            self._dashboard_adapter = None
            self._dashboard_adapter_key = None
            after = self.config_view()
        tracked = ("exchange", "mode", "symbol", "environment", "base_url", "poll_seconds", "paper_equity", "max_leverage", "stop_loss_pct", "risk_per_trade", "max_notional_pct", "min_score", "take_profit_r", "volume_sma_period", "min_volume_ratio", "require_full_alignment", "email_notifications")
        changed = {
            key: {"before": before.get(key), "after": after.get(key)}
            for key in tracked
            if before.get(key) != after.get(key)
        }
        self.operation_logger.record(
            "config_change",
            "save_config",
            summary="页面保存运行配置",
            details={"changed": changed, "credential_fields_received": [key for key in ("api_key", "api_secret", "passphrase") if str(payload.get(key, "")).strip()]},
            changed_files=[str(target)],
        )
        return after

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.running:
                self.operation_logger.record("engine_start", "start", status="skipped", summary="启动请求被忽略：引擎已经运行", result={"running": True})
                return {"running": True, "message": "交易引擎已经在运行"}
            config = self._config()
            exchange_name = self._exchange(config, str(payload.get("exchange", "")))
            mode = str(config.get("mode", "paper"))
            exchange_config = config.get("exchanges", {}).get(exchange_name, {})
            self.exchange_name = exchange_name
            # The dashboard may have opened a read-only adapter while the
            # engine was stopped. Binance returns the same active listenKey
            # for the account, so close that stream before creating the engine
            # stream; closing it afterwards would invalidate the new stream.
            dashboard_adapter = self._dashboard_adapter
            if dashboard_adapter is not None:
                close_dashboard_adapter = getattr(dashboard_adapter, "close", None)
                if close_dashboard_adapter is not None:
                    close_dashboard_adapter()
                self._dashboard_adapter = None
                self._dashboard_adapter_key = None
            self.reporter = TradeReporter(
                report_directory(config, exchange_name),
                config.get("report_timezone", "Asia/Shanghai"),
            )
            try:
                self.engine = build_engine(
                    exchange_name,
                    config,
                    reporter=self.reporter,
                    notifier=self.notifier,
                )
                preflight = (
                    self.engine.prepare_live()
                    if mode == "live"
                    else {"prepared": False, "mode": mode}
                )
            except Exception as error:
                self._notify_emergency(
                    error,
                    category="engine_runtime",
                    context="实盘启动预检失败",
                    incident="start",
                    config=config,
                    exchange_name=exchange_name,
                )
                if self.engine is not None:
                    self.engine.close()
                self.reporter.close()
                self.engine = None
                self.reporter = None
                raise
            self.last_result = None
            self.last_error = ""
            self.started_at = time.time()
            self.last_cycle_at = 0.0
            self._first_cycle_logged = False
            self._last_logged_error = ""
            self._last_macro_block = ""
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._run_loop, name="btc-dashboard-engine", daemon=True)
            self._thread.start()
            self.operation_logger.record(
                "engine_start",
                "start",
                summary="交易引擎启动成功",
                details={
                    "mode": mode,
                    "exchange": exchange_name,
                    "environment": exchange_config.get("environment", ""),
                    "symbol": exchange_config.get("symbol", ""),
                    "authorization": "saved_configuration",
                },
                result={"running": True, "mode": mode, "exchange": exchange_name, "preflight": preflight},
            )
        return {"running": True, "mode": mode, "exchange": exchange_name}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._stop_event:
                self._stop_event.set()
            thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3)
        engine = self.engine
        reporter = self.reporter
        if engine is not None:
            try:
                engine.close()
            except Exception:
                LOG.exception("email notifier shutdown failed; engine stop continues")
        if reporter is not None:
            reporter.close()
        with self._lock:
            self._thread = None
            self._stop_event = None
            self.engine = None
            self.reporter = None
        self.operation_logger.record("engine_stop", "stop", summary="交易引擎已停止", result={"running": False})
        return {"running": False}

    def shutdown(self) -> None:
        """Release the dashboard-owned adapter and email worker on process exit."""

        try:
            self.stop()
        finally:
            adapter = self._dashboard_adapter
            if adapter is not None:
                close_adapter = getattr(adapter, "close", None)
                if callable(close_adapter):
                    try:
                        close_adapter()
                    except Exception:
                        LOG.exception("dashboard adapter shutdown failed")
            self._dashboard_adapter = None
            self._dashboard_adapter_key = None
            self.notifier.close()

    def restart(self, payload: dict[str, Any]) -> dict[str, Any]:
        reason = str(payload.get("reason", "页面手动重启"))
        self.operation_logger.record("engine_restart", "restart", status="started", summary="开始重启交易引擎", details={"reason": reason})
        try:
            self.stop()
            # A user may have just switched the router's outbound proxy/IP.
            # Explicit restart is the safe boundary for discarding the old
            # process-local Binance retry deadline and probing the new route.
            clear_rate_limits()
            result = self.start(payload)
            self.operation_logger.record("engine_restart", "restart", summary="交易引擎重启成功", details={"reason": reason}, result=result)
            return result
        except Exception as error:
            self.operation_logger.record("engine_restart", "restart", status="error", summary="交易引擎重启失败", details={"reason": reason}, result={"error": str(error)})
            raise

    def _run_loop(self) -> None:
        stop_event = self._stop_event
        while stop_event and not stop_event.is_set():
            rate_limit_wait = 0.0
            try:
                result = self.engine.evaluate_once()
                self.engine.resolve_emergency("engine_runtime", "cycle")
                with self._lock:
                    self.last_result = result
                    self.last_cycle_at = time.time()
                    self.last_error = ""
                    self._last_logged_error = ""
                macro_status = self.engine.macro_risk.status() if self.engine and self.engine.macro_risk else {}
                macro_block = str(macro_status.get("reason") or "") if macro_status.get("blocked") else ""
                if macro_block != self._last_macro_block:
                    if macro_block:
                        self.operation_logger.record(
                            "macro_risk",
                            "entry_block",
                            status="skipped",
                            summary="重大宏观事件或突发波动期间暂停新开仓",
                            details={"reason": macro_block, "next_event": macro_status.get("next_event")},
                            result={"running": True, "position": bool(result.position)},
                        )
                    elif self._last_macro_block:
                        self.operation_logger.record(
                            "macro_risk",
                            "entry_resume",
                            summary="宏观风险窗口结束，恢复按策略评估开仓",
                            details={"previous_reason": self._last_macro_block},
                            result={"running": True},
                        )
                    self._last_macro_block = macro_block
                if not self._first_cycle_logged:
                    self._first_cycle_logged = True
                    self.operation_logger.record(
                        "engine_cycle",
                        "first_cycle",
                        summary="引擎首次行情周期执行完成",
                        details={"status": result.status, "signal_side": result.signal.side, "signal_score": result.signal.score},
                        result={"exchange": result.exchange, "position": bool(result.position)},
                    )
            except Exception as error:  # Do not terminate the dashboard on a transient venue error.
                if isinstance(error, ApiError):
                    LOG.warning("dashboard engine cycle API failure: %s", error)
                else:
                    LOG.exception("dashboard engine cycle failed")
                notify_emergency = getattr(self.engine, "notify_emergency", None)
                if callable(notify_emergency) and not bool(
                    getattr(error, "_btc_emergency_notified", False)
                ):
                    notify_emergency(
                        error,
                        category="engine_runtime",
                        context="行情周期执行失败",
                        incident="cycle",
                    )
                if isinstance(error, ApiError) and error.rate_limited:
                    rate_limit_wait = error.retry_after_seconds
                with self._lock:
                    self.last_error = str(error)
                    self.last_cycle_at = time.time()
                if str(error) != self._last_logged_error:
                    self._last_logged_error = str(error)
                    self.operation_logger.record("engine_cycle", "cycle", status="error", summary="行情周期执行失败，引擎保持运行", result={"error": str(error)})
            poll = normalized_poll_seconds(self.engine.config.poll_seconds) if self.engine else 15
            stop_event.wait(cycle_wait_seconds(poll, rate_limit_wait))

    def _adapter(self, config: dict[str, Any], exchange_name: str) -> Any:
        expected = config.get("exchanges", {}).get(exchange_name, {})
        if self.engine and self.engine.adapter.name == exchange_name:
            settings = self.engine.adapter.settings
            same_configuration = (
                str(settings.environment).strip().lower()
                == str(expected.get("environment", "")).strip().lower()
                and str(settings.base_url).rstrip("/")
                == str(expected.get("base_url", "")).rstrip("/")
                and str(settings.symbol).strip().upper()
                == str(expected.get("symbol", "")).strip().upper()
            )
            if same_configuration:
                return self.engine.adapter
        key = (
            exchange_name,
            str(expected.get("environment", "")).strip().lower(),
            str(expected.get("base_url", "")).rstrip("/"),
            str(expected.get("symbol", "")).strip().upper(),
        )
        with self._lock:
            if self._dashboard_adapter is not None and self._dashboard_adapter_key == key:
                return self._dashboard_adapter
        adapter = build_engine(exchange_name, config, reporter=None).adapter
        with self._lock:
            self._dashboard_adapter = adapter
            self._dashboard_adapter_key = key
        return adapter

    def _market_snapshot(self, config: dict[str, Any], exchange_name: str) -> dict[str, Any]:
        snapshot_seconds = max(5.0, float(config.get("dashboard_snapshot_seconds", 15)))
        with self._snapshot_condition:
            now = time.time()
            if self._exchange_snapshot and now - self._snapshot_at < snapshot_seconds:
                return self._exchange_snapshot
            if self._snapshot_refreshing:
                self._snapshot_condition.wait(timeout=10)
                if self._exchange_snapshot:
                    return self._exchange_snapshot
            self._snapshot_refreshing = True
        try:
            adapter = self._adapter(config, exchange_name)
            fetch_snapshot = getattr(adapter, "fetch_dashboard_snapshot_nonblocking", None)
            snapshot = (
                fetch_snapshot()
                if callable(fetch_snapshot)
                else adapter.fetch_dashboard_snapshot()
            )
            self._handle_snapshot_alerts(snapshot, config, exchange_name)
            with self._snapshot_condition:
                now = time.time()
                private_now = time.monotonic()
                previous = self._exchange_snapshot
                stale_limit = max(
                    0.0,
                    float(config.get("dashboard_private_stale_seconds", 90)),
                )
                last_private_at = float(getattr(self, "_private_snapshot_at", 0.0))
                preserve_private = (
                    bool(previous.get("private_available"))
                    and bool(snapshot.get("private_transient"))
                    and last_private_at > 0
                    and private_now - last_private_at <= stale_limit
                )
                if snapshot.get("private_available"):
                    self._private_snapshot_at = private_now
                elif preserve_private:
                    error_message = str(snapshot.get("private_error") or "私有 API 暂时不可用")
                    stale = dict(previous)
                    if snapshot.get("market"):
                        stale["market"] = dict(snapshot["market"])
                    if snapshot.get("order_limits"):
                        stale["order_limits"] = dict(snapshot["order_limits"])
                    stale["private_available"] = True
                    stale["private_stale"] = True
                    stale["private_error"] = ""
                    stale["private_warning"] = (
                        "Binance 私有网络短暂波动，暂时显示最近一次成功账户快照"
                    )
                    stale["private_snapshot_at"] = last_private_at
                    stale["snapshot_error"] = error_message
                    snapshot = stale
                self._exchange_snapshot = snapshot
                self._snapshot_at = now
            return snapshot
        except Exception as error:
            self._handle_snapshot_alerts(
                {"snapshot_error": str(error)},
                config,
                exchange_name,
            )
            with self._snapshot_condition:
                if not self._exchange_snapshot:
                    raise
                now = time.time()
                private_now = time.monotonic()
                stale = dict(self._exchange_snapshot)
                stale["market"] = dict(stale.get("market") or {})
                stale["market"]["stale"] = True
                stale_limit = max(
                    0.0,
                    float(config.get("dashboard_private_stale_seconds", 90)),
                )
                last_private_at = float(getattr(self, "_private_snapshot_at", 0.0))
                preserve_private = (
                    bool(stale.get("private_available"))
                    and last_private_at > 0
                    and private_now - last_private_at <= stale_limit
                )
                stale["private_stale"] = preserve_private
                if preserve_private:
                    stale["private_warning"] = (
                        "Binance 网络短暂波动，暂时显示最近一次成功账户快照"
                    )
                    stale["private_error"] = ""
                else:
                    stale.pop("account", None)
                    stale["positions"] = []
                    stale["open_orders"] = []
                    stale["private_available"] = False
                    stale["private_warning"] = ""
                    stale["private_error"] = str(error)
                stale["snapshot_error"] = str(error)
                # Back off for one normal snapshot interval. Without advancing
                # the cache timestamp, every browser refresh immediately sends
                # another group of signed REST requests while the network is
                # still down, amplifying a short TLS interruption.
                self._exchange_snapshot = stale
                self._snapshot_at = now
                return stale
        finally:
            with self._snapshot_condition:
                self._snapshot_refreshing = False
                self._snapshot_condition.notify_all()

    def _with_live_market(
        self,
        config: dict[str, Any],
        exchange_name: str,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Overlay cached private data with the newest in-memory market tick."""

        try:
            adapter = self._adapter(config, exchange_name)
            fetch_live_snapshot = getattr(adapter, "fetch_live_dashboard_snapshot", None)
            live_snapshot = fetch_live_snapshot() if callable(fetch_live_snapshot) else None
            if live_snapshot:
                refreshed = dict(snapshot)
                refreshed.update(live_snapshot)
                if "order_limits" not in live_snapshot and "order_limits" in snapshot:
                    refreshed["order_limits"] = snapshot["order_limits"]
                refreshed.pop("private_stale", None)
                refreshed.pop("private_warning", None)
                refreshed.pop("snapshot_error", None)
                market = dict(refreshed.get("market") or {})
                market.pop("stale", None)
                refreshed["market"] = market
                return refreshed
            fetch_live_market = getattr(adapter, "fetch_live_market_snapshot", None)
            if not callable(fetch_live_market):
                return snapshot
            live_market = fetch_live_market()
            if not live_market:
                return snapshot
        except Exception as error:
            if is_rate_limit_error(error):
                self._handle_snapshot_alerts(
                    {"snapshot_error": str(error)},
                    config,
                    exchange_name,
                )
            LOG.debug("live dashboard market refresh failed; using cached price", exc_info=True)
            return snapshot
        refreshed = dict(snapshot)
        market = dict(snapshot.get("market") or {})
        market.update(live_market)
        market.pop("stale", None)
        refreshed["market"] = market
        return refreshed

    def private_check(self) -> dict[str, Any]:
        config = self._config()
        exchange_name = self._exchange(config)
        snapshot = self._market_snapshot(config, exchange_name)
        if not snapshot.get("private_available") or snapshot.get("private_stale"):
            raise RuntimeError(str(snapshot.get("private_error") or "私有 API 未连接"))
        account = snapshot.get("account") or {}
        equity_raw = str(account.get("wallet_balance_raw") or account.get("wallet_balance") or "0")
        return {
            "exchange": exchange_name,
            "equity": float(equity_raw),
            "equity_raw": equity_raw,
            "message": "私有 API 读取成功，未下单",
        }

    def email_test(self) -> dict[str, Any]:
        notifier = self.notifier
        if not notifier.config.enabled:
            raise RuntimeError("请先启用邮件通知并保存配置")
        if not notifier.ready:
            raise RuntimeError(
                f"邮件配置不完整；请检查 SMTP、发件/收件邮箱，并设置环境变量 {notifier.config.password_env}"
            )
        sent_before = int(notifier.status().get("sent_count") or 0)
        if not notifier.send_test():
            raise RuntimeError("测试邮件未进入发送队列")
        if not notifier.flush(notifier.config.timeout_seconds + 2):
            raise RuntimeError("测试邮件发送超时，请检查 SMTP 网络和服务日志")
        status = notifier.status()
        if status.get("last_error"):
            raise RuntimeError(str(status["last_error"]))
        if int(status.get("sent_count") or 0) <= sent_before:
            raise RuntimeError("测试邮件没有确认发送成功，请检查 SMTP 服务日志")
        self.operation_logger.record(
            "email_notification",
            "test",
            summary="测试邮件发送成功",
            details={"recipients_count": len(notifier.config.recipients)},
            result={"sent": True},
        )
        return {"sent": True, "recipients_count": len(notifier.config.recipients)}

    def status(self) -> dict[str, Any]:
        config = self._config()
        exchange_name = self._exchange(config)
        exchange_config = config.get("exchanges", {}).get(exchange_name, {})
        try:
            snapshot = self._market_snapshot(config, exchange_name)
            snapshot = self._with_live_market(config, exchange_name, snapshot)
        except Exception as error:
            snapshot = {"market": {}, "positions": [], "open_orders": [], "private_available": False, "private_error": str(error)}
        market = snapshot.get("market") or {}
        mark_price = market.get("mark_price") or market.get("last_price")
        positions, account = _mark_to_market_view(
            snapshot,
            str(exchange_config.get("symbol") or ""),
            float(mark_price or 0),
        )
        paper_position = (
            _position_dict(
                self.engine.position,
                float(mark_price or 0),
                mode=str(config.get("mode", "paper")),
            )
            if self.engine and self.engine.position
            else None
        )
        for position in positions:
            if snapshot.get("private_stale"):
                position["source"] = "exchange_stale"
        if paper_position:
            positions = [paper_position]
        if account:
            account["source"] = (
                "exchange_stale" if snapshot.get("private_stale") else "exchange"
            )
            account["environment"] = exchange_config.get("environment", "")
            if snapshot.get("private_snapshot_at"):
                account["snapshot_at"] = snapshot["private_snapshot_at"]
        else:
            if str(config.get("mode", "paper")).strip().lower() == "live":
                account = {
                    "wallet_balance_raw": "",
                    "available_balance_raw": "",
                    "unrealized_pnl": None,
                    "margin_balance_raw": "",
                    "source": "unavailable",
                    "environment": exchange_config.get("environment", ""),
                }
            else:
                closed_pnl = float(self.engine.session_pnl) if self.engine else 0.0
                paper_unrealized = float(paper_position.get("unrealized_pnl", 0)) if paper_position else 0.0
                paper_pnl = closed_pnl + paper_unrealized
                account = {
                    "wallet_balance": float(config.get("paper_equity", 10000)) + paper_pnl,
                    "available_balance": float(config.get("paper_equity", 10000)) + paper_pnl,
                    "unrealized_pnl": paper_unrealized,
                    "margin_balance": float(config.get("paper_equity", 10000)) + paper_pnl,
                    "source": "paper",
                    "environment": "simulation",
                }
        result = self.last_result
        signal = None
        result_view = None
        if result is not None:
            signal = result.signal
            result_view = {
                "exchange": result.exchange,
                "status": result.status,
                "position": _position_dict(result.position, float(mark_price or 0)) if result.position else None,
                "raw": result.raw,
            }
        with self._lock:
            last_error = self.last_error
            last_cycle_at = self.last_cycle_at
            started_at = self.started_at
        notifier = getattr(self, "notifier", None)
        email_status = (
            notifier.status()
            if notifier is not None
            else {
                "enabled": bool(config.get("email_notifications", {}).get("enabled", False)),
                "ready": False,
                "recipients_count": len(config.get("email_notifications", {}).get("recipients", [])),
            }
        )
        risk_guard = {"enabled": False, "blocked": False}
        if self.engine is not None:
            threshold = int(self.engine.risk.config.max_consecutive_losses)
            pause_minutes = max(
                0,
                int(getattr(self.engine.risk.config, "loss_streak_pause_minutes", 0)),
            )
            blocked = threshold > 0 and self.engine.consecutive_losses >= threshold and (
                pause_minutes <= 0 or time.time() < self.engine.cooldown_until
            )
            risk_guard = {
                "enabled": threshold > 0,
                "blocked": blocked,
                "consecutive_losses": self.engine.consecutive_losses,
                "max_consecutive_losses": threshold,
                "cooldown_until": self.engine.cooldown_until,
                "loss_streak_pause_minutes": pause_minutes,
            }
        return {
            "running": self.running,
            "exchange": exchange_name,
            "symbol": exchange_config.get("symbol", ""),
            "mode": config.get("mode", "paper"),
            "environment": exchange_config.get("environment", ""),
            "base_url": exchange_config.get("base_url", ""),
            "started_at": started_at,
            "last_cycle_at": last_cycle_at,
            "last_error": last_error,
            "connection": {
                "market": bool(snapshot.get("market")),
                "private": bool(snapshot.get("private_available"))
                and not bool(snapshot.get("private_stale")),
                "private_source": snapshot.get("private_source", "rest"),
                "private_stream": snapshot.get("private_stream", {}),
                "private_error": snapshot.get("private_error", ""),
                "private_stale": bool(snapshot.get("private_stale")),
                "private_warning": snapshot.get("private_warning", ""),
            },
            "market": snapshot.get("market", {}),
            "account": account,
            "order_sizing": _order_sizing_view(config, snapshot, account),
            "positions": positions,
            "open_orders": snapshot.get("open_orders", []),
            "signal": {"side": signal.side, "score": signal.score, "timestamp": signal.timestamp, "reasons": list(signal.reasons)} if signal else None,
            "last_result": result_view,
            "macro_risk": self.engine.macro_risk.status() if self.engine and self.engine.macro_risk else {"enabled": False, "blocked": False},
            "risk_guard": risk_guard,
            "email_notifications": email_status,
        }

    def reports(self, query: dict[str, list[str]]) -> dict[str, Any]:
        config = self._config()
        exchange_name = self._exchange(config)
        timezone_name = config.get("report_timezone", "Asia/Shanghai")
        base_directory = Path(config.get("report_dir", "reports"))
        candidates = [
            base_directory,
            base_directory / "binance-production",
            report_directory(config, exchange_name),
        ]
        directories: list[Path] = []
        seen: set[str] = set()
        for directory in candidates:
            identity = str(directory.resolve())
            if identity in seen:
                continue
            if directory != base_directory and not (directory / "trades.sqlite3").exists():
                continue
            seen.add(identity)
            directories.append(directory)
        reporters: list[TradeReporter] = []
        try:
            rows: list[dict[str, Any]] = []
            start_date = (query.get("from") or [""])[0]
            end_date = (query.get("to") or [""])[0]
            selected_exchange = (query.get("exchange") or [""])[0]
            scope = (query.get("scope") or ["all"])[0]
            for directory in directories:
                reporter = TradeReporter(directory, timezone_name)
                reporters.append(reporter)
                rows.extend(
                    reporter.query_trades(
                        start_date=start_date,
                        end_date=end_date,
                        exchange=selected_exchange,
                        scope=scope,
                        limit=20_000,
                    )
                )
            if not reporters:
                reporters.append(TradeReporter(base_directory, timezone_name))
            result = reporters[0].report_data_from_rows(rows)
            result["scope"] = scope
            result["sources"] = [str(reporter.database_path) for reporter in reporters]
            return result
        finally:
            for reporter in reporters:
                reporter.close()

    def operations(self, query: dict[str, list[str]]) -> dict[str, Any]:
        rows = self.operation_logger.query(
            start_date=(query.get("from") or [""])[0],
            end_date=(query.get("to") or [""])[0],
            event_type=(query.get("type") or [""])[0],
            keyword=(query.get("q") or [""])[0],
            limit=int((query.get("limit") or ["500"])[0]),
        )
        return {"path": str(self.operation_logger.path), "rows": rows}

    def version(self) -> dict[str, Any]:
        code_files = (
            Path(__file__),
            Path(__file__).with_name("strategy.py"),
            Path(__file__).with_name("engine.py"),
            Path(__file__).with_name("operation_log.py"),
            Path(__file__).with_name("notifications.py"),
            Path(__file__).with_name("main.py"),
        )
        latest_mtime = max((path.stat().st_mtime_ns for path in code_files if path.exists()), default=0)
        return {"version": f"{self.server_started_at:.6f}-{latest_mtime}", "code_mtime": latest_mtime}


class DashboardHandler(BaseHTTPRequestHandler):
    server: "DashboardServer"

    def log_message(self, format: str, *args: Any) -> None:
        LOG.info("dashboard %s", format % args)

    def _json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html(self) -> None:
        data = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._html()
            elif parsed.path == "/api/config":
                self._json(self.server.service.config_view())
            elif parsed.path == "/api/status":
                self._json(self.server.service.status())
            elif parsed.path == "/api/reports":
                self._json(self.server.service.reports(parse_qs(parsed.query)))
            elif parsed.path == "/api/operations":
                self._json(self.server.service.operations(parse_qs(parsed.query)))
            elif parsed.path == "/api/version":
                self._json(self.server.service.version())
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            LOG.debug("dashboard client disconnected before GET response completed")
        except Exception as error:
            LOG.exception("dashboard GET failed")
            self._json({"error": str(error)}, 400)

    def do_POST(self) -> None:
        try:
            payload = self._body()
            if self.path == "/api/config":
                self._json(self.server.service.save_config(payload))
            elif self.path == "/api/start":
                self._json(self.server.service.start(payload))
            elif self.path == "/api/stop":
                self._json(self.server.service.stop())
            elif self.path == "/api/restart":
                self._json(self.server.service.restart(payload))
            elif self.path == "/api/private-check":
                self._json(self.server.service.private_check())
            elif self.path == "/api/email-test":
                self._json(self.server.service.email_test())
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            LOG.debug("dashboard client disconnected before POST response completed")
        except Exception as error:
            LOG.exception("dashboard POST failed")
            if self.path == "/api/start":
                self.server.service.operation_logger.record("engine_start", "start", status="error", summary="交易引擎启动失败", result={"error": str(error)})
            self._json({"error": str(error)}, 400)


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], service: DashboardService) -> None:
        super().__init__(address, DashboardHandler)
        self.service = service


def run_dashboard(config_path: str, *, host: str = "127.0.0.1", port: int = 8787) -> None:
    service = DashboardService(config_path)
    server = DashboardServer((host, port), service)
    print(f"BTC Futures dashboard: http://{host}:{port}")
    print(f"Config file: {local_config_path(config_path)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    finally:
        service.shutdown()
        server.server_close()


DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Futures Control Room</title>
<style>
:root{--bg:#0b1020;--panel:#121a2d;--panel2:#18223a;--line:#2a3858;--text:#e9eefc;--muted:#8e9bb7;--cyan:#59d7d2;--green:#66d99a;--red:#ff7c8b;--yellow:#ffd166}
.secret-wrap{position:relative}.secret-wrap input{padding-right:66px}.secret-toggle{position:absolute;right:6px;top:50%;transform:translateY(-50%);padding:5px 9px;border-radius:7px;background:#263656;color:var(--text);font-size:12px}.live-phrase{margin-top:10px}.hidden{display:none!important}
.scope-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:-2px 0 16px}.scope-tabs button{background:#263656;color:var(--text);border:1px solid var(--line)}.scope-tabs button.active{background:var(--cyan);border-color:var(--cyan);color:#07131c}.source-badge{color:var(--cyan)}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#1b2d54 0,#0b1020 42%);color:var(--text);font:14px/1.5 Inter,"Microsoft YaHei",sans-serif}button,input,select{font:inherit}button{cursor:pointer;border:0;border-radius:10px;padding:10px 16px;background:var(--cyan);color:#07131c;font-weight:700}button.secondary{background:#263656;color:var(--text)}button.danger{background:var(--red);color:#230811}button:disabled{opacity:.45;cursor:not-allowed}.shell{max-width:1400px;margin:auto;padding:24px}.topbar{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:22px}.brand{display:flex;align-items:center;gap:12px}.logo{width:42px;height:42px;border-radius:14px;background:linear-gradient(135deg,#f6bf4c,#f58b46);display:grid;place-items:center;color:#241507;font-size:22px;font-weight:900}.title{font-size:22px;font-weight:800}.subtitle{color:var(--muted);font-size:12px}.status{display:flex;align-items:center;gap:9px;color:var(--muted)}.dot{width:10px;height:10px;border-radius:50%;background:var(--yellow);box-shadow:0 0 12px currentColor}.dot.ok{background:var(--green)}.dot.bad{background:var(--red)}.nav{display:flex;gap:8px;margin-bottom:16px}.nav button{background:transparent;color:var(--muted);border:1px solid transparent}.nav button.active{background:var(--panel2);border-color:var(--line);color:var(--text)}.tab{display:none}.tab.active{display:block}.grid{display:grid;gap:14px}.metrics{grid-template-columns:repeat(4,minmax(0,1fr))}.layout{grid-template-columns:1.2fr .8fr}.panel{background:rgba(18,26,45,.92);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 16px 50px #05091455}.metric{min-height:108px}.eyebrow{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.metric-value{font-size:26px;font-weight:800;margin-top:12px}.positive{color:var(--green)}.negative{color:var(--red)}.neutral{color:var(--text)}.panel h2{font-size:16px;margin:0 0 16px}.panel h3{font-size:14px;margin:18px 0 8px}.toolbar{display:flex;flex-wrap:wrap;align-items:end;gap:10px;margin-bottom:14px}.field{display:flex;flex-direction:column;gap:6px;min-width:140px}.field.grow{flex:1}.field label{color:var(--muted);font-size:12px}.field input,.field select{width:100%;background:#0c1427;border:1px solid var(--line);border-radius:9px;color:var(--text);padding:10px 11px;outline:none}.field input:focus,.field select:focus{border-color:var(--cyan)}.note{color:var(--muted);font-size:12px;margin:8px 0}.notice{padding:11px 13px;border-radius:10px;background:#2a2230;color:#ffd8a4;margin-bottom:14px}.columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;min-width:720px}th,td{padding:10px 11px;border-bottom:1px solid #24314f;text-align:left;white-space:nowrap}th{color:var(--muted);font-size:12px;font-weight:600;background:#111a2d}tr:last-child td{border-bottom:0}.empty{padding:28px;text-align:center;color:var(--muted)}.actions{display:flex;gap:8px;flex-wrap:wrap}.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-bottom:14px}.kpi{background:#0d162a;border:1px solid var(--line);border-radius:11px;padding:12px}.kpi strong{display:block;font-size:18px;margin-top:6px}.check{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px}.check input{accent-color:var(--cyan)}.footer{color:var(--muted);font-size:12px;padding:18px 0 4px}.toast{position:fixed;right:24px;bottom:24px;background:#e9eefc;color:#0b1020;border-radius:10px;padding:12px 15px;display:none;box-shadow:0 10px 30px #0008;z-index:4}.small{font-size:12px;color:var(--muted)}
@media(max-width:980px){.metrics,.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.layout,.columns{grid-template-columns:1fr}.shell{padding:15px}.topbar{align-items:flex-start;flex-direction:column}}
</style>
</head>
<body>
<div class="shell">
  <header class="topbar"><div class="brand"><div class="logo">₿</div><div><div class="title">BTC Futures Control Room</div><div class="subtitle">本地控制台 · 默认模拟盘 · 币安测试网</div></div></div><div class="status"><span id="statusDot" class="dot"></span><span id="statusText">正在连接</span></div></header>
  <nav class="nav"><button class="active" data-tab="console">控制台</button><button data-tab="config">配置</button><button data-tab="reports">交易报表</button></nav>

  <section id="console" class="tab active">
    <div class="grid metrics">
      <div class="panel metric"><div class="eyebrow">账户权益</div><div id="equity" class="metric-value">—</div><div id="balanceSub" class="small">等待账户数据</div></div>
      <div class="panel metric"><div class="eyebrow">最新成交价格</div><div id="markPrice" class="metric-value">—</div><div id="marketSub" class="small">—</div></div>
      <div class="panel metric"><div class="eyebrow">未实现盈亏</div><div id="unrealized" class="metric-value">—</div><div id="pnlSub" class="small">含当前持仓</div></div>
      <div class="panel metric"><div class="eyebrow">当前挂单</div><div id="orderCount" class="metric-value">—</div><div id="engineSub" class="small">引擎未启动</div></div>
    </div>
    <div class="grid layout" style="margin-top:14px">
      <div class="panel"><h2>运行控制</h2><div class="toolbar"><div class="field"><label>交易平台</label><select id="runExchange"><option value="binance">Binance</option><option value="okx">OKX</option><option value="gate">Gate</option></select></div><div class="actions"><button id="startBtn">启动引擎</button><button id="stopBtn" class="danger">停止引擎</button><button id="privateBtn" class="secondary">检查私有 API</button></div></div><div class="note">启动权限直接采用“配置”中已保存的交易模式和网络；修改配置前必须先停止引擎。</div><div class="note">当前模式：<span id="modeLabel">paper</span>；最近周期：<span id="lastCycle">—</span></div><div id="runtimeNotice" class="notice">请先在“配置”中保存测试网 API Key。没有 Key 时仍可运行公开行情和模拟盘。</div><h3>合约下单额度</h3><div id="orderSizingBox" class="small">正在读取交易所限制…</div><h3>策略信号</h3><div id="signalBox" class="small">暂无信号</div></div>
      <div class="panel"><h2>连接状态</h2><div id="connectionBox" class="small">正在读取…</div><h3>最近一次结果</h3><div id="resultBox" class="small">—</div><h3>风险提示</h3><div class="note">5% 是价格止损距离，不等于账户亏损 5%。引擎会按风险比例、手续费、滑点和资金费估算仓位。</div></div>
    </div>
    <div class="grid columns" style="margin-top:14px"><div class="panel"><h2>当前合约持仓</h2><div class="table-wrap"><table><thead><tr><th>来源</th><th>方向</th><th>数量</th><th>开仓价</th><th>标记价</th><th>未实现盈亏</th><th>止损</th><th>止盈</th></tr></thead><tbody id="positionsBody"><tr><td colspan="8" class="empty">暂无持仓</td></tr></tbody></table></div></div><div class="panel"><h2>当前委托订单</h2><div class="table-wrap"><table><thead><tr><th>订单号</th><th>方向</th><th>类型</th><th>状态</th><th>数量</th><th>触发价</th><th>只减仓</th></tr></thead><tbody id="ordersBody"><tr><td colspan="7" class="empty">暂无挂单</td></tr></tbody></table></div></div></div>
  </section>

  <section id="config" class="tab"><div class="panel"><h2>运行配置</h2><div class="notice">API Secret 和 OKX Passphrase 不会回显。输入框留空表示保留已保存值；配置保存到本机的 <span id="configFile">—</span>，该文件已加入 Git 忽略。</div><div class="grid columns"><div><div class="field"><label>交易平台</label><select id="cfgExchange"><option value="binance">Binance</option><option value="okx">OKX</option><option value="gate">Gate</option></select></div><div class="field" style="margin-top:12px"><label>API Key</label><input id="apiKey" autocomplete="off" placeholder="首次填写；已保存时留空保持不变"></div><div class="field" style="margin-top:12px"><label>API Secret</label><input id="apiSecret" type="password" autocomplete="new-password" placeholder="首次填写；已保存时留空保持不变"></div><div class="field" style="margin-top:12px"><label>OKX Passphrase</label><input id="passphrase" type="password" autocomplete="new-password" placeholder="仅 OKX 需要；首次填写"></div><div class="field" style="margin-top:12px"><label>交易模式</label><select id="cfgMode"><option value="paper">paper · 模拟盘</option><option value="live">live · 测试网下单</option></select></div></div><div><div class="grid columns"><div class="field"><label>合约</label><input id="symbol"></div><div class="field"><label>最大杠杆</label><input id="maxLeverage" type="number" min="1" max="125" step="1"></div><div class="field"><label>轮询秒数</label><input id="pollSeconds" type="number" min="5" step="1"></div><div class="field"><label>模拟盘权益</label><input id="paperEquity" type="number" min="0" step="100"></div><div class="field"><label>止损价格距离 (%)</label><input id="stopLoss" type="number" min="0.1" max="50" step="0.1"></div><div class="field"><label>单笔风险 (%)</label><input id="riskPerTrade" type="number" min="0.1" max="10" step="0.1"></div><div class="field"><label>理论最大可开金额使用比例 (%)</label><input id="maxNotional" type="number" min="1" max="100" step="1"></div><div class="field"><label>最低信号分</label><input id="minScore" type="number" min="1" max="8" step="1"></div><div class="field"><label>最小放量倍数</label><input id="minVolumeRatio" type="number" min="1" max="5" step="0.1"></div><div class="field"><label>参考盈利目标 (R，真实盘动态退出)</label><input id="takeProfitR" type="number" min="0.2" max="10" step="0.1"></div></div></div></div><div class="actions" style="margin-top:18px"><button id="saveBtn">保存配置</button><button id="configCheckBtn" class="secondary">检查私有 API</button></div><div id="credentialState" class="note" style="margin-top:12px">—</div></div></section>

  <section id="reports" class="tab"><div class="panel"><h2>交易报表</h2><div class="toolbar"><div class="field"><label>开始日期</label><input id="fromDate" type="date"></div><div class="field"><label>结束日期</label><input id="toDate" type="date"></div><div class="field"><label>平台</label><select id="reportExchange"><option value="">全部平台</option><option value="binance">Binance</option><option value="okx">OKX</option><option value="gate">Gate</option></select></div><button id="reportBtn">查询报表</button></div><div class="kpis"><div class="kpi"><span class="eyebrow">交易笔数</span><strong id="statTrades">0</strong></div><div class="kpi"><span class="eyebrow">胜率</span><strong id="statWinRate">0%</strong></div><div class="kpi"><span class="eyebrow">毛收益</span><strong id="statGross">0</strong></div><div class="kpi"><span class="eyebrow">总成本</span><strong id="statCost">0</strong></div><div class="kpi"><span class="eyebrow">净收益</span><strong id="statNet">0</strong></div></div><h3>逐笔明细</h3><div class="table-wrap"><table><thead><tr><th>开仓时间</th><th>平仓时间</th><th>平台</th><th>方向</th><th>开仓价</th><th>平仓价</th><th>数量</th><th>毛收益</th><th>手续费</th><th>总成本</th><th>净收益</th><th>净收益率</th><th>手续费占比</th><th>平仓原因</th></tr></thead><tbody id="tradesBody"><tr><td colspan="14" class="empty">暂无交易记录</td></tr></tbody></table></div><div class="grid columns" style="margin-top:18px"><div><h3>按日统计</h3><div class="table-wrap"><table><thead><tr><th>日期</th><th>笔数</th><th>胜率</th><th>毛收益</th><th>手续费</th><th>总成本</th><th>净收益</th></tr></thead><tbody id="dailyBody"><tr><td colspan="7" class="empty">暂无数据</td></tr></tbody></table></div></div><div><h3>按月统计</h3><div class="table-wrap"><table><thead><tr><th>月份</th><th>笔数</th><th>胜率</th><th>毛收益</th><th>手续费</th><th>总成本</th><th>净收益</th></tr></thead><tbody id="monthlyBody"><tr><td colspan="7" class="empty">暂无数据</td></tr></tbody></table></div></div></div></div></section>
  <div class="footer">本页面默认只监听 127.0.0.1。不要把它暴露到公网，也不要把 API Secret 发给任何人。</div>
</div><div id="toast" class="toast"></div>
<script>
document.querySelector('.nav').insertAdjacentHTML('beforeend','<button data-tab="operations">操作日志</button>');
 document.getElementById('reports').insertAdjacentHTML('afterend',`<section id="operations" class="tab"><div class="panel"><h2>操作日志</h2><div class="toolbar"><div class="field"><label>开始日期</label><input id="operationsFrom" type="date"></div><div class="field"><label>结束日期</label><input id="operationsTo" type="date"></div><div class="field"><label>类型</label><select id="operationsType"><option value="">全部类型</option><option value="config_change">配置变更</option><option value="strategy_change">策略变更</option><option value="code_change">代码变更</option><option value="engine_start">启动引擎</option><option value="engine_stop">停止引擎</option><option value="engine_restart">重启引擎</option><option value="dashboard_restart">页面服务重启</option><option value="engine_cycle">运行周期</option></select></div><div class="field grow"><label>关键词</label><input id="operationsQuery" placeholder="搜索摘要、文件或结果"></div><button id="operationsBtn">查询日志</button></div><div class="note">日志文件：<span id="operationPath">—</span>；API Secret、Passphrase 等敏感字段不会写入日志。</div><div class="table-wrap"><table><thead><tr><th>本地时间</th><th>类型</th><th>动作</th><th>状态</th><th>摘要</th><th>变更文件</th><th>详情/结果</th><th>来源</th></tr></thead><tbody id="operationsBody"><tr><td colspan="8" class="empty">暂无操作日志</td></tr></tbody></table></div></div></section>`);
document.querySelector('#config .actions').insertAdjacentHTML('beforebegin',`<div style="margin-top:20px;border-top:1px solid var(--line);padding-top:4px"><h3>邮件通知</h3><div class="note">最多 5 个收件邮箱。SMTP 密码只从环境变量读取，不保存到配置或日志。</div><div class="grid columns"><div class="check"><input type="checkbox" id="emailEnabled"><label for="emailEnabled">启用开仓、平仓和每日邮件</label></div><div class="check"><input type="checkbox" id="dailyEmailEnabled"><label for="dailyEmailEnabled">每天 0 点发送上一日交易报告</label></div><div class="field"><label>SMTP 服务器</label><input id="smtpHost" placeholder="例如 smtp.qq.com"></div><div class="field"><label>SMTP 端口</label><input id="smtpPort" type="number" min="1" max="65535" value="465"></div><div class="field"><label>连接安全</label><select id="emailSecurity"><option value="ssl">SSL</option><option value="starttls">STARTTLS</option><option value="plain">Plain</option></select></div><div class="field"><label>发件邮箱</label><input id="emailSender" type="email" placeholder="sender@example.com"></div><div class="field"><label>收件邮箱（逗号分隔，最多5个）</label><input id="emailRecipients" placeholder="a@example.com,b@example.com"></div><div class="field"><label>SMTP 用户名环境变量</label><input id="emailUsernameEnv" value="BTC_EMAIL_USERNAME"></div><div class="field"><label>SMTP 密码环境变量</label><input id="emailPasswordEnv" value="BTC_EMAIL_PASSWORD"></div><div class="field"><label>报告时区</label><input id="emailTimezone" value="Asia/Shanghai"></div><div class="field"><label>每日报告小时</label><input id="dailyReportHour" type="number" min="0" max="23" value="0"></div></div><div id="emailState" class="note">邮件通知未配置</div></div>`);
const $=id=>document.getElementById(id);const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pollSecondsInput=$('pollSeconds');pollSecondsInput.min='1';pollSecondsInput.title='每次策略评估完成后等待的秒数；不改变 K 线周期';pollSecondsInput.closest('.field').querySelector('label').textContent='轮询间隔（秒，最小 1）';
document.querySelector('#reports h2').insertAdjacentHTML('afterend','<div id="reportScopeButtons" class="scope-tabs"><button type="button" class="active" data-scope="all">全部交易</button><button type="button" data-scope="testnet">测试网交易</button><button type="button" data-scope="production">正式交易</button></div>');
let reportScope='all';
const reportTradeHeader=document.querySelector('#tradesBody')?.closest('table')?.querySelector('thead tr');
if(reportTradeHeader&&reportTradeHeader.children[2]){reportTradeHeader.children[2].textContent='交易平台';reportTradeHeader.children[2].insertAdjacentHTML('afterend','<th data-environment-column>平台与环境</th>')}
function installSecretToggle(id){const input=$(id);if(!input)return;input.type='password';input.autocomplete='new-password';const wrap=document.createElement('div');wrap.className='secret-wrap';input.parentNode.insertBefore(wrap,input);wrap.appendChild(input);const button=document.createElement('button');button.type='button';button.className='secret-toggle';button.textContent='显示';button.setAttribute('aria-label','显示敏感信息');button.onclick=()=>{const hidden=input.type==='password';input.type=hidden?'text':'password';button.textContent=hidden?'隐藏':'显示';button.setAttribute('aria-label',hidden?'隐藏敏感信息':'显示敏感信息')};wrap.appendChild(button)}
['apiKey','apiSecret','passphrase'].forEach(installSecretToggle);
$('cfgExchange').closest('.field').insertAdjacentHTML('afterend','<div id="environmentField" class="field" style="margin-top:12px"><label>Binance 环境</label><select id="cfgEnvironment"><option value="testnet">testnet · 模拟测试网</option><option value="production">production · 正式环境</option></select></div>');
function syncEnvironmentUi(){const isBinance=$('cfgExchange').value==='binance';$('environmentField').classList.toggle('hidden',!isBinance);const production=isBinance&&$('cfgEnvironment').value==='production';const liveOption=$('cfgMode').querySelector('option[value="live"]');liveOption.textContent=production?'live · 正式实盘（真实资金）':'live · 测试环境下单'}
const num=(v,d=2)=>Number(v||0).toLocaleString('zh-CN',{minimumFractionDigits:d,maximumFractionDigits:d});const exact=v=>{const raw=String(v??'').trim();if(!raw)return '—';const match=raw.match(/^(-?)(\d+)(?:\.(\d+))?$/);if(!match)return raw;const whole=match[2].replace(/\B(?=(\d{3})+(?!\d))/g,',');return `${match[1]}${whole}${match[3]!==undefined?'.'+match[3]:''}`};const pct=v=>`${(Number(v||0)*100).toFixed(2)}%`;const pnl=v=>`<span class="${Number(v||0)>=0?'positive':'negative'}">${num(v)}</span>`;const time=v=>v?new Date(Number(v)).toLocaleString('zh-CN',{hour12:false}):'—';
async function api(url,opt){const r=await fetch(url,{headers:{'Content-Type':'application/json'},...opt});const d=await r.json();if(!r.ok||d.error)throw Error(d.error||'请求失败');return d}
function toast(s){$('toast').textContent=s;$('toast').style.display='block';setTimeout(()=>$('toast').style.display='none',2600)}
document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.nav button').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.tab).classList.add('active');if(b.dataset.tab==='reports')loadReports()});
function fillConfig(c){$('cfgExchange').value=c.exchange;$('runExchange').value=c.exchange;$('cfgEnvironment').value=c.environment==='production'?'production':'testnet';$('cfgMode').value=c.mode;$('symbol').value=c.symbol;$('maxLeverage').value=c.max_leverage;$('pollSeconds').value=c.poll_seconds;$('paperEquity').value=c.paper_equity;$('stopLoss').value=(Number(c.stop_loss_pct)*100).toFixed(2);$('riskPerTrade').value=(Number(c.risk_per_trade)*100).toFixed(2);$('maxNotional').value=(Number(c.max_notional_pct)*100).toFixed(1);$('minScore').value=c.min_score;$('minVolumeRatio').value=c.min_volume_ratio;$('takeProfitR').value=c.take_profit_r;$('configFile').textContent=c.config_file;$('credentialState').textContent=`API Key：${c.credentials.api_key_masked||'未配置'} · Secret：${c.credentials.secret_saved?'已保存':'未配置'} · Passphrase：${c.credentials.passphrase_saved?'已保存':'未配置'} · 环境：${c.environment} · 地址：${c.base_url} · 完整多周期对齐：${c.require_full_alignment?'是':'否'}`;syncEnvironmentUi()}
const fillBaseConfig=fillConfig;fillConfig=function(c){fillBaseConfig(c);const e=c.email_notifications||{};$('emailEnabled').checked=Boolean(e.enabled);$('dailyEmailEnabled').checked=e.daily_report_enabled!==false;$('smtpHost').value=e.smtp_host||'';$('smtpPort').value=e.smtp_port||465;$('emailSecurity').value=e.security||'ssl';$('emailSender').value=e.sender||'';$('emailRecipients').value=(e.recipients||[]).join(',');$('emailUsernameEnv').value=e.username_env||'BTC_EMAIL_USERNAME';$('emailPasswordEnv').value=e.password_env||'BTC_EMAIL_PASSWORD';$('emailTimezone').value=e.timezone||'Asia/Shanghai';$('dailyReportHour').value=Number(e.daily_report_hour||0);$('emailState').textContent=`邮件：${e.enabled?'已启用':'未启用'} · 收件人 ${(e.recipients||[]).length}/5 · SMTP 密码：${e.password_configured?'环境变量已设置':'未设置'}`};
async function loadConfig(){try{fillConfig(await api('/api/config'))}catch(e){toast(e.message)}}
async function saveConfig(){try{const d=await api('/api/config',{method:'POST',body:JSON.stringify({exchange:$('cfgExchange').value,api_key:$('apiKey').value,api_secret:$('apiSecret').value,passphrase:$('passphrase').value,mode:$('cfgMode').value,symbol:$('symbol').value,poll_seconds:Number($('pollSeconds').value),paper_equity:Number($('paperEquity').value),max_leverage:Number($('maxLeverage').value),stop_loss_pct:Number($('stopLoss').value)/100,risk_per_trade:Number($('riskPerTrade').value)/100,max_notional_pct:Number($('maxNotional').value)/100,min_score:Number($('minScore').value),min_volume_ratio:Number($('minVolumeRatio').value),take_profit_r:Number($('takeProfitR').value)})});$('apiKey').value='';$('apiSecret').value='';$('passphrase').value='';fillConfig(d);toast('配置已保存，请重新检查私有 API')}catch(e){toast(e.message)}}
function renderOrderSizing(s){const x=s.order_sizing||{};if(!x.available){$('orderSizingBox').textContent=x.error||'当前平台未提供下单额度数据';return}const ok=Boolean(x.strategy_cap_meets_minimum),fallback=Boolean(x.minimum_fallback_available);const message=ok?'30%容量上限满足交易所最小订单':fallback?'30%计算结果低于最小单时，将按交易所最小单兜底':'账户理论容量不足最小单：不下单并发送邮件通知';$('orderSizingBox').innerHTML=`交易所最小：<b>${exact(x.effective_min_notional_raw)} USDT</b>（${exact(x.effective_min_quantity_raw)} BTC；MIN_NOTIONAL ${exact(x.min_notional_filter_raw)} USDT）<br>账户理论最大：<b>${exact(x.estimated_max_open_notional_raw)} USDT</b>（${exact(x.estimated_max_open_quantity_raw)} BTC，${exact(x.current_leverage_raw)}x，按数量步进取整，未扣手续费/滑点）<br>理论最大可开金额使用上限：<b class="${ok||fallback?'positive':'negative'}">${exact(x.effective_strategy_cap_raw)} USDT</b>（理论最大 ${exact(x.strategy_cap_basis_raw)} USDT × ${pct(x.max_notional_pct_raw)}；单笔风险预算 ${exact(x.risk_budget_raw)} USDT）<br><span class="${ok||fallback?'positive':'negative'}">${message}</span><br>实际下单：${esc(x.formula||'—')}`}
function renderStatus(s){const connected=s.connection.market;const privateOk=s.connection.private;const privateLabel=s.connection.private_source==='websocket'?'私有 WebSocket':'私有 API';const markRaw=s.market.mark_price_raw??s.market.mark_price;const lastRaw=s.market.last_price_raw??s.market.last_price??markRaw;const priceSource=s.market.last_price_source||s.market.price_source||`${s.exchange} 合约行情`;const tickAt=Number(s.market.last_price_timestamp||s.market.timestamp||0);const age=tickAt?Math.max(0,(Date.now()-tickAt)/1000):null;const ageText=age===null?'':` · ${age.toFixed(1)} 秒前`;$('statusDot').className=`dot ${connected?'ok':'bad'}`;$('statusText').textContent=connected?`${s.exchange} · ${s.environment} · ${s.running?'引擎运行中':'引擎已停止'}`:'行情连接失败';$('equity').textContent=exact(s.account.wallet_balance_raw??s.account.wallet_balance);$('balanceSub').textContent=`可用 ${exact(s.account.available_balance_raw??s.account.available_balance)} · 保证金 ${exact(s.account.margin_balance_raw??s.account.margin_balance)}`;$('markPrice').textContent=Number(lastRaw)>0?exact(lastRaw):'—';$('marketSub').textContent=`${s.symbol||'—'} · ${priceSource} · 标记 ${exact(markRaw)}${ageText}`;$('unrealized').innerHTML=pnl(s.account.unrealized_pnl);$('pnlSub').textContent=privateOk?'交易所实时数据':'模拟盘估算';$('orderCount').textContent=(s.open_orders||[]).length;$('engineSub').textContent=s.running?`最近周期 ${time(s.last_cycle_at)}`:'引擎未启动';$('modeLabel').textContent=s.mode;$('lastCycle').textContent=time(s.last_cycle_at);$('runtimeNotice').textContent=s.last_error||s.connection.private_error||(s.running?'引擎正在按配置运行；页面每 1 秒刷新一次':'请在“配置”中保存测试网 API Key。没有 Key 时仍可运行公开行情和模拟盘。');$('runtimeNotice').className=s.last_error?'notice negative':'notice';$('connectionBox').innerHTML=`行情：<span class="${connected?'positive':'negative'}">${connected?'正常':'失败'}</span><br>${privateLabel}：<span class="${privateOk?'positive':'neutral'}">${privateOk?'已连接':'未连接'}</span>${s.connection.private_error?`<br><span class="negative">${esc(s.connection.private_error)}</span>`:''}<br>交易对：${esc(s.symbol)}<br>模式：${esc(s.mode)}`;renderOrderSizing(s);const sig=s.signal;$('signalBox').innerHTML=sig?`方向：<b>${esc(sig.side)}</b> · 分数：<b>${sig.score}</b><br>${esc((sig.reasons||[]).join(' · '))}`:'暂无信号';$('resultBox').textContent=s.last_result?`${s.last_result.status} · ${s.last_result.position?'持仓已更新':''}`:'—';$('positionsBody').innerHTML=(s.positions||[]).length?s.positions.map(p=>`<tr><td>${esc(p.source||'exchange')}</td><td>${esc(p.side)}</td><td>${num(p.quantity,5)}</td><td>${num(p.entry_price,2)}</td><td>${num(p.mark_price,2)}</td><td>${pnl(p.unrealized_pnl)}</td><td>${num(p.stop_price,2)}</td><td>${p.take_profit_mode==='dynamic'?'动态退出':num(p.take_profit_price,2)}</td></tr>`).join(''):'<tr><td colspan="8" class="empty">暂无持仓</td></tr>';$('ordersBody').innerHTML=(s.open_orders||[]).length?s.open_orders.map(o=>`<tr><td>${esc(o.order_id||o.client_order_id||'—')}</td><td>${esc(o.side)}</td><td>${esc(o.type)}</td><td>${esc(o.status)}</td><td>${num(o.quantity,5)}</td><td>${num(o.stop_price||o.price,2)}</td><td>${o.reduce_only?'是':'否'}</td></tr>`).join(''):'<tr><td colspan="7" class="empty">暂无挂单</td></tr>'}
let statusRequestInFlight=false;async function loadStatus(){if(statusRequestInFlight)return;statusRequestInFlight=true;try{renderStatus(await api('/api/status'))}catch(e){$('statusDot').className='dot bad';$('statusText').textContent=e.message}finally{statusRequestInFlight=false}}
async function startEngine(){try{await api('/api/start',{method:'POST',body:JSON.stringify({exchange:$('runExchange').value})});toast('引擎已启动');loadStatus()}catch(e){toast(e.message)}}async function stopEngine(){try{await api('/api/stop',{method:'POST',body:'{}'});toast('引擎已停止');loadStatus()}catch(e){toast(e.message)}}async function privateCheck(){try{const d=await api('/api/private-check',{method:'POST',body:'{}'});toast(`私有 API 成功，权益 ${exact(d.equity_raw??d.equity)}`);loadStatus()}catch(e){toast(e.message)}}
function summaryRows(rows){return rows.length?rows.map(r=>`<tr><td>${esc(r.period)}</td><td>${r.trades}</td><td>${pct(r.win_rate)}</td><td>${pnl(r.gross_pnl)}</td><td>${num(r.trading_fee)}</td><td>${num(r.total_cost)}</td><td>${pnl(r.net_pnl)}</td></tr>`).join(''):'<tr><td colspan="7" class="empty">暂无数据</td></tr>'}
async function loadReports(){try{const q=new URLSearchParams();if($('fromDate').value)q.set('from',$('fromDate').value);if($('toDate').value)q.set('to',$('toDate').value);if($('reportExchange').value)q.set('exchange',$('reportExchange').value);const d=await api('/api/reports?'+q.toString()),s=d.stats;$('statTrades').textContent=s.trades;$('statWinRate').textContent=s.trades?`${(s.wins/s.trades*100).toFixed(2)}%`:'0%';$('statGross').innerHTML=pnl(s.gross_pnl);$('statCost').innerHTML=num(s.total_cost);$('statNet').innerHTML=pnl(s.net_pnl);$('tradesBody').innerHTML=d.trades.length?d.trades.map(r=>`<tr><td>${esc(r.entry_time)}</td><td>${esc(r.exit_time)}</td><td>${esc(r.exchange)}</td><td>${esc(r.side)}</td><td>${num(r.entry_price,2)}</td><td>${num(r.exit_price,2)}</td><td>${num(r.quantity,5)}</td><td>${pnl(r.gross_pnl)}</td><td>${num(r.trading_fee)}</td><td>${num(r.total_cost)}</td><td>${pnl(r.net_pnl)}</td><td>${pct(r.net_pnl_pct)}</td><td>${pct(r.fee_ratio_pct)}</td><td>${esc(r.exit_reason)}</td></tr>`).join(''):'<tr><td colspan="14" class="empty">暂无交易记录</td></tr>';$('dailyBody').innerHTML=summaryRows(d.daily);$('monthlyBody').innerHTML=summaryRows(d.monthly)}catch(e){toast(e.message)}}
$('saveBtn').onclick=saveConfig;$('configCheckBtn').onclick=privateCheck;$('startBtn').onclick=startEngine;$('stopBtn').onclick=stopEngine;$('privateBtn').onclick=privateCheck;$('reportBtn').onclick=loadReports;$('cfgExchange').onchange=()=>{$('runExchange').value=$('cfgExchange').value;if($('cfgExchange').value==='binance'&&/[-_]/.test($('symbol').value))$('symbol').value='BTCUSDT';syncEnvironmentUi()};$('cfgEnvironment').onchange=syncEnvironmentUi;$('cfgMode').onchange=syncEnvironmentUi;loadConfig();loadStatus();setInterval(loadStatus,1000);
</script>
<script>
async function loadOperations(){try{const q=new URLSearchParams();if($('operationsFrom').value)q.set('from',$('operationsFrom').value);if($('operationsTo').value)q.set('to',$('operationsTo').value);if($('operationsType').value)q.set('type',$('operationsType').value);if($('operationsQuery').value)q.set('q',$('operationsQuery').value);const d=await api('/api/operations?'+q.toString());$('operationPath').textContent=d.path;$('operationsBody').innerHTML=d.rows.length?d.rows.map(r=>{const detail=JSON.stringify({details:r.details||{},result:r.result||{}});return `<tr><td>${esc(r.local_time||r.timestamp)}</td><td>${esc(r.event_type)}</td><td>${esc(r.action)}</td><td>${esc(r.status)}</td><td>${esc(r.summary)}</td><td>${esc((r.changed_files||[]).join(', '))}</td><td title="${esc(detail)}">${esc(detail)}</td><td>${esc(r.source)}</td></tr>`}).join(''):'<tr><td colspan="8" class="empty">暂无操作日志</td></tr>'}catch(e){toast(e.message)}}
async function restartEngine(){try{await api('/api/restart',{method:'POST',body:JSON.stringify({exchange:$('runExchange').value,reason:'页面手动安全重启'})});toast('引擎已安全重启');loadStatus();loadOperations()}catch(e){toast(e.message)}}
async function saveConfigWithEmail(){try{const recipients=$('emailRecipients').value.split(/[,;\n]/).map(x=>x.trim()).filter(Boolean);if(recipients.length>5)throw Error('收件邮箱最多支持 5 个');const d=await api('/api/config',{method:'POST',body:JSON.stringify({exchange:$('cfgExchange').value,environment:$('cfgEnvironment').value,api_key:$('apiKey').value,api_secret:$('apiSecret').value,passphrase:$('passphrase').value,mode:$('cfgMode').value,symbol:$('symbol').value,poll_seconds:Number($('pollSeconds').value),paper_equity:Number($('paperEquity').value),max_leverage:Number($('maxLeverage').value),stop_loss_pct:Number($('stopLoss').value)/100,risk_per_trade:Number($('riskPerTrade').value)/100,max_notional_pct:Number($('maxNotional').value)/100,min_score:Number($('minScore').value),min_volume_ratio:Number($('minVolumeRatio').value),take_profit_r:Number($('takeProfitR').value),email_notifications:{enabled:$('emailEnabled').checked,smtp_host:$('smtpHost').value.trim(),smtp_port:Number($('smtpPort').value),security:$('emailSecurity').value,sender:$('emailSender').value.trim(),recipients,username_env:$('emailUsernameEnv').value.trim(),password_env:$('emailPasswordEnv').value.trim(),daily_report_enabled:$('dailyEmailEnabled').checked,daily_report_hour:Number($('dailyReportHour').value),timezone:$('emailTimezone').value.trim()}})});$('apiKey').value='';$('apiSecret').value='';$('passphrase').value='';['apiKey','apiSecret','passphrase'].forEach(id=>{$(id).type='password';$(id).parentElement.querySelector('.secret-toggle').textContent='显示'});fillConfig(d);toast('配置已保存；重新启动引擎后生效')}catch(e){toast(e.message)}}
async function testEmail(){try{const d=await api('/api/email-test',{method:'POST',body:'{}'});toast(`测试邮件已发送给 ${d.recipients_count} 个收件人`)}catch(e){toast(e.message)}}
let dashboardVersion='';
async function watchDashboardVersion(){try{const d=await api('/api/version');if(!dashboardVersion){dashboardVersion=d.version}else if(d.version!==dashboardVersion){location.reload()}}catch(e){}}
document.getElementById('stopBtn').insertAdjacentHTML('afterend','<button id="restartBtn" class="secondary">安全重启</button>');
document.getElementById('configCheckBtn').insertAdjacentHTML('afterend','<button id="emailTestBtn" class="secondary">发送测试邮件</button>');
document.querySelector('[data-tab="operations"]').addEventListener('click',loadOperations);
$('saveBtn').onclick=saveConfigWithEmail;$('emailTestBtn').onclick=testEmail;$('restartBtn').onclick=restartEngine;$('operationsBtn').onclick=loadOperations;loadOperations();watchDashboardVersion();setInterval(()=>{if($('operations').classList.contains('active'))loadOperations()},1000);setInterval(watchDashboardVersion,2000);
const beijingFormatter=new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',hour12:false,year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'});
const formatBeijing=value=>{if(!value)return '—';const numeric=typeof value==='number'||/^\d+$/.test(String(value));const numericValue=numeric?Number(value):0;const date=numeric?new Date(numericValue<100000000000?numericValue*1000:numericValue):new Date(value);return Number.isNaN(date.getTime())?String(value):beijingFormatter.format(date)};
const renderPositionMetrics=s=>{const table=document.querySelector('#positionsBody')?.closest('table');const header=table?.querySelector('thead tr');if(header&&!header.querySelector('[data-pnl-pct]'))header.insertAdjacentHTML('beforeend','<th data-pnl-pct>盈亏比例</th>');const rows=[...document.querySelectorAll('#positionsBody tr')];if(!(s.positions||[]).length){const empty=rows[0]?.querySelector('td');if(empty)empty.colSpan=9;return}(s.positions||[]).forEach((position,index)=>{const row=rows[index];if(!row)return;const cell=document.createElement('td');cell.innerHTML=`<span class="${Number(position.unrealized_pnl_pct||0)>=0?'positive':'negative'}">${pct(position.unrealized_pnl_pct)}</span>`;row.appendChild(cell)})};
const originalRenderStatus=renderStatus;renderStatus=function(s){originalRenderStatus(s);const cycle=formatBeijing(s.last_cycle_at);const source=s.account?.source||'unavailable';const accountSource=source==='exchange'?'交易所账户':source==='exchange_stale'?'交易所账户（最近快照）':source==='paper'?'模拟权益':'账户数据暂不可用';const venue=s.exchange==='binance'?'币安':String(s.exchange||'').toUpperCase();const network=s.environment==='production'?'正式网络':'测试网';$('lastCycle').textContent=cycle;$('engineSub').textContent=s.running?`最近周期 ${cycle}`:'引擎未启动';if(source==='unavailable'){$('equity').textContent='—';$('balanceSub').innerHTML=`<span class="source-badge">${accountSource}</span> · 等待交易所私有接口恢复`;$('unrealized').textContent='—';$('pnlSub').textContent='未使用模拟权益替代'}else{$('balanceSub').innerHTML=`<span class="source-badge">${accountSource}</span> · 可用 ${exact(s.account?.available_balance_raw??s.account?.available_balance)} · 保证金 ${exact(s.account?.margin_balance_raw??s.account?.margin_balance)}`;$('pnlSub').textContent=source==='exchange'?'交易所实时账户':source==='exchange_stale'?'交易所最近成功快照':'模拟盘估算'}document.querySelector('.subtitle').textContent=`本地控制台 · ${s.mode} · ${venue}${network}`;if(s.connection.private_stale){$('connectionBox').insertAdjacentHTML('beforeend',`<br><span class="neutral">${esc(s.connection.private_warning||'私有数据为最近成功快照')}</span>`);if(!s.last_error){$('runtimeNotice').textContent=s.connection.private_warning||'私有数据暂时陈旧，禁止新开仓';$('runtimeNotice').className='notice'}}if(!s.running&&!s.last_error&&!s.connection.private_error){$('runtimeNotice').textContent=`引擎已停止；${venue}${network}${s.connection.private?'私有 API 已连接':'私有 API 未连接'}。`}renderPositionMetrics(s)};
loadReports=async function(){try{const q=new URLSearchParams();if($('fromDate').value)q.set('from',$('fromDate').value);if($('toDate').value)q.set('to',$('toDate').value);if($('reportExchange').value)q.set('exchange',$('reportExchange').value);q.set('scope',reportScope);const d=await api('/api/reports?'+q.toString()),s=d.stats;$('statTrades').textContent=s.trades;$('statWinRate').textContent=s.trades?`${(s.wins/s.trades*100).toFixed(2)}%`:'0%';$('statGross').innerHTML=pnl(s.gross_pnl);$('statCost').innerHTML=num(s.total_cost);$('statNet').innerHTML=pnl(s.net_pnl);$('tradesBody').innerHTML=d.trades.length?d.trades.map(r=>`<tr><td>${esc(formatBeijing(r.entry_time))}</td><td>${esc(formatBeijing(r.exit_time))}</td><td>${esc(r.exchange)}</td><td>${esc(r.exchange_environment_label)}</td><td>${esc(r.side)}</td><td>${num(r.entry_price,2)}</td><td>${num(r.exit_price,2)}</td><td>${num(r.quantity,5)}</td><td>${pnl(r.gross_pnl)}</td><td>${num(r.trading_fee)}</td><td>${num(r.total_cost)}</td><td>${pnl(r.net_pnl)}</td><td>${pct(r.net_pnl_pct)}</td><td>${pct(r.fee_ratio_pct)}</td><td>${esc(r.exit_reason)}</td></tr>`).join(''):'<tr><td colspan="15" class="empty">暂无交易记录</td></tr>';$('dailyBody').innerHTML=summaryRows(d.daily);$('monthlyBody').innerHTML=summaryRows(d.monthly)}catch(e){toast(e.message)}};
document.querySelectorAll('#reportScopeButtons button').forEach(button=>{button.onclick=()=>{reportScope=button.dataset.scope;document.querySelectorAll('#reportScopeButtons button').forEach(item=>item.classList.toggle('active',item===button));loadReports()}});$('reportBtn').onclick=loadReports;loadReports();
if($('stopLoss'))$('stopLoss').insertAdjacentHTML('afterend','<div class="note">动态止损：30秒 ATR × 1.4，自动限制在 0.25%～0.60%；页面上的止损比例是 ATR 不可用时的备用值。</div>');
if($('stopLoss'))$('stopLoss').insertAdjacentHTML('afterend','<div class="note">费用保护：按吃单 0.05% 双边、滑点 0.02%估算；反向信号不会在手续费后亏损时频繁平仓，除非出现满分强反转。</div>');
const emailAwareRenderStatus=renderStatus;renderStatus=function(s){emailAwareRenderStatus(s);const e=s.email_notifications||{};if($('emailState'))$('emailState').textContent=`邮件：${e.enabled?'已启用':'未启用'} · ${e.ready?'发送就绪':'配置未就绪'} · 收件人 ${e.recipients_count||0}/5${e.last_error?' · 最近错误：'+e.last_error:''}`};
</script></body></html>"""
