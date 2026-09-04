from __future__ import annotations

import json
import logging
import os
import queue
import re
import smtplib
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as datetime_time, timedelta
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from .http_client import ApiError, is_rate_limit_error, redact_url_credentials
from .models import Position, Signal
from .reporting import TradeRecord, TradeReporter


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmailNotificationConfig:
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 465
    security: str = "ssl"
    sender: str = ""
    recipients: tuple[str, ...] = ()
    username_env: str = "BTC_EMAIL_USERNAME"
    password_env: str = "BTC_EMAIL_PASSWORD"
    timeout_seconds: float = 10.0
    daily_report_enabled: bool = True
    daily_report_hour: int = 0
    timezone: str = "Asia/Shanghai"
    state_path: str = "reports/email_notification_state.json"
    retry_seconds: int = 1_800

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        default_timezone: str = "Asia/Shanghai",
        default_state_path: str = "",
    ) -> "EmailNotificationConfig":
        raw = raw or {}
        raw_recipients = raw.get("recipients", ()) or ()
        if isinstance(raw_recipients, str):
            raw_recipients = raw_recipients.replace(";", ",").replace("\n", ",").split(",")
        recipients: list[str] = []
        for item in raw_recipients:
            address = str(item).strip()
            if not address or address in recipients:
                continue
            parsed = parseaddr(address)[1]
            if not parsed or "@" not in parsed:
                raise ValueError(f"无效的收件邮箱：{address}")
            recipients.append(parsed)
        if len(recipients) > 5:
            raise ValueError("收件邮箱最多支持 5 个")
        security = str(raw.get("security") or cls.security).strip().lower()
        if security not in {"ssl", "starttls", "plain"}:
            raise ValueError("email_notifications.security 必须是 ssl、starttls 或 plain")
        timezone_name = str(raw.get("timezone") or default_timezone or cls.timezone)
        ZoneInfo(timezone_name)
        return cls(
            enabled=bool(raw.get("enabled", cls.enabled)),
            smtp_host=str(raw.get("smtp_host") or "").strip(),
            smtp_port=max(1, min(65_535, int(raw.get("smtp_port", cls.smtp_port)))),
            security=security,
            sender=parseaddr(str(raw.get("sender") or "").strip())[1],
            recipients=tuple(recipients),
            username_env=str(raw.get("username_env") or cls.username_env).strip(),
            password_env=str(raw.get("password_env") or cls.password_env).strip(),
            timeout_seconds=max(1.0, min(60.0, float(raw.get("timeout_seconds", cls.timeout_seconds)))),
            daily_report_enabled=bool(raw.get("daily_report_enabled", cls.daily_report_enabled)),
            daily_report_hour=max(0, min(23, int(raw.get("daily_report_hour", cls.daily_report_hour)))),
            timezone=timezone_name,
            state_path=str(raw.get("state_path") or default_state_path or cls.state_path),
            retry_seconds=max(60, int(raw.get("retry_seconds", cls.retry_seconds))),
        )


@dataclass(frozen=True)
class _QueuedEmail:
    subject: str
    body: str
    event_type: str
    report_date: str = ""
    body_factory: Callable[[], str] | None = None
    incident_key: str = ""
    receipt: "_DeliveryReceipt | None" = None


@dataclass
class _DeliveryReceipt:
    completed: threading.Event = field(default_factory=threading.Event)
    delivered: bool = False
    error: str = ""


class EmailNotifier:
    """Send trade notifications without blocking the market-data loop."""

    _EMERGENCY_COOLDOWN_SECONDS = 1_800.0

    def __init__(
        self,
        config: EmailNotificationConfig,
        *,
        send_fn: Callable[[EmailMessage], Mapping[str, Any] | None] | None = None,
        now_fn: Callable[[], datetime] | None = None,
        initial_error: str = "",
    ) -> None:
        self.config = config
        self._send_fn = send_fn
        self._now_fn = now_fn or (lambda: datetime.now(ZoneInfo(self.config.timezone)))
        self._queue: queue.Queue[_QueuedEmail] = queue.Queue(maxsize=100)
        self._urgent_queue: queue.Queue[_QueuedEmail] = queue.Queue(maxsize=20)
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._closed = False
        self._worker: threading.Thread | None = None
        self._last_sent_at = ""
        self._last_error = str(initial_error)
        self._sent_count = 0
        self._last_daily_report_date = ""
        self._daily_pending: set[str] = set()
        self._last_daily_failure_at = 0.0
        self._emergency_pending: dict[str, int] = {}
        self._emergency_resolved_pending: dict[str, int] = {}
        self._last_emergency_alerts: dict[str, float] = {}
        self._load_state()
        if self.config.enabled:
            self._worker = threading.Thread(
                target=self._run,
                name="btc-email-notifier",
                daemon=True,
            )
            self._worker.start()

    @property
    def ready(self) -> bool:
        sender = self.config.sender or os.getenv(self.config.username_env, "").strip()
        password = os.getenv(self.config.password_env, "").strip()
        return bool(
            self.config.enabled
            and self.config.smtp_host
            and sender
            and self.config.recipients
            and password
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._prune_expired_emergency_alerts(self._now_fn().timestamp()):
                self._save_state_safely()
            return {
                "enabled": self.config.enabled,
                "ready": self.ready,
                "recipients_count": len(self.config.recipients),
                "daily_report_enabled": self.config.daily_report_enabled,
                "daily_report_hour": self.config.daily_report_hour,
                "timezone": self.config.timezone,
                "password_configured": bool(os.getenv(self.config.password_env, "").strip()),
                "queued": self._queue.qsize() + self._urgent_queue.qsize(),
                "sent_count": self._sent_count,
                "last_sent_at": self._last_sent_at,
                "last_error": self._last_error,
                "last_daily_report_date": self._last_daily_report_date,
                "emergency_incidents_in_cooldown": len(self._last_emergency_alerts),
            }

    def notify_emergency(
        self,
        error: BaseException | str,
        *,
        category: str,
        exchange: str,
        symbol: str,
        mode: str,
        environment: str,
        context: str,
        incident: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> bool:
        """Queue a high-priority, deduplicated operational alert."""

        selected_category = self._emergency_category(category, error)
        incident_key = self._emergency_incident_key(
            selected_category,
            exchange,
            incident,
        )
        now = self._now_fn().astimezone(ZoneInfo(self.config.timezone))
        now_timestamp = now.timestamp()
        error_text = redact_url_credentials(str(error)).strip() or type(error).__name__
        api_error = self._find_api_error(error)
        venue_name = str(exchange or "交易所").strip()
        if selected_category == "ip_restricted":
            if venue_name.lower() == "binance":
                subject = "【紧急】Binance IP 限频/封禁"
                heading = "Binance 出口 IP 已被限制或封禁"
            else:
                subject = f"【紧急】{venue_name} API 限频/IP 限制"
                heading = f"{venue_name} API 已限频或出口 IP 受限"
            handling = (
                "请手动切换路由器代理节点，确认公网出口 IP 已变化后，"
                "从页面重启引擎或运行 scripts\\restart_bot.ps1。程序不会自动切换节点。"
            )
        else:
            subject, heading, handling = {
                "order_failure": (
                    "【紧急】下单失败",
                    "交易订单执行失败",
                    "请立即登录交易所核对实际仓位、成交和挂单；确认前不要重复手工下单。",
                ),
                "engine_runtime": (
                    "【紧急】交易引擎报错",
                    "交易引擎发生异常",
                    "请立即查看交易页面和错误日志，并核对交易所仓位及保护单状态。",
                ),
            }.get(
                selected_category,
                (
                    "【紧急】交易系统异常",
                    "交易系统发生异常",
                    "请立即查看交易页面、错误日志和交易所实际状态。",
                ),
            )
        lines = [
            heading,
            f"时间：{now:%Y-%m-%d %H:%M:%S %Z}",
            f"平台/模式：{exchange} / {mode}",
            f"网络：{environment or '—'}",
            f"合约：{symbol or '—'}",
            f"阶段：{context or '运行中'}",
            f"错误：{error_text}",
        ]
        if api_error is not None:
            if api_error.status_code is not None:
                lines.append(f"HTTP 状态：{api_error.status_code}")
            if api_error.api_code not in (None, ""):
                lines.append(f"交易所错误码：{api_error.api_code}")
            if api_error.retry_after_seconds > 0:
                try:
                    retry_at = datetime.fromtimestamp(
                        api_error.retry_at,
                        tz=ZoneInfo(self.config.timezone),
                    )
                    lines.append(
                        f"预计可重试：{retry_at:%Y-%m-%d %H:%M:%S %Z}"
                        f"（约 {max(1, int(round(api_error.retry_after_seconds)))} 秒后）"
                    )
                except (OSError, OverflowError, ValueError):
                    lines.append("预计可重试：交易所返回的限频时间无法在本机解析")
        for key, value in (details or {}).items():
            lines.append(f"{key}：{redact_url_credentials(str(value))}")
        lines.append(f"处理建议：{handling}")

        with self._lock:
            if self._prune_expired_emergency_alerts(now_timestamp):
                self._save_state_safely()
            last_sent = self._last_emergency_alerts.get(incident_key, 0.0)
            pending_count = self._emergency_pending.get(incident_key, 0)
            resolved_pending_count = self._emergency_resolved_pending.get(
                incident_key,
                0,
            )
            if pending_count > resolved_pending_count or (
                last_sent > 0
                and now_timestamp - last_sent < self._EMERGENCY_COOLDOWN_SECONDS
            ):
                return False
            self._emergency_pending[incident_key] = pending_count + 1

        accepted = self._enqueue(
            _QueuedEmail(
                subject,
                "\n".join(lines),
                f"emergency_{selected_category}",
                incident_key=incident_key,
            )
        )
        if not accepted:
            with self._lock:
                remaining = self._emergency_pending.get(incident_key, 1) - 1
                if remaining > 0:
                    self._emergency_pending[incident_key] = remaining
                else:
                    self._emergency_pending.pop(incident_key, None)
        return accepted

    def resolve_emergency(
        self,
        category: str,
        exchange: str,
        incident: str = "",
    ) -> None:
        """Mark a proven-recovered incident eligible for a future fresh alert."""

        incident_key = self._emergency_incident_key(category, exchange, incident)
        with self._lock:
            state_changed = incident_key in self._last_emergency_alerts
            self._last_emergency_alerts.pop(incident_key, None)
            pending_count = self._emergency_pending.get(incident_key, 0)
            if pending_count:
                # Delivery may complete after the caller has already proven
                # recovery. Count those older deliveries so a fresh incident
                # may queue immediately without inheriting their cooldown.
                self._emergency_resolved_pending[incident_key] = pending_count
            if state_changed:
                self._save_state_safely()

    @staticmethod
    def _emergency_category(category: str, error: BaseException | str) -> str:
        if is_rate_limit_error(error):
            return "ip_restricted"
        selected = str(category or "").strip().lower()
        if selected in {"order", "order_error", "order_failure", "private_api"}:
            return "order_failure"
        if selected in {"engine", "engine_error", "engine_runtime", "runtime"}:
            return "engine_runtime"
        return selected or "engine_runtime"

    @staticmethod
    def _emergency_incident_key(category: str, exchange: str, incident: str) -> str:
        selected = str(category or "engine_runtime").strip().lower()
        venue = str(exchange or "unknown").strip().lower()
        # A Binance ban is one host/IP incident even though its countdown text
        # and the operation that discovers it can change on every request.
        if selected == "ip_restricted":
            return f"ip_restricted:{venue}"
        scope = str(incident or "default").strip().lower()
        return f"{selected}:{venue}:{scope}"

    @staticmethod
    def _find_api_error(error: BaseException | str) -> ApiError | None:
        current: BaseException | None = error if isinstance(error, BaseException) else None
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, ApiError):
                return current
            current = current.__cause__ or current.__context__
        return None

    def _prune_expired_emergency_alerts(self, now_timestamp: float) -> bool:
        cutoff = float(now_timestamp) - self._EMERGENCY_COOLDOWN_SECONDS
        expired = [
            key
            for key, sent_at in self._last_emergency_alerts.items()
            if sent_at <= cutoff
        ]
        for key in expired:
            self._last_emergency_alerts.pop(key, None)
        return bool(expired)

    def notify_open(
        self,
        position: Position,
        signal: Signal | None,
        *,
        exchange: str,
        symbol: str,
        mode: str,
    ) -> bool:
        opened_at = datetime.fromtimestamp(position.opened_at / 1000, tz=ZoneInfo(self.config.timezone))
        direction = "做多" if position.side == "long" else "做空"
        reasons = "、".join(signal.reasons) if signal and signal.reasons else "—"
        body = "\n".join(
            (
                "交易开仓通知",
                f"时间：{opened_at:%Y-%m-%d %H:%M:%S %Z}",
                f"平台/模式：{exchange} / {mode}",
                f"合约：{symbol}",
                f"方向：{direction}",
                f"开仓价格：{position.entry_price:.4f}",
                f"数量：{position.quantity:.8f}",
                f"名义价值：{position.entry_price * position.quantity:.4f}",
                f"止损价格：{position.stop_price:.4f}",
                (
                    "止盈方式：动态趋势/移动止损退出（交易所不挂固定止盈单）"
                    if mode == "live"
                    else f"止盈价格：{position.take_profit_price:.4f}"
                ),
                f"信号分数：{signal.score if signal else 0}",
                f"信号原因：{reasons}",
            )
        )
        return self._enqueue(_QueuedEmail("开仓", body, "open"))

    def notify_order_unavailable(
        self,
        *,
        exchange: str,
        symbol: str,
        mode: str,
        side: str,
        current_price: float,
        minimum_quantity: float,
        minimum_notional: float,
        available_quantity: float,
        available_notional: float,
        reason: str,
    ) -> bool:
        """Notify that even the venue minimum order cannot be opened.

        Delivery remains asynchronous and isolated from the trading loop, just
        like entry and exit notifications.
        """
        now = self._now_fn().astimezone(ZoneInfo(self.config.timezone))
        direction = "做多" if side == "long" else "做空"
        body = "\n".join(
            (
                "合约下单金额不足通知",
                f"时间：{now:%Y-%m-%d %H:%M:%S %Z}",
                f"平台/模式：{exchange} / {mode}",
                f"合约：{symbol}",
                f"方向：{direction}",
                f"当前标记价格：{current_price:.4f}",
                f"交易所最小数量：{minimum_quantity:.8f}",
                f"交易所最小名义金额：{minimum_notional:.4f}",
                f"账户当前估算最大数量：{available_quantity:.8f}",
                f"账户当前估算最大名义金额：{available_notional:.4f}",
                f"未下单原因：{reason}",
                "处理结果：本次信号已放弃，没有发送开仓订单。",
            )
        )
        return self._enqueue(_QueuedEmail("下单金额不足", body, "order_unavailable"))

    def notify_private_api_unavailable(
        self,
        *,
        exchange: str,
        symbol: str,
        mode: str,
        side: str,
        current_price: float,
        reason: str,
        retryable: bool,
        retry_seconds: float,
    ) -> bool:
        """Notify that entry was blocked by private-API availability, not funds."""
        now = self._now_fn().astimezone(ZoneInfo(self.config.timezone))
        direction = "做多" if side == "long" else "做空"
        if retryable:
            handling = (
                f"本次未发送开仓订单；信号将在接下来约 {max(1, int(round(retry_seconds)))} 秒内保留。"
                "每次重试都会重新校验信号、当前价格和账户状态，条件失效则自动放弃。"
            )
        else:
            handling = "本次未发送开仓订单，信号已放弃；请检查 API 权限或网络连接。"
        body = "\n".join(
            (
                "Binance 私有 API 暂时不可用通知",
                f"时间：{now:%Y-%m-%d %H:%M:%S %Z}",
                f"平台/模式：{exchange} / {mode}",
                f"合约：{symbol}",
                f"方向：{direction}",
                f"当前标记价格：{current_price:.4f}",
                f"未下单原因：{reason}",
                f"处理结果：{handling}",
            )
        )
        return self._enqueue(
            _QueuedEmail("私有 API 暂时不可用", body, "private_api_unavailable")
        )

    def notify_close(
        self,
        record: TradeRecord,
        *,
        daily_summary: Mapping[str, Any] | None = None,
        reporter: TradeReporter | None = None,
        report_date: date | str | None = None,
    ) -> bool:
        def build_body() -> str:
            selected_summary = daily_summary
            if selected_summary is None and reporter is not None and report_date is not None:
                selected_summary = reporter.notification_summary(
                    report_date,
                    exchange=record.exchange,
                )
            return self._close_body(record, selected_summary)

        return self._enqueue(
            _QueuedEmail(
                "平仓",
                "",
                "close",
                body_factory=build_body,
            )
        )

    def notify_reconciled_open(
        self,
        *,
        exchange: str,
        symbol: str,
        mode: str,
        environment: str,
        side: str,
        quantity: float,
        entry_price: float,
        observed_at_ms: int,
    ) -> bool:
        """Notify about a venue position missing from local engine state."""
        observed_at = datetime.fromtimestamp(observed_at_ms / 1000, tz=ZoneInfo(self.config.timezone))
        direction = "做多" if side == "long" else "做空"
        body = "\n".join(
            (
                "交易开仓通知（交易所对账兜底）",
                f"发现时间：{observed_at:%Y-%m-%d %H:%M:%S %Z}",
                f"平台/模式：{exchange} / {mode}",
                f"网络：{environment}",
                f"合约：{symbol}",
                f"方向：{direction}",
                f"开仓价格：{entry_price:.4f}",
                f"数量：{quantity:.8f}",
                f"名义价值：{entry_price * quantity:.4f}",
                "本地状态：成交后本地建仓流程未完成，已禁止继续开新仓。",
                "安全提示：请在交易所确认该仓位已有止损保护。",
            )
        )
        return self._enqueue(_QueuedEmail("开仓", body, "reconciled_open"))

    def notify_reconciled_close(
        self,
        *,
        exchange: str,
        symbol: str,
        mode: str,
        environment: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        exit_time_ms: int,
        realized_pnl: float | None,
        commission: float | None,
        commission_assets: tuple[str, ...] = (),
        source: str = "",
    ) -> bool:
        """Notify when a previously unmanaged venue position becomes flat."""
        exit_time = datetime.fromtimestamp(exit_time_ms / 1000, tz=ZoneInfo(self.config.timezone))
        direction = "做多" if side == "long" else "做空"
        direction_factor = 1 if side == "long" else -1
        estimated_gross = (exit_price - entry_price) * quantity * direction_factor
        exact_realized = realized_pnl is not None
        gross = float(realized_pnl) if exact_realized else estimated_gross
        fee = float(commission) if commission is not None else 0.0
        fee_is_quote = not commission_assets or set(commission_assets) <= {"USDT", "USDC"}
        net = gross - fee if commission is not None and fee_is_quote else None
        outcome_value = net if net is not None else gross
        outcome = "盈利" if outcome_value > 0 else "亏损" if outcome_value < 0 else "持平"
        lines = [
            "交易平仓通知（交易所对账兜底）",
            f"时间：{exit_time:%Y-%m-%d %H:%M:%S %Z}",
            f"平台/模式：{exchange} / {mode}",
            f"网络：{environment}",
            f"合约：{symbol}",
            f"方向：{direction}",
            f"开仓价格：{entry_price:.4f}",
            f"平仓价格：{exit_price:.4f}",
            f"数量：{quantity:.8f}",
            f"已实现盈亏：{gross:+.4f}（{'交易所数据' if exact_realized else '按价格估算'}）",
        ]
        if commission is not None:
            asset_label = "/".join(commission_assets) if commission_assets else "计价币"
            lines.append(f"成交手续费：{fee:.8f} {asset_label}")
        if net is not None:
            lines.append(f"扣成交手续费后：{net:+.4f}（{outcome}）")
        else:
            lines.append(f"结果：{outcome}；非计价币手续费未折算")
        lines.extend(
            (
                "平仓原因：App 手动平仓、交易所条件单或本地成交状态异常后的外部平仓",
                f"数据来源：{source or '交易所仓位与成交记录'}",
                "说明：此邮件为交易所对账兜底通知，不会反向影响下单或平仓。",
            )
        )
        return self._enqueue(_QueuedEmail("平仓", "\n".join(lines), "reconciled_close"))

    @staticmethod
    def _close_body(
        record: TradeRecord,
        daily_summary: Mapping[str, Any] | None,
    ) -> str:
        direction = "做多" if record.side == "long" else "做空"
        outcome = "盈利" if record.net_pnl > 0 else "亏损" if record.net_pnl < 0 else "持平"
        lines = [
            "交易平仓通知",
            f"平台/模式：{record.exchange} / {record.mode}",
            f"合约：{record.symbol}",
            f"方向：{direction}",
            f"开仓价格：{record.entry_price:.4f}",
            f"平仓价格：{record.exit_price:.4f}",
            f"数量：{record.quantity:.8f}",
            f"毛收益：{record.gross_pnl:+.4f}",
            f"手续费：{record.trading_fee:.4f}",
            f"滑点成本：{record.slippage_cost:.4f}",
            f"资金费：{record.funding_fee:.4f}",
            f"净收益：{record.net_pnl:+.4f}（{outcome}）",
            f"净收益率：{record.net_pnl_pct * 100:+.4f}%",
            f"持仓时间：{record.holding_minutes:.2f} 分钟",
            f"平仓原因：{record.exit_reason}",
        ]
        if daily_summary:
            daily = daily_summary.get("daily", {})
            lines.extend(
                (
                    "",
                    "当日累计",
                    f"交易笔数：{int(daily.get('trades', 0))}",
                    f"胜率：{float(daily.get('win_rate', 0.0)) * 100:.2f}%",
                    f"净收益：{float(daily.get('net_pnl', 0.0)):+.4f}",
                )
            )
        return "\n".join(lines)

    def maybe_send_daily_report(
        self,
        reporter: TradeReporter,
        *,
        exchange: str,
        symbol: str,
        mode: str,
        now: datetime | None = None,
    ) -> bool:
        if not self.config.enabled or not self.config.daily_report_enabled:
            return False
        local_now = (now or self._now_fn()).astimezone(ZoneInfo(self.config.timezone))
        scheduled = datetime.combine(
            local_now.date(),
            datetime_time(self.config.daily_report_hour),
            tzinfo=ZoneInfo(self.config.timezone),
        )
        if local_now < scheduled:
            return False
        report_date = (local_now.date() - timedelta(days=1)).isoformat()
        with self._lock:
            if report_date == self._last_daily_report_date or report_date in self._daily_pending:
                return False
            if self._last_daily_failure_at and time.monotonic() - self._last_daily_failure_at < self.config.retry_seconds:
                return False
            self._daily_pending.add(report_date)
        def build_body() -> str:
            summary = reporter.notification_summary(report_date, exchange=exchange)
            daily = summary["daily"]
            cumulative = summary["cumulative"]
            return "\n".join(
                (
                    f"{report_date} 每日交易报告",
                    f"平台/模式：{exchange} / {mode}",
                    f"合约：{symbol}",
                    "",
                    "当日统计",
                    f"交易笔数：{daily['trades']}",
                    f"盈利/亏损：{daily['wins']} / {daily['losses']}",
                    f"胜率：{daily['win_rate'] * 100:.2f}%",
                    f"毛收益：{daily['gross_pnl']:+.4f}",
                    f"总成本：{daily['total_cost']:.4f}",
                    f"净收益：{daily['net_pnl']:+.4f}",
                    "",
                    "累计统计",
                    f"交易笔数：{cumulative['trades']}",
                    f"盈利/亏损：{cumulative['wins']} / {cumulative['losses']}",
                    f"当前胜率：{cumulative['win_rate'] * 100:.2f}%",
                    f"累计净收益：{cumulative['net_pnl']:+.4f}",
                    f"累计总成本：{cumulative['total_cost']:.4f}",
                )
            )

        accepted = self._enqueue(
            _QueuedEmail(
                "每日交易报告",
                "",
                "daily",
                report_date,
                body_factory=build_body,
            )
        )
        if not accepted:
            with self._lock:
                self._daily_pending.discard(report_date)
        return accepted

    def send_test(self) -> bool:
        now = self._now_fn().astimezone(ZoneInfo(self.config.timezone))
        return self._enqueue(
            _QueuedEmail(
                "邮件通知测试",
                f"BTC 合约交易邮件通知配置测试成功。\n时间：{now:%Y-%m-%d %H:%M:%S %Z}",
                "test",
            )
        )

    def send_strategy_inspection_report(
        self,
        report: str,
        *,
        status: str,
        run_id: str,
        timeout: float | None = None,
    ) -> bool:
        """Send one hourly strategy-inspection report and confirm its delivery."""

        if not self.config.enabled:
            raise RuntimeError("邮件通知未启用")
        if self._send_fn is None and not self.ready:
            raise RuntimeError("邮件通知配置不完整")
        normalized_status = str(status).strip().lower()
        status_details = {
            "no_change": ("无策略修改", "本次巡检完成，策略保持不变"),
            "changed": ("策略已更新", "本次巡检完成，策略或代码已经修改"),
            "failed": ("执行异常", "本次巡检未完整成功，需要检查失败项"),
        }
        if normalized_status not in status_details:
            raise ValueError("status 必须是 no_change、changed 或 failed")
        normalized_run_id = str(run_id).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:+-]{1,80}", normalized_run_id):
            raise ValueError("run_id 格式无效")
        normalized_report = redact_url_credentials(str(report)).strip()
        if not normalized_report:
            raise ValueError("巡检报告不能为空")
        if "\x00" in normalized_report:
            raise ValueError("巡检报告不能包含 NUL 字符")
        if len(normalized_report.encode("utf-8")) > 64 * 1024:
            raise ValueError("巡检报告不能超过 64 KiB")

        local_now = self._now_fn().astimezone(ZoneInfo(self.config.timezone))
        subject_status, status_text = status_details[normalized_status]
        receipt = _DeliveryReceipt()
        accepted = self._enqueue(
            _QueuedEmail(
                subject=f"【整点巡检】{subject_status}｜{local_now:%Y-%m-%d %H:00}",
                body="\n".join(
                    (
                        "BTC 自动交易策略整点巡检报告",
                        f"巡检批次：{normalized_run_id}",
                        f"完成时间：{local_now:%Y-%m-%d %H:%M:%S %Z}",
                        f"执行结果：{status_text}",
                        "",
                        normalized_report,
                    )
                ),
                event_type="strategy_inspection",
                receipt=receipt,
            )
        )
        if not accepted:
            raise RuntimeError("巡检报告邮件未进入发送队列")
        wait_seconds = (
            max(0.1, float(timeout))
            if timeout is not None
            else self.config.timeout_seconds + 2.0
        )
        if not receipt.completed.wait(wait_seconds):
            raise RuntimeError("巡检报告邮件发送超时")
        if not receipt.delivered:
            detail = redact_url_credentials(receipt.error).strip()
            raise RuntimeError(
                f"巡检报告邮件发送失败：{detail}"
                if detail
                else "巡检报告邮件发送失败"
            )
        return True

    def flush(self, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while (
            self._queue.unfinished_tasks or self._urgent_queue.unfinished_tasks
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        return (
            self._queue.unfinished_tasks == 0
            and self._urgent_queue.unfinished_tasks == 0
        )

    def close(self) -> bool:
        with self._lock:
            self._closed = True
            self._stop_event.set()
        worker = self._worker
        if worker and worker is not threading.current_thread():
            worker.join(timeout=max(3.0, self.config.timeout_seconds + 2.0))
        return worker is None or not worker.is_alive()

    def _enqueue(self, item: _QueuedEmail) -> bool:
        with self._lock:
            if not self.config.enabled or self._closed:
                return False
            selected_queue = self._urgent_queue if item.incident_key else self._queue
            try:
                selected_queue.put_nowait(item)
                return True
            except queue.Full:
                self._last_error = (
                    "紧急邮件通知队列已满"
                    if item.incident_key
                    else "邮件通知队列已满"
                )
                LOG.error("email notification queue is full; dropping %s", item.event_type)
                return False

    def _run(self) -> None:
        while (
            not self._stop_event.is_set()
            or not self._urgent_queue.empty()
            or not self._queue.empty()
        ):
            selected_queue = self._urgent_queue
            try:
                item = self._urgent_queue.get_nowait()
            except queue.Empty:
                selected_queue = self._queue
                try:
                    item = self._queue.get(timeout=0.25)
                except queue.Empty:
                    continue
            try:
                self._deliver(item)
                with self._lock:
                    self._last_sent_at = datetime.now(ZoneInfo(self.config.timezone)).isoformat(timespec="seconds")
                    self._last_error = ""
                    self._sent_count += 1
                    if item.report_date:
                        self._last_daily_report_date = item.report_date
                        self._daily_pending.discard(item.report_date)
                    if item.incident_key:
                        pending_count = self._emergency_pending.get(item.incident_key, 1)
                        if pending_count > 1:
                            self._emergency_pending[item.incident_key] = pending_count - 1
                        else:
                            self._emergency_pending.pop(item.incident_key, None)
                        resolved_count = self._emergency_resolved_pending.get(
                            item.incident_key,
                            0,
                        )
                        if resolved_count > 1:
                            self._emergency_resolved_pending[item.incident_key] = (
                                resolved_count - 1
                            )
                        elif resolved_count == 1:
                            self._emergency_resolved_pending.pop(item.incident_key, None)
                        else:
                            self._last_emergency_alerts[item.incident_key] = self._now_fn().timestamp()
                    if item.report_date or item.incident_key:
                        self._save_state_safely()
                if item.receipt is not None:
                    item.receipt.delivered = True
                    item.receipt.completed.set()
                LOG.info("email notification sent type=%s recipients=%s", item.event_type, len(self.config.recipients))
            except Exception as error:  # Email outages must never stop trading.
                with self._lock:
                    self._last_error = str(error)
                    if item.report_date:
                        self._daily_pending.discard(item.report_date)
                        self._last_daily_failure_at = time.monotonic()
                    if item.incident_key:
                        pending_count = self._emergency_pending.get(item.incident_key, 1)
                        if pending_count > 1:
                            self._emergency_pending[item.incident_key] = pending_count - 1
                        else:
                            self._emergency_pending.pop(item.incident_key, None)
                        resolved_count = self._emergency_resolved_pending.get(
                            item.incident_key,
                            0,
                        )
                        if resolved_count > 1:
                            self._emergency_resolved_pending[item.incident_key] = (
                                resolved_count - 1
                            )
                        elif resolved_count == 1:
                            self._emergency_resolved_pending.pop(item.incident_key, None)
                if item.receipt is not None:
                    item.receipt.error = redact_url_credentials(str(error))
                    item.receipt.completed.set()
                    LOG.error(
                        "email notification failed type=%s error_type=%s",
                        item.event_type,
                        type(error).__name__,
                    )
                else:
                    LOG.exception("email notification failed type=%s", item.event_type)
            finally:
                selected_queue.task_done()

    def _deliver(self, item: _QueuedEmail) -> None:
        sender = self.config.sender or os.getenv(self.config.username_env, "").strip()
        username = os.getenv(self.config.username_env, "").strip() or sender
        password = os.getenv(self.config.password_env, "").strip()
        if not self.config.smtp_host:
            raise RuntimeError("未配置 SMTP 服务器")
        if not sender:
            raise RuntimeError("未配置发件邮箱")
        if not self.config.recipients:
            raise RuntimeError("未配置收件邮箱")
        body = item.body_factory() if item.body_factory is not None else item.body
        message = EmailMessage()
        message["Subject"] = item.subject
        message["From"] = sender
        message["To"] = ", ".join(self.config.recipients)
        message.set_content(body)
        if self._send_fn is not None:
            self._raise_for_refused_recipients(self._send_fn(message))
            return
        if not password:
            raise RuntimeError(f"未设置 SMTP 密码环境变量 {self.config.password_env}")
        if self.config.security == "ssl":
            with smtplib.SMTP_SSL(
                self.config.smtp_host,
                self.config.smtp_port,
                timeout=self.config.timeout_seconds,
                context=ssl.create_default_context(),
            ) as client:
                if username:
                    client.login(username, password)
                self._raise_for_refused_recipients(client.send_message(message))
            return
        with smtplib.SMTP(
            self.config.smtp_host,
            self.config.smtp_port,
            timeout=self.config.timeout_seconds,
        ) as client:
            client.ehlo()
            if self.config.security == "starttls":
                client.starttls(context=ssl.create_default_context())
                client.ehlo()
            if username:
                client.login(username, password)
            self._raise_for_refused_recipients(client.send_message(message))

    @staticmethod
    def _raise_for_refused_recipients(refused: Mapping[str, Any] | None) -> None:
        if refused:
            raise RuntimeError(f"SMTP 拒绝了 {len(refused)} 个收件人")

    def _load_state(self) -> None:
        path = Path(self.config.state_path)
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            self._last_daily_report_date = str(state.get("last_daily_report_date") or "")
            emergency = state.get("last_emergency_alerts") or {}
            if isinstance(emergency, Mapping):
                cutoff = self._now_fn().timestamp() - self._EMERGENCY_COOLDOWN_SECONDS
                self._last_emergency_alerts = {
                    str(key): float(value)
                    for key, value in emergency.items()
                    if str(key) and float(value) > cutoff
                }
        except Exception as error:
            LOG.warning("email notification state ignored: %s", error)

    def _save_state(self) -> None:
        path = Path(self.config.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "last_daily_report_date": self._last_daily_report_date,
                    "last_emergency_alerts": self._last_emergency_alerts,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _save_state_safely(self) -> bool:
        try:
            self._save_state()
            return True
        except Exception:
            LOG.exception("email notification state could not be persisted")
            return False
