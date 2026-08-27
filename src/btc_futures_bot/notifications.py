from __future__ import annotations

import json
import logging
import os
import queue
import smtplib
import ssl
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

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


class EmailNotifier:
    """Send trade notifications without blocking the market-data loop."""

    def __init__(
        self,
        config: EmailNotificationConfig,
        *,
        send_fn: Callable[[EmailMessage], None] | None = None,
        now_fn: Callable[[], datetime] | None = None,
        initial_error: str = "",
    ) -> None:
        self.config = config
        self._send_fn = send_fn
        self._now_fn = now_fn or (lambda: datetime.now(ZoneInfo(self.config.timezone)))
        self._queue: queue.Queue[_QueuedEmail] = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._last_sent_at = ""
        self._last_error = str(initial_error)
        self._sent_count = 0
        self._last_daily_report_date = ""
        self._daily_pending: set[str] = set()
        self._last_daily_failure_at = 0.0
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
            return {
                "enabled": self.config.enabled,
                "ready": self.ready,
                "recipients_count": len(self.config.recipients),
                "daily_report_enabled": self.config.daily_report_enabled,
                "daily_report_hour": self.config.daily_report_hour,
                "timezone": self.config.timezone,
                "password_configured": bool(os.getenv(self.config.password_env, "").strip()),
                "queued": self._queue.qsize(),
                "sent_count": self._sent_count,
                "last_sent_at": self._last_sent_at,
                "last_error": self._last_error,
                "last_daily_report_date": self._last_daily_report_date,
            }

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

    def flush(self, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def close(self) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker and worker is not threading.current_thread():
            worker.join(timeout=3)

    def _enqueue(self, item: _QueuedEmail) -> bool:
        if not self.config.enabled:
            return False
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            with self._lock:
                self._last_error = "邮件通知队列已满"
            LOG.error("email notification queue is full; dropping %s", item.event_type)
            return False

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
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
                        self._save_state()
                LOG.info("email notification sent type=%s recipients=%s", item.event_type, len(self.config.recipients))
            except Exception as error:  # Email outages must never stop trading.
                with self._lock:
                    self._last_error = str(error)
                    if item.report_date:
                        self._daily_pending.discard(item.report_date)
                        self._last_daily_failure_at = time.monotonic()
                LOG.exception("email notification failed type=%s", item.event_type)
            finally:
                self._queue.task_done()

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
            self._send_fn(message)
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
                client.send_message(message)
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
            client.send_message(message)

    def _load_state(self) -> None:
        path = Path(self.config.state_path)
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            self._last_daily_report_date = str(state.get("last_daily_report_date") or "")
        except Exception as error:
            LOG.warning("email notification state ignored: %s", error)

    def _save_state(self) -> None:
        path = Path(self.config.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"last_daily_report_date": self._last_daily_report_date},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
