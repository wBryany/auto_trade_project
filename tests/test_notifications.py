from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from btc_futures_bot.costs import CostConfig
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.http_client import ApiError
from btc_futures_bot import main as main_module
from btc_futures_bot.models import Candle, Position, Signal
from btc_futures_bot.notifications import EmailNotificationConfig, EmailNotifier
from btc_futures_bot.reporting import TradeRecord, TradeReporter
from btc_futures_bot.risk import RiskManager
from btc_futures_bot.strategy import StrategyConfig


_VALID_INSPECTION_REPORT = """巡检时间与覆盖窗口：2026-09-04 12:00–13:00（Asia/Shanghai）
Git：main 与 origin/main 同步，工作区干净
页面服务：HTTP 200，页面正常
交易引擎：running=true，last_error 为空
交易所连接：Binance 公私链路正常；行情：https://example.test/kline?api_key=visible&signature=hidden
仓位与挂单：无仓位，无挂单
新增平仓交易：0 笔，本窗口无新增样本
K线环境（1m/5m/1h/4h）：各周期数据完整，大周期偏多
多单入场质量：未出现同时满足形态与量能的入场
空单入场质量：趋势否决有效，不宜逆势追空
手续费后期望：无合格信号，不计算新的实盘期望
MFE/MAE、退出原因与回撤：无新交易，MFE/MAE 和退出原因不适用，回撤不变
本轮结论：保持当前策略
策略/代码/配置修改：无修改，样本不足以支持调参
测试结果：本轮无修改，无需运行代码测试
重启与验证：无需重启，引擎持续正常
提交与推送：无提交，无需推送
失败与用户处理：无失败，无需用户处理
""".strip()


def _email_config(tmp_path: Path) -> EmailNotificationConfig:
    return EmailNotificationConfig(
        enabled=True,
        smtp_host="smtp.example.com",
        smtp_port=465,
        security="ssl",
        sender="sender@example.com",
        recipients=("one@example.com", "two@example.com"),
        state_path=str(tmp_path / "email_state.json"),
    )


def _record() -> TradeRecord:
    opened_at = 1_767_225_600_000
    return TradeRecord.from_position(
        exchange="okx",
        symbol="BTC-USDT-SWAP",
        mode="paper",
        position=Position(
            "long",
            0.1,
            100.0,
            99.0,
            102.5,
            opened_at,
            initial_stop_price=99.0,
            best_price=103.0,
            worst_price=99.8,
        ),
        exit_price=102.0,
        exit_time_ms=opened_at + 600_000,
        exit_reason="take_profit",
        costs=CostConfig(),
        equity_before=10_000,
        signal=Signal("long", 6, opened_at, ("trend", "volume")),
    )


def test_email_config_rejects_more_than_five_recipients() -> None:
    try:
        EmailNotificationConfig.from_mapping(
            {"recipients": [f"user{index}@example.com" for index in range(6)]}
        )
    except ValueError as error:
        assert "最多支持 5 个" in str(error)
    else:
        raise AssertionError("six recipients must be rejected")


def test_runtime_dotenv_uses_configured_file_outside_working_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = tmp_path / "runtime"
    secrets = tmp_path / "secrets"
    other_working_directory = tmp_path / "elsewhere"
    runtime.mkdir()
    secrets.mkdir()
    other_working_directory.mkdir()
    env_path = secrets / "mail.env"
    env_path.write_text("TEST_RUNTIME_EMAIL_PASSWORD=loaded-from-config\n", encoding="utf-8")
    config_path = runtime / "config.local.json"
    config_path.write_text(
        json.dumps({"env_file": str(env_path)}),
        encoding="utf-8",
    )
    monkeypatch.delenv("TEST_RUNTIME_EMAIL_PASSWORD", raising=False)
    monkeypatch.chdir(other_working_directory)

    loaded = main_module.load_runtime_dotenv(str(config_path))

    assert str(env_path.resolve()) in loaded
    assert os.environ["TEST_RUNTIME_EMAIL_PASSWORD"] == "loaded-from-config"


def test_open_and_close_messages_include_trade_details(tmp_path: Path) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)
    position = Position(
        "long",
        0.1,
        100.0,
        99.0,
        102.5,
        1_767_225_600_000,
        initial_stop_price=99.0,
    )

    notifier.notify_open(
        position,
        Signal("long", 6, position.opened_at, ("trend", "volume")),
        exchange="okx",
        symbol="BTC-USDT-SWAP",
        mode="paper",
    )
    notifier.notify_close(
        _record(),
        daily_summary={"daily": {"trades": 2, "win_rate": 0.5, "net_pnl": 3.25}},
    )
    assert notifier.flush()
    notifier.close()

    assert [message["Subject"] for message in messages] == ["开仓", "平仓"]
    assert "开仓价格：100.0000" in messages[0].get_content()
    assert "止盈价格：102.5000" in messages[0].get_content()
    assert "止损价格：99.0000" in messages[0].get_content()
    assert "平仓价格：102.0000" in messages[1].get_content()
    assert "净收益" in messages[1].get_content()
    assert "当日累计" in messages[1].get_content()


def test_live_open_message_describes_dynamic_profit_exit(tmp_path: Path) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)
    position = Position("long", 0.1, 100.0, 99.0, 102.5, 1_767_225_600_000)

    notifier.notify_open(
        position,
        Signal("long", 6, position.opened_at, ("trend", "volume")),
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
    )
    assert notifier.flush()
    notifier.close()

    content = messages[0].get_content()
    assert "止盈方式：动态趋势/移动止损退出" in content
    assert "止盈价格" not in content


def test_minimum_order_unavailable_email_includes_capacity_details(tmp_path: Path) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)

    notifier.notify_order_unavailable(
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
        side="short",
        current_price=78_000.0,
        minimum_quantity=0.001,
        minimum_notional=78.0,
        available_quantity=0.0005,
        available_notional=39.0,
        reason="账户当前理论可开金额低于交易所最小订单",
    )
    assert notifier.flush()
    notifier.close()

    assert len(messages) == 1
    assert messages[0]["Subject"] == "下单金额不足"
    assert "交易所最小数量：0.00100000" in messages[0].get_content()
    assert "账户当前估算最大名义金额：39.0000" in messages[0].get_content()
    assert "没有发送开仓订单" in messages[0].get_content()


def test_private_api_unavailable_email_is_not_labeled_as_insufficient_funds(
    tmp_path: Path,
) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)

    notifier.notify_private_api_unavailable(
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
        side="long",
        current_price=78_524.5,
        reason="network error GET /fapi/v1/openOrders: TLS handshake timed out",
        retryable=True,
        retry_seconds=32.4,
    )
    assert notifier.flush()
    notifier.close()

    assert len(messages) == 1
    assert messages[0]["Subject"] == "私有 API 暂时不可用"
    content = messages[0].get_content()
    assert "下单金额不足" not in content
    assert "接下来约 32 秒内保留" in content
    assert "重新校验信号、当前价格和账户状态" in content


def test_emergency_messages_cover_engine_order_and_ip_failures(tmp_path: Path) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)

    notifier.notify_emergency(
        RuntimeError("strategy evaluation exploded"),
        category="engine_runtime",
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
        environment="production",
        context="engine evaluate_once",
        incident="engine-cycle",
    )
    notifier.notify_emergency(
        RuntimeError("entry market order rejected"),
        category="order_failure",
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
        environment="production",
        context="entry place_market_order",
        incident="entry-order",
    )
    notifier.notify_emergency(
        RuntimeError('HTTP 418: {"code":-1003,"msg":"IP banned"}'),
        category="ip_restricted",
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
        environment="production",
        context="fetch candles",
        incident="binance-ip-limit",
    )
    assert notifier.flush()
    notifier.close()

    assert len(messages) == 3
    combined = [f"{message['Subject']}\n{message.get_content()}" for message in messages]
    assert all("紧急" in message["Subject"] for message in messages)
    assert "引擎" in combined[0]
    assert "strategy evaluation exploded" in combined[0]
    assert "engine evaluate_once" in combined[0]
    assert "下单" in combined[1]
    assert "entry market order rejected" in combined[1]
    assert "entry place_market_order" in combined[1]
    assert "IP" in combined[2]
    assert "HTTP 418" in combined[2]
    assert "fetch candles" in combined[2]
    assert "切换节点" in combined[2]
    assert "重启" in combined[2]
    assert all("binance" in content for content in combined)
    assert all("BTCUSDT" in content for content in combined)
    assert all("live" in content for content in combined)
    assert all("production" in content for content in combined)


def test_non_binance_rate_limit_alert_uses_the_actual_exchange_name(tmp_path: Path) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)

    assert notifier.notify_emergency(
        RuntimeError("HTTP 429: too many requests"),
        category="engine_runtime",
        exchange="okx",
        symbol="BTC-USDT-SWAP",
        mode="paper",
        environment="demo",
        context="fetch candles",
    )
    assert notifier.flush()
    notifier.close()

    assert len(messages) == 1
    assert "OKX" in messages[0]["Subject"].upper()
    assert "Binance" not in messages[0]["Subject"]


def test_entry_reconciliation_alert_distinguishes_filled_order_and_ip_ban(tmp_path: Path) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)
    context = dict(category="entry_reconciliation", exchange="binance", symbol="BTCUSDT",
                   mode="live", environment="production", incident="managed_entry_fill:77",
                   context="受保护仓位的开仓成交明细对账")
    try:
        assert notifier.notify_emergency(RuntimeError("fill metadata unavailable"), **context)
        assert notifier.notify_emergency(ApiError("IP banned", status_code=418), **context)
        assert notifier.flush()
    finally:
        notifier.close()
    assert len(messages) == 2
    assert "已开仓" in messages[0]["Subject"]
    assert "下单失败" not in messages[0]["Subject"]
    assert "硬止损" in messages[0].get_content()
    assert "IP" in messages[1]["Subject"]


def test_emergency_message_is_deduplicated_until_incident_is_resolved(tmp_path: Path) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)
    context = {
        "category": "engine_runtime",
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "mode": "live",
        "environment": "production",
        "context": "engine evaluate_once",
        "incident": "engine-cycle",
    }

    assert notifier.notify_emergency(RuntimeError("first cycle failure"), **context)
    assert not notifier.notify_emergency(RuntimeError("same incident, changed detail"), **context)
    assert notifier.flush()
    assert len(messages) == 1

    notifier.resolve_emergency(
        "engine_runtime",
        "binance",
        incident="engine-cycle",
    )
    assert notifier.notify_emergency(RuntimeError("failure after recovery"), **context)
    assert notifier.flush()
    notifier.close()

    assert len(messages) == 2
    assert "first cycle failure" in messages[0].get_content()
    assert "failure after recovery" in messages[1].get_content()


def test_emergency_recovery_during_delivery_does_not_restore_old_cooldown(
    tmp_path: Path,
) -> None:
    delivery_started = threading.Event()
    release_delivery = threading.Event()
    messages = []

    def blocking_send(message: object) -> None:
        delivery_started.set()
        assert release_delivery.wait(timeout=2)
        messages.append(message)

    notifier = EmailNotifier(_email_config(tmp_path), send_fn=blocking_send)
    context = {
        "category": "engine_runtime",
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "mode": "live",
        "environment": "production",
        "context": "engine evaluate_once",
        "incident": "engine-cycle",
    }

    assert notifier.notify_emergency(RuntimeError("first failure"), **context)
    assert delivery_started.wait(timeout=2)
    notifier.resolve_emergency("engine_runtime", "binance", incident="engine-cycle")
    assert notifier.notify_emergency(
        RuntimeError("new failure while recovered delivery is finishing"),
        **context,
    )
    release_delivery.set()
    assert notifier.flush()
    notifier.close()
    assert len(messages) == 2


def test_closed_notifier_rejects_new_emails(tmp_path: Path) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)
    assert notifier.close()

    assert not notifier.send_test()
    assert messages == []


def test_strategy_inspection_report_confirms_its_own_delivery(tmp_path: Path) -> None:
    messages = []
    now = datetime(2026, 9, 4, 13, 7, 8, tzinfo=ZoneInfo("Asia/Shanghai"))
    notifier = EmailNotifier(
        _email_config(tmp_path),
        send_fn=messages.append,
        now_fn=lambda: now,
    )

    report = "\n".join(
        (
            "BTC 自动交易策略整点巡检报告",
            "巡检批次：2026-09-04T13:00:00+08:00",
            "完成时间：2026-09-04 13:07:08 CST",
            "执行结果：本次巡检完成，策略保持不变",
            "",
            _VALID_INSPECTION_REPORT,
        )
    )
    assert notifier.send_strategy_inspection_report(
        report,
        status="no_change",
        run_id="2026-09-04T13:00:00+08:00",
        timeout=1,
    )
    notifier.close()

    assert len(messages) == 1
    assert messages[0]["Subject"] == "【整点巡检】无策略修改｜2026-09-04 13:00"
    content = messages[0].get_content()
    assert content.count("BTC 自动交易策略整点巡检报告") == 1
    assert content.count("巡检批次：") == 1
    assert "巡检批次：2026-09-04T13:00:00+08:00" in content
    assert "巡检时间与覆盖窗口：" in content
    assert "策略/代码/配置修改：无修改" in content
    assert "api_key=[REDACTED]" in content
    assert "signature=[REDACTED]" in content
    assert "visible" not in content
    assert "hidden" not in content


def test_strategy_inspection_report_rejects_the_accidental_draft(
    tmp_path: Path,
) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)
    report = "\n".join(
        (
            "BTC 自动交易整点巡检报告",
            "巡检批次：2026-09-04T17:00:00+08:00",
            "巡检执行时间：2026-09-04 17:01–17:09（Asia/Shanghai）",
            "... wait wrong?",
        )
    )

    try:
        notifier.send_strategy_inspection_report(
            report,
            status="no_change",
            run_id="2026-09-04T17:00:00+08:00",
            timeout=1,
        )
    except ValueError as error:
        assert "草稿占位" in str(error)
    else:
        raise AssertionError("an unfinished draft must be rejected before delivery")
    finally:
        notifier.close()

    assert messages == []


def test_strategy_inspection_report_requires_every_contract_field(
    tmp_path: Path,
) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)
    incomplete = _VALID_INSPECTION_REPORT.replace(
        "测试结果：本轮无修改，无需运行代码测试\n",
        "",
    )

    try:
        notifier.send_strategy_inspection_report(
            incomplete,
            status="no_change",
            run_id="2026-09-04T17:00:00+08:00",
            timeout=1,
        )
    except ValueError as error:
        assert "测试结果" in str(error)
    else:
        raise AssertionError("a report with a missing contract field must be rejected")
    finally:
        notifier.close()

    assert messages == []


def test_strategy_inspection_report_rejects_a_mismatched_embedded_run_id(
    tmp_path: Path,
) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)
    report = "\n".join(
        (
            "BTC 自动交易策略整点巡检报告",
            "巡检批次：2026-09-04T16:00:00+08:00",
            _VALID_INSPECTION_REPORT,
        )
    )

    try:
        notifier.send_strategy_inspection_report(
            report,
            status="no_change",
            run_id="2026-09-04T17:00:00+08:00",
            timeout=1,
        )
    except ValueError as error:
        assert "批次与 run_id 不一致" in str(error)
    else:
        raise AssertionError("a stale embedded run id must be rejected")
    finally:
        notifier.close()

    assert messages == []


def test_strategy_inspection_report_surfaces_smtp_failure(
    tmp_path: Path,
    caplog,
) -> None:
    secret_detail = "smtp unavailable with SECRET_REPORT_BODY"

    def fail_send(message: object) -> None:
        raise OSError(secret_detail)

    notifier = EmailNotifier(_email_config(tmp_path), send_fn=fail_send)
    try:
        notifier.send_strategy_inspection_report(
            _VALID_INSPECTION_REPORT,
            status="failed",
            run_id="2026-09-04T14:00:00+08:00",
            timeout=1,
        )
    except RuntimeError as error:
        assert "smtp unavailable" in str(error)
    else:
        raise AssertionError("SMTP failure must be returned to the inspection task")
    finally:
        notifier.close()
    assert secret_detail not in caplog.text
    assert "SECRET_REPORT_BODY" not in caplog.text
    assert "error_type=OSError" in caplog.text


def test_strategy_inspection_report_times_out_per_message(tmp_path: Path) -> None:
    release = threading.Event()

    def blocked_send(message: object) -> None:
        release.wait(timeout=2)

    notifier = EmailNotifier(_email_config(tmp_path), send_fn=blocked_send)
    try:
        notifier.send_strategy_inspection_report(
            _VALID_INSPECTION_REPORT,
            status="no_change",
            run_id="2026-09-04T15:00:00+08:00",
            timeout=0.01,
        )
    except RuntimeError as error:
        assert "超时" in str(error)
    else:
        raise AssertionError("blocked delivery must time out")
    finally:
        release.set()
        notifier.close()


def test_strategy_inspection_report_rejects_partial_recipient_delivery(
    tmp_path: Path,
    caplog,
) -> None:
    rejected_address = "two@example.com"

    def partially_rejected(message: object) -> dict[str, tuple[int, bytes]]:
        return {rejected_address: (550, b"mailbox unavailable")}

    notifier = EmailNotifier(_email_config(tmp_path), send_fn=partially_rejected)
    try:
        notifier.send_strategy_inspection_report(
            _VALID_INSPECTION_REPORT,
            status="no_change",
            run_id="2026-09-04T16:00:00+08:00",
            timeout=1,
        )
    except RuntimeError as error:
        assert "拒绝了 1 个收件人" in str(error)
    else:
        raise AssertionError("partial recipient refusal must fail delivery confirmation")
    finally:
        notifier.close()
    assert rejected_address not in caplog.text


def test_emergency_delivery_survives_state_persistence_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)

    def fail_state_write() -> None:
        raise OSError("state directory is read only")

    monkeypatch.setattr(notifier, "_save_state", fail_state_write)
    assert notifier.notify_emergency(
        RuntimeError("engine failed"),
        category="engine_runtime",
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
        environment="production",
        context="engine cycle",
    )
    assert notifier.flush()
    notifier.close()

    assert len(messages) == 1
    assert "engine failed" in messages[0].get_content()


def test_emergency_message_handles_unrepresentable_retry_timestamp(tmp_path: Path) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)
    error = ApiError(
        "HTTP 429",
        status_code=429,
        retry_at=float("inf"),
        api_code=-1003,
    )

    assert notifier.notify_emergency(
        error,
        category="engine_runtime",
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
        environment="production",
        context="engine cycle",
    )
    assert notifier.flush()
    notifier.close()

    assert len(messages) == 1
    assert "无法在本机解析" in messages[0].get_content()


def test_reconciled_live_position_messages_include_exchange_pnl(tmp_path: Path) -> None:
    messages = []
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)
    notifier.notify_reconciled_open(
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
        environment="production",
        side="long",
        quantity=0.001,
        entry_price=78_000.0,
        observed_at_ms=1_767_225_600_000,
    )
    notifier.notify_reconciled_close(
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
        environment="production",
        side="long",
        quantity=0.001,
        entry_price=78_000.0,
        exit_price=79_000.0,
        exit_time_ms=1_767_225_660_000,
        realized_pnl=1.0,
        commission=0.0314,
        commission_assets=("USDT",),
        source="Binance userTrades",
    )
    assert notifier.flush()
    notifier.close()

    assert [message["Subject"] for message in messages] == ["开仓", "平仓"]
    assert "交易所对账兜底" in messages[0].get_content()
    assert "已实现盈亏：+1.0000（交易所数据）" in messages[1].get_content()
    assert "扣成交手续费后：+0.9686（盈利）" in messages[1].get_content()


def test_midnight_report_sends_previous_day_only_once(tmp_path: Path) -> None:
    class Reporter:
        calls = 0

        def notification_summary(self, report_date: str, *, exchange: str = "") -> dict[str, object]:
            self.calls += 1
            stats = {
                "trades": 3,
                "wins": 2,
                "losses": 1,
                "win_rate": 2 / 3,
                "gross_pnl": 12.0,
                "total_cost": 4.0,
                "net_pnl": 8.0,
            }
            return {"date": report_date, "daily": stats, "cumulative": stats}

    messages = []
    reporter = Reporter()
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=messages.append)
    now = datetime(2026, 8, 5, 0, 0, 5, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert notifier.maybe_send_daily_report(
        reporter,
        exchange="okx",
        symbol="BTC-USDT-SWAP",
        mode="paper",
        now=now,
    )
    assert notifier.flush()
    assert not notifier.maybe_send_daily_report(
        reporter,
        exchange="okx",
        symbol="BTC-USDT-SWAP",
        mode="paper",
        now=now,
    )
    notifier.close()

    assert reporter.calls == 1
    assert len(messages) == 1
    assert messages[0]["Subject"] == "每日交易报告"
    assert "2026-08-04 每日交易报告" in messages[0].get_content()
    assert "当前胜率：66.67%" in messages[0].get_content()


def test_reporter_notification_summary_includes_daily_and_cumulative_stats(tmp_path: Path) -> None:
    reporter = TradeReporter(tmp_path)
    reporter.record_trade(_record())

    summary = reporter.notification_summary("2026-01-01", exchange="okx")
    reporter.close()

    assert summary["daily"]["trades"] == 1
    assert summary["daily"]["wins"] == 1
    assert summary["daily"]["win_rate"] == 1.0
    assert summary["cumulative"]["net_pnl"] == summary["daily"]["net_pnl"]


def test_engine_emits_open_and_close_notifications() -> None:
    class Adapter:
        name = "okx"
        settings = SimpleNamespace(symbol="BTC-USDT-SWAP")

        @staticmethod
        def fetch_candles(interval: str, limit: int) -> list[Candle]:
            price = {"5m": 100.0, "1m": 101.0, "1h": 99.0}[interval]
            step = {"5m": 300_000, "1m": 60_000, "1h": 3_600_000}[interval]
            return [
                Candle(0, price, price + 0.2, price - 0.2, price, 10.0),
                Candle(step, price, price + 0.2, price - 0.2, price, 10.0),
            ]

    class Strategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("long", 6, 1, ("fixed",))

    class Notifier:
        opened = []
        closed = []

        def notify_open(self, position: Position, signal: Signal, **context: object) -> None:
            self.opened.append((position, signal, context))

        def notify_close(self, record: TradeRecord, **context: object) -> None:
            self.closed.append((record, context))

    notifier = Notifier()
    engine = TradingEngine(
        Adapter(),
        Strategy(),
        RiskManager(),
        EngineConfig(mode="paper"),
        notifier=notifier,
    )

    result = engine.evaluate_once()
    engine._close_paper_position(102.0, "test_exit")

    assert result.position is not None
    assert len(notifier.opened) == 1
    assert len(notifier.closed) == 1
    assert notifier.closed[0][0].exit_price == 102.0


def test_email_exceptions_cannot_cancel_engine_entry_or_exit() -> None:
    class Adapter:
        name = "okx"
        settings = SimpleNamespace(symbol="BTC-USDT-SWAP")

        @staticmethod
        def fetch_candles(interval: str, limit: int) -> list[Candle]:
            price = {"5m": 100.0, "1m": 101.0, "1h": 99.0}[interval]
            step = {"5m": 300_000, "1m": 60_000, "1h": 3_600_000}[interval]
            return [
                Candle(0, price, price + 0.2, price - 0.2, price, 10.0),
                Candle(step, price, price + 0.2, price - 0.2, price, 10.0),
            ]

    class Strategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("long", 6, 1, ("fixed",))

    class ExplodingNotifier:
        @staticmethod
        def notify_open(*args: object, **kwargs: object) -> None:
            raise RuntimeError("smtp open failure")

        @staticmethod
        def notify_close(*args: object, **kwargs: object) -> None:
            raise RuntimeError("smtp close failure")

    engine = TradingEngine(
        Adapter(),
        Strategy(),
        RiskManager(),
        EngineConfig(mode="paper"),
        notifier=ExplodingNotifier(),
    )

    result = engine.evaluate_once()
    assert result.position is not None
    engine._close_paper_position(102.0, "test_exit")

    assert engine.position is None
    assert engine.session_pnl != 0.0


def test_daily_summary_runs_on_email_worker_thread(tmp_path: Path) -> None:
    class Reporter:
        worker_names = []

        def notification_summary(self, report_date: str, *, exchange: str = "") -> dict[str, object]:
            self.worker_names.append(threading.current_thread().name)
            stats = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "gross_pnl": 0.0,
                "total_cost": 0.0,
                "net_pnl": 0.0,
            }
            return {"date": report_date, "daily": stats, "cumulative": stats}

    reporter = Reporter()
    notifier = EmailNotifier(_email_config(tmp_path), send_fn=lambda message: None)

    notifier.maybe_send_daily_report(
        reporter,
        exchange="okx",
        symbol="BTC-USDT-SWAP",
        mode="paper",
        now=datetime(2026, 8, 5, 0, 0, 5, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert notifier.flush()
    notifier.close()

    assert reporter.worker_names == ["btc-email-notifier"]


def test_slow_smtp_does_not_delay_engine_entry(tmp_path: Path) -> None:
    release = threading.Event()

    def slow_send(message: object) -> None:
        release.wait(timeout=2)

    class Adapter:
        name = "okx"
        settings = SimpleNamespace(symbol="BTC-USDT-SWAP")

        @staticmethod
        def fetch_candles(interval: str, limit: int) -> list[Candle]:
            price = {"5m": 100.0, "1m": 101.0, "1h": 99.0}[interval]
            step = {"5m": 300_000, "1m": 60_000, "1h": 3_600_000}[interval]
            return [
                Candle(0, price, price + 0.2, price - 0.2, price, 10.0),
                Candle(step, price, price + 0.2, price - 0.2, price, 10.0),
            ]

    class Strategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("long", 6, 1, ("fixed",))

    notifier = EmailNotifier(_email_config(tmp_path), send_fn=slow_send)
    engine = TradingEngine(
        Adapter(),
        Strategy(),
        RiskManager(),
        EngineConfig(mode="paper"),
        notifier=notifier,
    )

    started = time.perf_counter()
    result = engine.evaluate_once()
    elapsed = time.perf_counter() - started
    release.set()
    notifier.flush()
    notifier.close()

    assert result.position is not None
    assert elapsed < 0.25


def test_invalid_email_config_does_not_prevent_engine_build(tmp_path: Path) -> None:
    class Adapter:
        name = "okx"
        settings = SimpleNamespace(symbol="BTC-USDT-SWAP")

    original_make_adapter = main_module.make_adapter
    main_module.make_adapter = lambda *args, **kwargs: Adapter()
    reporter = TradeReporter(tmp_path)
    try:
        engine = main_module.build_engine(
            "okx",
            {
                "mode": "paper",
                "report_dir": str(tmp_path),
                "report_timezone": "Asia/Shanghai",
                "strategy": {},
                "risk": {},
                "costs": {},
                "account": {"max_leverage": 3},
                "exchanges": {"okx": {"symbol": "BTC-USDT-SWAP"}},
                "email_notifications": {
                    "enabled": True,
                    "recipients": [f"user{index}@example.com" for index in range(6)],
                },
            },
            reporter=reporter,
        )
    finally:
        main_module.make_adapter = original_make_adapter
        reporter.close()

    assert engine.notifier is not None
    status = engine.notifier.status()
    assert status["enabled"] is False
    assert "隔离禁用" in status["last_error"]
