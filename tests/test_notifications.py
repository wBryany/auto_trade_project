from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from btc_futures_bot.costs import CostConfig
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot import main as main_module
from btc_futures_bot.models import Candle, Position, Signal
from btc_futures_bot.notifications import EmailNotificationConfig, EmailNotifier
from btc_futures_bot.reporting import TradeRecord, TradeReporter
from btc_futures_bot.risk import RiskManager
from btc_futures_bot.strategy import StrategyConfig


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
