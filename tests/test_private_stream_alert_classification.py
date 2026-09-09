"""Preserve bootstrap rate limits through the real adapter and engine alert path."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import btc_futures_bot.engine as engine_module
from btc_futures_bot.binance_user_stream import BinanceUserDataStream
from btc_futures_bot.dashboard import DashboardService
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.exchanges.binance import BinanceAdapter
from btc_futures_bot.http_client import ApiError
from btc_futures_bot.models import Candle, Signal
from btc_futures_bot.notifications import EmailNotificationConfig, EmailNotifier
from btc_futures_bot.risk import RiskManager
from btc_futures_bot.strategy import StrategyConfig


@pytest.mark.parametrize("runner", ["dashboard", "cli"])
def test_bootstrap_ban_retains_wait_and_actionable_alert_in_engine_loop(tmp_path, monkeypatch, runner):
    deadline = time.time() + 120
    original = ApiError("HTTP 418: IP banned", status_code=418, api_code=-1003, retry_at=deadline)
    loader = Mock(side_effect=original)
    stream = BinanceUserDataStream("BTCUSDT", "production", "https://fapi.binance.com", lambda: "test", loader)
    # No socket, account, order or email traffic is permitted in this test.
    monkeypatch.setattr(stream, "start", lambda **_: False)
    callback_socket = SimpleNamespace(keep_running=True, close=Mock())
    stream._on_open(callback_socket)
    stream._on_error(callback_socket, AttributeError("'NoneType' object has no attribute 'sock'"))

    class CachedBinance(BinanceAdapter):
        def __init__(self):
            self.settings = SimpleNamespace(name="binance", symbol="BTCUSDT", environment="production")
            self._private_stream = stream

        def fetch_candles(self, *_):
            return [Candle(60_000, 100, 101, 99, 100, 1), Candle(120_000, 100, 101, 99, 100, 1)]

        def place_market_order(self, *_):
            raise AssertionError("Unavailable private account must not submit an order")

    messages = []
    notifier = EmailNotifier(EmailNotificationConfig(
        enabled=True, smtp_host="smtp.example.com", sender="sender@example.com",
        recipients=("recipient@example.com",), state_path=str(tmp_path / "email.json"),
    ), send_fn=messages.append)
    engine = TradingEngine(
        CachedBinance(),
        SimpleNamespace(config=StrategyConfig(), evaluate=lambda _: Signal("flat", 0, 60_000)),
        RiskManager(), EngineConfig(mode="live", poll_seconds=1), notifier=notifier,
    )
    service = DashboardService.__new__(DashboardService)
    service.engine = engine
    service._lock = threading.RLock()
    service.last_error = service._last_logged_error = ""
    service.last_cycle_at = 0
    service.operation_logger = SimpleNamespace(record=Mock())
    waits = []

    class Stop:
        done = False

        def is_set(self):
            return self.done

        def wait(self, seconds):
            waits.append(seconds)
            self.done = True
            if runner == "cli":
                raise KeyboardInterrupt
            return True

    stop = Stop()
    service._stop_event = stop
    monkeypatch.setattr(engine_module, "time", SimpleNamespace(
        monotonic=time.monotonic, time=time.time, sleep=stop.wait,
    ))
    try:
        if runner == "dashboard":
            service._run_loop()
            assert service.last_error == str(original)
        else:
            with pytest.raises(KeyboardInterrupt):
                engine.run_forever()
        assert waits == [30.0]  # Existing bounded engine wait, not a fresh 1s error loop.
        assert notifier.flush()
        assert len(messages) == 1
        assert messages[0]["Subject"] == "【紧急】Binance IP 限频/封禁"
        body = messages[0].get_content()
        assert "HTTP 状态：418" in body
        assert "交易所错误码：-1003" in body
        assert "预计可重试：" in body
        assert "sock" not in body
        assert "限频等待时间后重试" in body
        assert engine.position is None
        assert engine.has_pending_live_reconciliation_error
        loader.assert_called_once()
    finally:
        notifier.close()
