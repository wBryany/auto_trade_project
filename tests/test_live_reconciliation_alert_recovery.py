from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import btc_futures_bot.engine as engine_module
from btc_futures_bot.dashboard import DashboardService
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.models import Candle, Position, Signal
from btc_futures_bot.notifications import EmailNotificationConfig, EmailNotifier
from btc_futures_bot.risk import RiskManager
from btc_futures_bot.strategy import StrategyConfig


ERROR = "Binance private WebSocket is not ready"
CANDLE = Candle(60_000, 100.0, 100.0, 100.0, 100.0, 1.0)


class PrivateAdapter:
    name = "binance"
    settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

    def __init__(self) -> None:
        self.available = False
        self.position_reads = 0
        self.remote_position = None
        self.orders = Mock()

    def fetch_candles(self, _timeframe, _limit):
        return [CANDLE, CANDLE]

    def fetch_live_position(self):
        self.position_reads += 1
        if not self.available:
            raise RuntimeError(ERROR)
        return self.remote_position

    def fetch_mark_price(self):
        return 100.0

    def fetch_equity(self):
        if not self.available:
            raise RuntimeError(ERROR)
        return 1_000.0

    def place_market_order(self, request):
        return self.orders(request)


def make_engine(adapter, notifier=None, *, mode="live"):
    strategy = SimpleNamespace(
        config=StrategyConfig(),
        evaluate=lambda _: Signal("flat", 0, CANDLE.timestamp),
    )
    return TradingEngine(
        adapter, strategy, RiskManager(),
        EngineConfig(mode=mode, poll_seconds=1, live_reconciliation_seconds=5),
        notifier=notifier,
    )


def make_dashboard(engine):
    service = DashboardService.__new__(DashboardService)
    service.engine = engine
    service._lock = threading.RLock()
    service.last_result = None
    service.last_cycle_at = 0.0
    service.last_error = ""
    service._last_logged_error = ""
    service._first_cycle_logged = False
    service._last_macro_block = ""
    service.operation_logger = SimpleNamespace(record=Mock())
    return service


def patch_engine_clock(monkeypatch, clock, sleep=time.sleep):
    # Replace only the engine's module reference; notifier threads keep a real
    # monotonic clock and real waits, so flush timeouts remain bounded.
    monkeypatch.setattr(
        engine_module, "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=time.time, sleep=sleep),
    )


@pytest.mark.parametrize("runner", ["dashboard", "cli"])
def test_outage_alert_stays_deduplicated_until_actual_reconciliation_recovers(
    tmp_path: Path, monkeypatch, runner: str,
) -> None:
    messages = []
    notifier = EmailNotifier(
        EmailNotificationConfig(
            enabled=True, smtp_host="smtp.example.com", sender="sender@example.com",
            recipients=("recipient@example.com",),
            state_path=str(tmp_path / "email-state.json"),
        ),
        send_fn=messages.append,
    )
    adapter = PrivateAdapter()
    engine = make_engine(adapter, notifier)
    service = make_dashboard(engine)
    times = [100.0, 101.0, 105.0, 106.0, 110.0, 115.0]
    clock = [times[0]]
    observed = []
    delivered = []
    resolve = Mock(wraps=engine.resolve_emergency)
    monkeypatch.setattr(engine, "resolve_emergency", resolve)

    class EndLoop(BaseException):
        pass

    class Stop:
        index = 0

        def is_set(self):
            return self.index >= len(times)

        def wait(self, timeout):
            assert timeout == 1.0
            assert notifier.flush()
            delivered.append(len(messages))
            observed.append((service.last_error, service._last_logged_error))
            self.index += 1
            if self.is_set():
                if runner == "cli":
                    raise EndLoop()
                return True
            clock[0] = times[self.index]
            adapter.available = clock[0] == 110.0
            return False

    stop = Stop()
    service._stop_event = stop
    patch_engine_clock(monkeypatch, clock, sleep=stop.wait)
    try:
        if runner == "dashboard":
            service._run_loop()
        else:
            with pytest.raises(EndLoop):
                engine.run_forever()
        # 101 and 106 completed market-only cycles, but neither retried the
        # failed private phase. 110 proves recovery; 115 is a fresh incident.
        assert adapter.position_reads == 4
        assert delivered == [1, 1, 1, 1, 1, 2]
        assert resolve.call_args_list == [call("engine_runtime", "cycle")]
        if runner == "dashboard":
            assert observed == [(ERROR, ERROR)] * 4 + [("", ""), (ERROR, ERROR)]
            errors = [call for call in service.operation_logger.record.call_args_list
                      if call.kwargs.get("status") == "error"]
            assert len(errors) == 2
            assert service.last_result.status == "no_action"
    finally:
        notifier.close()


def test_reconciliation_failure_raises_and_requires_complete_state_match_to_recover(monkeypatch):
    adapter = PrivateAdapter()
    engine = make_engine(adapter)
    engine.position = Position(
        "long", 1.0, 100.0, 99.0, 103.0, 1,
        initial_stop_price=99.0, stop_order_id="hard-stop",
    )
    clock = [100.0]
    patch_engine_clock(monkeypatch, clock)
    with pytest.raises(RuntimeError, match=ERROR):
        engine._reconcile_binance_live_position_if_due(CANDLE)
    assert engine.has_pending_live_reconciliation_error

    # A healthy private stream alone does not prove position reconciliation.
    adapter.available = True
    adapter.remote_position = {"side": "long", "quantity": 2.0}
    clock[0] = 101.0
    engine._reconcile_binance_live_position_if_due(CANDLE)
    assert adapter.position_reads == 1
    assert engine.has_pending_live_reconciliation_error
    clock[0] = 105.0
    with pytest.raises(RuntimeError, match="quantity differs"):
        engine._reconcile_binance_live_position_if_due(CANDLE)
    assert engine.has_pending_live_reconciliation_error

    adapter.remote_position["quantity"] = 1.0
    clock[0] = 110.0
    engine._reconcile_binance_live_position_if_due(CANDLE)
    assert not engine.has_pending_live_reconciliation_error
    assert adapter.position_reads == 3
    assert engine.position.stop_order_id == "hard-stop"
    adapter.orders.assert_not_called()


def test_skipped_failed_reconciliation_does_not_skip_local_position_management(monkeypatch):
    adapter = PrivateAdapter()
    engine = make_engine(adapter)
    engine.position = Position(
        "long", 1.0, 100.0, 99.0, 103.0, 1,
        initial_stop_price=99.0, stop_order_id="hard-stop",
    )
    manage = Mock(return_value=None)
    monkeypatch.setattr(engine, "_manage_live_position", manage)
    monkeypatch.setattr(engine, "_reconcile_managed_entry_fill", Mock())
    clock = [100.0]
    patch_engine_clock(monkeypatch, clock)
    with pytest.raises(RuntimeError, match=ERROR):
        engine.evaluate_once()
    clock[0] = 101.0
    result = engine.evaluate_once()
    manage.assert_called_once()
    assert result.position is engine.position
    assert result.position.stop_order_id == "hard-stop"
    assert engine.has_pending_live_reconciliation_error
    assert adapter.position_reads == 1


def test_skipped_failed_reconciliation_still_requires_available_equity_for_entry(monkeypatch):
    adapter = PrivateAdapter()
    engine = make_engine(adapter)
    clock = [100.0]
    patch_engine_clock(monkeypatch, clock)
    with pytest.raises(RuntimeError, match=ERROR):
        engine.evaluate_once()
    engine.strategy.evaluate = lambda _: Signal("long", 6, CANDLE.timestamp)
    monkeypatch.setattr(engine.risk, "observed_range_allows_entry", lambda *_: True)
    clock[0] = 101.0
    result = engine.evaluate_once()
    assert result.status in {"private_api_retry_scheduled", "private_api_unavailable"}
    assert engine.position is None
    assert engine.has_pending_live_reconciliation_error
    assert adapter.position_reads == 1
    adapter.orders.assert_not_called()


@pytest.mark.parametrize("mode,exchange", [("paper", "binance"), ("live", "okx")])
def test_other_modes_do_not_require_binance_reconciliation_for_cycle_recovery(mode, exchange):
    adapter = PrivateAdapter()
    adapter.name = exchange
    engine = make_engine(adapter, mode=mode)
    result = engine.evaluate_once()
    assert result.status == "no_action"
    assert not engine.has_pending_live_reconciliation_error
    assert adapter.position_reads == 0
