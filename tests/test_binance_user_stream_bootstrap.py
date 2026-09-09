from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import websocket
import websocket._app as websocket_app

import btc_futures_bot.binance_user_stream as stream_module
from btc_futures_bot.binance_user_stream import BinanceUserDataStream
from btc_futures_bot.http_client import ApiError


def make_stream(loader):
    return BinanceUserDataStream(
        "BTCUSDT", "production", "https://fapi.binance.com",
        lambda: "test-key-never-sent", loader,
    )


def seed(balance):
    return {"account": {"availableBalance": str(balance)}, "positions": [],
            "orders": [], "algo_orders": []}


@pytest.mark.parametrize("status", [418, 429, 500])
def test_real_websocket_app_tears_down_after_failed_open_without_sock_error(monkeypatch, status):
    original = ApiError("bootstrap account request failed", status_code=status,
                        retry_at=time.time() + 90 if status != 500 else 0,
                        api_code=-1003 if status != 500 else None)
    stream = make_stream(Mock(side_effect=original))
    stream.seed_snapshot(seed("stale"))
    monkeypatch.setattr(stream, "start", lambda **_: False)
    transports = []
    sdk_errors = []

    class Transport:
        def __init__(self, *_args, **_kwargs):
            self.sock = object()
            self.closed = False
            self.close_frame = None
            transports.append(self)

        def settimeout(self, _):
            pass

        def connect(self, *_args, **_kwargs):
            pass

        def close(self, *_args, **_kwargs):
            self.closed = True
            self.sock = None

    monkeypatch.setattr(websocket_app, "WebSocket", Transport)

    def on_error(app, error):
        sdk_errors.append(error)
        stream._on_error(app, error)

    app = websocket.WebSocketApp(
        "wss://example.invalid/ws/test", on_open=stream._on_open,
        on_error=on_error, on_close=stream._on_close,
    )

    class Dispatcher:
        def read(self, raw_socket, _read, _check):
            # Exercise the real SDK's post-on_open app.sock.sock access.
            # No real socket or selector is constructed by this test.
            assert raw_socket is app.sock.sock
            assert app.keep_running is False

    monkeypatch.setattr(app, "create_dispatcher", lambda *_: Dispatcher())
    app.run_forever(ping_interval=0)
    assert sdk_errors == []
    assert len(transports) == 1 and transports[0].closed
    assert app.sock is None
    assert not stream.healthy()
    assert not stream._ready_event.is_set()
    with pytest.raises(ApiError) as first:
        stream.snapshot()
    assert first.value is not original
    assert first.value.status_code == status
    assert first.value.api_code == original.api_code
    assert first.value.retry_at == original.retry_at
    first.value._btc_emergency_notified = True
    with pytest.raises(ApiError) as second:
        stream.snapshot()
    assert second.value is not first.value
    assert not hasattr(second.value, "_btc_emergency_notified")
    assert second.value.__cause__ is original


def test_secondary_teardown_error_cannot_replace_bootstrap_api_error(monkeypatch):
    error = ApiError("HTTP 418 account bootstrap banned", status_code=418,
                     retry_at=time.time()+120, api_code=-1003)
    stream = make_stream(Mock(side_effect=error))
    monkeypatch.setattr(stream, "start", lambda **_: False)
    app = SimpleNamespace(keep_running=True, close=Mock())
    stream._on_open(app)
    stream._on_error(app, AttributeError("'NoneType' object has no attribute 'sock'"))
    with pytest.raises(ApiError, match="bootstrap banned") as caught:
        stream.snapshot()
    assert caught.value.status_code == 418
    assert caught.value.retry_at == error.retry_at
    app.close.assert_not_called()
    status = stream.status()
    assert status["last_error"] == str(error)
    assert status["retry_at"] == error.retry_at
    assert 0 < status["retry_after_seconds"] <= 120
    json.dumps(status)  # Public status must not contain exception objects.


def test_failed_bootstrap_reconnect_waits_for_rate_limit_then_loads_fresh_snapshot(monkeypatch):
    wall = [1000.0]
    error = ApiError("HTTP 418 account bootstrap banned", status_code=418,
                     retry_at=1050.0, api_code=-1003)
    loader = Mock(side_effect=[error, seed("fresh")])
    stream = make_stream(loader)
    stream.seed_snapshot(seed("stale"))
    monkeypatch.setattr(stream, "start", lambda **_: False)
    ensure_key = Mock(return_value="test-key")
    monkeypatch.setattr(stream, "_ensure_listen_key", ensure_key)
    monkeypatch.setattr(stream_module, "time", SimpleNamespace(
        time=lambda: wall[0], monotonic=lambda: wall[0],
    ))
    waits, attempts, recovered = [], [], []

    class Stop:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, delay):
            if self.stopped:
                return True
            waits.append(delay)
            wall[0] += delay
            return False

    stream._stop = Stop()

    class App:
        def __init__(self, _url, **callbacks):
            self.callbacks = callbacks
            self.keep_running = True

        def run_forever(self, **_):
            attempts.append(wall[0])
            self.callbacks["on_open"](self)
            if len(attempts) == 1:
                assert not stream.healthy()
                with pytest.raises(ApiError):
                    stream.snapshot()
            else:
                recovered.append(stream.snapshot())
                stream._stop.stopped = True

    monkeypatch.setattr(stream_module, "websocket", SimpleNamespace(WebSocketApp=App))
    stream._run()
    assert waits == [50.0]
    assert attempts == [1000.0, 1050.0]
    assert ensure_key.call_count == loader.call_count == 2
    assert recovered[0]["account"]["availableBalance"] == "fresh"
    assert stream.status()["last_error"] == ""
    assert stream._last_exception is None
    # run_forever returned: even a correctly seeded cache is unusable offline.
    with pytest.raises(RuntimeError, match="not ready"):
        stream.snapshot()


def test_stop_interrupts_long_rate_limit_wait_without_connecting(monkeypatch):
    stream = make_stream(Mock())
    stream._retry_at = time.time()+3600
    ensure_key = Mock(side_effect=AssertionError("must not request REST during ban"))
    monkeypatch.setattr(stream, "_ensure_listen_key", ensure_key)
    entered = threading.Event()
    stopped = threading.Event()

    class Stop:
        is_set = stopped.is_set

        def wait(self, timeout):
            entered.set()
            return stopped.wait(timeout)

    stream._stop = Stop()
    thread = threading.Thread(target=stream._run, daemon=True)
    thread.start()
    try:
        assert entered.wait(timeout=1)
        stopped.set()
        thread.join(timeout=1)
        assert not thread.is_alive()
        ensure_key.assert_not_called()
    finally:
        stopped.set()
        thread.join(timeout=1)


def test_keepalive_respects_existing_rate_limit_deadline(monkeypatch):
    stream = make_stream(Mock())
    stream._listen_key = "test-key"
    stream._retry_at = 1090.0
    monkeypatch.setattr(stream_module, "time", SimpleNamespace(
        time=lambda: 1000.0, monotonic=lambda: 1000.0,
    ))
    request = Mock(side_effect=AssertionError("must not renew during ban"))
    monkeypatch.setattr(stream, "_listen_key_request", request)
    waits = []
    stream._stop = SimpleNamespace(wait=lambda delay: waits.append(delay) or True)
    stream._keepalive_loop()
    assert waits == [90.0]
    request.assert_not_called()


def test_expired_authentication_cannot_read_old_snapshot(monkeypatch):
    stream = make_stream(lambda: seed("current"))
    monkeypatch.setattr(stream, "start", lambda **_: False)
    app = SimpleNamespace(keep_running=True)
    stream._on_open(app)
    assert stream.snapshot()["account"]["availableBalance"] == "current"
    stream.process_message(json.dumps({"e": "listenKeyExpired"}))
    assert not stream.healthy()
    assert not stream._ready_event.is_set()
    with pytest.raises(RuntimeError, match="listenKey expired"):
        stream.snapshot()


def test_stop_during_bootstrap_does_not_publish_snapshot(monkeypatch):
    stream = make_stream(Mock())
    stream.seed_snapshot(seed("stale"))
    monkeypatch.setattr(stream, "start", lambda **_: False)

    def loader():
        stream._stop.set()
        return seed("new-but-stopped")

    stream._snapshot_loader = loader
    app = SimpleNamespace(keep_running=True)
    stream._on_open(app)
    assert not app.keep_running
    assert not stream.healthy()
    with pytest.raises(RuntimeError, match="not ready"):
        stream.snapshot()
