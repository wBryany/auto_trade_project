import json
from types import SimpleNamespace

import pytest

from btc_futures_bot import okx_ws
from btc_futures_bot.okx_ws import OkxWebSocketConnection


def connection(**overrides):
    return OkxWebSocketConnection("wss://wspap.okx.com:8443/ws/v5/public", **{
        "on_open": lambda socket: None, "on_message": lambda socket, payload: None,
        "on_disconnect": lambda: None, **overrides,
    })


def test_missing_dependency_is_rest_fallback(monkeypatch):
    monkeypatch.setattr(okx_ws, "websocket", None)
    stream = connection()
    assert stream.start() is False
    assert not stream.status()["connected"]
    assert "not installed" in stream.status()["last_error"]


def test_close_before_start_cannot_resurrect():
    stream = connection()
    stream.close()
    assert stream.start() is False
    assert stream.status()["closed"]


def test_transport_pong_not_forwarded_to_application():
    seen = []
    stream = connection(on_message=lambda socket, payload: (seen.append(payload), stream._stop.set()))
    messages = iter(["pong", json.dumps({"event": "subscribe"})])
    socket = SimpleNamespace(recv=lambda: next(messages), send=lambda _: None)
    stream._receive(socket)
    assert seen == [{"event": "subscribe"}]


def test_text_ping_and_missing_pong_are_bounded(monkeypatch):
    now = [0.0]
    sent = []
    monkeypatch.setattr(okx_ws.time, "monotonic", lambda: now[0])

    def timeout():
        now[0] += 16
        raise okx_ws.websocket.WebSocketTimeoutException()

    stream = connection()
    with pytest.raises(TimeoutError):
        stream._receive(SimpleNamespace(recv=timeout, send=sent.append))
    assert sent == ["ping"]


def test_disconnect_invalidates_cache_and_error_does_not_leak_payload(monkeypatch):
    invalidations = []
    closed = []
    delays = []
    socket = SimpleNamespace(settimeout=lambda _: None, close=lambda: closed.append(True))
    monkeypatch.setattr(okx_ws.websocket, "create_connection", lambda *a, **k: socket)
    stream = connection(on_open=lambda _: (_ for _ in ()).throw(ValueError("secret-key-in-raw-error")),
                        on_disconnect=lambda: invalidations.append(True))

    def wait(delay):
        delays.append(delay)
        stream._stop.set()

    monkeypatch.setattr(stream._stop, "wait", wait)
    stream._run()
    assert invalidations == [True]
    assert closed == [True]
    assert len(delays) == 1 and 1.6 <= delays[0] <= 2.4
    assert not stream.status()["connected"]
    assert "secret-key" not in stream.status()["last_error"]


def test_transport_rejects_nonobject_data():
    stream = connection()
    with pytest.raises(ValueError):
        stream._receive(SimpleNamespace(recv=lambda: "[]"))
