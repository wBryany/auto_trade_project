from __future__ import annotations

import base64
import hashlib
import hmac
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import btc_futures_bot.okx_user_stream as private_module
from btc_futures_bot.okx_user_stream import OkxPrivateStream


SYMBOL = "BTC-USDT-SWAP"


class Socket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(json.loads(payload))

    def close(self):
        self.closed = True


class Connection:
    def __init__(self, url, *, on_open, on_message, on_disconnect, name):
        self.url = url
        self.on_open, self.on_message, self.on_disconnect = on_open, on_message, on_disconnect
        self.socket = Socket()

    def start(self):
        self.on_open(self.socket)
        return True

    def close(self):
        self.socket.close()
        self.on_disconnect()

    def status(self):
        return {"available": True, "connected": not self.socket.closed}


@pytest.fixture
def setup_stream(monkeypatch):
    clock = SimpleNamespace(monotonic=100.0, server=1000.0)
    monkeypatch.setattr(private_module, "OkxWebSocketConnection", Connection)
    monkeypatch.setattr(private_module.time, "monotonic", lambda: clock.monotonic)

    def make(environment="demo", *, subscribe=True):
        stream = OkxPrivateStream(
            SYMBOL, environment, credentials=lambda: ("test-key", "test-secret", "test-passphrase"),
            timestamp=lambda: clock.server,
        )
        assert stream.start()
        socket = stream._connection.socket
        if subscribe:
            stream._on_message(socket, {"event": "login", "code": "0"})
            for channel in ("account", "positions", "orders"):
                arg = {"channel": channel}
                if channel != "account":
                    arg.update(instType="SWAP", instId=SYMBOL)
                stream._on_message(socket, {"event": "subscribe", "arg": arg})
        return stream, socket, clock

    return make


def balance(*, version=999_000, equity="1000", details=None):
    return {"totalEq": equity, "adjEq": equity, "availEq": equity, "uTime": str(version), "details": details or [{"ccy": "USDT", "eq": equity, "uTime": str(version)}]}


def position(*, version=999_000, quantity="1", identity="p1", side="net"):
    return {"instId": SYMBOL, "instType": "SWAP", "posId": identity, "posSide": side, "mgnMode": "isolated", "pos": quantity, "avgPx": "80000", "uTime": str(version)}


def order(*, version=999_000, state="live", identity="o1", filled="0"):
    return {"instId": SYMBOL, "ordId": identity, "clOrdId": "client", "side": "buy", "ordType": "limit", "state": state, "sz": "1", "accFillSz": filled, "px": "80000", "uTime": str(version)}


def baseline(*, positions=None, orders=None, account=None):
    return tuple({"code": "0", "data": data} for data in ([account or balance()], positions or [], orders or []))


def emit(stream, socket, channel, rows, **extra):
    arg = {"channel": channel}
    if channel != "account":
        arg.update(instType="SWAP", instId=SYMBOL)
    stream._on_message(socket, {"arg": arg, "data": rows, **extra})


def seed(stream, **kwargs):
    return stream.synchronize(lambda: baseline(**kwargs))


@pytest.mark.parametrize("environment,host", [("demo", "wspap.okx.com"), ("production", "ws.okx.com")])
def test_login_uses_server_seconds_and_official_private_endpoint(setup_stream, environment, host):
    stream, socket, _clock = setup_stream(environment, subscribe=False)
    assert stream._connection.url == f"wss://{host}:8443/ws/v5/private"
    login = socket.sent[0]
    assert login["op"] == "login"
    args = login["args"][0]
    assert float(args["timestamp"]) == 1000
    expected = base64.b64encode(hmac.new(b"test-secret", (args["timestamp"] + "GET/users/self/verify").encode(), hashlib.sha256).digest()).decode()
    assert args["sign"] == expected
    assert "test-secret" not in json.dumps(login)
    assert "test-key" not in json.dumps(stream.status())


def test_only_read_only_subscriptions_are_sent(setup_stream):
    _stream, socket, _clock = setup_stream()
    assert [row["op"] for row in socket.sent] == ["login", "subscribe"]
    assert {row["channel"] for row in socket.sent[1]["args"]} == {"account", "positions", "orders"}


def test_login_and_subscription_never_imply_flat_account(setup_stream):
    stream, socket, _clock = setup_stream()
    assert stream.status()["subscribed"] is True
    assert stream.snapshot() is None
    emit(stream, socket, "positions", [], eventType="snapshot", curPage=1, lastPage=True)
    emit(stream, socket, "orders", [])
    assert stream.snapshot() is None


def test_provider_is_not_called_before_all_subscription_acks(setup_stream):
    stream, socket, _clock = setup_stream(subscribe=False)
    provider = Mock(return_value=baseline())
    assert not stream.synchronize(provider)
    stream._on_message(socket, {"event": "login", "code": "0"})
    stream._on_message(socket, {"event": "subscribe", "arg": {"channel": "account"}})
    assert not stream.synchronize(provider)
    provider.assert_not_called()


def test_complete_rest_baseline_can_establish_flat_and_is_copied(setup_stream):
    stream, _socket, _clock = setup_stream()
    assert seed(stream)
    snapshot = stream.snapshot()
    assert snapshot["positions"] == snapshot["orders"] == []
    snapshot["account"]["totalEq"] = "corrupted"
    assert stream.snapshot()["account"]["totalEq"] == "1000"


def test_orders_have_no_initial_push_so_rest_existing_order_is_preserved(setup_stream):
    stream, socket, _clock = setup_stream()
    assert seed(stream, orders=[order()])
    emit(stream, socket, "orders", [])
    assert len(stream.snapshot()["orders"]) == 1


def test_order_updates_add_partial_then_remove_and_tombstone_blocks_resurrection(setup_stream):
    stream, socket, _clock = setup_stream()
    assert seed(stream)
    emit(stream, socket, "orders", [order(version=1_000_001)])
    emit(stream, socket, "orders", [order(version=1_000_002, state="partially_filled", filled="0.5")])
    assert stream.snapshot()["orders"][0]["accFillSz"] == "0.5"
    emit(stream, socket, "orders", [order(version=1_000_003, state="filled", filled="1")])
    emit(stream, socket, "orders", [order(version=1_000_002)])
    assert stream.snapshot()["orders"] == []


def test_positions_delta_keeps_other_side_and_zero_explicitly_removes(setup_stream):
    stream, socket, _clock = setup_stream()
    assert seed(stream, positions=[position(side="long"), position(identity="p2", side="short")])
    emit(stream, socket, "positions", [position(version=1_000_001, quantity="0", side="long")], eventType="event_update")
    positions = stream.snapshot()["positions"]
    assert len(positions) == 1 and positions[0]["posId"] == "p2"
    emit(stream, socket, "positions", [position(version=999_999, side="long")])
    assert len(stream.snapshot()["positions"]) == 1


def test_empty_position_delta_does_not_erase_and_conflicting_empty_snapshot_invalidates(setup_stream):
    stream, socket, _clock = setup_stream()
    assert seed(stream, positions=[position()])
    emit(stream, socket, "positions", [], eventType="event_update")
    assert len(stream.snapshot()["positions"]) == 1
    emit(stream, socket, "positions", [], eventType="snapshot", curPage=1, lastPage=True)
    assert stream.snapshot() is None


def test_complete_snapshot_omitting_one_known_position_requires_rest(setup_stream):
    stream, socket, _clock = setup_stream()
    assert seed(stream, positions=[position(), position(identity="p2", side="short")])
    emit(stream, socket, "positions", [position(version=1_000_001)], eventType="snapshot", curPage=1, lastPage=True)
    assert stream.snapshot() is None


def test_account_delta_merges_currencies_and_zero_balance_not_omission(setup_stream):
    stream, socket, _clock = setup_stream()
    assert seed(stream)
    emit(stream, socket, "account", [balance(version=1_000_001, equity="1200", details=[{"ccy": "BTC", "eq": "200"}])], eventType="event_update")
    assert {row["ccy"] for row in stream.snapshot()["account"]["details"]} == {"USDT", "BTC"}
    emit(stream, socket, "account", [balance(version=1_000_002, equity="200", details=[{"ccy": "USDT", "eq": "0"}])], eventType="event_update")
    assert next(row for row in stream.snapshot()["account"]["details"] if row["ccy"] == "USDT")["eq"] == "0"
    emit(stream, socket, "account", [balance(version=999_999, equity="9000")])
    assert stream.snapshot()["account"]["totalEq"] == "200"


def test_conflicting_account_totals_at_equal_version_require_rest(setup_stream):
    stream, socket, _clock = setup_stream()
    assert seed(stream)
    emit(stream, socket, "account", [balance(equity="2000")])
    assert stream.snapshot() is None


@pytest.mark.parametrize("channel", ["account", "positions"])
def test_paginated_snapshot_does_not_replace_complete_state(setup_stream, channel):
    stream, socket, _clock = setup_stream()
    assert seed(stream, positions=[position()])
    rows = [balance(version=1_000_001)] if channel == "account" else [position(version=1_000_001)]
    emit(stream, socket, channel, rows, eventType="snapshot", curPage=1, lastPage=False)
    assert stream.snapshot() is None


def test_account_push_during_baseline_can_merge_without_permanent_retry(setup_stream):
    stream, socket, _clock = setup_stream()

    def provider():
        emit(stream, socket, "account", [balance(version=1_000_001, equity="1234")], eventType="snapshot", curPage=1, lastPage=True)
        return baseline()

    assert stream.synchronize(provider)
    assert stream.snapshot()["account"]["totalEq"] == "1234"


def test_uncovered_order_during_rest_window_invalidates_and_throttles_retry(setup_stream):
    stream, socket, clock = setup_stream()

    def provider():
        emit(stream, socket, "orders", [order(version=1_000_100)])
        clock.server = 1000.2
        return baseline()

    assert not stream.synchronize(provider)
    assert stream.snapshot() is None
    retry = Mock(return_value=baseline())
    assert not stream.synchronize(retry)
    retry.assert_not_called()
    assert stream.status()["retry_after_seconds"] == 5
    clock.monotonic += 5.1
    assert stream.synchronize(retry)


def test_rest_row_that_covers_buffered_order_version_is_authoritative(setup_stream):
    stream, socket, clock = setup_stream()

    def provider():
        emit(stream, socket, "orders", [order(version=1_000_050)])
        clock.server = 1000.2
        return baseline(orders=[order(version=1_000_100, state="partially_filled", filled="0.5")])

    assert stream.synchronize(provider)
    assert stream.snapshot()["orders"][0]["accFillSz"] == "0.5"


def test_late_uncovered_delta_from_rest_window_requires_reconciliation(setup_stream):
    stream, socket, clock = setup_stream()

    def provider():
        clock.server = 1000.2
        return baseline()

    assert stream.synchronize(provider)
    emit(stream, socket, "orders", [order(version=1_000_100)])
    assert stream.snapshot() is None


def test_reconnect_during_rest_load_rejects_old_generation(setup_stream):
    stream, socket, _clock = setup_stream()
    original = stream.status()["generation"]

    def provider():
        stream._on_disconnect()
        stream._on_open(Socket())
        return baseline(positions=[position()])

    assert not stream.synchronize(provider)
    assert stream.status()["generation"] > original
    assert stream.snapshot() is None
    emit(stream, socket, "account", [balance(version=1_000_100)])
    assert stream.snapshot() is None


def test_disconnect_forgets_old_cache_and_requires_new_rest_even_after_acks(setup_stream):
    stream, socket, _clock = setup_stream()
    assert seed(stream, positions=[position()], orders=[order()])
    stream._on_disconnect()
    assert stream.snapshot() is None
    assert stream._positions == stream._orders == {}
    new_socket = Socket()
    stream._on_open(new_socket)
    stream._on_message(new_socket, {"event": "login", "code": "0"})
    for channel in ("account", "positions", "orders"):
        arg = {"channel": channel, "instId": SYMBOL, "instType": "SWAP"} if channel != "account" else {"channel": channel}
        stream._on_message(new_socket, {"event": "subscribe", "arg": arg})
    assert stream.snapshot() is None
    assert seed(stream)
    assert stream.snapshot()["positions"] == []


def test_account_push_and_heartbeat_cannot_extend_sixty_second_rest_fence(setup_stream):
    stream, socket, clock = setup_stream()
    assert seed(stream)
    provider = Mock(return_value=baseline())
    clock.monotonic += 30
    assert stream.synchronize(provider)
    provider.assert_not_called()
    clock.monotonic += 31
    emit(stream, socket, "account", [balance(version=1_060_000)])
    stream._on_message(socket, {"event": "pong"})
    assert stream.snapshot() is None
    assert stream.synchronize(provider)
    provider.assert_called_once()


def test_proactive_refresh_starts_at_45s_without_hiding_fresh_old_snapshot(setup_stream):
    stream, _socket, clock = setup_stream()
    assert seed(stream, positions=[position()], orders=[order()])
    clock.monotonic += 44.9
    provider = Mock(return_value=baseline())
    assert stream.synchronize(provider)
    provider.assert_not_called()
    clock.monotonic += 0.1
    clock.server = 1045

    def refresh():
        assert stream.status()["synchronizing"] is True
        assert stream.status()["ready"] is True
        old = stream.snapshot()
        assert old is not None and len(old["positions"]) == len(old["orders"]) == 1
        return baseline(positions=[position()], orders=[order()])

    assert stream.synchronize(refresh)
    assert stream.snapshot() is not None
    assert stream.status()["last_rest_age_seconds"] == 0


def test_refresh_applies_deltas_to_visible_old_state_and_replays_them_after_seed(setup_stream):
    stream, socket, clock = setup_stream()
    assert seed(stream, positions=[position()], orders=[order()])
    clock.monotonic += 45
    clock.server = 1045

    def refresh():
        emit(stream, socket, "orders", [order(version=1_045_100, state="filled", filled="1")])
        emit(stream, socket, "positions", [position(version=1_045_100, quantity="0")])
        emit(stream, socket, "account", [balance(version=1_045_100, equity="1100")])
        visible = stream.snapshot()
        assert visible is not None
        assert visible["orders"] == visible["positions"] == []
        assert visible["account"]["totalEq"] == "1100"
        clock.server = 1045.2
        return baseline()  # Complete REST state confirms removals.

    assert stream.synchronize(refresh)
    latest = stream.snapshot()
    assert latest["orders"] == latest["positions"] == []
    assert latest["account"]["totalEq"] == "1100"


def test_slow_refresh_does_not_extend_old_60s_deadline_even_with_account_push(setup_stream):
    stream, socket, clock = setup_stream()
    assert seed(stream)
    clock.monotonic += 45
    clock.server = 1045

    def refresh():
        assert stream.snapshot() is not None
        clock.monotonic += 15.1
        emit(stream, socket, "account", [balance(version=1_060_100)])
        stream._on_message(socket, {"event": "pong"})
        assert stream.snapshot() is None
        assert stream.status()["ready"] is False
        clock.server = 1060.1
        return baseline(account=balance(version=1_060_100))

    assert stream.synchronize(refresh)
    assert stream.snapshot() is not None  # Only successful REST completion renews it.


def test_failed_proactive_refresh_immediately_invalidates_otherwise_fresh_baseline(setup_stream):
    stream, _socket, clock = setup_stream()
    assert seed(stream)
    clock.monotonic += 45

    def refresh():
        assert stream.snapshot() is not None
        raise TimeoutError("REST request timed out")

    assert not stream.synchronize(refresh)
    assert stream.snapshot() is None
    assert stream.status()["last_rest_age_seconds"] == 45
    assert stream.status()["retry_after_seconds"] == 5


def test_bad_delta_during_refresh_invalidates_old_view_and_cannot_be_masked_by_rest_success(setup_stream):
    stream, socket, clock = setup_stream()
    assert seed(stream)
    clock.monotonic += 45
    clock.server = 1045

    def refresh():
        emit(stream, socket, "orders", [order(version=1_045_001, state="unknown")])
        assert stream.snapshot() is None
        return baseline()

    assert not stream.synchronize(refresh)
    assert stream.snapshot() is None


def test_disconnect_during_proactive_refresh_never_retains_old_ready(setup_stream):
    stream, _socket, clock = setup_stream()
    assert seed(stream)
    clock.monotonic += 45

    def refresh():
        assert stream.snapshot() is not None
        stream._on_disconnect()
        assert stream.snapshot() is None
        return baseline()

    assert not stream.synchronize(refresh)
    assert stream.snapshot() is None


@pytest.mark.parametrize("broken", [({"code": "1", "data": []}, {"code": "0", "data": []}, {"code": "0", "data": []}), ({"code": "0", "data": []}, {"code": "0", "data": []}, {"code": "0", "data": []})])
def test_failed_or_empty_balance_baseline_is_not_ready(setup_stream, broken):
    stream, _socket, _clock = setup_stream()
    assert not stream.synchronize(lambda: broken)
    assert stream.snapshot() is None


def test_unknown_order_state_or_missing_version_fails_closed(setup_stream):
    stream, socket, _clock = setup_stream()
    assert seed(stream)
    emit(stream, socket, "orders", [order(version=1_000_001, state="unknown")])
    assert stream.snapshot() is None


def test_login_error_does_not_echo_credentials_or_raw_remote_message(setup_stream):
    stream, socket, _clock = setup_stream(subscribe=False)
    stream._on_message(socket, {"event": "error", "code": "60009", "msg": "test-key test-passphrase"})
    status = json.dumps(stream.status())
    assert "test-key" not in status and "test-passphrase" not in status
    assert socket.closed and stream.snapshot() is None


def test_buffer_overflow_during_sync_cannot_be_followed_by_ready(setup_stream, monkeypatch):
    stream, socket, _clock = setup_stream()
    monkeypatch.setattr(private_module, "_MAX_BUFFER", 1)

    def provider():
        emit(stream, socket, "account", [balance()])
        emit(stream, socket, "account", [balance()])
        return baseline()

    assert not stream.synchronize(provider)
    assert stream.snapshot() is None


def test_close_clears_ready_and_releases_socket(setup_stream):
    stream, socket, _clock = setup_stream()
    assert seed(stream)
    stream.close()
    assert stream.snapshot() is None and socket.closed
    assert stream.status()["connected"] is False
    assert stream.start() is False


def test_invalid_environment_is_rejected_before_connect():
    with pytest.raises(ValueError, match="demo or production"):
        OkxPrivateStream(SYMBOL, "testnet", credentials=lambda: ("", "", ""), timestamp=lambda: 1000)
