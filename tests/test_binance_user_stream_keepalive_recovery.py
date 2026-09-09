from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import btc_futures_bot.binance_user_stream as stream_module
from btc_futures_bot.binance_user_stream import BinanceUserDataStream
from btc_futures_bot.http_client import ApiError


class _Clock:
    wall = 1_800_000_000.0
    monotonic = 1000.0

    def advance(self, seconds):
        self.wall += seconds
        self.monotonic += seconds


class _Steps:
    def __init__(self, clock, advances):
        self.clock = clock
        self.advances = iter(advances)
        self.stopped = False

    def is_set(self):
        return self.stopped

    def wait(self, delay):
        try:
            self.clock.advance(next(self.advances))
        except StopIteration:
            self.stopped = True
        return self.stopped


@pytest.fixture
def stream_and_clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(stream_module, "time", SimpleNamespace(
        time=lambda: clock.wall, monotonic=lambda: clock.monotonic,
    ))
    monkeypatch.setattr(stream_module, "request_json", Mock(side_effect=AssertionError("no network")))
    stream = BinanceUserDataStream(
        "BTCUSDT", "production", "https://fapi.binance.com", lambda: "fake-key",
        lambda: {"account": {"availableBalance": "123"}}, keepalive_seconds=60,
    )
    stream.seed_snapshot({"account": {"availableBalance": "123"}})
    stream._connected = True
    stream._last_transport_at = clock.monotonic
    stream._listen_key = "active-key"
    stream._last_keepalive_at = clock.monotonic - 60
    monkeypatch.setattr(stream, "start", lambda **kwargs: False)
    return stream, clock


def _renew(monkeypatch, stream, clock, outcome, *, advance=0):
    stream._stop = _Steps(clock, [advance])
    if isinstance(outcome, Exception):
        request = Mock(side_effect=outcome)
    elif callable(outcome):
        request = Mock(side_effect=outcome)
    else:
        request = Mock(return_value=outcome)
    monkeypatch.setattr(stream, "_listen_key_request", request)
    stream._keepalive_loop()
    request.assert_called_once_with("PUT")
    return request


def _fail_renewal(monkeypatch, stream, clock):
    error = ApiError("HTTP 429 PUT listenKey", status_code=429, retry_at=clock.wall + 30)
    _renew(monkeypatch, stream, clock, error)
    assert stream.status()["last_error"] == str(error)
    return error


@pytest.mark.parametrize("failure", ["429", "local_budget", "network"])
def test_only_actual_successful_renewal_clears_its_previous_error(monkeypatch, stream_and_clock, failure):
    stream, clock = stream_and_clock
    errors = {
        "429": ApiError("HTTP 429 PUT listenKey", status_code=429, retry_at=clock.wall + 30),
        "local_budget": ApiError("Binance REST deferred locally: renewal", retry_at=clock.wall + 30,
                                 api_code="LOCAL_REQUEST_BUDGET"),
        "network": RuntimeError("network error PUT listenKey"),
    }
    error = errors[failure]
    _renew(monkeypatch, stream, clock, error)
    assert stream.status()["last_error"] == str(error)
    assert stream._last_exception is error
    clock.advance(60)
    stream._last_transport_at = clock.monotonic
    # Transport health and the passage of a deadline are not proof that the
    # operation which failed has recovered.
    assert stream.healthy()
    assert stream.status()["retry_after_seconds"] == 0
    assert stream.status()["last_error"] == str(error)
    assert not stream.status().get("rest_recovery_confirmed", False)

    _renew(monkeypatch, stream, clock, {"listenKey": "active-key"})

    status = stream.status()
    assert status["healthy"] and status["ready"]
    assert status["last_error"] == ""
    assert stream._last_exception is None
    assert status["retry_at"] == 0
    assert status["rest_recovery_confirmed"] is True
    assert len(status["rest_recovery_id"]) == 32
    assert stream.snapshot()["account"]["availableBalance"] == "123"
    json.dumps(status)


@pytest.mark.parametrize("failure_source", ["bootstrap", "transport"])
def test_successful_renewal_does_not_clear_an_independent_failure(
    monkeypatch, stream_and_clock, failure_source,
):
    stream, clock = stream_and_clock
    if failure_source == "bootstrap":
        error = ApiError("account bootstrap failed", status_code=500)
        stream._snapshot_loader = Mock(side_effect=error)
        stream._on_open(SimpleNamespace(keep_running=True))
    else:
        error = RuntimeError("transport disconnected")
        stream._on_error(None, error)
        stream._on_close(None, None, "closed")

    _renew(monkeypatch, stream, clock, {})

    assert stream.status()["last_error"] == str(error)
    assert stream._last_exception is error
    assert not stream.status()["rest_recovery_confirmed"]
    assert stream.status()["rest_recovery_id"] == ""
    assert not stream.healthy()
    with pytest.raises(RuntimeError, match=str(error)):
        stream.snapshot()


def test_success_returning_after_a_new_transport_error_cannot_clear_it(monkeypatch, stream_and_clock):
    stream, clock = stream_and_clock
    _fail_renewal(monkeypatch, stream, clock)
    later = RuntimeError("new transport failure while PUT is in flight")

    def request(method):
        stream._on_error(None, later)
        return {}

    _renew(monkeypatch, stream, clock, request, advance=60)

    assert stream.status()["last_error"] == str(later)
    assert stream._last_exception is later
    assert not stream.status()["rest_recovery_confirmed"]


def test_success_returning_after_a_new_keepalive_failure_cannot_clear_that_version(monkeypatch, stream_and_clock):
    stream, clock = stream_and_clock
    _fail_renewal(monkeypatch, stream, clock)
    later = RuntimeError("new renewal failure")

    def request(method):
        stream._record_error(later, source="keepalive")
        return {}

    _renew(monkeypatch, stream, clock, request, advance=60)

    assert stream.status()["last_error"] == str(later)
    assert stream._last_exception is later
    assert not stream.status()["rest_recovery_confirmed"]


def test_success_returning_after_a_new_ban_cannot_clear_its_deadline(monkeypatch, stream_and_clock):
    stream, clock = stream_and_clock
    _fail_renewal(monkeypatch, stream, clock)
    later = ApiError("new HTTP 418 ban", status_code=418, retry_at=clock.wall + 600)

    def request(method):
        stream._record_error(later, bootstrap_failure=True)
        return {}

    _renew(monkeypatch, stream, clock, request, advance=60)

    assert stream.status()["last_error"] == str(later)
    assert stream.status()["retry_at"] == later.retry_at
    assert stream.status()["retry_after_seconds"] == 540
    assert not stream.status()["rest_recovery_confirmed"]


def test_old_success_cannot_revive_an_expired_listen_key_or_cached_account(monkeypatch, stream_and_clock):
    stream, clock = stream_and_clock
    _fail_renewal(monkeypatch, stream, clock)

    def request(method):
        stream.process_message(json.dumps({"e": "listenKeyExpired"}))
        return {"listenKey": "active-key"}

    _renew(monkeypatch, stream, clock, request, advance=60)

    assert stream._listen_key == ""
    assert "listenKey expired" in stream.status()["last_error"]
    assert not stream.status()["rest_recovery_confirmed"]
    assert not stream.healthy()
    with pytest.raises(RuntimeError, match="listenKey expired"):
        stream.snapshot()


def test_old_success_cannot_overwrite_a_replacement_key_or_its_timer(monkeypatch, stream_and_clock):
    stream, clock = stream_and_clock
    _fail_renewal(monkeypatch, stream, clock)
    replacement_time = clock.monotonic + 60

    def request(method):
        stream._listen_key = "replacement-key"
        stream._last_keepalive_at = replacement_time
        return {"listenKey": "active-key"}

    _renew(monkeypatch, stream, clock, request, advance=60)

    assert stream._listen_key == "replacement-key"
    assert stream._last_keepalive_at == replacement_time
    assert not stream.status()["rest_recovery_confirmed"]


def test_secondary_keepalive_failure_preserves_bootstrap_error_and_longer_ban(monkeypatch, stream_and_clock):
    stream, clock = stream_and_clock
    original = ApiError("account bootstrap failed", status_code=500)
    stream._record_error(original, bootstrap_failure=True)
    rate_limit = ApiError("PUT listenKey rate limited", status_code=429, retry_at=clock.wall + 120)

    _renew(monkeypatch, stream, clock, rate_limit)

    assert stream.status()["last_error"] == str(original)
    assert stream._last_exception is original
    assert stream.status()["retry_at"] == rate_limit.retry_at
    assert not stream.status()["rest_recovery_confirmed"]


@pytest.mark.parametrize("event", ["record_error", "expired", "close", "bootstrap"])
def test_recovery_confirmation_is_invalidated_by_a_later_stream_lifecycle_event(
    monkeypatch, stream_and_clock, event,
):
    stream, clock = stream_and_clock
    _fail_renewal(monkeypatch, stream, clock)
    _renew(monkeypatch, stream, clock, {}, advance=60)
    assert stream.status()["rest_recovery_confirmed"]
    if event == "record_error":
        stream._record_error(RuntimeError("new failure"))
    elif event == "expired":
        stream.process_message(json.dumps({"e": "listenKeyExpired"}))
    elif event == "close":
        stream._on_close(None, None, "closed")
    else:
        stream.seed_snapshot({"account": {"availableBalance": "new"}})
    assert not stream.status()["rest_recovery_confirmed"]


def test_only_actual_bootstrap_rest_success_can_confirm_fresh_process_recovery(stream_and_clock):
    stream, clock = stream_and_clock
    # A seeded fixture/cache on its own does not establish REST recovery.
    assert not stream.status()["rest_recovery_confirmed"]
    stream._retry_at = clock.wall - 1
    app = SimpleNamespace(keep_running=True)

    stream._on_open(app)

    assert app.keep_running
    assert stream.status()["healthy"]
    assert stream.status()["last_error"] == ""
    assert stream.status()["retry_at"] == 0
    assert stream.status()["rest_recovery_confirmed"]
    assert len(stream.status()["rest_recovery_id"]) == 32


@pytest.mark.parametrize("failure", ["transport", "keepalive", "expired", "new_ban"])
def test_successful_bootstrap_does_not_erase_a_concurrent_failure(monkeypatch, stream_and_clock, failure):
    stream, clock = stream_and_clock
    error = RuntimeError("concurrent stream failure")
    if failure == "new_ban":
        error = ApiError("concurrent HTTP 429", status_code=429, retry_at=clock.wall + 120)

    def loader():
        if failure == "expired":
            stream.process_message(json.dumps({"e": "listenKeyExpired"}))
        elif failure == "transport":
            stream._on_error(None, error)
        else:
            stream._record_error(error, source="keepalive")
        return {"account": {"availableBalance": "must-not-publish"}}

    stream._snapshot_loader = loader
    app = SimpleNamespace(keep_running=True)
    stream._on_open(app)

    assert not app.keep_running
    assert not stream.status()["healthy"]
    assert not stream.status()["ready"]
    assert not stream.status()["rest_recovery_confirmed"]
    if failure == "expired":
        assert "listenKey expired" in stream.status()["last_error"]
    else:
        assert stream.status()["last_error"] == str(error)
    if failure == "new_ban":
        assert stream.status()["retry_at"] == error.retry_at
    with pytest.raises(RuntimeError):
        stream.snapshot()


def test_recovery_id_is_stable_until_a_new_proven_recovery(monkeypatch, stream_and_clock):
    stream, clock = stream_and_clock
    _fail_renewal(monkeypatch, stream, clock)
    _renew(monkeypatch, stream, clock, {}, advance=60)
    first_id = stream.status()["rest_recovery_id"]
    _renew(monkeypatch, stream, clock, {}, advance=60)
    assert stream.status()["rest_recovery_id"] == first_id

    stream._record_error(RuntimeError("new renewal error"), source="keepalive")
    assert stream.status()["rest_recovery_id"] == ""
    _renew(monkeypatch, stream, clock, {}, advance=60)
    second_id = stream.status()["rest_recovery_id"]
    assert second_id and second_id != first_id


def test_each_actual_bootstrap_has_a_distinct_recovery_id(stream_and_clock):
    stream, clock = stream_and_clock
    stream._on_open(SimpleNamespace(keep_running=True))
    first_id = stream.status()["rest_recovery_id"]
    stream._on_open(SimpleNamespace(keep_running=True))
    assert stream.status()["rest_recovery_id"] != first_id
