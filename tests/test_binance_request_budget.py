from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from btc_futures_bot import binance_request_budget as budget
from btc_futures_bot import http_client
from btc_futures_bot.exchanges.base import ExchangeSettings
from btc_futures_bot.exchanges.binance import BinanceAdapter


BASE = "https://fapi.binance.com"
MINUTE = 1_800_000_000.0


class _Clock:
    def __init__(self, wall=MINUTE + 10, monotonic=100.0):
        self.wall = wall
        self.monotonic = monotonic

    def advance(self, seconds):
        self.wall += seconds
        self.monotonic += seconds


class _Response:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return b'{"ok":true}'


def _headers(weight, server_time=MINUTE + 10):
    return {"X-MBX-USED-WEIGHT-1M": str(weight), "Date": formatdate(server_time, usegmt=True)}


@pytest.fixture(autouse=True)
def isolated_budget(monkeypatch):
    http_client.clear_rate_limits()

    def no_network(*args, **kwargs):
        raise AssertionError("test must mock every network request")

    def no_sleep(*args, **kwargs):
        raise AssertionError("local scheduling must not sleep on a signed request")

    monkeypatch.setattr(http_client, "urlopen", no_network)
    monkeypatch.setattr(http_client.time, "sleep", no_sleep)
    yield
    http_client.clear_rate_limits()


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(budget.time, "time", lambda: clock.wall)
    monkeypatch.setattr(budget.time, "monotonic", lambda: clock.monotonic)
    return clock


def _assert_local_defer(call):
    with pytest.raises(http_client.ApiError) as caught:
        call()
    assert http_client.is_local_request_deferred(caught.value)
    assert not http_client.is_rate_limit_error(caught.value)
    assert caught.value.retry_after_seconds > 0
    return caught.value


@pytest.mark.parametrize("cross_minute", [False, True])
def test_response_does_not_discard_another_threads_inflight_reservation(
    monkeypatch, clock, cross_minute,
):
    # A's header includes shared-IP traffic but not B, whose request has not
    # reached the exchange yet. B's reserved cost must survive A's response.
    if cross_minute:
        clock.wall = MINUTE - 2
        budget.observe(BASE, _headers(0, MINUTE - 2))
        clock.advance(3)
    else:
        budget.observe(BASE, _headers(0, clock.wall))
    first_entered, second_entered = threading.Event(), threading.Event()
    release_first, release_second = threading.Event(), threading.Event()
    sent = []

    def urlopen(request, **kwargs):
        path = urlsplit(request.full_url).path
        sent.append(path)
        if path == "/fapi/v2/account":
            first_entered.set()
            assert release_first.wait(3)
            return _Response(_headers(1794, clock.wall))
        if path == "/fapi/v2/positionRisk":
            second_entered.set()
            assert release_second.wait(3)
            return _Response(_headers(1799, clock.wall))
        return _Response()

    monkeypatch.setattr(http_client, "urlopen", urlopen)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(http_client.request_json, "GET", BASE + "/fapi/v2/account")
        assert first_entered.wait(3)
        second = executor.submit(http_client.request_json, "GET", BASE + "/fapi/v2/positionRisk")
        assert second_entered.wait(3)
        try:
            release_first.set()
            assert first.result(timeout=3) == {"ok": True}
            _assert_local_defer(lambda: http_client.request_json(
                "POST", BASE + "/fapi/v1/leverage", body={"symbol": "BTCUSDT", "leverage": 20},
            ))
            assert sent == ["/fapi/v2/account", "/fapi/v2/positionRisk"]
        finally:
            release_first.set()
            release_second.set()
        assert second.result(timeout=3) == {"ok": True}


def test_local_clock_offset_cannot_reopen_budget_twice_in_one_server_minute(clock):
    # Local clock is 50 seconds ahead. After the first server-based expiry,
    # using local wall-clock boundaries would reset this window 10s later.
    clock.wall = MINUTE + 50
    budget.observe(BASE, _headers(1800, MINUTE))
    assert budget.reserve("GET", BASE + "/fapi/v1/klines", None) > clock.wall
    clock.advance(61)
    budget.configure(BASE + "/fapi/v1/exchangeInfo", {"rateLimits": [{
        "rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 200,
    }]})
    # Unknown endpoints conservatively reserve 50. The 60s request counter is
    # far below its ceiling, so only the weight accounting can prevent this.
    assert budget.reserve("GET", BASE + "/fapi/custom/a", None) == 0
    assert budget.reserve("GET", BASE + "/fapi/custom/b", None) == 0
    assert budget.reserve("GET", BASE + "/fapi/custom/c", None) > clock.wall
    clock.advance(11)
    assert budget.reserve("GET", BASE + "/fapi/custom/d", None) > clock.wall
    clock.advance(50)
    assert budget.reserve("GET", BASE + "/fapi/custom/e", None) == 0


def test_late_previous_minute_response_cannot_replace_current_high_weight(clock):
    budget.observe(BASE, _headers(1801, MINUTE + 10))
    budget.observe(BASE, _headers(2, MINUTE - 1))
    assert budget.reserve("GET", BASE + "/fapi/v1/klines", None) > clock.wall


@pytest.mark.parametrize("date", [None, "invalid-date"])
def test_missing_server_date_cannot_advance_window_using_a_fast_local_clock(clock, date):
    clock.wall = MINUTE + 130  # Local clock is two minutes ahead of Binance.
    budget.observe(BASE, _headers(1900, MINUTE + 10))
    headers = {"X-MBX-USED-WEIGHT-1M": "5"}
    if date is not None:
        headers["Date"] = date
    budget.observe(BASE, headers)
    # A response without a usable Date does not prove an exchange rollover.
    assert budget.reserve("GET", BASE + "/fapi/v1/klines", None) > clock.wall
    # A subsequent valid Date must still be accepted after that response.
    budget.observe(BASE, _headers(2000, MINUTE + 11))
    assert budget.reserve("GET", BASE + "/fapi/v1/klines", None) > clock.wall


def test_first_real_server_date_can_replace_an_initial_local_clock_fallback(clock):
    clock.wall = MINUTE + 130
    budget.observe(BASE, {"X-MBX-USED-WEIGHT-1M": "5"})
    budget.observe(BASE, _headers(1900, MINUTE + 10))
    assert budget.reserve("GET", BASE + "/fapi/v1/klines", None) > clock.wall


def test_high_weight_http_400_defers_followup_without_another_network_request(monkeypatch, clock):
    sent = []

    def urlopen(request, **kwargs):
        sent.append(request.full_url)
        raise HTTPError(request.full_url, 400, "bad request", _headers(1805), BytesIO(
            b'{"code":-4046,"msg":"No need to change margin type."}',
        ))

    monkeypatch.setattr(http_client, "urlopen", urlopen)
    with pytest.raises(http_client.ApiError) as caught:
        http_client.request_json("POST", BASE + "/fapi/v1/marginType", body={"symbol": "BTCUSDT"})
    assert caught.value.api_code == -4046
    assert not http_client.is_rate_limit_error(caught.value)
    _assert_local_defer(lambda: http_client.request_json(
        "GET", BASE + "/fapi/v1/klines", params={"symbol": "BTCUSDT", "limit": 300},
    ))
    assert len(sent) == 1


@pytest.mark.parametrize("method,path,params", [
    ("POST", "/fapi/v1/order", {"symbol": "BTCUSDT", "reduceOnly": "true"}),
    ("POST", "/fapi/v1/algoOrder", {"symbol": "BTCUSDT", "closePosition": "true"}),
    ("DELETE", "/fapi/v1/algoOrder", {"symbol": "BTCUSDT", "algoId": "123"}),
    ("GET", "/fapi/v1/order", {"symbol": "BTCUSDT", "orderId": "123"}),
    ("GET", "/fapi/v1/userTrades", {"symbol": "BTCUSDT", "orderId": "123"}),
    ("GET", "/fapi/v1/openAlgoOrders", {"symbol": "BTCUSDT"}),
    ("GET", "/fapi/v1/time", {}),
    ("PUT", "/fapi/v1/listenKey", {}),
])
def test_safety_operations_can_use_headroom_but_new_entry_cannot(monkeypatch, clock, method, path, params):
    budget.observe(BASE, _headers(1900))
    sent = []
    monkeypatch.setattr(http_client, "urlopen", lambda request, **kwargs: sent.append(request) or _Response())
    _assert_local_defer(lambda: http_client.request_json(
        "POST", BASE + "/fapi/v1/order", body={"symbol": "BTCUSDT", "side": "BUY"},
    ))
    kwargs = {"params": params} if method in {"GET", "DELETE"} else {"body": params}
    assert http_client.request_json(method, BASE + path, **kwargs) == {"ok": True}
    assert len(sent) == 1


@pytest.mark.parametrize("path,flag", [
    ("/fapi/v1/order", "reduceOnly"), ("/fapi/v1/algoOrder", "closePosition"),
])
def test_zero_ip_weight_risk_reduction_is_not_blocked_by_preventive_weight_or_count(
    monkeypatch, clock, path, flag,
):
    sent = []
    monkeypatch.setattr(http_client, "urlopen", lambda request, **kwargs: sent.append(request) or _Response())
    body = urlencode({"symbol": "BTCUSDT", flag: "true", "timestamp": int(clock.wall * 1000)})
    # A noisy local caller has used the request-count allowance; fresh weight
    # observations can separately be above the preventive exchange ceiling.
    for _ in range(120):
        assert http_client.request_json("POST", BASE + path, body=body) == {"ok": True}
    budget.observe(BASE, _headers(2401))
    assert http_client.request_json("POST", BASE + path, body=body) == {"ok": True}
    assert len(sent) == 121
    _assert_local_defer(lambda: http_client.request_json(
        "POST", BASE + "/fapi/v1/order", body=urlencode({"symbol": "BTCUSDT", "side": "BUY"}),
    ))
    assert len(sent) == 121


def test_real_exchange_429_blocks_even_protection_and_fill_recovery(monkeypatch, clock):
    sent = []

    def urlopen(request, **kwargs):
        sent.append(request)
        raise HTTPError(request.full_url, 429, "too many requests", {
            **_headers(2401), "Retry-After": "15",
        }, BytesIO(b'{"code":-1003,"msg":"Too many requests."}'))

    monkeypatch.setattr(http_client, "urlopen", urlopen)
    with pytest.raises(http_client.ApiError) as caught:
        http_client.request_json("POST", BASE + "/fapi/v1/order", body={"reduceOnly": "true"})
    assert caught.value.status_code == 429
    for method, path, kwargs in [
        ("POST", "/fapi/v1/algoOrder", {"body": {"closePosition": "true"}}),
        ("GET", "/fapi/v1/userTrades", {"params": {"symbol": "BTCUSDT"}}),
        ("GET", "/fapi/v1/time", {}),
    ]:
        with pytest.raises(http_client.ApiError) as blocked:
            http_client.request_json(method, BASE + path, **kwargs)
        assert http_client.is_rate_limit_error(blocked.value)
        assert not http_client.is_local_request_deferred(blocked.value)
        assert blocked.value.retry_after_seconds == 15
    assert len(sent) == 1


def test_signed_defer_does_not_sleep_and_next_call_gets_a_fresh_timestamp(monkeypatch, clock):
    adapter = BinanceAdapter(ExchangeSettings(
        name="binance", environment="production", base_url=BASE, symbol="BTCUSDT",
    ))
    monkeypatch.setattr(adapter, "credentials", lambda: ("test-key", "test-secret", ""))
    monkeypatch.setattr(adapter, "_ensure_server_time", lambda **kwargs: None)
    signed_at = []

    def timestamp():
        value = int(clock.wall * 1000)
        signed_at.append(value)
        return value

    monkeypatch.setattr(adapter, "_server_timestamp_ms", timestamp)
    budget.observe(BASE, _headers(1900))
    sent = []
    monkeypatch.setattr(http_client, "urlopen", lambda request, **kwargs: sent.append(request) or _Response())
    _assert_local_defer(lambda: adapter._signed("GET", "/fapi/v2/account"))
    assert sent == []
    assert len(signed_at) == 1

    clock.advance(61)
    assert adapter._signed("GET", "/fapi/v2/account") == {"ok": True}
    assert len(sent) == 1
    timestamp_query = parse_qs(urlsplit(sent[0].full_url).query)["timestamp"]
    assert timestamp_query == [str(signed_at[-1])]
    assert signed_at[-1] - signed_at[0] == 61_000
