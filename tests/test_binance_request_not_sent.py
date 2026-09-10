import time
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from btc_futures_bot import http_client
from btc_futures_bot.exchanges.base import ExchangeSettings
from btc_futures_bot.exchanges.binance import BinanceAdapter
from btc_futures_bot.http_client import ApiError
from btc_futures_bot.models import OrderRequest


BASE = "https://fapi.binance.com"


@pytest.fixture(autouse=True)
def isolated_transport(monkeypatch):
    http_client.clear_rate_limits()
    monkeypatch.setattr(http_client, "urlopen", lambda *_a, **_k: pytest.fail("unmocked HTTP"))
    yield
    http_client.clear_rate_limits()


def adapter(monkeypatch):
    result = BinanceAdapter(ExchangeSettings("binance", "production", BASE, "BTCUSDT"))
    monkeypatch.setattr(result, "credentials", lambda: ("test-key", "test-secret", ""))
    result._server_time_anchor_ms = 1_800_000_000_000
    result._server_time_anchor_monotonic = time.monotonic()
    result._server_time_synced_at = time.monotonic() - 1_000
    return result


@pytest.mark.parametrize("gate", ["host", "budget"])
def test_only_before_send_gate_marks_transport_request_not_sent(monkeypatch, gate):
    if gate == "host":
        monkeypatch.setattr(http_client, "rate_limit_remaining", lambda _url: 20)
    else:
        monkeypatch.setattr(http_client.binance_request_budget, "reserve", lambda *_a: time.time() + 20)
    with pytest.raises(ApiError) as caught:
        http_client.request_json("POST", BASE + "/fapi/v1/order", body="reduceOnly=true")
    assert caught.value.request_not_sent is True


@pytest.mark.parametrize("status", [418, 429, 503])
def test_actual_exchange_response_is_never_marked_unsent(monkeypatch, status):
    def reject(request, **kwargs):
        raise HTTPError(request.full_url, status, "rejected", {"Retry-After": "20"}, BytesIO(b'{"code":-1003}'))
    monkeypatch.setattr(http_client, "urlopen", reject)
    with pytest.raises(ApiError) as caught:
        http_client.request_json("POST", BASE + "/fapi/v1/order")
    assert caught.value.request_not_sent is False


def test_transport_timeout_is_uncertain_by_default(monkeypatch):
    monkeypatch.setattr(http_client, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(URLError("timeout")))
    with pytest.raises(ApiError) as caught:
        http_client.request_json("POST", BASE + "/fapi/v1/order")
    assert caught.value.request_not_sent is False
    assert ApiError("unknown failure").request_not_sent is False


@pytest.mark.parametrize("status", [418, 429])
def test_pre_submit_clock_rejection_preserves_original_cause_and_marks_target_order_unsent(monkeypatch, status):
    venue = adapter(monkeypatch)
    calls = []
    def request(method, url, **kwargs):
        calls.append((method, url))
        assert url.endswith("/time")
        raise ApiError(f"HTTP {status} GET {BASE}/fapi/v1/time: IP banned", status_code=status, retry_at=time.time() + 20)
    monkeypatch.setattr("btc_futures_bot.exchanges.binance.request_json", request)
    with pytest.raises(ApiError) as caught:
        venue.place_market_order(OrderRequest("buy", 1, True, "close-test"))
    assert caught.value.request_not_sent is True
    assert caught.value.status_code == status
    assert "GET" in str(caught.value) and "IP banned" in str(caught.value)
    assert calls == [("GET", BASE + "/fapi/v1/time")]


def test_non_rate_limit_clock_failure_still_uses_existing_anchor(monkeypatch):
    venue = adapter(monkeypatch)
    calls = []
    def request(method, url, **kwargs):
        calls.append((method, url))
        if url.endswith("/time"):
            raise ApiError("temporary clock network error")
        return {"status": "FILLED", "executedQty": "1", "avgPrice": "100"}
    monkeypatch.setattr("btc_futures_bot.exchanges.binance.request_json", request)
    assert venue.place_market_order(OrderRequest("buy", 1, True, "close-test"))["status"] == "FILLED"
    assert [method for method, _ in calls] == ["GET", "POST"]


def test_signed_request_preserves_local_gate_marker_without_marking_real_rejection(monkeypatch):
    venue = adapter(monkeypatch)
    monkeypatch.setattr(venue, "_ensure_server_time", lambda: None)
    for not_sent in (True, False):
        def request(*_args, **_kwargs):
            raise ApiError("local gate" if not_sent else "HTTP 429 POST order", status_code=429,
                           retry_at=time.time() + 20, request_not_sent=not_sent)
        monkeypatch.setattr("btc_futures_bot.exchanges.binance.request_json", request)
        with pytest.raises(ApiError) as caught:
            venue.place_market_order(OrderRequest("buy", 1, True, "close-test"))
        assert caught.value.request_not_sent is not_sent


@pytest.mark.parametrize("gate_stage", ["clock_resync", "second_order_attempt"])
def test_gate_after_an_already_submitted_1021_request_does_not_grant_unsent_retry_permission(monkeypatch, gate_stage):
    venue = adapter(monkeypatch)
    monkeypatch.setattr(venue, "_ensure_server_time", lambda: None)
    submitted = []
    def request(method, url, **kwargs):
        submitted.append((method, url))
        if len(submitted) == 1:
            raise ApiError("HTTP 400 POST order: -1021", status_code=400, api_code=-1021)
        if gate_stage == "second_order_attempt" and url.endswith("/time"):
            return {"serverTime": 1_800_000_000_000}
        raise ApiError("Binance REST deferred locally: preventive request budget",
                       retry_at=time.time() + 20, api_code="LOCAL_REQUEST_BUDGET", request_not_sent=True)
    monkeypatch.setattr("btc_futures_bot.exchanges.binance.request_json", request)
    with pytest.raises(ApiError) as caught:
        venue.place_market_order(OrderRequest("buy", 1, True, "close-test"))
    assert caught.value.request_not_sent is False
    assert submitted[0] == ("POST", BASE + "/fapi/v1/order")


@pytest.mark.parametrize("lookup_succeeds", [True, False])
def test_ambiguous_order_uses_original_client_id_lookup_and_never_resubmits(monkeypatch, lookup_succeeds):
    venue = adapter(monkeypatch)
    calls = []
    def signed(method, path, params):
        calls.append((method, path, params))
        if method == "POST":
            raise ApiError("network error POST order: timeout")
        assert params["origClientOrderId"] == "close-original"
        if not lookup_succeeds:
            raise ApiError("rate limit active for host", status_code=418, request_not_sent=True)
        return {"status": "FILLED", "executedQty": "1", "avgPrice": "100"}
    monkeypatch.setattr(venue, "_signed", signed)
    order = OrderRequest("buy", 1, True, "close-original")
    if lookup_succeeds:
        assert venue.place_market_order(order)["status"] == "FILLED"
    else:
        with pytest.raises(ApiError, match="network error") as caught:
            venue.place_market_order(order)
        assert caught.value.request_not_sent is False
    assert [method for method, _, _ in calls] == ["POST", "GET"]
