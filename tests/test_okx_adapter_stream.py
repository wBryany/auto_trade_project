from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from btc_futures_bot.exchanges.base import ExchangeSettings
from btc_futures_bot.exchanges.okx import OkxAdapter
from btc_futures_bot.http_client import ApiError
from btc_futures_bot.models import Candle


def adapter():
    return OkxAdapter(ExchangeSettings(name="okx", environment="demo",
        base_url="https://openapi.okx.com", symbol="BTC-USDT-SWAP"))


def streams(monkeypatch, item, *, candles=None, private=None):
    market = Mock()
    market.candles.return_value = candles
    market.seed_generation.return_value = 7
    market.market_snapshot.return_value = {"mark_price": 100.0, "timestamp": 1000}
    market.status.return_value = {"connected": True}
    account = Mock()
    account.snapshot.return_value = private
    account.status.return_value = {"ready": private is not None}
    item._market_stream, item._private_stream = market, account
    monkeypatch.setattr(item, "_ensure_server_time", lambda: None)
    return market, account


def test_ready_candles_and_mark_avoid_rest(monkeypatch):
    item = adapter()
    rows = [Candle(0, 100, 101, 99, 100, 1)]
    streams(monkeypatch, item, candles=rows)
    rest = Mock(side_effect=AssertionError("REST not expected"))
    monkeypatch.setattr("btc_futures_bot.exchanges.okx.request_json", rest)
    assert item.fetch_candles("1m", 100) == rows
    assert item.fetch_mark_price() == 100
    assert not rest.called


def test_rest_backfill_passes_generation_and_okx_limit(monkeypatch):
    item = adapter()
    market, _ = streams(monkeypatch, item)
    monkeypatch.setattr(item, "_server_timestamp_ms", lambda: 1000)
    rest = Mock(return_value={"code": "0", "data": [[0, "100", "101", "99", "100", "1", "1", "100", "0"]]})
    monkeypatch.setattr("btc_futures_bot.exchanges.okx.request_json", rest)
    rows = item.fetch_candles("1m", 999)
    assert rest.call_args.kwargs["params"]["limit"] == 300
    assert rest.call_args.kwargs["max_attempts"] == 1
    assert rest.call_args.kwargs["headers"] == {"x-simulated-trading": "1"}
    market.seed_candles.assert_called_once_with("1m", rows, generation=7)


def test_rest_mark_price_does_not_fetch_private_account(monkeypatch):
    item = adapter()
    rest = Mock(return_value={"code": "0", "data": [{"markPx": "100", "ts": "1000"}]})
    monkeypatch.setattr("btc_futures_bot.exchanges.okx.request_json", rest)
    assert item.fetch_mark_price() == 100
    assert rest.call_count == 1
    assert rest.call_args.args[1].endswith("/public/mark-price")


def test_ws_equity_and_snapshot_no_signed_rest(monkeypatch):
    item = adapter()
    private = {"account": {"totalEq": "200", "availEq": "180"}, "positions": [], "orders": []}
    streams(monkeypatch, item, private=private)
    monkeypatch.setattr(item, "has_credentials", lambda: True)
    monkeypatch.setattr(item, "_fetch_private_baseline", Mock(side_effect=AssertionError("REST not expected")))
    assert item.fetch_equity() == 200
    snapshot = item.fetch_dashboard_snapshot()
    assert snapshot["private_source"] == "websocket"
    assert snapshot["account"]["wallet_balance"] == 200


def test_reconcile_failure_reuses_one_rest_baseline(monkeypatch):
    item = adapter()
    _, private = streams(monkeypatch, item)
    monkeypatch.setattr(item, "has_credentials", lambda: True)
    private.synchronize.side_effect = lambda provider: provider() and False
    baseline = ({"code": "0", "data": [{"totalEq": "200"}]}, {"code": "0", "data": []}, {"code": "0", "data": []})
    provider = Mock(return_value=baseline)
    monkeypatch.setattr(item, "_fetch_private_baseline", provider)
    snapshot = item.fetch_dashboard_snapshot()
    assert provider.call_count == 1
    assert snapshot["private_source"] == "rest"
    assert snapshot["private_available"]


def test_ui_stream_snapshot_is_nonblocking_and_expires_rest(monkeypatch):
    item = adapter()
    market, _ = streams(monkeypatch, item)
    market.market_snapshot.return_value = None
    item._last_rest_snapshot = {"account": {"wallet_balance": 123}, "private_available": True}
    item._last_rest_snapshot_at = 10
    item._last_rest_market = {"mark_price": 100}
    item._last_rest_market_at = 10
    monkeypatch.setattr("btc_futures_bot.exchanges.okx.time.monotonic", lambda: 30)
    monkeypatch.setattr(item, "_public", Mock(side_effect=AssertionError("no UI REST")))
    snapshot = item.fetch_live_dashboard_snapshot()
    assert snapshot["market"]["stale"]
    assert snapshot["private_stale"]
    assert not snapshot["private_available"]
    assert snapshot["account"]["wallet_balance"] == 123


@pytest.mark.parametrize("code", ["50011", "50040"])
def test_business_rate_limit_is_structured_and_has_cooldown(monkeypatch, code):
    item = adapter()
    request = Mock(return_value={"code": code, "msg": "Too many requests"})
    monkeypatch.setattr("btc_futures_bot.exchanges.okx.request_json", request)
    for _ in range(2):
        with pytest.raises(ApiError) as error:
            item._public("/api/v5/public/mark-price")
        assert error.value.rate_limited and error.value.retry_after_seconds > 50
    assert request.call_count == 1


def test_partial_private_baseline_failure_does_not_retry_same_cycle(monkeypatch):
    item = adapter()
    _, private = streams(monkeypatch, item)
    monkeypatch.setattr(item, "has_credentials", lambda: True)

    def synchronize(provider):
        try:
            provider()
        except ApiError:
            pass
        return False

    private.synchronize.side_effect = synchronize
    baseline = Mock(side_effect=ApiError("positions request unavailable"))
    monkeypatch.setattr(item, "_fetch_private_baseline", baseline)
    snapshot = item.fetch_dashboard_snapshot()
    assert not snapshot["private_available"]
    assert baseline.call_count == 1


@pytest.mark.parametrize("rows", [
    [[0, "100", "101", "99", "100", "1", "1", "100", "1"]],
    [[0, "100", "101", "99", "nan", "1", "1", "100", "0"]],
    [[0, "100", "101", "99", "100", "1", "1", "100", "0"], [120000, "100", "101", "99", "100", "1", "1", "100", "0"]],
])
def test_ws_fallback_rejects_closed_nan_or_gapped_rest_history(monkeypatch, rows):
    item = adapter()
    market, _ = streams(monkeypatch, item)
    monkeypatch.setattr(item, "_server_timestamp_ms", lambda: int(rows[-1][0]) + 1000)
    monkeypatch.setattr("btc_futures_bot.exchanges.okx.request_json", Mock(return_value={"code": "0", "data": rows}))
    with pytest.raises(ApiError, match="candle history"):
        item.fetch_candles("1m", 100)
    market.seed_candles.assert_not_called()


@pytest.mark.parametrize("mark", [
    {"markPx": "100", "ts": "1000"}, {"markPx": "nan", "ts": "100000"}, {"markPx": "100"},
])
def test_ws_fallback_rejects_stale_invalid_or_undated_mark(monkeypatch, mark):
    item = adapter()
    market, _ = streams(monkeypatch, item)
    market.market_snapshot.return_value = None
    monkeypatch.setattr(item, "_server_timestamp_ms", lambda: 100000)
    monkeypatch.setattr("btc_futures_bot.exchanges.okx.request_json", Mock(return_value={"code": "0", "data": [mark]}))
    with pytest.raises(ApiError, match="stale mark"):
        item.fetch_mark_price()
    assert not item._last_rest_market


def test_close_stops_both_streams_and_prevents_restart(monkeypatch):
    item = adapter()
    market, private = streams(monkeypatch, item)
    item.close()
    market.close.assert_called_once()
    private.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        item.fetch_candles("1m", 100)


def test_hedge_short_positive_contract_count_is_short():
    item = adapter()
    result = item._private_view({}, [{"instId": "BTC-USDT-SWAP", "posSide": "short", "pos": "2"}], [])
    assert result["positions"][0]["side"] == "short"


def test_private_baseline_paginates_full_order_list(monkeypatch):
    item = adapter()
    pages = []

    def request(method, path, *, params=None):
        if path.endswith("orders-pending"):
            pages.append(dict(params))
            if "after" not in params:
                return {"code": "0", "data": [{"ordId": str(i)} for i in range(200, 100, -1)]}
            return {"code": "0", "data": [{"ordId": "100"}]}
        return {"code": "0", "data": []}

    monkeypatch.setattr(item, "_private", request)
    _, _, orders = item._fetch_private_baseline()
    assert len(orders["data"]) == 101
    assert pages[1]["after"] == "101"


def test_private_baseline_rejects_repeated_pagination(monkeypatch):
    item = adapter()
    monkeypatch.setattr(item, "_private", lambda *a, **k: {"code": "0", "data": [{"ordId": str(i)} for i in range(100)]})
    with pytest.raises(ApiError, match="ambiguous"):
        item._fetch_private_baseline()
