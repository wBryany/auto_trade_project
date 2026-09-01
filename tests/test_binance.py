from __future__ import annotations

from decimal import Decimal

import pytest

from btc_futures_bot.exchanges.base import ExchangeSettings
from btc_futures_bot.exchanges.binance import BinanceAdapter
from btc_futures_bot.http_client import ApiError
from btc_futures_bot.models import Candle, OrderRequest, Position


def _adapter() -> BinanceAdapter:
    return BinanceAdapter(
        ExchangeSettings(
            name="binance",
            environment="testnet",
            base_url="https://demo-fapi.binance.com",
            symbol="BTCUSDT",
            api_key_env="TEST_BINANCE_KEY",
            api_secret_env="TEST_BINANCE_SECRET",
        )
    )


def test_binance_websocket_cache_bootstraps_rest_only_once(monkeypatch) -> None:
    adapter = _adapter()

    class FakeStream:
        def __init__(self) -> None:
            self.rows: list[Candle] = []

        def start(self) -> bool:
            return True

        def has_history(self, _interval: str, minimum: int) -> bool:
            return len(self.rows) >= minimum

        def candles(self, _interval: str, limit: int) -> list[Candle]:
            return self.rows[-limit:]

        def seed_candles(self, _interval: str, candles: list[Candle]) -> None:
            self.rows = list(candles)

        def healthy(self) -> bool:
            return True

    stream = FakeStream()
    adapter._market_stream = stream
    rest_rows = [
        Candle(index, 100.0, 101.0, 99.0, 100.5, 1.0)
        for index in range(300)
    ]
    reads = []

    def fetch_rest(interval: str, limit: int) -> list[Candle]:
        reads.append((interval, limit))
        return rest_rows

    monkeypatch.setattr(adapter, "_fetch_candles_rest", fetch_rest)

    first = adapter.fetch_candles("1m", 300)
    second = adapter.fetch_candles("1m", 300)

    assert first == rest_rows
    assert second == rest_rows
    assert reads == [("1m", 300)]


def test_binance_mark_price_prefers_websocket_without_rest(monkeypatch) -> None:
    adapter = _adapter()

    class FakeStream:
        def start(self) -> bool:
            return True

        def mark_price(self):
            return 65_001.25, 65_000.5, 1_700_000_000_000

    adapter._market_stream = FakeStream()
    monkeypatch.setattr(
        "btc_futures_bot.exchanges.binance.request_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("REST must not be called")),
    )

    assert adapter.fetch_mark_price() == 65_001.25


def _exchange_info() -> dict:
    return {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "contractType": "PERPETUAL",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {
                        "filterType": "MARKET_LOT_SIZE",
                        "minQty": "0.001",
                        "maxQty": "1000",
                        "stepSize": "0.001",
                    },
                    {"filterType": "MIN_NOTIONAL", "notional": "50"},
                ],
            }
        ]
    }


def _set_symbol_rules(adapter: BinanceAdapter) -> None:
    adapter._symbol_rules = {
        "quantity_step": Decimal("0.001"),
        "min_quantity": Decimal("0.001"),
        "max_quantity": Decimal("1000"),
        "price_tick": Decimal("0.1"),
        "min_notional": Decimal("50"),
    }


def _algo_row(params: dict, algo_id: int) -> dict:
    return {
        "algoId": algo_id,
        "clientAlgoId": params["clientAlgoId"],
        "algoType": params["algoType"],
        "orderType": params["type"],
        "symbol": params["symbol"],
        "side": params["side"],
        "positionSide": params["positionSide"],
        "triggerPrice": params["triggerPrice"],
        "closePosition": True,
        "workingType": params["workingType"],
        "algoStatus": "NEW",
    }


def test_binance_rejects_mismatched_environment_endpoint() -> None:
    try:
        BinanceAdapter(
            ExchangeSettings(
                name="binance",
                environment="production",
                base_url="https://demo-fapi.binance.com",
                symbol="BTCUSDT",
            )
        )
    except ValueError as error:
        assert "endpoint" in str(error)
    else:
        raise AssertionError("production adapter accepted the testnet endpoint")


def test_binance_normalizes_quantity_and_trigger_prices(monkeypatch) -> None:
    adapter = _adapter()
    monkeypatch.setattr("btc_futures_bot.exchanges.binance.request_json", lambda *args, **kwargs: _exchange_info())

    assert adapter.normalize_order_quantity(0.0019, 65_000) == 0.001
    assert adapter.normalize_trigger_price(64_999.99, "SELL") == 64_999.9
    assert adapter.normalize_trigger_price(65_000.01, "BUY") == 65_000.1


def test_binance_market_order_waits_for_result(monkeypatch) -> None:
    adapter = _adapter()
    captured = {}

    def signed(method, path, params=None):
        captured.update({"method": method, "path": path, "params": params})
        return {"status": "FILLED", "executedQty": "0.001", "avgPrice": "65000"}

    monkeypatch.setattr(adapter, "_signed", signed)
    payload = adapter.place_market_order(OrderRequest("buy", 0.001, False, "btcbot-entry-test"))

    assert captured["params"]["newOrderRespType"] == "RESULT"
    assert captured["params"]["newClientOrderId"] == "btcbot-entry-test"
    assert adapter.market_fill(payload, fallback_price=1) == (0.001, 65_000.0)


def test_binance_market_fill_reconciles_missing_post_details(monkeypatch) -> None:
    adapter = _adapter()
    calls = []

    def signed(method, path, params=None):
        calls.append((method, path, params))
        return {
            "status": "FILLED",
            "orderId": 123,
            "executedQty": "0.001",
            "avgPrice": "0",
            "cumQuote": "65.00000",
        }

    monkeypatch.setattr(adapter, "_signed", signed)

    fill = adapter.market_fill(
        {"status": "FILLED", "orderId": 123},
        fallback_price=64_999,
    )

    assert fill == (0.001, 65_000.0)
    assert calls == [("GET", "/fapi/v1/order", {"symbol": "BTCUSDT", "orderId": 123})]


def test_binance_market_fill_retries_transient_order_not_found(monkeypatch) -> None:
    adapter = _adapter()
    calls = []
    sleeps = []

    def signed(method, path, params=None):
        calls.append((method, path, params))
        if len(calls) == 1:
            raise ApiError('HTTP 400: {"code":-2013,"msg":"Order does not exist."}')
        return {
            "status": "FILLED",
            "orderId": 123,
            "executedQty": "0.001",
            "avgPrice": "65000",
        }

    monkeypatch.setattr(adapter, "_signed", signed)
    monkeypatch.setattr("btc_futures_bot.exchanges.binance.time.sleep", sleeps.append)

    fill = adapter.market_fill(
        {"status": "NEW", "orderId": 123, "origQty": "0.001", "side": "BUY"},
        fallback_price=64_999,
    )

    assert fill == (0.001, 65_000.0)
    assert [call[1] for call in calls] == ["/fapi/v1/order", "/fapi/v1/order"]
    assert sleeps == [0.25]


def test_binance_market_fill_recovers_exact_entry_position_after_visibility_timeout(monkeypatch) -> None:
    adapter = _adapter()

    def signed(method, path, params=None):
        if path == "/fapi/v1/order":
            raise ApiError('HTTP 400: {"code":-2013,"msg":"Order does not exist."}')
        if path in {"/fapi/v1/allOrders", "/fapi/v1/userTrades"}:
            return []
        raise AssertionError((method, path, params))

    monkeypatch.setattr(adapter, "_signed", signed)
    monkeypatch.setattr("btc_futures_bot.exchanges.binance.time.sleep", lambda _: None)
    monkeypatch.setattr(
        adapter,
        "fetch_live_position",
        lambda: {"side": "long", "quantity": 0.001, "entry_price": 65_001.25, "mark_price": 65_002},
    )

    fill = adapter.market_fill(
        {
            "status": "NEW",
            "orderId": 123,
            "clientOrderId": "btcbot-entry-test",
            "origQty": "0.001",
            "side": "BUY",
            "reduceOnly": False,
        },
        fallback_price=64_999,
    )

    assert fill == (0.001, 65_001.25)


def test_binance_aggregates_recent_position_close_fills(monkeypatch) -> None:
    adapter = _adapter()
    opened_at = adapter.now_ms() - 30_000
    rows = [
        {
            "id": 1,
            "orderId": 10,
            "side": "BUY",
            "qty": "0.002",
            "quoteQty": "156.0",
            "price": "78000",
            "realizedPnl": "0",
            "commission": "0.0312",
            "commissionAsset": "USDT",
            "time": opened_at + 1_000,
        },
        {
            "id": 2,
            "orderId": 11,
            "side": "SELL",
            "qty": "0.001",
            "quoteQty": "79.0",
            "price": "79000",
            "realizedPnl": "1.0",
            "commission": "0.0158",
            "commissionAsset": "USDT",
            "time": opened_at + 10_000,
        },
        {
            "id": 3,
            "orderId": 12,
            "side": "SELL",
            "qty": "0.001",
            "quoteQty": "80.0",
            "price": "80000",
            "realizedPnl": "2.0",
            "commission": "0.0160",
            "commissionAsset": "USDT",
            "time": opened_at + 20_000,
        },
    ]

    def signed(method: str, path: str, params: dict | None = None) -> list[dict]:
        assert method == "GET"
        assert path == "/fapi/v1/userTrades"
        assert params and params["symbol"] == "BTCUSDT"
        return rows

    monkeypatch.setattr(adapter, "_signed", signed)
    fill = adapter.fetch_latest_close_fill(
        Position("long", 0.002, 78_000.0, 77_000.0, 80_000.0, opened_at),
        after_ms=opened_at,
    )

    assert fill is not None
    assert fill["quantity"] == pytest.approx(0.002)
    assert fill["price"] == pytest.approx(79_500.0)
    assert fill["realized_pnl"] == pytest.approx(3.0)
    assert fill["commission"] == pytest.approx(0.0318)
    assert fill["commission_assets"] == ["USDT"]


def test_binance_protection_uses_only_server_side_stop(monkeypatch) -> None:
    adapter = _adapter()
    _set_symbol_rules(adapter)
    calls = []
    open_orders = []

    def signed(method, path, params=None):
        calls.append((method, path, params))
        if method == "POST":
            row = _algo_row(params, len(open_orders) + 1)
            open_orders.append(row)
            return {"algoId": row["algoId"]}
        if method == "GET" and path == "/fapi/v1/openAlgoOrders":
            return list(open_orders)
        raise AssertionError((method, path))

    monkeypatch.setattr(adapter, "_signed", signed)
    position = Position("long", 0.001, 65_000, 64_700, 65_750, 1)
    result = adapter.place_protection_orders(
        position,
        stop_client_id="btcbot-stop-test",
        take_profit_client_id="btcbot-tp-test",
    )

    assert result["stop"]["algoId"] == 1
    assert result["take_profit"] is None
    assert result["confirmed"] is True
    assert result["mode"] == "stop_only_dynamic_profit_exit"
    submitted = [call for call in calls if call[0] == "POST"]
    confirmations = [call for call in calls if call[0] == "GET"]
    assert [call[2]["type"] for call in submitted] == ["STOP_MARKET"]
    assert all(call[1] == "/fapi/v1/algoOrder" for call in submitted)
    assert all(call[2]["closePosition"] == "true" for call in submitted)
    assert all(call[2]["positionSide"] == "BOTH" for call in submitted)
    assert len(confirmations) == 1


def test_binance_protection_serializes_trigger_at_exchange_tick_precision(monkeypatch) -> None:
    adapter = _adapter()
    _set_symbol_rules(adapter)
    open_orders = []
    submitted_prices = []

    def signed(method, path, params=None):
        if method == "POST":
            submitted_prices.append(params["triggerPrice"])
            row = _algo_row(params, len(open_orders) + 1)
            open_orders.append(row)
            return {"algoId": row["algoId"]}
        if method == "GET" and path == "/fapi/v1/openAlgoOrders":
            return list(open_orders)
        raise AssertionError((method, path, params))

    monkeypatch.setattr(adapter, "_signed", signed)
    position = Position(
        "short",
        0.001,
        77_950.9,
        78_157.57217988612,
        77_434.21955028469,
        1,
    )

    adapter.place_protection_orders(
        position,
        stop_client_id="btcbot-stop-precision",
        take_profit_client_id="btcbot-tp-precision",
    )

    assert submitted_prices == ["78157.6"]


def test_binance_protection_recovers_ambiguous_submit_by_client_id(monkeypatch) -> None:
    adapter = _adapter()
    _set_symbol_rules(adapter)
    position = Position(
        "long",
        0.001,
        65_000,
        64_700,
        65_750,
        1,
        stop_client_id="btcbot-stop-recover",
    )
    calls = []
    open_orders = []

    def signed(method, path, params=None):
        calls.append((method, path, params))
        if method == "POST":
            open_orders.append(_algo_row(params, 91))
            raise ApiError("network error POST /fapi/v1/algoOrder: connection reset")
        if method == "GET" and path == "/fapi/v1/openAlgoOrders":
            return list(open_orders)
        raise AssertionError((method, path))

    monkeypatch.setattr(adapter, "_signed", signed)

    result = adapter.place_stop_order(position)

    assert result["algoId"] == 91
    assert [call[0] for call in calls].count("POST") == 1
    assert [call[0] for call in calls].count("GET") == 1


def test_binance_protection_rejects_unconfirmed_stop(monkeypatch) -> None:
    adapter = _adapter()
    _set_symbol_rules(adapter)
    position = Position("long", 0.001, 65_000, 64_700, 65_750, 1)
    open_orders = []
    submitted_types = []

    def signed(method, path, params=None):
        if method == "POST":
            submitted_types.append(params["type"])
            return {"algoId": 1}
        if method == "GET" and path == "/fapi/v1/openAlgoOrders":
            return list(open_orders)
        raise AssertionError((method, path))

    monkeypatch.setattr(adapter, "_signed", signed)
    monkeypatch.setattr("btc_futures_bot.exchanges.binance.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="STOP_MARKET.*open server-side"):
        adapter.place_protection_orders(
            position,
            stop_client_id="btcbot-stop-test",
            take_profit_client_id="btcbot-tp-test",
        )

    assert submitted_types == ["STOP_MARKET"]


def test_binance_private_failure_is_not_reported_as_connected(monkeypatch) -> None:
    adapter = _adapter()
    monkeypatch.setenv("TEST_BINANCE_KEY", "configured")
    monkeypatch.setenv("TEST_BINANCE_SECRET", "configured")
    monkeypatch.setattr(
        "btc_futures_bot.exchanges.binance.request_json",
        lambda *args, **kwargs: {
            "symbol": "BTCUSDT",
            "markPrice": "65001.25000000",
            "indexPrice": "65002.50000000",
            "time": 1_700_000_000_000,
        },
    )

    def failed_private_call(*args, **kwargs):
        raise RuntimeError("invalid credentials")

    monkeypatch.setattr(adapter, "_signed", failed_private_call)
    snapshot = adapter.fetch_dashboard_snapshot()

    assert snapshot["market"]["mark_price"] == 65_001.25
    assert snapshot["market"]["mark_price_raw"] == "65001.25000000"
    assert snapshot["market"]["price_endpoint"] == "/fapi/v1/premiumIndex"
    assert snapshot["private_available"] is False
    assert "invalid credentials" in snapshot["private_error"]
    assert snapshot["private_transient"] is False
    assert snapshot["private_retryable"] is False


def test_binance_dashboard_preserves_balance_precision_and_estimates_order_capacity(monkeypatch) -> None:
    adapter = _adapter()
    _set_symbol_rules(adapter)
    monkeypatch.setenv("TEST_BINANCE_KEY", "configured")
    monkeypatch.setenv("TEST_BINANCE_SECRET", "configured")
    monkeypatch.setattr(
        "btc_futures_bot.exchanges.binance.request_json",
        lambda *args, **kwargs: {
            "symbol": "BTCUSDT",
            "markPrice": "77500.0",
            "indexPrice": "77502.0",
            "time": 1_700_000_000_000,
        },
    )

    def signed(method, path, params=None):
        if path == "/fapi/v2/account":
            return {
                "totalWalletBalance": "27.39737452",
                "availableBalance": "27.39737452",
                "totalUnrealizedProfit": "0.00000000",
                "totalMarginBalance": "27.39737452",
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "positionSide": "BOTH",
                        "positionAmt": "0.000",
                        "markPrice": "77500.0",
                        "leverage": "20",
                        "maxNotionalValue": "100000000",
                    }
                ],
            }
        if path in {"/fapi/v1/openOrders", "/fapi/v1/openAlgoOrders"}:
            return []
        raise AssertionError((method, path, params))

    monkeypatch.setattr(adapter, "_signed", signed)
    snapshot = adapter.fetch_dashboard_snapshot()

    assert snapshot["private_available"] is True
    assert snapshot["account"]["wallet_balance_raw"] == "27.39737452"
    assert snapshot["account"]["available_balance_raw"] == "27.39737452"
    assert snapshot["order_limits"]["effective_min_quantity_raw"] == "0.001"
    assert snapshot["order_limits"]["effective_min_notional_raw"] == "77.5000"
    assert snapshot["order_limits"]["estimated_max_open_quantity_raw"] == "0.007"
    assert snapshot["order_limits"]["estimated_max_open_notional_raw"] == "542.5000"
    assert snapshot["order_limits"]["current_leverage_raw"] == "20"


def test_binance_signed_request_resyncs_server_time_on_1021(monkeypatch) -> None:
    adapter = _adapter()
    monkeypatch.setenv("TEST_BINANCE_KEY", "configured")
    monkeypatch.setenv("TEST_BINANCE_SECRET", "configured")
    monkeypatch.setattr(adapter, "now_ms", lambda: 900_000)
    monotonic_values = iter((10.0, 10.0, 20.0, 20.0))
    monkeypatch.setattr(
        "btc_futures_bot.exchanges.binance.time.monotonic",
        lambda: next(monotonic_values),
    )
    account_timestamps = []
    server_times = iter((1_000_000, 1_001_000))

    def request(method, url, *, params=None, headers=None, body=None, **kwargs):
        if url.endswith("/fapi/v1/time"):
            return {"serverTime": next(server_times)}
        account_timestamps.append(params["timestamp"])
        if len(account_timestamps) == 1:
            raise ApiError('HTTP 400: {"code":-1021,"msg":"timestamp outside recvWindow"}')
        return {"totalWalletBalance": "123.45"}

    monkeypatch.setattr("btc_futures_bot.exchanges.binance.request_json", request)

    assert adapter.fetch_equity() == 123.45
    assert account_timestamps == [1_000_000, 1_001_000]


def test_binance_signed_get_retries_with_a_fresh_signature(monkeypatch) -> None:
    adapter = _adapter()
    monkeypatch.setenv("TEST_BINANCE_KEY", "configured")
    monkeypatch.setenv("TEST_BINANCE_SECRET", "configured")
    monkeypatch.setattr(adapter, "_ensure_server_time", lambda: None)
    timestamps = iter((1_000_000, 1_000_001))
    monkeypatch.setattr(adapter, "_server_timestamp_ms", lambda: next(timestamps))
    calls = []

    def request(method, url, *, params=None, **kwargs):
        calls.append((dict(params or {}), kwargs))
        if len(calls) == 1:
            raise ApiError("network error: temporary TLS timeout")
        return {"totalWalletBalance": "123.45"}

    monkeypatch.setattr("btc_futures_bot.exchanges.binance.request_json", request)
    monkeypatch.setattr("btc_futures_bot.exchanges.binance.time.sleep", lambda _delay: None)

    assert adapter.fetch_equity() == 123.45
    assert [call[0]["timestamp"] for call in calls] == [1_000_000, 1_000_001]
    assert calls[0][0]["signature"] != calls[1][0]["signature"]
    assert all(call[1]["max_attempts"] == 1 for call in calls)


def test_binance_signed_post_does_not_retry_an_ambiguous_transport_failure(monkeypatch) -> None:
    adapter = _adapter()
    monkeypatch.setenv("TEST_BINANCE_KEY", "configured")
    monkeypatch.setenv("TEST_BINANCE_SECRET", "configured")
    monkeypatch.setattr(adapter, "_ensure_server_time", lambda: None)
    monkeypatch.setattr(adapter, "_server_timestamp_ms", lambda: 1_000_000)
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        raise ApiError("network error: response unknown")

    monkeypatch.setattr("btc_futures_bot.exchanges.binance.request_json", request)

    with pytest.raises(ApiError, match="response unknown"):
        adapter._signed("POST", "/fapi/v1/order", {"symbol": "BTCUSDT"})

    assert len(calls) == 1


def test_binance_time_sync_anchors_to_response_completion(monkeypatch) -> None:
    adapter = _adapter()
    monkeypatch.setattr(adapter, "now_ms", lambda: 1_010_000)
    monotonic_values = iter((50.0, 50.0))
    monkeypatch.setattr(
        "btc_futures_bot.exchanges.binance.time.monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        "btc_futures_bot.exchanges.binance.request_json",
        lambda *args, **kwargs: {"serverTime": 1_000_000},
    )

    assert adapter._sync_server_time() == -10_000
    assert adapter._server_timestamp_ms() == 1_000_000


def test_binance_signed_get_does_not_retry_rate_limit(monkeypatch) -> None:
    adapter = _adapter()
    monkeypatch.setenv("TEST_BINANCE_KEY", "configured")
    monkeypatch.setenv("TEST_BINANCE_SECRET", "configured")
    monkeypatch.setattr(adapter, "_ensure_server_time", lambda: None)
    monkeypatch.setattr(adapter, "_server_timestamp_ms", lambda: 1_000_000)
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        raise ApiError("HTTP 429", status_code=429, retry_at=9_999_999_999)

    monkeypatch.setattr("btc_futures_bot.exchanges.binance.request_json", request)

    with pytest.raises(ApiError, match="429"):
        adapter.fetch_equity()

    assert len(calls) == 1


def test_binance_private_network_failure_is_marked_transient(monkeypatch) -> None:
    adapter = _adapter()
    monkeypatch.setenv("TEST_BINANCE_KEY", "configured")
    monkeypatch.setenv("TEST_BINANCE_SECRET", "configured")
    monkeypatch.setattr(
        "btc_futures_bot.exchanges.binance.request_json",
        lambda *args, **kwargs: {
            "symbol": "BTCUSDT",
            "markPrice": "65001.25",
            "indexPrice": "65002.5",
            "time": 1_700_000_000_000,
        },
    )
    monkeypatch.setattr(
        adapter,
        "_signed",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ApiError("network error: TLS handshake timed out")
        ),
    )

    snapshot = adapter.fetch_dashboard_snapshot()

    assert snapshot["private_available"] is False
    assert snapshot["private_transient"] is True
    assert snapshot["private_retryable"] is True


def test_binance_rate_limit_is_transient_for_display_but_not_entry_retry() -> None:
    error = ApiError("HTTP 429 rate limited", status_code=429)

    assert BinanceAdapter._private_error_is_transient(error) is True
    assert BinanceAdapter._private_error_is_retryable(error) is False
