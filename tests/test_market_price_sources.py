from __future__ import annotations

from btc_futures_bot.exchanges.base import ExchangeSettings
from btc_futures_bot.exchanges.gate import GateAdapter
from btc_futures_bot.exchanges.okx import OkxAdapter
from btc_futures_bot.http_client import ApiError


def test_okx_dashboard_uses_configured_swap_mark_price(monkeypatch) -> None:
    adapter = OkxAdapter(
        ExchangeSettings(
            name="okx",
            environment="demo",
            base_url="https://openapi.okx.com",
            symbol="BTC-USDT-SWAP",
        )
    )
    calls: list[tuple[str, dict]] = []

    def request(method, url, *, params=None, **kwargs):
        calls.append((url, params or {}))
        return {
            "code": "0",
            "data": [
                {
                    "instType": "SWAP",
                    "instId": "BTC-USDT-SWAP",
                    "markPx": "78400.125",
                    "ts": "1700000000000",
                }
            ],
        }

    monkeypatch.setattr("btc_futures_bot.exchanges.okx.request_json", request)

    snapshot = adapter.fetch_dashboard_snapshot()

    assert calls == [
        (
            "https://openapi.okx.com/api/v5/public/mark-price",
            {"instType": "SWAP", "instId": "BTC-USDT-SWAP"},
        )
    ]
    assert snapshot["market"]["mark_price_raw"] == "78400.125"
    assert snapshot["market"]["price_source"] == "OKX 永续合约标记价格"


def test_okx_dashboard_marks_private_api_unavailable_after_auth_failure(monkeypatch) -> None:
    settings = ExchangeSettings(
        name="okx",
        environment="demo",
        base_url="https://openapi.okx.com",
        symbol="BTC-USDT-SWAP",
        api_key_env="TEST_OKX_API_KEY",
        api_secret_env="TEST_OKX_API_SECRET",
        passphrase_env="TEST_OKX_API_PASSPHRASE",
    )
    adapter = OkxAdapter(settings)
    monkeypatch.setenv(settings.api_key_env, "test-key")
    monkeypatch.setenv(settings.api_secret_env, "test-secret")
    monkeypatch.setenv(settings.passphrase_env, "test-passphrase")
    monkeypatch.setattr(
        "btc_futures_bot.exchanges.okx.request_json",
        lambda *args, **kwargs: {
            "code": "0",
            "data": [{"instId": settings.symbol, "markPx": "78400.125"}],
        },
    )
    monkeypatch.setattr(
        adapter,
        "_private",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Invalid OK-ACCESS-KEY")
        ),
    )

    snapshot = adapter.fetch_dashboard_snapshot()

    assert snapshot["private_available"] is False
    assert snapshot["private_error"] == "Invalid OK-ACCESS-KEY"


def test_okx_private_request_signs_with_synchronized_server_time(monkeypatch) -> None:
    settings = ExchangeSettings(
        name="okx",
        environment="demo",
        base_url="https://openapi.okx.com",
        symbol="BTC-USDT-SWAP",
        api_key_env="TEST_OKX_TIME_KEY",
        api_secret_env="TEST_OKX_TIME_SECRET",
        passphrase_env="TEST_OKX_TIME_PASSPHRASE",
    )
    adapter = OkxAdapter(settings)
    monkeypatch.setenv(settings.api_key_env, "test-key")
    monkeypatch.setenv(settings.api_secret_env, "test-secret")
    monkeypatch.setenv(settings.passphrase_env, "test-passphrase")
    monkeypatch.setattr(adapter, "now_ms", lambda: 900_000)
    monotonic_values = iter((10.0, 10.0))
    monkeypatch.setattr(
        "btc_futures_bot.exchanges.okx.time.monotonic",
        lambda: next(monotonic_values),
    )
    private_headers: list[dict[str, str]] = []

    def request(method, url, *, headers=None, **kwargs):
        if url.endswith("/api/v5/public/time"):
            return {"code": "0", "data": [{"ts": "1000000"}]}
        private_headers.append(dict(headers or {}))
        return {"code": "0", "data": [{"totalEq": "123.45"}]}

    monkeypatch.setattr("btc_futures_bot.exchanges.okx.request_json", request)

    assert adapter.fetch_equity() == 123.45
    assert private_headers[0]["OK-ACCESS-TIMESTAMP"] == "1970-01-01T00:16:40.000Z"
    assert private_headers[0]["x-simulated-trading"] == "1"


def test_okx_private_request_resynchronizes_once_on_50102(monkeypatch) -> None:
    settings = ExchangeSettings(
        name="okx",
        environment="demo",
        base_url="https://openapi.okx.com",
        symbol="BTC-USDT-SWAP",
        api_key_env="TEST_OKX_RETRY_KEY",
        api_secret_env="TEST_OKX_RETRY_SECRET",
        passphrase_env="TEST_OKX_RETRY_PASSPHRASE",
    )
    adapter = OkxAdapter(settings)
    monkeypatch.setenv(settings.api_key_env, "test-key")
    monkeypatch.setenv(settings.api_secret_env, "test-secret")
    monkeypatch.setenv(settings.passphrase_env, "test-passphrase")
    monkeypatch.setattr(adapter, "_ensure_server_time", lambda: None)
    timestamps = iter(
        ("2026-09-04T13:41:00.000Z", "2026-09-04T13:41:31.500Z")
    )
    monkeypatch.setattr(adapter, "_server_timestamp", lambda: next(timestamps))
    forced_syncs: list[bool] = []
    monkeypatch.setattr(
        adapter,
        "_sync_server_time",
        lambda *, force=True: forced_syncs.append(force) or 31_500,
    )
    signed_timestamps: list[str] = []

    def request(method, url, *, headers=None, **kwargs):
        signed_timestamps.append(headers["OK-ACCESS-TIMESTAMP"])
        if len(signed_timestamps) == 1:
            raise ApiError(
                'HTTP 401: {"msg":"Timestamp request expired","code":"50102"}',
                status_code=401,
                api_code="50102",
            )
        return {"code": "0", "data": [{"totalEq": "123.45"}]}

    monkeypatch.setattr("btc_futures_bot.exchanges.okx.request_json", request)

    assert adapter.fetch_equity() == 123.45
    assert forced_syncs == [True]
    assert signed_timestamps == [
        "2026-09-04T13:41:00.000Z",
        "2026-09-04T13:41:31.500Z",
    ]


def test_gate_dashboard_uses_configured_contract_mark_price(monkeypatch) -> None:
    adapter = GateAdapter(
        ExchangeSettings(
            name="gate",
            environment="testnet",
            base_url="https://api-testnet.gateapi.io/api/v4",
            symbol="BTC_USDT",
            settle="usdt",
        )
    )
    calls: list[tuple[str, dict]] = []

    def request(method, path, *, params=None, **kwargs):
        calls.append((path, params or {}))
        return [{"contract": "BTC_USDT", "mark_price": "78401.75", "last": "78402.1"}]

    monkeypatch.setattr(adapter, "_request", request)

    snapshot = adapter.fetch_dashboard_snapshot()

    assert calls == [("/futures/usdt/tickers", {"contract": "BTC_USDT"})]
    assert snapshot["market"]["mark_price_raw"] == "78401.75"
    assert snapshot["market"]["last_trade_price_raw"] == "78402.1"
    assert snapshot["market"]["price_source"] == "Gate USDT 永续合约标记价格"
