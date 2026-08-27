from __future__ import annotations

from btc_futures_bot.exchanges.base import ExchangeSettings
from btc_futures_bot.exchanges.gate import GateAdapter
from btc_futures_bot.exchanges.okx import OkxAdapter


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
