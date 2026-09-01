from __future__ import annotations

import json
import time

import pytest

from btc_futures_bot.binance_user_stream import BinanceUserDataStream
from btc_futures_bot.exchanges.base import ExchangeSettings
from btc_futures_bot.exchanges.binance import BinanceAdapter


def _seed() -> dict:
    return {
        "account": {
            "totalWalletBalance": "25.00000000",
            "availableBalance": "25.00000000",
            "totalUnrealizedProfit": "0",
            "totalMarginBalance": "25.00000000",
            "assets": [
                {
                    "asset": "USDT",
                    "walletBalance": "25.00000000",
                    "availableBalance": "25.00000000",
                    "crossWalletBalance": "25.00000000",
                }
            ],
        },
        "positions": [
            {
                "symbol": "BTCUSDT",
                "positionSide": "BOTH",
                "positionAmt": "0",
                "entryPrice": "0",
                "unRealizedProfit": "0",
                "leverage": "20",
                "maxNotionalValue": "100000000",
            }
        ],
        "orders": [],
        "algo_orders": [],
    }


def _stream() -> BinanceUserDataStream:
    stream = BinanceUserDataStream(
        "BTCUSDT",
        "production",
        "https://fapi.binance.com",
        lambda: "api-key",
        _seed,
    )
    with stream._condition:
        stream._connected = True
        stream._last_transport_at = time.monotonic()
    stream.seed_snapshot(_seed())
    return stream


def test_private_stream_uses_new_private_path_and_applies_account_deltas(monkeypatch) -> None:
    stream = _stream()
    monkeypatch.setattr(stream, "start", lambda **_kwargs: True)

    assert stream.stream_base == "wss://fstream.binance.com/private"
    stream.process_message(
        json.dumps(
            {
                "e": "ACCOUNT_UPDATE",
                "a": {
                    "B": [{"a": "USDT", "wb": "24.9", "cw": "21.0", "bc": "-0.1"}],
                    "P": [
                        {
                            "s": "BTCUSDT",
                            "pa": "0.001",
                            "ep": "78000",
                            "bep": "78050",
                            "up": "0.25",
                            "mt": "isolated",
                            "iw": "3.9",
                            "ps": "BOTH",
                        }
                    ],
                },
            }
        )
    )

    snapshot = stream.snapshot()

    assert snapshot["account"]["totalWalletBalance"] == "24.9"
    assert snapshot["account"]["availableBalance"] == "21.0"
    assert snapshot["account"]["totalUnrealizedProfit"] == "0.25"
    assert snapshot["account"]["totalMarginBalance"] == "25.15"
    assert snapshot["positions"][0]["positionAmt"] == "0.001"
    assert snapshot["positions"][0]["entryPrice"] == "78000"


def test_private_stream_tracks_regular_and_algo_order_lifecycle(monkeypatch) -> None:
    stream = _stream()
    monkeypatch.setattr(stream, "start", lambda **_kwargs: True)
    regular_new = {
        "e": "ORDER_TRADE_UPDATE",
        "o": {
            "s": "BTCUSDT",
            "c": "btcbot-entry-1",
            "S": "BUY",
            "o": "LIMIT",
            "f": "GTC",
            "q": "0.001",
            "p": "78000",
            "ap": "0",
            "sp": "0",
            "x": "NEW",
            "X": "NEW",
            "i": 101,
            "z": "0",
            "R": False,
            "cp": False,
            "ps": "BOTH",
            "T": 1_700_000_000_000,
        },
    }
    stream.process_message(json.dumps(regular_new))
    assert len(stream.snapshot()["orders"]) == 1
    assert stream.wait_for_order("btcbot-entry-1", timeout=0) is not None

    regular_new["o"].update({"x": "TRADE", "X": "FILLED", "z": "0.001", "ap": "78000"})
    stream.process_message(json.dumps(regular_new))
    assert stream.snapshot()["orders"] == []
    assert stream.wait_for_order("btcbot-entry-1", timeout=0)["status"] == "FILLED"

    algo = {
        "e": "ALGO_UPDATE",
        "o": {
            "algoId": 202,
            "clientAlgoId": "btcbot-stop-1",
            "algoType": "CONDITIONAL",
            "orderType": "STOP_MARKET",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "positionSide": "BOTH",
            "triggerPrice": "77000",
            "closePosition": True,
            "workingType": "MARK_PRICE",
            "algoStatus": "NEW",
        },
    }
    stream.process_message(json.dumps(algo))
    assert len(stream.snapshot()["algo_orders"]) == 1
    assert stream.wait_for_algo_order("btcbot-stop-1", timeout=0) is not None

    algo["o"]["algoStatus"] = "FINISHED"
    stream.process_message(json.dumps(algo))
    assert stream.snapshot()["algo_orders"] == []
    assert stream.wait_for_algo_order("btcbot-stop-1", timeout=0)["algoStatus"] == "FINISHED"


def test_private_stream_maps_real_binance_algo_trigger_price_field(monkeypatch) -> None:
    stream = _stream()
    monkeypatch.setattr(stream, "start", lambda **_kwargs: True)

    stream.process_message(
        json.dumps(
            {
                "e": "ALGO_UPDATE",
                "T": 1_788_295_142_173,
                "E": 1_788_295_142_180,
                "o": {
                    "caid": "btcbot-stop-live-shape",
                    "aid": 2_000_001_402_291_671,
                    "at": "CONDITIONAL",
                    "o": "STOP_MARKET",
                    "s": "BTCUSDT",
                    "S": "BUY",
                    "ps": "BOTH",
                    "q": "0",
                    "X": "NEW",
                    "tp": "77683.80",
                    "p": "0",
                    "wt": "MARK_PRICE",
                    "cp": True,
                    "R": True,
                },
            }
        )
    )

    cached = stream.wait_for_algo_order("btcbot-stop-live-shape", timeout=0)

    assert cached is not None
    assert cached["triggerPrice"] == "77683.80"
    assert cached["clientAlgoId"] == "btcbot-stop-live-shape"
    assert cached["orderType"] == "STOP_MARKET"
    assert cached["workingType"] == "MARK_PRICE"
    assert cached["closePosition"] is True


def test_listen_key_management_uses_api_key_without_signature(monkeypatch) -> None:
    stream = _stream()
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return {"listenKey": "secret-listen-key"}

    monkeypatch.setattr("btc_futures_bot.binance_user_stream.request_json", request)

    assert stream._ensure_listen_key() == "secret-listen-key"
    method, url, kwargs = calls[0]
    assert method == "POST"
    assert url == "https://fapi.binance.com/fapi/v1/listenKey"
    assert kwargs["headers"] == {"X-MBX-APIKEY": "api-key"}
    assert "signature" not in str(kwargs)


def test_adapter_reads_live_position_from_private_stream_without_signed_rest(monkeypatch) -> None:
    adapter = BinanceAdapter(
        ExchangeSettings(
            name="binance",
            environment="production",
            base_url="https://fapi.binance.com",
            symbol="BTCUSDT",
            api_key_env="TEST_BINANCE_KEY",
            api_secret_env="TEST_BINANCE_SECRET",
            websocket_enabled=True,
        )
    )

    class PrivateCache:
        def snapshot(self, *, wait_seconds=0):
            return {
                "account": {},
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "positionSide": "BOTH",
                        "positionAmt": "0.001",
                        "entryPrice": "78000",
                    }
                ],
                "orders": [],
                "algo_orders": [],
            }

    class MarketCache:
        def start(self):
            return True

        def mark_price(self):
            return 78_125.0, 78_120.0, 1_700_000_000_000

    adapter._private_stream = PrivateCache()
    adapter._market_stream = MarketCache()
    monkeypatch.setattr(
        adapter,
        "_signed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("periodic position reconciliation must not call REST")
        ),
    )

    assert adapter.fetch_live_position() == {
        "side": "long",
        "quantity": pytest.approx(0.001),
        "entry_price": pytest.approx(78_000.0),
        "mark_price": pytest.approx(78_125.0),
    }


def test_private_bootstrap_reuses_account_positions_and_never_calls_position_risk(
    monkeypatch,
) -> None:
    adapter = BinanceAdapter(
        ExchangeSettings(
            name="binance",
            environment="production",
            base_url="https://fapi.binance.com",
            symbol="BTCUSDT",
            api_key_env="TEST_BINANCE_KEY",
            api_secret_env="TEST_BINANCE_SECRET",
            websocket_enabled=True,
        )
    )
    calls: list[str] = []

    def signed(_method, path, _params=None):
        calls.append(path)
        if path == "/fapi/v2/account":
            return _seed()["account"] | {"positions": _seed()["positions"]}
        return []

    monkeypatch.setattr(adapter, "_signed", signed)

    snapshot = adapter._fetch_private_snapshot_rest()

    assert snapshot["positions"] == _seed()["positions"]
    assert "/fapi/v2/positionRisk" not in calls
