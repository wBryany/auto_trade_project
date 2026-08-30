from __future__ import annotations

import json
from unittest.mock import patch

from btc_futures_bot.binance_stream import BinanceMarketStream
from btc_futures_bot.models import Candle


def test_combined_stream_updates_seeded_kline_and_mark_price() -> None:
    stream = BinanceMarketStream("BTCUSDT", "production")
    assert stream.url.startswith("wss://fstream.binance.com/market/stream?streams=")
    stream.seed_candles(
        "1m",
        [Candle(1_700_000_000_000, 100.0, 101.0, 99.0, 100.5, 10.0)],
    )
    stream._on_open(None)

    kline_message = json.dumps(
        {
            "stream": "btcusdt@kline_1m",
            "data": {
                "e": "kline",
                "k": {
                    "t": 1_700_000_000_000,
                    "i": "1m",
                    "o": "100.0",
                    "h": "102.0",
                    "l": "99.0",
                    "c": "101.5",
                    "v": "12.0",
                    "q": "1212.0",
                },
            },
        }
    )
    mark_message = json.dumps(
        {
            "stream": "btcusdt@markPrice@1s",
            "data": {
                "e": "markPriceUpdate",
                "E": 1_700_000_001_000,
                "p": "101.25",
                "i": "101.20",
            },
        }
    )

    with patch("btc_futures_bot.binance_stream.time.monotonic", return_value=100.0):
        stream.process_message(kline_message)
        stream.process_message(mark_message)
        assert stream.healthy()
        assert stream.mark_price() == (101.25, 101.2, 1_700_000_001_000)

    candles = stream.candles("1m", 300)
    assert len(candles) == 1
    assert candles[0].close == 101.5
    assert candles[0].quote_volume == 1212.0


def test_stream_ignores_unconfigured_events() -> None:
    stream = BinanceMarketStream("BTCUSDT", "testnet")
    stream.process_message(json.dumps({"data": {"e": "bookTicker", "p": "100"}}))

    assert stream.candles("1m", 10) == []
    assert stream.mark_price() is None
