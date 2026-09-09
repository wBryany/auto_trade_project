from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from btc_futures_bot.models import Candle
from btc_futures_bot.okx_stream import OkxMarketStream


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False

    def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, url, *, on_open, on_message, on_disconnect, name):
        self.url = url
        self.name = name
        self.on_open = on_open
        self.on_message = on_message
        self.on_disconnect = on_disconnect
        self.socket = None
        self.closed = False

    def start(self) -> bool:
        if self.closed:
            return False
        if self.socket is None:
            self.socket = FakeSocket()
            self.on_open(self.socket)
        return True

    def close(self) -> None:
        self.socket = None
        self.closed = True
        self.on_disconnect()

    def disconnect(self) -> None:
        self.socket = None
        self.on_disconnect()

    def reconnect(self) -> None:
        self.disconnect()
        self.start()

    def push(self, payload) -> None:
        self.on_message(self.socket, payload)

    def status(self) -> dict:
        return {"connected": self.socket is not None, "last_error": "", "available": True}


@pytest.fixture
def market():
    # Exact hour boundary plus ten seconds. One controllable exchange clock
    # and an independent monotonic clock model both timestamp/receipt ages.
    clocks = {"wall": 1_800_000_000_000 // 3_600_000 * 3_600_000 + 10_000, "mono": 100.0}
    with patch("btc_futures_bot.okx_stream.OkxWebSocketConnection", FakeConnection), patch(
        "btc_futures_bot.okx_stream.time.monotonic", side_effect=lambda: clocks["mono"]
    ):
        stream = OkxMarketStream("btc-usdt-swap", "demo", now_ms=lambda: clocks["wall"])
        stream.start()
        yield stream, clocks
        stream.close()


def _candles(now_ms: int, interval: str = "1m", count: int = 4) -> list[Candle]:
    duration = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}[interval]
    latest = now_ms // duration * duration
    return [Candle(latest - (count - index - 1) * duration, 100.0, 101.0, 99.0, 100.5, 10.0, 1000.0)
            for index in range(count)]


def _bar_payload(candle: Candle, interval: str = "1m", confirm: str = "0", symbol: str = "BTC-USDT-SWAP") -> dict:
    channel = {"1m": "candle1m", "5m": "candle5m", "1h": "candle1H"}[interval]
    return {"arg": {"channel": channel, "instId": symbol}, "data": [[
        str(candle.timestamp), str(candle.open), str(candle.high), str(candle.low),
        str(candle.close), str(candle.volume), "0.01", str(candle.quote_volume), confirm,
    ]]}


def _price_payload(now_ms: int, channel: str = "mark-price", price: str = "80000.12345678", symbol: str = "BTC-USDT-SWAP") -> dict:
    return {"arg": {"channel": channel, "instId": symbol}, "data": [
        {"instType": "SWAP", "instId": symbol, "ts": str(now_ms),
         "markPx" if channel == "mark-price" else "last": price},
    ]}


def _warm(stream: OkxMarketStream, clocks: dict, interval: str = "1m") -> list[Candle]:
    candles = _candles(clocks["wall"], interval)
    assert stream.seed_candles(interval, candles, generation=stream.seed_generation())
    stream._business.push(_bar_payload(candles[-1], interval))
    return candles


def test_correct_demo_endpoint_channel_subscriptions_and_idempotent_start(market) -> None:
    stream, _ = market
    assert stream.business_url == "wss://wspap.okx.com:8443/ws/v5/business"
    assert stream.public_url == "wss://wspap.okx.com:8443/ws/v5/public"
    assert stream._business.socket.sent == [{"op": "subscribe", "args": [
        {"channel": "candle1m", "instId": "BTC-USDT-SWAP"},
        {"channel": "candle5m", "instId": "BTC-USDT-SWAP"},
        {"channel": "candle1H", "instId": "BTC-USDT-SWAP"},
    ]}]
    assert [row["channel"] for row in stream._public.socket.sent[0]["args"]] == ["mark-price", "tickers"]
    assert stream.start()
    assert len(stream._business.socket.sent) == 1


def test_production_endpoints_and_invalid_constructor_inputs() -> None:
    with patch("btc_futures_bot.okx_stream.OkxWebSocketConnection", FakeConnection):
        stream = OkxMarketStream("BTC-USDT-SWAP", "production")
        assert stream.public_url == "wss://ws.okx.com:8443/ws/v5/public"
        assert stream.business_url == "wss://ws.okx.com:8443/ws/v5/business"
        for overrides in ({"environment": "testnet"}, {"stale_seconds": float("nan")},
                          {"stale_seconds": 0}, {"max_history": 1}, {"max_history": True}):
            arguments = {"symbol": "BTC-USDT-SWAP", "environment": "demo", **overrides}
            with pytest.raises(ValueError):
                OkxMarketStream(**arguments)


@pytest.mark.parametrize("interval", ["1m", "5m", "1h"])
def test_rest_seed_alone_never_makes_ws_history_fresh(market, interval: str) -> None:
    stream, clocks = market
    rows = _candles(clocks["wall"], interval)
    assert stream.seed_candles(interval, rows)
    assert stream.candles(interval, 4) is None
    stream._business.push({"event": "subscribe", "arg": {"channel": "candle1m", "instId": stream.symbol}})
    assert stream.candles(interval, 4) is None
    stream._business.push(_bar_payload(rows[-1], interval))
    assert stream.candles(interval, 4) == rows
    assert stream.candles(interval, 5) is None


def test_live_push_without_rest_history_does_not_supply_partial_history(market) -> None:
    stream, clocks = market
    rows = _candles(clocks["wall"])
    stream._business.push(_bar_payload(rows[-1]))
    assert stream.candles("1m", 1) is None
    assert not stream.status()["channels"]["candle1m"]["seeded"]


def test_receipt_expiry_cannot_be_refreshed_by_seed_or_subscription_ack(market) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    clocks["mono"] += 16
    assert stream.seed_candles("1m", rows)
    stream._business.push({"event": "subscribe", "arg": {"channel": "candle1m", "instId": stream.symbol}})
    assert stream.candles("1m", 4) is None
    assert not stream.status()["channels"]["candle1m"]["fresh"]


def test_seed_for_new_forming_bar_needs_ws_data_for_that_same_bar(market) -> None:
    stream, clocks = market
    _warm(stream, clocks)
    clocks["wall"] += 60_000
    rows = _candles(clocks["wall"])
    assert stream.seed_candles("1m", rows)
    assert stream.candles("1m", 4) is None
    stream._business.push(_bar_payload(rows[-1]))
    assert stream.candles("1m", 4) == rows


def test_newest_confirmed_only_bar_does_not_cause_engine_to_drop_a_closed_bar(market) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    stream._business.push(_bar_payload(rows[-1], confirm="1"))
    assert stream.candles("1m", 4) is None
    clocks["wall"] += 60_000
    next_bar = replace(rows[-1], timestamp=rows[-1].timestamp + 60_000)
    stream._business.push(_bar_payload(next_bar))
    returned = stream.candles("1m", 4)
    assert returned[-1] == next_bar
    assert returned[:-1][-1] == rows[-1]


def test_confirmed_candle_cannot_be_rolled_back_by_late_partial_or_reseed(market) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    confirmed = replace(rows[-1], volume=15.0, quote_volume=1500.0, high=102.0, close=101.5)
    stream._business.push(_bar_payload(confirmed, confirm="1"))
    stream._business.push(_bar_payload(rows[-1], confirm="0"))
    clocks["wall"] += 60_000
    next_bar = replace(rows[-1], timestamp=rows[-1].timestamp + 60_000)
    stream._business.push(_bar_payload(next_bar))
    stream.seed_candles("1m", [*rows[1:], next_bar])
    returned = stream.candles("1m", 4)
    assert returned[-2] == confirmed


def test_unconfirmed_cumulative_volume_and_range_cannot_go_backwards(market) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    updated = replace(rows[-1], volume=15.0, quote_volume=1500.0, high=102.0, close=101.5)
    stream._business.push(_bar_payload(updated))
    stream._business.push(_bar_payload(rows[-1]))
    assert stream.candles("1m", 4)[-1] == updated
    stream.seed_candles("1m", rows)
    assert stream.candles("1m", 4)[-1] == updated


def test_equal_cumulative_values_with_conflicting_close_do_not_overwrite(market) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    stream._business.push(_bar_payload(replace(rows[-1], close=100.7)))
    assert stream.candles("1m", 4)[-1] == rows[-1]


def test_previous_final_confirmation_may_arrive_after_the_next_forming_bar(market) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    clocks["wall"] += 60_000
    next_bar = replace(rows[-1], timestamp=rows[-1].timestamp + 60_000)
    stream._business.push(_bar_payload(next_bar))
    assert stream.candles("1m", 4) is None
    stream._business.push(_bar_payload(rows[-1], confirm="1"))
    assert stream.candles("1m", 4) == [*rows[1:], next_bar]


def test_observed_gap_requires_explicit_rest_reseed(market) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    clocks["wall"] += 120_000
    jumped = replace(rows[-1], timestamp=rows[-1].timestamp + 120_000)
    stream._business.push(_bar_payload(jumped))
    assert stream.candles("1m", 2) is None
    assert not stream.status()["channels"]["candle1m"]["seeded"]
    repaired = _candles(clocks["wall"])
    assert stream.seed_candles("1m", repaired)
    assert stream.candles("1m", 4) == repaired


def test_reconnect_invalidates_history_and_rejects_inflight_old_generation_seed(market) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    generation = stream.seed_generation()
    old_socket = stream._business.socket
    stream._business.reconnect()
    assert stream.seed_generation() != generation
    stream._business.push(_bar_payload(rows[-1]))
    assert not stream.seed_candles("1m", rows, generation=generation)
    assert stream.candles("1m", 4) is None
    stream._on_business_message(old_socket, _bar_payload(rows[-1]))
    assert stream.candles("1m", 4) is None
    assert stream.seed_candles("1m", rows, generation=stream.seed_generation())
    assert stream.candles("1m", 4) == rows


def test_disconnect_alone_makes_all_history_unavailable(market) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    stream._business.disconnect()
    assert not stream.seed_candles("1m", rows)
    assert stream.candles("1m", 4) is None


@pytest.mark.parametrize("bad", ["gap", "reverse", "duplicate", "nan", "bad_range", "wrong_alignment"])
def test_bad_rest_seed_does_not_establish_history(market, bad: str) -> None:
    stream, clocks = market
    rows = _candles(clocks["wall"])
    if bad == "gap":
        rows.pop(1)
    elif bad == "reverse":
        rows.reverse()
    elif bad == "duplicate":
        rows[1] = rows[0]
    elif bad == "nan":
        rows[1] = replace(rows[1], high=float("nan"))
    elif bad == "bad_range":
        rows[1] = replace(rows[1], high=98.0)
    else:
        rows[1] = replace(rows[1], timestamp=rows[1].timestamp + 1)
    assert not stream.seed_candles("1m", rows)
    assert stream.candles("1m", 4) is None


@pytest.mark.parametrize("bad", ["missing", "nan", "negative_volume", "bad_confirm", "bad_range"])
def test_malformed_ws_candle_invalidates_seed_until_rest_repair(market, bad: str) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    payload = _bar_payload(rows[-1])
    row = payload["data"][0]
    if bad == "missing":
        row.pop()
    elif bad == "nan":
        row[4] = "NaN"
    elif bad == "negative_volume":
        row[5] = "-1"
    elif bad == "bad_confirm":
        row[8] = "2"
    else:
        row[2] = "98"
    stream._business.push(payload)
    assert stream.candles("1m", 4) is None
    assert stream.status()["last_error"]


def test_other_symbol_unknown_channels_and_ack_never_refresh_cache(market) -> None:
    stream, clocks = market
    rows = _candles(clocks["wall"])
    stream.seed_candles("1m", rows)
    stream._business.push(_bar_payload(rows[-1], symbol="ETH-USDT-SWAP"))
    stream._business.push({"arg": {"channel": "unknown", "instId": stream.symbol}, "data": []})
    assert stream.candles("1m", 4) is None
    stream._public.push(_price_payload(clocks["wall"], symbol="ETH-USDT-SWAP"))
    assert stream.market_snapshot() is None


def test_mark_and_real_last_prices_keep_raw_precision_and_independent_timestamps(market) -> None:
    stream, clocks = market
    stream._public.push(_price_payload(clocks["wall"]))
    snapshot = stream.market_snapshot()
    assert snapshot["mark_price_raw"] == "80000.12345678"
    assert snapshot["last_price_source"] == "mark_price_fallback"
    stream._public.push(_price_payload(clocks["wall"] - 100, "tickers", "80001.01234567"))
    snapshot = stream.market_snapshot()
    assert snapshot["last_price_raw"] == "80001.01234567"
    assert snapshot["last_price_source"] == "tickers"
    assert snapshot["timestamp"] == clocks["wall"]
    assert snapshot["last_price_timestamp"] == clocks["wall"] - 100
    assert snapshot["price_source"] == "websocket"


def test_tickers_alone_cannot_substitute_for_fresh_mark_price(market) -> None:
    stream, clocks = market
    stream._public.push(_price_payload(clocks["wall"], "tickers"))
    assert stream.market_snapshot() is None


def test_late_price_updates_cannot_regress_new_prices_or_refresh_duplicate_age(market) -> None:
    stream, clocks = market
    stream._public.push(_price_payload(clocks["wall"]))
    stream._public.push(_price_payload(clocks["wall"] - 1, price="1"))
    assert stream.market_snapshot()["mark_price_raw"] == "80000.12345678"
    clocks["mono"] += 16
    stream._public.push(_price_payload(clocks["wall"], price="2"))
    assert stream.market_snapshot() is None


def test_documented_same_timestamp_mark_correction_updates_value_without_refreshing_age(market) -> None:
    stream, clocks = market
    stream._public.push(_price_payload(clocks["wall"]))
    clocks["mono"] += 10
    stream._public.push(_price_payload(clocks["wall"], price="80000.12345679"))
    assert stream.market_snapshot()["mark_price_raw"] == "80000.12345679"
    assert stream.status()["channels"]["mark-price"]["last_data_age_seconds"] == 10
    clocks["mono"] += 6
    assert stream.market_snapshot() is None


def test_fresh_receipt_cannot_resurrect_an_old_event_timestamp(market) -> None:
    stream, clocks = market
    stream._public.push(_price_payload(clocks["wall"] - 16_000))
    assert stream.market_snapshot() is None
    stream._public.push(_price_payload(clocks["wall"] + 10_000))
    assert stream.market_snapshot() is None
    stream._public.push(_price_payload(clocks["wall"]))
    assert stream.market_snapshot() is not None
    clocks["wall"] += 16_000
    assert stream.market_snapshot() is None


def test_stale_ticker_falls_back_explicitly_without_invalidating_fresh_mark(market) -> None:
    stream, clocks = market
    stream._public.push(_price_payload(clocks["wall"], "tickers", "90000"))
    clocks["mono"] += 16
    clocks["wall"] += 16_000
    stream._public.push(_price_payload(clocks["wall"]))
    snapshot = stream.market_snapshot()
    assert snapshot["last_price_source"] == "mark_price_fallback"
    assert snapshot["last_price"] == snapshot["mark_price"]


def test_price_reconnect_and_old_socket_messages_do_not_reuse_old_snapshot(market) -> None:
    stream, clocks = market
    stream._public.push(_price_payload(clocks["wall"]))
    old_socket = stream._public.socket
    stream._public.reconnect()
    stream._on_public_message(old_socket, _price_payload(clocks["wall"]))
    assert stream.market_snapshot() is None
    stream._public.push(_price_payload(clocks["wall"]))
    assert stream.market_snapshot() is not None


def test_price_disconnect_does_not_invalidate_independent_business_history(market) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    stream._public.disconnect()
    assert stream.market_snapshot() is None
    assert stream.candles("1m", 4) == rows


def test_bad_prices_and_unavailable_exchange_clock_fail_closed(market) -> None:
    stream, clocks = market
    for price in ("NaN", "Inf", "0", "-1", "not-a-number"):
        stream._public.push(_price_payload(clocks["wall"], price=price))
        assert stream.market_snapshot() is None
    _warm(stream, clocks)
    stream._public.push(_price_payload(clocks["wall"]))
    stream._now_ms = lambda: None
    assert stream.market_snapshot() is None
    assert stream.candles("1m", 4) is None
    assert not stream.status()["clock_available"]


def test_injected_exchange_clock_does_not_depend_on_wrong_machine_clock(market) -> None:
    stream, clocks = market
    with patch("btc_futures_bot.okx_stream.time.time", return_value=0):
        _warm(stream, clocks)
        stream._public.push(_price_payload(clocks["wall"]))
        assert stream.candles("1m", 4) is not None
        assert stream.market_snapshot() is not None


def test_status_exposes_each_channel_history_readiness_and_prices(market) -> None:
    stream, clocks = market
    for interval in stream.intervals:
        _warm(stream, clocks, interval)
    for channel in ("mark-price", "tickers"):
        stream._public.push(_price_payload(clocks["wall"], channel))
    status = stream.status()
    assert status["healthy"]
    assert status["connected"]
    for channel in ("candle1m", "candle5m", "candle1H"):
        assert status["channels"][channel]["fresh"]
        assert status["channels"][channel]["history_ready"]
        assert status["channels"][channel]["history_count"] == 4
    stream.close()
    assert not stream.status()["connected"]
    assert stream.candles("1m", 4) is None


def test_max_history_is_bounded_and_returns_only_requested_candles(market) -> None:
    stream, clocks = market
    stream.max_history = 4
    rows = _candles(clocks["wall"], count=10)
    assert stream.seed_candles("1m", rows)
    stream._business.push(_bar_payload(rows[-1]))
    assert stream.candles("1m", 4) == rows[-4:]
    assert stream.candles("1m", 2) == rows[-2:]
    assert stream.candles("1m", 5) is None
    assert stream.candles("30s", 2) is None


def test_late_open_callback_after_close_cannot_resurrect_stream(market) -> None:
    stream, _ = market
    stream.close()
    business_socket = FakeSocket()
    public_socket = FakeSocket()
    stream._on_business_open(business_socket)
    stream._on_public_open(public_socket)
    assert not stream.start()
    assert not stream.status()["connected"]
    assert business_socket.sent == []
    assert public_socket.sent == []


def test_empty_minute_placeholder_can_initialize_real_open_on_first_trade(market) -> None:
    stream, clocks = market
    rows = _candles(clocks["wall"])
    empty = replace(rows[-1], open=100.5, high=100.5, low=100.5, close=100.5,
                    volume=0.0, quote_volume=0.0)
    rows[-1] = empty
    stream.seed_candles("1m", rows)
    stream._business.push(_bar_payload(empty))
    # Live OKX pushes have this sequence: previous close / zero turnover,
    # followed by a new opening trade whose price may gap above that close.
    traded = replace(empty, open=102.0, high=103.0, low=102.0, close=102.5,
                     volume=3.0, quote_volume=307.5)
    stream._business.push(_bar_payload(traded))
    assert stream.candles("1m", 4)[-1] == traded
    stream._business.push(_bar_payload(empty))
    stream.seed_candles("1m", rows)
    assert stream.candles("1m", 4)[-1] == traded


def test_real_candle_open_cannot_change_when_turnover_already_exists(market) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    changed_open = replace(rows[-1], open=100.1, volume=11.0, quote_volume=1100.0)
    stream._business.push(_bar_payload(changed_open))
    assert stream.candles("1m", 4)[-1] == rows[-1]


def test_confirmed_empty_candle_remains_immutable(market) -> None:
    stream, clocks = market
    rows = _candles(clocks["wall"])
    empty = replace(rows[-1], open=100.5, high=100.5, low=100.5, close=100.5,
                    volume=0.0, quote_volume=0.0)
    rows[-1] = empty
    stream.seed_candles("1m", rows)
    stream._business.push(_bar_payload(empty, confirm="1"))
    traded = replace(empty, open=102.0, high=103.0, low=102.0, close=102.5,
                     volume=3.0, quote_volume=307.5)
    stream._business.push(_bar_payload(traded))
    assert stream._bars["1m"][empty.timestamp].confirmed
    assert stream._bars["1m"][empty.timestamp].candle == empty


@pytest.mark.parametrize("event", ["error", "channel-conn-count-error", "unsubscribe"])
def test_business_control_failure_invalidates_all_candles_and_closes_socket(market, event: str) -> None:
    stream, clocks = market
    for interval in stream.intervals:
        _warm(stream, clocks, interval)
    stream._public.push(_price_payload(clocks["wall"]))
    socket = stream._business.socket
    generation = stream.seed_generation()
    stream._business.push({"event": event, "code": "60012", "msg": "sensitive-message", "arg": {"private": "sensitive-arg"}})
    assert socket.closed
    assert stream.seed_generation() != generation
    for interval in stream.intervals:
        assert stream.candles(interval, 4) is None
    assert stream.market_snapshot() is not None  # Independent public link survives.
    status = stream.status()
    assert status["last_error"] == f"OKX business {event} code=60012"
    assert "sensitive" not in json.dumps(status)


@pytest.mark.parametrize("event", ["error", "channel-conn-count-error", "unsubscribe"])
def test_public_control_failure_invalidates_prices_and_closes_socket(market, event: str) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    stream._public.push(_price_payload(clocks["wall"]))
    stream._public.push(_price_payload(clocks["wall"], "tickers"))
    socket = stream._public.socket
    stream._public.push({"event": event, "code": "60011", "msg": "sensitive-message"})
    assert socket.closed
    assert stream.market_snapshot() is None
    assert stream.candles("1m", 4) == rows
    assert stream.status()["last_error"] == f"OKX public {event} code=60011"


def test_control_failure_does_not_echo_untrusted_code_or_close_a_new_connection(market) -> None:
    stream, clocks = market
    old_socket = stream._public.socket
    stream._public.reconnect()
    stream._public.push(_price_payload(clocks["wall"]))
    new_socket = stream._public.socket
    stream._on_public_message(old_socket, {"event": "error", "code": "60012"})
    assert not new_socket.closed
    assert stream.market_snapshot() is not None
    stream._public.push({"event": "error", "code": "sensitive-credential", "msg": "sensitive-message"})
    assert stream.status()["last_error"] == "OKX public error code=unknown"
    assert "sensitive" not in json.dumps(stream.status())


def test_business_subscription_failure_reconnect_requires_new_rest_seed(market) -> None:
    stream, clocks = market
    rows = _warm(stream, clocks)
    old_generation = stream.seed_generation()
    stream._business.push({"event": "error", "code": "60012"})
    stream._business.reconnect()
    assert stream._business.socket.sent[0]["op"] == "subscribe"
    stream._business.push(_bar_payload(rows[-1]))
    assert not stream.seed_candles("1m", rows, generation=old_generation)
    assert stream.candles("1m", 4) is None
    assert stream.seed_candles("1m", rows, generation=stream.seed_generation())
    assert stream.candles("1m", 4) == rows
