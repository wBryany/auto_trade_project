"""Public OKX v5 caches; REST warms history, WebSocket proves freshness.

Candles use /business and mark-price/tickers use /public. Returned histories
include the newest *forming* candle because the engine removes that last bar.
See https://www.okx.com/docs-v5/en/#order-book-trading-market-data-ws-candlesticks-channel
and https://www.okx.com/docs-v5/en/#public-data-websocket-mark-price-channel.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Sequence

from .models import Candle
from .okx_ws import OkxWebSocketConnection


OKX_STREAM_HOSTS = {"demo": "wspap.okx.com:8443", "production": "ws.okx.com:8443"}
_CHANNELS = {"1m": "candle1m", "5m": "candle5m", "1h": "candle1H"}
_DURATIONS = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}


@dataclass(frozen=True)
class _Bar:
    candle: Candle
    confirmed: bool
    websocket: bool = False


class OkxMarketStream:
    intervals = tuple(_CHANNELS)

    def __init__(
        self,
        symbol: str,
        environment: str,
        *,
        stale_seconds: float = 15.0,
        max_history: int = 1000,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self.symbol = symbol.strip().upper()
        self.environment = environment.strip().lower()
        if not self.symbol:
            raise ValueError("OKX stream symbol cannot be empty")
        if self.environment not in OKX_STREAM_HOSTS:
            raise ValueError("OKX stream environment must be demo or production")
        if not isfinite(stale_seconds) or stale_seconds <= 0:
            raise ValueError("stale_seconds must be positive and finite")
        if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 2:
            raise ValueError("max_history must be an integer of at least 2")
        self.stale_seconds = float(stale_seconds)
        self.max_history = max_history
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._lock = threading.RLock()
        self._closed = False
        self._business_socket: Any = None
        self._public_socket: Any = None
        self._generation = 0
        self._bars: dict[str, dict[int, _Bar]] = {interval: {} for interval in self.intervals}
        self._seeded = {interval: False for interval in self.intervals}
        self._received: dict[str, float | None] = {interval: None for interval in self.intervals}
        self._received_bar: dict[str, int | None] = {interval: None for interval in self.intervals}
        self._mark: tuple[str, float, int, float] | None = None
        self._ticker: tuple[str, float, int, float] | None = None
        self._last_error = ""
        host = OKX_STREAM_HOSTS[self.environment]
        self.business_url = f"wss://{host}/ws/v5/business"
        self.public_url = f"wss://{host}/ws/v5/public"
        self._business = OkxWebSocketConnection(
            self.business_url, on_open=self._on_business_open,
            on_message=self._on_business_message, on_disconnect=self._on_business_disconnect,
            name=f"okx-candles-{self.symbol}",
        )
        self._public = OkxWebSocketConnection(
            self.public_url, on_open=self._on_public_open,
            on_message=self._on_public_message, on_disconnect=self._on_public_disconnect,
            name=f"okx-prices-{self.symbol}",
        )

    def start(self) -> bool:
        with self._lock:
            if self._closed:
                return False
        business_started = self._business.start()
        public_started = self._public.start()
        return bool(business_started and public_started)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._business.close()
        self._public.close()
        # Also invalidate if a transport was never started or had already died.
        self._on_business_disconnect()
        self._on_public_disconnect()

    def seed_generation(self) -> int:
        """Capture BEFORE a REST request; pass back to seed_candles afterwards."""
        with self._lock:
            return self._generation

    def seed_candles(
        self, interval: str, candles: Sequence[Candle], *, generation: int | None = None,
    ) -> bool:
        """Merge a REST seed without overwriting newer/confirmed WS updates.

        A request begun before a disconnect must not repair the next session.
        Callers doing network I/O must pass the captured generation token.
        Seeding never changes WebSocket receipt timestamps.
        """
        if interval not in self._bars:
            return False
        duration = _DURATIONS[interval]
        rows = list(candles[-self.max_history:])
        if len(rows) < 2 or not all(self._valid_candle(row, duration) for row in rows):
            return False
        if any(right.timestamp - left.timestamp != duration for left, right in zip(rows, rows[1:])):
            return False
        with self._lock:
            if self._business_socket is None or (generation is not None and generation != self._generation):
                return False
            existing = self._bars[interval]
            merged = {
                candle.timestamp: _Bar(candle, index < len(rows) - 1)
                for index, candle in enumerate(rows)
            }
            for timestamp, old in existing.items():
                if timestamp < rows[0].timestamp:
                    continue
                seeded = merged.get(timestamp)
                if old.websocket and (
                    seeded is None or old.confirmed or not self._forward_update(old.candle, seeded.candle)
                ):
                    merged[timestamp] = old
            self._bars[interval] = dict(sorted(merged.items())[-self.max_history:])
            self._seeded[interval] = True
        return True

    def candles(self, interval: str, limit: int) -> list[Candle] | None:
        now_ms = self._clock()
        if interval not in self._bars or isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return None
        with self._lock:
            if not self._candle_fresh(interval, now_ms):
                return None
            bars = [bar for _, bar in sorted(self._bars[interval].items())][-limit:]
            if len(bars) < limit or not self._history_ready(interval, bars):
                return None
            return [bar.candle for bar in bars]

    def market_snapshot(self) -> dict[str, Any] | None:
        now_ms = self._clock()
        with self._lock:
            if not self._price_fresh(self._mark, now_ms):
                return None
            mark_raw, mark_price, timestamp, _ = self._mark
            ticker_fresh = self._price_fresh(self._ticker, now_ms)
            last_raw, last_price, last_timestamp, _ = self._ticker if ticker_fresh else self._mark
            snapshot = {
                "symbol": self.symbol,
                "mark_price": mark_price, "mark_price_raw": mark_raw,
                "last_price": last_price, "last_price_raw": last_raw,
                "timestamp": timestamp, "last_price_timestamp": last_timestamp,
                "last_price_source": "tickers" if ticker_fresh else "mark_price_fallback",
                "price_source": "websocket", "price_endpoint": self.public_url,
            }
        snapshot["stream"] = self.status()
        return snapshot

    def status(self) -> dict[str, Any]:
        now_ms = self._clock()
        with self._lock:
            channels = {}
            for interval in self.intervals:
                bars = [bar for _, bar in sorted(self._bars[interval].items())]
                channels[_CHANNELS[interval]] = {
                    "fresh": self._candle_fresh(interval, now_ms),
                    "history_ready": self._seeded[interval] and self._history_ready(interval, bars),
                    "seeded": self._seeded[interval], "history_count": len(bars),
                    "last_data_age_seconds": self._age(self._received[interval]),
                }
            channels["mark-price"] = {"fresh": self._price_fresh(self._mark, now_ms), "last_data_age_seconds": self._age(self._mark[3] if self._mark else None)}
            channels["tickers"] = {"fresh": self._price_fresh(self._ticker, now_ms), "last_data_age_seconds": self._age(self._ticker[3] if self._ticker else None)}
            result = {
                "connected": self._business_socket is not None and self._public_socket is not None,
                "healthy": all(row["fresh"] and row.get("history_ready", True) for row in channels.values()),
                "generation": self._generation, "channels": channels,
                "clock_available": now_ms is not None, "last_error": self._last_error,
            }
        result["business"] = self._business.status()
        result["public"] = self._public.status()
        result["available"] = bool(result["business"].get("available", True) and result["public"].get("available", True))
        return result

    def _on_business_open(self, socket: Any) -> None:
        with self._lock:
            if self._closed:
                return
            self._reset_history()
            self._business_socket = socket
        socket.send(json.dumps({"op": "subscribe", "args": [{"channel": channel, "instId": self.symbol} for channel in _CHANNELS.values()]}))

    def _on_public_open(self, socket: Any) -> None:
        with self._lock:
            if self._closed:
                return
            self._public_socket = socket
            self._mark = self._ticker = None
        socket.send(json.dumps({"op": "subscribe", "args": [{"channel": channel, "instId": self.symbol} for channel in ("mark-price", "tickers")]}))

    def _reset_history(self) -> None:
        self._generation += 1
        for interval in self.intervals:
            self._bars[interval].clear()
            self._seeded[interval] = False
            self._received[interval] = None
            self._received_bar[interval] = None

    def _on_business_disconnect(self) -> None:
        with self._lock:
            self._business_socket = None
            self._reset_history()

    def _on_public_disconnect(self) -> None:
        with self._lock:
            self._public_socket = None
            self._mark = self._ticker = None

    def _on_business_message(self, socket: Any, payload: dict[str, Any]) -> None:
        if self._handle_control_failure(socket, payload, business=True):
            return
        now_ms = self._clock()
        with self._lock:
            if socket is not self._business_socket:
                return
            channel = self._channel(payload)
            interval = next((name for name, value in _CHANNELS.items() if value == channel), None)
            if interval is None:
                return
            data = payload.get("data")
            if not isinstance(data, list):
                return
            duration = _DURATIONS[interval]
            for row in data:
                try:
                    if not isinstance(row, (list, tuple)) or len(row) < 9 or str(row[8]) not in {"0", "1"}:
                        raise ValueError("invalid candle row")
                    candle = Candle(int(row[0]), *[float(value) for value in row[1:6]], quote_volume=float(row[7]))
                    if not self._valid_candle(candle, duration):
                        raise ValueError("invalid candle values")
                except (TypeError, ValueError, OverflowError):
                    self._seeded[interval] = False
                    self._received[interval] = None
                    self._last_error = f"invalid {channel} payload"
                    continue
                if now_ms is None or candle.timestamp > now_ms // duration * duration:
                    continue
                confirmed = str(row[8]) == "1"
                bucket = self._bars[interval]
                latest = max(bucket, default=candle.timestamp)
                old = bucket.get(candle.timestamp)
                if old is not None:
                    if old.confirmed or not self._forward_update(old.candle, candle):
                        continue
                elif candle.timestamp < latest:
                    # Never use an unordered historical push to fill an unseen gap.
                    continue
                if bucket and candle.timestamp > latest + duration:
                    self._seeded[interval] = False
                bucket[candle.timestamp] = _Bar(candle, confirmed, True)
                if candle.timestamp >= latest:
                    self._received[interval] = time.monotonic()
                    self._received_bar[interval] = candle.timestamp
                for timestamp in sorted(bucket)[:-self.max_history]:
                    bucket.pop(timestamp)

    def _on_public_message(self, socket: Any, payload: dict[str, Any]) -> None:
        if self._handle_control_failure(socket, payload, business=False):
            return
        now_ms = self._clock()
        with self._lock:
            if socket is not self._public_socket:
                return
            channel = self._channel(payload)
            if channel not in {"mark-price", "tickers"} or not isinstance(payload.get("data"), list):
                return
            for row in payload["data"]:
                try:
                    if not isinstance(row, dict) or row.get("instId") != self.symbol:
                        continue
                    timestamp = int(row["ts"])
                    raw_price = str(row["markPx" if channel == "mark-price" else "last"])
                    price = float(raw_price)
                    if not isfinite(price) or price <= 0 or not self._event_time_fresh(timestamp, now_ms):
                        continue
                except (KeyError, TypeError, ValueError, OverflowError):
                    self._last_error = f"invalid {channel} payload"
                    continue
                old = self._mark if channel == "mark-price" else self._ticker
                if old is not None and timestamp < old[2]:
                    continue
                if old is not None and timestamp == old[2]:
                    if channel != "mark-price":
                        continue
                    # OKX documents same-ts mark-price corrections during
                    # deployments: the later received value is authoritative.
                    # Do not let corrections/duplicates extend receipt age.
                    received_at = old[3]
                else:
                    received_at = time.monotonic()
                value = (raw_price, price, timestamp, received_at)
                if channel == "mark-price":
                    self._mark = value
                else:
                    self._ticker = value

    def _handle_control_failure(self, socket: Any, payload: dict[str, Any], *, business: bool) -> bool:
        """An unusable subscription must reconnect, not idle on heartbeats."""
        if not isinstance(payload, dict):
            return False
        event = payload.get("event")
        if event not in ("error", "channel-conn-count-error", "unsubscribe"):
            return False
        with self._lock:
            active_socket = self._business_socket if business else self._public_socket
            if active_socket is None or socket is not active_socket:
                return True
            raw_code = str(payload.get("code", ""))
            code = raw_code if raw_code.isascii() and raw_code.isdecimal() and len(raw_code) <= 12 else "unknown"
            # Never log msg/arg or the raw payload: some exchange errors echo
            # the submitted request. The known event and numeric code suffice.
            self._last_error = f"OKX {'business' if business else 'public'} {event} code={code}"
            if business:
                self._business_socket = None
                self._reset_history()
            else:
                self._public_socket = None
                self._mark = self._ticker = None
        # Closing outside the cache lock allows the transport's disconnect
        # callback to run and its normal bounded backoff to resubscribe.
        try:
            socket.close()
        except Exception:
            pass
        return True

    def _channel(self, payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        if payload.get("event"):
            return ""
        argument = payload.get("arg")
        if not isinstance(argument, dict) or argument.get("instId") != self.symbol:
            return ""
        return str(argument.get("channel") or "")

    def _candle_fresh(self, interval: str, now_ms: int | None) -> bool:
        age = self._age(self._received[interval])
        bucket = self._bars[interval]
        if self._business_socket is None or now_ms is None or not self._seeded[interval] or age is None or age > self.stale_seconds or not bucket:
            return False
        latest = bucket[max(bucket)]
        return (not latest.confirmed and self._received_bar[interval] == latest.candle.timestamp
                and latest.candle.timestamp == now_ms // _DURATIONS[interval] * _DURATIONS[interval])

    @staticmethod
    def _history_ready(interval: str, bars: list[_Bar]) -> bool:
        return bool(
            bars and not bars[-1].confirmed
            and all(bar.confirmed for bar in bars[:-1])
            and all(right.candle.timestamp - left.candle.timestamp == _DURATIONS[interval] for left, right in zip(bars, bars[1:]))
        )

    def _price_fresh(self, value: tuple[str, float, int, float] | None, now_ms: int | None) -> bool:
        return bool(self._public_socket is not None and value is not None
                    and time.monotonic() - value[3] <= self.stale_seconds
                    and self._event_time_fresh(value[2], now_ms))

    def _event_time_fresh(self, timestamp: int, now_ms: int | None) -> bool:
        return bool(now_ms is not None and timestamp > 0 and -5000 <= now_ms - timestamp <= self.stale_seconds * 1000)

    def _clock(self) -> int | None:
        try:
            value = self._now_ms()
            return int(value) if isfinite(value) and value > 0 else None
        except (TypeError, ValueError, OverflowError, OSError, RuntimeError):
            return None

    @staticmethod
    def _age(received: float | None) -> float | None:
        return max(0.0, time.monotonic() - received) if received is not None else None

    @staticmethod
    def _valid_candle(candle: Candle, duration: int) -> bool:
        try:
            return bool(
                not isinstance(candle.timestamp, bool) and candle.timestamp >= 0 and candle.timestamp % duration == 0
                and all(isfinite(value) and value > 0 for value in (candle.open, candle.high, candle.low, candle.close))
                and candle.low <= min(candle.open, candle.close) <= max(candle.open, candle.close) <= candle.high
                and isfinite(candle.volume) and candle.volume >= 0
                and (candle.quote_volume is None or (isfinite(candle.quote_volume) and candle.quote_volume >= 0))
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _forward_update(old: Candle, new: Candle) -> bool:
        """Candle pushes have no sequence ID; cumulative fields reject rollback.

        If all cumulative fields tie but close differs, order is unknowable:
        retain the existing value and let a later update or REST recover it.
        """
        # OKX initially publishes an empty minute at the preceding close.
        # Its first real trade establishes the actual open/high/low; those
        # prices need not contain the provisional zero-volume flat range.
        # This one-way transition is not a rollback. A later empty update
        # can never overwrite a candle which already contains real trades.
        if (old.volume == 0 and old.quote_volume in (None, 0)
                and old.open == old.high == old.low == old.close
                and new.volume > 0):
            return True
        if new.open != old.open or new.volume < old.volume or new.high < old.high or new.low > old.low:
            return False
        if old.quote_volume is not None and new.quote_volume is not None and new.quote_volume < old.quote_volume:
            return False
        return not (new.volume == old.volume and new.quote_volume == old.quote_volume
                    and new.high == old.high and new.low == old.low and new.close != old.close)
