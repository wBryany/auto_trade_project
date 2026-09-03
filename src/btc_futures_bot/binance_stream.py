from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from .models import Candle

try:  # Installed through the project dependency; kept optional for safe fallback.
    import websocket
except ImportError:  # pragma: no cover - exercised only in incomplete installations.
    websocket = None


LOG = logging.getLogger(__name__)

BINANCE_STREAM_URLS = {
    # Binance retired the legacy /stream endpoint on 2026-04-23. K-lines
    # and mark prices are regular market feeds and must use /market.
    "production": "wss://fstream.binance.com/market",
    "testnet": "wss://fstream.binancefuture.com/market",
}


class BinanceMarketStream:
    """Maintain Binance trade, mark price, and K-line caches over one stream."""

    intervals = ("1m", "5m", "1h")

    def __init__(
        self,
        symbol: str,
        environment: str,
        *,
        stale_seconds: float = 15.0,
        max_history: int = 1000,
    ) -> None:
        self.symbol = symbol.strip().lower()
        self.environment = environment.strip().lower()
        self.stale_seconds = max(3.0, float(stale_seconds))
        self.max_history = max(300, int(max_history))
        base_url = BINANCE_STREAM_URLS.get(self.environment)
        if base_url is None:
            raise ValueError("Binance stream environment must be testnet or production")
        streams = [f"{self.symbol}@kline_{interval}" for interval in self.intervals]
        streams.append(f"{self.symbol}@aggTrade")
        streams.append(f"{self.symbol}@markPrice@1s")
        self.url = f"{base_url}/stream?streams={'/'.join(streams)}"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: Any = None
        self._candles: dict[str, dict[int, Candle]] = {
            interval: {} for interval in self.intervals
        }
        self._mark_price = 0.0
        self._index_price = 0.0
        self._market_timestamp = 0
        self._mark_received_at = 0.0
        self._last_trade_price = 0.0
        self._last_trade_timestamp = 0
        self._trade_received_at = 0.0
        self._last_message_at = 0.0
        self._connected = False
        self._last_error = ""

    @property
    def available(self) -> bool:
        return websocket is not None

    def start(self) -> bool:
        if not self.available:
            self._last_error = "websocket-client is not installed"
            return False
        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"binance-market-stream-{self.symbol}",
                daemon=True,
            )
            self._thread.start()
        return True

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            active_socket = self._socket
            thread = self._thread
        if active_socket is not None:
            try:
                active_socket.close()
            except Exception:
                LOG.debug("Binance market stream close failed", exc_info=True)
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3)
        with self._lock:
            self._connected = False
            self._thread = None
            self._socket = None

    def seed_candles(self, interval: str, candles: list[Candle]) -> None:
        if interval not in self._candles:
            return
        with self._lock:
            bucket = self._candles[interval]
            for candle in candles[-self.max_history :]:
                bucket[int(candle.timestamp)] = candle
            self._trim(bucket)

    def candles(self, interval: str, limit: int) -> list[Candle]:
        with self._lock:
            bucket = self._candles.get(interval, {})
            timestamps = sorted(bucket)[-max(1, int(limit)) :]
            return [bucket[timestamp] for timestamp in timestamps]

    def has_history(self, interval: str, minimum: int) -> bool:
        with self._lock:
            return len(self._candles.get(interval, {})) >= max(1, int(minimum))

    def healthy(self) -> bool:
        with self._lock:
            return bool(
                self._connected
                and self._last_message_at
                and time.monotonic() - self._last_message_at <= self.stale_seconds
            )

    def mark_price(self) -> tuple[float, float, int] | None:
        with self._lock:
            if (
                not self._connected
                or self._mark_price <= 0
                or not self._mark_received_at
                or time.monotonic() - self._mark_received_at > self.stale_seconds
            ):
                return None
            return self._mark_price, self._index_price, self._market_timestamp

    def latest_trade(self) -> tuple[float, int] | None:
        with self._lock:
            if (
                not self._connected
                or self._last_trade_price <= 0
                or not self._trade_received_at
                or time.monotonic() - self._trade_received_at > self.stale_seconds
            ):
                return None
            return self._last_trade_price, self._last_trade_timestamp

    def status(self) -> dict[str, Any]:
        with self._lock:
            age = (
                max(0.0, time.monotonic() - self._last_message_at)
                if self._last_message_at
                else None
            )
            return {
                "available": self.available,
                "connected": self._connected,
                "healthy": self.healthy(),
                "last_message_age_seconds": age,
                "mark_price_age_seconds": (
                    max(0.0, time.monotonic() - self._mark_received_at)
                    if self._mark_received_at
                    else None
                ),
                "trade_price_age_seconds": (
                    max(0.0, time.monotonic() - self._trade_received_at)
                    if self._trade_received_at
                    else None
                ),
                "last_error": self._last_error,
            }

    def process_message(self, raw_message: str) -> None:
        payload = json.loads(raw_message)
        data = payload.get("data", payload)
        event = str(data.get("e") or "")
        if event == "kline":
            row = data.get("k") or {}
            interval = str(row.get("i") or "")
            if interval not in self._candles:
                return
            candle = Candle(
                int(row["t"]),
                float(row["o"]),
                float(row["h"]),
                float(row["l"]),
                float(row["c"]),
                float(row["v"]),
                quote_volume=float(row["q"]) if row.get("q") is not None else None,
            )
            with self._lock:
                bucket = self._candles[interval]
                bucket[int(candle.timestamp)] = candle
                self._trim(bucket)
                self._last_message_at = time.monotonic()
            return
        if event == "markPriceUpdate":
            with self._lock:
                self._mark_price = float(data.get("p") or 0)
                self._index_price = float(data.get("i") or 0)
                self._market_timestamp = int(data.get("E") or time.time() * 1000)
                received_at = time.monotonic()
                self._mark_received_at = received_at
                self._last_message_at = received_at
            return
        if event == "aggTrade":
            with self._lock:
                self._last_trade_price = float(data.get("p") or 0)
                self._last_trade_timestamp = int(
                    data.get("T") or data.get("E") or time.time() * 1000
                )
                received_at = time.monotonic()
                self._trade_received_at = received_at
                self._last_message_at = received_at

    def _trim(self, bucket: dict[int, Candle]) -> None:
        overflow = len(bucket) - self.max_history
        if overflow <= 0:
            return
        for timestamp in sorted(bucket)[:overflow]:
            bucket.pop(timestamp, None)

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            had_recent_messages = False
            try:
                active_socket = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                with self._lock:
                    self._socket = active_socket
                active_socket.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as error:
                self._record_error(error)
            finally:
                with self._lock:
                    had_recent_messages = bool(
                        self._last_message_at
                        and time.monotonic() - self._last_message_at <= self.stale_seconds
                    )
                    self._connected = False
                    self._socket = None
            reconnect_delay = 1.0 if had_recent_messages else backoff
            if self._stop.wait(reconnect_delay):
                break
            backoff = 1.0 if had_recent_messages else min(30.0, backoff * 2)

    def _on_open(self, _socket: Any) -> None:
        with self._lock:
            self._connected = True
            self._last_error = ""
        LOG.info("Binance WebSocket market stream connected for %s", self.symbol.upper())

    def _on_message(self, _socket: Any, message: str) -> None:
        try:
            self.process_message(message)
        except Exception as error:
            self._record_error(error)

    def _on_error(self, _socket: Any, error: Any) -> None:
        self._record_error(error)

    def _on_close(self, _socket: Any, status_code: Any, message: Any) -> None:
        with self._lock:
            self._connected = False
        if not self._stop.is_set():
            LOG.warning(
                "Binance WebSocket market stream closed code=%s message=%s",
                status_code,
                message,
            )

    def _record_error(self, error: Any) -> None:
        message = str(error)
        with self._lock:
            self._last_error = message
        if not self._stop.is_set():
            LOG.warning("Binance WebSocket market stream error: %s", message)
