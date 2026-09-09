"""Bounded, reconnecting OKX transport; application caches own data freshness."""
from __future__ import annotations

import json
import random
import threading
import time
from typing import Any, Callable

try:
    import websocket
except ImportError:  # optional dependency failure must permit REST fallback
    websocket = None


class OkxWebSocketConnection:
    def __init__(
        self, url: str, *, on_open: Callable[[Any], None],
        on_message: Callable[[Any, dict[str, Any]], None],
        on_disconnect: Callable[[], None], name: str = "okx-ws",
    ) -> None:
        self.url = url
        self.name = name
        self._on_open = on_open
        self._on_message = on_message
        self._on_disconnect = on_disconnect
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._socket: Any = None
        self._connected = False
        self._generation = 0
        self._attempts = 0
        self._last_error = ""
        self._last_message_at = 0.0
        self._retry_at = 0.0

    def start(self) -> bool:
        with self._lock:
            if self._stop.is_set():
                return False
            if websocket is None:
                self._last_error = "websocket-client is not installed"
                return False
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
                self._thread.start()
            return True

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            socket = self._socket
            self._connected = False
            thread = self._thread
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": websocket is not None,
                "connected": self._connected and not self._stop.is_set(),
                "closed": self._stop.is_set(), "generation": self._generation,
                "reconnect_attempts": self._attempts, "last_error": self._last_error,
                "last_message_at": self._last_message_at,
                "retry_in_seconds": max(0.0, self._retry_at - time.monotonic()),
            }

    def _receive(self, socket: Any) -> None:
        last_received = time.monotonic()
        ping_at: float | None = None
        while not self._stop.is_set():
            try:
                message = socket.recv()
            except websocket.WebSocketTimeoutException:
                message = None
            now = time.monotonic()
            if message == "" or message == b"":
                raise ConnectionError("connection closed")
            if message is not None:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                last_received = now
                with self._lock:
                    self._last_message_at = time.time()
                if message == "pong":
                    ping_at = None
                elif message == "ping":
                    socket.send("pong")
                else:
                    payload = json.loads(message)
                    if not isinstance(payload, dict):
                        raise ValueError("unexpected WebSocket message")
                    if not self._stop.is_set():
                        self._on_message(socket, payload)
            # Heartbeats keep the transport alive, never the market/account cache.
            if ping_at is not None and now - ping_at >= 10:
                raise TimeoutError("OKX heartbeat timed out")
            if ping_at is None and now - last_received >= 15:
                socket.send("ping")
                ping_at = now

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            socket = None
            opened_at = time.monotonic()
            try:
                socket = websocket.create_connection(self.url, timeout=5, enable_multithread=True)
                socket.settimeout(1)
                with self._lock:
                    self._socket = socket
                    if self._stop.is_set():
                        break
                    self._connected = True
                    self._generation += 1
                    self._retry_at = 0.0
                    self._last_error = ""
                self._on_open(socket)
                self._receive(socket)
            except Exception as error:
                # Never log raw frames, login requests, API credentials or
                # exception bodies that may echo signed application messages.
                with self._lock:
                    self._last_error = f"{type(error).__name__}: connection interrupted"
            finally:
                with self._lock:
                    self._connected = False
                    self._socket = None
                try:
                    self._on_disconnect()
                except Exception:
                    pass
                if socket is not None:
                    try:
                        socket.close()
                    except Exception:
                        pass
            if self._stop.is_set():
                break
            failures = 1 if time.monotonic() - opened_at >= 60 else min(failures + 1, 6)
            delay = min(60.0, 2.0 ** failures * random.uniform(0.8, 1.2))
            with self._lock:
                self._attempts += 1
                self._retry_at = time.monotonic() + delay
            self._stop.wait(delay)
