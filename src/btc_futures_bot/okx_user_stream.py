"""Read-only OKX private cache, fenced by periodic REST reconciliation.

Protocol: https://www.okx.com/docs-v5/en/ and /docs-v5/trick_en/.
Orders have no subscription snapshot. Account/position messages may be
partial, so login, subscription ACKs, heartbeats and empty updates cannot
establish a known-flat account. Only a validated REST baseline can do that.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import math
import threading
import time
from typing import Any, Callable

from .okx_ws import OkxWebSocketConnection


PRIVATE_URLS = {
    "demo": "wss://wspap.okx.com:8443/ws/v5/private",
    "production": "wss://ws.okx.com:8443/ws/v5/private",
}
_CHANNELS = {"account", "positions", "orders"}
_OPEN_STATES = {"live", "partially_filled"}
_TERMINAL_STATES = {"filled", "canceled", "mmp_canceled"}
_RECONCILE_SECONDS = 60.0
_REFRESH_SECONDS = 45.0
_RETRY_SECONDS = 5.0
_MAX_BUFFER = 2000


class OkxPrivateStream:
    def __init__(
        self, symbol: str, environment: str, *,
        credentials: Callable[[], tuple[str, str, str]],
        timestamp: Callable[[], float | str], stale_seconds: float = 90,
    ) -> None:
        self.symbol = symbol.strip().upper()
        self.environment = environment.strip().lower()
        if self.environment not in PRIVATE_URLS:
            raise ValueError("OKX private stream environment must be demo or production")
        if not self.symbol or not math.isfinite(stale_seconds) or stale_seconds <= 0:
            raise ValueError("OKX private stream requires a symbol and positive finite stale_seconds")
        self.stale_seconds = float(stale_seconds)
        self._credentials = credentials
        self._timestamp = timestamp
        self._lock = threading.RLock()
        self._socket: Any = None
        self._closed = False
        self._generation = 0
        self._connected = False
        self._logged_in = False
        self._subscriptions: set[str] = set()
        self._ready = False
        self._synchronizing = False
        self._sync_failed = False
        self._last_error = "REST baseline has not been synchronized"
        self._last_rest_at: float | None = None
        self._last_data_at: float | None = None
        self._next_sync_at = 0.0
        self._sync_start_ms = 0
        self._sync_end_ms = 0
        self._buffer: list[dict[str, Any]] = []
        self._account: dict[str, Any] = {}
        self._positions: dict[str, dict[str, Any]] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        # Tombstones survive removals until the next authoritative REST seed.
        self._position_versions: dict[str, tuple[int, int]] = {}
        self._order_versions: dict[str, tuple[int, str, float]] = {}
        self._connection = OkxWebSocketConnection(
            PRIVATE_URLS[self.environment], on_open=self._on_open,
            on_message=self._on_message, on_disconnect=self._on_disconnect,
            name=f"okx-private-{self.symbol.lower()}",
        )

    def start(self) -> bool:
        with self._lock:
            if self._closed:
                return False
        return bool(self._connection.start())

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._disconnect_locked("OKX private stream closed")
        self._connection.close()

    def _reset_cache_locked(self) -> None:
        self._ready = False
        self._last_rest_at = None
        self._last_data_at = None
        self._account = {}
        self._positions = {}
        self._orders = {}
        self._position_versions = {}
        self._order_versions = {}
        self._buffer = []
        self._sync_start_ms = 0
        self._sync_end_ms = 0

    def _disconnect_locked(self, reason: str) -> None:
        self._generation += 1
        self._connected = False
        self._logged_in = False
        self._subscriptions.clear()
        self._socket = None
        self._synchronizing = False
        self._reset_cache_locked()
        self._last_error = reason

    def _on_disconnect(self) -> None:
        with self._lock:
            self._disconnect_locked("OKX private stream disconnected; REST resynchronization required")

    def _server_ms(self) -> int:
        value = float(self._timestamp())
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid server timestamp")
        return int(value * 1000)

    def _on_open(self, socket: Any) -> None:
        with self._lock:
            if self._closed:
                socket.close()
                return
            self._disconnect_locked("OKX private login pending")
            self._socket = socket
            self._connected = True
            self._next_sync_at = 0.0
        try:
            key, secret, passphrase = self._credentials()
            if not all(isinstance(value, str) and value for value in (key, secret, passphrase)):
                raise ValueError("incomplete credentials")
            stamp = str(self._server_ms() / 1000)
            signature = base64.b64encode(hmac.new(
                secret.encode(), (stamp + "GET/users/self/verify").encode(), hashlib.sha256,
            ).digest()).decode()
            socket.send(json.dumps({"op": "login", "args": [{
                "apiKey": key, "passphrase": passphrase, "timestamp": stamp, "sign": signature,
            }]}))
        except Exception:
            # Never retain exception/payload text that might echo login secrets.
            with self._lock:
                self._last_error = "OKX private login preparation failed"
                self._ready = False
            socket.close()

    def _invalidate_locked(self, reason: str) -> None:
        self._ready = False
        if self._synchronizing:
            self._sync_failed = True
        self._last_error = reason
        self._next_sync_at = max(self._next_sync_at, time.monotonic() + _RETRY_SECONDS)

    def _on_message(self, socket: Any, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._closed or socket is not self._socket or not self._connected:
                return
            event = payload.get("event")
            if event == "login":
                if str(payload.get("code")) != "0":
                    self._invalidate_locked("OKX private login rejected")
                    self._logged_in = False
                    socket.close()
                    return
                self._logged_in = True
                socket.send(json.dumps({"op": "subscribe", "args": [
                    {"channel": "account"},
                    {"channel": "positions", "instType": "SWAP", "instId": self.symbol},
                    {"channel": "orders", "instType": "SWAP", "instId": self.symbol},
                ]}))
                return
            arg = payload.get("arg") or {}
            channel = arg.get("channel") if isinstance(arg, dict) else None
            if event in {"error", "channel-conn-count-error", "unsubscribe"}:
                self._invalidate_locked("OKX private subscription or protocol error; REST required")
                self._subscriptions.clear()
                socket.close()
                return
            if event == "subscribe":
                if str(payload.get("code", "0")) != "0":
                    self._invalidate_locked("OKX private subscription rejected")
                    self._subscriptions.clear()
                    socket.close()
                    return
                if self._logged_in and channel in _CHANNELS and str(payload.get("code", "0")) == "0":
                    if channel == "account" or (
                        arg.get("instId") == self.symbol and arg.get("instType") == "SWAP"
                    ):
                        self._subscriptions.add(channel)
                return
            if event or channel not in _CHANNELS or not self._logged_in:
                return
            if channel != "account" and arg.get("instId") not in (None, "", self.symbol):
                return
            if self._synchronizing or not self._ready:
                if len(self._buffer) >= _MAX_BUFFER:
                    self._buffer.clear()
                    self._invalidate_locked("OKX private update buffer overflow; REST required")
                    return
                else:
                    self._buffer.append(copy.deepcopy(payload))
                if not self._synchronizing or not self._ready:
                    return
                # During proactive reconciliation, keep the still-valid old
                # view current as well as buffering updates for the new seed.
                # Its original 60s REST deadline is never extended here.
            try:
                self._apply_locked(payload)
            except (ValueError, TypeError, KeyError, OverflowError):
                self._invalidate_locked("OKX private data is incomplete or ambiguous; REST required")

    def _healthy_locked(self) -> bool:
        now = time.monotonic()
        return bool(
            not self._closed and self._connected and self._logged_in
            and self._subscriptions == _CHANNELS and self._ready
            and self._last_rest_at is not None and now - self._last_rest_at <= _RECONCILE_SECONDS
            and self._last_data_at is not None and now - self._last_data_at <= self.stale_seconds
        )

    def status(self) -> dict[str, Any]:
        transport = self._connection.status()
        with self._lock:
            now = time.monotonic()
            ready = self._healthy_locked()
            return {
                "connected": self._connected, "logged_in": self._logged_in,
                "subscribed": self._subscriptions == _CHANNELS,
                "subscribed_channels": sorted(self._subscriptions),
                "ready": ready, "healthy": ready, "generation": self._generation,
                "synchronizing": self._synchronizing, "source": "OKX private WebSocket + REST baseline",
                "last_error": self._last_error or ("REST reconciliation due" if not ready else ""),
                "retry_after_seconds": max(0.0, self._next_sync_at - now),
                "last_rest_age_seconds": None if self._last_rest_at is None else max(0.0, now - self._last_rest_at),
                "last_event_age_seconds": None if self._last_data_at is None else max(0.0, now - self._last_data_at),
                "reconcile_seconds": _RECONCILE_SECONDS,
                "refresh_seconds": _REFRESH_SECONDS,
                "transport": transport,
            }

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._healthy_locked():
                return None
            return {
                "account": copy.deepcopy(self._account),
                "positions": copy.deepcopy(list(self._positions.values())),
                "orders": copy.deepcopy(list(self._orders.values())),
                "generation": self._generation,
            }

    def synchronize(self, snapshot_provider: Callable[[], tuple[dict, dict, dict]]) -> bool:
        """Seed three REST responses after all subscriptions; throttle retries.

        A provider returns (balance_payload, positions_payload, orders_payload).
        It must obtain complete, symbol-filtered positions and pending orders.
        Refresh is attempted from age 45s, before the strict 60s deadline.
        During refresh, a still-valid old view continues receiving deltas.
        Failures invalidate it immediately; pongs, data pushes and an in-flight
        REST request never extend its existing 60s deadline.
        """
        with self._lock:
            healthy = self._healthy_locked()
            if healthy and time.monotonic() - self._last_rest_at < _REFRESH_SECONDS:
                return True
            if (
                self._closed or not self._connected or not self._logged_in
                or self._subscriptions != _CHANNELS or self._synchronizing
                or time.monotonic() < self._next_sync_at
            ):
                return healthy
            self._ready = healthy
            self._synchronizing = True
            self._sync_failed = False
            generation = self._generation
            self._buffer = []
        try:
            start_ms = self._server_ms()
            balance, positions, orders = snapshot_provider()
            end_ms = self._server_ms()
            account_rows = self._response_rows(balance)
            position_rows = self._response_rows(positions)
            order_rows = self._response_rows(orders)
            if len(account_rows) != 1 or end_ms < start_ms:
                raise ValueError("incomplete account baseline")
            with self._lock:
                if generation != self._generation or not self._connected or self._closed:
                    return False
                if self._sync_failed or not self._logged_in or self._subscriptions != _CHANNELS:
                    raise ValueError("private synchronization interrupted")
                # The lock makes successful publication atomic. Mark invalid
                # before changing any data, so even a failed seed cannot expose
                # half-rebuilt state in the exception-handler handoff.
                self._ready = False
                self._account = self._validated_account(account_rows[0])
                self._positions, self._orders = {}, {}
                self._position_versions, self._order_versions = {}, {}
                self._sync_start_ms, self._sync_end_ms = start_ms, end_ms
                for row in position_rows:
                    self._merge_position(row, baseline=True)
                for row in order_rows:
                    if str(row.get("state")) not in _OPEN_STATES:
                        raise ValueError("pending-orders baseline contains a non-open order")
                    self._merge_order(row, baseline=True)
                # Account pushes are frequent. Version merging permits them;
                # uncovered position/order changes during REST reads do not.
                for payload in self._buffer:
                    self._apply_locked(payload)
                self._buffer = []
                self._last_rest_at = time.monotonic()
                self._last_data_at = time.monotonic()
                self._last_error = ""
                self._next_sync_at = 0.0
                self._ready = True
                return True
        except Exception:
            with self._lock:
                if generation == self._generation:
                    self._invalidate_locked("OKX private REST synchronization failed or overlapped uncertain updates")
            return False
        finally:
            with self._lock:
                if generation == self._generation:
                    self._synchronizing = False

    @staticmethod
    def _response_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or str(payload.get("code")) != "0" or not isinstance(payload.get("data"), list):
            raise ValueError("invalid REST response")
        if not all(isinstance(row, dict) for row in payload["data"]):
            raise ValueError("invalid REST rows")
        return payload["data"]

    @staticmethod
    def _version(row: dict[str, Any], field: str = "uTime") -> int:
        value = int(row.get(field) or 0)
        if value <= 0:
            raise ValueError("missing update version")
        return value

    @staticmethod
    def _finite(row: dict[str, Any], field: str) -> float:
        value = float(row[field])
        if not math.isfinite(value):
            raise ValueError("nonfinite value")
        return value

    def _validated_account(self, row: dict[str, Any]) -> dict[str, Any]:
        self._version(row)
        self._finite(row, "totalEq")
        details = row.get("details", [])
        if not isinstance(details, list) or not all(
            isinstance(detail, dict) and isinstance(detail.get("ccy"), str) and detail["ccy"]
            for detail in details
        ):
            raise ValueError("invalid account details")
        return copy.deepcopy(row)

    def _apply_locked(self, payload: dict[str, Any]) -> None:
        channel = payload["arg"]["channel"]
        rows = payload.get("data")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("missing channel data")
        # Multi-page snapshots must never replace complete REST state with an
        # incomplete page. A fresh REST reconciliation is the safe fallback.
        if payload.get("lastPage") is False or int(payload.get("curPage", 1)) != 1:
            raise ValueError("paginated private snapshot requires REST")
        if channel == "account":
            if len(rows) != 1:
                raise ValueError("empty or fragmented account update")
            incoming = self._validated_account(rows[0])
            if self._version(incoming) < self._version(self._account):
                return
            if self._version(incoming) == self._version(self._account):
                for field in ("totalEq", "adjEq", "availEq"):
                    old, new = self._account.get(field), incoming.get(field)
                    if old not in (None, "") and new not in (None, "") and float(old) != float(new):
                        raise ValueError("conflicting account totals at the same version")
            details = {row["ccy"]: row for row in self._account.get("details", [])}
            for detail in incoming.get("details", []):
                old = details.get(detail["ccy"], {})
                if int(detail.get("uTime") or incoming["uTime"]) < int(old.get("uTime") or self._account["uTime"]):
                    continue
                details[detail["ccy"]] = copy.deepcopy(detail)
            self._account.update(incoming)
            self._account["details"] = list(details.values())
        elif channel == "positions":
            if payload.get("eventType") == "snapshot":
                incoming_keys = {self._position_key(row) for row in rows if row.get("instId") == self.symbol}
                if set(self._positions) - incoming_keys:
                    raise ValueError("snapshot omits previously known positions")
            for row in rows:
                self._merge_position(row)
        elif channel == "orders":
            for row in rows:
                self._merge_order(row)
        if rows:
            self._last_data_at = time.monotonic()

    def _check_fence(self, version: int, previous: int | None, *, terminal: bool) -> bool:
        if previous is not None and version <= previous:
            return True  # Entity-specific version handling below is authoritative.
        if version < self._sync_start_ms:
            return previous is not None
        if self._sync_start_ms <= version <= self._sync_end_ms:
            if terminal and previous is None:
                return False  # REST already confirmed this entity absent.
            raise ValueError("uncovered private delta overlaps REST snapshot window")
        return True

    def _position_key(self, row: dict[str, Any]) -> str:
        return str(row.get("posId") or f"{self.symbol}:{row.get('posSide', '')}:{row.get('mgnMode', '')}")

    def _merge_position(self, row: dict[str, Any], *, baseline: bool = False) -> None:
        if row.get("instId") != self.symbol:
            return
        side = str(row.get("posSide") or "")
        if side not in {"net", "long", "short"}:
            raise ValueError("invalid position side")
        key = self._position_key(row)
        version = (self._version(row), int(row.get("pTime") or 0))
        quantity = self._finite(row, "pos")
        previous = self._position_versions.get(key)
        if not baseline and not self._check_fence(version[0], previous[0] if previous else None, terminal=quantity == 0):
            return
        if previous and version < previous:
            return
        if previous == version and key not in self._positions:
            return
        if previous == version and key in self._positions and quantity != float(self._positions[key]["pos"]):
            raise ValueError("same-version conflicting position size")
        if quantity != 0:
            if self._finite(row, "avgPx") <= 0:
                raise ValueError("invalid position entry price")
            self._positions[key] = {**self._positions.get(key, {}), **copy.deepcopy(row)}
        else:
            self._positions.pop(key, None)
        self._position_versions[key] = version

    def _merge_order(self, row: dict[str, Any], *, baseline: bool = False) -> None:
        if row.get("instId") != self.symbol:
            return
        key = str(row.get("ordId") or "")
        state = str(row.get("state") or "")
        version = self._version(row)
        if not key or state not in _OPEN_STATES | _TERMINAL_STATES:
            raise ValueError("unknown order identity or state")
        fill = self._finite(row, "accFillSz")
        if fill < 0:
            raise ValueError("invalid order fill")
        previous = self._order_versions.get(key)
        if not baseline and not self._check_fence(version, previous[0] if previous else None, terminal=state in _TERMINAL_STATES):
            return
        if previous:
            if version < previous[0] or (previous[1] in _TERMINAL_STATES and state in _OPEN_STATES):
                return
            if fill < previous[2]:
                raise ValueError("order cumulative fill moved backwards")
            if version == previous[0] and previous[1] == "partially_filled" and state == "live":
                return
            if version == previous[0] and previous[1] in _TERMINAL_STATES and state != previous[1]:
                raise ValueError("conflicting terminal order states")
        if state in _OPEN_STATES:
            if self._finite(row, "sz") <= 0 or row.get("side") not in {"buy", "sell"}:
                raise ValueError("invalid open order")
            self._orders[key] = {**self._orders.get(key, {}), **copy.deepcopy(row)}
        else:
            self._orders.pop(key, None)
        self._order_versions[key] = (version, state, fill)
