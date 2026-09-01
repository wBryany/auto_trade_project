from __future__ import annotations

import copy
import json
import logging
import threading
import time
from decimal import Decimal
from typing import Any, Callable

from .http_client import request_json

try:  # Installed through the project dependency; kept optional for safe fallback.
    import websocket
except ImportError:  # pragma: no cover - exercised only in incomplete installations.
    websocket = None


LOG = logging.getLogger(__name__)

BINANCE_PRIVATE_STREAM_URLS = {
    # Binance separated public market and private user-data paths in 2026.
    "production": "wss://fstream.binance.com/private",
    "testnet": "wss://stream.binancefuture.com/private",
}

_OPEN_ORDER_STATUSES = {"NEW", "PARTIALLY_FILLED", "PENDING_NEW"}
_OPEN_ALGO_STATUSES = {"NEW", "WORKING", "TRIGGERING", "TRIGGERED"}


class BinanceUserDataStream:
    """Maintain a bootstrapped Binance futures account cache over WebSocket.

    Binance user-data events are deltas, not complete snapshots.  Every new
    socket is therefore opened first and then bootstrapped once through the
    supplied loader.  Messages received after ``on_open`` remain queued in the
    WebSocket transport while the loader runs, so applying them after the
    snapshot closes the usual REST/WebSocket race window.
    """

    def __init__(
        self,
        symbol: str,
        environment: str,
        rest_base_url: str,
        api_key_provider: Callable[[], str],
        snapshot_loader: Callable[[], dict[str, Any]],
        *,
        stale_seconds: float = 75.0,
        keepalive_seconds: float = 30 * 60,
    ) -> None:
        self.symbol = symbol.strip().upper()
        self.environment = environment.strip().lower()
        stream_base = BINANCE_PRIVATE_STREAM_URLS.get(self.environment)
        if stream_base is None:
            raise ValueError("Binance user stream environment must be testnet or production")
        self.stream_base = stream_base
        self.rest_base_url = rest_base_url.rstrip("/")
        self._api_key_provider = api_key_provider
        self._snapshot_loader = snapshot_loader
        self.stale_seconds = max(45.0, float(stale_seconds))
        self.keepalive_seconds = max(60.0, float(keepalive_seconds))

        self._condition = threading.Condition(threading.RLock())
        self._stop = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._socket: Any = None
        self._listen_key = ""
        self._last_keepalive_at = 0.0
        self._connected = False
        self._ready = False
        self._last_transport_at = 0.0
        self._last_event_at = 0.0
        self._last_bootstrap_at = 0.0
        self._last_error = ""
        self._reconnects = 0

        self._account: dict[str, Any] = {}
        self._positions: dict[tuple[str, str], dict[str, Any]] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._algo_orders: dict[str, dict[str, Any]] = {}
        self._recent_orders: dict[str, dict[str, Any]] = {}
        self._recent_algo_orders: dict[str, dict[str, Any]] = {}

    @property
    def available(self) -> bool:
        return websocket is not None

    def start(self, *, wait_seconds: float = 0.0) -> bool:
        if not self.available:
            with self._condition:
                self._last_error = "websocket-client is not installed"
            return False
        with self._condition:
            if not self._thread or not self._thread.is_alive():
                self._stop.clear()
                self._ready_event.clear()
                self._thread = threading.Thread(
                    target=self._run,
                    name=f"binance-user-stream-{self.symbol.lower()}",
                    daemon=True,
                )
                self._keepalive_thread = threading.Thread(
                    target=self._keepalive_loop,
                    name=f"binance-user-keepalive-{self.symbol.lower()}",
                    daemon=True,
                )
                self._thread.start()
                self._keepalive_thread.start()
        if wait_seconds > 0:
            self._ready_event.wait(float(wait_seconds))
        return self.healthy()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            active_socket = self._socket
            stream_thread = self._thread
            keepalive_thread = self._keepalive_thread
        if active_socket is not None:
            try:
                active_socket.close()
            except Exception:
                LOG.debug("Binance user stream socket close failed", exc_info=True)
        for thread in (stream_thread, keepalive_thread):
            if thread and thread is not threading.current_thread():
                thread.join(timeout=3)
        self._close_listen_key()
        with self._condition:
            self._connected = False
            self._ready = False
            self._thread = None
            self._keepalive_thread = None
            self._socket = None
            self._ready_event.clear()
            self._condition.notify_all()

    def healthy(self) -> bool:
        with self._condition:
            return bool(
                self._connected
                and self._ready
                and self._last_transport_at
                and time.monotonic() - self._last_transport_at <= self.stale_seconds
            )

    def status(self) -> dict[str, Any]:
        with self._condition:
            transport_age = (
                max(0.0, time.monotonic() - self._last_transport_at)
                if self._last_transport_at
                else None
            )
            event_age = (
                max(0.0, time.monotonic() - self._last_event_at)
                if self._last_event_at
                else None
            )
            return {
                "available": self.available,
                "connected": self._connected,
                "ready": self._ready,
                "healthy": self.healthy(),
                "source": "Binance USDⓈ-M User Data Stream",
                "last_transport_age_seconds": transport_age,
                "last_event_age_seconds": event_age,
                "last_error": self._last_error,
                "reconnects": self._reconnects,
            }

    def snapshot(self, *, wait_seconds: float = 0.0) -> dict[str, Any]:
        self.start(wait_seconds=wait_seconds)
        with self._condition:
            if not self.healthy():
                reason = self._last_error or "Binance private WebSocket is not ready"
                raise RuntimeError(reason)
            return {
                "account": copy.deepcopy(self._account),
                "positions": copy.deepcopy(list(self._positions.values())),
                "orders": copy.deepcopy(list(self._orders.values())),
                "algo_orders": copy.deepcopy(list(self._algo_orders.values())),
            }

    def wait_for_order(self, client_id: str, *, timeout: float = 2.0) -> dict[str, Any] | None:
        return self._wait_for_cached(self._recent_orders, client_id, timeout)

    def wait_for_algo_order(self, client_id: str, *, timeout: float = 2.0) -> dict[str, Any] | None:
        return self._wait_for_cached(self._recent_algo_orders, client_id, timeout)

    def record_order_response(self, row: dict[str, Any]) -> None:
        """Make a successful synchronous order response visible immediately."""

        if not isinstance(row, dict):
            return
        client_id = str(row.get("clientOrderId") or "")
        order_id = str(row.get("orderId") or "")
        if not client_id and not order_id:
            return
        with self._condition:
            key = client_id or order_id
            cached = dict(row)
            self._recent_orders[key] = cached
            if client_id:
                self._recent_orders[client_id] = cached
            if order_id:
                self._recent_orders[order_id] = cached
            status = str(row.get("status") or "").upper()
            if status in _OPEN_ORDER_STATUSES:
                self._orders[key] = cached
            self._trim_recent(self._recent_orders)
            self._condition.notify_all()

    def seed_snapshot(self, payload: dict[str, Any]) -> None:
        account = copy.deepcopy(dict(payload.get("account") or {}))
        positions = [copy.deepcopy(dict(row)) for row in payload.get("positions", [])]
        orders = [copy.deepcopy(dict(row)) for row in payload.get("orders", [])]
        algo_orders = [copy.deepcopy(dict(row)) for row in payload.get("algo_orders", [])]
        with self._condition:
            self._account = account
            self._positions = {
                self._position_key(row): row
                for row in positions
                if str(row.get("symbol") or "").upper() == self.symbol
            }
            self._orders = {}
            self._recent_orders = {}
            for row in orders:
                self._store_regular_order(row)
            self._algo_orders = {}
            self._recent_algo_orders = {}
            for row in algo_orders:
                self._store_algo_order(row)
            self._last_bootstrap_at = time.monotonic()
            self._ready = True
            self._last_error = ""
            self._ready_event.set()
            self._condition.notify_all()

    def process_message(self, raw_message: str) -> None:
        payload = json.loads(raw_message)
        data = payload.get("data", payload)
        event = str(data.get("e") or "")
        with self._condition:
            self._last_transport_at = time.monotonic()
            if event:
                self._last_event_at = self._last_transport_at
            if event == "ACCOUNT_UPDATE":
                self._apply_account_update(data.get("a") or {})
            elif event == "ORDER_TRADE_UPDATE":
                self._apply_order_update(data.get("o") or {})
            elif event == "ALGO_UPDATE":
                self._apply_algo_update(data.get("o") or data.get("ao") or {})
            elif event == "ACCOUNT_CONFIG_UPDATE":
                self._apply_config_update(data)
            elif event == "MARGIN_CALL":
                for row in data.get("p") or []:
                    self._apply_margin_position(row)
            elif event == "listenKeyExpired":
                self._last_error = "Binance private WebSocket listenKey expired"
                self._ready = False
                self._listen_key = ""
                active_socket = self._socket
                if active_socket is not None:
                    active_socket.close()
            self._condition.notify_all()

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            had_ready_connection = False
            try:
                listen_key = self._ensure_listen_key()
                active_socket = websocket.WebSocketApp(
                    f"{self.stream_base}/ws/{listen_key}",
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_pong=self._on_pong,
                )
                with self._condition:
                    self._socket = active_socket
                active_socket.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as error:
                self._record_error(error)
            finally:
                with self._condition:
                    had_ready_connection = self._ready
                    self._connected = False
                    self._ready = False
                    self._socket = None
                    self._ready_event.clear()
                    self._condition.notify_all()
            reconnect_delay = 1.0 if had_ready_connection else backoff
            if self._stop.wait(reconnect_delay):
                break
            with self._condition:
                self._reconnects += 1
            backoff = 1.0 if had_ready_connection else min(30.0, backoff * 2)

    def _keepalive_loop(self) -> None:
        while not self._stop.wait(1.0):
            with self._condition:
                listen_key = self._listen_key
                due = bool(
                    listen_key
                    and time.monotonic() - self._last_keepalive_at >= self.keepalive_seconds
                )
            if not due:
                continue
            try:
                payload = self._listen_key_request("PUT")
                replacement = str((payload or {}).get("listenKey") or "")
                with self._condition:
                    self._last_keepalive_at = time.monotonic()
                    if replacement and replacement != self._listen_key:
                        self._listen_key = replacement
                        active_socket = self._socket
                        if active_socket is not None:
                            active_socket.close()
            except Exception as error:
                self._record_error(error)
                # A temporary network failure should not discard a still-valid
                # key. Retry after one minute, well before its 60-minute TTL.
                with self._condition:
                    self._last_keepalive_at = time.monotonic() - self.keepalive_seconds + 60.0

    def _ensure_listen_key(self) -> str:
        with self._condition:
            if self._listen_key:
                return self._listen_key
        payload = self._listen_key_request("POST")
        listen_key = str((payload or {}).get("listenKey") or "")
        if not listen_key:
            raise RuntimeError("Binance did not return a private WebSocket listenKey")
        with self._condition:
            self._listen_key = listen_key
            self._last_keepalive_at = time.monotonic()
        return listen_key

    def _close_listen_key(self) -> None:
        with self._condition:
            listen_key = self._listen_key
            self._listen_key = ""
        if not listen_key:
            return
        try:
            self._listen_key_request("DELETE")
        except Exception:
            LOG.debug("Binance listenKey close failed", exc_info=True)

    def _listen_key_request(self, method: str) -> Any:
        api_key = str(self._api_key_provider() or "")
        if not api_key:
            raise RuntimeError("Binance private WebSocket requires an API key")
        return request_json(
            method,
            f"{self.rest_base_url}/fapi/v1/listenKey",
            headers={"X-MBX-APIKEY": api_key},
            timeout=5.0,
            max_attempts=1,
        )

    def _on_open(self, _socket: Any) -> None:
        with self._condition:
            self._connected = True
            self._ready = False
            self._last_transport_at = time.monotonic()
            self._last_error = ""
        try:
            self.seed_snapshot(self._snapshot_loader())
        except Exception as error:
            self._record_error(error)
            with self._condition:
                self._ready = False
            _socket.close()
            return
        LOG.info("Binance private WebSocket user stream connected for %s", self.symbol)

    def _on_message(self, _socket: Any, message: str) -> None:
        try:
            self.process_message(message)
        except Exception as error:
            self._record_error(error)

    def _on_pong(self, _socket: Any, _payload: Any) -> None:
        with self._condition:
            self._last_transport_at = time.monotonic()

    def _on_error(self, _socket: Any, error: Any) -> None:
        self._record_error(error)

    def _on_close(self, _socket: Any, status_code: Any, message: Any) -> None:
        with self._condition:
            self._connected = False
            self._ready = False
            self._ready_event.clear()
            self._condition.notify_all()
        if not self._stop.is_set():
            LOG.warning(
                "Binance private WebSocket user stream closed code=%s message=%s",
                status_code,
                message,
            )

    def _record_error(self, error: Any) -> None:
        message = str(error)
        with self._condition:
            self._last_error = message
            self._condition.notify_all()
        if not self._stop.is_set():
            LOG.warning("Binance private WebSocket user stream error: %s", message)

    def _apply_account_update(self, update: dict[str, Any]) -> None:
        assets = self._account.setdefault("assets", [])
        for incoming in update.get("B") or []:
            asset_name = str(incoming.get("a") or "")
            row = next((item for item in assets if str(item.get("asset") or "") == asset_name), None)
            if row is None:
                row = {"asset": asset_name}
                assets.append(row)
            if incoming.get("wb") is not None:
                row["walletBalance"] = str(incoming["wb"])
            if incoming.get("cw") is not None:
                row["crossWalletBalance"] = str(incoming["cw"])
                row["availableBalance"] = str(incoming["cw"])
        for incoming in update.get("P") or []:
            symbol = str(incoming.get("s") or "").upper()
            if symbol != self.symbol:
                continue
            key = (symbol, str(incoming.get("ps") or "BOTH"))
            row = self._positions.get(key, {"symbol": symbol, "positionSide": key[1]})
            mappings = {
                "pa": "positionAmt",
                "ep": "entryPrice",
                "bep": "breakEvenPrice",
                "up": "unRealizedProfit",
                "iw": "isolatedWallet",
                "mt": "marginType",
            }
            for source, target in mappings.items():
                if incoming.get(source) is not None:
                    row[target] = incoming[source]
            self._positions[key] = row
        self._recalculate_account_totals()

    def _apply_order_update(self, update: dict[str, Any]) -> None:
        row = {
            "symbol": update.get("s", self.symbol),
            "clientOrderId": update.get("c"),
            "side": update.get("S"),
            "type": update.get("o"),
            "timeInForce": update.get("f"),
            "origQty": update.get("q", "0"),
            "price": update.get("p", "0"),
            "avgPrice": update.get("ap", "0"),
            "stopPrice": update.get("sp", "0"),
            "executionType": update.get("x"),
            "status": update.get("X"),
            "orderId": update.get("i"),
            "executedQty": update.get("z", "0"),
            "lastFilledQty": update.get("l", "0"),
            "lastFilledPrice": update.get("L", "0"),
            "reduceOnly": update.get("R", False),
            "closePosition": update.get("cp", False),
            "positionSide": update.get("ps", "BOTH"),
            "cumQuote": update.get("b", "0"),
            "realizedProfit": update.get("rp", "0"),
            "updateTime": update.get("T") or int(time.time() * 1000),
        }
        self._store_regular_order(row)

    def _apply_algo_update(self, update: dict[str, Any]) -> None:
        row = {
            "algoId": update.get("algoId") or update.get("i") or update.get("aid"),
            "clientAlgoId": update.get("clientAlgoId") or update.get("c") or update.get("caid"),
            "algoType": update.get("algoType") or update.get("at") or "CONDITIONAL",
            "orderType": update.get("orderType") or update.get("o") or update.get("ot"),
            "symbol": update.get("symbol") or update.get("s") or self.symbol,
            "side": update.get("side") or update.get("S"),
            "positionSide": update.get("positionSide") or update.get("ps") or "BOTH",
            # ALGO_UPDATE uses the compact ``tp`` field for trigger price.
            # ``sp`` belongs to the legacy regular-order event shape.
            "triggerPrice": (
                update.get("triggerPrice")
                or update.get("tp")
                or update.get("sp")
                or "0"
            ),
            "price": update.get("price") or update.get("p") or "0",
            "quantity": update.get("quantity") or update.get("q") or "0",
            "closePosition": update.get("closePosition") or update.get("cp") or False,
            "reduceOnly": update.get("reduceOnly") or update.get("R") or False,
            "workingType": update.get("workingType") or update.get("wt") or "MARK_PRICE",
            "algoStatus": update.get("algoStatus") or update.get("X") or update.get("status"),
            "updateTime": update.get("updateTime") or update.get("T") or int(time.time() * 1000),
        }
        self._store_algo_order(row)

    def _apply_config_update(self, event: dict[str, Any]) -> None:
        config = event.get("ac") or {}
        symbol = str(config.get("s") or "").upper()
        if symbol != self.symbol or config.get("l") is None:
            return
        for (row_symbol, _side), row in self._positions.items():
            if row_symbol == symbol:
                row["leverage"] = str(config["l"])

    def _apply_margin_position(self, incoming: dict[str, Any]) -> None:
        symbol = str(incoming.get("s") or "").upper()
        if symbol != self.symbol:
            return
        key = (symbol, str(incoming.get("ps") or "BOTH"))
        row = self._positions.get(key, {"symbol": symbol, "positionSide": key[1]})
        for source, target in {
            "pa": "positionAmt",
            "ep": "entryPrice",
            "up": "unRealizedProfit",
            "iw": "isolatedWallet",
            "mp": "markPrice",
        }.items():
            if incoming.get(source) is not None:
                row[target] = incoming[source]
        self._positions[key] = row
        self._recalculate_account_totals()

    def _store_regular_order(self, row: dict[str, Any]) -> None:
        order_id = str(row.get("orderId") or "")
        client_id = str(row.get("clientOrderId") or "")
        key = order_id or client_id
        if not key:
            return
        cached = copy.deepcopy(row)
        self._recent_orders[key] = cached
        if client_id:
            self._recent_orders[client_id] = cached
        if order_id:
            self._recent_orders[order_id] = cached
        status = str(row.get("status") or "").upper()
        if status in _OPEN_ORDER_STATUSES:
            self._orders[key] = cached
        else:
            self._orders.pop(key, None)
            for existing_key, existing in list(self._orders.items()):
                if order_id and str(existing.get("orderId") or "") == order_id:
                    self._orders.pop(existing_key, None)
                elif client_id and str(existing.get("clientOrderId") or "") == client_id:
                    self._orders.pop(existing_key, None)
        self._trim_recent(self._recent_orders)

    def _store_algo_order(self, row: dict[str, Any]) -> None:
        algo_id = str(row.get("algoId") or "")
        client_id = str(row.get("clientAlgoId") or "")
        key = algo_id or client_id
        if not key:
            return
        cached = copy.deepcopy(row)
        self._recent_algo_orders[key] = cached
        if client_id:
            self._recent_algo_orders[client_id] = cached
        if algo_id:
            self._recent_algo_orders[algo_id] = cached
        status = str(row.get("algoStatus") or "").upper()
        if not status or status in _OPEN_ALGO_STATUSES:
            self._algo_orders[key] = cached
        else:
            self._algo_orders.pop(key, None)
            for existing_key, existing in list(self._algo_orders.items()):
                if algo_id and str(existing.get("algoId") or "") == algo_id:
                    self._algo_orders.pop(existing_key, None)
                elif client_id and str(existing.get("clientAlgoId") or "") == client_id:
                    self._algo_orders.pop(existing_key, None)
        self._trim_recent(self._recent_algo_orders)

    def _recalculate_account_totals(self) -> None:
        assets = self._account.get("assets") or []
        usdt = next((row for row in assets if str(row.get("asset") or "") == "USDT"), None)
        if usdt:
            wallet = str(usdt.get("walletBalance") or self._account.get("totalWalletBalance") or "0")
            available = str(usdt.get("availableBalance") or usdt.get("crossWalletBalance") or wallet)
            self._account["totalWalletBalance"] = wallet
            self._account["availableBalance"] = available
        unrealized = sum(
            Decimal(str(row.get("unRealizedProfit") or "0"))
            for row in self._positions.values()
        )
        wallet_total = Decimal(str(self._account.get("totalWalletBalance") or "0"))
        self._account["totalUnrealizedProfit"] = str(unrealized)
        self._account["totalMarginBalance"] = str(wallet_total + unrealized)

    def _wait_for_cached(
        self,
        cache: dict[str, dict[str, Any]],
        key: str,
        timeout: float,
    ) -> dict[str, Any] | None:
        lookup = str(key or "")
        if not lookup:
            return None
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                row = cache.get(lookup)
                if row is not None:
                    return copy.deepcopy(row)
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._connected:
                    return None
                self._condition.wait(min(remaining, 0.25))

    @staticmethod
    def _position_key(row: dict[str, Any]) -> tuple[str, str]:
        return (
            str(row.get("symbol") or "").upper(),
            str(row.get("positionSide") or "BOTH"),
        )

    @staticmethod
    def _trim_recent(cache: dict[str, dict[str, Any]], maximum: int = 300) -> None:
        overflow = len(cache) - maximum
        if overflow <= 0:
            return
        for key in list(cache)[:overflow]:
            cache.pop(key, None)
