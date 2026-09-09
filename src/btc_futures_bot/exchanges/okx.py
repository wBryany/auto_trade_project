from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlencode
from typing import Any

from ..http_client import ApiError, format_number, request_json
from ..models import Candle, OrderRequest, Position
from ..okx_stream import OkxMarketStream
from ..okx_user_stream import OkxPrivateStream
from .base import ExchangeAdapter, ExchangeSettings, candle_limit_for


LOG = logging.getLogger(__name__)


class OkxAdapter(ExchangeAdapter):
    def __init__(self, settings: ExchangeSettings) -> None:
        super().__init__(settings)
        self.base_url = settings.base_url.rstrip("/")
        self._scalp_cache: dict[int, Candle] = {}
        self._server_time_offset_ms = 0
        self._server_time_anchor_ms = 0
        self._server_time_anchor_monotonic = 0.0
        self._server_time_synced_at = 0.0
        self._server_time_lock = threading.RLock()
        self._rest_lock = threading.RLock()
        self._rest_failures: dict[str, tuple[float, Exception]] = {}
        self._snapshot_lock = threading.RLock()
        self._reconcile_lock = threading.Lock()
        self._last_rest_snapshot: dict[str, Any] | None = None
        self._last_rest_snapshot_at = 0.0
        self._last_rest_market: dict[str, Any] = {}
        self._last_rest_market_at = 0.0
        self._closed = False
        self._market_stream = (
            OkxMarketStream(settings.symbol, settings.environment,
                            stale_seconds=settings.websocket_stale_seconds,
                            now_ms=self._server_timestamp_ms)
            if settings.websocket_enabled else None
        )
        self._private_stream = (
            OkxPrivateStream(settings.symbol, settings.environment,
                             credentials=self.credentials, timestamp=self._ws_timestamp,
                             stale_seconds=90.0)
            if settings.websocket_enabled else None
        )

    def _ensure_streams(self) -> None:
        if self._closed:
            raise RuntimeError("OKX adapter is closed")
        if self._market_stream is not None:
            # Signed requests and cache freshness share the exchange-time anchor.
            self._ensure_server_time()
            self._market_stream.start()
        if self._private_stream is not None and self.has_credentials():
            self._private_stream.start()

    def _ws_timestamp(self) -> str:
        self._ensure_server_time()
        return str(self._server_timestamp_ms() / 1000)

    def close(self) -> None:
        self._closed = True
        for stream in (self._market_stream, self._private_stream):
            if stream is not None:
                stream.close()

    def _guard_rest(self, key: str) -> None:
        with self._rest_lock:
            host_failure = self._rest_failures.get("*")
            failure = host_failure if host_failure and time.monotonic() < host_failure[0] else self._rest_failures.get(key)
        if failure and time.monotonic() < failure[0]:
            raise failure[1]

    def _rest_failed(self, key: str, error: Exception) -> None:
        delay = max(5.0, getattr(error, "retry_after_seconds", 0.0))
        if getattr(error, "status_code", None) in {401, 403, 451}:
            delay = max(delay, 60.0)
        with self._rest_lock:
            self._rest_failures[key] = (time.monotonic() + delay, error)
            if isinstance(error, ApiError) and error.rate_limited:
                self._rest_failures["*"] = self._rest_failures[key]

    def _public(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        key = path + "?" + urlencode(params or {})
        self._guard_rest(key)
        try:
            payload = request_json("GET", f"{self.base_url}{path}", params=params,
                                   headers={"x-simulated-trading": "1"} if self.settings.environment == "demo" else None,
                                   timeout=5.0, max_attempts=1)
            self._check_response(payload)
            return payload
        except (ApiError, RuntimeError, TimeoutError, OSError) as error:
            self._rest_failed(key, error)
            raise

    def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
        self._ensure_streams()
        stream = self._market_stream
        if stream is not None:
            streamed = stream.candles(interval, limit)
            if streamed is not None:
                return streamed
        generation = stream.seed_generation() if stream is not None else None
        if interval == "30s":
            payload = self._public(
                "/api/v5/market/candles",
                params={"instId": self.settings.symbol, "bar": "1s", "limit": candle_limit_for("1s", 300)},
            )
            self._check_response(payload)
            one_second = [self._candle_from_row(row) for row in payload.get("data", [])]
            one_second.sort(key=lambda candle: candle.timestamp)
            for candle in self._aggregate_30s(one_second):
                self._scalp_cache[candle.timestamp] = candle
            if len(self._scalp_cache) > 600:
                self._scalp_cache = dict(sorted(self._scalp_cache.items())[-600:])
            return list(sorted(self._scalp_cache.values(), key=lambda candle: candle.timestamp))[-max(10, min(int(limit), 600)):]

        bar = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H"}[interval]
        params = {"instId": self.settings.symbol, "bar": bar, "limit": min(300, candle_limit_for(interval, limit))}
        try:
            payload = self._public(
                "/api/v5/market/candles",
                params=params,
            )
            self._check_response(payload)
        except (ApiError, RuntimeError, TimeoutError, OSError) as error:
            if interval != "5m" or stream is not None:
                raise
            return self._fetch_5m_from_official_1m(limit, error)
        candles = [self._candle_from_row(row) for row in payload.get("data", [])]
        candles.sort(key=lambda candle: candle.timestamp)
        if stream is not None:
            duration = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}[interval]
            current_bucket = self._server_timestamp_ms() // duration * duration
            raw_rows = sorted(payload.get("data", []), key=lambda row: int(row[0]))
            newest_row = raw_rows[-1] if raw_rows else []
            if (not candles or candles[-1].timestamp != current_bucket
                    or not all(OkxMarketStream._valid_candle(candle, duration) for candle in candles)
                    or len(newest_row) < 9 or str(newest_row[8]) != "0"
                    or any(len(row) < 9 or str(row[8]) != "1" for row in raw_rows[:-1])
                    or any(right.timestamp - left.timestamp != duration for left, right in zip(candles, candles[1:]))):
                error = ApiError("OKX REST candle history is stale, incomplete or missing the current forming bar")
                self._rest_failed("/api/v5/market/candles?" + urlencode(params), error)
                raise error
            stream.seed_candles(interval, candles, generation=generation)
        return candles

    def _fetch_5m_from_official_1m(self, limit: int, primary_error: Exception) -> list[Candle]:
        """Fall back to OKX 1m candles when its direct 5m request is unavailable.

        The engine already removes the newest forming candle, so this method
        deliberately keeps the newest (possibly partial) five-minute bucket.
        """
        LOG.warning("OKX 5m candles unavailable; aggregating official 1m candles: %s", primary_error)
        try:
            payload = self._public(
                "/api/v5/market/candles",
                params={
                    "instId": self.settings.symbol,
                    "bar": "1m",
                    "limit": candle_limit_for("1m", 300),
                },
            )
            self._check_response(payload)
            one_minute = [self._candle_from_row(row) for row in payload.get("data", [])]
            one_minute.sort(key=lambda candle: candle.timestamp)
            aggregated = self._aggregate_timeframe(one_minute, 300_000)
            if len(aggregated) < 2:
                raise RuntimeError("OKX official 1m fallback returned insufficient data")
            return aggregated[-max(10, min(int(limit), 300)) :]
        except Exception as fallback_error:
            raise ApiError(
                f"OKX 5m candles failed and official 1m fallback failed: {fallback_error}"
            ) from primary_error

    @staticmethod
    def _candle_from_row(row: list[Any]) -> Candle:
        quote_volume = None
        if len(row) > 7 and row[7] not in (None, ""):
            quote_volume = float(row[7])
        return Candle(
            int(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            quote_volume=quote_volume,
        )

    @staticmethod
    def _aggregate_30s(candles: list[Candle]) -> list[Candle]:
        return OkxAdapter._aggregate_timeframe(candles, 30_000)

    @staticmethod
    def _aggregate_timeframe(candles: list[Candle], bucket_ms: int) -> list[Candle]:
        if bucket_ms <= 0:
            raise ValueError("bucket_ms must be positive")
        buckets: dict[int, list[Candle]] = {}
        for candle in candles:
            bucket = candle.timestamp // bucket_ms * bucket_ms
            buckets.setdefault(bucket, []).append(candle)
        result: list[Candle] = []
        for timestamp, group in sorted(buckets.items()):
            quote_volume = None
            if any(candle.quote_volume is not None for candle in group):
                quote_volume = sum(candle.quote_volume or 0.0 for candle in group)
            result.append(
                Candle(
                    timestamp,
                    group[0].open,
                    max(candle.high for candle in group),
                    min(candle.low for candle in group),
                    group[-1].close,
                    sum(candle.volume for candle in group),
                    quote_volume=quote_volume,
                )
            )
        return result

    def fetch_equity(self) -> float:
        self._ensure_streams()
        private = self._private_stream.snapshot() if self._private_stream is not None else None
        if private is not None:
            row = private["account"]
            return float(row.get("totalEq") or row.get("adjEq") or 0)
        payload = self._private("GET", "/api/v5/account/balance")
        self._check_response(payload)
        row = (payload.get("data") or [{}])[0]
        return float(row.get("totalEq") or row.get("adjEq") or 0)

    def fetch_live_market_snapshot(self) -> dict[str, Any] | None:
        return self._market_stream.market_snapshot() if self._market_stream is not None else None

    def _fetch_rest_market(self) -> dict[str, Any]:
        mark_payload = self._public(
            "/api/v5/public/mark-price",
            params={"instType": "SWAP", "instId": self.settings.symbol},
        )
        self._check_response(mark_payload)
        mark = (mark_payload.get("data") or [{}])[0]
        mark_price_raw = str(mark.get("markPx") or "0")
        timestamp = int(mark.get("ts") or 0)
        if (not math.isfinite(float(mark_price_raw)) or float(mark_price_raw) <= 0
                or (self._market_stream is not None and (
                    timestamp <= 0 or not -5000 <= self._server_timestamp_ms() - timestamp <= self.settings.websocket_stale_seconds * 1000))):
            error = ApiError(f"OKX returned an invalid or stale mark price for {self.settings.symbol}")
            self._rest_failed("/api/v5/public/mark-price?" + urlencode({"instType": "SWAP", "instId": self.settings.symbol}), error)
            raise error
        market = {
            "symbol": self.settings.symbol,
            "mark_price": float(mark_price_raw),
            "mark_price_raw": mark_price_raw,
            "last_price": float(mark_price_raw),
            "last_price_raw": mark_price_raw,
            "price_source": "OKX 永续合约标记价格",
            "price_endpoint": "/api/v5/public/mark-price",
            "timestamp": int(mark.get("ts") or self.now_ms()),
        }
        with self._snapshot_lock:
            self._last_rest_market = dict(market)
            self._last_rest_market_at = time.monotonic()
        return market

    def fetch_mark_price(self) -> float:
        self._ensure_streams()
        market = self.fetch_live_market_snapshot() or self._fetch_rest_market()
        return float(market["mark_price"])

    def _fetch_private_baseline(self) -> tuple[Any, Any, Any]:
        balance = self._private("GET", "/api/v5/account/balance")
        positions = self._private("GET", "/api/v5/account/positions", params={"instId": self.settings.symbol})
        # An orders subscription has no initial snapshot. A truncated REST
        # first page must never become an authoritative list of pending orders.
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        cursor = ""
        for _ in range(10):
            params = {"instType": "SWAP", "instId": self.settings.symbol, "limit": 100}
            if cursor:
                params["after"] = cursor
            page = self._private("GET", "/api/v5/trade/orders-pending", params=params)
            self._check_response(page)
            batch = page.get("data")
            if not isinstance(batch, list):
                raise ApiError("OKX pending-order snapshot has invalid data")
            for row in batch:
                order_id = str(row.get("ordId") or "")
                if not order_id or order_id in seen:
                    raise ApiError("OKX pending-order pagination is ambiguous; refusing partial snapshot")
                seen.add(order_id)
                rows.append(row)
            if len(batch) < 100:
                break
            cursor = str(batch[-1]["ordId"])
        else:
            raise ApiError("OKX pending-order snapshot exceeds bounded pagination; refusing partial snapshot")
        orders = {"code": "0", "data": rows}
        for payload in (balance, positions, orders):
            self._check_response(payload)
        return balance, positions, orders

    def _stream_status(self) -> dict[str, Any]:
        return {
            "market_stream": self._market_stream.status() if self._market_stream is not None else {},
            "private_stream": self._private_stream.status() if self._private_stream is not None else {},
        }

    def fetch_live_dashboard_snapshot(self) -> dict[str, Any] | None:
        """No REST or waits on the browser's one-second refresh path."""
        if self._market_stream is None:
            return None
        market = self.fetch_live_market_snapshot()
        private = self._private_stream.snapshot() if self._private_stream is not None else None
        with self._snapshot_lock:
            baseline = dict(self._last_rest_snapshot or {})
            baseline_age = time.monotonic() - self._last_rest_snapshot_at
            rest_market = dict(self._last_rest_market)
            market_age = time.monotonic() - self._last_rest_market_at
        snapshot = {**baseline, **self._stream_status()}
        if market is not None:
            snapshot["market"] = market
        else:
            snapshot["market"] = {**rest_market, "stale": market_age >= 15 or not rest_market}
        if private is not None:
            snapshot.update(self._private_view(private["account"], private["positions"], private["orders"]))
            snapshot.update(private_available=True, private_source="websocket", private_stale=False, private_error="", private_warning="")
        else:
            snapshot.update(private_source="rest", private_warning="OKX 私有 WebSocket 未就绪，使用有时限的 REST 核对结果")
            if baseline_age >= 15 or not baseline.get("private_available"):
                snapshot.update(private_available=False, private_stale=True,
                                private_error=str(baseline.get("private_error") or "等待 OKX 私有数据重新同步"))
        return snapshot

    def fetch_dashboard_snapshot(self) -> dict[str, Any]:
        self._ensure_streams()
        snapshot: dict[str, Any] = {
            "market": self.fetch_live_market_snapshot() or self._fetch_rest_market(),
            "positions": [], "open_orders": [], "private_available": self.has_credentials(),
            "private_source": "rest", **self._stream_status(),
        }
        if not self.has_credentials():
            snapshot["private_error"] = "未配置完整的 OKX API Key、Secret 或 Passphrase"
            return snapshot
        try:
            with self._reconcile_lock:
                observed: list[tuple[Any, Any, Any]] = []
                baseline_error: list[Exception] = []

                def baseline_provider() -> tuple[Any, Any, Any]:
                    try:
                        result = self._fetch_private_baseline()
                    except Exception as error:
                        baseline_error.append(error)
                        raise
                    observed.append(result)
                    return result

                if self._private_stream is not None:
                    self._private_stream.synchronize(baseline_provider)
                    private = self._private_stream.snapshot()
                else:
                    private = None
                if private is not None:
                    snapshot.update(self._private_view(private["account"], private["positions"], private["orders"]))
                    snapshot["private_source"] = "websocket"
                else:
                    if baseline_error:
                        raise baseline_error[-1]
                    payloads = observed[-1] if observed else baseline_provider()
                    balance = (payloads[0].get("data") or [{}])[0]
                    snapshot.update(self._private_view(balance, payloads[1].get("data", []), payloads[2].get("data", [])))
                    if self._private_stream is not None:
                        snapshot["private_warning"] = "OKX 私有 WebSocket 未就绪，已使用 REST 核对"
                snapshot.update(self._stream_status())
                if snapshot["private_source"] == "rest":
                    with self._snapshot_lock:
                        self._last_rest_snapshot = dict(snapshot)
                        self._last_rest_snapshot_at = time.monotonic()
        except Exception as error:
            snapshot["private_available"] = False
            snapshot["private_error"] = str(error)
            with self._snapshot_lock:
                if self._last_rest_snapshot:
                    self._last_rest_snapshot.update(private_available=False, private_stale=True, private_error=str(error))
        return snapshot

    def _private_view(self, balance: dict[str, Any], positions: list[Any], orders: list[Any]) -> dict[str, Any]:
        return {
            "account": {
                "wallet_balance": float(balance.get("totalEq") or balance.get("adjEq") or 0),
                "available_balance": float(balance.get("availEq") or 0),
                "unrealized_pnl": 0.0,
                "margin_balance": float(balance.get("totalEq") or balance.get("adjEq") or 0),
            },
            "positions": [
                {
                    "symbol": row.get("instId", self.settings.symbol),
                    "side": ("short" if row.get("posSide") == "short" or float(row.get("pos") or 0) < 0 else "long"),
                    "quantity": abs(float(row.get("pos") or 0)),
                    "entry_price": float(row.get("avgPx") or 0),
                    "mark_price": float(row.get("markPx") or 0),
                    "unrealized_pnl": float(row.get("upl") or 0),
                    "liquidation_price": float(row.get("liqPx") or 0),
                    "leverage": float(row.get("lever") or 0),
                    "position_side": row.get("posSide", "net"),
                }
                for row in positions
                if row.get("instId") == self.settings.symbol and abs(float(row.get("pos") or 0)) > 0
            ],
            "open_orders": [
                {
                    "order_id": row.get("ordId"),
                    "client_order_id": row.get("clOrdId"),
                    "side": row.get("side"),
                    "type": row.get("ordType"),
                    "status": row.get("state"),
                    "quantity": float(row.get("sz") or 0),
                    "executed_quantity": float(row.get("accFillSz") or 0),
                    "price": float(row.get("px") or 0),
                    "stop_price": float(row.get("slTriggerPx") or 0),
                    "reduce_only": row.get("reduceOnly") == "true",
                    "update_time": row.get("uTime"),
                }
                for row in orders if row.get("instId") == self.settings.symbol
            ],
        }

    def place_market_order(self, request: OrderRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "instId": self.settings.symbol,
            "tdMode": self.settings.margin_mode,
            "side": request.side,
            "ordType": "market",
            "sz": format_number(self._to_contracts(request.quantity)),
            "clOrdId": request.client_id,
        }
        if request.reduce_only:
            body["reduceOnly"] = "true"
        payload = self._private("POST", "/api/v5/trade/order", body=body)
        self._check_response(payload)
        return payload

    def place_stop_order(self, position: Position) -> dict[str, Any]:
        side = "sell" if position.side == "long" else "buy"
        body: dict[str, Any] = {
            "instId": self.settings.symbol,
            "tdMode": self.settings.margin_mode,
            "side": side,
            "ordType": "conditional",
            "sz": format_number(self._to_contracts(position.quantity)),
            "slTriggerPx": format_number(position.stop_price),
            "slOrdPx": "-1",
            "slTriggerPxType": "mark",
            "tpTriggerPx": format_number(position.take_profit_price),
            "tpOrdPx": "-1",
            "tpTriggerPxType": "mark",
            "reduceOnly": "true",
        }
        payload = self._private("POST", "/api/v5/trade/order-algo", body=body)
        self._check_response(payload)
        return payload

    def _to_contracts(self, quantity: float) -> float:
        if self.settings.contract_size <= 0:
            raise ValueError("OKX contract_size must be positive")
        return quantity / self.settings.contract_size

    def _private(self, method: str, path: str, *, params: dict[str, Any] | None = None, body: dict[str, Any] | None = None) -> Any:
        key, secret, passphrase = self.credentials()
        body_string = json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body else ""
        query = urlencode(
            [(name, value) for name, value in (params or {}).items() if value is not None]
        )
        request_path = f"{path}?{query}" if query else path
        if method.upper() == "GET":
            self._guard_rest(request_path)
        self._ensure_server_time()
        for attempt in range(2):
            timestamp = self._server_timestamp()
            message = f"{timestamp}{method.upper()}{request_path}{body_string}".encode("utf-8")
            signature = base64.b64encode(
                hmac.new(secret.encode(), message, hashlib.sha256).digest()
            ).decode()
            headers = {
                "Content-Type": "application/json",
                "OK-ACCESS-KEY": key,
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": passphrase,
            }
            if self.settings.environment == "demo":
                headers["x-simulated-trading"] = "1"
            try:
                payload = request_json(
                    method,
                    f"{self.base_url}{path}",
                    params=params,
                    headers=headers,
                    body=body_string or None,
                    timeout=5.0,
                    max_attempts=1,
                )
            except ApiError as error:
                if attempt == 0 and self._is_timestamp_error(error):
                    self._sync_server_time(force=True)
                    continue
                if method.upper() == "GET":
                    self._rest_failed(request_path, error)
                raise
            if attempt == 0 and self._is_timestamp_payload(payload):
                self._sync_server_time(force=True)
                continue
            try:
                self._check_response(payload)
            except (ApiError, RuntimeError) as error:
                if method.upper() == "GET":
                    self._rest_failed(request_path, error)
                raise
            return payload
        raise ApiError("OKX private request failed after server-time synchronization")

    def _server_timestamp(self) -> str:
        timestamp_ms = self._server_timestamp_ms()
        return (
            datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def _server_timestamp_ms(self) -> int:
        with self._server_time_lock:
            if self._server_time_anchor_ms > 0:
                elapsed_ms = int(
                    max(0.0, time.monotonic() - self._server_time_anchor_monotonic)
                    * 1000
                )
                return self._server_time_anchor_ms + elapsed_ms
            return self.now_ms() + self._server_time_offset_ms

    def _ensure_server_time(self) -> None:
        with self._server_time_lock:
            fresh = (
                self._server_time_anchor_ms > 0
                and time.monotonic() - self._server_time_synced_at < 900.0
            )
        if fresh:
            return
        try:
            self._sync_server_time(force=False)
        except ApiError:
            with self._server_time_lock:
                if self._server_time_anchor_ms <= 0:
                    raise

    def _sync_server_time(self, *, force: bool = True) -> int:
        with self._server_time_lock:
            if (
                not force
                and self._server_time_anchor_ms > 0
                and time.monotonic() - self._server_time_synced_at < 900.0
            ):
                return self._server_time_offset_ms
            payload = self._public("/api/v5/public/time")
            finished_at = self.now_ms()
            finished_monotonic = time.monotonic()
            if not isinstance(payload, dict) or str(payload.get("code", "0")) != "0":
                raise ApiError(f"OKX server-time response is invalid: {payload}")
            rows = payload.get("data") or []
            server_time = int(rows[0].get("ts") or 0) if rows else 0
            if server_time <= 0:
                raise ApiError("OKX server-time response has no valid timestamp")
            self._server_time_offset_ms = server_time - finished_at
            self._server_time_anchor_ms = server_time
            self._server_time_anchor_monotonic = finished_monotonic
            self._server_time_synced_at = finished_monotonic
            return self._server_time_offset_ms

    @staticmethod
    def _is_timestamp_error(error: ApiError) -> bool:
        message = str(error).lower()
        return str(error.api_code or "") == "50102" or "50102" in message or "timestamp request expired" in message

    @staticmethod
    def _is_timestamp_payload(payload: Any) -> bool:
        return isinstance(payload, dict) and str(payload.get("code") or "") == "50102"

    @staticmethod
    def _check_response(payload: Any) -> None:
        if not isinstance(payload, dict) or payload.get("code") not in (None, "0", 0):
            code = payload.get("code") if isinstance(payload, dict) else None
            limited = str(code) in {"50011", "50040"}
            raise ApiError(f"OKX API error: {payload}", api_code=code,
                           status_code=429 if limited else None,
                           retry_at=time.time() + 60.0 if limited else 0.0)
