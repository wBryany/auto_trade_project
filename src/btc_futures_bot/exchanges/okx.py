from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlencode
from typing import Any

from ..http_client import ApiError, format_number, request_json
from ..models import Candle, OrderRequest, Position
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

    def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
        if interval == "30s":
            payload = request_json(
                "GET",
                f"{self.base_url}/api/v5/market/candles",
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
        try:
            payload = request_json(
                "GET",
                f"{self.base_url}/api/v5/market/candles",
                params={"instId": self.settings.symbol, "bar": bar, "limit": candle_limit_for(interval, limit)},
            )
            self._check_response(payload)
        except (ApiError, RuntimeError, TimeoutError, OSError) as error:
            if interval != "5m":
                raise
            return self._fetch_5m_from_official_1m(limit, error)
        candles = [self._candle_from_row(row) for row in payload.get("data", [])]
        return sorted(candles, key=lambda candle: candle.timestamp)

    def _fetch_5m_from_official_1m(self, limit: int, primary_error: Exception) -> list[Candle]:
        """Fall back to OKX 1m candles when its direct 5m request is unavailable.

        The engine already removes the newest forming candle, so this method
        deliberately keeps the newest (possibly partial) five-minute bucket.
        """
        LOG.warning("OKX 5m candles unavailable; aggregating official 1m candles: %s", primary_error)
        try:
            payload = request_json(
                "GET",
                f"{self.base_url}/api/v5/market/candles",
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
        payload = self._private("GET", "/api/v5/account/balance")
        self._check_response(payload)
        row = (payload.get("data") or [{}])[0]
        return float(row.get("totalEq") or row.get("adjEq") or 0)

    def fetch_dashboard_snapshot(self) -> dict[str, Any]:
        mark_payload = request_json(
            "GET",
            f"{self.base_url}/api/v5/public/mark-price",
            params={"instType": "SWAP", "instId": self.settings.symbol},
        )
        self._check_response(mark_payload)
        mark = (mark_payload.get("data") or [{}])[0]
        mark_price_raw = str(mark.get("markPx") or "0")
        if float(mark_price_raw) <= 0:
            raise RuntimeError(f"OKX returned no mark price for {self.settings.symbol}")
        snapshot: dict[str, Any] = {
            "market": {
                "symbol": self.settings.symbol,
                "mark_price": float(mark_price_raw),
                "mark_price_raw": mark_price_raw,
                "last_price": float(mark_price_raw),
                "last_price_raw": mark_price_raw,
                "price_source": "OKX 永续合约标记价格",
                "price_endpoint": "/api/v5/public/mark-price",
                "timestamp": int(mark.get("ts") or self.now_ms()),
            },
            "positions": [],
            "open_orders": [],
            "private_available": self.has_credentials(),
        }
        if not self.has_credentials():
            snapshot["private_error"] = "未配置完整的 OKX API Key、Secret 或 Passphrase"
            return snapshot
        try:
            balance_payload = self._private("GET", "/api/v5/account/balance")
            positions_payload = self._private("GET", "/api/v5/account/positions", params={"instId": self.settings.symbol})
            orders_payload = self._private("GET", "/api/v5/trade/orders-pending", params={"instType": "SWAP", "instId": self.settings.symbol})
            self._check_response(balance_payload)
            self._check_response(positions_payload)
            self._check_response(orders_payload)
            balance = (balance_payload.get("data") or [{}])[0]
            snapshot["account"] = {
                "wallet_balance": float(balance.get("totalEq") or balance.get("adjEq") or 0),
                "available_balance": float(balance.get("availEq") or 0),
                "unrealized_pnl": 0.0,
                "margin_balance": float(balance.get("totalEq") or balance.get("adjEq") or 0),
            }
            snapshot["positions"] = [
                {
                    "symbol": row.get("instId", self.settings.symbol),
                    "side": ("short" if float(row.get("pos") or 0) < 0 else "long"),
                    "quantity": abs(float(row.get("pos") or 0)),
                    "entry_price": float(row.get("avgPx") or 0),
                    "mark_price": float(row.get("markPx") or 0),
                    "unrealized_pnl": float(row.get("upl") or 0),
                    "liquidation_price": float(row.get("liqPx") or 0),
                    "leverage": float(row.get("lever") or 0),
                    "position_side": row.get("posSide", "net"),
                }
                for row in positions_payload.get("data", [])
                if abs(float(row.get("pos") or 0)) > 0
            ]
            snapshot["open_orders"] = [
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
                for row in orders_payload.get("data", [])
            ]
        except Exception as error:
            snapshot["private_available"] = False
            snapshot["private_error"] = str(error)
        return snapshot

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
                )
            except ApiError as error:
                if attempt == 0 and self._is_timestamp_error(error):
                    self._sync_server_time(force=True)
                    continue
                raise
            if attempt == 0 and self._is_timestamp_payload(payload):
                self._sync_server_time(force=True)
                continue
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
            payload = request_json(
                "GET",
                f"{self.base_url}/api/v5/public/time",
                timeout=5.0,
                max_attempts=3,
            )
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
            raise RuntimeError(f"OKX API error: {payload}")
