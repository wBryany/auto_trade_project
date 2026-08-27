from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any
from urllib.parse import urlencode

from ..http_client import format_number, request_json
from ..models import Candle, OrderRequest, Position
from .base import ExchangeAdapter, ExchangeSettings, candle_limit_for


class GateAdapter(ExchangeAdapter):
    def __init__(self, settings: ExchangeSettings) -> None:
        super().__init__(settings)
        self.base_url = settings.base_url.rstrip("/")
        self.prefix = "/api/v4"
        if self.base_url.endswith(self.prefix):
            self.host = self.base_url[: -len(self.prefix)]
        else:
            self.host = self.base_url
            self.base_url = f"{self.host}{self.prefix}"

    def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
        payload = self._request(
            "GET",
            f"/futures/{self.settings.settle}/candlesticks",
            params={"contract": self.settings.symbol, "interval": interval, "limit": candle_limit_for(interval, limit)},
        )
        candles: list[Candle] = []
        for row in payload:
            if isinstance(row, dict):
                candles.append(Candle(int(row["t"]) * 1000, float(row["o"]), float(row["h"]), float(row["l"]), float(row["c"]), float(row["v"])))
            else:
                candles.append(Candle(int(row[0]) * 1000, float(row[5]), float(row[3]), float(row[4]), float(row[2]), float(row[1])))
        return sorted(candles, key=lambda candle: candle.timestamp)

    def fetch_equity(self) -> float:
        payload = self._request("GET", f"/futures/{self.settings.settle}/accounts", authenticated=True)
        return float(payload.get("total") or payload.get("available") or 0)

    def fetch_dashboard_snapshot(self) -> dict[str, Any]:
        rows = self._request(
            "GET",
            f"/futures/{self.settings.settle}/tickers",
            params={"contract": self.settings.symbol},
        )
        ticker = (rows or [{}])[0] if isinstance(rows, list) else (rows or {})
        mark_price_raw = str(ticker.get("mark_price") or "0")
        if float(mark_price_raw) <= 0:
            raise RuntimeError(f"Gate returned no mark price for {self.settings.symbol}")
        return {
            "market": {
                "symbol": self.settings.symbol,
                "mark_price": float(mark_price_raw),
                "mark_price_raw": mark_price_raw,
                "last_price": float(mark_price_raw),
                "last_price_raw": mark_price_raw,
                "last_trade_price": float(ticker.get("last") or 0),
                "last_trade_price_raw": str(ticker.get("last") or "0"),
                "price_source": "Gate USDT 永续合约标记价格",
                "price_endpoint": f"/futures/{self.settings.settle}/tickers",
                "timestamp": self.now_ms(),
            },
            "positions": [],
            "open_orders": [],
            "private_available": False,
        }

    def place_market_order(self, request: OrderRequest) -> dict[str, Any]:
        signed_size = self._signed_size(request.side, request.quantity, request.reduce_only)
        body: dict[str, Any] = {
            "contract": self.settings.symbol,
            "size": signed_size,
            "price": "0",
            "tif": "ioc",
            "reduce_only": request.reduce_only,
            "text": request.client_id or "t-btc-bot",
        }
        return self._request("POST", f"/futures/{self.settings.settle}/orders", body=body, authenticated=True)

    def place_stop_order(self, position: Position) -> dict[str, Any]:
        quantity = self._to_contracts(position.quantity)
        body = {
            "initial": {
                "contract": self.settings.symbol,
                "size": 0,
                "price": "0",
                "reduce_only": True,
                "close": True,
            },
            "trigger": {
                "strategy_type": 0,
                "price_type": 1,
                "price": format_number(position.stop_price),
                "rule": 2 if position.side == "long" else 1,
                "expiration": 0,
            },
            "order_type": "close-long-position" if position.side == "long" else "close-short-position",
            "pos_margin_mode": self.settings.margin_mode,
        }
        # Keep the conversion visible for future dual-position support; size=0 means full close here.
        _ = quantity
        return self._request("POST", f"/futures/{self.settings.settle}/price_orders", body=body, authenticated=True)

    def _to_contracts(self, quantity: float) -> int:
        if self.settings.contract_size <= 0:
            raise ValueError("Gate contract_size must be positive")
        return max(1, int(round(quantity / self.settings.contract_size)))

    def _signed_size(self, side: str, quantity: float, reduce_only: bool) -> int:
        contracts = self._to_contracts(quantity)
        if side == "buy":
            return -contracts if reduce_only else contracts
        return contracts if reduce_only else -contracts

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        authenticated: bool = False,
    ) -> Any:
        query = urlencode([(key, value) for key, value in (params or {}).items() if value is not None])
        body_string = json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body is not None else ""
        timestamp = str(int(time.time()))
        body_hash = hashlib.sha512(body_string.encode()).hexdigest()
        request_path = f"{self.prefix}{path}"
        signature_string = f"{method.upper()}\n{request_path}\n{query}\n{body_hash}\n{timestamp}"
        headers: dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
        if authenticated:
            key, secret, _ = self.credentials()
            headers.update({
                "KEY": key,
                "Timestamp": timestamp,
                "SIGN": hmac.new(secret.encode(), signature_string.encode(), hashlib.sha512).hexdigest(),
            })
        return request_json(method, f"{self.host}{request_path}", params=params, headers=headers, body=body_string or None)
