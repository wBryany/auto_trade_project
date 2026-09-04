from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any
from urllib.parse import urlencode

from ..binance_stream import BinanceMarketStream
from ..binance_user_stream import BinanceUserDataStream
from ..http_client import ApiError, format_number, redact_url_credentials, request_json
from ..models import Candle, OrderRequest, Position
from .base import ExchangeAdapter, ExchangeSettings, candle_limit_for


BINANCE_BASE_URLS = {
    "testnet": "https://demo-fapi.binance.com",
    "production": "https://fapi.binance.com",
}

LOG = logging.getLogger(__name__)


class BinanceAdapter(ExchangeAdapter):
    def __init__(self, settings: ExchangeSettings) -> None:
        super().__init__(settings)
        environment = settings.environment.strip().lower()
        expected_base = BINANCE_BASE_URLS.get(environment)
        if expected_base is None:
            raise ValueError("Binance environment must be testnet or production")
        self.base_url = settings.base_url.rstrip("/")
        if self.base_url != expected_base:
            raise ValueError(f"Binance {environment} endpoint must be {expected_base}")
        if not settings.symbol.isascii() or not settings.symbol.isalnum():
            raise ValueError("Binance USD-M symbol must use exchange format, for example BTCUSDT")
        self._symbol_rules: dict[str, Decimal] | None = None
        self._server_time_offset_ms = 0
        self._server_time_anchor_ms = 0
        self._server_time_anchor_monotonic = 0.0
        self._server_time_synced_at = 0.0
        self._server_time_lock = threading.RLock()
        self._market_stream = (
            BinanceMarketStream(
                settings.symbol,
                environment,
                stale_seconds=settings.websocket_stale_seconds,
            )
            if settings.websocket_enabled
            else None
        )
        self._private_stream = (
            BinanceUserDataStream(
                settings.symbol,
                environment,
                self.base_url,
                self._api_key,
                self._fetch_private_snapshot_rest,
                stale_seconds=max(75.0, settings.websocket_stale_seconds * 5),
            )
            if settings.websocket_enabled
            else None
        )
        self._rest_candle_refresh_at: dict[str, float] = {}
        self._rest_mark_price: tuple[float, float, int] | None = None
        self._rest_mark_price_at = 0.0

    def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
        requested_limit = candle_limit_for(interval, limit)
        stream = self._market_stream
        if stream is not None:
            stream.start()
            if stream.has_history(interval, requested_limit):
                cached = stream.candles(interval, requested_limit)
                if stream.healthy():
                    return cached
                refresh_at = self._rest_candle_refresh_at.get(interval, 0.0)
                if time.monotonic() < refresh_at:
                    return cached
        candles = self._fetch_candles_rest(interval, requested_limit)
        if stream is not None:
            stream.seed_candles(interval, candles)
            self._rest_candle_refresh_at[interval] = time.monotonic() + max(
                30.0,
                float(self.settings.websocket_stale_seconds),
            )
            return stream.candles(interval, requested_limit)
        return candles

    def _fetch_candles_rest(self, interval: str, limit: int) -> list[Candle]:
        payload = request_json(
            "GET",
            f"{self.base_url}/fapi/v1/klines",
            params={"symbol": self.settings.symbol, "interval": interval, "limit": limit},
        )
        return [
            Candle(
                int(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                quote_volume=float(row[7]) if len(row) > 7 else None,
            )
            for row in payload
        ]

    def fetch_equity(self) -> float:
        payload = (
            self._private_snapshot(wait_seconds=10.0)["account"]
            if self._private_stream is not None
            else self._signed("GET", "/fapi/v2/account")
        )
        return float(payload.get("totalWalletBalance") or payload.get("totalMarginBalance") or 0)

    def _api_key(self) -> str:
        key, _secret, _passphrase = self.credentials()
        return key

    def _fetch_private_snapshot_rest(self) -> dict[str, Any]:
        """Bootstrap the delta-only user stream once per connection."""

        account = self._signed("GET", "/fapi/v2/account")
        # The account response already contains the complete USD-M position
        # array.  Reusing it removes the high-frequency /positionRisk endpoint
        # entirely, including during private-stream reconnects.
        positions = list(account.get("positions") or [])
        orders = self._signed(
            "GET",
            "/fapi/v1/openOrders",
            {"symbol": self.settings.symbol},
        )
        # Existing conditional orders are state, not deltas.  Refuse to mark
        # the stream ready until this snapshot succeeds; treating a failed
        # query as an empty list could hide a live exchange-side stop.
        algo_orders = self._signed(
            "GET",
            "/fapi/v1/openAlgoOrders",
            {"symbol": self.settings.symbol, "algoType": "CONDITIONAL"},
        )
        return {
            "account": account,
            "positions": positions,
            "orders": orders,
            "algo_orders": algo_orders,
        }

    def _private_snapshot(self, *, wait_seconds: float = 0.0) -> dict[str, Any]:
        stream = self._private_stream
        if stream is None:
            return self._fetch_private_snapshot_rest()
        return stream.snapshot(wait_seconds=wait_seconds)

    def fetch_mark_price(self) -> float:
        mark_price_raw, _index_price_raw, _timestamp, _source = self._market_prices()
        return float(mark_price_raw)

    def _market_prices(self) -> tuple[str, str, int, str]:
        stream = self._market_stream
        if stream is not None:
            stream.start()
            streamed = stream.mark_price()
            if streamed is not None:
                mark_price, index_price, timestamp = streamed
                return str(mark_price), str(index_price), timestamp, "websocket"
        if self._rest_mark_price is not None and time.monotonic() - self._rest_mark_price_at < 5.0:
            mark_price, index_price, timestamp = self._rest_mark_price
            return str(mark_price), str(index_price), timestamp, "rest_cache"
        premium_index = request_json(
            "GET",
            f"{self.base_url}/fapi/v1/premiumIndex",
            params={"symbol": self.settings.symbol},
        )
        mark_price = Decimal(str(premium_index.get("markPrice") or "0"))
        if mark_price <= 0:
            raise RuntimeError(f"Binance returned no mark price for {self.settings.symbol}")
        index_price = Decimal(str(premium_index.get("indexPrice") or "0"))
        timestamp = int(premium_index.get("time") or self.now_ms())
        self._rest_mark_price = (float(mark_price), float(index_price), timestamp)
        self._rest_mark_price_at = time.monotonic()
        return str(mark_price), str(index_price), timestamp, "rest"

    def _dashboard_market_view(
        self,
        mark_price_raw: str,
        index_price_raw: str,
        market_timestamp: int,
        price_source: str,
    ) -> dict[str, Any]:
        last_price_raw = mark_price_raw
        last_price_timestamp = market_timestamp
        last_price_source = (
            "Binance USDⓈ-M WebSocket 标记价格"
            if price_source == "websocket"
            else "Binance USDⓈ-M REST 标记价格"
        )
        last_price_endpoint = (
            "@markPrice@1s" if price_source == "websocket" else "/fapi/v1/premiumIndex"
        )
        stream = self._market_stream
        latest_trade = getattr(stream, "latest_trade", None) if stream is not None else None
        streamed_trade = latest_trade() if callable(latest_trade) else None
        if streamed_trade is not None:
            trade_price, trade_timestamp = streamed_trade
            last_price_raw = str(trade_price)
            last_price_timestamp = trade_timestamp
            last_price_source = "Binance USDⓈ-M WebSocket 最新成交价"
            last_price_endpoint = "@aggTrade"
        market = {
            "symbol": self.settings.symbol,
            "mark_price": float(mark_price_raw),
            "mark_price_raw": mark_price_raw,
            "last_price": float(last_price_raw),
            "last_price_raw": last_price_raw,
            "last_price_timestamp": last_price_timestamp,
            "last_price_source": last_price_source,
            "last_price_endpoint": last_price_endpoint,
            "index_price": float(index_price_raw),
            "index_price_raw": index_price_raw,
            "price_source": (
                "Binance USDⓈ-M WebSocket 标记价格"
                if price_source == "websocket"
                else "Binance USDⓈ-M REST 标记价格"
            ),
            "price_endpoint": (
                "@markPrice@1s"
                if price_source == "websocket"
                else "/fapi/v1/premiumIndex"
            ),
            "timestamp": market_timestamp,
        }
        if stream is not None:
            market["stream"] = stream.status()
        return market

    def fetch_live_market_snapshot(self) -> dict[str, Any] | None:
        """Return only fresh in-memory WebSocket prices without making REST calls."""

        stream = self._market_stream
        if stream is None:
            return None
        stream.start()
        streamed = stream.mark_price()
        if streamed is None:
            return None
        mark_price, index_price, timestamp = streamed
        return self._dashboard_market_view(
            str(mark_price),
            str(index_price),
            timestamp,
            "websocket",
        )

    def fetch_dashboard_snapshot(
        self,
        *,
        private_wait_seconds: float = 10.0,
        include_order_limits: bool = True,
    ) -> dict[str, Any]:
        mark_price_raw, index_price_raw, market_timestamp, price_source = self._market_prices()
        if Decimal(mark_price_raw) <= 0:
            raise RuntimeError(f"Binance returned no mark price for {self.settings.symbol}")
        snapshot: dict[str, Any] = {
            "market": self._dashboard_market_view(
                mark_price_raw,
                index_price_raw,
                market_timestamp,
                price_source,
            ),
            "positions": [],
            "open_orders": [],
            "private_available": False,
        }
        if self._private_stream is not None:
            snapshot["private_stream"] = self._private_stream.status()
        if include_order_limits:
            try:
                snapshot["order_limits"] = self.order_limits(mark_price_raw)
            except Exception as error:
                snapshot["order_limits"] = {"error": str(error)}
        if not self.has_credentials():
            snapshot["private_error"] = "未配置当前环境的 API Key；只显示公开行情"
            return snapshot

        try:
            private = self._private_snapshot(wait_seconds=max(0.0, private_wait_seconds))
            account = private["account"]
            position_rows = private["positions"]
            order_rows = private["orders"]
            algo_rows = private["algo_orders"]
            wallet_balance_raw = str(account.get("totalWalletBalance") or "0")
            available_balance_raw = str(account.get("availableBalance") or "0")
            active_rows = [
                row for row in position_rows if abs(float(row.get("positionAmt") or 0)) > 0
            ]
            live_unrealized = sum(
                (float(mark_price_raw) - float(row.get("entryPrice") or 0))
                * abs(float(row.get("positionAmt") or 0))
                * (1 if float(row.get("positionAmt") or 0) > 0 else -1)
                for row in active_rows
            )
            unrealized_pnl_raw = self._decimal_string(Decimal(str(live_unrealized)))
            margin_balance_raw = self._decimal_string(
                Decimal(wallet_balance_raw) + Decimal(unrealized_pnl_raw)
            )
            snapshot["account"] = {
                "wallet_balance": float(wallet_balance_raw),
                "wallet_balance_raw": wallet_balance_raw,
                "available_balance": float(available_balance_raw),
                "available_balance_raw": available_balance_raw,
                "unrealized_pnl": float(unrealized_pnl_raw),
                "unrealized_pnl_raw": unrealized_pnl_raw,
                "margin_balance": float(margin_balance_raw),
                "margin_balance_raw": margin_balance_raw,
            }
            snapshot["positions"] = [
                {
                    "symbol": row.get("symbol", self.settings.symbol),
                    "side": "long" if float(row.get("positionAmt") or 0) > 0 else "short",
                    "quantity": abs(float(row.get("positionAmt") or 0)),
                    "entry_price": float(row.get("entryPrice") or 0),
                    "mark_price": float(mark_price_raw),
                    "unrealized_pnl": (
                        (float(mark_price_raw) - float(row.get("entryPrice") or 0))
                        * abs(float(row.get("positionAmt") or 0))
                        * (1 if float(row.get("positionAmt") or 0) > 0 else -1)
                    ),
                    "liquidation_price": float(row.get("liquidationPrice") or 0),
                    "leverage": float(row.get("leverage") or 0),
                    "position_side": row.get("positionSide", "BOTH"),
                }
                for row in position_rows
                if abs(float(row.get("positionAmt") or 0)) > 0
            ]
            snapshot["open_orders"] = (
                [self._regular_order_view(row) for row in order_rows]
                + [self._algo_order_view(row) for row in algo_rows]
            )
            limits = snapshot.get("order_limits") or {}
            if not limits.get("error"):
                position_row = next(
                    (
                        row
                        for row in position_rows
                        if str(row.get("symbol") or "") == self.settings.symbol
                        and str(row.get("positionSide") or "BOTH") == "BOTH"
                    ),
                    position_rows[0] if position_rows else {},
                )
                account_position = next(
                    (
                        row
                        for row in account.get("positions", [])
                        if str(row.get("symbol") or "") == self.settings.symbol
                        and str(row.get("positionSide") or "BOTH") == "BOTH"
                    ),
                    {},
                )
                leverage = Decimal(str(position_row.get("leverage") or account_position.get("leverage") or "0"))
                max_position_notional = Decimal(
                    str(
                        position_row.get("maxNotionalValue")
                        or account_position.get("maxNotional")
                        or "0"
                    )
                )
                reference_price = Decimal(mark_price_raw)
                available_balance = Decimal(available_balance_raw)
                current_position_notional = abs(
                    Decimal(str(position_row.get("positionAmt") or "0"))
                    * Decimal(str(position_row.get("markPrice") or mark_price_raw))
                )
                pending_entry_notional = Decimal("0")
                for row in order_rows:
                    if self._truthy(row.get("reduceOnly")) or self._truthy(row.get("closePosition")):
                        continue
                    order_quantity = Decimal(str(row.get("origQty") or row.get("quantity") or "0"))
                    order_price = Decimal(str(row.get("price") or row.get("stopPrice") or mark_price_raw))
                    if order_price <= 0:
                        order_price = reference_price
                    pending_entry_notional += abs(order_quantity * order_price)
                balance_cap = available_balance * leverage if leverage > 0 else Decimal("0")
                remaining_position_cap = max(
                    Decimal("0"),
                    max_position_notional - current_position_notional - pending_entry_notional,
                )
                venue_quantity_cap = Decimal(str(limits.get("max_market_quantity_raw") or "0"))
                venue_notional_cap = venue_quantity_cap * reference_price
                caps = [item for item in (balance_cap, remaining_position_cap, venue_notional_cap) if item > 0]
                raw_capacity = min(caps) if caps else Decimal("0")
                quantity_step = Decimal(str(limits.get("quantity_step_raw") or "0"))
                estimated_quantity = (
                    self._round_step(raw_capacity / reference_price, quantity_step, ROUND_DOWN)
                    if raw_capacity > 0 and reference_price > 0 and quantity_step > 0
                    else Decimal("0")
                )
                estimated_notional = estimated_quantity * reference_price
                limits.update(
                    {
                        "current_leverage_raw": self._decimal_string(leverage),
                        "max_position_notional_raw": self._decimal_string(max_position_notional),
                        "balance_leverage_cap_raw": self._decimal_string(balance_cap),
                        "estimated_max_open_quantity_raw": self._decimal_string(estimated_quantity),
                        "estimated_max_open_notional_raw": self._decimal_string(estimated_notional),
                        "maximum_is_estimate": True,
                    }
                )
            snapshot["private_available"] = True
            snapshot["private_source"] = (
                "websocket" if self._private_stream is not None else "rest"
            )
            if self._private_stream is not None:
                snapshot["private_stream"] = self._private_stream.status()
        except Exception as error:  # Keep public market data visible if private auth fails.
            snapshot["private_available"] = False
            snapshot["private_error"] = str(error)
            snapshot["private_transient"] = self._private_error_is_transient(error)
            snapshot["private_retryable"] = self._private_error_is_retryable(error)
        return snapshot

    def fetch_dashboard_snapshot_nonblocking(self) -> dict[str, Any]:
        """Refresh dashboard state without ever waiting for private-stream startup."""

        return self.fetch_dashboard_snapshot(private_wait_seconds=0.0)

    def fetch_live_dashboard_snapshot(self) -> dict[str, Any] | None:
        """Return WebSocket-backed market/private state without waiting or REST."""

        market_stream = self._market_stream
        private_stream = self._private_stream
        if (
            market_stream is None
            or private_stream is None
            or market_stream.mark_price() is None
            or not private_stream.healthy()
        ):
            return None
        return self.fetch_dashboard_snapshot(
            private_wait_seconds=0.0,
            include_order_limits=False,
        )

    def close(self) -> None:
        if self._market_stream is not None:
            self._market_stream.close()
        if self._private_stream is not None:
            self._private_stream.close()

    def symbol_rules(self) -> dict[str, Decimal]:
        if self._symbol_rules is not None:
            return self._symbol_rules
        payload = request_json(
            "GET",
            f"{self.base_url}/fapi/v1/exchangeInfo",
            params={"symbol": self.settings.symbol},
        )
        rows = [row for row in payload.get("symbols", []) if row.get("symbol") == self.settings.symbol]
        if len(rows) != 1:
            raise RuntimeError(f"Binance symbol is unavailable: {self.settings.symbol}")
        symbol = rows[0]
        if symbol.get("status") != "TRADING" or symbol.get("contractType") != "PERPETUAL":
            raise RuntimeError(f"Binance symbol is not a tradable perpetual contract: {self.settings.symbol}")
        filters = {item.get("filterType"): item for item in symbol.get("filters", [])}
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
        price_filter = filters.get("PRICE_FILTER") or {}
        notional_filter = filters.get("MIN_NOTIONAL") or {}
        rules = {
            "quantity_step": Decimal(str(lot.get("stepSize") or "0")),
            "min_quantity": Decimal(str(lot.get("minQty") or "0")),
            "max_quantity": Decimal(str(lot.get("maxQty") or "0")),
            "price_tick": Decimal(str(price_filter.get("tickSize") or "0")),
            "min_notional": Decimal(str(notional_filter.get("notional") or "0")),
        }
        if rules["quantity_step"] <= 0 or rules["price_tick"] <= 0:
            raise RuntimeError(f"Binance exchangeInfo has incomplete filters for {self.settings.symbol}")
        self._symbol_rules = rules
        return rules

    def order_limits(self, reference_price: str | float) -> dict[str, Any]:
        price = Decimal(str(reference_price))
        if price <= 0:
            raise ValueError("reference_price must be positive")
        rules = self.symbol_rules()
        quantity_step = rules["quantity_step"]
        min_quantity = rules["min_quantity"]
        min_notional = rules["min_notional"]
        quantity_for_notional = (
            self._round_step(min_notional / price, quantity_step, ROUND_UP)
            if min_notional > 0
            else Decimal("0")
        )
        effective_min_quantity = max(min_quantity, quantity_for_notional)
        effective_min_notional = effective_min_quantity * price
        max_market_quantity = rules["max_quantity"]
        return {
            "quantity_step_raw": self._decimal_string(quantity_step),
            "min_quantity_raw": self._decimal_string(min_quantity),
            "max_market_quantity_raw": self._decimal_string(max_market_quantity),
            "min_notional_filter_raw": self._decimal_string(min_notional),
            "effective_min_quantity_raw": self._decimal_string(effective_min_quantity),
            "effective_min_notional_raw": self._decimal_string(effective_min_notional),
            "max_market_notional_raw": self._decimal_string(max_market_quantity * price),
        }

    def normalize_order_quantity(self, quantity: float, reference_price: float) -> float:
        if quantity <= 0 or reference_price <= 0:
            raise ValueError("quantity and reference_price must be positive")
        rules = self.symbol_rules()
        selected = self._round_step(Decimal(str(quantity)), rules["quantity_step"], ROUND_DOWN)
        if selected < rules["min_quantity"]:
            raise ValueError(f"Binance quantity {selected} is below minimum {rules['min_quantity']}")
        if rules["max_quantity"] > 0 and selected > rules["max_quantity"]:
            raise ValueError(f"Binance quantity {selected} exceeds maximum {rules['max_quantity']}")
        if selected * Decimal(str(reference_price)) < rules["min_notional"]:
            raise ValueError(f"Binance order notional is below minimum {rules['min_notional']} USDT")
        return float(selected)

    def normalize_trigger_price(self, price: float, side: str) -> float:
        return float(self._normalize_trigger_price_decimal(price, side))

    def _normalize_trigger_price_decimal(self, price: float, side: str) -> Decimal:
        if price <= 0 or side.lower() not in {"buy", "sell"}:
            raise ValueError("invalid trigger price or side")
        rounding = ROUND_DOWN if side.lower() == "sell" else ROUND_UP
        return self._round_step(Decimal(str(price)), self.symbol_rules()["price_tick"], rounding)

    def prepare_live(
        self,
        *,
        max_leverage: float,
        managed_position: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.has_credentials():
            raise RuntimeError("Binance live mode requires credentials for the selected environment")
        self.symbol_rules()
        private = self._private_snapshot(wait_seconds=15.0)
        positions = private["positions"]
        regular_orders = private["orders"]
        algo_orders = private["algo_orders"]
        active_positions = [row for row in positions if abs(float(row.get("positionAmt") or 0)) > 0]
        if active_positions or regular_orders or algo_orders:
            if managed_position is not None:
                return self._resume_protected_live_position(
                    managed_position,
                    active_positions,
                    regular_orders,
                    algo_orders,
                    max_leverage=max_leverage,
                )
            raise RuntimeError(
                "Binance live preflight refused: existing position/order found; resolve it on Binance before starting"
            )

        position_mode = self._signed("GET", "/fapi/v1/positionSide/dual")
        if bool(position_mode.get("dualSidePosition")):
            self._signed("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "false"})
        try:
            self._signed(
                "POST",
                "/fapi/v1/marginType",
                {"symbol": self.settings.symbol, "marginType": self.settings.margin_mode.upper()},
            )
        except ApiError as error:
            if '"code":-4046' not in str(error).replace(" ", ""):
                raise
        leverage = max(1, min(125, int(max_leverage)))
        leverage_result = self._signed(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": self.settings.symbol, "leverage": leverage},
        )
        if int(leverage_result.get("leverage") or 0) != leverage:
            raise RuntimeError("Binance did not confirm the requested leverage")
        return {
            "exchange": "binance",
            "environment": self.settings.environment,
            "symbol": self.settings.symbol,
            "margin_mode": self.settings.margin_mode,
            "position_mode": "one-way",
            "leverage": leverage,
            "flat": True,
            "private_stream": (
                self._private_stream.status() if self._private_stream is not None else None
            ),
        }

    def _resume_protected_live_position(
        self,
        managed_state: dict[str, Any],
        active_positions: list[dict[str, Any]],
        regular_orders: list[dict[str, Any]],
        algo_orders: list[dict[str, Any]],
        *,
        max_leverage: float,
    ) -> dict[str, Any]:
        """Resume only an exactly matching bot position with one valid hard stop."""

        expected = managed_state.get("position") or {}
        if len(active_positions) != 1 or regular_orders or len(algo_orders) != 1:
            raise RuntimeError(
                "Binance live resume refused: saved position requires exactly one position and one hard stop"
            )
        remote = active_positions[0]
        if str(remote.get("positionSide") or "BOTH").upper() != "BOTH":
            raise RuntimeError("Binance live resume refused: position is not in one-way mode")
        amount = float(remote.get("positionAmt") or 0)
        remote_side = "long" if amount > 0 else "short"
        remote_quantity = abs(amount)
        remote_entry = float(remote.get("entryPrice") or 0)
        remote_leverage = float(remote.get("leverage") or 0)
        expected_side = str(expected.get("side") or "")
        expected_quantity = float(expected.get("quantity") or 0)
        expected_entry = float(expected.get("entry_price") or 0)
        if remote_side != expected_side:
            raise RuntimeError("Binance live resume refused: saved position side does not match exchange")
        if abs(remote_quantity - expected_quantity) > max(1e-12, expected_quantity * 0.001):
            raise RuntimeError("Binance live resume refused: saved position quantity does not match exchange")
        if abs(remote_entry - expected_entry) > max(1e-8, expected_entry * 0.00001):
            raise RuntimeError("Binance live resume refused: saved entry price does not match exchange")
        if remote_leverage <= 0 or remote_leverage > max(1.0, float(max_leverage)):
            raise RuntimeError("Binance live resume refused: exchange leverage exceeds configured maximum")
        if (
            self.settings.margin_mode.lower() == "isolated"
            and "isolated" in remote
            and not self._truthy(remote.get("isolated"))
        ):
            raise RuntimeError("Binance live resume refused: exchange position is not isolated")

        stop_client_id = str(expected.get("stop_client_id") or "")
        stop_order_id = str(expected.get("stop_order_id") or "")
        stop_price = float(expected.get("initial_stop_price") or expected.get("stop_price") or 0)
        if not stop_client_id.startswith("btcbot-stop-"):
            raise RuntimeError("Binance live resume refused: saved hard stop is not bot-owned")
        if stop_price <= 0 or (
            remote_side == "long" and stop_price >= remote_entry
        ) or (
            remote_side == "short" and stop_price <= remote_entry
        ):
            raise RuntimeError("Binance live resume refused: saved hard stop is not protective")
        stop = algo_orders[0]
        if str(stop.get("algoId") or "") != stop_order_id:
            raise RuntimeError("Binance live resume refused: saved hard-stop id does not match exchange")
        self._validate_open_algo_order(
            stop,
            client_id=stop_client_id,
            order_type="STOP_MARKET",
            exit_side="SELL" if remote_side == "long" else "BUY",
            trigger_price=stop_price,
        )
        return {
            "exchange": "binance",
            "environment": self.settings.environment,
            "symbol": self.settings.symbol,
            "margin_mode": self.settings.margin_mode,
            "position_mode": "one-way",
            "leverage": remote_leverage,
            "flat": False,
            "resumed": True,
            "stop_order_id": stop_order_id,
            "stop_client_id": stop_client_id,
            "private_stream": (
                self._private_stream.status() if self._private_stream is not None else None
            ),
        }

    def place_market_order(self, request: OrderRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": self.settings.symbol,
            "side": request.side.upper(),
            "type": "MARKET",
            "quantity": format_number(self._to_contracts(request.quantity)),
            "newOrderRespType": "RESULT",
        }
        if request.client_id:
            params["newClientOrderId"] = request.client_id
        if request.reduce_only:
            params["reduceOnly"] = "true"
        try:
            result = self._signed("POST", "/fapi/v1/order", params)
            if self._private_stream is not None:
                self._private_stream.record_order_response(result)
            return result
        except ApiError as error:
            message = str(error)
            ambiguous = "network error" in message or any(f"HTTP {code}" in message for code in range(500, 600))
            if not request.client_id or not ambiguous:
                raise
            try:
                if self._private_stream is not None:
                    streamed = self._private_stream.wait_for_order(request.client_id, timeout=2.0)
                    if streamed is not None:
                        return streamed
                return self._signed(
                    "GET",
                    "/fapi/v1/order",
                    {"symbol": self.settings.symbol, "origClientOrderId": request.client_id},
                )
            except Exception:
                raise error

    def market_fill(self, payload: dict[str, Any], *, fallback_price: float) -> tuple[float, float]:
        details = dict(payload)

        def parsed_fill(row: dict[str, Any]) -> tuple[str, float, float]:
            status = str(row.get("status") or "").upper()
            quantity = float(row.get("executedQty") or row.get("cumQty") or 0)
            price = float(row.get("avgPrice") or 0)
            if price <= 0 and quantity > 0:
                cumulative_quote = float(row.get("cumQuote") or 0)
                if cumulative_quote > 0:
                    price = cumulative_quote / quantity
            return status, quantity, price

        status, quantity, price = parsed_fill(details)
        if status == "FILLED" and quantity > 0 and price > 0:
            return quantity, price

        # Binance production can acknowledge a MARKET order as FILLED before
        # returning its executed quantity and average price in the POST body.
        # The regular-order lookup is briefly eventually consistent and can
        # return -2013 even though the MARKET order has already filled. Poll by
        # both exchange and client ids, then consult order/trade history. Never
        # resubmit the order while its outcome is ambiguous.
        order_id = details.get("orderId")
        client_id = str(details.get("clientOrderId") or "")
        expected_quantity = float(details.get("origQty") or details.get("executedQty") or 0)
        queries: list[dict[str, Any]] = []
        if order_id:
            queries.append({"symbol": self.settings.symbol, "orderId": order_id})
        if client_id:
            queries.append({"symbol": self.settings.symbol, "origClientOrderId": client_id})

        def order_not_visible(error: Exception) -> bool:
            compact = str(error).lower().replace(" ", "")
            return '"code":-2013' in compact or "orderdoesnotexist" in compact

        def matches_order(row: dict[str, Any]) -> bool:
            if order_id and str(row.get("orderId") or "") == str(order_id):
                return True
            return bool(client_id and str(row.get("clientOrderId") or "") == client_id)

        for attempt in range(6):
            for query in queries:
                try:
                    candidate = self._signed("GET", "/fapi/v1/order", query)
                except ApiError as error:
                    if order_not_visible(error):
                        continue
                    raise
                status, quantity, price = parsed_fill(candidate)
                if status == "FILLED" and quantity > 0 and price > 0:
                    return quantity, price

            # The history endpoint can expose the order before the direct
            # lookup does. It is read-only and keyed against deterministic ids.
            if attempt >= 1 and queries:
                rows = self._signed(
                    "GET",
                    "/fapi/v1/allOrders",
                    {"symbol": self.settings.symbol, "limit": 100},
                )
                if not isinstance(rows, list):
                    raise RuntimeError("Binance all-orders response is not a list")
                matches = [row for row in rows if matches_order(row)]
                if len(matches) > 1:
                    raise RuntimeError("Binance returned duplicate matches for a market order")
                if matches:
                    status, quantity, price = parsed_fill(matches[0])
                    if status == "FILLED" and quantity > 0 and price > 0:
                        return quantity, price
            if attempt < 5:
                time.sleep(min(0.25 * (2**attempt), 1.0))

        # A filled trade is stronger evidence than an eventually-consistent
        # order lookup. Require the complete expected quantity when available.
        if order_id:
            rows = self._signed(
                "GET",
                "/fapi/v1/userTrades",
                {"symbol": self.settings.symbol, "limit": 100},
            )
            if not isinstance(rows, list):
                raise RuntimeError("Binance user-trades response is not a list")
            fills = [row for row in rows if str(row.get("orderId") or "") == str(order_id)]
            filled_quantity = sum(float(row.get("qty") or 0) for row in fills)
            filled_quote = sum(
                float(row.get("quoteQty") or 0)
                or float(row.get("qty") or 0) * float(row.get("price") or 0)
                for row in fills
            )
            complete = expected_quantity <= 0 or filled_quantity + 1e-12 >= expected_quantity
            if complete and filled_quantity > 0 and filled_quote > 0:
                return filled_quantity, filled_quote / filled_quantity

        # Last-resort entry recovery: live preflight was flat before this
        # request, so an exact side/quantity match proves the submitted entry
        # created the venue position. This lets the engine attach protection
        # immediately instead of abandoning a real naked position.
        is_entry = not self._truthy(details.get("reduceOnly"))
        expected_side = "long" if str(details.get("side") or "").upper() == "BUY" else "short"
        if is_entry and expected_quantity > 0 and str(details.get("side") or "").upper() in {"BUY", "SELL"}:
            remote = self.fetch_live_position()
            if (
                remote is not None
                and remote.get("side") == expected_side
                and abs(float(remote.get("quantity") or 0) - expected_quantity)
                <= max(1e-12, expected_quantity * 0.001)
                and float(remote.get("entry_price") or 0) > 0
            ):
                LOG.warning(
                    "Binance market fill recovered from exact live-position match order_id=%s client_id=%s",
                    order_id,
                    client_id,
                )
                return expected_quantity, float(remote["entry_price"])
        if status != "FILLED" or quantity <= 0 or price <= 0:
            raise RuntimeError(f"Binance market order is not confirmed FILLED (status={status or 'unknown'})")
        return quantity, price

    def place_stop_order(self, position: Position) -> dict[str, Any]:
        return self._place_algo_order(
            position,
            order_type="STOP_MARKET",
            trigger_price=position.stop_price,
            client_id=position.stop_client_id,
        )

    def place_protection_orders(
        self,
        position: Position,
        *,
        stop_client_id: str = "",
        take_profit_client_id: str = "",
    ) -> dict[str, Any]:
        # Keep one outage-safe downside order at the venue. Profit-taking is
        # intentionally managed by the engine from live mark price, trend
        # invalidation and local break-even/trailing rules.
        stop = self._place_algo_order(
            position,
            order_type="STOP_MARKET",
            trigger_price=position.stop_price,
            client_id=stop_client_id,
        )
        return {
            "stop": stop,
            "take_profit": None,
            "confirmed": True,
            "mode": "stop_only_dynamic_profit_exit",
        }

    def cancel_protection_orders(self, position: Position) -> dict[str, Any]:
        cancelled: list[str] = []
        errors: list[str] = []
        for client_id in (position.stop_client_id, position.take_profit_client_id):
            if not client_id:
                continue
            try:
                self._signed(
                    "DELETE",
                    "/fapi/v1/algoOrder",
                    {"symbol": self.settings.symbol, "clientAlgoId": client_id},
                )
                cancelled.append(client_id)
            except ApiError as error:
                compact = str(error).replace(" ", "")
                if '"code":-2011' not in compact and '"code":-2013' not in compact:
                    errors.append(str(error))
        return {"cancelled": cancelled, "errors": errors}

    def fetch_live_position(self) -> dict[str, Any] | None:
        rows = self._private_snapshot(wait_seconds=3.0)["positions"]
        active = [row for row in rows if abs(float(row.get("positionAmt") or 0)) > 0]
        if not active:
            return None
        if len(active) != 1 or str(active[0].get("positionSide", "BOTH")) != "BOTH":
            raise RuntimeError("Binance live reconciliation requires one-way mode with one active position")
        row = active[0]
        amount = float(row.get("positionAmt") or 0)
        return {
            "side": "long" if amount > 0 else "short",
            "quantity": abs(amount),
            "entry_price": float(row.get("entryPrice") or 0),
            "mark_price": self.fetch_mark_price(),
        }

    def fetch_latest_close_fill(
        self,
        position: Position,
        *,
        after_ms: int = 0,
    ) -> dict[str, Any] | None:
        """Aggregate recent Binance fills that closed a one-way position."""
        params: dict[str, Any] = {"symbol": self.settings.symbol, "limit": 1000}
        if after_ms > 0:
            # Binance restricts the start/end window for this endpoint. The
            # six-day floor leaves room for clock differences and stale local
            # reconciliation state without exceeding the seven-day window.
            params["startTime"] = max(int(after_ms) - 60_000, self.now_ms() - 6 * 86_400_000)
        rows = self._signed("GET", "/fapi/v1/userTrades", params)
        if not isinstance(rows, list):
            raise RuntimeError("Binance user-trades response is not a list")

        close_side = "SELL" if position.side == "long" else "BUY"
        candidates = [
            row
            for row in rows
            if str(row.get("side") or "").upper() == close_side
            and int(row.get("time") or 0) >= max(0, int(after_ms) - 60_000)
        ]
        if not candidates:
            return None

        selected: list[dict[str, Any]] = []
        selected_quantity = 0.0
        for row in sorted(candidates, key=lambda item: int(item.get("time") or 0), reverse=True):
            quantity = float(row.get("qty") or 0)
            if quantity <= 0:
                continue
            selected.append(row)
            selected_quantity += quantity
            if selected_quantity + 1e-12 >= position.quantity * 0.999:
                break
        if not selected:
            return None

        selected.reverse()
        quantity = sum(float(row.get("qty") or 0) for row in selected)
        quote_quantity = sum(
            float(row.get("quoteQty") or 0)
            or float(row.get("qty") or 0) * float(row.get("price") or 0)
            for row in selected
        )
        if quantity <= 0 or quote_quantity <= 0:
            return None
        commission_assets = sorted(
            {str(row.get("commissionAsset") or "").strip() for row in selected if row.get("commissionAsset")}
        )
        order_ids = list(dict.fromkeys(str(row.get("orderId") or "") for row in selected))
        return {
            "quantity": quantity,
            "price": quote_quantity / quantity,
            "realized_pnl": sum(float(row.get("realizedPnl") or 0) for row in selected),
            "commission": sum(float(row.get("commission") or 0) for row in selected),
            "commission_assets": commission_assets,
            "timestamp": max(int(row.get("time") or 0) for row in selected),
            "order_ids": [order_id for order_id in order_ids if order_id],
            "source": "Binance /fapi/v1/userTrades",
        }

    def fetch_protection_status(self, position: Position) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, client_id in (
            ("stop", position.stop_client_id),
            ("take_profit", position.take_profit_client_id),
        ):
            if not client_id:
                continue
            if self._private_stream is not None:
                streamed = self._private_stream.wait_for_algo_order(client_id, timeout=0.25)
                if streamed is not None:
                    result[name] = streamed
                    continue
            try:
                result[name] = self._signed(
                    "GET",
                    "/fapi/v1/algoOrder",
                    {"symbol": self.settings.symbol, "clientAlgoId": client_id},
                )
            except ApiError as error:
                result[name] = {"error": str(error)}
        return result

    def _place_algo_order(
        self,
        position: Position,
        *,
        order_type: str,
        trigger_price: float,
        client_id: str,
    ) -> dict[str, Any]:
        if not client_id:
            raise ValueError("Binance protection orders require a client id for safe reconciliation")
        exit_side = "SELL" if position.side == "long" else "BUY"
        normalized_trigger_decimal = self._normalize_trigger_price_decimal(trigger_price, exit_side)
        normalized_trigger = float(normalized_trigger_decimal)
        params: dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": self.settings.symbol,
            "side": exit_side,
            "type": order_type,
            "positionSide": "BOTH",
            # Keep the exchange tick-size Decimal through serialization. A
            # float round-trip can turn 78157.6 into 78157.600000000006 and
            # Binance rejects that value with -1111 over-precision.
            "triggerPrice": self._decimal_string(normalized_trigger_decimal),
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "newOrderRespType": "RESULT",
            "clientAlgoId": client_id,
        }
        submitted: dict[str, Any] = {}
        try:
            submitted = self._signed("POST", "/fapi/v1/algoOrder", params)
        except ApiError as error:
            if not self._is_ambiguous_order_error(error):
                raise
            # A lost HTTP response does not prove that Binance rejected the
            # request. Reconcile by the deterministic client id before any
            # retry can create a duplicate conditional order.
            try:
                return self._confirm_open_algo_order(
                    client_id=client_id,
                    order_type=order_type,
                    exit_side=exit_side,
                    trigger_price=normalized_trigger,
                )
            except Exception as confirmation_error:
                raise RuntimeError(
                    f"Binance {order_type} submission outcome is unknown and could not be reconciled: "
                    f"{confirmation_error}"
                ) from error

        confirmed = self._confirm_open_algo_order(
            client_id=client_id,
            order_type=order_type,
            exit_side=exit_side,
            trigger_price=normalized_trigger,
        )
        submitted_id = str(submitted.get("algoId") or "")
        confirmed_id = str(confirmed.get("algoId") or "")
        if submitted_id and confirmed_id and submitted_id != confirmed_id:
            raise RuntimeError(f"Binance {order_type} confirmation returned a different algo id")
        return confirmed

    def _confirm_open_algo_order(
        self,
        *,
        client_id: str,
        order_type: str,
        exit_side: str,
        trigger_price: float,
    ) -> dict[str, Any]:
        if self._private_stream is not None:
            streamed = self._private_stream.wait_for_algo_order(client_id, timeout=2.0)
            if streamed is not None:
                return self._validate_open_algo_order(
                    streamed,
                    client_id=client_id,
                    order_type=order_type,
                    exit_side=exit_side,
                    trigger_price=trigger_price,
                )
        # The Algo service can be very briefly eventually consistent after a
        # successful POST. GET requests already have transport retries; these
        # short polls only cover the visibility delay.
        for attempt in range(3):
            rows = self._signed(
                "GET",
                "/fapi/v1/openAlgoOrders",
                {"symbol": self.settings.symbol, "algoType": "CONDITIONAL"},
            )
            if not isinstance(rows, list):
                raise RuntimeError("Binance open algo orders response is not a list")
            matches = [row for row in rows if str(row.get("clientAlgoId") or "") == client_id]
            if len(matches) > 1:
                raise RuntimeError(f"Binance returned duplicate open algo orders for {client_id}")
            if matches:
                return self._validate_open_algo_order(
                    matches[0],
                    client_id=client_id,
                    order_type=order_type,
                    exit_side=exit_side,
                    trigger_price=trigger_price,
                )
            if attempt < 2:
                time.sleep(0.1 * (attempt + 1))
        raise RuntimeError(f"Binance did not confirm {order_type} as an open server-side algo order")

    def _validate_open_algo_order(
        self,
        row: dict[str, Any],
        *,
        client_id: str,
        order_type: str,
        exit_side: str,
        trigger_price: float,
    ) -> dict[str, Any]:
        if str(row.get("clientAlgoId") or "") != client_id:
            raise RuntimeError(f"Binance {order_type} confirmation has unexpected clientAlgoId")
        expected = {
            "symbol": self.settings.symbol,
            "algoType": "CONDITIONAL",
            "orderType": order_type,
            "side": exit_side,
            "positionSide": "BOTH",
            "workingType": "MARK_PRICE",
        }
        for field, value in expected.items():
            actual = str(row.get(field) or "").upper()
            if actual != str(value).upper():
                raise RuntimeError(f"Binance {order_type} confirmation has unexpected {field}")
        if not self._truthy(row.get("closePosition")):
            raise RuntimeError(f"Binance {order_type} is not a close-position protection order")
        if not row.get("algoId"):
            raise RuntimeError(f"Binance {order_type} confirmation has no algo id")
        try:
            actual_trigger = float(row.get("triggerPrice") or 0)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Binance {order_type} confirmation has an invalid trigger price") from error
        tick = float(self.symbol_rules()["price_tick"])
        if actual_trigger <= 0 or abs(actual_trigger - trigger_price) > max(tick / 2, 1e-12):
            raise RuntimeError(f"Binance {order_type} confirmation has an unexpected trigger price")
        status = str(row.get("algoStatus") or "").upper()
        if status in {"CANCELED", "EXPIRED", "REJECTED", "FINISHED"}:
            raise RuntimeError(f"Binance {order_type} is not active (status={status})")
        return row

    @staticmethod
    def _is_ambiguous_order_error(error: Exception) -> bool:
        message = str(error)
        return "network error" in message or any(f"HTTP {code}" in message for code in range(500, 600))

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _private_error_is_transient(error: Exception) -> bool:
        message = str(error).lower()
        if isinstance(error, ApiError):
            if error.status_code in {None, 408, 418, 425, 429, 500, 502, 503, 504}:
                return True
            if "-1021" in message:
                return True
        return isinstance(error, (TimeoutError, ConnectionError, OSError)) or any(
            marker in message
            for marker in (
                "network error",
                "timed out",
                "timeout",
                "ssl",
                "connection reset",
                "connection aborted",
                "temporary failure",
            )
        )

    @staticmethod
    def _private_error_is_retryable(error: Exception) -> bool:
        """Return whether an entry may safely retry after a short backoff."""
        message = str(error).lower()
        if isinstance(error, ApiError):
            status = error.status_code
            if status is None or status in {408, 425, 500, 502, 503, 504}:
                return True
            if "-1021" in message:
                return True
            return False
        return isinstance(error, (TimeoutError, ConnectionError, OSError)) or any(
            marker in message
            for marker in (
                "network error",
                "timed out",
                "timeout",
                "ssl",
                "connection reset",
                "connection aborted",
                "temporary failure",
            )
        )

    @staticmethod
    def _decimal_string(value: Decimal) -> str:
        return format(value, "f")

    @staticmethod
    def _round_step(value: Decimal, step: Decimal, rounding: str) -> Decimal:
        if step <= 0:
            raise ValueError("step must be positive")
        units = (value / step).to_integral_value(rounding=rounding)
        return units * step

    @staticmethod
    def _regular_order_view(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "order_id": row.get("orderId"),
            "client_order_id": row.get("clientOrderId"),
            "side": row.get("side"),
            "type": row.get("type"),
            "status": row.get("status"),
            "quantity": float(row.get("origQty") or 0),
            "executed_quantity": float(row.get("executedQty") or 0),
            "price": float(row.get("price") or 0),
            "stop_price": float(row.get("stopPrice") or 0),
            "reduce_only": bool(row.get("reduceOnly")),
            "update_time": row.get("updateTime"),
        }

    @staticmethod
    def _algo_order_view(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "order_id": row.get("algoId"),
            "client_order_id": row.get("clientAlgoId"),
            "side": row.get("side"),
            "type": row.get("orderType"),
            "status": row.get("algoStatus"),
            "quantity": float(row.get("quantity") or 0),
            "executed_quantity": 0.0,
            "price": float(row.get("price") or 0),
            "stop_price": float(row.get("triggerPrice") or 0),
            "reduce_only": bool(row.get("reduceOnly") or row.get("closePosition")),
            "update_time": row.get("updateTime"),
        }

    @staticmethod
    def _to_contracts(quantity: float) -> float:
        return quantity

    def _signed(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        key, secret, _ = self.credentials()
        headers = {"X-MBX-APIKEY": key, "Content-Type": "application/x-www-form-urlencoded"}
        request_method = method.upper()
        self._ensure_server_time()
        max_attempts = 3 if request_method == "GET" else 2
        for attempt in range(max_attempts):
            signed_params = dict(params or {})
            signed_params["timestamp"] = self._server_timestamp_ms()
            signed_params.setdefault("recvWindow", 10_000)
            query = urlencode([(name, value) for name, value in signed_params.items() if value is not None])
            signed_params["signature"] = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            try:
                if request_method in {"GET", "DELETE"}:
                    return request_json(
                        method,
                        f"{self.base_url}{path}",
                        params=signed_params,
                        headers=headers,
                        timeout=5.0,
                        max_attempts=1,
                    )
                return request_json(
                    method,
                    f"{self.base_url}{path}",
                    headers=headers,
                    body=urlencode(signed_params),
                    timeout=5.0,
                )
            except ApiError as error:
                safe_message = redact_url_credentials(str(error))
                has_next_attempt = attempt + 1 < max_attempts
                if has_next_attempt and "-1021" in safe_message:
                    self._sync_server_time(force=True)
                    continue
                retryable_get = (
                    request_method == "GET"
                    and has_next_attempt
                    and not error.rate_limited
                    and error.status_code in {None, 408, 425, 500, 502, 503, 504}
                )
                if retryable_get:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise ApiError(
                    safe_message,
                    status_code=error.status_code,
                    retry_at=error.retry_at,
                    api_code=error.api_code,
                ) from None
        raise ApiError("Binance signed request failed after server-time synchronization")

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
                f"{self.base_url}/fapi/v1/time",
                timeout=5.0,
                max_attempts=3,
            )
            finished_at = self.now_ms()
            finished_monotonic = time.monotonic()
            server_time = int(payload.get("serverTime") or 0)
            if server_time <= 0:
                raise ApiError("Binance server-time response is invalid")
            # Anchor exchange time to monotonic time at response completion.
            # This is deliberately conservative: asymmetric/slow network paths
            # can make midpoint estimation run ahead of Binance by >1000 ms.
            self._server_time_offset_ms = server_time - finished_at
            self._server_time_anchor_ms = server_time
            self._server_time_anchor_monotonic = finished_monotonic
            self._server_time_synced_at = finished_monotonic
            return self._server_time_offset_ms
