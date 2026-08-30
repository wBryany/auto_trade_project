from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from ..models import Candle, OrderRequest, Position


@dataclass(frozen=True)
class ExchangeSettings:
    name: str
    environment: str
    base_url: str
    symbol: str
    api_key_env: str = ""
    api_secret_env: str = ""
    passphrase_env: str = ""
    settle: str = "usdt"
    margin_mode: str = "isolated"
    position_mode: str = "net"
    contract_size: float = 1.0
    websocket_enabled: bool = False
    websocket_stale_seconds: float = 15.0


class ExchangeAdapter(ABC):
    def __init__(self, settings: ExchangeSettings) -> None:
        self.settings = settings

    @property
    def name(self) -> str:
        return self.settings.name

    @abstractmethod
    def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
        raise NotImplementedError

    @abstractmethod
    def fetch_equity(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def place_market_order(self, request: OrderRequest) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def place_stop_order(self, position: Position) -> dict[str, Any]:
        raise NotImplementedError

    def fetch_dashboard_snapshot(self) -> dict[str, Any]:
        """Return optional account/order data used by the local dashboard."""
        return {"market": {}, "positions": [], "open_orders": [], "private_available": False}

    def fetch_mark_price(self) -> float:
        """Return the configured venue's current derivative mark price.

        Adapters with a lightweight mark-price endpoint should override this
        method.  The snapshot fallback keeps the interface usable for venues
        that already expose an official mark price through their dashboard
        payload.
        """
        snapshot = self.fetch_dashboard_snapshot()
        market = snapshot.get("market") or {}
        price = float(market.get("mark_price") or market.get("last_price") or 0)
        if price <= 0:
            raise RuntimeError(f"{self.name}: mark price is unavailable for {self.settings.symbol}")
        return price

    def close(self) -> None:
        """Release optional exchange-specific background resources."""

        return None

    def prepare_live(self, *, max_leverage: float) -> dict[str, Any]:
        """Validate credentials/account state before order-capable mode starts."""
        if not self.has_credentials():
            raise RuntimeError(f"{self.name}: live mode requires API credentials")
        return {"exchange": self.name, "prepared": True, "max_leverage": max_leverage}

    def normalize_order_quantity(self, quantity: float, reference_price: float) -> float:
        """Apply venue quantity filters before submission."""
        if quantity <= 0 or reference_price <= 0:
            raise ValueError("quantity and reference_price must be positive")
        return float(quantity)

    def normalize_trigger_price(self, price: float, side: str) -> float:
        if price <= 0 or side.lower() not in {"buy", "sell"}:
            raise ValueError("invalid trigger price or side")
        return float(price)

    def market_fill(self, payload: dict[str, Any], *, fallback_price: float) -> tuple[float, float]:
        """Return confirmed filled quantity and average price."""
        quantity = float(payload.get("executedQty") or payload.get("filled_size") or 0)
        price = float(payload.get("avgPrice") or payload.get("fill_price") or fallback_price)
        status = str(payload.get("status") or "").upper()
        if status and status not in {"FILLED", "SUCCESS", "FINISHED"}:
            raise RuntimeError(f"{self.name}: market order not filled; status={status}")
        if quantity <= 0 or price <= 0:
            raise RuntimeError(f"{self.name}: market order response has no confirmed fill")
        return quantity, price

    def place_protection_orders(
        self,
        position: Position,
        *,
        stop_client_id: str = "",
        take_profit_client_id: str = "",
    ) -> dict[str, Any]:
        return {"stop": self.place_stop_order(position), "take_profit": None}

    def emergency_close(self, position: Position, *, client_id: str) -> dict[str, Any]:
        side = "sell" if position.side == "long" else "buy"
        return self.place_market_order(OrderRequest(side, position.quantity, True, client_id))

    def cancel_protection_orders(self, position: Position) -> dict[str, Any]:
        return {"cancelled": False, "reason": "adapter_has_no_protection_cancellation"}

    def fetch_live_position(self) -> dict[str, Any] | None:
        """Return the venue position for the configured symbol, or None when flat."""
        return None

    def fetch_protection_status(self, position: Position) -> dict[str, Any]:
        return {}

    def fetch_latest_close_fill(
        self,
        position: Position,
        *,
        after_ms: int = 0,
    ) -> dict[str, Any] | None:
        """Return the venue's latest fills that flattened ``position``.

        This read-only reconciliation hook is used only as a notification
        fallback when the venue has a position that the local engine could not
        register. Adapters without private fill history can return ``None``.
        """
        return None

    def has_credentials(self) -> bool:
        if not self.settings.api_key_env or not self.settings.api_secret_env:
            return False
        if not (os.getenv(self.settings.api_key_env) and os.getenv(self.settings.api_secret_env)):
            return False
        if self.name == "okx" and self.settings.passphrase_env:
            return bool(os.getenv(self.settings.passphrase_env))
        return True

    def credentials(self) -> tuple[str, str, str]:
        key = os.getenv(self.settings.api_key_env, "") if self.settings.api_key_env else ""
        secret = os.getenv(self.settings.api_secret_env, "") if self.settings.api_secret_env else ""
        passphrase = os.getenv(self.settings.passphrase_env, "") if self.settings.passphrase_env else ""
        if not key or not secret or (self.name == "okx" and not passphrase):
            raise RuntimeError(f"{self.name}: missing API credentials in environment")
        return key, secret, passphrase

    @staticmethod
    def now_ms() -> int:
        return int(time.time() * 1000)


def candle_limit_for(interval: str, requested: int) -> int:
    if interval == "1s":
        return max(60, min(int(requested), 300))
    if interval not in {"1m", "5m", "15m", "1h", "4h"}:
        raise ValueError(f"unsupported interval: {interval}")
    return max(10, min(int(requested), 1500))
