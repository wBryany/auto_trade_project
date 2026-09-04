from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    # OKX derivative candles expose quote-currency turnover separately from
    # contract volume.  Keep it optional so other adapters and old fixtures
    # remain compatible.
    quote_volume: float | None = None


@dataclass(frozen=True)
class Signal:
    side: str
    score: int
    timestamp: int
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Position:
    side: str
    quantity: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    opened_at: int
    initial_stop_price: float | None = None
    best_price: float | None = None
    stop_reason: str = "stop_loss"
    worst_price: float | None = None
    entry_order_id: str = ""
    entry_client_id: str = ""
    stop_order_id: str = ""
    stop_client_id: str = ""
    take_profit_order_id: str = ""
    take_profit_client_id: str = ""
    # Exact entry commission is populated only when Binance reports every
    # fill in the quote asset. ``None`` deliberately distinguishes an
    # unavailable/non-quote fee from a confirmed zero commission.
    entry_fee: float | None = None
    entry_fee_asset: str = ""


@dataclass(frozen=True)
class OrderRequest:
    side: str
    quantity: float
    reduce_only: bool = False
    client_id: str = ""


@dataclass
class TradeResult:
    exchange: str
    status: str
    signal: Signal | None = None
    position: Position | None = None
    raw: dict[str, Any] | None = None
