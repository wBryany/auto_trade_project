from __future__ import annotations

from dataclasses import dataclass
from math import floor

from .costs import CostConfig


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.005
    stop_loss_pct: float = 0.05
    max_notional_pct: float = 0.20
    max_daily_loss_pct: float = 0.02
    max_consecutive_losses: int = 3
    cooldown_minutes: int = 15


@dataclass(frozen=True)
class Protection:
    stop_price: float
    take_profit_price: float
    quantity: float
    risk_amount: float


class RiskManager:
    def __init__(
        self,
        config: RiskConfig | None = None,
        quantity_step: float = 0.000001,
        max_leverage: float = 3.0,
        costs: CostConfig | None = None,
    ) -> None:
        self.config = config or RiskConfig()
        self.quantity_step = quantity_step
        self.max_leverage = max_leverage
        self.costs = costs or CostConfig()
        if not 0 < self.config.risk_per_trade <= 0.02:
            raise ValueError("risk_per_trade must be between 0 and 2%")
        if not 0 < self.config.stop_loss_pct < 0.20:
            raise ValueError("stop_loss_pct must be between 0 and 20%")
        if not 0 < self.config.max_notional_pct <= 1:
            raise ValueError("max_notional_pct must be between 0 and 100%")
        if not 0 < self.max_leverage <= 125:
            raise ValueError("max_leverage must be between 0 and 125")

    def protection(
        self,
        side: str,
        equity: float,
        entry_price: float,
        take_profit_r: float = 1.5,
        stop_loss_pct: float | None = None,
        size_multiplier: float = 1.0,
    ) -> Protection:
        if equity <= 0 or entry_price <= 0:
            raise ValueError("equity and entry_price must be positive")
        if side not in {"long", "short"}:
            raise ValueError("side must be long or short")
        selected_stop_loss_pct = self.config.stop_loss_pct if stop_loss_pct is None else float(stop_loss_pct)
        if not 0 < selected_stop_loss_pct < 0.20:
            raise ValueError("stop_loss_pct must be between 0 and 20%")
        if take_profit_r <= 0:
            raise ValueError("take_profit_r must be positive")
        selected_size_multiplier = float(size_multiplier)
        if not 0 < selected_size_multiplier <= 1:
            raise ValueError("size_multiplier must be between 0 and 1")
        stop_distance = entry_price * selected_stop_loss_pct
        stop_price = entry_price - stop_distance if side == "long" else entry_price + stop_distance
        take_profit_price = entry_price + stop_distance * take_profit_r if side == "long" else entry_price - stop_distance * take_profit_r
        risk_amount = equity * self.config.risk_per_trade * selected_size_multiplier
        risk_cost_per_unit = self.costs.estimate_round_trip_cost(entry_price, stop_price, 1.0)
        quantity_by_risk = risk_amount / (stop_distance + risk_cost_per_unit)
        # ``max_notional_pct`` is the share of the account's theoretical
        # leveraged capacity, not a percentage of unleveraged wallet equity.
        # The risk budget remains an independent (and usually tighter) cap.
        theoretical_max_quantity = equity * self.max_leverage / entry_price
        quantity_by_notional = (
            theoretical_max_quantity
            * self.config.max_notional_pct
            * selected_size_multiplier
        )
        quantity_by_leverage = theoretical_max_quantity * selected_size_multiplier
        quantity = min(quantity_by_risk, quantity_by_notional, quantity_by_leverage)
        quantity = floor(quantity / self.quantity_step) * self.quantity_step
        if quantity <= 0:
            raise ValueError("computed quantity is below quantity_step")
        return Protection(stop_price, take_profit_price, quantity, risk_amount)

    def is_cost_effective(self, side: str, entry_price: float, take_profit_price: float, quantity: float) -> bool:
        net_profit = self.costs.estimate_net_pnl(side, entry_price, take_profit_price, quantity)
        notional = entry_price * quantity
        return net_profit > 0 and notional > 0 and net_profit / notional >= self.costs.min_net_edge_pct

    def break_even_price(self, side: str, entry_price: float, *, holding_hours: float | None = None) -> float:
        """Return the exit price that covers estimated round-trip costs."""
        if side not in {"long", "short"}:
            raise ValueError("side must be long or short")
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        variable_rate = self.costs.fee_pct + self.costs.slippage_pct
        funding_rate = abs(self.costs.funding_rate_pct_per_8h) * self.costs.funding_intervals_for(holding_hours)
        if side == "long":
            return entry_price * (1 + variable_rate + funding_rate) / (1 - variable_rate)
        return entry_price * (1 - variable_rate - funding_rate) / (1 + variable_rate)

    def estimate_net_pnl(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        *,
        holding_hours: float | None = None,
    ) -> float:
        return self.costs.estimate_net_pnl(
            side,
            entry_price,
            exit_price,
            quantity,
            holding_hours=holding_hours,
        )
