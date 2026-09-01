from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    entry_fee: float
    exit_fee: float
    slippage_cost: float
    funding_fee: float

    @property
    def trading_fee(self) -> float:
        return self.entry_fee + self.exit_fee

    @property
    def total_cost(self) -> float:
        return self.trading_fee + self.slippage_cost + self.funding_fee


@dataclass(frozen=True)
class CostConfig:
    """Trading-cost assumptions, expressed as decimal rates.

    Example: 0.0005 means 0.05%. Rates must be changed to the user's actual
    VIP/account rate before live trading.
    """

    execution: str = "taker"
    maker_fee_pct: float = 0.0002
    taker_fee_pct: float = 0.0005
    slippage_pct: float = 0.0002
    funding_rate_pct_per_8h: float = 0.0001
    expected_holding_hours: float = 4.0
    min_net_edge_pct: float = 0.001

    def __post_init__(self) -> None:
        if self.execution not in {"maker", "taker"}:
            raise ValueError("execution must be maker or taker")
        for name in ("maker_fee_pct", "taker_fee_pct", "slippage_pct", "funding_rate_pct_per_8h", "min_net_edge_pct"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.expected_holding_hours < 0:
            raise ValueError("expected_holding_hours cannot be negative")

    @property
    def fee_pct(self) -> float:
        return self.maker_fee_pct if self.execution == "maker" else self.taker_fee_pct

    @property
    def round_trip_pct(self) -> float:
        """Entry plus exit fee and slippage, as a share of entry notional.

        Funding is excluded because it depends on the actual hold. This is the
        price move a position must make before it breaks even, so it is also
        the numerator of "how much of one risk unit the costs consume":
        ``round_trip_pct / stop_loss_pct``.
        """
        return 2.0 * (self.fee_pct + self.slippage_pct)

    @property
    def funding_intervals(self) -> int:
        return self.funding_intervals_for()

    def funding_intervals_for(self, holding_hours: float | None = None) -> int:
        """Estimate settled funding intervals for an expected or actual hold."""
        hours = self.expected_holding_hours if holding_hours is None else max(0.0, float(holding_hours))
        if hours == 0 or self.funding_rate_pct_per_8h == 0:
            return 0
        # Do not charge a full funding interval to a scalp that is expected to
        # close before the next funding settlement.
        return int(hours / 8.0)

    def estimate_round_trip_cost(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        *,
        holding_hours: float | None = None,
    ) -> float:
        """Estimate fees + slippage + conservative funding for one round trip."""
        return self.breakdown(entry_price, exit_price, quantity, holding_hours=holding_hours).total_cost

    def breakdown(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        *,
        holding_hours: float | None = None,
    ) -> CostBreakdown:
        if entry_price <= 0 or exit_price <= 0 or quantity <= 0:
            raise ValueError("entry_price, exit_price and quantity must be positive")
        notional = (entry_price + exit_price) * quantity
        entry_fee = entry_price * quantity * self.fee_pct
        exit_fee = exit_price * quantity * self.fee_pct
        slippage = notional * self.slippage_pct
        funding = entry_price * quantity * abs(self.funding_rate_pct_per_8h) * self.funding_intervals_for(holding_hours)
        return CostBreakdown(entry_fee, exit_fee, slippage, funding)

    def estimate_net_pnl(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        *,
        holding_hours: float | None = None,
    ) -> float:
        gross = (exit_price - entry_price) * quantity * (1 if side == "long" else -1)
        return gross - self.estimate_round_trip_cost(
            entry_price,
            exit_price,
            quantity,
            holding_hours=holding_hours,
        )
