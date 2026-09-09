from __future__ import annotations

import json
from pathlib import Path

import pytest

from btc_futures_bot.costs import CostConfig
from btc_futures_bot.entry_costs import evaluate_scalp_cost_coverage
from btc_futures_bot.exchanges.okx import OkxAdapter


# This fixed public-data regression does not download anything or inspect an
# account. These four past losses are examples, not an out-of-sample test or
# a reason to tune the guard's parameters.
_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "scalp_cost_sep6.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda case: case["case"])
def test_sep6_low_activity_entries_fail_fee_coverage_using_only_prior_closed_bars(case: dict) -> None:
    rows = case["rows"]
    timestamp = case["signal_timestamp_ms"]
    assert len(rows) == 60
    assert all(len(row) == 9 and row[8] == "1" for row in rows)
    assert int(rows[-1][0]) == timestamp
    assert all(int(row[0]) <= timestamp for row in rows)
    assert all(int(current[0]) - int(previous[0]) == 60_000 for previous, current in zip(rows, rows[1:]))
    assert case["query"]["after"] == timestamp + 60_000
    candles = [OkxAdapter._candle_from_row(row) for row in rows]
    costs = CostConfig(**_FIXTURE["costs"])

    decision = evaluate_scalp_cost_coverage(
        candles, case["side"], _FIXTURE["horizon_seconds"], costs, _FIXTURE["lookback_windows"],
    )

    assert decision.allowed is False
    assert decision.reason == "median_below_cost_hurdle"
    assert decision.horizon_bars == 10
    assert decision.lookback_windows == 6
    expected = case["expected"]
    assert decision.observed_move_pct == pytest.approx(expected["observed_move_pct"])
    assert decision.latest_move_pct == pytest.approx(expected["latest_move_pct"])
    assert decision.required_move_pct == pytest.approx(expected["required_move_pct"])
    assert decision.window_move_pcts == pytest.approx(expected["window_move_pcts"])
    assert decision.observed_move_pct < decision.required_move_pct
    assert decision.latest_move_pct < decision.required_move_pct
    # Independently verify the hurdle through the shared net-PnL calculation:
    # exactly this favorable move earns the configured net edge after costs.
    direction = 1.0 if case["side"] == "long" else -1.0
    entry = float(candles[-1].close)
    exit_price = entry * (1.0 + direction * decision.required_move_pct)
    net_return = costs.estimate_net_pnl(
        case["side"], entry, exit_price, 1.0,
        holding_hours=_FIXTURE["horizon_seconds"] / 3600.0,
    ) / entry
    assert net_return == pytest.approx(costs.min_net_edge_pct)
