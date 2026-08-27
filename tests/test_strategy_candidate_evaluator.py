from __future__ import annotations

import json

from scripts.evaluate_live_strategy_candidates import _bootstrap_expectancy_ci, _metrics


def _row(net_pnl: float, gross_pnl: float | None = None, total_cost: float = 0.0) -> dict[str, str]:
    return {
        "net_pnl": str(net_pnl),
        "gross_pnl": str(net_pnl if gross_pnl is None else gross_pnl),
        "total_cost": str(total_cost),
    }


def test_metrics_serializes_profit_without_losses_as_standard_json() -> None:
    metrics = _metrics([_row(1.0), _row(2.0)])

    assert metrics["profit_factor"] is None
    json.dumps(metrics, allow_nan=False)


def test_bootstrap_expectancy_interval_is_deterministic() -> None:
    rows = [_row(1.0), _row(1.0), _row(1.0)]

    interval = _bootstrap_expectancy_ci(rows, samples=200, seed=7)

    assert interval == {"samples": 200, "lower": 1.0, "upper": 1.0}
