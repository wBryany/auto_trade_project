from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from btc_futures_bot.backtest import run_backtest
from btc_futures_bot.costs import CostConfig
from btc_futures_bot.main import load_config
from btc_futures_bot.reporting import TradeReporter
from btc_futures_bot.risk import RiskConfig, RiskManager
from btc_futures_bot.strategy import MultiTimeframeStrategy, StrategyConfig


FOLDS = {
    "wf_1": (datetime(2026, 8, 6, tzinfo=timezone.utc), datetime(2026, 8, 13, tzinfo=timezone.utc)),
    "wf_2": (datetime(2026, 8, 13, tzinfo=timezone.utc), datetime(2026, 8, 20, tzinfo=timezone.utc)),
    "wf_3_holdout": (datetime(2026, 8, 20, tzinfo=timezone.utc), None),
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    pnl = [float(row["net_pnl"]) for row in rows]
    gross = [float(row["gross_pnl"]) for row in rows]
    costs = [float(row["total_cost"]) for row in rows]
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    wins = sum(value > 0 for value in pnl)
    gross_profit = sum(value for value in pnl if value > 0)
    gross_loss = abs(sum(value for value in pnl if value < 0))
    return {
        "trades": len(rows),
        "wins": wins,
        "win_rate_pct": (100.0 * wins / len(rows)) if rows else 0.0,
        "gross_pnl": sum(gross),
        "total_cost": sum(costs),
        "net_pnl": sum(pnl),
        "expectancy_per_trade": (sum(pnl) / len(rows)) if rows else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss else (float("inf") if gross_profit else 0.0),
        "max_drawdown_usdt": max_drawdown,
    }


def _fold_rows(rows: list[dict[str, str]], start: datetime, end: datetime | None) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        entered = datetime.fromisoformat(row["entry_time"].replace("Z", "+00:00"))
        if entered >= start and (end is None or entered < end):
            selected.append(row)
    return selected


def _candidates() -> list[dict[str, Any]]:
    return [
        {"name": "baseline", "strategy": {}, "risk": {}},
        {
            "name": "regime_slope_macd",
            "strategy": {
                "traditional_strong_regime_require_fast_slope": True,
                "traditional_strong_regime_require_macd": True,
            },
            "risk": {},
        },
        *[
            {
                "name": f"regime_q_{str(gap).replace('.', '_')}",
                "strategy": {
                    "traditional_strong_regime_min_gap_atr": gap,
                    "traditional_strong_regime_require_fast_slope": True,
                    "traditional_strong_regime_require_macd": True,
                },
                "risk": {},
            }
            for gap in (0.25, 0.5, 0.75, 1.0)
        ],
        {
            "name": "cross_q_035_060",
            "strategy": {
                "traditional_cross_min_body_ratio": 0.35,
                "traditional_cross_min_close_location": 0.60,
                "traditional_cross_max_extension_atr": 1.25,
            },
            "risk": {},
        },
        {
            "name": "cross_q_050_070",
            "strategy": {
                "traditional_cross_min_body_ratio": 0.50,
                "traditional_cross_min_close_location": 0.70,
                "traditional_cross_max_extension_atr": 1.00,
            },
            "risk": {},
        },
        *[
            {
                "name": f"combo_gap_{str(gap).replace('.', '_')}_cross_{body}_{close}",
                "strategy": {
                    "traditional_strong_regime_min_gap_atr": gap,
                    "traditional_strong_regime_require_fast_slope": True,
                    "traditional_strong_regime_require_macd": True,
                    "traditional_cross_min_body_ratio": body,
                    "traditional_cross_min_close_location": close,
                    "traditional_cross_max_extension_atr": 1.25,
                },
                "risk": {},
            }
            for gap, body, close in ((0.5, 0.35, 0.60), (0.5, 0.50, 0.70), (0.75, 0.35, 0.60))
        ],
        *[
            {
                "name": f"combo_gap_0_5_cross_035_cooldown_{minutes}",
                "strategy": {
                    "traditional_strong_regime_min_gap_atr": 0.5,
                    "traditional_strong_regime_require_fast_slope": True,
                    "traditional_strong_regime_require_macd": True,
                    "traditional_cross_min_body_ratio": 0.35,
                    "traditional_cross_min_close_location": 0.60,
                    "traditional_cross_max_extension_atr": 1.25,
                },
                "risk": {"cooldown_minutes": minutes},
            }
            for minutes in (30, 60)
        ],
        {
            "name": "combo_gap_0_5_cross_035_be_1_0",
            "strategy": {
                "traditional_strong_regime_min_gap_atr": 0.5,
                "traditional_strong_regime_require_fast_slope": True,
                "traditional_strong_regime_require_macd": True,
                "traditional_cross_min_body_ratio": 0.35,
                "traditional_cross_min_close_location": 0.60,
                "traditional_cross_max_extension_atr": 1.25,
                "break_even_trigger_r": 1.0,
            },
            "risk": {},
        },
        {
            "name": "combo_gap_0_5_cross_035_volume_1_2",
            "strategy": {
                "traditional_strong_regime_min_gap_atr": 0.5,
                "traditional_strong_regime_require_fast_slope": True,
                "traditional_strong_regime_require_macd": True,
                "traditional_cross_min_body_ratio": 0.35,
                "traditional_cross_min_close_location": 0.60,
                "traditional_cross_max_extension_atr": 1.25,
                "traditional_min_volume_ratio": 1.2,
            },
            "risk": {},
        },
        {
            "name": "cross_family_only",
            "strategy": {
                "traditional_allow_pullback": False,
                "traditional_allow_breakout": False,
            },
            "risk": {},
        },
        {
            "name": "cross_family_only_be_1_0",
            "strategy": {
                "traditional_allow_pullback": False,
                "traditional_allow_breakout": False,
                "break_even_trigger_r": 1.0,
            },
            "risk": {},
        },
        {
            "name": "cross_family_only_cooldown_30",
            "strategy": {
                "traditional_allow_pullback": False,
                "traditional_allow_breakout": False,
            },
            "risk": {"cooldown_minutes": 30},
        },
        {
            "name": "strict_directional_regime",
            "strategy": {
                "traditional_allow_early_regime": False,
                "traditional_countertrend_cross_max_regime_gap_pct": 0.0,
                "traditional_countertrend_pullback_max_regime_gap_pct": 0.0,
                "traditional_neutral_transition_max_regime_gap_pct": 0.0,
            },
            "risk": {},
        },
        {
            "name": "strict_directional_cross_family",
            "strategy": {
                "traditional_allow_pullback": False,
                "traditional_allow_breakout": False,
                "traditional_allow_early_regime": False,
                "traditional_countertrend_cross_max_regime_gap_pct": 0.0,
                "traditional_countertrend_pullback_max_regime_gap_pct": 0.0,
                "traditional_neutral_transition_max_regime_gap_pct": 0.0,
            },
            "risk": {},
        },
        {
            "name": "trigger_15m",
            "strategy": {"trigger_timeframe": "15m"},
            "risk": {},
        },
        {
            "name": "trigger_15m_be_1_0",
            "strategy": {"trigger_timeframe": "15m", "break_even_trigger_r": 1.0},
            "risk": {},
        },
        {
            "name": "trigger_15m_strict_directional",
            "strategy": {
                "trigger_timeframe": "15m",
                "traditional_allow_early_regime": False,
                "traditional_countertrend_cross_max_regime_gap_pct": 0.0,
                "traditional_countertrend_pullback_max_regime_gap_pct": 0.0,
                "traditional_neutral_transition_max_regime_gap_pct": 0.0,
            },
            "risk": {},
        },
        {
            "name": "regime_4h_20_50",
            "strategy": {
                "regime_timeframe": "4h",
                "traditional_trend_fast": 20,
                "traditional_trend_slow": 50,
            },
            "risk": {},
        },
        {
            "name": "regime_4h_20_50_cross_family",
            "strategy": {
                "regime_timeframe": "4h",
                "traditional_trend_fast": 20,
                "traditional_trend_slow": 50,
                "traditional_allow_pullback": False,
                "traditional_allow_breakout": False,
            },
            "risk": {},
        },
        {
            "name": "regime_4h_20_50_be_1_0",
            "strategy": {
                "regime_timeframe": "4h",
                "traditional_trend_fast": 20,
                "traditional_trend_slow": 50,
                "break_even_trigger_r": 1.0,
            },
            "risk": {},
        },
        {
            "name": "trigger_15m_regime_4h_20_50",
            "strategy": {
                "trigger_timeframe": "15m",
                "regime_timeframe": "4h",
                "traditional_trend_fast": 20,
                "traditional_trend_slow": 50,
            },
            "risk": {},
        },
        {
            "name": "fixed_tp_2_5",
            "strategy": {},
            "risk": {},
            "fixed_take_profit": True,
        },
        {
            "name": "fixed_tp_2_0",
            "strategy": {"take_profit_r": 2.0},
            "risk": {},
            "fixed_take_profit": True,
        },
        {
            "name": "fixed_tp_1_5",
            "strategy": {"take_profit_r": 1.5},
            "risk": {},
            "fixed_take_profit": True,
        },
        {
            "name": "fixed_tp_2_0_be_1_0",
            "strategy": {"take_profit_r": 2.0, "break_even_trigger_r": 1.0},
            "risk": {},
            "fixed_take_profit": True,
        },
        *[
            {
                "name": f"loss_streak_3_pause_{minutes}",
                "strategy": {},
                "risk": {
                    "max_consecutive_losses": 3,
                    "loss_streak_pause_minutes": minutes,
                },
            }
            for minutes in (240, 720, 1440)
        ],
        *[
            {
                "name": f"loss_streak_{losses}_pause_720",
                "strategy": {},
                "risk": {
                    "max_consecutive_losses": losses,
                    "loss_streak_pause_minutes": 720,
                },
            }
            for losses in (2, 4)
        ],
        {
            "name": "cross_family_be_1_0_loss_streak_3_pause_720",
            "strategy": {
                "traditional_allow_pullback": False,
                "traditional_allow_breakout": False,
                "break_even_trigger_r": 1.0,
            },
            "risk": {
                "max_consecutive_losses": 3,
                "loss_streak_pause_minutes": 720,
            },
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed live-strategy safety candidates")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[], help="evaluate only the named candidate")
    args = parser.parse_args()

    raw = load_config(str(args.config))
    base_strategy = StrategyConfig(**raw.get("strategy", {}))
    base_risk = RiskConfig(**raw.get("risk", {}))
    active_exchange = str(raw.get("active_exchange") or "binance")
    account = raw.get("account", {})
    exchange = raw.get("exchanges", {}).get(active_exchange, {})
    costs = CostConfig(**exchange.get("costs", raw.get("costs", {})))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_candidates = _candidates()
    if args.only:
        requested = set(args.only)
        selected_candidates = [item for item in selected_candidates if item["name"] in requested]
        missing = requested - {str(item["name"]) for item in selected_candidates}
        if missing:
            raise ValueError(f"unknown candidates: {', '.join(sorted(missing))}")

    results: list[dict[str, Any]] = []
    for candidate in selected_candidates:
        name = str(candidate["name"])
        candidate_dir = args.output_dir / name
        reporter = TradeReporter(candidate_dir)
        strategy_config = replace(base_strategy, **candidate["strategy"])
        risk_config = replace(base_risk, **candidate["risk"])
        risk = RiskManager(
            risk_config,
            max_leverage=float(account.get("max_leverage", 3)),
            costs=costs,
        )
        try:
            summary = run_backtest(
                args.data_dir,
                initial_equity=10_000.0,
                candle_limit=int(raw.get("candle_limit", 300)),
                strategy=MultiTimeframeStrategy(strategy_config),
                risk=risk,
                reporter=reporter,
                use_fixed_take_profit=bool(candidate.get("fixed_take_profit", False)),
            )
        finally:
            reporter.close()
        rows = _read_rows(candidate_dir / "trade_report.csv")
        folds = {
            fold_name: _metrics(_fold_rows(rows, start, end))
            for fold_name, (start, end) in FOLDS.items()
        }
        aggregate_rows = [
            row
            for row in rows
            if datetime.fromisoformat(row["entry_time"].replace("Z", "+00:00")) >= FOLDS["wf_1"][0]
        ]
        results.append(
            {
                "name": name,
                "strategy_override": candidate["strategy"],
                "risk_override": candidate["risk"],
                "fixed_take_profit": bool(candidate.get("fixed_take_profit", False)),
                "summary": asdict(summary),
                "full": _metrics(rows),
                "walk_forward_folds": folds,
                "walk_forward_aggregate": _metrics(aggregate_rows),
            }
        )
        print(name, json.dumps(results[-1]["full"], ensure_ascii=False), flush=True)

    baseline = results[0]
    for item in results:
        item["deployment_gate"] = {
            "full_expectancy_positive": item["full"]["expectancy_per_trade"] > 0,
            "walk_forward_expectancy_positive": item["walk_forward_aggregate"]["expectancy_per_trade"] > 0,
            "latest_holdout_expectancy_positive": item["walk_forward_folds"]["wf_3_holdout"]["expectancy_per_trade"] > 0,
            "drawdown_not_worse_than_baseline": item["full"]["max_drawdown_usdt"] <= baseline["full"]["max_drawdown_usdt"],
        }
        item["deployment_gate"]["passed"] = all(item["deployment_gate"].values())

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.data_dir),
        "config": str(args.config),
        "folds": {
            name: {"start": start.isoformat(), "end": end.isoformat() if end else None}
            for name, (start, end) in FOLDS.items()
        },
        "results": results,
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
