from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import pytest

from btc_futures_bot.trade_analysis import (
    analyze_trade_csv,
    analyze_trades,
    render_markdown,
    write_analysis_reports,
)


def _row(
    *,
    trade_id: str,
    side: str,
    entry_time: str,
    exit_time: str,
    net_pnl: float,
    net_pnl_pct: float,
    signal_score: int,
    cost: float,
    mfe_r: float,
    mae_r: float,
    equity_before: float,
) -> dict[str, str]:
    return {
        "trade_id": trade_id,
        "side": side,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_notional": "1000",
        "net_pnl": str(net_pnl),
        "net_pnl_pct": str(net_pnl_pct),
        "signal_score": str(signal_score),
        "trading_fee": str(cost / 2),
        "slippage_cost": str(cost / 4),
        "funding_fee": str(cost / 4),
        "total_cost": str(cost),
        "mfe_r": str(mfe_r),
        "mae_r": str(mae_r),
        "mfe_price": str(mfe_r * 10),
        "mae_price": str(mae_r * 10),
        "equity_before": str(equity_before),
    }


def test_analyze_trades_calculates_overall_and_group_metrics() -> None:
    # Deliberately not chronological: drawdown must be based on exit time.
    rows = [
        _row(
            trade_id="four",
            side="short",
            entry_time="2026-01-01T03:00:00Z",
            exit_time="2026-01-01T03:10:00Z",
            net_pnl=25,
            net_pnl_pct=0.025,
            signal_score=8,
            cost=1,
            mfe_r=1.5,
            mae_r=0.1,
            equity_before=1050,
        ),
        _row(
            trade_id="one",
            side="long",
            entry_time="2026-01-01T00:00:00Z",
            exit_time="2026-01-01T00:10:00Z",
            net_pnl=100,
            net_pnl_pct=0.1,
            signal_score=7,
            cost=4,
            mfe_r=2.0,
            mae_r=0.2,
            equity_before=1000,
        ),
        _row(
            trade_id="three",
            side="long",
            entry_time="2026-01-01T02:00:00Z",
            exit_time="2026-01-01T02:10:00Z",
            net_pnl=0,
            net_pnl_pct=0,
            signal_score=7,
            cost=2,
            mfe_r=0.5,
            mae_r=0.4,
            equity_before=1050,
        ),
        _row(
            trade_id="two",
            side="short",
            entry_time="2026-01-01T01:00:00Z",
            exit_time="2026-01-01T01:10:00Z",
            net_pnl=-50,
            net_pnl_pct=-0.05,
            signal_score=8,
            cost=3,
            mfe_r=0.25,
            mae_r=1.0,
            equity_before=1100,
        ),
    ]

    report = analyze_trades(rows, timezone_name="Asia/Shanghai")
    overall = report["overall"]

    assert overall["trades"] == 4
    assert overall["wins"] == 2
    assert overall["losses"] == 1
    assert overall["breakevens"] == 1
    assert overall["win_rate"] == pytest.approx(0.5)
    assert overall["gross_profit"] == pytest.approx(125)
    assert overall["gross_loss"] == pytest.approx(50)
    assert overall["profit_factor"] == pytest.approx(2.5)
    assert overall["average_win"] == pytest.approx(62.5)
    assert overall["average_loss"] == pytest.approx(-50)
    assert overall["payoff_ratio"] == pytest.approx(1.25)
    assert overall["expectancy"] == pytest.approx(18.75)
    assert overall["net_pnl"] == pytest.approx(75)
    assert overall["max_drawdown"] == pytest.approx(50)
    assert overall["max_drawdown_pct"] == pytest.approx(50 / 1100)
    returns = [0.1, -0.05, 0.0, 0.025]
    assert overall["sharpe"] == pytest.approx(statistics.fmean(returns) / statistics.stdev(returns))
    assert overall["average_mfe_r"] == pytest.approx(1.0625)
    assert overall["average_mae_r"] == pytest.approx(0.425)
    assert overall["total_cost"] == pytest.approx(10)
    assert list(report["by_side"]) == ["long", "short"]
    assert report["by_side"]["long"]["trades"] == 2
    assert report["by_signal_score"]["8"]["profit_factor"] == pytest.approx(0.5)
    # Midnight UTC is 08:00 in the configured report timezone.
    assert report["by_entry_hour"]["08"]["net_pnl"] == pytest.approx(100)


def test_empty_and_no_loss_samples_are_json_safe() -> None:
    empty = analyze_trades([])
    overall = empty["overall"]
    assert overall["trades"] == 0
    assert overall["win_rate"] == 0.0
    assert overall["profit_factor"] is None
    assert overall["average_win"] is None
    assert overall["average_loss"] is None
    assert overall["payoff_ratio"] is None
    assert overall["sharpe"] is None
    assert overall["max_drawdown"] == 0.0
    assert overall["max_drawdown_pct"] is None
    json.dumps(empty, allow_nan=False)

    wins_only = analyze_trades(
        [
            {"net_pnl": "10", "net_pnl_pct": "0.01", "side": "long", "signal_score": "5"},
            {"net_pnl": "20", "net_pnl_pct": "0.02", "side": "long", "signal_score": "5"},
        ]
    )
    metrics = wins_only["overall"]
    assert metrics["profit_factor"] is None
    assert metrics["profit_factor_reason"] == "no_losing_trades"
    assert metrics["average_loss"] is None
    assert metrics["payoff_ratio"] is None
    assert "| Profit Factor | ∞ |" in render_markdown(wins_only)
    json.dumps(wins_only, allow_nan=False)


def test_csv_analysis_and_both_output_formats(tmp_path: Path) -> None:
    csv_path = tmp_path / "trade_report.csv"
    rows = [
        {
            "net_pnl": "12",
            "entry_notional": "1000",
            "side": "LONG",
            "signal_score": "7.0",
            "entry_time": "2026-01-01T00:00:00Z",
            "exit_time": "2026-01-01T00:05:00Z",
            # Exercise fallback cost fields and a missing MFE/MAE sample.
            "entry_fee": "1",
            "exit_fee": "1.5",
            "slippage_cost": "0.5",
            "funding_fee": "0.25",
        }
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = analyze_trade_csv(csv_path, timezone_name="UTC")
    assert report["overall"]["average_trade_return"] == pytest.approx(0.012)
    assert report["overall"]["trading_fee"] == pytest.approx(2.5)
    assert report["overall"]["total_cost"] == pytest.approx(3.25)
    assert report["overall"]["average_mfe_r"] is None
    assert report["by_side"]["long"]["trades"] == 1
    assert list(report["by_signal_score"]) == ["7"]

    json_path = tmp_path / "analysis.json"
    markdown_path = tmp_path / "analysis.md"
    write_analysis_reports(report, json_path=json_path, markdown_path=markdown_path)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["overall"]["net_pnl"] == pytest.approx(12)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# 交易尸检报告" in markdown
    assert "按信号分数" in markdown
    assert "逐笔 Sharpe（不年化）" in markdown


def test_invalid_timezone_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown timezone"):
        analyze_trades([], timezone_name="Mars/Olympus_Mons")
