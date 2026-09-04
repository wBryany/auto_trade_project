from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo


DEFAULT_TIMEZONE = "Asia/Shanghai"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _Trade:
    index: int
    side: str
    signal_score: str
    entry_hour: str
    entry_time: datetime | None
    exit_time: datetime | None
    net_pnl: float
    trade_return: float | None
    equity_before: float | None
    trading_fee: float
    slippage_cost: float
    funding_fee: float
    total_cost: float
    mfe_r: float | None
    mae_r: float | None
    mfe_price: float | None
    mae_price: float | None


def read_trade_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a ``TradeReporter`` detail CSV (including its UTF-8 BOM)."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        if "net_pnl" not in reader.fieldnames:
            raise ValueError(f"{source} is not a TradeReporter CSV: missing net_pnl column")
        return [dict(row) for row in reader]


def analyze_trade_csv(
    path: str | Path,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    """Read and analyze a ``trade_report.csv`` file."""
    return analyze_trades(read_trade_csv(path), timezone_name=timezone_name)


def analyze_trades(
    rows: Iterable[Mapping[str, Any]],
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    """Calculate a deterministic, JSON-safe trade post-mortem.

    Profit factor and all PnL statistics use *net* PnL.  The Sharpe ratio is
    the arithmetic mean of per-trade ``net_pnl_pct`` divided by its sample
    standard deviation, with a zero risk-free return and no annualization.
    Missing ``net_pnl_pct`` values are derived as ``net_pnl / entry_notional``.
    """
    try:
        report_timezone = ZoneInfo(timezone_name)
    except Exception as exc:
        raise ValueError(f"unknown timezone: {timezone_name}") from exc

    trades = [
        _normalize_trade(row, index=index, report_timezone=report_timezone)
        for index, row in enumerate(rows)
    ]
    trades.sort(key=_chronological_key)

    return {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "trades": len(trades),
            "entry_hour_timezone": timezone_name,
        },
        "methodology": {
            "pnl_basis": "net_pnl_after_costs",
            "win_definition": "net_pnl > 0; breakeven trades remain in the win-rate denominator",
            "profit_factor": "sum(positive net_pnl) / abs(sum(negative net_pnl))",
            "average_loss_sign": "negative",
            "max_drawdown": "peak-to-trough drawdown of chronological cumulative net_pnl, starting at zero",
            "max_drawdown_pct": "same synthetic curve rebased to the first positive equity_before; null when unavailable",
            "sharpe": "mean per-trade net_pnl_pct / sample standard deviation; zero risk-free return; not annualized",
            "mfe_mae": "arithmetic means of available TradeReporter mfe_r/mae_r and price-distance fields",
        },
        "overall": _metrics(trades),
        "by_side": _group_metrics(trades, lambda trade: trade.side, _side_sort_key),
        "by_signal_score": _group_metrics(
            trades,
            lambda trade: trade.signal_score,
            _numeric_label_sort_key,
        ),
        "by_entry_hour": _group_metrics(
            trades,
            lambda trade: trade.entry_hour,
            _numeric_label_sort_key,
        ),
    }


def render_markdown(report: Mapping[str, Any], *, title: str = "交易尸检报告") -> str:
    """Render :func:`analyze_trades` output as a human-readable Markdown report."""
    overall = _mapping(report.get("overall"))
    input_data = _mapping(report.get("input"))
    timezone_name = str(input_data.get("entry_hour_timezone") or DEFAULT_TIMEZONE)

    lines = [
        f"# {title}",
        "",
        "## 统计口径",
        "",
        "- 盈亏、Profit Factor、期望和回撤均按扣除手续费、滑点及资金费后的 `net_pnl` 计算。",
        "- 胜率 = 盈利笔数 / 全部交易笔数；持平交易计入分母，但不计入盈利或亏损。",
        "- Sharpe 使用逐笔 `net_pnl_pct` 收益率、样本标准差、无风险收益 0，且不年化。",
        f"- 入场小时按 `{_escape_markdown(timezone_name)}` 时区统计；MFE/MAE 是已有 `mfe_r`/`mae_r` 样本的算术平均。",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 交易数 | {_integer(overall.get('trades'))} |",
        f"| 盈利 / 亏损 / 持平 | {_integer(overall.get('wins'))} / {_integer(overall.get('losses'))} / {_integer(overall.get('breakevens'))} |",
        f"| 胜率 | {_format_percent(overall.get('win_rate'))} |",
        f"| Profit Factor | {_format_profit_factor(overall)} |",
        f"| 平均盈利 | {_format_number(overall.get('average_win'))} |",
        f"| 平均亏损 | {_format_number(overall.get('average_loss'))} |",
        f"| 盈亏比 | {_format_number(overall.get('payoff_ratio'))} |",
        f"| 单笔期望 | {_format_number(overall.get('expectancy'))} |",
        f"| 净利润 | {_format_number(overall.get('net_pnl'))} |",
        f"| 最大回撤 | {_format_number(overall.get('max_drawdown'))} |",
        f"| 最大回撤率 | {_format_percent(overall.get('max_drawdown_pct'))} |",
        f"| 逐笔 Sharpe（不年化） | {_format_number(overall.get('sharpe'))} |",
        f"| 平均 MFE | {_format_r(overall.get('average_mfe_r'), overall.get('mfe_samples'))} |",
        f"| 平均 MAE | {_format_r(overall.get('average_mae_r'), overall.get('mae_samples'))} |",
        f"| 总成本 | {_format_number(overall.get('total_cost'))} |",
        f"| 手续费 / 滑点 / 资金费 | {_format_number(overall.get('trading_fee'))} / {_format_number(overall.get('slippage_cost'))} / {_format_number(overall.get('funding_fee'))} |",
    ]

    lines.extend(_group_markdown("按方向", "方向", _mapping(report.get("by_side"))))
    lines.extend(_group_markdown("按信号分数", "signal_score", _mapping(report.get("by_signal_score"))))
    lines.extend(
        _group_markdown(
            f"按入场小时（{timezone_name}）",
            "小时",
            _mapping(report.get("by_entry_hour")),
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def write_analysis_reports(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    title: str = "交易尸检报告",
) -> None:
    """Write JSON and Markdown versions of an analysis report."""
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_target.write_text(render_markdown(report, title=title), encoding="utf-8")


def _normalize_trade(
    row: Mapping[str, Any],
    *,
    index: int,
    report_timezone: ZoneInfo,
) -> _Trade:
    entry_time = _parse_datetime(row.get("entry_time"))
    exit_time = _parse_datetime(row.get("exit_time"))
    entry_notional = _optional_float(row.get("entry_notional"))
    net_pnl = _optional_float(row.get("net_pnl")) or 0.0
    trade_return = _optional_float(row.get("net_pnl_pct"))
    if trade_return is None and entry_notional not in {None, 0.0}:
        trade_return = net_pnl / abs(entry_notional)

    trading_fee = _optional_float(row.get("trading_fee"))
    if trading_fee is None:
        trading_fee = (_optional_float(row.get("entry_fee")) or 0.0) + (
            _optional_float(row.get("exit_fee")) or 0.0
        )
    slippage_cost = _optional_float(row.get("slippage_cost")) or 0.0
    funding_fee = _optional_float(row.get("funding_fee")) or 0.0
    total_cost = _optional_float(row.get("total_cost"))
    if total_cost is None:
        total_cost = trading_fee + slippage_cost + funding_fee

    side = str(row.get("side") or "unknown").strip().lower() or "unknown"
    signal_score = _signal_score_label(row.get("signal_score"))
    entry_hour = (
        f"{entry_time.astimezone(report_timezone).hour:02d}"
        if entry_time is not None
        else "unknown"
    )
    return _Trade(
        index=index,
        side=side,
        signal_score=signal_score,
        entry_hour=entry_hour,
        entry_time=entry_time,
        exit_time=exit_time,
        net_pnl=net_pnl,
        trade_return=trade_return,
        equity_before=_optional_float(row.get("equity_before")),
        trading_fee=trading_fee,
        slippage_cost=slippage_cost,
        funding_fee=funding_fee,
        total_cost=total_cost,
        mfe_r=_optional_float(row.get("mfe_r")),
        mae_r=_optional_float(row.get("mae_r")),
        mfe_price=_optional_float(row.get("mfe_price")),
        mae_price=_optional_float(row.get("mae_price")),
    )


def _metrics(trades: list[_Trade]) -> dict[str, Any]:
    pnl = [trade.net_pnl for trade in trades]
    winning_pnl = [value for value in pnl if value > 0.0]
    losing_pnl = [value for value in pnl if value < 0.0]
    breakevens = len(pnl) - len(winning_pnl) - len(losing_pnl)
    gross_profit = sum(winning_pnl)
    gross_loss = abs(sum(losing_pnl))
    average_win = _mean_or_none(winning_pnl)
    average_loss = _mean_or_none(losing_pnl)

    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
        profit_factor_reason = None
    elif gross_profit > 0.0:
        profit_factor = None
        profit_factor_reason = "no_losing_trades"
    else:
        profit_factor = None
        profit_factor_reason = "no_profit_or_loss"

    if average_win is not None and average_loss not in {None, 0.0}:
        payoff_ratio = average_win / abs(average_loss)
        payoff_ratio_reason = None
    else:
        payoff_ratio = None
        payoff_ratio_reason = "requires_winning_and_losing_trades"

    returns = [trade.trade_return for trade in trades if trade.trade_return is not None]
    sharpe: float | None = None
    sharpe_reason: str | None = None
    if len(returns) < 2:
        sharpe_reason = "requires_at_least_two_returns"
    else:
        return_stddev = statistics.stdev(returns)
        if return_stddev == 0.0:
            sharpe_reason = "zero_return_variance"
        else:
            sharpe = statistics.fmean(returns) / return_stddev

    max_drawdown, max_drawdown_pct = _drawdown(trades)
    mfe_r = [trade.mfe_r for trade in trades if trade.mfe_r is not None]
    mae_r = [trade.mae_r for trade in trades if trade.mae_r is not None]
    mfe_price = [trade.mfe_price for trade in trades if trade.mfe_price is not None]
    mae_price = [trade.mae_price for trade in trades if trade.mae_price is not None]

    return {
        "trades": len(trades),
        "wins": len(winning_pnl),
        "losses": len(losing_pnl),
        "breakevens": breakevens,
        "win_rate": len(winning_pnl) / len(trades) if trades else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "profit_factor_reason": profit_factor_reason,
        "average_win": average_win,
        "average_loss": average_loss,
        "payoff_ratio": payoff_ratio,
        "payoff_ratio_reason": payoff_ratio_reason,
        "expectancy": statistics.fmean(pnl) if pnl else 0.0,
        "net_pnl": sum(pnl),
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "return_samples": len(returns),
        "average_trade_return": statistics.fmean(returns) if returns else None,
        "sharpe": sharpe,
        "sharpe_reason": sharpe_reason,
        "mfe_samples": len(mfe_r),
        "mae_samples": len(mae_r),
        "average_mfe_r": _mean_or_none(mfe_r),
        "average_mae_r": _mean_or_none(mae_r),
        "average_mfe_price_distance": _mean_or_none(mfe_price),
        "average_mae_price_distance": _mean_or_none(mae_price),
        "trading_fee": sum(trade.trading_fee for trade in trades),
        "slippage_cost": sum(trade.slippage_cost for trade in trades),
        "funding_fee": sum(trade.funding_fee for trade in trades),
        "total_cost": sum(trade.total_cost for trade in trades),
    }


def _drawdown(trades: list[_Trade]) -> tuple[float, float | None]:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    first_equity = next(
        (
            trade.equity_before
            for trade in trades
            if trade.equity_before is not None and trade.equity_before > 0.0
        ),
        None,
    )
    equity = first_equity
    equity_peak = first_equity
    max_drawdown_pct: float | None = 0.0 if first_equity is not None else None

    for trade in trades:
        cumulative += trade.net_pnl
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
        if equity is not None and equity_peak is not None:
            equity += trade.net_pnl
            equity_peak = max(equity_peak, equity)
            if equity_peak > 0.0:
                drawdown_pct = (equity_peak - equity) / equity_peak
                max_drawdown_pct = max(max_drawdown_pct or 0.0, drawdown_pct)
    return max_drawdown, max_drawdown_pct


def _group_metrics(
    trades: list[_Trade],
    key_function: Any,
    sort_key: Any,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[_Trade]] = {}
    for trade in trades:
        groups.setdefault(str(key_function(trade)), []).append(trade)
    return {
        key: _metrics(groups[key])
        for key in sorted(groups, key=sort_key)
    }


def _chronological_key(trade: _Trade) -> tuple[int, datetime, int]:
    timestamp = trade.exit_time or trade.entry_time
    if timestamp is None:
        return (1, datetime.max.replace(tzinfo=timezone.utc), trade.index)
    return (0, timestamp, trade.index)


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(text)
    except ValueError:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _signal_score_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    number = _optional_float(value)
    if number is not None and number.is_integer():
        return str(int(number))
    return text


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _numeric_label_sort_key(value: str) -> tuple[int, float, str]:
    number = _optional_float(value)
    if number is None:
        return (1, 0.0, value)
    return (0, number, value)


def _side_sort_key(value: str) -> tuple[int, str]:
    priority = {"long": 0, "short": 1, "unknown": 3}
    return (priority.get(value, 2), value)


def _group_markdown(
    heading: str,
    label_heading: str,
    groups: Mapping[str, Any],
) -> list[str]:
    lines = [
        "",
        f"## {heading}",
        "",
        f"| {label_heading} | 交易数 | 胜率 | PF | 平均盈利 | 平均亏损 | 盈亏比 | 单笔期望 | 净利润 | 最大回撤 | Sharpe | MFE | MAE | 总成本 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not groups:
        lines.append("| 无样本 | 0 | 0.00% | — | — | — | — | 0.000000 | 0.000000 | 0.000000 | — | — | — | 0.000000 |")
        return lines
    for label, raw_metrics in groups.items():
        metrics = _mapping(raw_metrics)
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(str(label)),
                    _integer(metrics.get("trades")),
                    _format_percent(metrics.get("win_rate")),
                    _format_profit_factor(metrics),
                    _format_number(metrics.get("average_win")),
                    _format_number(metrics.get("average_loss")),
                    _format_number(metrics.get("payoff_ratio")),
                    _format_number(metrics.get("expectancy")),
                    _format_number(metrics.get("net_pnl")),
                    _format_number(metrics.get("max_drawdown")),
                    _format_number(metrics.get("sharpe")),
                    _format_r(metrics.get("average_mfe_r"), metrics.get("mfe_samples")),
                    _format_r(metrics.get("average_mae_r"), metrics.get("mae_samples")),
                    _format_number(metrics.get("total_cost")),
                ]
            )
            + " |"
        )
    return lines


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "0"


def _format_number(value: Any) -> str:
    number = _optional_float(value)
    return f"{number:,.6f}" if number is not None else "—"


def _format_percent(value: Any) -> str:
    number = _optional_float(value)
    return f"{number * 100:.2f}%" if number is not None else "—"


def _format_profit_factor(metrics: Mapping[str, Any]) -> str:
    value = _optional_float(metrics.get("profit_factor"))
    if value is not None:
        return f"{value:.4f}"
    if metrics.get("profit_factor_reason") == "no_losing_trades" and (
        _optional_float(metrics.get("gross_profit")) or 0.0
    ) > 0.0:
        return "∞"
    return "—"


def _format_r(value: Any, samples: Any) -> str:
    if _integer(samples) == "0":
        return "—"
    number = _optional_float(value)
    return f"{number:.4f}R" if number is not None else "—"


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")
