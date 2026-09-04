from __future__ import annotations

import csv
import os
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Iterable

from .costs import CostConfig
from .models import Position, Signal


_EXPORT_LOCKS_GUARD = threading.Lock()
_EXPORT_LOCKS: dict[str, threading.RLock] = {}


def _export_lock(path: Path) -> threading.RLock:
    identity = str(path.resolve())
    with _EXPORT_LOCKS_GUARD:
        return _EXPORT_LOCKS.setdefault(identity, threading.RLock())


def _iso_utc(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_trade_environment(value: Any) -> str:
    """Normalize venue-specific environment names for persisted trade history."""
    environment = str(value or "testnet").strip().lower()
    if environment in {"production", "prod", "mainnet", "live"}:
        return "production"
    return "testnet"


def trade_scope(row: dict[str, Any]) -> str:
    """Classify a completed trade as test or real execution.

    Public production-market data used by a paper run is still a test trade.
    A trade is considered real only when both the venue environment is
    production and the engine mode was live at execution time.
    """
    environment = canonical_trade_environment(row.get("environment"))
    mode = str(row.get("mode") or "paper").strip().lower()
    return "production" if environment == "production" and mode == "live" else "testnet"


def exchange_environment_label(exchange: Any, environment: Any) -> str:
    venue = str(exchange or "").strip().lower()
    venue_label = {"okx": "OKX", "binance": "币安", "gate": "Gate"}.get(
        venue,
        str(exchange or "未知平台"),
    )
    environment_label = "正式网络" if canonical_trade_environment(environment) == "production" else "测试网"
    return f"{venue_label}-{environment_label}"


@dataclass(frozen=True)
class TradeRecord:
    trade_id: str
    exchange: str
    symbol: str
    mode: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_notional: float
    exit_notional: float
    gross_pnl: float
    entry_fee: float
    exit_fee: float
    trading_fee: float
    slippage_cost: float
    funding_fee: float
    total_cost: float
    net_pnl: float
    net_pnl_pct: float
    fee_ratio_pct: float
    fee_to_gross_pct: float
    total_cost_ratio_pct: float
    equity_before: float
    equity_after: float
    exit_reason: str
    signal_score: int
    signal_reasons: str
    signal_timestamp_ms: int
    initial_stop_price: float
    take_profit_price: float
    best_price: float
    worst_price: float
    mfe_price: float
    mae_price: float
    mfe_r: float
    mae_r: float
    realized_r: float
    holding_minutes: float
    environment: str = "testnet"
    model_name: str = ""
    model_version: str = ""
    meta_score: float = 0.0
    meta_threshold: float = 0.0
    meta_decision: str = ""

    @classmethod
    def from_position(
        cls,
        *,
        exchange: str,
        symbol: str,
        mode: str,
        position: Position,
        exit_price: float,
        exit_time_ms: int,
        exit_reason: str,
        costs: CostConfig,
        equity_before: float,
        signal: Signal | None = None,
        environment: str = "testnet",
    ) -> "TradeRecord":
        direction = 1 if position.side == "long" else -1
        initial_stop = position.initial_stop_price or position.stop_price
        risk_distance = abs(position.entry_price - initial_stop)
        best_price = position.best_price or position.entry_price
        worst_price = position.worst_price or position.entry_price
        mfe_price = (
            max(0.0, best_price - position.entry_price)
            if position.side == "long"
            else max(0.0, position.entry_price - best_price)
        )
        mae_price = (
            max(0.0, position.entry_price - worst_price)
            if position.side == "long"
            else max(0.0, worst_price - position.entry_price)
        )
        realized_distance = (exit_price - position.entry_price) * direction
        holding_hours = max(0.0, (exit_time_ms - position.opened_at) / 3_600_000)
        entry_notional = position.entry_price * position.quantity
        exit_notional = exit_price * position.quantity
        gross_pnl = (exit_price - position.entry_price) * position.quantity * direction
        breakdown = costs.breakdown(
            position.entry_price,
            exit_price,
            position.quantity,
            holding_hours=holding_hours,
        )
        net_pnl = gross_pnl - breakdown.total_cost
        return cls(
            trade_id=uuid.uuid4().hex,
            exchange=exchange,
            symbol=symbol,
            mode=mode,
            side=position.side,
            entry_time=_iso_utc(position.opened_at),
            exit_time=_iso_utc(exit_time_ms),
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_notional=entry_notional,
            exit_notional=exit_notional,
            gross_pnl=gross_pnl,
            entry_fee=breakdown.entry_fee,
            exit_fee=breakdown.exit_fee,
            trading_fee=breakdown.trading_fee,
            slippage_cost=breakdown.slippage_cost,
            funding_fee=breakdown.funding_fee,
            total_cost=breakdown.total_cost,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl / entry_notional if entry_notional else 0.0,
            fee_ratio_pct=breakdown.trading_fee / entry_notional if entry_notional else 0.0,
            fee_to_gross_pct=breakdown.trading_fee / abs(gross_pnl) if gross_pnl else 0.0,
            total_cost_ratio_pct=breakdown.total_cost / entry_notional if entry_notional else 0.0,
            equity_before=equity_before,
            equity_after=equity_before + net_pnl,
            exit_reason=exit_reason,
            signal_score=signal.score if signal else 0,
            signal_reasons="|".join(signal.reasons) if signal else "",
            signal_timestamp_ms=signal.timestamp if signal else 0,
            initial_stop_price=initial_stop,
            take_profit_price=position.take_profit_price,
            best_price=best_price,
            worst_price=worst_price,
            mfe_price=mfe_price,
            mae_price=mae_price,
            mfe_r=mfe_price / risk_distance if risk_distance else 0.0,
            mae_r=mae_price / risk_distance if risk_distance else 0.0,
            realized_r=realized_distance / risk_distance if risk_distance else 0.0,
            holding_minutes=holding_hours * 60,
            environment=canonical_trade_environment(environment),
            model_name=signal.model_name if signal else "",
            model_version=signal.model_version if signal else "",
            meta_score=signal.meta_score if signal else 0.0,
            meta_threshold=signal.meta_threshold if signal else 0.0,
            meta_decision=signal.meta_decision if signal else "",
        )


class TradeReporter:
    """SQLite source of truth plus Excel-compatible CSV exports."""

    def __init__(self, report_dir: str | Path = "reports", timezone_name: str = "Asia/Shanghai") -> None:
        self.directory = Path(report_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database_path = self.directory / "trades.sqlite3"
        self.timezone = ZoneInfo(timezone_name)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()
        self.export_csv()

    def _create_schema(self) -> None:
        columns = ",\n".join(f"{field.name} {self._sqlite_type(field.type)}" for field in fields(TradeRecord))
        with self._connection:
            self._connection.execute(f"CREATE TABLE IF NOT EXISTS trades ({columns}, PRIMARY KEY (trade_id))")
            existing = {
                row[1]
                for row in self._connection.execute("PRAGMA table_info(trades)").fetchall()
            }
            for field in fields(TradeRecord):
                if field.name not in existing:
                    self._connection.execute(
                        f"ALTER TABLE trades ADD COLUMN {field.name} {self._sqlite_type(field.type)}"
                    )
            # All rows created before environment tracking were OKX demo/paper
            # trades. Keep that historical meaning stable instead of inferring
            # it from whichever exchange is configured today.
            self._connection.execute(
                "UPDATE trades SET environment = 'testnet' "
                "WHERE environment IS NULL OR TRIM(environment) = ''"
            )
            self._connection.execute(
                "UPDATE trades SET environment = 'production' "
                "WHERE LOWER(environment) IN ('prod', 'mainnet', 'live')"
            )
            self._connection.execute(
                "UPDATE trades SET environment = 'testnet' "
                "WHERE LOWER(environment) IN ('demo', 'paper', 'sandbox')"
            )

    @staticmethod
    def _sqlite_type(annotation: Any) -> str:
        if annotation is int or annotation == "int":
            return "INTEGER"
        if annotation is float or annotation == "float":
            return "REAL"
        return "TEXT"

    def record_trade(self, record: TradeRecord) -> None:
        values = asdict(record)
        names = list(values)
        placeholders = ",".join("?" for _ in names)
        with self._lock:
            with self._connection:
                self._connection.execute(
                    f"INSERT OR REPLACE INTO trades ({','.join(names)}) VALUES ({placeholders})",
                    [values[name] for name in names],
                )
            self.export_csv()

    def export_csv(self) -> None:
        with self._lock:
            trade_rows = self._rows("SELECT * FROM trades ORDER BY exit_time, entry_time")
            self._write_csv("trade_report.csv", [field.name for field in fields(TradeRecord)], trade_rows)
            summary_headers = [
                "period", "trades", "wins", "losses", "win_rate", "entry_notional",
                "gross_pnl", "trading_fee", "slippage_cost", "funding_fee", "total_cost",
                "net_pnl", "net_pnl_pct", "fee_ratio_pct", "fee_to_gross_pct", "average_net_pnl",
                "max_win", "max_loss",
            ]
            self._write_csv("daily_summary.csv", summary_headers, self._summary_rows("day"))
            self._write_csv("monthly_summary.csv", summary_headers, self._summary_rows("month"))

    def _rows(self, query: str) -> list[dict[str, Any]]:
        cursor = self._connection.execute(query)
        names = [description[0] for description in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def query_trades(
        self,
        *,
        start_date: str = "",
        end_date: str = "",
        exchange: str = "",
        scope: str = "all",
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Query completed trades using local report dates, while storing UTC."""
        conditions: list[str] = []
        params: list[Any] = []
        if start_date:
            start = datetime.combine(date.fromisoformat(start_date), datetime_time.min, tzinfo=self.timezone)
            conditions.append("exit_time >= ?")
            params.append(start.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
        if end_date:
            end = datetime.combine(date.fromisoformat(end_date) + timedelta(days=1), datetime_time.min, tzinfo=self.timezone)
            conditions.append("exit_time < ?")
            params.append(end.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
        if exchange:
            conditions.append("exchange = ?")
            params.append(exchange)
        selected_scope = str(scope or "all").strip().lower()
        if selected_scope not in {"all", "testnet", "production"}:
            raise ValueError("scope must be all, testnet, or production")
        production_condition = (
            "(LOWER(COALESCE(environment, '')) = 'production' "
            "AND LOWER(COALESCE(mode, '')) = 'live')"
        )
        if selected_scope == "production":
            conditions.append(production_condition)
        elif selected_scope == "testnet":
            conditions.append(f"NOT {production_condition}")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        safe_limit = max(1, min(int(limit), 20000))
        cursor = self._connection.execute(
            f"SELECT * FROM trades{where} ORDER BY exit_time DESC, entry_time DESC LIMIT {safe_limit}",
            params,
        )
        names = [description[0] for description in cursor.description]
        return [self._trade_view(dict(zip(names, row))) for row in cursor.fetchall()]

    @staticmethod
    def _trade_view(row: dict[str, Any]) -> dict[str, Any]:
        view = dict(row)
        view["environment"] = canonical_trade_environment(view.get("environment"))
        view["trade_scope"] = trade_scope(view)
        view["exchange_environment_label"] = exchange_environment_label(
            view.get("exchange"),
            view.get("environment"),
        )
        return view

    def report_data(
        self,
        *,
        start_date: str = "",
        end_date: str = "",
        exchange: str = "",
        scope: str = "all",
    ) -> dict[str, Any]:
        rows = self.query_trades(
            start_date=start_date,
            end_date=end_date,
            exchange=exchange,
            scope=scope,
        )
        return self.report_data_from_rows(rows)

    def report_data_from_rows(self, trade_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        rows_by_id: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(trade_rows):
            view = self._trade_view(row)
            identity = str(view.get("trade_id") or f"row-{index}")
            rows_by_id[identity] = view
        rows = sorted(
            rows_by_id.values(),
            key=lambda row: (str(row.get("exit_time") or ""), str(row.get("entry_time") or "")),
            reverse=True,
        )
        summary_headers = [
            "period", "trades", "wins", "losses", "win_rate", "entry_notional",
            "gross_pnl", "trading_fee", "slippage_cost", "funding_fee", "total_cost",
            "net_pnl", "net_pnl_pct", "fee_ratio_pct", "fee_to_gross_pct", "average_net_pnl",
            "max_win", "max_loss",
        ]
        daily = self._summary_rows("day", rows)
        monthly = self._summary_rows("month", rows)
        net_pnl = sum(float(row.get("net_pnl") or 0) for row in rows)
        gross_pnl = sum(float(row.get("gross_pnl") or 0) for row in rows)
        total_cost = sum(float(row.get("total_cost") or 0) for row in rows)
        trading_fee = sum(float(row.get("trading_fee") or 0) for row in rows)
        notional = sum(float(row.get("entry_notional") or 0) for row in rows)
        return {
            "trades": rows,
            "daily": [{key: row.get(key, 0) for key in summary_headers} for row in daily],
            "monthly": [{key: row.get(key, 0) for key in summary_headers} for row in monthly],
            "stats": {
                "trades": len(rows),
                "wins": sum(1 for row in rows if float(row.get("net_pnl") or 0) > 0),
                "losses": sum(1 for row in rows if float(row.get("net_pnl") or 0) <= 0),
                "gross_pnl": gross_pnl,
                "total_cost": total_cost,
                "trading_fee": trading_fee,
                "net_pnl": net_pnl,
                "net_pnl_pct": net_pnl / notional if notional else 0.0,
                "fee_ratio_pct": trading_fee / notional if notional else 0.0,
            },
        }

    def notification_summary(self, report_date: str | date, *, exchange: str = "") -> dict[str, Any]:
        """Return daily and cumulative completed-trade statistics for email."""
        selected_date = report_date.isoformat() if isinstance(report_date, date) else str(report_date)
        date.fromisoformat(selected_date)
        with self._lock:
            daily_rows = self.query_trades(
                start_date=selected_date,
                end_date=selected_date,
                exchange=exchange,
                limit=20_000,
            )
            cumulative_rows = self.query_trades(exchange=exchange, limit=20_000)
        return {
            "date": selected_date,
            "daily": self._notification_stats(daily_rows),
            "cumulative": self._notification_stats(cumulative_rows),
        }

    @staticmethod
    def _notification_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        trades = len(rows)
        wins = sum(1 for row in rows if float(row["net_pnl"]) > 0)
        losses = trades - wins
        gross_pnl = sum(float(row["gross_pnl"]) for row in rows)
        total_cost = sum(float(row["total_cost"]) for row in rows)
        net_pnl = sum(float(row["net_pnl"]) for row in rows)
        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / trades if trades else 0.0,
            "gross_pnl": gross_pnl,
            "total_cost": total_cost,
            "net_pnl": net_pnl,
        }

    def _summary_rows(self, granularity: str, trade_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        if granularity not in {"day", "month"}:
            raise ValueError("granularity must be day or month")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in (trade_rows if trade_rows is not None else self._rows("SELECT * FROM trades ORDER BY exit_time")):
            exit_time = datetime.fromisoformat(row["exit_time"].replace("Z", "+00:00")).astimezone(self.timezone)
            period = exit_time.strftime("%Y-%m-%d" if granularity == "day" else "%Y-%m")
            grouped.setdefault(period, []).append(row)
        rows: list[dict[str, Any]] = []
        for period, items in grouped.items():
            net_values = [float(item["net_pnl"]) for item in items]
            row = {
                "period": period,
                "trades": len(items),
                "wins": sum(1 for value in net_values if value > 0),
                "losses": sum(1 for value in net_values if value <= 0),
                "entry_notional": sum(float(item["entry_notional"]) for item in items),
                "gross_pnl": sum(float(item["gross_pnl"]) for item in items),
                "trading_fee": sum(float(item["trading_fee"]) for item in items),
                "slippage_cost": sum(float(item["slippage_cost"]) for item in items),
                "funding_fee": sum(float(item["funding_fee"]) for item in items),
                "total_cost": sum(float(item["total_cost"]) for item in items),
                "net_pnl": sum(net_values),
                "max_win": max(net_values),
                "max_loss": min(net_values),
                "average_net_pnl": sum(net_values) / len(net_values),
            }
            rows.append(row)
        rows.sort(key=lambda row: row["period"])
        for row in rows:
            trades = row["trades"] or 0
            entry_notional = row["entry_notional"] or 0.0
            gross_pnl = row["gross_pnl"] or 0.0
            row["win_rate"] = (row["wins"] or 0) / trades if trades else 0.0
            row["net_pnl_pct"] = (row["net_pnl"] or 0.0) / entry_notional if entry_notional else 0.0
            row["fee_ratio_pct"] = (row["trading_fee"] or 0.0) / entry_notional if entry_notional else 0.0
            row["fee_to_gross_pct"] = (row["trading_fee"] or 0.0) / abs(gross_pnl) if gross_pnl else 0.0
        return rows

    def _write_csv(self, filename: str, headers: Iterable[str], rows: list[dict[str, Any]]) -> None:
        target = self.directory / filename
        # Dashboard report requests can overlap in the threaded HTTP server.
        # A unique temporary name prevents concurrent exports from fighting
        # over one shared .tmp file on Windows.
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        with _export_lock(target):
            try:
                with temporary.open("w", encoding="utf-8-sig", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(headers), extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                try:
                    os.replace(temporary, target)
                except PermissionError:
                    # Excel or antivirus can briefly hold the existing CSV on
                    # Windows. SQLite has already committed the trade, so keep
                    # the previous export and retry on the next refresh.
                    if not target.exists():
                        raise
            finally:
                if temporary.exists():
                    temporary.unlink()

    def close(self) -> None:
        with self._lock:
            self._connection.close()
