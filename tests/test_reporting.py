from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from btc_futures_bot.costs import CostConfig
from btc_futures_bot.models import Position, Signal
from btc_futures_bot.reporting import TradeRecord, TradeReporter


def test_reporter_writes_detail_and_aggregates(tmp_path: Path) -> None:
    position = Position(
        "long",
        1.0,
        100.0,
        95.0,
        107.5,
        1763200000000,
        initial_stop_price=95.0,
        best_price=108.0,
        worst_price=98.0,
    )
    record = TradeRecord.from_position(
        exchange="test",
        symbol="BTC-USDT",
        mode="paper",
        position=position,
        exit_price=107.5,
        exit_time_ms=1763203600000,
        exit_reason="take_profit",
        costs=CostConfig(),
        equity_before=10_000,
        signal=Signal("long", 6, 1763200000000, ("4h_uptrend", "1m_breakout")),
    )
    reporter = TradeReporter(tmp_path)
    reporter.record_trade(record)
    reporter.close()

    detail = list(csv.DictReader((tmp_path / "trade_report.csv").open(encoding="utf-8-sig")))
    daily = list(csv.DictReader((tmp_path / "daily_summary.csv").open(encoding="utf-8-sig")))
    monthly = list(csv.DictReader((tmp_path / "monthly_summary.csv").open(encoding="utf-8-sig")))
    assert len(detail) == len(daily) == len(monthly) == 1
    assert float(detail[0]["trading_fee"]) > 0
    assert float(detail[0]["net_pnl"]) < float(detail[0]["gross_pnl"])
    assert float(detail[0]["mfe_r"]) == 1.6
    assert float(detail[0]["mae_r"]) == 0.4
    assert float(detail[0]["holding_minutes"]) == 60.0
    assert detail[0]["environment"] == "testnet"
    assert daily[0]["trades"] == "1"


def test_report_uses_actual_holding_time_for_funding() -> None:
    opened_at = 1763200000000
    position = Position(
        "short",
        1.0,
        100.0,
        101.0,
        97.5,
        opened_at,
        initial_stop_price=101.0,
        best_price=99.0,
        worst_price=100.5,
    )

    record = TradeRecord.from_position(
        exchange="test",
        symbol="BTC-USDT",
        mode="paper",
        position=position,
        exit_price=99.0,
        exit_time_ms=opened_at + int(8.1 * 3_600_000),
        exit_reason="trend_invalidation",
        costs=CostConfig(expected_holding_hours=0.1, funding_rate_pct_per_8h=0.0001),
        equity_before=10_000,
    )

    assert record.funding_fee == 0.01
    assert record.holding_minutes == 486.0


def test_report_uses_confirmed_usdt_entry_fee_including_zero() -> None:
    costs = CostConfig(
        taker_fee_pct=0.001,
        slippage_pct=0.0,
        funding_rate_pct_per_8h=0.0,
    )
    position = Position(
        "long",
        1.0,
        100.0,
        95.0,
        110.0,
        1_763_200_000_000,
        entry_fee=0.0,
        entry_fee_asset="USDT",
    )

    record = TradeRecord.from_position(
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
        position=position,
        exit_price=101.0,
        exit_time_ms=position.opened_at + 60_000,
        exit_reason="trend_invalidation",
        costs=costs,
        equity_before=100.0,
        environment="production",
    )

    assert record.entry_fee == 0.0
    assert record.exit_fee == 0.101
    assert record.trading_fee == 0.101


def test_report_estimates_entry_fee_when_actual_commission_is_not_usdt() -> None:
    costs = CostConfig(
        taker_fee_pct=0.001,
        slippage_pct=0.0,
        funding_rate_pct_per_8h=0.0,
    )
    position = Position(
        "long",
        1.0,
        100.0,
        95.0,
        110.0,
        1_763_200_000_000,
        entry_fee=2.5,
        entry_fee_asset="BNB",
    )

    record = TradeRecord.from_position(
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
        position=position,
        exit_price=101.0,
        exit_time_ms=position.opened_at + 60_000,
        exit_reason="trend_invalidation",
        costs=costs,
        equity_before=100.0,
        environment="production",
    )

    assert record.entry_fee == 0.1
    assert record.entry_fee != position.entry_fee


def test_reporter_tracks_environment_and_filters_real_trades(tmp_path: Path) -> None:
    position = Position(
        "long",
        1.0,
        100.0,
        95.0,
        107.5,
        1763200000000,
        initial_stop_price=95.0,
    )
    test_record = TradeRecord.from_position(
        exchange="okx",
        symbol="BTC-USDT-SWAP",
        mode="paper",
        environment="demo",
        position=position,
        exit_price=101.0,
        exit_time_ms=1763203600000,
        exit_reason="take_profit",
        costs=CostConfig(),
        equity_before=10_000,
    )
    production_record = TradeRecord.from_position(
        exchange="binance",
        symbol="BTCUSDT",
        mode="live",
        environment="production",
        position=position,
        exit_price=102.0,
        exit_time_ms=1763207200000,
        exit_reason="take_profit",
        costs=CostConfig(),
        equity_before=10_000,
    )
    reporter = TradeReporter(tmp_path)
    reporter.record_trade(test_record)
    reporter.record_trade(production_record)

    all_rows = reporter.query_trades(scope="all")
    test_rows = reporter.query_trades(scope="testnet")
    production_rows = reporter.query_trades(scope="production")

    assert len(all_rows) == 2
    assert len(test_rows) == len(production_rows) == 1
    assert test_rows[0]["exchange_environment_label"] == "OKX-测试网"
    assert test_rows[0]["trade_scope"] == "testnet"
    assert production_rows[0]["exchange_environment_label"] == "币安-正式网络"
    assert production_rows[0]["trade_scope"] == "production"
    reporter.close()


def test_reporter_backfills_missing_historical_environment(tmp_path: Path) -> None:
    position = Position("short", 1.0, 100.0, 105.0, 95.0, 1763200000000)
    record = TradeRecord.from_position(
        exchange="okx",
        symbol="BTC-USDT-SWAP",
        mode="paper",
        position=position,
        exit_price=99.0,
        exit_time_ms=1763203600000,
        exit_reason="take_profit",
        costs=CostConfig(),
        equity_before=10_000,
    )
    reporter = TradeReporter(tmp_path)
    reporter.record_trade(record)
    with reporter._connection:
        reporter._connection.execute("UPDATE trades SET environment = NULL")
    reporter.close()

    reopened = TradeReporter(tmp_path)
    rows = reopened.query_trades()
    assert rows[0]["environment"] == "testnet"
    assert rows[0]["exchange_environment_label"] == "OKX-测试网"
    reopened.close()


def test_parallel_csv_exports_use_independent_temporary_files(tmp_path: Path) -> None:
    first = TradeReporter(tmp_path)
    second = TradeReporter(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda index: (first if index % 2 else second).export_csv(), range(20)))

    assert (tmp_path / "trade_report.csv").exists()
    assert list(tmp_path.glob("*.tmp")) == []
    first.close()
    second.close()
