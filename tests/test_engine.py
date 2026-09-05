from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.costs import CostConfig
from btc_futures_bot.http_client import ApiError
from btc_futures_bot.models import Candle, Position, Signal
from btc_futures_bot.reporting import TradeRecord, TradeReporter
from btc_futures_bot.risk import RiskConfig, RiskManager
from btc_futures_bot.strategy import MultiTimeframeStrategy, StrategyConfig


def test_live_engine_restores_timed_loss_streak_after_restart(tmp_path: object) -> None:
    reporter = TradeReporter(tmp_path)
    now_ms = int(time.time() * 1000)
    costs = CostConfig(maker_fee_pct=0.0, taker_fee_pct=0.0, slippage_pct=0.0, funding_rate_pct_per_8h=0.0)
    try:
        for index in range(3):
            opened_at = now_ms - (3 - index) * 120_000
            position = Position(
                "long",
                1.0,
                100.0,
                99.0,
                102.5,
                opened_at,
                initial_stop_price=99.0,
                best_price=100.0,
                worst_price=99.0,
            )
            reporter.record_trade(
                TradeRecord.from_position(
                    exchange="binance",
                    symbol="BTCUSDT",
                    mode="live",
                    environment="production",
                    position=position,
                    exit_price=99.0,
                    exit_time_ms=opened_at + 60_000,
                    exit_reason="stop_loss",
                    costs=costs,
                    equity_before=100.0,
                    signal=Signal("long", 6, opened_at, ("test",)),
                )
            )

        adapter = SimpleNamespace(
            name="binance",
            settings=SimpleNamespace(environment="production"),
        )
        engine = TradingEngine(
            adapter,
            MultiTimeframeStrategy(StrategyConfig()),
            RiskManager(
                RiskConfig(
                    max_consecutive_losses=3,
                    cooldown_minutes=5,
                    loss_streak_pause_minutes=720,
                )
            ),
            EngineConfig(mode="live"),
            reporter=reporter,
        )

        assert engine.consecutive_losses == 3
        assert engine.cooldown_until > time.time()
        assert not engine._entry_allowed()

        engine.cooldown_until = time.time() - 1
        assert engine._entry_allowed()
        assert engine.consecutive_losses == 0
    finally:
        reporter.close()


def test_zero_loss_streak_threshold_never_blocks_entry() -> None:
    engine = TradingEngine(
        SimpleNamespace(name="test"),
        MultiTimeframeStrategy(StrategyConfig()),
        RiskManager(
            RiskConfig(
                max_consecutive_losses=0,
                cooldown_minutes=0,
                loss_streak_pause_minutes=720,
            )
        ),
        EngineConfig(mode="paper"),
    )
    engine.consecutive_losses = 100
    engine.cooldown_until = time.time() - 1

    assert engine._entry_allowed()
    assert engine.consecutive_losses == 100


def test_live_position_reconciliation_is_throttled_between_poll_cycles() -> None:
    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        def __init__(self) -> None:
            self.position_reads = 0

        def fetch_live_position(self) -> None:
            self.position_reads += 1
            return None

    adapter = Adapter()
    engine = TradingEngine(
        adapter,
        MultiTimeframeStrategy(StrategyConfig()),
        RiskManager(),
        EngineConfig(mode="live", live_reconciliation_seconds=5),
    )
    candle = Candle(1, 100.0, 100.0, 100.0, 100.0, 1.0)

    with patch("btc_futures_bot.engine.time.monotonic", side_effect=(100.0, 102.0, 105.1)):
        engine._reconcile_binance_live_position_if_due(candle)
        engine._reconcile_binance_live_position_if_due(candle)
        engine._reconcile_binance_live_position_if_due(candle)

    assert adapter.position_reads == 2


def test_market_candles_are_cached_at_timeframe_specific_refresh_rate() -> None:
    class Adapter:
        name = "test"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="testnet")

        def __init__(self) -> None:
            self.candle_reads = 0

        def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
            self.candle_reads += 1
            return [Candle(self.candle_reads, 100.0, 100.0, 100.0, 100.0, 1.0)]

    adapter = Adapter()
    engine = TradingEngine(
        adapter,
        MultiTimeframeStrategy(StrategyConfig()),
        RiskManager(),
        EngineConfig(mode="paper", candle_refresh_seconds={"1m": 1.0}),
    )

    with patch("btc_futures_bot.engine.time.monotonic", side_effect=(100.0, 100.5, 101.1)):
        first = engine._fetch_candles_cached("1m", "1m")
        cached = engine._fetch_candles_cached("1m", "1m")
        refreshed = engine._fetch_candles_cached("1m", "1m")

    assert first is cached
    assert refreshed is not first
    assert adapter.candle_reads == 2


def _engine(*, enable_time_exit: bool) -> TradingEngine:
    strategy = MultiTimeframeStrategy(
        StrategyConfig(
            enable_time_exit=enable_time_exit,
            max_hold_seconds=1,
            hard_max_hold_seconds=1,
        )
    )
    return TradingEngine(
        SimpleNamespace(name="test"),
        strategy,
        RiskManager(),
        EngineConfig(mode="paper"),
    )


def _stale_position() -> Position:
    return Position(
        "long",
        1.0,
        100.0,
        99.0,
        102.0,
        int(time.time() * 1000) - 60_000,
        initial_stop_price=99.0,
        best_price=100.0,
    )


def test_time_exit_disabled_keeps_position_open() -> None:
    engine = _engine(enable_time_exit=False)
    engine.position = _stale_position()

    engine._manage_paper_position(Candle(1, 100.0, 100.1, 99.9, 100.0, 1.0))

    assert engine.position is not None


def test_time_exit_can_be_reenabled_for_comparison() -> None:
    engine = _engine(enable_time_exit=True)
    engine.position = _stale_position()

    engine._manage_paper_position(Candle(1, 100.0, 100.1, 99.9, 100.0, 1.0))

    assert engine.position is None


def test_profit_trend_invalidation_closes_only_after_one_r_mfe() -> None:
    class InvalidatedStrategy:
        config = StrategyConfig(
            mode="traditional_kline",
            trigger_timeframe="5m",
            enable_profit_trend_exit=True,
            profit_trend_exit_trigger_r=1.0,
        )

        @staticmethod
        def position_trend_invalidated(side: str, candles_by_timeframe: object) -> bool:
            return True

    engine = TradingEngine(
        SimpleNamespace(name="test"),
        InvalidatedStrategy(),
        RiskManager(),
        EngineConfig(mode="paper"),
    )
    engine.position = Position(
        "long",
        1.0,
        100.0,
        99.0,
        102.5,
        int(time.time() * 1000) - 60_000,
        initial_stop_price=99.0,
        best_price=100.9,
    )

    engine._manage_paper_position(Candle(1, 100.9, 101.1, 100.7, 100.8, 1.0), {"5m": []})

    assert engine.position is None


def test_profit_trend_invalidation_waits_for_one_r_mfe() -> None:
    class InvalidatedStrategy:
        config = StrategyConfig(
            mode="traditional_kline",
            trigger_timeframe="5m",
            enable_profit_trend_exit=True,
            profit_trend_exit_trigger_r=1.0,
        )

        @staticmethod
        def position_trend_invalidated(side: str, candles_by_timeframe: object) -> bool:
            return True

    engine = TradingEngine(
        SimpleNamespace(name="test"),
        InvalidatedStrategy(),
        RiskManager(),
        EngineConfig(mode="paper"),
    )
    engine.position = Position(
        "long",
        1.0,
        100.0,
        99.0,
        102.5,
        int(time.time() * 1000) - 60_000,
        initial_stop_price=99.0,
        best_price=100.8,
    )

    engine._manage_paper_position(Candle(1, 100.8, 100.9, 100.7, 100.8, 1.0), {"5m": []})

    assert engine.position is not None


def test_paper_entry_uses_latest_execution_close_not_stale_trigger_close() -> None:
    class Adapter:
        name = "test"
        settings = SimpleNamespace(symbol="BTC-USDT")

        def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
            closed_price = {"5m": 100.0, "1m": 105.0, "1h": 101.0}[interval]
            interval_ms = {"5m": 300_000, "1m": 60_000, "1h": 3_600_000}[interval]
            return [
                Candle(0, closed_price, closed_price + 1, closed_price - 1, closed_price, 10),
                Candle(interval_ms, closed_price, closed_price + 1, closed_price - 1, closed_price, 10),
            ]

    class FixedStrategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("long", 6, 1, ("fixed",))

    engine = TradingEngine(Adapter(), FixedStrategy(), RiskManager(), EngineConfig(mode="paper"))

    result = engine.evaluate_once()

    assert result.position is not None
    assert result.position.entry_price == 105.0


def test_paper_entry_candle_is_not_reprocessed_on_next_poll() -> None:
    class Adapter:
        name = "test"
        settings = SimpleNamespace(symbol="BTC-USDT")

        def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
            interval_ms = {"5m": 300_000, "1m": 60_000, "1h": 3_600_000}[interval]
            if interval == "1m":
                return [
                    Candle(0, 100.0, 100.1, 99.9, 100.0, 10.0),
                    Candle(60_000, 100.0, 110.0, 90.0, 100.0, 10.0),
                    Candle(120_000, 100.0, 100.1, 99.9, 100.0, 1.0),
                ]
            return [
                Candle(0, 100.0, 100.1, 99.9, 100.0, 10.0),
                Candle(interval_ms, 100.0, 100.1, 99.9, 100.0, 1.0),
            ]

    class FixedLongStrategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("long", 6, 1, ("fixed",))

    engine = TradingEngine(Adapter(), FixedLongStrategy(), RiskManager(), EngineConfig(mode="paper"))

    first = engine.evaluate_once()
    second = engine.evaluate_once()

    assert first.position is not None
    assert engine.last_position_candle_timestamp == 60_000
    assert second.position is not None
    assert engine.position is not None


def test_protected_stop_waits_for_a_new_closed_execution_candle() -> None:
    class Adapter:
        name = "test"
        settings = SimpleNamespace(symbol="BTC-USDT")

        def __init__(self) -> None:
            self.advanced = False

        def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
            interval_ms = {"5m": 300_000, "1m": 60_000, "1h": 3_600_000}[interval]
            if interval == "1m":
                rows = [
                    Candle(0, 100.0, 100.1, 99.9, 100.0, 10.0),
                    Candle(60_000, 100.0, 101.2, 99.8, 101.1, 10.0),
                ]
                if self.advanced:
                    rows.append(Candle(120_000, 101.1, 101.2, 100.0, 100.5, 10.0))
                    rows.append(Candle(180_000, 100.5, 100.6, 100.4, 100.5, 1.0))
                else:
                    rows.append(Candle(120_000, 101.1, 101.2, 101.0, 101.1, 1.0))
                return rows
            return [
                Candle(0, 100.0, 100.1, 99.9, 100.0, 10.0),
                Candle(interval_ms, 100.0, 100.1, 99.9, 100.0, 1.0),
            ]

    class FlatStrategy:
        config = StrategyConfig(
            trigger_timeframe="5m",
            regime_timeframe="1h",
            break_even_trigger_r=1.0,
            break_even_lock_r=0.5,
            enable_profit_trend_exit=False,
        )

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("flat", 0, 1, ("fixed_flat",))

    adapter = Adapter()
    engine = TradingEngine(adapter, FlatStrategy(), RiskManager(), EngineConfig(mode="paper"))
    engine.position = Position(
        "long",
        1.0,
        100.0,
        99.0,
        103.0,
        int(time.time() * 1000) - 60_000,
        initial_stop_price=99.0,
        best_price=100.0,
    )

    engine.evaluate_once()
    protected_stop = engine.position.stop_price if engine.position is not None else 0.0
    engine.evaluate_once()

    assert protected_stop > 100.0
    assert engine.position is not None
    assert engine.position.stop_price == protected_stop

    adapter.advanced = True
    engine.evaluate_once()

    assert engine.position is None


def test_live_entry_is_reported_only_after_server_stop_is_confirmed() -> None:
    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        def __init__(self) -> None:
            self.events = []

        def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
            return [
                Candle(1, 100.0, 100.1, 99.9, 100.0, 10.0),
                Candle(2, 100.0, 100.1, 99.9, 100.0, 1.0),
            ]

        def fetch_live_position(self) -> None:
            return None

        def fetch_equity(self) -> float:
            return 10_000.0

        def normalize_order_quantity(self, quantity: float, reference_price: float) -> float:
            return quantity

        def place_market_order(self, request: object) -> dict:
            self.events.append("entry")
            return {"orderId": 11, "status": "FILLED", "executedQty": "1", "avgPrice": "100"}

        def market_fill(self, payload: dict, *, fallback_price: float) -> tuple[float, float]:
            return float(payload["executedQty"]), float(payload["avgPrice"])

        def place_protection_orders(self, position: Position, **kwargs: object) -> dict:
            self.events.append("protection")
            return {
                "stop": {"algoId": 21},
                "take_profit": None,
                "confirmed": True,
            }

    class Strategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("long", 6, 1, ("fixed",))

    adapter = Adapter()
    risk = RiskManager(RiskConfig(stop_loss_pct=0.005))
    engine = TradingEngine(adapter, Strategy(), risk, EngineConfig(mode="live"))

    result = engine.evaluate_once()

    assert adapter.events == ["entry", "protection"]
    assert result.status == "live_order_protected"
    assert result.position is not None
    assert result.position.stop_order_id == "21"
    assert result.position.take_profit_order_id == ""
    assert result.position.take_profit_client_id == ""
    assert result.raw["server_side_stop_confirmed"] is True
    assert result.raw["server_side_take_profit_enabled"] is False
    assert result.raw["profit_exit_mode"] == "dynamic_trend_and_trailing"
    assert result.raw["risk_exit_mode"] == "dynamic_adverse_soft_stop"


def test_binance_live_reconciliation_accepts_stop_only_position() -> None:
    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        @staticmethod
        def fetch_live_position() -> dict:
            return {
                "side": "long",
                "quantity": 0.001,
                "entry_price": 100.0,
                "mark_price": 101.0,
            }

        @staticmethod
        def emergency_close(position: Position, *, client_id: str) -> dict:
            raise AssertionError("a stop-protected position must not be closed")

    engine = TradingEngine(
        Adapter(),
        MultiTimeframeStrategy(StrategyConfig()),
        RiskManager(),
        EngineConfig(mode="live"),
    )
    engine.position = Position(
        "long",
        0.001,
        100.0,
        99.0,
        102.5,
        1,
        initial_stop_price=99.0,
        stop_order_id="21",
        stop_client_id="btcbot-stop-test",
        take_profit_order_id="",
        take_profit_client_id="",
    )

    engine._reconcile_binance_live_position(Candle(2, 101.0, 101.0, 101.0, 101.0, 1.0))

    assert engine.position is not None
    assert engine.position.stop_order_id == "21"
    assert engine.position.take_profit_order_id == ""


def test_binance_unmanaged_position_emails_open_and_close_once_across_restart(tmp_path) -> None:
    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        def __init__(self) -> None:
            self.remote = {
                "side": "long",
                "quantity": 0.001,
                "entry_price": 78_000.0,
                "mark_price": 78_100.0,
            }

        def fetch_live_position(self) -> dict | None:
            return self.remote

        @staticmethod
        def fetch_latest_close_fill(position: Position, *, after_ms: int = 0) -> dict:
            assert position.side == "long"
            assert after_ms > 0
            return {
                "quantity": 0.001,
                "price": 79_000.0,
                "realized_pnl": 1.0,
                "commission": 0.0314,
                "commission_assets": ["USDT"],
                "timestamp": after_ms + 30_000,
                "source": "test userTrades",
            }

    class Notifier:
        def __init__(self) -> None:
            self.opened = []
            self.closed = []

        def notify_reconciled_open(self, **context: object) -> bool:
            self.opened.append(context)
            return True

        def notify_reconciled_close(self, **context: object) -> bool:
            self.closed.append(context)
            return True

    state_path = tmp_path / "live_reconciliation_state.json"
    adapter = Adapter()
    first_notifier = Notifier()
    first_engine = TradingEngine(
        adapter,
        MultiTimeframeStrategy(StrategyConfig()),
        RiskManager(),
        EngineConfig(mode="live", reconciliation_state_path=str(state_path)),
        notifier=first_notifier,
    )
    candle = Candle(1, 78_100.0, 78_100.0, 78_100.0, 78_100.0, 1.0)

    try:
        first_engine._reconcile_binance_live_position(candle)
    except RuntimeError as error:
        assert "unmanaged Binance position" in str(error)
    else:
        raise AssertionError("unmanaged live position must still block the engine")
    assert len(first_notifier.opened) == 1

    try:
        first_engine._reconcile_binance_live_position(candle)
    except RuntimeError:
        pass
    assert len(first_notifier.opened) == 1

    second_notifier = Notifier()
    second_engine = TradingEngine(
        adapter,
        MultiTimeframeStrategy(StrategyConfig()),
        RiskManager(),
        EngineConfig(mode="live", reconciliation_state_path=str(state_path)),
        notifier=second_notifier,
    )
    try:
        second_engine._reconcile_binance_live_position(candle)
    except RuntimeError:
        pass
    assert second_notifier.opened == []

    adapter.remote = None
    second_engine._reconcile_binance_live_position(candle)
    assert len(second_notifier.closed) == 1
    assert second_notifier.closed[0]["exit_price"] == 79_000.0
    assert second_notifier.closed[0]["realized_pnl"] == 1.0
    assert second_engine.unmanaged_live_position is None


def test_binance_managed_position_is_restored_after_safe_restart(tmp_path) -> None:
    state_path = tmp_path / "live_reconciliation_state.json"
    managed = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "environment": "production",
        "position": {
            "side": "short",
            "quantity": 0.001,
            "entry_price": 77_852.6,
            "stop_price": 78_203.0,
            "take_profit_price": 76_976.6,
            "opened_at": 1_788_435_361_907,
            "initial_stop_price": 78_203.0,
            "best_price": 77_841.8,
            "stop_reason": "stop_loss",
            "worst_price": 77_976.0,
            "entry_order_id": "10",
            "entry_client_id": "btcbot-entry-test",
            "stop_order_id": "20",
            "stop_client_id": "btcbot-stop-test",
            "take_profit_order_id": "",
            "take_profit_client_id": "",
        },
        "signal": {
            "side": "short",
            "score": 6,
            "timestamp": 1_788_435_300_000,
            "reasons": ["1m_ultra_short_reversal_short"],
        },
        "position_equity_before": 25.5,
        "last_position_candle_timestamp": 1_788_435_300_000,
    }
    state_path.write_text(
        json.dumps({"unmanaged_position": None, "managed_position": managed}),
        encoding="utf-8",
    )

    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        def prepare_live(self, *, max_leverage: float, managed_position: dict | None = None):
            assert max_leverage == 3
            assert managed_position == managed
            return {"prepared": True, "resumed": True, "flat": False}

        def close(self) -> None:
            return None

    engine = TradingEngine(
        Adapter(),
        MultiTimeframeStrategy(StrategyConfig()),
        RiskManager(),
        EngineConfig(mode="live", reconciliation_state_path=str(state_path)),
    )

    preflight = engine.prepare_live()

    assert preflight["resumed"] is True
    assert engine.position is not None
    assert engine.position.side == "short"
    assert engine.position.stop_client_id == "btcbot-stop-test"
    assert engine.position.best_price == 77_841.8
    assert engine.position_signal is not None
    assert engine.position_signal.reasons == ("1m_ultra_short_reversal_short",)
    engine.close()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["managed_position"]["position"]["stop_order_id"] == "20"


def test_live_entry_reconciles_only_after_protected_state_is_saved(tmp_path) -> None:
    state_path = tmp_path / "live_reconciliation_state.json"
    fill_timestamp = 1_788_525_001_234

    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        def __init__(self) -> None:
            self.events: list[str] = []
            self.requested_quantity = 0.0

        @staticmethod
        def fetch_candles(interval: str, limit: int) -> list[Candle]:
            return [
                Candle(1, 100.0, 100.1, 99.9, 100.0, 10.0),
                Candle(2, 100.0, 100.1, 99.9, 100.0, 1.0),
            ]

        @staticmethod
        def fetch_live_position() -> None:
            return None

        @staticmethod
        def fetch_equity() -> float:
            return 10_000.0

        @staticmethod
        def normalize_order_quantity(quantity: float, reference_price: float) -> float:
            return quantity

        def place_market_order(self, request: object) -> dict:
            self.events.append("entry")
            self.requested_quantity = float(request.quantity)
            return {
                "orderId": 11,
                "status": "FILLED",
                "executedQty": str(self.requested_quantity),
                "avgPrice": "100",
                "updateTime": 1_788_525_001_000,
            }

        @staticmethod
        def market_fill(payload: dict, *, fallback_price: float) -> tuple[float, float]:
            return float(payload["executedQty"]), float(payload["avgPrice"])

        def place_protection_orders(self, position: Position, **kwargs: object) -> dict:
            self.events.append("protection")
            return {"stop": {"algoId": 21}, "take_profit": None, "confirmed": True}

        def fetch_entry_fill(self, position: Position) -> dict:
            self.events.append("entry_fill")
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            persisted = saved["managed_position"]["position"]
            assert persisted["entry_order_id"] == "11"
            assert persisted["stop_order_id"] == "21"
            assert persisted["entry_price"] == 100.0
            assert persisted["opened_at"] == 1_788_525_001_000
            return {
                "side": "BUY",
                "quantity": position.quantity,
                "price": 100.25,
                "timestamp": fill_timestamp,
                "commission": 0.0,
                "commission_assets": ["USDT"],
            }

    class Strategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("long", 6, 1, ("fixed",))

    class Notifier:
        def __init__(self) -> None:
            self.opened: list[Position] = []

        def notify_open(self, position: Position, *args: object, **kwargs: object) -> None:
            self.opened.append(position)

    adapter = Adapter()
    notifier = Notifier()
    engine = TradingEngine(
        adapter,
        Strategy(),
        RiskManager(RiskConfig(stop_loss_pct=0.005)),
        EngineConfig(mode="live", reconciliation_state_path=str(state_path)),
        notifier=notifier,
    )

    result = engine.evaluate_once()

    assert result.status == "live_order_protected"
    assert adapter.events == ["entry", "protection", "entry_fill"]
    assert result.position is not None
    assert result.position.entry_price == 100.25
    assert result.position.opened_at == fill_timestamp
    assert result.position.entry_fee == 0.0
    assert result.position.entry_fee_asset == "USDT"
    assert notifier.opened == [result.position]
    saved_position = json.loads(state_path.read_text(encoding="utf-8"))["managed_position"]["position"]
    assert saved_position["opened_at"] == fill_timestamp
    assert saved_position["entry_fee"] == 0.0


def test_restart_entry_fill_failure_keeps_restored_protected_position_and_alerts(tmp_path) -> None:
    lookup_error = RuntimeError("entry trades are temporarily unavailable")
    state_path = tmp_path / "live_reconciliation_state.json"
    managed = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "environment": "production",
        "position": {
            "side": "long",
            "quantity": 0.001,
            "entry_price": 79_000.0,
            "stop_price": 78_600.0,
            "take_profit_price": 80_000.0,
            "opened_at": 1_788_525_000_000,
            "entry_order_id": "77",
            "stop_order_id": "88",
            "stop_client_id": "btcbot-stop-test",
        },
        "signal": None,
        "position_equity_before": 25.0,
        "last_position_candle_timestamp": 1_788_524_980_000,
    }
    state_path.write_text(
        json.dumps({"unmanaged_position": None, "managed_position": managed}),
        encoding="utf-8",
    )

    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        @staticmethod
        def prepare_live(*, max_leverage: float, managed_position: dict | None = None) -> dict:
            assert managed_position == managed
            return {"prepared": True, "resumed": True, "flat": False}

        @staticmethod
        def fetch_entry_fill(position: Position) -> dict:
            raise lookup_error

        @staticmethod
        def emergency_close(*args: object, **kwargs: object) -> None:
            raise AssertionError("entry reconciliation must not close exposure")

        @staticmethod
        def cancel_protection_orders(*args: object, **kwargs: object) -> None:
            raise AssertionError("entry reconciliation must not cancel protection")

    notifier = _EmergencyNotifier()
    engine = TradingEngine(
        Adapter(),
        _FixedLongSignalStrategy(),
        RiskManager(),
        EngineConfig(mode="live", reconciliation_state_path=str(state_path)),
        notifier=notifier,
    )

    preflight = engine.prepare_live()

    assert preflight["resumed"] is True
    assert engine.position is not None
    assert engine.position.opened_at == managed["position"]["opened_at"]
    assert engine.position.stop_order_id == "88"
    assert engine.position.entry_fee is None
    assert len(notifier.emergencies) == 1
    assert notifier.emergencies[0]["error"] is lookup_error
    assert notifier.emergencies[0]["category"] == "entry_reconciliation"
    assert notifier.emergencies[0]["incident"] == "managed_entry_fill:77"
    saved_position = json.loads(state_path.read_text(encoding="utf-8"))["managed_position"]["position"]
    assert saved_position["opened_at"] == managed["position"]["opened_at"]
    assert saved_position["stop_order_id"] == "88"


def test_safe_restart_backfills_exact_entry_metadata_after_restore(tmp_path) -> None:
    state_path = tmp_path / "live_reconciliation_state.json"
    original_opened_at = 1_788_525_000_000
    exact_opened_at = 1_788_525_001_234
    managed = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "environment": "production",
        "position": {
            "side": "short",
            "quantity": 0.001,
            "entry_price": 79_000.0,
            "stop_price": 79_500.0,
            "take_profit_price": 78_000.0,
            "opened_at": original_opened_at,
            "entry_order_id": "77",
            "entry_client_id": "btcbot-entry-test",
            "stop_order_id": "88",
            "stop_client_id": "btcbot-stop-test",
        },
        "signal": None,
        "position_equity_before": 25.0,
        "last_position_candle_timestamp": 1_788_524_980_000,
    }
    state_path.write_text(
        json.dumps({"unmanaged_position": None, "managed_position": managed}),
        encoding="utf-8",
    )

    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        @staticmethod
        def prepare_live(*, max_leverage: float, managed_position: dict | None = None) -> dict:
            assert managed_position == managed
            return {"prepared": True, "resumed": True, "flat": False}

        @staticmethod
        def fetch_entry_fill(position: Position) -> dict:
            persisted = json.loads(state_path.read_text(encoding="utf-8"))["managed_position"]["position"]
            assert persisted["opened_at"] == original_opened_at
            assert persisted["stop_order_id"] == "88"
            return {
                "side": "SELL",
                "quantity": position.quantity,
                "price": 79_012.5,
                "timestamp": exact_opened_at,
                "commission": 0.031605,
                "commission_assets": ["USDT"],
            }

    engine = TradingEngine(
        Adapter(),
        MultiTimeframeStrategy(StrategyConfig()),
        RiskManager(),
        EngineConfig(mode="live", reconciliation_state_path=str(state_path)),
    )

    preflight = engine.prepare_live()

    assert preflight["resumed"] is True
    assert engine.position is not None
    assert engine.position.entry_price == 79_012.5
    assert engine.position.opened_at == exact_opened_at
    assert engine.position.entry_fee == pytest.approx(0.031605)
    saved_position = json.loads(state_path.read_text(encoding="utf-8"))["managed_position"]["position"]
    assert saved_position["entry_price"] == 79_012.5
    assert saved_position["opened_at"] == exact_opened_at
    assert saved_position["entry_fee_asset"] == "USDT"


def test_pending_entry_fills_retry_without_alerts_or_blocking_risk_loop(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("btc_futures_bot.engine.time.monotonic", lambda: clock[0])

    class Adapter(_EmergencyAdapter):
        calls = 0

        def fetch_entry_fill(self, position: Position) -> dict | None:
            self.calls += 1
            if self.calls < 3:
                return None
            return {
                "side": "BUY", "quantity": position.quantity, "price": 100.0,
                "timestamp": 1_788_525_001_234, "commission": 0.04,
                "commission_assets": ["USDT"],
                "source": "Binance /fapi/v1/userTrades",
            }

    adapter = Adapter()
    notifier = _EmergencyNotifier()
    engine = _alert_test_engine(adapter, notifier)
    engine.position = Position("long", 1.0, 100.0, 99.0, 103.0, 123_456,
                               entry_order_id="77", stop_order_id="88")
    for timestamp in (100.0, 114.0, 115.0, 144.0):
        clock[0] = timestamp
        engine._reconcile_managed_entry_fill()
    assert adapter.calls == 2
    assert notifier.emergencies == []
    clock[0] = 145.0
    engine._reconcile_managed_entry_fill()
    assert engine.position.entry_fee == 0.04
    assert engine.position.stop_order_id == "88"
    clock[0] = 1_000.0
    engine._reconcile_managed_entry_fill()
    assert adapter.calls == 3
    assert notifier.emergencies == []
    assert ("ip_restricted", "") in notifier.resolved


def test_entry_fill_pending_timeout_is_reconciliation_alert(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("btc_futures_bot.engine.time.monotonic", lambda: clock[0])

    class Adapter(_EmergencyAdapter):
        @staticmethod
        def fetch_entry_fill(position: Position) -> None:
            return None

    notifier = _EmergencyNotifier()
    engine = _alert_test_engine(Adapter(), notifier)
    engine.position = Position("long", 1.0, 100.0, 99.0, 103.0, 123_456,
                               entry_order_id="77", stop_order_id="88")
    for timestamp in (100.0, 115.0, 145.0, 205.0, 325.0):
        clock[0] = timestamp
        engine._reconcile_managed_entry_fill()
    assert notifier.emergencies == []
    assert engine._entry_fill_next_at == 400.0
    clock[0] = 400.0
    engine._reconcile_managed_entry_fill()
    assert len(notifier.emergencies) == 1
    assert notifier.emergencies[0]["category"] == "entry_reconciliation"
    assert engine.position.stop_order_id == "88"


def test_entry_fill_rate_limit_alert_is_immediate_and_respects_ban(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("btc_futures_bot.engine.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("btc_futures_bot.http_client.time.time", lambda: 1_000.0)
    error = ApiError("IP banned", status_code=418, retry_at=1_600.0)

    class Adapter(_EmergencyAdapter):
        calls = 0

        def fetch_entry_fill(self, position: Position) -> None:
            self.calls += 1
            raise error

    adapter = Adapter()
    notifier = _EmergencyNotifier()
    engine = _alert_test_engine(adapter, notifier)
    engine.position = Position("long", 1.0, 100.0, 99.0, 103.0, 123_456,
                               entry_order_id="77", stop_order_id="88")
    engine._reconcile_managed_entry_fill()
    assert len(notifier.emergencies) == 1
    assert notifier.emergencies[0]["error"] is error
    clock[0] = 699.0
    engine._reconcile_managed_entry_fill()
    assert adapter.calls == 1
    assert engine.position.stop_order_id == "88"


def test_cli_loop_does_not_sleep_through_an_entire_ip_ban(monkeypatch) -> None:
    engine = _alert_test_engine(_EmergencyAdapter(), _EmergencyNotifier())
    calls = [0]
    waits = []

    def evaluate():
        calls[0] += 1
        if calls[0] == 1:
            raise ApiError("IP banned", status_code=418, retry_at=time.time() + 8_000)
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "evaluate_once", evaluate)
    monkeypatch.setattr("btc_futures_bot.engine.time.sleep", waits.append)
    with pytest.raises(KeyboardInterrupt):
        engine.run_forever()
    assert calls[0] == 2
    assert waits == [30.0]


def test_failed_live_preflight_does_not_erase_saved_managed_position(tmp_path) -> None:
    state_path = tmp_path / "live_reconciliation_state.json"
    managed = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "environment": "production",
        "position": {
            "side": "long",
            "quantity": 0.001,
            "entry_price": 81_119.1,
            "stop_price": 80_754.06405,
            "take_profit_price": 82_031.69,
            "opened_at": 1_788_513_122_845,
            "initial_stop_price": 80_754.06405,
            "best_price": 81_119.1,
            "stop_reason": "stop_loss",
            "worst_price": 81_119.1,
            "entry_order_id": "10",
            "entry_client_id": "btcbot-entry-test",
            "stop_order_id": "20",
            "stop_client_id": "btcbot-stop-test",
            "take_profit_order_id": "",
            "take_profit_client_id": "",
        },
        "signal": None,
        "position_equity_before": 25.0,
        "last_position_candle_timestamp": 1_788_513_120_000,
    }
    state_path.write_text(
        json.dumps({"unmanaged_position": None, "managed_position": managed}),
        encoding="utf-8",
    )

    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        @staticmethod
        def prepare_live(*, max_leverage: float, managed_position: dict | None = None):
            assert max_leverage == 3
            assert managed_position == managed
            raise RuntimeError("preflight rejected")

        @staticmethod
        def close() -> None:
            return None

    engine = TradingEngine(
        Adapter(),
        MultiTimeframeStrategy(StrategyConfig()),
        RiskManager(),
        EngineConfig(mode="live", reconciliation_state_path=str(state_path)),
    )

    with pytest.raises(RuntimeError, match="preflight rejected"):
        engine.prepare_live()
    engine.close()

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["managed_position"] == managed


def test_failed_managed_position_restore_does_not_erase_saved_state(tmp_path) -> None:
    state_path = tmp_path / "live_reconciliation_state.json"
    managed = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "environment": "production",
        "position": {
            "side": "long",
            "quantity": 0.001,
            "entry_price": 81_119.1,
            "stop_price": 80_754.06405,
            "take_profit_price": 82_031.689875,
            "opened_at": 1_788_513_092_000,
            "initial_stop_price": 80_754.06405,
            "best_price": 81_119.1,
            "stop_reason": "stop_loss",
            "worst_price": 81_119.1,
            "entry_order_id": "10",
            "entry_client_id": "btcbot-entry-test",
            "stop_order_id": "20",
            "stop_client_id": "btcbot-stop-test",
            "take_profit_order_id": "",
            "take_profit_client_id": "",
        },
        "signal": {"side": "long", "score": "invalid", "timestamp": 1},
        "position_equity_before": 25.0,
        "last_position_candle_timestamp": 0,
    }
    state_path.write_text(
        json.dumps({"unmanaged_position": None, "managed_position": managed}),
        encoding="utf-8",
    )

    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        @staticmethod
        def prepare_live(*, max_leverage: float, managed_position: dict | None = None):
            assert managed_position == managed
            return {"prepared": True, "resumed": True, "flat": False}

        @staticmethod
        def close() -> None:
            return None

    engine = TradingEngine(
        Adapter(),
        MultiTimeframeStrategy(StrategyConfig()),
        RiskManager(),
        EngineConfig(mode="live", reconciliation_state_path=str(state_path)),
    )

    with pytest.raises(RuntimeError, match="saved managed position is invalid"):
        engine.prepare_live()
    assert engine.position is not None
    engine.close()

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["managed_position"] == managed


def test_live_entry_rolls_back_when_stop_is_not_confirmed() -> None:
    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        def __init__(self) -> None:
            self.events = []

        def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
            return [
                Candle(1, 100.0, 100.1, 99.9, 100.0, 10.0),
                Candle(2, 100.0, 100.1, 99.9, 100.0, 1.0),
            ]

        def fetch_live_position(self) -> None:
            return None

        def fetch_equity(self) -> float:
            return 10_000.0

        def normalize_order_quantity(self, quantity: float, reference_price: float) -> float:
            return quantity

        def place_market_order(self, request: object) -> dict:
            self.events.append("entry")
            return {"orderId": 11, "status": "FILLED", "executedQty": "1", "avgPrice": "100"}

        def market_fill(self, payload: dict, *, fallback_price: float) -> tuple[float, float]:
            return float(payload["executedQty"]), float(payload["avgPrice"])

        def place_protection_orders(self, position: Position, **kwargs: object) -> dict:
            self.events.append("protection_failed")
            raise RuntimeError("stop was not confirmed open")

        def emergency_close(self, position: Position, *, client_id: str) -> dict:
            self.events.append("emergency_close")
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "99.9"}

        def cancel_protection_orders(self, position: Position) -> dict:
            self.events.append("cancel_protection")
            return {"cancelled": True}

    class Strategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("long", 6, 1, ("fixed",))

    adapter = Adapter()
    risk = RiskManager(RiskConfig(stop_loss_pct=0.005))
    engine = TradingEngine(adapter, Strategy(), risk, EngineConfig(mode="live"))

    result = engine.evaluate_once()

    assert result.status == "live_entry_rolled_back"
    assert result.position is None
    assert engine.position is None
    assert adapter.events == ["entry", "protection_failed", "emergency_close", "cancel_protection"]


def test_binance_client_ids_fit_exchange_limit() -> None:
    for prefix in ("entry", "stop", "tp", "close", "emergency-close", "emergency-retry"):
        client_id = TradingEngine._client_id(prefix)
        assert len(client_id) <= 36
        assert client_id.startswith(f"btcbot-{prefix}-")


def test_live_position_checks_mark_price_each_poll_and_closes_at_protected_stop(
    tmp_path,
) -> None:
    exchange_exit_time_ms = 1_788_552_003_195

    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        def __init__(self) -> None:
            self.closed = False
            self.mark_calls = 0
            self.close_request = None

        @staticmethod
        def fetch_candles(interval: str, limit: int) -> list[Candle]:
            return [
                Candle(1, 100.0, 100.1, 99.9, 100.0, 10.0),
                Candle(2, 100.0, 100.1, 99.9, 100.0, 1.0),
            ]

        def fetch_live_position(self) -> dict | None:
            if self.closed:
                return None
            return {"side": "long", "quantity": 1.0, "mark_price": 100.9}

        def fetch_mark_price(self) -> float:
            self.mark_calls += 1
            return 100.9

        def place_market_order(self, request: object) -> dict:
            self.close_request = request
            self.closed = True
            return {
                "status": "FILLED",
                "executedQty": "1",
                "avgPrice": "100.9",
                "updateTime": exchange_exit_time_ms,
            }

        @staticmethod
        def market_fill(payload: dict, *, fallback_price: float) -> tuple[float, float]:
            return float(payload["executedQty"]), float(payload["avgPrice"])

        @staticmethod
        def cancel_protection_orders(position: Position) -> dict:
            return {"cancelled": [position.stop_client_id, position.take_profit_client_id], "errors": []}

    class Strategy:
        config = StrategyConfig(
            trigger_timeframe="5m",
            regime_timeframe="1h",
            enable_profit_trend_exit=False,
        )

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("flat", 0, 1, ("no_entry",))

    adapter = Adapter()
    reporter = TradeReporter(tmp_path)
    engine = TradingEngine(
        adapter,
        Strategy(),
        RiskManager(),
        EngineConfig(mode="live"),
        reporter=reporter,
    )
    engine.position = Position(
        "long",
        1.0,
        100.0,
        101.0,
        103.0,
        int(time.time() * 1000) - 120_000,
        initial_stop_price=99.0,
        best_price=102.0,
        stop_reason="break_even_stop",
        stop_order_id="1",
        stop_client_id="stop",
        take_profit_order_id="2",
        take_profit_client_id="tp",
    )

    result = engine.evaluate_once()

    assert result.status == "live_active_exit"
    assert result.raw["exit_reason"] == "break_even_stop"
    assert adapter.mark_calls == 1
    assert adapter.close_request.reduce_only is True
    assert adapter.close_request.side == "sell"
    assert engine.position is None
    rows = reporter.query_trades(limit=1)
    assert rows[0]["exit_time"] == "2026-09-04T20:00:03Z"
    reporter.close()


def test_live_adverse_dynamic_exit_closes_locally_and_keeps_hard_stop_until_fill() -> None:
    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        def __init__(self) -> None:
            self.closed = False
            self.cancelled_stop_id = ""

        @staticmethod
        def fetch_candles(interval: str, limit: int) -> list[Candle]:
            return [
                Candle(1, 100.0, 100.1, 99.9, 100.0, 10.0),
                Candle(2, 100.0, 100.1, 99.9, 100.0, 10.0),
            ]

        def fetch_live_position(self) -> dict | None:
            return None if self.closed else {"side": "long", "quantity": 1.0, "mark_price": 95.0}

        @staticmethod
        def fetch_mark_price() -> float:
            return 95.0

        def place_market_order(self, request: object) -> dict:
            self.closed = True
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "95"}

        @staticmethod
        def market_fill(payload: dict, *, fallback_price: float) -> tuple[float, float]:
            return float(payload["executedQty"]), float(payload["avgPrice"])

        def cancel_protection_orders(self, position: Position) -> dict:
            self.cancelled_stop_id = position.stop_order_id
            return {"cancelled": [position.stop_client_id], "errors": []}

    class Strategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("flat", 0, 1, ("no_entry",))

    adapter = Adapter()
    engine = TradingEngine(adapter, Strategy(), RiskManager(), EngineConfig(mode="live"))
    engine.position = Position(
        "long",
        1.0,
        100.0,
        90.0,
        125.0,
        int(time.time() * 1000) - 120_000,
        initial_stop_price=90.0,
        best_price=100.0,
        stop_order_id="hard-stop-1",
        stop_client_id="hard-stop-client",
    )

    with patch(
        "btc_futures_bot.engine.adverse_dynamic_exit_reason",
        return_value="dynamic_stop_loss",
    ):
        result = engine.evaluate_once()

    assert result.status == "live_active_exit"
    assert result.raw["exit_reason"] == "dynamic_stop_loss"
    assert adapter.cancelled_stop_id == "hard-stop-1"
    assert engine.position is None


def test_live_close_race_reuses_confirmed_flat_position_without_second_query() -> None:
    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        def __init__(self) -> None:
            self.position_reads = 0

        @staticmethod
        def place_market_order(request: object) -> dict:
            raise RuntimeError("reduce-only order rejected because the venue is already flat")

        def fetch_live_position(self) -> None:
            self.position_reads += 1
            return None

        @staticmethod
        def fetch_protection_status(position: Position) -> dict:
            return {"stop": {"status": "FILLED", "avgPrice": "99.0"}}

        @staticmethod
        def cancel_protection_orders(position: Position) -> dict:
            return {"cancelled": [position.stop_client_id], "errors": []}

    adapter = Adapter()
    engine = TradingEngine(adapter, MultiTimeframeStrategy(), RiskManager(), EngineConfig(mode="live"))
    engine.position = Position(
        "long",
        1.0,
        100.0,
        99.0,
        103.0,
        int(time.time() * 1000) - 120_000,
        initial_stop_price=99.0,
        stop_order_id="1",
        stop_client_id="stop",
    )

    result = engine._close_live_position(98.9, "stop_loss")

    assert result["exchange_race_reconciled"] is True
    assert adapter.position_reads == 1
    assert engine.position is None


def test_live_minimum_order_fallback_uses_venue_minimum_when_capacity_allows() -> None:
    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        def __init__(self) -> None:
            self.entry_request = None

        @staticmethod
        def fetch_candles(interval: str, limit: int) -> list[Candle]:
            return [
                Candle(1, 100.0, 100.1, 99.9, 100.0, 10.0),
                Candle(2, 100.0, 100.1, 99.9, 100.0, 1.0),
            ]

        @staticmethod
        def fetch_live_position() -> None:
            return None

        @staticmethod
        def fetch_equity() -> float:
            return 1.0

        @staticmethod
        def normalize_order_quantity(quantity: float, reference_price: float) -> float:
            if quantity < 1.0:
                raise ValueError("Binance quantity is below minimum")
            return 1.0

        @staticmethod
        def fetch_dashboard_snapshot() -> dict:
            return {
                "private_available": True,
                "order_limits": {
                    "effective_min_quantity_raw": "1",
                    "effective_min_notional_raw": "100",
                    "estimated_max_open_quantity_raw": "2",
                    "estimated_max_open_notional_raw": "200",
                },
            }

        def place_market_order(self, request: object) -> dict:
            self.entry_request = request
            return {"orderId": 11, "status": "FILLED", "executedQty": "1", "avgPrice": "100"}

        @staticmethod
        def market_fill(payload: dict, *, fallback_price: float) -> tuple[float, float]:
            return float(payload["executedQty"]), float(payload["avgPrice"])

        @staticmethod
        def place_protection_orders(position: Position, **kwargs: object) -> dict:
            return {"stop": {"algoId": 21}, "take_profit": None, "confirmed": True}

    class Strategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("long", 6, 1, ("fixed",))

    adapter = Adapter()
    risk = RiskManager(
        RiskConfig(risk_per_trade=0.02, stop_loss_pct=0.005, max_notional_pct=0.3),
        max_leverage=20,
    )
    engine = TradingEngine(adapter, Strategy(), risk, EngineConfig(mode="live"))

    result = engine.evaluate_once()

    assert result.status == "live_order_protected"
    assert result.position is not None
    assert result.position.quantity == 1.0
    assert adapter.entry_request.quantity == 1.0
    assert result.raw["sizing"]["minimum_fallback_used"] is True


def test_live_minimum_order_fallback_emails_and_skips_when_capacity_is_too_low() -> None:
    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        @staticmethod
        def fetch_candles(interval: str, limit: int) -> list[Candle]:
            return [
                Candle(1, 100.0, 100.1, 99.9, 100.0, 10.0),
                Candle(2, 100.0, 100.1, 99.9, 100.0, 1.0),
            ]

        @staticmethod
        def fetch_live_position() -> None:
            return None

        @staticmethod
        def fetch_equity() -> float:
            return 1.0

        @staticmethod
        def normalize_order_quantity(quantity: float, reference_price: float) -> float:
            raise ValueError("Binance quantity is below minimum")

        @staticmethod
        def fetch_dashboard_snapshot() -> dict:
            return {
                "private_available": True,
                "order_limits": {
                    "effective_min_quantity_raw": "1",
                    "effective_min_notional_raw": "100",
                    "estimated_max_open_quantity_raw": "0.5",
                    "estimated_max_open_notional_raw": "50",
                },
            }

        @staticmethod
        def place_market_order(request: object) -> dict:
            raise AssertionError("an unaffordable minimum order must not be submitted")

    class Strategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("long", 6, 1, ("fixed",))

    class Notifier:
        def __init__(self) -> None:
            self.messages = []

        def notify_order_unavailable(self, **payload: object) -> None:
            self.messages.append(payload)

    notifier = Notifier()
    risk = RiskManager(
        RiskConfig(risk_per_trade=0.02, stop_loss_pct=0.005, max_notional_pct=0.3),
        max_leverage=20,
    )
    engine = TradingEngine(
        Adapter(),
        Strategy(),
        risk,
        EngineConfig(mode="live"),
        notifier=notifier,
    )

    result = engine.evaluate_once()

    assert result.status == "minimum_order_unavailable"
    assert result.position is None
    assert len(notifier.messages) == 1
    assert notifier.messages[0]["minimum_notional"] == 100.0


def test_transient_private_failure_retries_same_signal_with_fresh_price() -> None:
    class Adapter:
        name = "binance"
        settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

        def __init__(self) -> None:
            self.snapshot_reads = 0
            self.mark_reads = 0
            self.entry_request = None

        @staticmethod
        def fetch_candles(interval: str, limit: int) -> list[Candle]:
            return [
                Candle(1, 100.0, 100.1, 99.9, 100.0, 10.0),
                Candle(2, 100.0, 100.1, 99.9, 100.0, 1.0),
            ]

        @staticmethod
        def fetch_live_position() -> None:
            return None

        @staticmethod
        def fetch_equity() -> float:
            return 1.0

        def fetch_mark_price(self) -> float:
            self.mark_reads += 1
            return 101.0

        @staticmethod
        def normalize_order_quantity(quantity: float, reference_price: float) -> float:
            if quantity < 1.0:
                raise ValueError("Binance quantity is below minimum")
            return 1.0

        def fetch_dashboard_snapshot(self) -> dict:
            self.snapshot_reads += 1
            if self.snapshot_reads == 1:
                return {
                    "private_available": False,
                    "private_transient": True,
                    "private_error": "TLS handshake timed out while reading openOrders",
                    "order_limits": {
                        "effective_min_quantity_raw": "1",
                        "effective_min_notional_raw": "100",
                        "estimated_max_open_quantity_raw": "0",
                        "estimated_max_open_notional_raw": "0",
                    },
                }
            return {
                "private_available": True,
                "order_limits": {
                    "effective_min_quantity_raw": "1",
                    "effective_min_notional_raw": "101",
                    "estimated_max_open_quantity_raw": "2",
                    "estimated_max_open_notional_raw": "202",
                },
            }

        def place_market_order(self, request: object) -> dict:
            self.entry_request = request
            return {"orderId": 11, "status": "FILLED", "executedQty": "1", "avgPrice": "101"}

        @staticmethod
        def market_fill(payload: dict, *, fallback_price: float) -> tuple[float, float]:
            return float(payload["executedQty"]), float(payload["avgPrice"])

        @staticmethod
        def place_protection_orders(position: Position, **kwargs: object) -> dict:
            return {"stop": {"algoId": 21}, "take_profit": None, "confirmed": True}

    class Strategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("long", 6, 1, ("fixed",))

    class Notifier:
        def __init__(self) -> None:
            self.private_messages = []
            self.capacity_messages = []

        def notify_private_api_unavailable(self, **payload: object) -> None:
            self.private_messages.append(payload)

        def notify_order_unavailable(self, **payload: object) -> None:
            self.capacity_messages.append(payload)

        @staticmethod
        def notify_open(*args: object, **kwargs: object) -> None:
            return None

    adapter = Adapter()
    notifier = Notifier()
    risk = RiskManager(
        RiskConfig(risk_per_trade=0.02, stop_loss_pct=0.005, max_notional_pct=0.3),
        max_leverage=20,
    )
    engine = TradingEngine(
        adapter,
        Strategy(),
        risk,
        EngineConfig(
            mode="live",
            private_entry_retry_seconds=45,
            private_entry_retry_interval_seconds=1,
        ),
        notifier=notifier,
    )

    first = engine.evaluate_once()
    assert first.status == "private_api_retry_scheduled"
    assert first.position is None
    assert first.raw["retry_scheduled"] is True
    assert len(notifier.private_messages) == 1
    assert notifier.capacity_messages == []

    # Make the scheduled retry immediately eligible without slowing the test.
    engine._private_entry_retry_next_at = 0.0
    second = engine.evaluate_once()

    assert second.status == "live_order_protected"
    assert second.position is not None
    assert second.position.entry_price == 101.0
    assert adapter.mark_reads == 1
    assert adapter.snapshot_reads == 2
    assert adapter.entry_request.quantity == 1.0
    assert len(notifier.private_messages) == 1
    assert notifier.capacity_messages == []
    assert engine._private_entry_retry_signal_timestamp == 0


class _EmergencyNotifier:
    def __init__(self) -> None:
        self.emergencies: list[dict[str, object]] = []
        self.resolved: list[tuple[str, str]] = []

    def notify_emergency(self, error: Exception, **payload: object) -> bool:
        self.emergencies.append({"error": error, **payload})
        return True

    def resolve_emergency(
        self,
        category: str,
        exchange: str,
        incident: str = "",
    ) -> None:
        self.resolved.append((category, incident))

    @staticmethod
    def notify_open(*args: object, **kwargs: object) -> None:
        return None

    @staticmethod
    def notify_close(*args: object, **kwargs: object) -> None:
        return None


class _FixedLongSignalStrategy:
    config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

    @staticmethod
    def evaluate(candles_by_timeframe: object) -> Signal:
        return Signal("long", 6, 1, ("fixed",))


class _EmergencyAdapter:
    name = "binance"
    settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

    def __init__(
        self,
        *,
        entry_error: Exception | None = None,
        protection_error: Exception | None = None,
        close_error: Exception | None = None,
        equity_error: Exception | None = None,
        live_position: dict[str, object] | None = None,
    ) -> None:
        self.entry_error = entry_error
        self.protection_error = protection_error
        self.close_error = close_error
        self.equity_error = equity_error
        self.live_position = live_position

    @staticmethod
    def fetch_candles(interval: str, limit: int) -> list[Candle]:
        return [
            Candle(1, 100.0, 100.1, 99.9, 100.0, 10.0),
            Candle(2, 100.0, 100.1, 99.9, 100.0, 1.0),
        ]

    def fetch_live_position(self) -> dict[str, object] | None:
        return self.live_position

    def fetch_equity(self) -> float:
        if self.equity_error is not None:
            raise self.equity_error
        return 10_000.0

    @staticmethod
    def normalize_order_quantity(quantity: float, reference_price: float) -> float:
        return quantity

    def place_market_order(self, request: object) -> dict:
        error = self.close_error if getattr(request, "reduce_only", False) else self.entry_error
        if error is not None:
            raise error
        return {"orderId": 11, "status": "FILLED", "executedQty": "1", "avgPrice": "100"}

    @staticmethod
    def market_fill(payload: dict, *, fallback_price: float) -> tuple[float, float]:
        return float(payload["executedQty"]), float(payload["avgPrice"])

    def place_protection_orders(self, position: Position, **kwargs: object) -> dict:
        if self.protection_error is not None:
            raise self.protection_error
        return {"stop": {"algoId": 21}, "take_profit": None, "confirmed": True}

    @staticmethod
    def emergency_close(position: Position, *, client_id: str) -> dict:
        return {"status": "FILLED", "executedQty": "1", "avgPrice": "99.9"}

    @staticmethod
    def cancel_protection_orders(position: Position) -> dict:
        return {"cancelled": True, "errors": []}


def _alert_test_engine(adapter: object, notifier: _EmergencyNotifier) -> TradingEngine:
    return TradingEngine(
        adapter,
        _FixedLongSignalStrategy(),
        RiskManager(RiskConfig(stop_loss_pct=0.005)),
        EngineConfig(mode="live"),
        notifier=notifier,
    )


def test_live_entry_order_error_emits_emergency_notification() -> None:
    entry_error = RuntimeError("entry market order rejected")
    notifier = _EmergencyNotifier()
    engine = _alert_test_engine(_EmergencyAdapter(entry_error=entry_error), notifier)

    try:
        engine.evaluate_once()
    except RuntimeError as raised:
        assert raised is entry_error
    else:
        raise AssertionError("entry order error must remain visible to the engine loop")

    assert len(notifier.emergencies) == 1
    assert notifier.emergencies[0]["error"] is entry_error
    assert notifier.emergencies[0]["category"] == "order_failure"
    assert notifier.emergencies[0]["incident"] == "entry_submit"


def test_deduplicated_capacity_emergency_does_not_fall_back_to_normal_email() -> None:
    class Notifier(_EmergencyNotifier):
        def __init__(self) -> None:
            super().__init__()
            self.capacity_messages: list[dict[str, object]] = []

        def notify_emergency(self, error: Exception, **payload: object) -> bool:
            self.emergencies.append({"error": error, **payload})
            return False

        def notify_order_unavailable(self, **payload: object) -> None:
            self.capacity_messages.append(payload)

    error = RuntimeError('{"code":-4164,"msg":"minimum notional"}')
    notifier = Notifier()
    result = _alert_test_engine(
        _EmergencyAdapter(entry_error=error),
        notifier,
    ).evaluate_once()

    assert result.status == "minimum_order_unavailable"
    assert len(notifier.emergencies) == 1
    assert notifier.capacity_messages == []


def test_live_protection_error_emits_emergency_even_after_successful_rollback() -> None:
    protection_error = RuntimeError("stop order was not confirmed open")
    notifier = _EmergencyNotifier()
    result = _alert_test_engine(
        _EmergencyAdapter(protection_error=protection_error),
        notifier,
    ).evaluate_once()

    assert result.status == "live_entry_rolled_back"
    assert result.position is None
    assert len(notifier.emergencies) == 1
    assert notifier.emergencies[0]["error"] is protection_error
    assert notifier.emergencies[0]["category"] == "order_failure"
    assert notifier.emergencies[0]["incident"] == "protection_submit"
    assert ("order_failure", "entry_submit") in notifier.resolved
    assert ("ip_restricted", "") in notifier.resolved


def test_embedded_protection_cancel_errors_emit_emergency_notification() -> None:
    class Adapter(_EmergencyAdapter):
        @staticmethod
        def cancel_protection_orders(position: Position) -> dict:
            return {
                "cancelled": [],
                "errors": ['HTTP 429: {"code":-1003,"msg":"too many requests"}'],
            }

    notifier = _EmergencyNotifier()
    engine = _alert_test_engine(Adapter(), notifier)
    position = Position("long", 1.0, 100.0, 99.0, 103.0, 123_456)

    result = engine._cancel_protection_orders_with_alert(
        position,
        context="test cancellation",
    )

    assert result["errors"]
    assert len(notifier.emergencies) == 1
    assert notifier.emergencies[0]["category"] == "order_failure"
    assert notifier.emergencies[0]["incident"] == "protection_cancel:123456"


def test_embedded_protection_status_errors_emit_emergency_notification() -> None:
    class Adapter(_EmergencyAdapter):
        @staticmethod
        def fetch_protection_status(position: Position) -> dict:
            return {"stop": {"error": "HTTP 418: IP banned"}}

    notifier = _EmergencyNotifier()
    engine = _alert_test_engine(Adapter(), notifier)
    engine.position = Position(
        "long",
        1.0,
        100.0,
        99.0,
        103.0,
        123_456,
        stop_order_id="1",
        stop_client_id="stop",
    )

    engine._finalize_binance_live_position_closed(
        Candle(123_999, 99.0, 99.0, 99.0, 99.0, 0.0)
    )

    assert engine.position is None
    assert len(notifier.emergencies) == 1
    assert notifier.emergencies[0]["category"] == "order_failure"
    assert notifier.emergencies[0]["incident"] == "protection_status:123456"


def test_managed_stop_reconciliation_records_exchange_fill_price_and_time(tmp_path) -> None:
    fill_timestamp = 1_788_525_008_035

    class Adapter(_EmergencyAdapter):
        @staticmethod
        def fetch_protection_status(position: Position) -> dict:
            return {"stop": {"status": "FILLED", "triggerPrice": "80863.6"}}

        @staticmethod
        def fetch_latest_close_fill(position: Position, *, after_ms: int) -> dict:
            assert after_ms == position.opened_at
            return {
                "quantity": position.quantity,
                "price": 80777.1,
                "timestamp": fill_timestamp,
            }

    reporter = TradeReporter(tmp_path)
    engine = TradingEngine(
        Adapter(),
        _FixedLongSignalStrategy(),
        RiskManager(
            RiskConfig(stop_loss_pct=0.005),
            costs=CostConfig(
                maker_fee_pct=0.0,
                taker_fee_pct=0.0,
                slippage_pct=0.0,
                funding_rate_pct_per_8h=0.0,
            ),
        ),
        EngineConfig(mode="live"),
        reporter=reporter,
    )
    engine.position = Position(
        "long",
        0.001,
        81229.2,
        80863.6686,
        82143.0285,
        1_788_521_161_832,
        initial_stop_price=80863.6686,
        stop_order_id="1",
        stop_client_id="stop",
    )

    try:
        engine._finalize_binance_live_position_closed(
            Candle(fill_timestamp, 81158.0, 81158.0, 81158.0, 81158.0, 0.0)
        )
        rows = reporter.query_trades(exchange="binance", scope="production")
    finally:
        reporter.close()

    assert engine.position is None
    assert len(rows) == 1
    assert rows[0]["exit_reason"] == "stop_loss"
    assert rows[0]["exit_price"] == pytest.approx(80777.1)
    assert rows[0]["exit_time"] == "2026-09-04T12:30:08Z"
    assert rows[0]["mae_price"] == pytest.approx(81229.2 - 80777.1)
    assert rows[0]["mae_r"] == pytest.approx(
        (81229.2 - 80777.1) / (81229.2 - 80863.6686)
    )
    assert rows[0]["holding_minutes"] == pytest.approx(
        (fill_timestamp - 1_788_521_161_832) / 60_000
    )


def test_managed_close_fill_lookup_error_alerts_and_uses_status_fallback() -> None:
    fill_error = ApiError(
        'HTTP 429: {"code":-1003,"msg":"too many requests"}',
        status_code=429,
        api_code=-1003,
    )

    class Adapter(_EmergencyAdapter):
        @staticmethod
        def fetch_protection_status(position: Position) -> dict:
            return {"stop": {"status": "FILLED", "triggerPrice": "99.0"}}

        @staticmethod
        def fetch_latest_close_fill(position: Position, *, after_ms: int) -> dict:
            raise fill_error

    notifier = _EmergencyNotifier()
    engine = _alert_test_engine(Adapter(), notifier)
    engine.position = Position(
        "long",
        1.0,
        100.0,
        99.0,
        103.0,
        123_456,
        initial_stop_price=99.0,
        stop_order_id="1",
        stop_client_id="stop",
    )

    engine._finalize_binance_live_position_closed(
        Candle(123_999, 98.5, 99.1, 98.0, 98.5, 0.0)
    )

    assert engine.position is None
    assert len(notifier.emergencies) == 1
    assert notifier.emergencies[0]["error"] is fill_error
    assert notifier.emergencies[0]["incident"] == "managed_close_fill:123456"


def test_unmanaged_close_fill_lookup_error_emits_emergency_before_fallback() -> None:
    fill_error = ApiError(
        'HTTP 429: {"code":-1003,"msg":"too many requests"}',
        status_code=429,
        api_code=-1003,
    )

    class Adapter(_EmergencyAdapter):
        @staticmethod
        def fetch_latest_close_fill(position: Position, *, after_ms: int) -> dict:
            raise fill_error

    class Notifier(_EmergencyNotifier):
        @staticmethod
        def notify_reconciled_close(**payload: object) -> bool:
            return True

    notifier = Notifier()
    engine = _alert_test_engine(Adapter(), notifier)
    engine.unmanaged_live_position = {
        "side": "short",
        "quantity": 1.0,
        "entry_price": 100.0,
        "observed_at_ms": 123_456,
    }

    engine._resolve_unmanaged_live_position(
        Candle(123_999, 99.0, 99.0, 99.0, 99.0, 0.0)
    )

    assert engine.unmanaged_live_position is None
    assert len(notifier.emergencies) == 1
    assert notifier.emergencies[0]["error"] is fill_error
    assert notifier.emergencies[0]["incident"] == "reconcile_close_fill:123456"


def test_live_close_order_error_emits_emergency_notification() -> None:
    close_error = RuntimeError("reduce-only close order rejected")
    notifier = _EmergencyNotifier()
    engine = _alert_test_engine(
        _EmergencyAdapter(
            close_error=close_error,
            live_position={"side": "long", "quantity": 1.0, "mark_price": 99.0},
        ),
        notifier,
    )
    engine.position = Position(
        "long",
        1.0,
        100.0,
        99.0,
        103.0,
        int(time.time() * 1000) - 60_000,
        stop_order_id="1",
        stop_client_id="stop",
    )

    try:
        engine._close_live_position(99.0, "stop_loss")
    except RuntimeError as raised:
        assert raised is close_error
    else:
        raise AssertionError("failed close order must remain visible to the engine loop")

    assert len(notifier.emergencies) == 1
    assert notifier.emergencies[0]["error"] is close_error
    assert notifier.emergencies[0]["category"] == "order_failure"
    assert notifier.emergencies[0]["incident"] == "close_submit"


def test_rate_limited_private_api_trade_result_forwards_raw_error_to_notifier() -> None:
    rate_limit_error = ApiError(
        'HTTP 429 GET https://fapi.binance.com/fapi/v2/account: {"code":-1003}',
        status_code=429,
        retry_at=time.time() + 60,
    )

    notifier = _EmergencyNotifier()
    result = _alert_test_engine(
        _EmergencyAdapter(equity_error=rate_limit_error),
        notifier,
    ).evaluate_once()

    assert result.status == "private_api_unavailable"
    assert result.position is None
    assert len(notifier.emergencies) == 1
    assert notifier.emergencies[0]["error"] is rate_limit_error
    assert notifier.emergencies[0]["category"] == "order_failure"
    assert notifier.emergencies[0]["incident"] == "entry_private_api"
    assert rate_limit_error.rate_limited is True
    assert ("ip_restricted", "") not in notifier.resolved
