from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.costs import CostConfig
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


def test_live_position_checks_mark_price_each_poll_and_closes_at_protected_stop() -> None:
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
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.9"}

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
    engine = TradingEngine(adapter, Strategy(), RiskManager(), EngineConfig(mode="live"))
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
