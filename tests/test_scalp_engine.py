from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from btc_futures_bot.dashboard import _macro_entry_block_summary
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.main import save_dashboard_config
from btc_futures_bot.models import Candle, Position, Signal
from btc_futures_bot.risk import RiskConfig, RiskManager
from btc_futures_bot.strategy import MultiTimeframeStrategy, StrategyConfig


class CandleAdapter:
    name = "okx"
    settings = SimpleNamespace(symbol="BTC-USDT-SWAP", environment="demo")

    def __init__(self, *, count: int = 2) -> None:
        self.calls: list[str] = []
        self.count = count

    def fetch_candles(self, interval: str, limit: int) -> list[Candle]:
        self.calls.append(interval)
        step = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}[interval]
        return [Candle(index * step, 100.0, 100.1, 99.9, 100.0, 10.0) for index in range(self.count)]


class RecordingStrategy:
    config = StrategyConfig(mode="scalp_v2", trigger_timeframe="1m", regime_timeframe="5m")

    def __init__(self) -> None:
        self.candles: dict[str, list[Candle]] = {}

    def evaluate(self, candles: dict[str, list[Candle]]) -> Signal:
        self.candles = candles
        return Signal("flat", 0, 0, ("recorded",))


@pytest.mark.parametrize("gate_mode", ["shadow", "enforce"])
def test_minute_scalp_fetches_closed_hourly_features_for_active_gate(gate_mode: str) -> None:
    adapter = CandleAdapter()
    strategy = RecordingStrategy()
    engine = TradingEngine(
        adapter, strategy, RiskManager(), EngineConfig(mode="paper"),
        entry_gate=SimpleNamespace(config=SimpleNamespace(mode=gate_mode)),
    )

    engine.evaluate_once()

    assert adapter.calls == ["1m", "5m", "1h"]
    assert strategy.config.regime_timeframe == "5m"
    assert all(len(candles) == 1 and candles[0].timestamp == 0 for candles in strategy.candles.values())


@pytest.mark.parametrize("gate", [None, SimpleNamespace(config=SimpleNamespace(mode="off"))])
def test_gate_off_keeps_primary_timeframe_fetches_only(gate: object) -> None:
    adapter = CandleAdapter()
    engine = TradingEngine(adapter, RecordingStrategy(), RiskManager(), EngineConfig(), entry_gate=gate)

    engine.evaluate_once()

    assert adapter.calls == ["1m", "5m"]


def test_single_forming_candle_is_not_passed_to_primary_or_model() -> None:
    adapter = CandleAdapter(count=1)
    strategy = RecordingStrategy()
    engine = TradingEngine(
        adapter, strategy, RiskManager(), EngineConfig(),
        entry_gate=SimpleNamespace(config=SimpleNamespace(mode="shadow")),
    )

    engine.evaluate_once()

    assert strategy.candles == {"1m": [], "5m": [], "1h": []}


@pytest.mark.parametrize("gate_mode", ["off", "shadow", "enforce"])
@pytest.mark.parametrize("environment", ["demo", "production"])
def test_experimental_scalp_cannot_enable_exchange_orders(gate_mode: str, environment: str) -> None:
    adapter = CandleAdapter()
    adapter.settings = SimpleNamespace(symbol="BTC-USDT-SWAP", environment=environment)
    with pytest.raises(ValueError, match="scalp_v2 requires mode=paper"):
        TradingEngine(
            adapter, RecordingStrategy(), RiskManager(), EngineConfig(mode="live"),
            entry_gate=SimpleNamespace(config=SimpleNamespace(mode=gate_mode)),
        )
    assert adapter.calls == []


def test_dashboard_can_save_scalp_v2_without_changing_execution_network(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "mode": "paper", "active_exchange": "okx",
        "exchanges": {"okx": {"enabled": True, "environment": "demo"}},
        "strategy": {"mode": "traditional_kline"},
    }), encoding="utf-8")

    target = save_dashboard_config(path, {"exchange": "okx", "strategy_mode": "scalp_v2"})
    saved = json.loads(target.read_text(encoding="utf-8"))

    assert saved["strategy"]["mode"] == "scalp_v2"
    assert saved["mode"] == "paper"
    assert saved["exchanges"]["okx"]["environment"] == "demo"


def _scalp_engine(side: str = "long") -> TradingEngine:
    strategy = MultiTimeframeStrategy(StrategyConfig(
        mode="scalp_v2", trigger_timeframe="1m", regime_timeframe="5m",
        enable_time_exit=True, max_hold_seconds=300, hard_max_hold_seconds=600,
        time_exit_min_r=0.2, break_even_trigger_r=0, trailing_trigger_r=0,
    ))
    engine = TradingEngine(CandleAdapter(), strategy, RiskManager(RiskConfig()), EngineConfig())
    engine.position = Position(
        side, 1.0, 100.0, 99.0 if side == "long" else 101.0,
        102.0 if side == "long" else 98.0, 1_000_000,
        initial_stop_price=99.0 if side == "long" else 101.0,
    )
    return engine


@pytest.mark.parametrize("side,close", [("long", 99.9), ("short", 100.1)])
def test_scalp_paper_hard_timeout_closes_loser_after_ten_minutes(side: str, close: float) -> None:
    engine = _scalp_engine(side)
    candle = Candle(1_540_000, 100.0, 100.2, 99.8, close, 10)
    with patch("btc_futures_bot.engine.time.time", return_value=1599):
        engine._manage_paper_position(candle)
    assert engine.position is not None

    with patch("btc_futures_bot.engine.time.time", return_value=1600), patch.object(
        engine, "_close_paper_position", wraps=engine._close_paper_position,
    ) as close_position:
        engine._manage_paper_position(candle)

    close_position.assert_called_once_with(close, "hard_time_exit")
    assert engine.position is None
    assert engine.session_pnl < 0
    assert engine.consecutive_losses == 1


def test_scalp_paper_soft_timeout_requires_net_profit() -> None:
    engine = _scalp_engine()
    # The tiny gross gain does not cover modeled fees/slippage.
    with patch("btc_futures_bot.engine.time.time", return_value=1300):
        engine._manage_paper_position(Candle(1_240_000, 100.0, 100.1, 99.9, 100.05, 10))
    assert engine.position is not None

    with patch("btc_futures_bot.engine.time.time", return_value=1300), patch.object(
        engine, "_close_paper_position", wraps=engine._close_paper_position,
    ) as close_position:
        engine._manage_paper_position(Candle(1_240_000, 100.0, 100.35, 99.9, 100.3, 10))
    close_position.assert_called_once_with(100.3, "time_exit")
    assert engine.position is None
    assert engine.session_pnl > 0


@pytest.mark.parametrize("side,mark", [("long", 99.9), ("short", 100.1)])
def test_scalp_live_hard_timeout_does_not_require_profit(side: str, mark: float) -> None:
    engine = _scalp_engine(side)
    with patch("btc_futures_bot.engine.time.time", return_value=1599):
        assert engine._live_time_exit_reason(mark) == ""
    with patch("btc_futures_bot.engine.time.time", return_value=1600):
        assert engine._live_time_exit_reason(mark) == "hard_time_exit"


def test_scalp_stop_is_honored_before_time_exit() -> None:
    engine = _scalp_engine()
    with patch("btc_futures_bot.engine.time.time", return_value=1600), patch.object(
        engine, "_close_paper_position", wraps=engine._close_paper_position,
    ) as close_position:
        engine._manage_paper_position(Candle(1_540_000, 99.5, 100.1, 98.9, 99.7, 10))
    close_position.assert_called_once_with(99.0, "stop_loss")


def test_macro_summary_names_actual_shock_and_its_end() -> None:
    summary = _macro_entry_block_summary({
        "reason": "macro_shock:range=7.53x,volume=1.92x,range_pct=0.0278%",
        "shock_until_ms": 1_788_676_926_000,
        "next_event": {"name": "future FOMC", "timestamp_ms": 1_789_999_999_000},
    })
    assert "range_pct=0.0278%" in summary
    assert "当前窗口截止" in summary
    assert "future FOMC" not in summary


def test_macro_calendar_summary_uses_event_window_end() -> None:
    summary = _macro_entry_block_summary({
        "reason": "macro_event:CPI", "shock_until_ms": 0,
        "next_event": {"name": "CPI", "timestamp_ms": 1_788_676_926_000, "post_minutes": 30},
    })
    assert "macro_event:CPI" in summary
    assert "当前窗口截止" in summary
