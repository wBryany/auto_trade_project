import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.exchanges.base import ExchangeSettings
from btc_futures_bot.exchanges.binance import BinanceAdapter
from btc_futures_bot.models import Position, Signal
from btc_futures_bot.risk import RiskManager
from btc_futures_bot.strategy import MultiTimeframeStrategy, StrategyConfig


@pytest.fixture
def clock(monkeypatch):
    values = {"wall": 1_788_840_000.0, "monotonic": 100.0}
    monkeypatch.setattr("btc_futures_bot.engine.time.time", lambda: values["wall"])
    monkeypatch.setattr("btc_futures_bot.engine.time.monotonic", lambda: values["monotonic"])
    return values


def _engine(tmp_path, clock, side="short"):
    adapter = SimpleNamespace(
        name="binance", settings=SimpleNamespace(symbol="BTCUSDT", environment="production"),
    )
    strategy = MultiTimeframeStrategy(StrategyConfig(
        enable_time_exit=False, break_even_trigger_r=0.5, break_even_lock_r=0.1,
        trailing_trigger_r=2.0,
    ))
    engine = TradingEngine(adapter, strategy, RiskManager(), EngineConfig(
        mode="live", reconciliation_state_path=str(tmp_path / "state.json"),
    ))
    engine.position = Position(
        side, 1, 100, 90 if side == "long" else 110, 120 if side == "long" else 80,
        int(clock["wall"] * 1000) - 60_000,
        initial_stop_price=90 if side == "long" else 110,
        best_price=100, worst_price=100,
        stop_order_id="123", stop_client_id="btcbot-stop-checkpoint",
    )
    engine.position_signal = Signal(side, 6, 1, ("1m_ultra_short_reversal_short",))
    engine._save_live_reconciliation_state()
    return engine


def _tick(engine, price):
    return engine._manage_live_position(price, {}, Signal("flat", 0, 0, ()))


def _saved(engine):
    return json.loads(Path(engine.config.reconciliation_state_path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("side", ["long", "short"])
def test_armed_stop_survives_abrupt_restart_and_keeps_original_exchange_protection(tmp_path, clock, monkeypatch, side):
    engine = _engine(tmp_path, clock, side)
    original_hard_stop = engine.position.initial_stop_price
    clock["monotonic"] += 0.1  # Stop changes must bypass the five-second batching.
    _tick(engine, 108 if side == "long" else 92)
    protected = engine.position
    assert protected.stop_reason == "break_even_stop"
    assert protected.stop_price != original_hard_stop

    # No close()/shutdown hook is invoked: load the state a crash would leave.
    resumed = TradingEngine(engine.adapter, engine.strategy, engine.risk, engine.config)
    resumed._restore_managed_live_position()
    assert resumed.position == protected
    assert resumed.position_signal == engine.position_signal

    # The actual Binance resume validator must still match the original hard
    # protection, not require the exchange order to move to the saved soft stop.
    monkeypatch.setattr("btc_futures_bot.exchanges.binance.request_json", lambda *_args, **_kwargs: pytest.fail("unexpected HTTP request"))
    adapter = BinanceAdapter(ExchangeSettings("binance", "production", "https://fapi.binance.com", "BTCUSDT"))
    monkeypatch.setattr(adapter, "symbol_rules", lambda: {"price_tick": Decimal("0.1")})
    state = _saved(engine)["managed_position"]
    remote = [{
        "positionSide": "BOTH", "positionAmt": "1" if side == "long" else "-1",
        "entryPrice": "100", "leverage": "3", "isolated": True,
    }]
    stop = {
        "algoId": "123", "clientAlgoId": "btcbot-stop-checkpoint", "symbol": "BTCUSDT",
        "algoType": "CONDITIONAL", "orderType": "STOP_MARKET",
        "side": "SELL" if side == "long" else "BUY", "positionSide": "BOTH",
        "workingType": "MARK_PRICE", "closePosition": True,
        "triggerPrice": str(original_hard_stop), "algoStatus": "NEW",
    }
    assert adapter._resume_protected_live_position(state, remote, [], [stop], max_leverage=3)["resumed"]
    with pytest.raises(RuntimeError, match="unexpected trigger price"):
        adapter._resume_protected_live_position(
            state, remote, [], [dict(stop, triggerPrice=str(protected.stop_price))], max_leverage=3,
        )

    closes = []
    monkeypatch.setattr(resumed, "_close_live_position", lambda price, reason: closes.append((price, reason)) or {})
    result = _tick(resumed, 100)
    assert result.raw["exit_reason"] == "break_even_stop"
    assert closes == [(100, "break_even_stop")]


def test_price_extrema_are_batched_and_idle_cycles_do_not_rewrite_state(tmp_path, clock, monkeypatch):
    engine = _engine(tmp_path, clock)
    writes = []
    save = engine._save_live_reconciliation_state
    monkeypatch.setattr(engine, "_save_live_reconciliation_state", lambda: writes.append(clock["monotonic"]) or save())
    clock["monotonic"] = 101
    _tick(engine, 99)
    clock["monotonic"] = 102
    _tick(engine, 101)
    assert writes == []
    assert _saved(engine)["managed_position"]["position"]["best_price"] == 100
    clock["monotonic"] = 105
    _tick(engine, 100)
    assert writes == [105]
    snapshot = _saved(engine)["managed_position"]["position"]
    assert (snapshot["best_price"], snapshot["worst_price"]) == (99, 101)
    for now in (110, 115, 120):
        clock["monotonic"] = now
        _tick(engine, 100)
    assert writes == [105]


def test_failed_atomic_save_keeps_old_snapshot_and_retries_armed_stop(tmp_path, clock, monkeypatch):
    engine = _engine(tmp_path, clock)
    original_state = _saved(engine)
    original_replace = Path.replace
    monkeypatch.setattr(Path, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk unavailable")))
    clock["monotonic"] += 0.1
    assert _tick(engine, 92) is None
    assert engine.position.stop_reason == "break_even_stop"
    assert engine.position.initial_stop_price == 110
    assert engine.position.stop_order_id == "123"
    assert _saved(engine) == original_state

    monkeypatch.setattr(Path, "replace", original_replace)
    clock["monotonic"] += 0.1
    _tick(engine, 93)  # No new high/low or stop is needed to retry the failed write.
    snapshot = _saved(engine)["managed_position"]["position"]
    assert snapshot["stop_price"] == engine.position.stop_price
    assert snapshot["best_price"] == 92


def test_failed_checkpoint_does_not_prevent_next_cycle_protective_exit(tmp_path, clock, monkeypatch):
    engine = _engine(tmp_path, clock)
    monkeypatch.setattr(Path, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk unavailable")))
    _tick(engine, 92)
    closes = []
    monkeypatch.setattr(engine, "_close_live_position", lambda price, reason: closes.append((price, reason)) or {})
    result = _tick(engine, 100)
    assert result.raw["exit_reason"] == "break_even_stop"
    assert closes == [(100, "break_even_stop")]


@pytest.mark.parametrize("mode,exchange,path", [("paper", "binance", "state.json"), ("live", "okx", "state.json"), ("live", "binance", "")])
def test_checkpoint_is_limited_to_configured_binance_live_state(tmp_path, clock, monkeypatch, mode, exchange, path):
    engine = _engine(tmp_path, clock)
    engine.config.mode = mode
    engine.adapter.name = exchange
    engine.config.reconciliation_state_path = str(tmp_path / path) if path else ""
    monkeypatch.setattr(engine, "_save_live_reconciliation_state", lambda: pytest.fail("unexpected checkpoint"))
    _tick(engine, 92)
