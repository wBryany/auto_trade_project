import json
import logging
import math
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from btc_futures_bot.costs import CostConfig
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.models import Candle, Signal
from btc_futures_bot.risk import RiskConfig, RiskManager
from btc_futures_bot.strategy import StrategyConfig


def _bars(count=60, spread=0.2):
    return [Candle(i * 60_000, 100, 100 + spread / 2, 100 - spread / 2, 100, 10)
            for i in range(count)]


def _risk(**kwargs):
    return RiskManager(RiskConfig(**kwargs), costs=CostConfig(
        taker_fee_pct=0.0005, slippage_pct=0.0002,
        min_net_edge_pct=0.0015, expected_holding_hours=0.1,
    ))


class EntryAdapter:
    name = "binance"
    settings = SimpleNamespace(symbol="BTCUSDT", environment="production", api_key="credential-sentinel")

    def __init__(self):
        self.equity_reads = 0
        self.order_calls = 0

    def fetch_candles(self, *_args):
        # The forming candle would pass the range gate if accidentally included.
        return _bars() + [Candle(60 * 60_000, 100, 200, 1, 100, 10)]

    def fetch_live_position(self):
        return None

    def fetch_equity(self):
        self.equity_reads += 1
        return 10_000

    def place_market_order(self, *_args):
        self.order_calls += 1
        pytest.fail("rejected entry must not submit an order")


def _engine(risk):
    strategy = SimpleNamespace(
        config=StrategyConfig(trigger_timeframe="1m", regime_timeframe="1m",
                              min_stop_loss_pct=0.05, max_stop_loss_pct=0.05),
        signal=Signal("short", 6, 59 * 60_000, ("signal-prose-sentinel",)),
    )
    strategy.evaluate = lambda _: strategy.signal
    return TradingEngine(EntryAdapter(), strategy, risk, EngineConfig(mode="live"))


def _rejections(caplog):
    marker = "entry_gate_rejected "
    return [json.loads(record.getMessage().split(marker, 1)[1])
            for record in caplog.records if marker in record.getMessage()]


def test_range_diagnostic_describes_the_same_lookback_and_cost_formula():
    risk = _risk(entry_range_lookback_minutes=60)
    candles = [Candle(-60_000, 100, 200, 1, 100, 10)] + _bars()
    result = risk.assess_entry_range(candles, 100)
    assert not result.allowed
    assert result.reason == "range_below_required"
    assert result.lookback_minutes == result.window_count == 60
    assert result.available_candles == 61
    assert result.entry_price == 100
    assert result.first_candle_timestamp == 0
    assert result.last_candle_timestamp == 59 * 60_000
    assert result.range_points == pytest.approx(0.2)
    assert result.observed_pct == pytest.approx(0.002)
    assert result.required_pct == pytest.approx(0.0029)
    assert risk.observed_range_allows_entry(candles, 100) is result.allowed
    json.dumps(asdict(result), allow_nan=False)


def test_range_equal_boundary_passes_and_one_float_below_fails():
    risk = RiskManager(RiskConfig(entry_range_lookback_minutes=2), costs=CostConfig(
        taker_fee_pct=0, slippage_pct=0, funding_rate_pct_per_8h=0, min_net_edge_pct=0.125,
    ))
    candles = [Candle(i * 60_000, 8, 8.5, 7.5, 8, 1) for i in range(2)]
    equal = risk.assess_entry_range(candles, 8)
    assert equal.observed_pct == equal.required_pct == 0.125
    assert equal.allowed and equal.reason == "range_sufficient"
    below = [replace(c, high=math.nextafter(8.5, -math.inf)) for c in candles]
    assert not risk.assess_entry_range(below, 8).allowed
    assert not risk.observed_range_allows_entry(below, 8)


@pytest.mark.parametrize("kind,reason", [
    ("insufficient", "insufficient_candles"),
    ("gap", "non_contiguous_candles"),
    ("nan_high", "invalid_candle_range"),
    ("inf_low", "invalid_candle_range"),
    ("zero_low", "invalid_candle_range"),
    ("inverted", "invalid_candle_range"),
    ("nan_price", "invalid_entry_price"),
    ("zero_price", "invalid_entry_price"),
])
def test_invalid_range_assessment_explains_rejection_without_nonfinite_output(kind, reason):
    risk = _risk(entry_range_lookback_minutes=60)
    candles, price = _bars(), 100
    if kind == "insufficient": candles.pop()
    if kind == "gap": candles[10] = replace(candles[10], timestamp=1)
    if kind == "nan_high": candles[10] = replace(candles[10], high=math.nan)
    if kind == "inf_low": candles[10] = replace(candles[10], low=math.inf)
    if kind == "zero_low": candles[10] = replace(candles[10], low=0)
    if kind == "inverted": candles[10] = replace(candles[10], high=99, low=101)
    if kind == "nan_price": price = math.nan
    if kind == "zero_price": price = 0
    result = risk.assess_entry_range(candles, price)
    assert not result.allowed and result.reason == reason
    assert not risk.observed_range_allows_entry(candles, price)
    assert result.observed_pct is None and result.range_points is None
    json.dumps(asdict(result), allow_nan=False)


def test_disabled_range_gate_still_allows_missing_or_invalid_observations():
    risk = _risk(entry_range_lookback_minutes=0)
    assert risk.observed_range_allows_entry([], math.nan)
    result = risk.assess_entry_range([], math.nan)
    assert result.allowed and result.reason == "disabled"
    assert result.window_count == result.available_candles == 0
    json.dumps(asdict(result), allow_nan=False)


def test_range_rejection_is_logged_once_and_survives_later_no_action_cycles(caplog):
    engine = _engine(_risk(entry_range_lookback_minutes=60))
    with caplog.at_level(logging.INFO):
        first = engine.evaluate_once()
        for _ in range(5):
            assert engine.evaluate_once().status == "no_action"
    assert first.status == "insufficient_market_range"
    assert first.raw["entry_blocked"] == "closed_1m_range_below_costs_and_minimum_net_edge"
    assert first.raw["entry_diagnostics"]["range_points"] == pytest.approx(0.2)
    assert engine.adapter.equity_reads == engine.adapter.order_calls == 0
    assert engine.position is None
    messages = _rejections(caplog)
    assert len(messages) == 1
    logged = messages[0]
    assert logged["status"] == first.status
    assert logged["signal_timestamp"] == engine.strategy.signal.timestamp
    assert logged["side"] == "short" and logged["score"] == 6
    assert logged["window_count"] == logged["available_candles"] == 60
    assert logged["entry_price"] == 100
    assert logged["first_candle_timestamp"] == 0
    assert logged["last_candle_timestamp"] == 59 * 60_000
    assert logged["observed_pct"] == pytest.approx(0.002)
    assert logged["required_pct"] == pytest.approx(0.0029)
    assert "credential-sentinel" not in json.dumps(logged)
    assert "signal-prose-sentinel" not in json.dumps(logged)
    # The next distinct signal gets its own rejection; no long-lived set grows.
    engine.strategy.signal = replace(engine.strategy.signal, timestamp=60 * 60_000)
    with caplog.at_level(logging.INFO):
        assert engine.evaluate_once().status == "insufficient_market_range"
    assert len(_rejections(caplog)) == 2


@pytest.mark.parametrize("kind,reason", [
    ("cooldown", "cooldown_active"),
    ("session", "session_loss_limit"),
    ("streak", "consecutive_loss_limit"),
])
def test_risk_rejection_logs_actual_gate_without_changing_risk_state(caplog, monkeypatch, kind, reason):
    engine = _engine(_risk(max_consecutive_losses=3))
    monkeypatch.setattr("btc_futures_bot.engine.time.time", lambda: 1_000)
    if kind == "cooldown": engine.cooldown_until = 1_060
    if kind == "session": engine.session_pnl = -200
    if kind == "streak": engine.consecutive_losses = 3
    before = (engine.consecutive_losses, engine.session_pnl, engine.cooldown_until)
    with caplog.at_level(logging.INFO):
        result = engine.evaluate_once()
        assert engine.evaluate_once().status == "no_action"
    assert result.status == "risk_blocked"
    assert result.raw["entry_diagnostics"]["reason"] == reason
    assert len(_rejections(caplog)) == 1
    assert _rejections(caplog)[0]["reason"] == reason
    assert before == (engine.consecutive_losses, engine.session_pnl, engine.cooldown_until)
    assert engine.adapter.equity_reads == engine.adapter.order_calls == 0


@pytest.mark.parametrize("stage", ["before_live_quantity", "after_live_quantity"])
def test_cost_rejection_logs_stage_and_requested_quantity_without_order(caplog, monkeypatch, stage):
    engine = _engine(_risk())
    if stage == "before_live_quantity":
        engine.risk.costs = replace(engine.risk.costs, min_net_edge_pct=0.2)
    else:
        # Model a sizing-time cost change to exercise the existing second gate.
        # Both decisions still use the real cost predicate and estimator.
        def quantity(*_args):
            engine.risk.costs = replace(engine.risk.costs, min_net_edge_pct=0.2)
            return 0.001, {"private_payload": "raw-payload-sentinel"}
        monkeypatch.setattr(engine, "_select_live_entry_quantity", quantity)
    with caplog.at_level(logging.INFO):
        result = engine.evaluate_once()
        assert engine.evaluate_once().status == "no_action"
    assert result.status == "cost_blocked"
    assert engine.position is None and engine.adapter.order_calls == 0
    assert engine.adapter.equity_reads == 1
    messages = _rejections(caplog)
    assert len(messages) == 1
    logged = messages[0]
    assert logged["stage"] == stage
    assert logged["reason"] == "target_net_edge_below_minimum"
    assert logged["net_edge_pct"] < logged["required_pct"]
    if stage == "after_live_quantity": assert logged["quantity"] == 0.001
    assert "raw-payload-sentinel" not in json.dumps(logged)
    assert result.raw["entry_diagnostics"]["stage"] == stage


def test_private_retry_rechecking_risk_does_not_repeat_diagnostic_or_change_retry_state(caplog, monkeypatch):
    engine = _engine(_risk())
    monkeypatch.setattr("btc_futures_bot.engine.time.time", lambda: 1_000)
    engine.cooldown_until = 1_060
    engine._private_entry_retry_signal_timestamp = engine.strategy.signal.timestamp
    engine._private_entry_retry_side = "short"
    engine._private_entry_retry_started_at = 10
    engine._private_entry_retry_deadline = 100
    engine._private_entry_retry_next_at = 0
    monkeypatch.setattr(engine, "_private_entry_retry_pending", lambda _: True)
    monkeypatch.setattr(engine.adapter, "fetch_mark_price", lambda: 100, raising=False)
    original_retry = (engine._private_entry_retry_signal_timestamp, engine._private_entry_retry_deadline)
    with caplog.at_level(logging.INFO):
        assert engine.evaluate_once().status == "risk_blocked"
        assert engine.evaluate_once().status == "risk_blocked"
    assert len(_rejections(caplog)) == 1
    assert original_retry == (engine._private_entry_retry_signal_timestamp, engine._private_entry_retry_deadline)
    assert engine.adapter.order_calls == engine.adapter.equity_reads == 0
