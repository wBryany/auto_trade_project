import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from btc_futures_bot import http_client
from btc_futures_bot.costs import CostConfig
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.exchanges.base import ExchangeSettings
from btc_futures_bot.exchanges.binance import BinanceAdapter
from btc_futures_bot.http_client import ApiError
from btc_futures_bot.models import Position, Signal
from btc_futures_bot.risk import RiskConfig, RiskManager
from btc_futures_bot.strategy import MultiTimeframeStrategy, StrategyConfig


class Adapter:
    name = "binance"
    settings = SimpleNamespace(symbol="BTCUSDT", environment="production")

    def __init__(self):
        self.remote = {"side": "short", "quantity": 1.0, "entry_price": 100.0}
        self.orders = []
        self.outcomes = []
        self.reads = 0
        self.cancellations = 0
        self.fill_error = None

    def fetch_live_position(self):
        self.reads += 1
        return self.remote

    def place_market_order(self, request):
        self.orders.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        if self.fill_error is None:
            self.remote = None
        return {"status": "FILLED", "executedQty": "1", "avgPrice": "99"}

    def market_fill(self, payload, *, fallback_price):
        if self.fill_error is not None:
            raise self.fill_error
        return 1.0, 99.0

    def fetch_protection_status(self, position):
        return {}

    def cancel_protection_orders(self, position):
        self.cancellations += 1
        return {}


@pytest.fixture
def clock(monkeypatch):
    clock = {"wall": 1_800_000_000.0, "mono": 100.0}
    monkeypatch.setattr("btc_futures_bot.engine.time.time", lambda: clock["wall"])
    monkeypatch.setattr("btc_futures_bot.engine.time.monotonic", lambda: clock["mono"])
    monkeypatch.setattr(http_client, "urlopen", lambda *_a, **_kw: pytest.fail("unexpected network"))
    return clock


def make_engine(tmp_path, clock):
    adapter = Adapter()
    engine = TradingEngine(
        adapter, MultiTimeframeStrategy(StrategyConfig(
            enable_time_exit=False, enable_profit_trend_exit=False,
            break_even_trigger_r=100, trailing_trigger_r=100,
        )), RiskManager(RiskConfig(), costs=CostConfig()),
        EngineConfig(mode="live", reconciliation_state_path=str(tmp_path / "state.json")),
    )
    engine.position = Position(
        "short", 1, 100, 110, 80, int(clock["wall"] * 1000) - 60_000,
        initial_stop_price=110, best_price=100, worst_price=100,
        entry_order_id="entry-1", entry_client_id="client-1",
        stop_order_id="stop-1", stop_client_id="stop-client-1",
    )
    engine.position_signal = Signal("short", 6, 1, ())
    engine._save_live_reconciliation_state()
    return engine


def deferred(clock, kind="host"):
    return ApiError(
        "rate limit active for https://fapi.binance.com" if kind == "host"
        else "Binance REST deferred locally: preventive request budget",
        status_code=418 if kind == "host" else None,
        api_code=None if kind == "host" else "LOCAL_REQUEST_BUDGET",
        retry_at=clock["wall"] + 20, request_not_sent=True,
    )


def saved(engine):
    return json.loads(Path(engine.config.reconciliation_state_path).read_text(encoding="utf-8"))


def tick(engine, price=100):
    return engine._manage_live_position(price, {}, Signal("flat", 0, 0, ()))


def arm(engine, clock, kind="host"):
    engine.adapter.outcomes.append(deferred(clock, kind))
    with pytest.raises(ApiError):
        engine._close_live_position(111, "dynamic_stop_loss")
    assert engine._pending_live_exit is not None


@pytest.mark.parametrize("kind", ["host", "budget"])
def test_unsent_exit_survives_recovered_price_and_resumes_only_after_fresh_reconciliation(tmp_path, clock, caplog, kind):
    engine = make_engine(tmp_path, clock)
    with caplog.at_level("INFO"):
        arm(engine, clock, kind)
    assert "live exit request reason=dynamic_stop_loss" in caplog.text
    assert saved(engine)["managed_position"]["pending_exit"]["decision_price"] == 111
    assert tick(engine).status == "live_exit_retry_wait"
    assert len(engine.adapter.orders) == 1
    assert engine.adapter.cancellations == 0
    reads = engine.adapter.reads
    clock["wall"] += 21
    result = tick(engine)  # Price no longer satisfies the original stop.
    assert result.status == "live_active_exit"
    assert result.raw["exit_reason"] == "dynamic_stop_loss"
    assert result.raw["original_decision_mark_price"] == 111
    assert engine.adapter.reads >= reads + 2  # Pre-submit and filled/flat checks.
    assert len(engine.adapter.orders) == 2
    assert all(order.reduce_only for order in engine.adapter.orders)
    assert engine.adapter.cancellations == 1
    assert engine.position is None and engine._pending_live_exit is None
    assert saved(engine)["managed_position"] is None
    assert tick(engine) is None
    assert len(engine.adapter.orders) == 2


def test_pending_exit_survives_restart_of_same_protected_position(tmp_path, clock):
    engine = make_engine(tmp_path, clock)
    arm(engine, clock)
    resumed = TradingEngine(engine.adapter, engine.strategy, engine.risk, engine.config)
    resumed._restore_managed_live_position()
    assert resumed._pending_live_exit == engine._pending_live_exit
    clock["wall"] += 21
    assert tick(resumed).raw["exit_reason"] == "dynamic_stop_loss"
    assert len(engine.adapter.orders) == 2


@pytest.mark.parametrize("exact_price", [100.0, 100.25])
def test_same_entry_late_fill_metadata_preserves_unsent_exit_across_restart(tmp_path, clock, exact_price):
    engine = make_engine(tmp_path, clock)
    arm(engine, clock)
    original = engine.position
    pending = dict(engine._pending_live_exit)
    engine.adapter.fetch_entry_fill = lambda position: {
        "order_id": original.entry_order_id, "side": "SELL", "quantity": 1.0,
        "price": exact_price, "timestamp": original.opened_at + 1,
    }
    engine._reconcile_managed_entry_fill()
    assert engine.position.opened_at == original.opened_at + 1
    assert engine.position.entry_price == exact_price
    persisted = saved(engine)["managed_position"]["pending_exit"]
    assert persisted["position_identity"]["opened_at"] == original.opened_at + 1
    assert persisted["exit_reason"] == pending["exit_reason"]
    assert persisted["decision_price"] == pending["decision_price"]
    assert persisted["retry_at"] == pending["retry_at"]
    engine.adapter.remote["entry_price"] = exact_price
    resumed = TradingEngine(engine.adapter, engine.strategy, engine.risk, engine.config)
    resumed._restore_managed_live_position()
    clock["wall"] += 21
    assert tick(resumed, 100).raw["original_decision_mark_price"] == 111
    assert len(engine.adapter.orders) == 2


def test_late_fill_for_different_entry_cannot_rewrite_pending_exit_identity(tmp_path, clock):
    engine = make_engine(tmp_path, clock)
    arm(engine, clock)
    original = engine.position
    pending = dict(engine._pending_live_exit)
    engine.adapter.fetch_entry_fill = lambda position: {
        "order_id": "other-order", "side": "SELL", "quantity": 1.0,
        "price": 105, "timestamp": original.opened_at + 1,
    }
    engine._reconcile_managed_entry_fill()
    assert engine.position is original
    assert saved(engine)["managed_position"]["pending_exit"] == pending
    assert not engine._entry_fill_complete
    assert len(engine.adapter.orders) == 1


def test_exchange_hard_stop_wins_while_waiting_no_second_order(tmp_path, clock):
    engine = make_engine(tmp_path, clock)
    arm(engine, clock)
    engine.adapter.remote = None
    clock["wall"] += 21
    assert tick(engine).status == "live_exit_reconciled"
    assert len(engine.adapter.orders) == 1
    assert engine.position is None and engine._pending_live_exit is None
    assert saved(engine)["managed_position"] is None


@pytest.mark.parametrize("field,value", [("entry_order_id", "entry-2"), ("opened_at", 1), ("quantity", 2.0), ("side", "long")])
def test_deferred_exit_never_reused_for_another_local_position(tmp_path, clock, field, value):
    engine = make_engine(tmp_path, clock)
    arm(engine, clock)
    engine.position = replace(engine.position, **{field: value})
    clock["wall"] += 21
    assert tick(engine) is None
    assert engine._pending_live_exit is None
    assert saved(engine)["managed_position"]["pending_exit"] is None
    assert len(engine.adapter.orders) == 1


@pytest.mark.parametrize("remote", [
    {"side": "long", "quantity": 1, "entry_price": 100},
    {"side": "short", "quantity": 2, "entry_price": 100},
    {"side": "short", "quantity": 1, "entry_price": 105},
])
def test_different_remote_position_invalidates_deferred_exit_without_order(tmp_path, clock, remote):
    engine = make_engine(tmp_path, clock)
    arm(engine, clock)
    engine.adapter.remote = remote
    clock["wall"] += 21
    with pytest.raises(RuntimeError, match="differs"):
        tick(engine)
    assert engine._pending_live_exit is None
    assert len(engine.adapter.orders) == 1


def test_repeated_local_gate_preserves_original_reason_and_decision(tmp_path, clock):
    engine = make_engine(tmp_path, clock)
    arm(engine, clock)
    clock["wall"] += 21
    engine.adapter.outcomes.append(deferred(clock))
    with pytest.raises(ApiError):
        tick(engine, 99)
    pending = saved(engine)["managed_position"]["pending_exit"]
    assert pending["exit_reason"] == "dynamic_stop_loss"
    assert pending["decision_price"] == 111
    assert pending["retry_at"] == clock["wall"] + 20


@pytest.mark.parametrize("error", [
    ApiError("network error POST order"),
    ApiError("HTTP 503 POST order", status_code=503),
    ApiError("HTTP 418 POST order", status_code=418),
    ApiError("HTTP 429 POST order", status_code=429),
])
def test_uncertain_submission_or_actual_exchange_rejection_never_creates_retry_permission(tmp_path, clock, error):
    engine = make_engine(tmp_path, clock)
    engine.adapter.outcomes.append(error)
    with pytest.raises(ApiError):
        engine._close_live_position(111, "dynamic_stop_loss")
    assert engine._pending_live_exit is None
    assert tick(engine) is None
    assert len(engine.adapter.orders) == 1


@pytest.mark.parametrize("stage", ["submit_timeout", "fill_lookup_local_gate"])
def test_retry_consumption_precedes_submission_and_does_not_survive_uncertain_outcome(tmp_path, clock, stage):
    engine = make_engine(tmp_path, clock)
    arm(engine, clock)
    clock["wall"] += 21
    if stage == "submit_timeout":
        engine.adapter.outcomes.append(ApiError("network error POST order"))
    else:
        engine.adapter.fill_error = deferred(clock)
    with pytest.raises(ApiError):
        tick(engine)
    assert engine._pending_live_exit is None
    assert saved(engine)["managed_position"]["pending_exit"] is None
    assert tick(engine) is None
    assert len(engine.adapter.orders) == 2


def test_failed_retry_consumption_cannot_leave_replayable_disk_state_after_submission(tmp_path, clock, monkeypatch):
    engine = make_engine(tmp_path, clock)
    arm(engine, clock)
    clock["wall"] += 21
    original_replace = Path.replace
    monkeypatch.setattr(Path, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(RuntimeError, match="could not persist"):
        tick(engine)
    assert len(engine.adapter.orders) == 1
    assert engine.adapter.cancellations == 0
    assert engine._pending_live_exit is not None
    monkeypatch.setattr(Path, "replace", original_replace)
    assert tick(engine).status == "live_active_exit"
    assert len(engine.adapter.orders) == 2


def test_real_clock_http_418_keeps_original_deadline_and_latches_only_unsent_close(tmp_path, clock, monkeypatch):
    engine = make_engine(tmp_path, clock)
    remote = engine.adapter.remote
    venue = BinanceAdapter(ExchangeSettings("binance", "production", "https://fapi.binance.com", "BTCUSDT"))
    monkeypatch.setattr(venue, "credentials", lambda: ("test-key", "test-secret", ""))
    monkeypatch.setattr(venue, "fetch_live_position", lambda: remote)
    venue._server_time_anchor_ms = int(clock["wall"] * 1000)
    venue._server_time_anchor_monotonic = clock["mono"]
    venue._server_time_synced_at = clock["mono"] - 1_000
    engine.adapter = venue
    deadline_ms = int((clock["wall"] + 20) * 1000)
    calls = []
    def reject_clock(request, **kwargs):
        calls.append((request.method, request.full_url))
        assert request.full_url.endswith("/time")
        detail = json.dumps({"code": -1003, "msg": f"IP(198.51.100.10) banned until {deadline_ms}"}).encode()
        raise HTTPError(request.full_url, 418, "banned", {}, BytesIO(detail))
    http_client.clear_rate_limits()
    monkeypatch.setattr(http_client, "urlopen", reject_clock)
    try:
        with pytest.raises(ApiError) as caught:
            engine._close_live_position(111, "dynamic_stop_loss")
        assert caught.value.status_code == 418
        assert caught.value.request_not_sent is True
        assert caught.value.retry_at == deadline_ms / 1000
        assert "HTTP 418 GET" in str(caught.value) and "198.51.100.10" in str(caught.value)
        assert calls == [("GET", "https://fapi.binance.com/fapi/v1/time")]
        assert saved(engine)["managed_position"]["pending_exit"]["retry_at"] == deadline_ms / 1000
    finally:
        http_client.clear_rate_limits()
