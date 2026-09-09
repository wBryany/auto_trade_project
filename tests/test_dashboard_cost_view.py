from __future__ import annotations

import copy
import shutil
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from btc_futures_bot.costs import CostConfig
from btc_futures_bot.dashboard import DASHBOARD_HTML, DashboardService, _position_cost_view, _position_dict
from btc_futures_bot.models import Position
from btc_futures_bot.risk import RiskManager


def _position(side: str = "long", *, opened_at: int = 1_000_000) -> Position:
    return Position(side, 0.01, 80_000, 79_800 if side == "long" else 80_200, 80_300 if side == "long" else 79_700, opened_at)


def _view(side: str, mark: float, costs: CostConfig | None = None) -> dict:
    return _position_cost_view(
        _position_dict(_position(side), mark), costs or CostConfig(), now_ms=1_600_000,
        cost_source="managed_engine",
    )


@pytest.mark.parametrize("side", ["long", "short"])
def test_unchanged_price_has_zero_gross_and_negative_estimated_net(side: str) -> None:
    view = _view(side, 80_000)
    assert view["unrealized_pnl"] == 0
    assert view["estimated_total_cost"] == pytest.approx(1.12)
    assert view["estimated_net_pnl"] == pytest.approx(-1.12)
    assert view["estimated_net_pnl_pct"] == pytest.approx(-0.0014)
    assert view["estimated_holding_hours"] == pytest.approx(1 / 6)
    assert view["estimated_holding_source"] == "actual_open_time"


@pytest.mark.parametrize("side", ["long", "short"])
def test_estimated_net_at_cost_break_even_is_zero(side: str) -> None:
    first = _view(side, 80_000)
    break_even = first["cost_break_even_price"]
    assert break_even > 80_000 if side == "long" else break_even < 80_000
    assert _view(side, break_even)["estimated_net_pnl"] == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("side", ["long", "short"])
def test_higher_fees_move_break_even_farther_from_entry(side: str) -> None:
    low = _view(side, 80_000, CostConfig(taker_fee_pct=0.0005))
    high = _view(side, 80_000, CostConfig(taker_fee_pct=0.001))
    assert abs(high["cost_break_even_price"] - 80_000) > abs(low["cost_break_even_price"] - 80_000)
    assert high["estimated_net_pnl"] < low["estimated_net_pnl"]


def test_cost_annotation_never_mutates_legacy_fields_or_input() -> None:
    original = _position_dict(_position(), 80_050)
    before = copy.deepcopy(original)
    view = _position_cost_view(original, CostConfig(), now_ms=1_600_000, cost_source="managed_engine")
    assert original == before
    assert {key: view[key] for key in before} == before
    assert view["unrealized_pnl"] == pytest.approx(0.5)
    assert view["estimated_net_pnl"] < 0


def test_managed_estimate_uses_actual_duration_for_funding() -> None:
    costs = CostConfig(expected_holding_hours=0.1)
    view = _position_cost_view(
        _position_dict(_position(), 80_000), costs, now_ms=1_000_000 + 9 * 3_600_000,
        cost_source="managed_engine",
    )
    assert view["estimated_holding_hours"] == 9
    assert view["estimated_funding_fee"] == pytest.approx(0.08)
    assert view["estimated_total_cost"] == pytest.approx(1.20)


def test_unknown_open_time_uses_labeled_expected_hold_and_contract_conversion() -> None:
    raw = {"side": "short", "quantity": 2, "entry_price": 80_000, "mark_price": 80_000, "unrealized_pnl": 0}
    costs = CostConfig(expected_holding_hours=8)
    view = _position_cost_view(raw, costs, now_ms=1_000_000, cost_source="configured_exchange", quantity_multiplier=0.01)
    assert view["quantity"] == 2
    assert view["estimated_holding_source"] == "configured_expected_hold"
    assert view["estimated_funding_fee"] == pytest.approx(0.16)
    assert view["estimated_total_cost"] == pytest.approx(2.40)


@pytest.mark.parametrize("multiplier", [None, 0, "invalid"])
def test_unknown_contract_size_does_not_fabricate_cost_amount(multiplier) -> None:
    original = _position_dict(_position(), 80_000)
    view = _position_cost_view(original, CostConfig(), now_ms=1_600_000, cost_source="configured_exchange", quantity_multiplier=multiplier)
    assert view["estimated_costs_available"] is False
    assert view["estimated_total_cost"] is None and view["estimated_net_pnl"] is None
    assert view["estimated_cost_error"]
    assert view["unrealized_pnl"] == 0


def _service(*, engine=None, config=None, snapshot=None) -> DashboardService:
    service = DashboardService.__new__(DashboardService)
    service._config = lambda: config or {
        "mode": "paper", "active_exchange": "okx", "exchanges": {"okx": {"symbol": "BTC-USDT-SWAP", "environment": "demo"}},
    }
    service._thread = None
    service._stop_event = None
    service._lock = threading.RLock()
    service.engine = engine
    service.exchange_name = "okx"
    service.last_result = None
    service.last_error = ""
    service.last_cycle_at = 0.0
    service.started_at = 0.0
    service.notifier = None
    service._market_snapshot = lambda _config, _exchange: snapshot or {"market": {"mark_price": 80_000}}
    service._with_live_market = lambda _config, _exchange, value: value
    return service


def test_stopped_retained_position_uses_original_engine_costs_not_changed_config() -> None:
    engine = SimpleNamespace(
        config=SimpleNamespace(mode="paper"), position=_position(),
        risk=RiskManager(costs=CostConfig(taker_fee_pct=0.0005)),
        session_pnl=-1, consecutive_losses=1, cooldown_until=0,
        macro_risk=None, entry_gate=None,
    )
    config = {
        "mode": "paper", "active_exchange": "okx", "costs": {"taker_fee_pct": 0.01},
        "exchanges": {"okx": {"symbol": "BTC-USDT-SWAP", "environment": "demo", "costs": {"taker_fee_pct": 0.02}}},
    }
    service = _service(engine=engine, config=config)
    with patch("btc_futures_bot.dashboard.time.time", return_value=1600):
        status = service.status()
    assert status["paper_position_retained"] is True
    assert status["cost_assumptions"]["source"] == "managed_engine"
    assert status["cost_assumptions"]["fee_pct_per_side"] == 0.0005
    assert status["positions"][0]["estimated_total_cost"] == pytest.approx(1.12)
    assert status["positions"][0]["estimated_holding_hours"] == pytest.approx(1 / 6)
    assert status["account"]["wallet_balance"] == 9999  # Net estimate is not a second account debit.


def test_exchange_estimate_uses_venue_override_and_does_not_debit_account() -> None:
    config = {
        "mode": "live", "active_exchange": "binance", "costs": {"taker_fee_pct": 0.009},
        "exchanges": {"binance": {"symbol": "BTCUSDT", "environment": "production", "costs": {"taker_fee_pct": 0.001}}},
    }
    snapshot = {
        "market": {"mark_price": 80_000}, "account": {"wallet_balance": 1000, "margin_balance": 1000, "unrealized_pnl": 0},
        "positions": [{"symbol": "BTCUSDT", "side": "long", "quantity": 0.01, "entry_price": 80_000, "mark_price": 80_000, "unrealized_pnl": 0}],
    }
    before = copy.deepcopy(snapshot)
    status = _service(config=config, snapshot=snapshot).status()
    assert status["positions"][0]["estimated_net_pnl"] == pytest.approx(-1.92)
    assert status["cost_assumptions"]["fee_pct_per_side"] == 0.001
    assert status["cost_assumptions"]["estimated_only"] is True
    assert status["account"]["wallet_balance"] == status["account"]["margin_balance"] == 1000
    assert status["account"]["unrealized_pnl"] == 0
    assert snapshot == before


def test_empty_positions_still_expose_cost_assumptions() -> None:
    status = _service().status()
    assert status["positions"] == []
    assert status["cost_assumptions"]["round_trip_pct_excluding_funding"] == pytest.approx(0.0014)
    assert status["cost_assumptions"]["estimated_only"] is True


def test_unmanaged_okx_unknown_contract_size_shows_unavailable_not_btc_assumption() -> None:
    snapshot = {"market": {"mark_price": 80_000}, "positions": [{"symbol": "BTC-USDT-SWAP", "side": "long", "quantity": 2, "entry_price": 80_000, "mark_price": 80_000}]}
    status = _service(snapshot=snapshot).status()
    position = status["positions"][0]
    assert position["estimated_costs_available"] is False
    assert "合约面值" in position["estimated_cost_error"]


def test_legacy_position_dict_signature_stays_usable() -> None:
    assert _position_dict(None) is None
    view = _position_dict(_position(), 80_100, mode="paper")
    assert view["unrealized_pnl"] == 1.0
    assert view["entry_price"] == 80_000
    assert view["take_profit_mode"] == "fixed"


def test_cost_ui_labels_and_embedded_javascript_syntax() -> None:
    assert "毛浮盈（未扣费用）" in DASHBOARD_HTML
    assert "预计平仓净盈亏" in DASHBOARD_HTML
    assert "预计双边总成本" in DASHBOARD_HTML
    assert "含费保本价" in DASHBOARD_HTML
    assert "不是交易所实扣，不修改账户余额" in DASHBOARD_HTML
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable for syntax verification")
    javascript = DASHBOARD_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    checked = subprocess.run([node, "--check"], input=javascript, text=True, capture_output=True, encoding="utf-8")
    assert checked.returncode == 0, checked.stderr
