from __future__ import annotations

from typing import Any

import pytest

from btc_futures_bot.exchanges.base import ExchangeSettings
from btc_futures_bot.exchanges.binance import BinanceAdapter
from btc_futures_bot.http_client import ApiError


MARGIN = ("POST", "/fapi/v1/marginType")
LEVERAGE = ("POST", "/fapi/v1/leverage")
POSITION_MODE = ("GET", "/fapi/v1/positionSide/dual")


def _position(**changes: Any) -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "0",
        "isolated": True, "leverage": "20", **changes,
    }


def _preflight(monkeypatch, positions=None, *, margin_mode="isolated", dual=False):
    adapter = BinanceAdapter(ExchangeSettings(
        name="binance", environment="testnet", base_url="https://demo-fapi.binance.com",
        symbol="BTCUSDT", margin_mode=margin_mode,
    ))
    monkeypatch.setattr(adapter, "has_credentials", lambda: True)
    monkeypatch.setattr(adapter, "symbol_rules", lambda: {})
    private = {"positions": [_position()] if positions is None else positions,
               "orders": [], "algo_orders": []}
    snapshot_calls = []

    def snapshot(**kwargs):
        snapshot_calls.append(kwargs)
        return private

    monkeypatch.setattr(adapter, "_private_snapshot", snapshot)
    calls = []

    def signed(method, path, params=None):
        calls.append((method, path, params))
        if (method, path) == POSITION_MODE:
            return {"dualSidePosition": dual}
        if (method, path) == LEVERAGE:
            return {"leverage": params["leverage"]}
        return {}

    monkeypatch.setattr(adapter, "_signed", signed)
    return adapter, private, calls, snapshot_calls


@pytest.mark.parametrize("isolated,margin_mode,margin_type", [
    (True, "isolated", None), ("true", "isolated", "isolated"),
    (False, "crossed", None), ("false", "crossed", "cross"),
    (False, "crossed", "CROSSED"),
])
def test_matching_bootstrap_settings_skip_both_configuration_posts(
    monkeypatch, isolated, margin_mode, margin_type,
):
    position = _position(isolated=isolated)
    if margin_type is not None:
        position["marginType"] = margin_type
    adapter, _, calls, snapshot_calls = _preflight(
        monkeypatch, [position, _position(symbol="ETHUSDT", isolated=False, leverage="3")],
        margin_mode=margin_mode,
    )

    result = adapter.prepare_live(max_leverage=20)

    assert [(method, path) for method, path, _ in calls] == [POSITION_MODE]
    assert snapshot_calls == [{"wait_seconds": 15.0}]
    assert result["flat"] is True
    assert result["leverage"] == 20
    assert result["margin_mode"] == margin_mode


@pytest.mark.parametrize("changes,expected", [
    ({"isolated": False}, [MARGIN]),
    ({"leverage": "10"}, [LEVERAGE]),
    ({"isolated": False, "leverage": "10"}, [MARGIN, LEVERAGE]),
    ({"isolated": None}, [MARGIN]),
    ({"isolated": 1}, [MARGIN]),
    ({"isolated": "unknown"}, [MARGIN]),
    ({"marginType": "cross"}, [MARGIN]),
    ({"marginType": None}, [MARGIN]),
    ({"marginType": "unknown"}, [MARGIN]),
    ({"leverage": None}, [LEVERAGE]),
    ({"leverage": True}, [LEVERAGE]),
    ({"leverage": "invalid"}, [LEVERAGE]),
    ({"leverage": "NaN"}, [LEVERAGE]),
    ({"leverage": "Infinity"}, [LEVERAGE]),
    ({"leverage": "20.5"}, [LEVERAGE]),
])
def test_only_unknown_or_mismatched_setting_is_written(monkeypatch, changes, expected):
    adapter, _, calls, _ = _preflight(monkeypatch, [_position(**changes)])

    adapter.prepare_live(max_leverage=20)

    assert [(method, path) for method, path, _ in calls] == [POSITION_MODE, *expected]
    for method, path, params in calls:
        if (method, path) == MARGIN:
            assert params == {"symbol": "BTCUSDT", "marginType": "ISOLATED"}
        if (method, path) == LEVERAGE:
            assert params == {"symbol": "BTCUSDT", "leverage": 20}


@pytest.mark.parametrize("missing,expected", [("isolated", MARGIN), ("leverage", LEVERAGE)])
def test_absent_setting_retains_original_configuration_request(monkeypatch, missing, expected):
    position = _position()
    del position[missing]
    adapter, _, calls, _ = _preflight(monkeypatch, [position])

    adapter.prepare_live(max_leverage=20)

    assert [(method, path) for method, path, _ in calls] == [POSITION_MODE, expected]


@pytest.mark.parametrize("positions", [
    [], [_position(symbol="ETHUSDT")], [_position(positionSide="LONG")],
    [_position(positionSide=None)], [_position(), _position()],
    [_position(), _position(positionSide="SHORT")],
])
def test_ambiguous_or_missing_one_way_row_never_skips_settings(monkeypatch, positions):
    adapter, _, calls, _ = _preflight(monkeypatch, positions)

    adapter.prepare_live(max_leverage=20)

    assert [(method, path) for method, path, _ in calls] == [POSITION_MODE, MARGIN, LEVERAGE]


def test_switching_position_mode_invalidates_old_bootstrap_settings(monkeypatch):
    adapter, _, calls, _ = _preflight(monkeypatch, dual=True)

    adapter.prepare_live(max_leverage=20)

    assert [(method, path) for method, path, _ in calls] == [
        POSITION_MODE, ("POST", "/fapi/v1/positionSide/dual"), MARGIN, LEVERAGE,
    ]


def test_unhealthy_private_snapshot_still_fails_before_configuration(monkeypatch):
    adapter, _, calls, _ = _preflight(monkeypatch)
    error = ApiError("account bootstrap rate limited", status_code=429, retry_at=123456)

    def unavailable(**kwargs):
        raise error

    monkeypatch.setattr(adapter, "_private_snapshot", unavailable)
    with pytest.raises(ApiError) as caught:
        adapter.prepare_live(max_leverage=20)

    assert caught.value is error
    assert calls == []


def test_failed_margin_setting_still_fails_closed(monkeypatch):
    adapter, _, calls, _ = _preflight(monkeypatch, [_position(isolated=False, leverage="10")])
    original = adapter._signed

    def signed(method, path, params=None):
        if (method, path) == MARGIN:
            raise ApiError("rate limited", status_code=429, retry_at=123456)
        return original(method, path, params)

    monkeypatch.setattr(adapter, "_signed", signed)
    with pytest.raises(ApiError, match="rate limited"):
        adapter.prepare_live(max_leverage=20)
    assert [(method, path) for method, path, _ in calls] == [POSITION_MODE]


def test_unchanged_margin_response_still_allows_leverage_confirmation(monkeypatch):
    adapter, _, calls, _ = _preflight(monkeypatch, [{}])
    original = adapter._signed

    def signed(method, path, params=None):
        if (method, path) == MARGIN:
            raise ApiError('HTTP 400: {"code": -4046, "msg": "No need to change margin type."}')
        return original(method, path, params)

    monkeypatch.setattr(adapter, "_signed", signed)
    assert adapter.prepare_live(max_leverage=20)["flat"] is True
    assert [(method, path) for method, path, _ in calls] == [POSITION_MODE, LEVERAGE]


def test_requested_leverage_must_still_be_confirmed(monkeypatch):
    adapter, _, _, _ = _preflight(monkeypatch, [_position(leverage="10")])
    original = adapter._signed
    monkeypatch.setattr(adapter, "_signed", lambda method, path, params=None: (
        {"leverage": 10} if (method, path) == LEVERAGE else original(method, path, params)
    ))

    with pytest.raises(RuntimeError, match="did not confirm the requested leverage"):
        adapter.prepare_live(max_leverage=20)


def test_existing_exposure_is_checked_before_flat_settings_optimization(monkeypatch):
    adapter, private, calls, _ = _preflight(monkeypatch, [_position(positionAmt="0.001")])
    managed = {"position": {"side": "long"}}
    resumed = {"resumed": True}
    received = []

    def resume(*args, **kwargs):
        received.append((args, kwargs))
        return resumed

    monkeypatch.setattr(adapter, "_resume_protected_live_position", resume)
    with pytest.raises(RuntimeError, match="existing position/order found"):
        adapter.prepare_live(max_leverage=20)
    assert adapter.prepare_live(max_leverage=20, managed_position=managed) is resumed
    assert received == [((managed, private["positions"], [], []), {"max_leverage": 20})]
    assert calls == []
