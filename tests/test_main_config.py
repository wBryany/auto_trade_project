from __future__ import annotations

import json

from btc_futures_bot.main import (
    build_engine,
    credential_values,
    ensure_exchange_defaults,
    load_config,
    save_dashboard_config,
)


def _source_config() -> dict:
    return {
        "mode": "paper",
        "active_exchange": "okx",
        "exchanges": {"okx": {"enabled": True}},
        "credentials": {"okx": {}},
        "strategy": {
            "mode": "traditional_kline",
            "trigger_timeframe": "5m",
            "regime_timeframe": "1h",
        },
    }


def test_dashboard_execution_mode_does_not_overwrite_strategy_mode(tmp_path) -> None:
    source = tmp_path / "config.json"
    source.write_text(json.dumps(_source_config()), encoding="utf-8")

    target = save_dashboard_config(source, {"exchange": "okx", "mode": "paper"})
    saved = json.loads(target.read_text(encoding="utf-8"))

    assert saved["mode"] == "paper"
    assert saved["strategy"]["mode"] == "traditional_kline"


def test_dashboard_accepts_one_second_polling_and_clamps_lower_values(tmp_path) -> None:
    source = tmp_path / "config.json"
    source.write_text(json.dumps(_source_config()), encoding="utf-8")

    target = save_dashboard_config(source, {"exchange": "okx", "poll_seconds": 1})
    assert json.loads(target.read_text(encoding="utf-8"))["poll_seconds"] == 1

    target = save_dashboard_config(source, {"exchange": "okx", "poll_seconds": 0})
    assert json.loads(target.read_text(encoding="utf-8"))["poll_seconds"] == 1


def test_dashboard_strategy_mode_requires_explicit_valid_field(tmp_path) -> None:
    source = tmp_path / "config.json"
    source.write_text(json.dumps(_source_config()), encoding="utf-8")

    target = save_dashboard_config(
        source,
        {"exchange": "okx", "mode": "paper", "strategy_mode": "scalp"},
    )
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["strategy"]["mode"] == "scalp"

    try:
        save_dashboard_config(
            source,
            {"exchange": "okx", "mode": "paper", "strategy_mode": "paper"},
        )
    except ValueError as error:
        assert "strategy_mode" in str(error)
    else:
        raise AssertionError("invalid strategy_mode was accepted")


def _tracked_and_local(tmp_path) -> tuple:
    tracked = tmp_path / "config.binance.testnet.json"
    tracked.write_text(
        json.dumps(
            {
                "mode": "paper",
                "active_exchange": "binance",
                "exchanges": {"binance": {"enabled": True, "environment": "testnet"}},
                "strategy": {
                    "mode": "traditional_kline",
                    "min_stop_cost_multiple": 3.0,
                    "traditional_ultra_short_countertrend_allow_long": False,
                    "traditional_ultra_short_reversal_enabled": True,
                    "traditional_ultra_short_reversal_allow_long": False,
                },
            }
        ),
        encoding="utf-8",
    )
    local = tmp_path / "config.binance.testnet.local.json"
    local.write_text(
        json.dumps(
            {
                "mode": "live",
                "active_exchange": "binance",
                "exchanges": {"binance": {"enabled": True, "environment": "production"}},
                "strategy": {"mode": "traditional_kline", "min_score": 4},
            }
        ),
        encoding="utf-8",
    )
    return tracked, local


def test_local_config_inherits_tracked_keys_it_does_not_override(tmp_path) -> None:
    tracked, _local = _tracked_and_local(tmp_path)

    config = load_config(str(tracked))

    # The local file still wins wherever it sets a value...
    assert config["mode"] == "live"
    assert config["exchanges"]["binance"]["environment"] == "production"
    assert config["strategy"]["min_score"] == 4
    # ...but keys it never learned about no longer collapse to the code default.
    assert config["strategy"]["min_stop_cost_multiple"] == 3.0
    assert config["strategy"]["traditional_ultra_short_countertrend_allow_long"] is False
    assert config["strategy"]["traditional_ultra_short_reversal_enabled"] is True
    assert config["strategy"]["traditional_ultra_short_reversal_allow_long"] is False


def test_local_config_path_is_merged_when_passed_directly(tmp_path) -> None:
    _tracked, local = _tracked_and_local(tmp_path)

    config = load_config(str(local))

    assert config["mode"] == "live"
    assert config["strategy"]["traditional_ultra_short_countertrend_allow_long"] is False
    assert config["strategy"]["traditional_ultra_short_reversal_enabled"] is True
    assert config["strategy"]["traditional_ultra_short_reversal_allow_long"] is False


def test_config_without_a_local_companion_is_unchanged(tmp_path) -> None:
    tracked, local = _tracked_and_local(tmp_path)
    local.unlink()

    config = load_config(str(tracked))

    assert config["mode"] == "paper"
    assert config["exchanges"]["binance"]["environment"] == "testnet"


def test_config_extends_parent_without_copying_parent_credentials(tmp_path) -> None:
    parent = tmp_path / "base.json"
    parent.write_text(
        json.dumps(
            {
                "mode": "live",
                "report_dir": "reports/base",
                "credentials": {"binance": {"production": {"api_key": "base-key"}}},
                "exchanges": {"binance": {"enabled": True, "environment": "production"}},
                "strategy": {"mode": "traditional_kline", "min_score": 6},
            }
        ),
        encoding="utf-8",
    )
    child = tmp_path / "model2.json"
    child.write_text(
        json.dumps(
            {
                "extends": "base.json",
                "mode": "paper",
                "report_dir": "reports/model2",
                "credentials": {
                    "binance": {"production": {"api_key": "", "api_secret": ""}}
                },
                "strategy": {"min_score": 7},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(str(child))

    assert config["mode"] == "paper"
    assert config["report_dir"] == "reports/model2"
    assert config["strategy"]["mode"] == "traditional_kline"
    assert config["strategy"]["min_score"] == 7
    assert credential_values(config, "binance").get("api_key", "") == ""


def test_config_extends_cycle_is_rejected(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"extends": "second.json"}), encoding="utf-8")
    second.write_text(json.dumps({"extends": "first.json"}), encoding="utf-8")

    try:
        load_config(str(first))
    except ValueError as error:
        assert "extends cycle" in str(error)
    else:
        raise AssertionError("recursive config inheritance was accepted")


def test_build_engine_attaches_configured_entry_gate(tmp_path, monkeypatch) -> None:
    source = tmp_path / "model2.json"
    source.write_text(
        json.dumps(
            {
                "mode": "paper",
                "report_dir": str(tmp_path / "reports"),
                "active_exchange": "binance",
                "account": {"max_leverage": 3},
                "risk": {},
                "exchanges": {
                    "binance": {"enabled": True, "environment": "testnet"}
                },
                "strategy": {"mode": "traditional_kline"},
                "trade_model": {"type": "lightgbm_meta", "mode": "off"},
            }
        ),
        encoding="utf-8",
    )

    class Adapter:
        name = "binance"

        def close(self) -> None:
            return None

    monkeypatch.setattr("btc_futures_bot.main.make_adapter", lambda *_args: Adapter())
    engine = build_engine("binance", load_config(str(source)))
    try:
        assert engine.entry_gate.status()["mode"] == "off"
        assert engine.entry_gate.config.strategy_config_hash
    finally:
        engine.close()


def test_binance_custom_credential_prefix_isolated_from_standard_environment() -> None:
    config = {
        "exchanges": {
            "binance": {
                "environment": "production",
                "credential_env_prefix": "BINANCE_MODEL2",
            }
        }
    }

    exchange = ensure_exchange_defaults(config, "binance")

    assert exchange["api_key_env"] == "BINANCE_MODEL2_API_KEY"
    assert exchange["api_secret_env"] == "BINANCE_MODEL2_API_SECRET"


def test_binance_cross_exchange_symbol_and_old_endpoint_are_corrected() -> None:
    config = {
        "exchanges": {
            "binance": {
                "environment": "testnet",
                "base_url": "https://testnet.binancefuture.com",
                "symbol": "BTC-USDT-SWAP",
            }
        }
    }

    exchange = ensure_exchange_defaults(config, "binance")

    assert exchange["symbol"] == "BTCUSDT"
    assert exchange["base_url"] == "https://demo-fapi.binance.com"
    assert exchange["api_key_env"] == "BINANCE_TESTNET_API_KEY"


def test_binance_production_credentials_are_isolated(tmp_path) -> None:
    config = {
        "mode": "paper",
        "active_exchange": "binance",
        "exchanges": {"binance": {"enabled": True, "environment": "testnet"}},
        "credentials": {
            "binance": {
                "testnet": {"api_key": "test-key", "api_secret": "test-secret"}
            }
        },
        "strategy": {"mode": "traditional_kline"},
    }
    source = tmp_path / "config.json"
    source.write_text(json.dumps(config), encoding="utf-8")

    target = save_dashboard_config(
        source,
        {
            "exchange": "binance",
            "environment": "production",
            "mode": "paper",
            "symbol": "BTC-USDT-SWAP",
            "api_key": "live-key",
            "api_secret": "live-secret",
        },
    )
    saved = json.loads(target.read_text(encoding="utf-8"))

    assert saved["exchanges"]["binance"]["environment"] == "production"
    assert saved["exchanges"]["binance"]["base_url"] == "https://fapi.binance.com"
    assert saved["exchanges"]["binance"]["symbol"] == "BTCUSDT"
    assert saved["credentials"]["binance"]["testnet"]["api_key"] == "test-key"
    assert credential_values(saved, "binance")["api_key"] == "live-key"
