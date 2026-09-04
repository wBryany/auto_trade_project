from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import btc_futures_bot.trade_model.policy as policy_module

from btc_futures_bot.models import Candle, Signal
from btc_futures_bot.strategy import StrategyConfig
from btc_futures_bot.trade_model import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_HASH,
    MANIFEST_VERSION,
    MODEL_TYPE,
    TRAINING_PIPELINE_VERSION,
    EntryGate,
    FeatureValidationError,
    MetaModelConfig,
    build_execution_policy,
    execution_policy_hash,
    extract_features,
    strategy_config_hash,
    normalized_source_sha256,
)
from btc_futures_bot.trade_model.decision_log import DecisionLog


_INTERVALS = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}
_STRATEGY = {"mode": "traditional_kline", "min_score": 6}


def _base_runtime_config() -> dict[str, Any]:
    return {
        "instance_id": "model2-test",
        "mode": "paper",
        "active_exchange": "binance",
        "exchanges": {
            "binance": {"symbol": "BTCUSDT", "environment": "testnet"}
        },
        "strategy": deepcopy(_STRATEGY),
        "trade_model": {},
    }


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


class FakePredictor:
    def __init__(self, score: float | Exception) -> None:
        self.score = score
        self.calls = 0
        self.rows: list[tuple[float, ...]] = []

    def predict(self, features: Any) -> float:
        self.calls += 1
        self.rows.append(tuple(float(value) for value in features))
        if isinstance(self.score, Exception):
            raise self.score
        return self.score


def _market(end_ms: int = 100 * 3_600_000) -> dict[str, list[Candle]]:
    market: dict[str, list[Candle]] = {}
    for timeframe, interval_ms in _INTERVALS.items():
        rows: list[Candle] = []
        for index in range(24):
            timestamp = end_ms - (24 - index) * interval_ms
            open_price = 100.0 + index * 0.2
            close = open_price + 0.1
            volume = 10.0 if index < 23 else 25.0
            rows.append(
                Candle(
                    timestamp,
                    open_price,
                    close + 0.1,
                    open_price - 0.1,
                    close,
                    volume,
                    quote_volume=volume * close,
                )
            )
        market[timeframe] = rows
    return market


def _write_artifact(
    root: Path,
    *,
    approved_for_live: bool = True,
    threshold: float = 0.65,
    manifest_updates: dict[str, Any] | None = None,
    policy_config: dict[str, Any] | None = None,
    barrier_config: dict[str, Any] | None = None,
    closed_history_limit: int | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model.txt"
    model_path.write_text("tree\nnot-a-real-model-but-hashable\n", encoding="utf-8")
    policy = build_execution_policy(
        policy_config or _base_runtime_config(),
        barrier_config=barrier_config,
        closed_history_limit=closed_history_limit,
    )
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "model_type": MODEL_TYPE,
        "training_pipeline_version": TRAINING_PIPELINE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "schema_hash": FEATURE_SCHEMA_HASH,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_version": "meta-test-v1",
        "threshold": threshold,
        "approved_for_live": approved_for_live,
        "strategy_config_hash": strategy_config_hash(_STRATEGY),
        "execution_policy_hash": execution_policy_hash(policy),
        "live_execution_equivalent": approved_for_live,
        "live_approval_blockers": [] if approved_for_live else ["test blocker"],
    }
    manifest.update(manifest_updates or {})
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return model_path, manifest_path, manifest


def _config(
    root: Path,
    mode: str,
    *,
    approved_for_live: bool = True,
    threshold_override: float | None = None,
    manifest_updates: dict[str, Any] | None = None,
    runtime_updates: dict[str, Any] | None = None,
    training_barrier_config: dict[str, Any] | None = None,
    training_closed_history_limit: int | None = None,
) -> MetaModelConfig:
    runtime_config = _base_runtime_config()
    if runtime_updates:
        _deep_update(runtime_config, runtime_updates)
    model_path, manifest_path, _manifest = _write_artifact(
        root / "artifact",
        approved_for_live=approved_for_live,
        manifest_updates=manifest_updates,
        policy_config=_base_runtime_config(),
        barrier_config=training_barrier_config,
        closed_history_limit=training_closed_history_limit,
    )
    section: dict[str, Any] = {
        "type": "lightgbm_meta",
        "mode": mode,
        "artifact_path": str(model_path.relative_to(root)),
        "manifest_path": str(manifest_path.relative_to(root)),
        "decision_log_path": "audit/decisions.sqlite3",
        "require_approved_for_live": True,
    }
    if threshold_override is not None:
        section["threshold_override"] = threshold_override
    runtime_config["trade_model"] = section
    return MetaModelConfig.from_mapping(runtime_config, root)


def test_config_parses_full_mapping_and_rejects_live_threshold_override(
    tmp_path: Path,
) -> None:
    model_path, manifest_path, _manifest = _write_artifact(tmp_path / "models")
    mapping = {
        "mode": "paper",
        "strategy": _STRATEGY,
        "trade_model": {
            "mode": "shadow",
            "artifact_path": str(model_path.relative_to(tmp_path)),
            "manifest_path": str(manifest_path.relative_to(tmp_path)),
            "decision_log_path": "reports/meta.sqlite3",
            "threshold_override": 0.7,
            "require_approved_for_live": "true",
        },
    }

    parsed = MetaModelConfig.from_mapping(mapping, tmp_path)

    assert parsed.mode == "shadow"
    assert parsed.execution_mode == "paper"
    assert parsed.artifact_path == model_path.resolve()
    assert parsed.manifest_path == manifest_path.resolve()
    assert parsed.decision_log_path == (tmp_path / "reports/meta.sqlite3").resolve()
    assert parsed.strategy_config_hash == strategy_config_hash(_STRATEGY)
    assert len(parsed.execution_policy_hash) == 64
    assert parsed.threshold_override == pytest.approx(0.7)

    mapping["mode"] = "live"
    with pytest.raises(ValueError, match="paper/backtest only"):
        MetaModelConfig.from_mapping(mapping, tmp_path)


def test_full_legacy_config_without_trade_model_defaults_to_off(tmp_path: Path) -> None:
    parsed = MetaModelConfig.from_mapping(
        {
            "mode": "paper",
            "strategy": {},
            "exchanges": {"binance": {}},
        },
        tmp_path,
    )

    assert parsed.mode == "off"
    assert parsed.execution_mode == "paper"
    assert parsed.artifact_path is None


def test_strategy_hash_materializes_all_defaults() -> None:
    defaults = StrategyConfig()

    assert strategy_config_hash({}) == strategy_config_hash(defaults)
    assert strategy_config_hash({"min_score": defaults.min_score}) == strategy_config_hash(
        defaults
    )
    assert strategy_config_hash({"min_score": defaults.min_score + 1}) != strategy_config_hash(
        defaults
    )

    with pytest.raises(ValueError, match="threshold_override"):
        MetaModelConfig(
            mode="enforce",
            execution_mode="live",
            threshold_override=0.8,
        )


def test_shared_features_are_ordered_directional_and_require_all_closed_histories() -> None:
    market = _market()
    long_features = extract_features(
        Signal("long", 7, market["5m"][-1].timestamp, ("breakout", "volume")),
        market,
    )
    short_features = extract_features(
        Signal("short", 7, market["5m"][-1].timestamp, ("breakout", "volume")),
        market,
    )

    assert tuple(long_features) == FEATURE_NAMES
    assert set(long_features) == set(short_features)
    assert long_features["tf_1m_side_return_5"] == pytest.approx(
        -short_features["tf_1m_side_return_5"]
    )
    assert long_features["tf_1m_quote_volume_ratio_20"] > 1.0
    assert all(isinstance(value, float) for value in long_features.values())

    missing = dict(market)
    missing.pop("1h")
    with pytest.raises(FeatureValidationError, match="missing required timeframe: 1h"):
        extract_features(Signal("long", 6, 1, ()), missing)

    gapped = dict(market)
    bad_5m = list(gapped["5m"])
    candle = bad_5m[-2]
    bad_5m[-2] = Candle(
        candle.timestamp + 1,
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
        candle.quote_volume,
    )
    gapped["5m"] = bad_5m
    with pytest.raises(FeatureValidationError, match="history has a gap"):
        extract_features(Signal("long", 6, 1, ()), gapped)


def test_enforce_accepts_enriches_signal_caches_and_audits_unique_candidate(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "enforce")
    predictor = FakePredictor(0.8)
    gate = EntryGate(config, predictor=predictor)
    market = _market()
    signal = Signal("long", 7, market["5m"][-1].timestamp, ("breakout", "volume"))

    first = gate.evaluate(signal, market)
    second = gate.evaluate(signal, market)

    assert first.accepted is True
    assert first.decision == "enforce_accept"
    assert first.score == pytest.approx(0.8)
    assert first.threshold == pytest.approx(0.65)
    assert first.signal.model_name == "lightgbm_meta"
    assert first.signal.model_version == "meta-test-v1"
    assert first.signal.meta_score == pytest.approx(0.8)
    assert isinstance(first.signal.meta_score, float)
    assert first.signal.meta_threshold == pytest.approx(0.65)
    assert isinstance(first.signal.meta_threshold, float)
    assert first.signal.meta_decision == "enforce_accept"
    assert second is first
    assert predictor.calls == 1
    assert len(predictor.rows[0]) == len(FEATURE_NAMES)
    assert gate.status()["recorded_candidates"] == 1
    gate.ensure_live_ready()

    with sqlite3.connect(config.decision_log_path) as connection:
        candidates = connection.execute(
            "SELECT COUNT(*), accepted, decision FROM entry_candidates"
        ).fetchone()
        feature_count = connection.execute(
            "SELECT COUNT(*) FROM entry_candidate_features"
        ).fetchone()[0]
        provenance = connection.execute(
            """
            SELECT execution_policy_hash, model_sha256, instance_id,
                   exchange_name, symbol, exchange_environment
            FROM entry_candidates
            """
        ).fetchone()
    assert candidates == (1, 1, "enforce_accept")
    assert feature_count == len(FEATURE_NAMES)
    assert provenance == (
        config.execution_policy_hash,
        hashlib.sha256(config.artifact_path.read_bytes()).hexdigest(),
        "model2-test",
        "binance",
        "BTCUSDT",
        "testnet",
    )
    gate.close()


def test_decision_log_migrates_legacy_schema_without_losing_rows(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE entry_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL UNIQUE,
                signal_timestamp INTEGER NOT NULL,
                side TEXT NOT NULL,
                signal_score REAL NOT NULL,
                reasons_json TEXT NOT NULL,
                gate_mode TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                meta_score REAL,
                threshold REAL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                schema_hash TEXT NOT NULL,
                features_json TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                UNIQUE(signal_timestamp, side)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO entry_candidates (
                candidate_id, signal_timestamp, side, signal_score,
                reasons_json, gate_mode, accepted, meta_score, threshold,
                decision, reason, model_name, model_version, schema_hash,
                features_json, created_at_ms
            ) VALUES ('1:long', 1, 'long', 6, '[]', 'shadow', 1, NULL,
                      NULL, 'legacy', 'legacy', 'lightgbm_meta', 'v0', '', '{}', 1)
            """
        )

    decision_log = DecisionLog(path)
    with sqlite3.connect(path) as connection:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(entry_candidates)")
        }
        legacy = connection.execute(
            "SELECT candidate_id, execution_policy_hash FROM entry_candidates"
        ).fetchone()

    assert {
        "execution_policy_hash",
        "model_sha256",
        "instance_id",
        "exchange_name",
        "symbol",
        "exchange_environment",
    } <= columns
    assert legacy == ("1:long", "")
    assert decision_log.record(
        timestamp=2,
        side="short",
        signal_score=7,
        reasons=("candidate",),
        gate_mode="enforce",
        accepted=False,
        meta_score=0.4,
        threshold=0.65,
        decision="enforce_reject",
        reason="below threshold",
        model_name="lightgbm_meta",
        model_version="v1",
        schema_hash=FEATURE_SCHEMA_HASH,
        execution_policy_hash="a" * 64,
        model_sha256="b" * 64,
        instance_id="model2",
        exchange_name="binance",
        symbol="BTCUSDT",
        exchange_environment="production",
        features={"signal_score": 7.0},
    ) is True
    decision_log.close()


def test_enforce_rejects_below_threshold_but_shadow_only_observes(
    tmp_path: Path,
) -> None:
    market = _market()
    signal = Signal("short", 6, market["5m"][-1].timestamp, ("reversal",))

    enforce = EntryGate(_config(tmp_path / "enforce", "enforce"), predictor=FakePredictor(0.4))
    enforced = enforce.evaluate(signal, market)
    assert enforced.accepted is False
    assert enforced.decision == "enforce_reject"
    assert enforced.signal.meta_score == pytest.approx(0.4)

    shadow = EntryGate(_config(tmp_path / "shadow", "shadow"), predictor=FakePredictor(0.4))
    observed = shadow.evaluate(signal, market)
    assert observed.accepted is True
    assert observed.decision == "shadow_reject"
    assert observed.score == pytest.approx(0.4)
    enforce.close()
    shadow.close()


@pytest.mark.parametrize(
    ("mode", "expected_accepted", "expected_decision"),
    [
        ("shadow", True, "shadow_error_accept"),
        ("enforce", False, "enforce_error_reject"),
    ],
)
def test_prediction_exception_is_fail_open_only_in_shadow(
    tmp_path: Path,
    mode: str,
    expected_accepted: bool,
    expected_decision: str,
) -> None:
    market = _market()
    signal = Signal("long", 6, market["5m"][-1].timestamp, ("candidate",))
    gate = EntryGate(
        _config(tmp_path / mode, mode),
        predictor=FakePredictor(RuntimeError("predict exploded")),
    )

    result = gate.evaluate(signal, market)

    assert result.accepted is expected_accepted
    assert result.decision == expected_decision
    assert "predict exploded" in result.reason
    assert result.signal.meta_score == 0.0
    assert isinstance(result.signal.meta_score, float)
    gate.close()


@pytest.mark.parametrize(
    ("manifest_updates", "error_text"),
    [
        ({"schema_hash": "0" * 64}, "schema_hash"),
        ({"model_sha256": "0" * 64}, "model_sha256"),
        ({"strategy_config_hash": "0" * 64}, "strategy_config_hash"),
        ({"feature_names": list(reversed(FEATURE_NAMES))}, "feature_names"),
        ({"manifest_version": "legacy-v0"}, "manifest_version"),
        ({"model_type": "different_model"}, "model_type"),
        ({"training_pipeline_version": "meta-training-v0"}, "training_pipeline_version"),
    ],
)
def test_manifest_integrity_errors_fail_closed_without_calling_predictor(
    tmp_path: Path,
    manifest_updates: dict[str, Any],
    error_text: str,
) -> None:
    predictor = FakePredictor(0.99)
    gate = EntryGate(
        _config(
            tmp_path,
            "enforce",
            manifest_updates=manifest_updates,
        ),
        predictor=predictor,
    )
    market = _market()

    result = gate.evaluate(
        Signal("long", 8, market["5m"][-1].timestamp, ("candidate",)),
        market,
    )

    assert result.accepted is False
    assert result.decision == "enforce_error_reject"
    assert error_text in result.reason
    assert predictor.calls == 0
    assert gate.status()["ready"] is False
    with pytest.raises(RuntimeError, match="not live-ready"):
        gate.ensure_live_ready()
    gate.close()


def test_implementation_source_hash_normalizes_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"value = 1\nprint(value)\n")
    crlf.write_bytes(b"value = 1\r\nprint(value)\r\n")

    assert normalized_source_sha256(lf) == normalized_source_sha256(crlf)
    policy = build_execution_policy(_base_runtime_config())
    source_hashes = policy["implementation_source_sha256"]
    assert set(source_hashes) == set(policy_module.IMPLEMENTATION_SOURCE_PATHS)
    assert all(len(digest) == 64 for digest in source_hashes.values())


def test_implementation_source_change_fails_artifact_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, manifest_path, _manifest = _write_artifact(tmp_path / "artifact")
    changed_hashes = policy_module.implementation_source_hashes()
    changed_hashes["src/btc_futures_bot/engine.py"] = "0" * 64
    monkeypatch.setattr(
        policy_module,
        "implementation_source_hashes",
        lambda: dict(changed_hashes),
    )
    mapping = _base_runtime_config()
    mapping["trade_model"] = {
        "mode": "enforce",
        "artifact_path": str(model_path.relative_to(tmp_path)),
        "manifest_path": str(manifest_path.relative_to(tmp_path)),
        "decision_log_path": "audit/source-change.sqlite3",
    }
    predictor = FakePredictor(0.99)
    gate = EntryGate(
        MetaModelConfig.from_mapping(mapping, tmp_path),
        predictor=predictor,
    )
    market = _market()

    result = gate.evaluate(
        Signal("long", 8, market["5m"][-1].timestamp, ("candidate",)),
        market,
    )

    assert result.accepted is False
    assert "execution_policy_hash" in result.reason
    assert predictor.calls == 0
    gate.close()


@pytest.mark.parametrize(
    ("runtime_updates", "training_barrier", "training_history"),
    [
        ({"risk": {"risk_per_trade": 0.006}}, None, None),
        ({"costs": {"taker_fee_pct": 0.0007}}, None, None),
        ({"candle_limit": 301}, None, None),
        (
            None,
            {
                "take_profit_pct": 0.02,
                "stop_loss_pct": 0.01,
                "horizon_bars": 45,
                "fee_pct_per_side": 0.0005,
                "slippage_pct_per_side": 0.0002,
                "fallback_funding_rate_per_8h": 0.0001,
            },
            None,
        ),
        (None, None, 123),
    ],
    ids=["risk", "cost", "candle-limit", "cli-label-override", "cli-history-override"],
)
def test_execution_policy_mismatch_fails_closed_before_prediction(
    tmp_path: Path,
    runtime_updates: dict[str, Any] | None,
    training_barrier: dict[str, Any] | None,
    training_history: int | None,
) -> None:
    predictor = FakePredictor(0.99)
    gate = EntryGate(
        _config(
            tmp_path,
            "enforce",
            runtime_updates=runtime_updates,
            training_barrier_config=training_barrier,
            training_closed_history_limit=training_history,
        ),
        predictor=predictor,
    )
    market = _market()

    result = gate.evaluate(
        Signal("long", 8, market["5m"][-1].timestamp, ("candidate",)),
        market,
    )

    assert result.accepted is False
    assert result.decision == "enforce_error_reject"
    assert "execution_policy_hash" in result.reason
    assert predictor.calls == 0
    assert gate.status()["ready"] is False
    gate.close()


def test_manifest_missing_execution_policy_hash_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path, "enforce")
    assert config.manifest_path is not None
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    manifest.pop("execution_policy_hash")
    config.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    predictor = FakePredictor(0.99)
    gate = EntryGate(config, predictor=predictor)
    market = _market()

    result = gate.evaluate(
        Signal("long", 8, market["5m"][-1].timestamp, ("candidate",)),
        market,
    )

    assert result.accepted is False
    assert "missing required fields: execution_policy_hash" in result.reason
    assert predictor.calls == 0
    gate.close()


def test_off_and_shadow_never_block_live_but_enforce_requires_approval(
    tmp_path: Path,
) -> None:
    off_config = MetaModelConfig.from_mapping(
        {"mode": "paper", "strategy": _STRATEGY, "trade_model": {"mode": "off"}},
        tmp_path,
    )
    off = EntryGate(off_config)
    result = off.evaluate(Signal("long", 6, 123, ("candidate",)), {})
    assert result.accepted is True
    assert result.decision == "off_accept"
    assert result.signal.meta_score == 0.0
    assert result.signal.meta_threshold == 0.0
    assert off.status()["ready"] is True
    off.ensure_live_ready()
    off.close()

    unavailable_shadow = EntryGate(MetaModelConfig(mode="shadow"))
    unavailable_shadow.ensure_live_ready()
    shadow_result = unavailable_shadow.evaluate(Signal("long", 6, 124, ()), {})
    assert shadow_result.accepted is True
    assert shadow_result.decision == "shadow_error_accept"
    unavailable_shadow.close()

    unapproved_enforce = EntryGate(
        _config(tmp_path / "unapproved", "enforce", approved_for_live=False),
        predictor=FakePredictor(0.8),
    )
    with pytest.raises(RuntimeError, match="not approved_for_live"):
        unapproved_enforce.ensure_live_ready()
    unapproved_enforce.close()


def test_paper_threshold_override_is_used_but_never_live_ready(tmp_path: Path) -> None:
    config = _config(tmp_path, "enforce", threshold_override=0.9)
    gate = EntryGate(config, predictor=FakePredictor(0.8))
    market = _market()

    result = gate.evaluate(
        Signal("long", 7, market["5m"][-1].timestamp, ("candidate",)),
        market,
    )

    assert result.accepted is False
    assert result.threshold == pytest.approx(0.9)
    with pytest.raises(RuntimeError, match="threshold_override"):
        gate.ensure_live_ready()
    gate.close()
