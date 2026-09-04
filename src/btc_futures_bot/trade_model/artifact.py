from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .features import FEATURE_NAMES, FEATURE_SCHEMA_HASH, strategy_config_hash
from .policy import (
    MANIFEST_VERSION,
    MODEL_TYPE,
    TRAINING_PIPELINE_VERSION,
    execution_policy_for_config,
)


SUPPORTED_MODEL_TYPE = MODEL_TYPE
SUPPORTED_GATE_MODES = frozenset({"off", "shadow", "enforce"})


@dataclass(frozen=True)
class MetaModelConfig:
    """Configuration for the independently enforced entry gate."""

    type: str = SUPPORTED_MODEL_TYPE
    mode: str = "off"
    artifact_path: Path | None = None
    manifest_path: Path | None = None
    decision_log_path: Path | None = None
    require_approved_for_live: bool = True
    threshold_override: float | None = None
    strategy_config_hash: str = ""
    execution_policy_hash: str = ""
    execution_mode: str = "paper"
    instance_id: str = ""
    exchange_name: str = "binance"
    symbol: str = "BTCUSDT"
    exchange_environment: str = "testnet"

    def __post_init__(self) -> None:
        if self.type != SUPPORTED_MODEL_TYPE:
            raise ValueError(f"unsupported trade_model type: {self.type}")
        if self.mode not in SUPPORTED_GATE_MODES:
            raise ValueError("trade_model.mode must be off, shadow, or enforce")
        if self.execution_mode not in {"paper", "live", "backtest", "testnet"}:
            raise ValueError(
                "trade_model execution_mode must be paper, live, backtest, or testnet"
            )
        if not isinstance(self.require_approved_for_live, bool):
            raise ValueError("require_approved_for_live must be a boolean")
        if self.threshold_override is not None:
            _required_probability(self.threshold_override, "threshold_override")
            if self.execution_mode == "live":
                raise ValueError(
                    "threshold_override is paper/backtest only and cannot be used live"
                )
        for name in ("artifact_path", "manifest_path", "decision_log_path"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        base_dir: str | Path,
    ) -> "MetaModelConfig":
        """Parse either the full bot config or its ``trade_model`` section."""

        root: Mapping[str, Any] = mapping or {}
        has_section = "trade_model" in root
        direct_section_keys = {
            "type",
            "artifact_path",
            "manifest_path",
            "decision_log_path",
            "require_approved_for_live",
            "threshold_override",
            "strategy_config_hash",
            "execution_policy_hash",
            "execution_mode",
            "runtime_mode",
            "instance_id",
            "exchange_name",
            "symbol",
            "exchange_environment",
        }
        looks_like_full_config = not has_section and (
            "strategy" in root
            or "exchanges" in root
            or str(root.get("mode") or "").strip().lower() in {"paper", "live"}
        )
        is_direct_section = not has_section and (
            not looks_like_full_config
            and (
                bool(direct_section_keys.intersection(root))
                or str(root.get("mode") or "").strip().lower()
                in SUPPORTED_GATE_MODES
            )
        )
        raw_value = root.get("trade_model", {}) if has_section else (root if is_direct_section else {})
        if raw_value is None:
            raw_value = {}
        if not isinstance(raw_value, Mapping):
            raise ValueError("trade_model must be an object")
        raw = raw_value

        model_type = str(raw.get("type") or SUPPORTED_MODEL_TYPE).strip().lower()
        if model_type != SUPPORTED_MODEL_TYPE:
            raise ValueError(f"unsupported trade_model type: {model_type}")
        mode = str(raw.get("mode") or "off").strip().lower()
        if mode not in SUPPORTED_GATE_MODES:
            raise ValueError("trade_model.mode must be off, shadow, or enforce")

        execution_mode = str(
            root.get("mode", "paper")
            if has_section or looks_like_full_config
            else raw.get("execution_mode", raw.get("runtime_mode", "paper"))
        ).strip().lower()
        if execution_mode not in {"paper", "live", "backtest", "testnet"}:
            raise ValueError(
                "trade_model execution_mode must be paper, live, backtest, or testnet"
            )

        root_dir = Path(base_dir).expanduser().resolve()
        artifact_path = _optional_path(raw.get("artifact_path"), root_dir)
        manifest_path = _optional_path(raw.get("manifest_path"), root_dir)
        if manifest_path is None and artifact_path is not None:
            manifest_path = artifact_path.with_name("manifest.json")
        decision_log_path = _optional_path(raw.get("decision_log_path"), root_dir)

        threshold_override = _optional_probability(
            raw.get("threshold_override"), "threshold_override"
        )
        if threshold_override is not None and execution_mode == "live":
            raise ValueError("threshold_override is paper/backtest only and cannot be used live")

        configured_strategy_hash = str(raw.get("strategy_config_hash") or "").strip().lower()
        calculated_strategy_hash = ""
        calculated_execution_policy_hash = ""
        if has_section or looks_like_full_config:
            strategy_value = root.get("strategy", {})
            if not isinstance(strategy_value, Mapping):
                raise ValueError("strategy must be an object")
            calculated_strategy_hash = strategy_config_hash(strategy_value)
            _policy, calculated_execution_policy_hash = execution_policy_for_config(root)
        if (
            configured_strategy_hash
            and calculated_strategy_hash
            and not hmac.compare_digest(configured_strategy_hash, calculated_strategy_hash)
        ):
            raise ValueError(
                "trade_model.strategy_config_hash does not match the current strategy config"
            )
        configured_execution_policy_hash = str(
            raw.get("execution_policy_hash") or ""
        ).strip().lower()
        if (
            configured_execution_policy_hash
            and calculated_execution_policy_hash
            and not hmac.compare_digest(
                configured_execution_policy_hash,
                calculated_execution_policy_hash,
            )
        ):
            raise ValueError(
                "trade_model.execution_policy_hash does not match the current effective config"
            )

        if has_section or looks_like_full_config:
            instance_id = str(root.get("instance_id") or "").strip()
            exchange_name = str(root.get("active_exchange") or "binance").strip().lower()
            exchanges = root.get("exchanges") or {}
            exchange_context = (
                exchanges.get(exchange_name, {}) if isinstance(exchanges, Mapping) else {}
            )
            if not isinstance(exchange_context, Mapping):
                exchange_context = {}
            symbol_default = "BTCUSDT" if exchange_name == "binance" else ""
            environment_default = "testnet" if exchange_name == "binance" else ""
            symbol = str(exchange_context.get("symbol") or symbol_default).strip().upper()
            exchange_environment = str(
                exchange_context.get("environment") or environment_default
            ).strip().lower()
        else:
            instance_id = str(raw.get("instance_id") or "").strip()
            exchange_name = str(raw.get("exchange_name") or "binance").strip().lower()
            symbol = str(raw.get("symbol") or "BTCUSDT").strip().upper()
            exchange_environment = str(
                raw.get("exchange_environment") or "testnet"
            ).strip().lower()

        return cls(
            type=model_type,
            mode=mode,
            artifact_path=artifact_path,
            manifest_path=manifest_path,
            decision_log_path=decision_log_path,
            require_approved_for_live=_as_bool(
                raw.get("require_approved_for_live", True),
                "require_approved_for_live",
            ),
            threshold_override=threshold_override,
            strategy_config_hash=calculated_strategy_hash or configured_strategy_hash,
            execution_policy_hash=(
                calculated_execution_policy_hash or configured_execution_policy_hash
            ),
            execution_mode=execution_mode,
            instance_id=instance_id,
            exchange_name=exchange_name,
            symbol=symbol,
            exchange_environment=exchange_environment,
        )


@dataclass(frozen=True)
class ModelManifest:
    manifest_version: str
    model_type: str
    training_pipeline_version: str
    feature_names: tuple[str, ...]
    schema_hash: str
    model_sha256: str
    model_version: str
    threshold: float
    approved_for_live: bool
    strategy_config_hash: str
    execution_policy_hash: str
    live_execution_equivalent: bool
    live_approval_blockers: tuple[str, ...]

    @classmethod
    def from_path(cls, path: Path) -> "ModelManifest":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"model manifest does not exist: {path}") from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read model manifest {path}: {error}") from error
        if not isinstance(raw, Mapping):
            raise ValueError("model manifest must contain a JSON object")

        missing = [
            name
            for name in (
                "manifest_version",
                "model_type",
                "training_pipeline_version",
                "feature_names",
                "schema_hash",
                "model_sha256",
                "model_version",
                "threshold",
                "approved_for_live",
                "strategy_config_hash",
                "execution_policy_hash",
                "live_execution_equivalent",
                "live_approval_blockers",
            )
            if name not in raw
        ]
        if missing:
            raise ValueError(f"model manifest is missing required fields: {', '.join(missing)}")
        identities = (
            ("manifest_version", MANIFEST_VERSION),
            ("model_type", MODEL_TYPE),
            ("training_pipeline_version", TRAINING_PIPELINE_VERSION),
        )
        for field_name, expected in identities:
            actual = raw[field_name]
            if not isinstance(actual, str) or actual != expected:
                raise ValueError(
                    f"manifest {field_name} must equal {expected!r}"
                )
        raw_names = raw["feature_names"]
        if not isinstance(raw_names, list) or not all(
            isinstance(name, str) and name for name in raw_names
        ):
            raise ValueError("manifest feature_names must be a list of non-empty strings")
        approved = raw["approved_for_live"]
        if not isinstance(approved, bool):
            raise ValueError("manifest approved_for_live must be a boolean")
        live_equivalent = raw["live_execution_equivalent"]
        if not isinstance(live_equivalent, bool):
            raise ValueError("manifest live_execution_equivalent must be a boolean")
        raw_blockers = raw["live_approval_blockers"]
        if not isinstance(raw_blockers, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_blockers
        ):
            raise ValueError(
                "manifest live_approval_blockers must be a list of non-empty strings"
            )
        if approved and (not live_equivalent or raw_blockers):
            raise ValueError(
                "manifest cannot be approved_for_live while live execution blockers remain"
            )
        threshold = _required_probability(raw["threshold"], "manifest threshold")
        model_version = raw["model_version"]
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("manifest model_version must be a non-empty string")

        return cls(
            manifest_version=MANIFEST_VERSION,
            model_type=MODEL_TYPE,
            training_pipeline_version=TRAINING_PIPELINE_VERSION,
            feature_names=tuple(raw_names),
            schema_hash=_required_digest(raw["schema_hash"], "schema_hash"),
            model_sha256=_required_digest(raw["model_sha256"], "model_sha256"),
            model_version=model_version.strip(),
            threshold=threshold,
            approved_for_live=approved,
            strategy_config_hash=_required_digest(
                raw["strategy_config_hash"], "strategy_config_hash"
            ),
            execution_policy_hash=_required_digest(
                raw["execution_policy_hash"], "execution_policy_hash"
            ),
            live_execution_equivalent=live_equivalent,
            live_approval_blockers=tuple(item.strip() for item in raw_blockers),
        )


@dataclass(frozen=True)
class ValidatedArtifact:
    path: Path
    manifest: ModelManifest
    threshold: float


def validate_artifact(config: MetaModelConfig) -> ValidatedArtifact:
    if config.artifact_path is None:
        raise ValueError("trade_model.artifact_path is required when the gate is enabled")
    if config.manifest_path is None:
        raise ValueError("trade_model.manifest_path is required when the gate is enabled")
    artifact_path = Path(config.artifact_path)
    manifest = ModelManifest.from_path(Path(config.manifest_path))
    if not manifest.model_version:
        raise ValueError("manifest model_version must not be empty")
    if manifest.feature_names != FEATURE_NAMES:
        raise ValueError("manifest feature_names do not exactly match the runtime feature order")
    if not hmac.compare_digest(manifest.schema_hash, FEATURE_SCHEMA_HASH):
        raise ValueError("manifest schema_hash does not match the runtime feature schema")
    if not config.strategy_config_hash:
        raise ValueError("current strategy_config_hash is unavailable")
    if not hmac.compare_digest(
        manifest.strategy_config_hash,
        config.strategy_config_hash.lower(),
    ):
        raise ValueError("manifest strategy_config_hash does not match current strategy config")
    if not config.execution_policy_hash:
        raise ValueError("current execution_policy_hash is unavailable")
    if not hmac.compare_digest(
        manifest.execution_policy_hash,
        config.execution_policy_hash.lower(),
    ):
        raise ValueError(
            "manifest execution_policy_hash does not match current execution policy"
        )

    actual_model_hash = sha256_file(artifact_path)
    if not hmac.compare_digest(manifest.model_sha256, actual_model_hash):
        raise ValueError("manifest model_sha256 does not match the model artifact")
    threshold = (
        config.threshold_override
        if config.threshold_override is not None
        else manifest.threshold
    )
    return ValidatedArtifact(artifact_path, manifest, threshold)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"cannot read model artifact {path}: {error}") from error
    return digest.hexdigest()


def _optional_path(value: Any, base_dir: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    if str(value).strip() == ":memory:":
        return Path(":memory:")
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if value == 0 or value == 1:
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


def _optional_probability(value: Any, name: str) -> float | None:
    if value is None or value == "":
        return None
    return _required_probability(value, name)


def _required_probability(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return probability


def _required_digest(value: Any, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"manifest {name} must be a SHA-256 hex digest")
    return digest
