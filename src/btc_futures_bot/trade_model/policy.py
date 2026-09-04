from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from ..costs import CostConfig
from ..risk import RiskConfig
from ..strategy import StrategyConfig
from .features import strategy_config_hash


MANIFEST_VERSION = "lightgbm-meta-manifest-v1"
MODEL_TYPE = "lightgbm_meta"
TRAINING_PIPELINE_VERSION = "meta-training-v1"
EXECUTION_POLICY_VERSION = "meta-execution-policy-v2"
LIVE_EXECUTION_BLOCKER = (
    "fixed triple-barrier labels do not reproduce the engine's dynamic live "
    "exit, protective-order, and position-management execution"
)

CANDIDATE_GENERATION_POLICY: dict[str, Any] = {
    "version": "primary-candidate-v1",
    "source": "MultiTimeframeStrategy.evaluate",
    "eligible_signal_sides": ["long", "short"],
    "strategy_is_frozen": True,
    "uniqueness": "first non-flat signal per primary signal timestamp",
    "feature_information_set": "only candles closed by decision_timestamp",
    "entry_rule": "next contiguous 1m candle open",
}

LABEL_RULES: dict[str, Any] = {
    "version": "fixed-triple-barrier-v1",
    "method": "fixed_tp_sl_horizon_triple_barrier",
    "same_candle_tp_and_sl": "stop_loss",
    "binary_target": "net_return_after_fees_slippage_and_funding_gt_zero",
    "funding": "actual settlements when available, configured fallback otherwise",
}

_BARRIER_FIELDS = (
    "take_profit_pct",
    "stop_loss_pct",
    "horizon_bars",
    "fee_pct_per_side",
    "slippage_pct_per_side",
    "fallback_funding_rate_per_8h",
)

_PACKAGE_DIR = Path(__file__).resolve().parent
IMPLEMENTATION_SOURCE_PATHS: dict[str, Path] = {
    "src/btc_futures_bot/strategy.py": _PACKAGE_DIR.parent / "strategy.py",
    "src/btc_futures_bot/engine.py": _PACKAGE_DIR.parent / "engine.py",
    "src/btc_futures_bot/trade_model/features.py": _PACKAGE_DIR / "features.py",
    "src/btc_futures_bot/trade_model/training.py": _PACKAGE_DIR / "training.py",
    "src/btc_futures_bot/trade_model/policy.py": Path(__file__).resolve(),
}


def build_execution_policy(
    config: Mapping[str, Any],
    *,
    barrier_config: Mapping[str, Any] | object | None = None,
    closed_history_limit: int | None = None,
) -> dict[str, Any]:
    """Build the one canonical training/runtime execution-policy document.

    ``barrier_config`` and ``closed_history_limit`` are training-only inputs.
    Passing a CLI override changes the resulting digest. Runtime deliberately
    omits both and reconstructs the policy implied by its complete effective
    configuration, so any training/runtime divergence fails artifact loading.
    """

    if not isinstance(config, Mapping):
        raise TypeError("effective bot configuration must be a mapping")

    strategy_raw = _mapping(config.get("strategy"), "strategy")
    risk_raw = _mapping(config.get("risk"), "risk")
    strategy = StrategyConfig(**strategy_raw)
    risk = RiskConfig(**risk_raw)

    active_exchange = str(config.get("active_exchange") or "binance").strip().lower()
    exchanges = _mapping(config.get("exchanges"), "exchanges")
    exchange = _mapping(exchanges.get(active_exchange), f"exchanges.{active_exchange}")
    root_costs = _mapping(config.get("costs"), "costs")
    selected_costs = exchange.get("costs", root_costs)
    costs = CostConfig(**_mapping(selected_costs, f"exchanges.{active_exchange}.costs"))

    configured_candle_limit = _integer(config.get("candle_limit", 300), "candle_limit")
    if configured_candle_limit < 2:
        raise ValueError("candle_limit must be at least 2")
    selected_closed_history = (
        configured_candle_limit - 1
        if closed_history_limit is None
        else _integer(closed_history_limit, "closed_history_limit")
    )
    if selected_closed_history < 21:
        raise ValueError("closed_history_limit must be at least 21")

    default_barrier = {
        "take_profit_pct": float(risk.stop_loss_pct) * float(strategy.take_profit_r),
        "stop_loss_pct": float(risk.stop_loss_pct),
        "horizon_bars": max(1, int(strategy.hard_max_hold_seconds) // 60),
        "fee_pct_per_side": float(costs.fee_pct),
        "slippage_pct_per_side": float(costs.slippage_pct),
        "fallback_funding_rate_per_8h": float(costs.funding_rate_pct_per_8h),
    }
    barrier = _normalise_barrier(barrier_config or default_barrier)

    return {
        "version": EXECUTION_POLICY_VERSION,
        "implementation_source_sha256": implementation_source_hashes(),
        "primary_strategy_hash": strategy_config_hash(strategy),
        "risk_config": _normalise_dataclass(risk),
        "cost_config": _normalise_dataclass(costs),
        "history_window": {
            "configured_candle_limit": configured_candle_limit,
            "closed_history_limit": selected_closed_history,
            "forming_bar_excluded": True,
        },
        "candidate_generation": dict(CANDIDATE_GENERATION_POLICY),
        "labeling": {
            "barrier_config": barrier,
            "rules": dict(LABEL_RULES),
            "live_execution_equivalent": False,
        },
    }


def execution_policy_hash(policy: Mapping[str, Any]) -> str:
    """Hash a canonical execution-policy document."""

    if not isinstance(policy, Mapping):
        raise TypeError("execution policy must be a mapping")
    try:
        encoded = json.dumps(
            policy,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"execution policy is not canonical JSON: {error}") from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def execution_policy_for_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    policy = build_execution_policy(config)
    return policy, execution_policy_hash(policy)


def implementation_source_hashes() -> dict[str, str]:
    """Fingerprint execution/training sources with platform-neutral newlines."""

    return {
        logical_name: normalized_source_sha256(path)
        for logical_name, path in IMPLEMENTATION_SOURCE_PATHS.items()
    }


def normalized_source_sha256(path: str | Path) -> str:
    """Hash UTF-8 source after CRLF/CR normalization to LF."""

    selected = Path(path)
    try:
        source = selected.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot fingerprint implementation source {selected}: {error}") from error
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalise_barrier(value: Mapping[str, Any] | object) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        raw = asdict(value)
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise TypeError("barrier_config must be a mapping or dataclass")
    missing = [name for name in _BARRIER_FIELDS if name not in raw]
    extra = sorted(set(raw).difference(_BARRIER_FIELDS))
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unknown " + ", ".join(extra))
        raise ValueError("invalid barrier_config: " + "; ".join(details))
    return {
        "take_profit_pct": _number(raw["take_profit_pct"], "take_profit_pct"),
        "stop_loss_pct": _number(raw["stop_loss_pct"], "stop_loss_pct"),
        "horizon_bars": _integer(raw["horizon_bars"], "horizon_bars"),
        "fee_pct_per_side": _number(raw["fee_pct_per_side"], "fee_pct_per_side"),
        "slippage_pct_per_side": _number(
            raw["slippage_pct_per_side"], "slippage_pct_per_side"
        ),
        "fallback_funding_rate_per_8h": _number(
            raw["fallback_funding_rate_per_8h"],
            "fallback_funding_rate_per_8h",
        ),
    }


def _normalise_dataclass(value: object) -> dict[str, Any]:
    defaults = type(value)()
    result: dict[str, Any] = {}
    for field in fields(value):
        selected = getattr(value, field.name)
        expected = type(getattr(defaults, field.name))
        if expected is bool:
            if not isinstance(selected, bool):
                raise ValueError(f"{field.name} must be a boolean")
            result[field.name] = selected
        elif expected is int:
            result[field.name] = _integer(selected, field.name)
        elif expected is float:
            result[field.name] = _number(selected, field.name)
        elif expected is str:
            result[field.name] = str(selected)
        else:
            result[field.name] = selected
    return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        selected = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if isinstance(value, float) and value != selected:
        raise ValueError(f"{name} must be an integer")
    return selected
