from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from ..models import Candle, Signal
from .artifact import MetaModelConfig, ValidatedArtifact, validate_artifact
from .decision_log import DecisionLog
from .features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_HASH,
    extract_features,
    feature_values,
)


class Predictor(Protocol):
    def predict(self, features: Sequence[float]) -> Any: ...


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    signal: Signal
    score: float | None
    threshold: float | None
    decision: str
    reason: str
    model_name: str = "lightgbm_meta"
    model_version: str = ""

    @property
    def decision_reason(self) -> str:
        return self.reason


class EntryGate:
    """Meta-model entry gate with fail-closed live enforcement.

    The object never changes exits, position sizing, stops or take profit.  Its
    only output is whether one primary-strategy entry candidate is accepted.
    """

    _MAX_CACHE_SIZE = 4096

    def __init__(
        self,
        config: MetaModelConfig,
        *,
        predictor: Predictor | Callable[[Sequence[float]], Any] | None = None,
        predictor_factory: Callable[[Path], Predictor] | None = None,
    ) -> None:
        if config.mode not in {"off", "shadow", "enforce"}:
            raise ValueError("entry gate mode must be off, shadow, or enforce")
        self.config = config
        self._lock = threading.RLock()
        self._cache: OrderedDict[tuple[str, int], GateDecision] = OrderedDict()
        self._closed = False
        self._artifact: ValidatedArtifact | None = None
        self._predictor: Predictor | Callable[[Sequence[float]], Any] | None = None
        self._model_error = ""
        self._log_error = ""
        self._decision_log: DecisionLog | None = None
        self._record_count = 0

        if config.decision_log_path is not None:
            try:
                self._decision_log = DecisionLog(Path(config.decision_log_path))
            except Exception as error:
                self._log_error = _format_error("decision log initialization failed", error)
        elif config.mode != "off":
            self._log_error = "decision_log_path is required when the entry gate is enabled"

        if config.mode == "off":
            return
        try:
            self._artifact = validate_artifact(config)
            if predictor is not None and predictor_factory is not None:
                raise ValueError("provide predictor or predictor_factory, not both")
            if predictor is not None:
                self._predictor = predictor
            elif predictor_factory is not None:
                self._predictor = predictor_factory(self._artifact.path)
            else:
                self._predictor = _LightGBMPredictor(self._artifact.path)
        except Exception as error:
            self._model_error = _format_error("model initialization failed", error)

    @property
    def threshold(self) -> float | None:
        return self._artifact.threshold if self._artifact is not None else None

    @property
    def model_version(self) -> str:
        if self._artifact is None:
            return ""
        return self._artifact.manifest.model_version

    def evaluate(
        self,
        signal: Signal,
        candles_by_timeframe: Mapping[str, Sequence[Candle]],
    ) -> GateDecision:
        side = str(signal.side).lower()
        if side == "flat":
            return self._make_decision(
                signal,
                accepted=True,
                score=None,
                threshold=self.threshold,
                decision="not_applicable",
                reason="flat signals are not entry candidates",
            )

        try:
            timestamp = int(signal.timestamp)
        except (TypeError, ValueError):
            timestamp = 0
        cache_key = (side, timestamp)
        with self._lock:
            if self._closed and self.config.mode != "off":
                decision = self._error_decision(signal, "entry gate is closed")
                return self._cache_decision(cache_key, decision)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached

            if self.config.mode == "off":
                decision = self._make_decision(
                    signal,
                    accepted=True,
                    score=None,
                    threshold=None,
                    decision="off_accept",
                    reason="entry gate is disabled",
                )
                return self._finish(cache_key, decision, features={})

            if side not in {"long", "short"}:
                decision = self._error_decision(
                    signal, f"unsupported entry signal side: {signal.side}"
                )
                return self._finish(cache_key, decision, features={})
            if self._model_error:
                decision = self._error_decision(signal, self._model_error)
                return self._finish(cache_key, decision, features={})
            if self._log_error:
                decision = self._error_decision(signal, self._log_error)
                return self._finish(cache_key, decision, features={})

            features: dict[str, float] = {}
            try:
                features = extract_features(signal, candles_by_timeframe)
                score = self._predict(feature_values(features))
                threshold = self.threshold
                if threshold is None:
                    raise RuntimeError("validated model threshold is unavailable")
                would_accept = score >= threshold
                if self.config.mode == "shadow":
                    decision = self._make_decision(
                        signal,
                        accepted=True,
                        score=score,
                        threshold=threshold,
                        decision="shadow_accept" if would_accept else "shadow_reject",
                        reason=(
                            "shadow score meets threshold; primary signal remains allowed"
                            if would_accept
                            else "shadow score is below threshold; primary signal remains allowed"
                        ),
                    )
                else:
                    decision = self._make_decision(
                        signal,
                        accepted=would_accept,
                        score=score,
                        threshold=threshold,
                        decision="enforce_accept" if would_accept else "enforce_reject",
                        reason=(
                            "meta-model score meets the frozen threshold"
                            if would_accept
                            else "meta-model score is below the frozen threshold"
                        ),
                    )
            except Exception as error:
                decision = self._error_decision(
                    signal,
                    _format_error("entry gate evaluation failed", error),
                )
            return self._finish(cache_key, decision, features=features)

    def status(self) -> dict[str, Any]:
        with self._lock:
            manifest = self._artifact.manifest if self._artifact is not None else None
            errors = [error for error in (self._model_error, self._log_error) if error]
            return {
                "type": self.config.type,
                "mode": self.config.mode,
                "ready": not self._closed
                and (self.config.mode == "off" or not errors),
                "closed": self._closed,
                "model_version": manifest.model_version if manifest else "",
                "threshold": self.threshold,
                "approved_for_live": manifest.approved_for_live if manifest else False,
                "live_execution_equivalent": (
                    manifest.live_execution_equivalent if manifest else False
                ),
                "live_approval_blockers": (
                    list(manifest.live_approval_blockers) if manifest else []
                ),
                "schema_hash": manifest.schema_hash if manifest else FEATURE_SCHEMA_HASH,
                "execution_policy_hash": self.config.execution_policy_hash,
                "model_sha256": manifest.model_sha256 if manifest else "",
                "instance_id": self.config.instance_id,
                "exchange_name": self.config.exchange_name,
                "symbol": self.config.symbol,
                "exchange_environment": self.config.exchange_environment,
                "artifact_path": (
                    str(self.config.artifact_path) if self.config.artifact_path else ""
                ),
                "manifest_path": (
                    str(self.config.manifest_path) if self.config.manifest_path else ""
                ),
                "decision_log_path": (
                    str(self.config.decision_log_path)
                    if self.config.decision_log_path
                    else ""
                ),
                "error": "; ".join(errors),
                "cache_size": len(self._cache),
                "recorded_candidates": self._record_count,
            }

    def ensure_live_ready(self) -> None:
        """Raise unless this gate is safe to attach to a live trading engine."""

        with self._lock:
            # Shadow is observational only.  It must never keep the proven
            # primary strategy from starting or entering; its unavailable
            # state remains visible through status() and decision metadata.
            if self.config.mode in {"off", "shadow"}:
                return
            if self._closed:
                raise RuntimeError("entry gate is closed")
            if self.config.threshold_override is not None:
                raise RuntimeError("threshold_override is not permitted for live trading")
            errors = [error for error in (self._model_error, self._log_error) if error]
            if errors:
                raise RuntimeError("entry gate is not live-ready: " + "; ".join(errors))
            if self._artifact is None or self._predictor is None:
                raise RuntimeError("entry gate is not live-ready: model is unavailable")
            if (
                self.config.require_approved_for_live
                and not self._artifact.manifest.approved_for_live
            ):
                raise RuntimeError("entry gate model is not approved_for_live")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._decision_log is not None:
                self._decision_log.close()
                self._decision_log = None

    def _predict(self, values: Sequence[float]) -> float:
        predictor = self._predictor
        if predictor is None:
            raise RuntimeError("model predictor is unavailable")
        if hasattr(predictor, "predict"):
            raw_score = predictor.predict(values)  # type: ignore[union-attr]
        elif callable(predictor):
            raw_score = predictor(values)
        else:
            raise TypeError("predictor must be callable or expose predict()")
        score = _scalar_probability(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("model prediction must be a finite probability from 0 to 1")
        return score

    def _finish(
        self,
        cache_key: tuple[str, int],
        decision: GateDecision,
        *,
        features: Mapping[str, float],
    ) -> GateDecision:
        try:
            self._persist(decision, features)
        except Exception as error:
            self._log_error = _format_error("decision logging failed", error)
            if self.config.mode in {"shadow", "enforce"}:
                decision = self._error_decision(
                    decision.signal,
                    self._log_error,
                    score=decision.score,
                    threshold=decision.threshold,
                )
        return self._cache_decision(cache_key, decision)

    def _persist(
        self,
        decision: GateDecision,
        features: Mapping[str, float],
    ) -> None:
        if self._decision_log is None:
            if self.config.mode != "off":
                raise RuntimeError(self._log_error or "decision log is unavailable")
            return
        inserted = self._decision_log.record(
            timestamp=int(decision.signal.timestamp),
            side=str(decision.signal.side),
            signal_score=float(decision.signal.score),
            reasons=tuple(str(reason) for reason in decision.signal.reasons),
            gate_mode=self.config.mode,
            accepted=decision.accepted,
            meta_score=decision.score,
            threshold=decision.threshold,
            decision=decision.decision,
            reason=decision.reason,
            model_name=decision.model_name,
            model_version=decision.model_version,
            schema_hash=FEATURE_SCHEMA_HASH,
            execution_policy_hash=self.config.execution_policy_hash,
            model_sha256=(
                self._artifact.manifest.model_sha256 if self._artifact is not None else ""
            ),
            instance_id=self.config.instance_id,
            exchange_name=self.config.exchange_name,
            symbol=self.config.symbol,
            exchange_environment=self.config.exchange_environment,
            features=features,
        )
        if inserted:
            self._record_count += 1

    def _error_decision(
        self,
        signal: Signal,
        reason: str,
        *,
        score: float | None = None,
        threshold: float | None = None,
    ) -> GateDecision:
        threshold = self.threshold if threshold is None else threshold
        if self.config.mode == "shadow":
            return self._make_decision(
                signal,
                accepted=True,
                score=score,
                threshold=threshold,
                decision="shadow_error_accept",
                reason=reason,
            )
        return self._make_decision(
            signal,
            accepted=False,
            score=score,
            threshold=threshold,
            decision="enforce_error_reject",
            reason=reason,
        )

    def _make_decision(
        self,
        signal: Signal,
        *,
        accepted: bool,
        score: float | None,
        threshold: float | None,
        decision: str,
        reason: str,
    ) -> GateDecision:
        enhanced = _enhance_signal(
            signal,
            model_name=self.config.type,
            model_version=self.model_version,
            meta_score=float(score) if score is not None else 0.0,
            meta_threshold=float(threshold) if threshold is not None else 0.0,
            meta_decision=decision,
        )
        return GateDecision(
            accepted=accepted,
            signal=enhanced,
            score=score,
            threshold=threshold,
            decision=decision,
            reason=reason,
            model_name=self.config.type,
            model_version=self.model_version,
        )

    def _cache_decision(
        self,
        key: tuple[str, int],
        decision: GateDecision,
    ) -> GateDecision:
        self._cache[key] = decision
        self._cache.move_to_end(key)
        while len(self._cache) > self._MAX_CACHE_SIZE:
            self._cache.popitem(last=False)
        return decision


class _LightGBMPredictor:
    def __init__(self, model_path: Path) -> None:
        try:
            import lightgbm as lightgbm  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "LightGBM is required when trade_model mode is shadow or enforce"
            ) from error
        self._booster = lightgbm.Booster(model_file=str(model_path))
        model_names = tuple(self._booster.feature_name())
        if model_names and model_names != FEATURE_NAMES:
            raise ValueError("native LightGBM feature names do not match the manifest schema")

    def predict(self, features: Sequence[float]) -> Any:
        return self._booster.predict([list(features)])


def _scalar_probability(value: Any) -> float:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("model prediction must contain exactly one probability")
        value = value[0]
        if hasattr(value, "tolist"):
            value = value.tolist()
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("model prediction is not numeric") from error


def _enhance_signal(signal: Signal, **updates: Any) -> Signal:
    # During rolling deployments an old process may still use the four-field
    # Signal.  Apply all metadata present in its dataclass schema; the current
    # schema receives every structured meta-model field.
    if not is_dataclass(signal):
        return signal
    available = {field.name for field in fields(signal)}
    selected = {name: value for name, value in updates.items() if name in available}
    return replace(signal, **selected) if selected else signal


def _format_error(prefix: str, error: Exception) -> str:
    detail = str(error).strip() or error.__class__.__name__
    return f"{prefix}: {detail}"
