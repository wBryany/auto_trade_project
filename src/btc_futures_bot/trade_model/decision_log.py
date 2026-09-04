from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path


class DecisionLog:
    """Append-only audit log with one row per primary-model candidate."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.execute("PRAGMA foreign_keys = ON")
        if str(path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS entry_candidates (
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
                    execution_policy_hash TEXT NOT NULL DEFAULT '',
                    model_sha256 TEXT NOT NULL DEFAULT '',
                    instance_id TEXT NOT NULL DEFAULT '',
                    exchange_name TEXT NOT NULL DEFAULT '',
                    symbol TEXT NOT NULL DEFAULT '',
                    exchange_environment TEXT NOT NULL DEFAULT '',
                    features_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    UNIQUE(signal_timestamp, side)
                )
                """
            )
            # Existing Model2 databases predate the immutable execution/model
            # provenance fields. SQLite's additive migration preserves every
            # historical candidate and gives those legacy rows an explicit
            # empty/unknown value.
            existing_columns = {
                str(row[1])
                for row in self._connection.execute("PRAGMA table_info(entry_candidates)")
            }
            for column in (
                "execution_policy_hash",
                "model_sha256",
                "instance_id",
                "exchange_name",
                "symbol",
                "exchange_environment",
            ):
                if column not in existing_columns:
                    self._connection.execute(
                        f"ALTER TABLE entry_candidates ADD COLUMN {column} "
                        "TEXT NOT NULL DEFAULT ''"
                    )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS entry_candidate_features (
                    candidate_id TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL NOT NULL,
                    PRIMARY KEY(candidate_id, feature_name),
                    FOREIGN KEY(candidate_id) REFERENCES entry_candidates(candidate_id)
                        ON DELETE CASCADE
                )
                """
            )

    def record(
        self,
        *,
        timestamp: int,
        side: str,
        signal_score: float,
        reasons: Sequence[str],
        gate_mode: str,
        accepted: bool,
        meta_score: float | None,
        threshold: float | None,
        decision: str,
        reason: str,
        model_name: str,
        model_version: str,
        schema_hash: str,
        execution_policy_hash: str,
        model_sha256: str,
        instance_id: str,
        exchange_name: str,
        symbol: str,
        exchange_environment: str,
        features: Mapping[str, float],
    ) -> bool:
        """Persist a decision, returning false when the candidate already exists."""

        candidate_id = f"{int(timestamp)}:{str(side).lower()}"
        features_payload = {
            str(name): float(value) for name, value in features.items()
        }
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO entry_candidates (
                    candidate_id, signal_timestamp, side, signal_score,
                    reasons_json, gate_mode, accepted, meta_score, threshold,
                    decision, reason, model_name, model_version, schema_hash,
                    execution_policy_hash, model_sha256, instance_id,
                    exchange_name, symbol, exchange_environment,
                    features_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    int(timestamp),
                    str(side).lower(),
                    float(signal_score),
                    json.dumps(list(reasons), ensure_ascii=False, separators=(",", ":")),
                    gate_mode,
                    int(bool(accepted)),
                    meta_score,
                    threshold,
                    decision,
                    reason,
                    model_name,
                    model_version,
                    schema_hash,
                    execution_policy_hash,
                    model_sha256,
                    instance_id,
                    exchange_name,
                    symbol,
                    exchange_environment,
                    json.dumps(
                        features_payload,
                        ensure_ascii=True,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    time.time_ns() // 1_000_000,
                ),
            )
            if cursor.rowcount != 1:
                return False
            self._connection.executemany(
                """
                INSERT INTO entry_candidate_features (
                    candidate_id, feature_name, feature_value
                ) VALUES (?, ?, ?)
                """,
                (
                    (candidate_id, name, value)
                    for name, value in features_payload.items()
                ),
            )
            return True

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM entry_candidates"
            ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._connection.close()
