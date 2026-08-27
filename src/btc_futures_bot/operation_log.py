from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


BEIJING_TZ = ZoneInfo("Asia/Shanghai")


SENSITIVE_KEY_PARTS = (
    "secret",
    "passphrase",
    "password",
    "token",
    "private_key",
    "api_key",
    "apikey",
)


def sanitize(value: Any, key: str = "") -> Any:
    """Remove credentials from structured operation-log data."""
    lowered = key.lower()
    if any(part in lowered for part in SENSITIVE_KEY_PARTS):
        return "[已保存，不记录]"
    if isinstance(value, dict):
        return {str(name): sanitize(item, str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item, key) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_json_or_csv(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if isinstance(default, list):
            return [item.strip() for item in value.split(",") if item.strip()]
        return default


class OperationLogger:
    """Append-only local JSONL audit log for dashboard and automation actions."""

    def __init__(self, log_path: str | Path = "logs/operation_log.jsonl") -> None:
        self.path = Path(log_path)
        if self.path.suffix.lower() != ".jsonl":
            self.path = self.path / "operation_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        event_type: str,
        action: str,
        *,
        status: str = "success",
        summary: str = "",
        details: Any = None,
        changed_files: list[str] | None = None,
        result: Any = None,
        source: str = "dashboard",
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        entry = {
            "timestamp": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "local_time": now.astimezone(BEIJING_TZ).isoformat(timespec="seconds"),
            "event_type": str(event_type),
            "action": str(action),
            "status": str(status),
            "summary": str(summary),
            "details": sanitize(details or {}),
            "changed_files": sanitize(changed_files or []),
            "result": sanitize(result or {}),
            "source": str(source),
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
            file.flush()
        return entry

    def query(
        self,
        *,
        start_date: str = "",
        end_date: str = "",
        event_type: str = "",
        keyword: str = "",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        needle = keyword.strip().lower()
        safe_limit = max(1, min(int(limit), 5000))
        if not self.path.exists():
            return rows
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            local_date = str(row.get("local_time", ""))[:10]
            if start_date and local_date < start_date:
                continue
            if end_date and local_date > end_date:
                continue
            if event_type and str(row.get("event_type", "")) != event_type:
                continue
            if needle:
                haystack = json.dumps(row, ensure_ascii=False).lower()
                if needle not in haystack:
                    continue
            rows.append(row)
        rows.sort(key=lambda row: str(row.get("timestamp", "")), reverse=True)
        return rows[:safe_limit]


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Append or inspect the local BTC bot operation log")
    subparsers = parser.add_subparsers(dest="command", required=True)
    append = subparsers.add_parser("append")
    append.add_argument("--log-path", default="logs/operation_log.jsonl")
    append.add_argument("--event-type", required=True)
    append.add_argument("--action", required=True)
    append.add_argument("--status", default="success")
    append.add_argument("--summary", default="")
    append.add_argument("--details", default="{}")
    append.add_argument("--changed-files", default="[]")
    append.add_argument("--result", default="{}")
    append.add_argument("--source", default="automation")
    args = parser.parse_args()
    if args.command == "append":
        logger = OperationLogger(args.log_path)
        entry = logger.record(
            args.event_type,
            args.action,
            status=args.status,
            summary=args.summary,
            details=_parse_json_or_csv(args.details, {}),
            changed_files=_parse_json_or_csv(args.changed_files, []),
            result=_parse_json_or_csv(args.result, {}),
            source=args.source,
        )
        print(json.dumps({"timestamp": entry["timestamp"], "status": entry["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
