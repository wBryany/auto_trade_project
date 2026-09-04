from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
DATASET_VERSION = "binance-usdm-klines-v1"
INTERVAL_MILLISECONDS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}
_SYMBOL = re.compile(r"^[A-Z0-9]{5,24}$")


class BinancePublicClient:
    """Small credential-free REST client with retry and local pacing."""

    def __init__(
        self,
        *,
        requests_per_second: float = 5.0,
        max_retries: int = 6,
        timeout_seconds: float = 30.0,
        opener: Any = urlopen,
        sleep: Any = time.sleep,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.minimum_interval = 1.0 / requests_per_second
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.opener = opener
        self.sleep = sleep
        self._last_request_at = 0.0

    def get(self, path: str, params: dict[str, Any]) -> Any:
        if not path.startswith("/fapi/v1/"):
            raise ValueError("only Binance USD-M public v1 endpoints are allowed")
        url = f"{BINANCE_FUTURES_BASE_URL}{path}?{urlencode(params)}"
        for attempt in range(self.max_retries + 1):
            self._pace()
            request = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "btc-futures-bot-dataset/1.0",
                },
                method="GET",
            )
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if isinstance(payload, dict) and int(payload.get("code", 0)) < 0:
                    raise RuntimeError(
                        f"Binance API error {payload.get('code')}: {payload.get('msg', '')}"
                    )
                return payload
            except HTTPError as error:
                retryable = error.code in {418, 429} or 500 <= error.code < 600
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(f"Binance HTTP {error.code} for {path}") from error
                retry_after = error.headers.get("Retry-After") if error.headers else None
                self.sleep(self._retry_delay(attempt, retry_after))
            except (URLError, TimeoutError, ConnectionError) as error:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Binance request failed for {path}: {error}") from error
                self.sleep(self._retry_delay(attempt, None))
            except (UnicodeError, json.JSONDecodeError) as error:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"invalid Binance JSON for {path}") from error
                self.sleep(self._retry_delay(attempt, None))
        raise AssertionError("unreachable retry loop")

    def _pace(self) -> None:
        remaining = self.minimum_interval - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            self.sleep(remaining)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(retry_after)
                    return min(
                        60.0,
                        max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds()),
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(30.0, (2**attempt) + random.random() * 0.25)


def download_klines(
    client: BinancePublicClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    """Page closed USD-M futures candles over the half-open [start, end) range."""

    selected_symbol = _normalise_symbol(symbol)
    if interval not in INTERVAL_MILLISECONDS:
        raise ValueError(f"unsupported interval: {interval}")
    if end_ms <= start_ms:
        raise ValueError("end must be later than start")
    interval_ms = INTERVAL_MILLISECONDS[interval]
    cursor = (int(start_ms) // interval_ms) * interval_ms
    rows: dict[int, dict[str, Any]] = {}
    while cursor < end_ms:
        payload = client.get(
            "/fapi/v1/klines",
            {
                "symbol": selected_symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 1500,
            },
        )
        if not isinstance(payload, list):
            raise RuntimeError("Binance kline response was not a list")
        page = [_normalise_kline(row) for row in payload]
        page = [
            row
            for row in page
            if start_ms <= row["open_time"] < end_ms and row["close_time"] < end_ms
        ]
        for row in page:
            rows[row["open_time"]] = row
        if not payload:
            break
        last_open = int(payload[-1][0])
        next_cursor = last_open + interval_ms
        if next_cursor <= cursor:
            raise RuntimeError("Binance kline pagination did not advance")
        cursor = next_cursor
        if len(payload) < 1500:
            break
    return [rows[timestamp] for timestamp in sorted(rows)]


def download_funding_rates(
    client: BinancePublicClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    """Page public historical funding settlements over [start, end)."""

    selected_symbol = _normalise_symbol(symbol)
    cursor = int(start_ms)
    rows: dict[int, dict[str, Any]] = {}
    while cursor < end_ms:
        payload = client.get(
            "/fapi/v1/fundingRate",
            {
                "symbol": selected_symbol,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 1000,
            },
        )
        if not isinstance(payload, list):
            raise RuntimeError("Binance funding response was not a list")
        for raw in payload:
            timestamp = int(raw["fundingTime"])
            if start_ms <= timestamp < end_ms:
                rows[timestamp] = {
                    "funding_time": timestamp,
                    "funding_rate": str(raw["fundingRate"]),
                    "mark_price": str(raw.get("markPrice") or ""),
                }
        if not payload:
            break
        next_cursor = int(payload[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise RuntimeError("Binance funding pagination did not advance")
        cursor = next_cursor
        if len(payload) < 1000:
            break
    return [rows[timestamp] for timestamp in sorted(rows)]


def write_jsonl_atomic(path: str | Path, rows: list[dict[str, Any]]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                json.dump(row, handle, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return _sha256_file(destination)


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _normalise_kline(row: Any) -> dict[str, Any]:
    if not isinstance(row, list) or len(row) < 11:
        raise RuntimeError("invalid Binance kline row")
    return {
        "open_time": int(row[0]),
        "open": str(row[1]),
        "high": str(row[2]),
        "low": str(row[3]),
        "close": str(row[4]),
        "volume": str(row[5]),
        "close_time": int(row[6]),
        "quote_volume": str(row[7]),
        "trade_count": int(row[8]),
        "taker_buy_base_volume": str(row[9]),
        "taker_buy_quote_volume": str(row[10]),
    }


def _normalise_symbol(symbol: str) -> str:
    selected = str(symbol).replace("/", "").replace("-", "").strip().upper()
    if not _SYMBOL.fullmatch(selected):
        raise ValueError("symbol must be a Binance USD-M symbol such as BTCUSDT")
    return selected


def _parse_timestamp(value: str) -> int:
    text = str(value).strip()
    if text.isdigit():
        timestamp = int(text)
        return timestamp * 1000 if timestamp < 10_000_000_000 else timestamp
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public Binance USD-M closed klines for meta-model training."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True, help="UTC ISO-8601 or Unix seconds/ms (inclusive)")
    parser.add_argument("--end", help="UTC ISO-8601 or Unix seconds/ms (exclusive; default now)")
    parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "1h"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/binance_meta"))
    parser.add_argument("--requests-per-second", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument(
        "--include-funding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also download public funding settlements (default: enabled)",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    symbol = _normalise_symbol(args.symbol)
    start_ms = _parse_timestamp(args.start)
    requested_end_ms = _parse_timestamp(args.end) if args.end else int(time.time() * 1000)
    # Never persist a forming candle even if a future end was supplied.
    end_ms = min(requested_end_ms, int(time.time() * 1000))
    if end_ms <= start_ms:
        raise SystemExit("--end must be later than --start")
    intervals = tuple(dict.fromkeys(args.intervals))
    unsupported = [value for value in intervals if value not in INTERVAL_MILLISECONDS]
    if unsupported:
        raise SystemExit(f"unsupported intervals: {', '.join(unsupported)}")
    client = BinancePublicClient(
        requests_per_second=args.requests_per_second,
        max_retries=args.max_retries,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_files: dict[str, Any] = {}
    for interval in intervals:
        rows = download_klines(client, symbol, interval, start_ms, end_ms)
        destination = args.output_dir / f"{symbol}-{interval}.jsonl"
        digest = write_jsonl_atomic(destination, rows)
        manifest_files[interval] = {
            "path": destination.name,
            "sha256": digest,
            "rows": len(rows),
            "open_time_start": rows[0]["open_time"] if rows else None,
            "open_time_end": rows[-1]["open_time"] if rows else None,
        }
        print(f"{interval}: {len(rows)} closed candles -> {destination}")
    if args.include_funding:
        funding = download_funding_rates(client, symbol, start_ms, end_ms)
        destination = args.output_dir / f"{symbol}-funding.jsonl"
        digest = write_jsonl_atomic(destination, funding)
        manifest_files["funding"] = {
            "path": destination.name,
            "sha256": digest,
            "rows": len(funding),
            "funding_time_start": funding[0]["funding_time"] if funding else None,
            "funding_time_end": funding[-1]["funding_time"] if funding else None,
        }
        print(f"funding: {len(funding)} settlements -> {destination}")
    manifest = {
        "dataset_version": DATASET_VERSION,
        "source": "Binance USD-M public REST",
        "base_url": BINANCE_FUTURES_BASE_URL,
        "symbol": symbol,
        "requested_start_ms": start_ms,
        "requested_end_ms": requested_end_ms,
        "effective_end_ms": end_ms,
        "downloaded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": manifest_files,
    }
    write_json_atomic(args.output_dir / "dataset_manifest.json", manifest)
    print(f"manifest -> {args.output_dir / 'dataset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
