from __future__ import annotations

import json
import re
import threading
import time
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from typing import Any, Mapping


class ApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_at: float = 0.0,
        api_code: int | str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_at = float(retry_at)
        self.api_code = api_code

    @property
    def retry_after_seconds(self) -> float:
        return max(0.0, self.retry_at - time.time())

    @property
    def rate_limited(self) -> bool:
        return (
            self.status_code in {418, 429}
            or str(self.api_code or "") == "-1003"
            or self.retry_at > time.time()
        )


_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_UNTIL: dict[str, float] = {}
_RATE_LIMIT_MARKERS = (
    "rate limit active",
    "http 418",
    "http 429",
    '"code":-1003',
    '"code": -1003',
    "'code':-1003",
    "'code': -1003",
    "ip banned",
    "too many requests",
    "way too many requests",
)
_SENSITIVE_QUERY_VALUE = re.compile(
    r"([?&](?:signature|api[_-]?key|api[_-]?secret|token)=)[^&\s]+",
    re.IGNORECASE,
)


def redact_url_credentials(url: str) -> str:
    """Remove signed/auth query values before a URL reaches logs or UI."""

    return _SENSITIVE_QUERY_VALUE.sub(r"\1[REDACTED]", str(url))


def _rate_limit_key(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def rate_limit_remaining(url: str) -> float:
    """Return seconds before another request may be sent to this host."""

    key = _rate_limit_key(url)
    with _RATE_LIMIT_LOCK:
        retry_at = _RATE_LIMIT_UNTIL.get(key, 0.0)
        if retry_at <= time.time():
            _RATE_LIMIT_UNTIL.pop(key, None)
            return 0.0
    return max(0.0, retry_at - time.time())


def _set_rate_limit(url: str, retry_at: float) -> float:
    key = _rate_limit_key(url)
    with _RATE_LIMIT_LOCK:
        retry_at = max(float(retry_at), _RATE_LIMIT_UNTIL.get(key, 0.0))
        _RATE_LIMIT_UNTIL[key] = retry_at
    return retry_at


def clear_rate_limits() -> None:
    """Clear process-local limits; intended for tests and explicit recovery."""

    with _RATE_LIMIT_LOCK:
        _RATE_LIMIT_UNTIL.clear()


def is_rate_limit_error(error: BaseException | str | object) -> bool:
    """Recognize Binance IP throttling even after an exception was wrapped/stringified."""

    current: object | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ApiError) and current.rate_limited:
            return True
        message = str(current).lower().replace("\r", " ").replace("\n", " ")
        if any(marker in message for marker in _RATE_LIMIT_MARKERS):
            return True
        if isinstance(current, BaseException):
            current = current.__cause__ or current.__context__
        else:
            current = None
    return False


def _response_api_code(detail: str) -> int | str | None:
    try:
        payload = json.loads(detail)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    code = payload.get("code")
    if isinstance(code, (int, str)) and not isinstance(code, bool):
        return code
    return None


def _http_retry_at(error: HTTPError, detail: str) -> float:
    now = time.time()
    banned_match = re.search(r"banned until\s+(\d{10,16})", detail, re.IGNORECASE)
    if banned_match:
        timestamp = int(banned_match.group(1))
        return timestamp / 1000 if timestamp > 10_000_000_000 else float(timestamp)
    response_headers = getattr(error, "headers", None) or {}
    retry_after = str(response_headers.get("Retry-After") or "").strip()
    if retry_after:
        try:
            return now + max(0.0, float(retry_after))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(retry_after)
                return parsed.timestamp()
            except (TypeError, ValueError, OverflowError):
                pass
    return now + (300.0 if error.code == 418 else 60.0)


def request_json(
    method: str,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    body: Mapping[str, Any] | str | None = None,
    timeout: float = 10.0,
    max_attempts: int | None = None,
) -> Any:
    query = urlencode([(key, value) for key, value in (params or {}).items() if value is not None])
    if query:
        url = f"{url}?{query}"
    data: bytes | None = None
    final_headers = {
        "Accept": "application/json",
        "User-Agent": "btc-futures-bot/0.1 (+local-dashboard)",
        **(headers or {}),
    }
    if body is not None:
        if isinstance(body, str):
            data = body.encode("utf-8")
        else:
            data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        final_headers.setdefault("Content-Type", "application/json")
    request_method = method.upper()
    attempts = (
        max(1, int(max_attempts))
        if max_attempts is not None and request_method == "GET"
        else (3 if request_method == "GET" else 1)
    )
    safe_url = redact_url_credentials(url)
    payload = ""
    for attempt in range(attempts):
        blocked_for = rate_limit_remaining(url)
        if blocked_for > 0:
            retry_at = time.time() + blocked_for
            raise ApiError(
                f"rate limit active for {_rate_limit_key(url)}; retry in {blocked_for:.0f}s",
                status_code=418,
                retry_at=retry_at,
            )
        request = Request(url, data=data, headers=final_headers, method=request_method)
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            break
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            api_code = _response_api_code(detail)
            if error.code in {418, 429} or str(api_code or "") == "-1003":
                retry_at = _set_rate_limit(url, _http_retry_at(error, detail))
                raise ApiError(
                    f"HTTP {error.code} {method} {safe_url}: {detail[:500]}",
                    status_code=error.code,
                    retry_at=retry_at,
                    api_code=api_code,
                ) from error
            retryable = error.code in {408, 425, 500, 502, 503, 504}
            if retryable and attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
                continue
            raise ApiError(
                f"HTTP {error.code} {method} {safe_url}: {detail[:500]}",
                status_code=error.code,
                api_code=api_code,
            ) from error
        except URLError as error:
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
                continue
            raise ApiError(f"network error {method} {safe_url}: {error.reason}") from error
        except (TimeoutError, ConnectionError, OSError) as error:
            # urllib can surface transient socket/SSL failures directly instead
            # of wrapping them in URLError. Retrying GET is safe and prevents a
            # single market-data disconnect from aborting an engine cycle.
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
                continue
            raise ApiError(f"network error {method} {safe_url}: {error}") from error
    try:
        return json.loads(payload) if payload else {}
    except json.JSONDecodeError as error:
        raise ApiError(f"invalid JSON from {method} {safe_url}: {payload[:500]}") from error


def format_number(value: float, decimals: int = 12) -> str:
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")
