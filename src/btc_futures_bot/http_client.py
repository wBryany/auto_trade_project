from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing import Any, Mapping


class ApiError(RuntimeError):
    pass


def request_json(
    method: str,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    body: Mapping[str, Any] | str | None = None,
    timeout: float = 10.0,
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
    max_attempts = 3 if request_method == "GET" else 1
    payload = ""
    for attempt in range(max_attempts):
        request = Request(url, data=data, headers=final_headers, method=request_method)
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            break
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            retryable = error.code in {408, 425, 429, 500, 502, 503, 504}
            if retryable and attempt + 1 < max_attempts:
                time.sleep(0.5 * (2**attempt))
                continue
            raise ApiError(f"HTTP {error.code} {method} {url}: {detail[:500]}") from error
        except URLError as error:
            if attempt + 1 < max_attempts:
                time.sleep(0.5 * (2**attempt))
                continue
            raise ApiError(f"network error {method} {url}: {error.reason}") from error
        except (TimeoutError, ConnectionError, OSError) as error:
            # urllib can surface transient socket/SSL failures directly instead
            # of wrapping them in URLError. Retrying GET is safe and prevents a
            # single market-data disconnect from aborting an engine cycle.
            if attempt + 1 < max_attempts:
                time.sleep(0.5 * (2**attempt))
                continue
            raise ApiError(f"network error {method} {url}: {error}") from error
    try:
        return json.loads(payload) if payload else {}
    except json.JSONDecodeError as error:
        raise ApiError(f"invalid JSON from {method} {url}: {payload[:500]}") from error


def format_number(value: float, decimals: int = 12) -> str:
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")
