"""Process-wide preventive budget for the remaining USD-M REST traffic.

WebSocket deltas need no REST requests. Bootstrap/configuration/history still
do. Reserve headroom for order protection, and never sleep on a signed request
or an engine lock. The HTTP client separately enforces actual exchange bans.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlsplit


LOG = logging.getLogger(__name__)
HOSTS = {"fapi.binance.com", "demo-fapi.binance.com"}
_LOCK = threading.Lock()


@dataclass
class _Budget:
    observed: int | None = None
    expires: float = 0.0
    server_window: int = -1
    server_time: float | None = None
    server_monotonic: float = 0.0
    has_server_date: bool = False
    limit: int = 2400
    attempts: deque[float] = field(default_factory=deque)
    reservations: deque[tuple[float, int]] = field(default_factory=deque)
    total_attempts: int = 0
    deferred: int = 0


_BUDGETS: dict[str, _Budget] = {}


def clear() -> None:
    with _LOCK:
        _BUDGETS.clear()


def _current(host: str) -> _Budget:
    state = _BUDGETS.setdefault(host, _Budget())
    now = time.monotonic()
    if now >= state.expires:
        state.observed = None
        server_now = (
            state.server_time + max(0.0, now - state.server_monotonic)
            if state.server_time is not None else time.time()
        )
        if state.server_time is not None:
            state.server_window = max(state.server_window, int(server_now // 60))
        # One extra second avoids reopening on a rounded server Date boundary.
        state.expires = now + 61.0 - server_now % 60.0
    while state.attempts and state.attempts[0] <= now - 60.0:
        state.attempts.popleft()
    while state.reservations and state.reservations[0][0] <= now - 60.0:
        state.reservations.popleft()
    return state


def _projected(state: _Budget) -> int:
    # Count the last minute's local reservations in addition to the exchange
    # high water. Some completed requests are deliberately counted twice. This
    # small conservative margin avoids losing in-flight requests when responses
    # arrive out of order or cross an exchange minute boundary.
    return (state.observed or 0) + sum(cost for _at, cost in state.reservations)


def _parameters(url: str, body: Mapping[str, Any] | str | None) -> dict[str, Any]:
    result = dict(parse_qsl(urlsplit(url).query))
    if isinstance(body, str):
        result.update(parse_qsl(body))
    elif isinstance(body, Mapping):
        result.update(body)
    return result


def _cost(method: str, path: str, params: Mapping[str, Any]) -> int:
    # Only estimate endpoints used by this bot; unknown endpoints reserve a
    # conservative 50. Exchange response headers remain the authoritative total.
    if path == "/fapi/v1/klines":
        try:
            limit = int(params.get("limit", 500))
        except (TypeError, ValueError):
            return 10
        return 1 if limit < 100 else 2 if limit < 500 else 5 if limit <= 1000 else 10
    if path == "/fapi/v1/positionSide/dual":
        return 30 if method == "GET" else 1
    if path in {"/fapi/v1/openOrders", "/fapi/v1/openAlgoOrders"}:
        return 1 if params.get("symbol") else 40
    if path == "/fapi/v1/premiumIndex":
        return 1 if params.get("symbol") else 10
    if path in {"/fapi/v1/order", "/fapi/v1/algoOrder"} and method == "POST":
        return 0
    if path in {"/fapi/v2/account", "/fapi/v2/positionRisk", "/fapi/v1/userTrades", "/fapi/v1/allOrders"}:
        return 5
    if path in {
        "/fapi/v1/time", "/fapi/v1/exchangeInfo", "/fapi/v1/listenKey",
        "/fapi/v1/marginType", "/fapi/v1/leverage", "/fapi/v1/order",
        "/fapi/v1/algoOrder", "/fapi/v1/openAlgoOrders",
    }:
        return 1
    return 50


def _priority(method: str, path: str, params: Mapping[str, Any]) -> bool:
    if path == "/fapi/v1/time":
        return True  # Required to freshly sign a protective/cancel request.
    if path == "/fapi/v1/listenKey":
        return method in {"PUT", "DELETE"}
    if method == "DELETE" and path in {
        "/fapi/v1/order", "/fapi/v1/algoOrder", "/fapi/v1/allOpenOrders",
        "/fapi/v1/algoOpenOrders",
    }:
        return True
    if method == "GET" and path in {
        "/fapi/v1/order", "/fapi/v1/algoOrder", "/fapi/v1/allOrders",
        "/fapi/v1/userTrades", "/fapi/v1/openOrders", "/fapi/v1/openAlgoOrders",
    }:
        return True  # Fill recovery and verifying exchange-side protection.
    if method == "POST" and path in {"/fapi/v1/order", "/fapi/v1/algoOrder"}:
        return any(str(params.get(flag, "")).lower() == "true" for flag in ("reduceOnly", "closePosition"))
    return False


def reserve(method: str, url: str, body: Mapping[str, Any] | str | None) -> float:
    """Reserve an actual HTTP attempt, or return its local retry timestamp."""
    parsed = urlsplit(url)
    host, path = parsed.hostname or "", parsed.path
    if host not in HOSTS or not path.startswith("/fapi/"):
        return 0.0
    params = _parameters(url, body)
    cost = _cost(method, path, params)
    priority = _priority(method, path, params)
    risk_reduction = priority and cost == 0 and method == "POST"
    with _LOCK:
        state = _current(host)
        # Normal work uses at most 75% of the observed IP budget; risk reduction
        # and fill reconciliation can use the reserve up to the exchange limit.
        ceiling = state.limit if priority else int(state.limit * 0.75)
        retry_in = 0.0
        if not risk_reduction and _projected(state) + cost >= ceiling:
            retry_in = max(0.0, state.expires - time.monotonic())
        if not risk_reduction and len(state.attempts) >= (120 if priority else 60):
            retry_in = max(retry_in, state.attempts[0] + 60.1 - time.monotonic())
        if retry_in > 0:
            state.deferred += 1
            return time.time() + retry_in
        now = time.monotonic()
        state.attempts.append(now)
        state.reservations.append((now, cost))
        state.total_attempts += 1
        count, total = len(state.attempts), state.total_attempts
    # Full attempt count, including retries/network errors, separate from the
    # old sampled response-weight log. Never log query/body/auth headers.
    LOG.info(
        "exchange_http_request method=%s host=%s path=%s local_requests_60s=%s local_requests_total=%s priority=%s",
        method, host, path, count, total, priority,
    )
    return 0.0


def observe(url: str, headers: Mapping[str, Any] | object) -> None:
    host = urlsplit(url).hostname or ""
    if host not in HOSTS:
        return
    try:
        normalized = {str(k).lower(): str(v) for k, v in headers.items()}  # type: ignore[union-attr]
        weight = int(normalized.get("x-mbx-used-weight-1m", ""))
    except (AttributeError, ValueError, TypeError):
        return
    if weight < 0:
        return
    try:
        response_time = parsedate_to_datetime(normalized["date"]).timestamp()
    except (KeyError, ValueError, TypeError, OverflowError):
        response_time = None
    with _LOCK:
        state = _current(host)
        server_now = response_time if response_time is not None else (
            state.server_time + max(0.0, time.monotonic() - state.server_monotonic)
            if state.server_time is not None else time.time()
        )
        window = int(server_now // 60)
        first_server_date = response_time is not None and not state.has_server_date
        if window < state.server_window and not first_server_date:
            return  # Late response from an older exchange minute.
        if window > state.server_window and state.server_window >= 0 and not first_server_date:
            state.observed = None
        same_window = window == state.server_window and state.server_time is not None and not first_server_date
        state.server_window = window
        expires = time.monotonic() + 61.0 - server_now % 60.0
        state.expires = max(state.expires, expires) if same_window else expires
        state.server_time = server_now
        state.server_monotonic = time.monotonic()
        state.has_server_date = state.has_server_date or response_time is not None
        state.observed = max(state.observed or 0, weight)


def configure(url: str, payload: Any) -> None:
    parsed = urlsplit(url)
    if parsed.hostname not in HOSTS or parsed.path != "/fapi/v1/exchangeInfo" or not isinstance(payload, dict):
        return
    for row in payload.get("rateLimits") or []:
        if not isinstance(row, dict) or row.get("rateLimitType") != "REQUEST_WEIGHT" or row.get("interval") != "MINUTE" or row.get("intervalNum") != 1:
            continue
        try:
            limit = int(row["limit"])
        except (KeyError, ValueError, TypeError):
            continue
        if limit > 0:
            with _LOCK:
                # Never loosen the known 2400 ceiling from a malformed or
                # unexpectedly larger response; honor lower exchange limits.
                _current(parsed.hostname).limit = min(2400, limit)


def status(url: str) -> dict[str, Any]:
    host = urlsplit(url).hostname or ""
    if host not in HOSTS:
        return {}
    with _LOCK:
        state = _current(host)
        return {
            "local_requests_60s": len(state.attempts),
            "local_requests_total": state.total_attempts,
            "observed_ip_weight_1m": state.observed,
            "reserved_ip_weight_1m": _projected(state),
            "normal_weight_ceiling": int(state.limit * 0.75),
            "exchange_weight_ceiling": state.limit,
            "local_deferred_total": state.deferred,
        }
