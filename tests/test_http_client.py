from __future__ import annotations

import logging
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from btc_futures_bot.http_client import (
    ApiError,
    clear_rate_limits,
    is_rate_limit_error,
    request_json,
)


class _Response:
    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        status: int = 200,
    ) -> None:
        self.headers = headers or {}
        self.status = status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"ok":true}'


def test_get_retries_direct_timeout_errors() -> None:
    with (
        patch(
            "btc_futures_bot.http_client.urlopen",
            side_effect=[TimeoutError("timed out"), _Response()],
        ) as mocked_urlopen,
        patch("btc_futures_bot.http_client.time.sleep") as mocked_sleep,
    ):
        result = request_json("GET", "https://example.test/market")

    assert result == {"ok": True}
    assert mocked_urlopen.call_count == 2
    mocked_sleep.assert_called_once_with(0.5)


def test_get_retry_count_can_be_disabled_for_expiring_signed_requests() -> None:
    with (
        patch(
            "btc_futures_bot.http_client.urlopen",
            side_effect=TimeoutError("timed out"),
        ) as mocked_urlopen,
        patch("btc_futures_bot.http_client.time.sleep") as mocked_sleep,
    ):
        try:
            request_json(
                "GET",
                "https://example.test/private",
                params={"timestamp": 1, "signature": "secret-signature"},
                max_attempts=1,
            )
        except ApiError as error:
            assert "[REDACTED]" in str(error)
            assert "secret-signature" not in str(error)
        else:
            raise AssertionError("single-attempt timeout must be surfaced")

    assert mocked_urlopen.call_count == 1
    mocked_sleep.assert_not_called()


def test_signed_url_is_redacted_from_urlerror_chain_message() -> None:
    with patch(
        "btc_futures_bot.http_client.urlopen",
        side_effect=URLError("temporary TLS failure"),
    ):
        try:
            request_json(
                "GET",
                "https://example.test/private",
                params={"signature": "must-not-leak"},
                max_attempts=1,
            )
        except ApiError as error:
            assert "[REDACTED]" in str(error)
            assert "must-not-leak" not in str(error)
        else:
            raise AssertionError("network failure must be surfaced")


def test_post_does_not_retry_direct_timeout_errors() -> None:
    with (
        patch(
            "btc_futures_bot.http_client.urlopen",
            side_effect=TimeoutError("timed out"),
        ) as mocked_urlopen,
        patch("btc_futures_bot.http_client.time.sleep") as mocked_sleep,
    ):
        try:
            request_json("POST", "https://example.test/order", body={"side": "buy"})
        except ApiError as error:
            assert "network error" in str(error)
        else:
            raise AssertionError("POST timeout must be surfaced")

    assert mocked_urlopen.call_count == 1
    mocked_sleep.assert_not_called()


def test_success_weight_logging_is_query_free_and_bounded(caplog: object) -> None:
    clear_rate_limits()
    clock = {"now": 1_000.0}
    responses = [
        _Response(headers={"X-MBX-USED-WEIGHT-1M": value})
        for value in ("10", "50", "110", "111")
    ]
    try:
        with (
            patch("btc_futures_bot.http_client.urlopen", side_effect=responses),
            patch(
                "btc_futures_bot.http_client.time.time",
                side_effect=lambda: clock["now"],
            ),
            caplog.at_level(logging.INFO, logger="btc_futures_bot.http_client"),  # type: ignore[attr-defined]
        ):
            for _index in range(4):
                request_json(
                    "GET",
                    "https://fapi.binance.com/fapi/v1/userTrades",
                    params={"symbol": "BTCUSDT", "signature": "must-not-leak"},
                )

        messages = [
            record.getMessage()
            for record in caplog.records  # type: ignore[attr-defined]
            if record.name == "btc_futures_bot.http_client"
        ]
        assert len(messages) == 2
        assert "used_weight=1m:10" in messages[0]
        assert "used_weight=1m:110" in messages[1]
        assert all("host=fapi.binance.com" in message for message in messages)
        assert all("path=/fapi/v1/userTrades" in message for message in messages)
        assert all("status=200" in message for message in messages)
        assert all("retry_at=0" in message for message in messages)
        assert all("?" not in message for message in messages)
        assert all("must-not-leak" not in message for message in messages)
        assert all("signature" not in message for message in messages)
    finally:
        clear_rate_limits()


def test_success_weight_logging_emits_again_after_timebox(caplog: object) -> None:
    clear_rate_limits()
    clock = {"now": 1_000.0}
    try:
        with (
            patch(
                "btc_futures_bot.http_client.urlopen",
                side_effect=[
                    _Response(headers={"x-mbx-used-weight-1m": "5"}),
                    _Response(headers={"x-mbx-used-weight-1m": "6"}),
                ],
            ),
            patch(
                "btc_futures_bot.http_client.time.time",
                side_effect=lambda: clock["now"],
            ),
            caplog.at_level(logging.INFO, logger="btc_futures_bot.http_client"),  # type: ignore[attr-defined]
        ):
            request_json("GET", "https://fapi.binance.com/fapi/v1/time")
            clock["now"] += 61.0
            request_json("GET", "https://fapi.binance.com/fapi/v1/time")

        messages = [
            record.getMessage()
            for record in caplog.records  # type: ignore[attr-defined]
            if record.name == "btc_futures_bot.http_client"
        ]
        assert len(messages) == 2
        assert "used_weight=1m:5" in messages[0]
        assert "used_weight=1m:6" in messages[1]
    finally:
        clear_rate_limits()


def test_rate_limit_logging_is_query_free_and_records_deadline(caplog: object) -> None:
    clear_rate_limits()
    error = HTTPError(
        "https://fapi.binance.com/fapi/v1/userTrades",
        429,
        "Too Many Requests",
        {
            "Retry-After": "30",
            "X-MBX-USED-WEIGHT-1M": "2401",
        },
        BytesIO(b'{"code":-1003,"msg":"Too many requests"}'),
    )
    try:
        with (
            patch("btc_futures_bot.http_client.urlopen", side_effect=error),
            patch("btc_futures_bot.http_client.time.time", return_value=1_000.0),
            caplog.at_level(logging.WARNING, logger="btc_futures_bot.http_client"),  # type: ignore[attr-defined]
        ):
            with pytest.raises(ApiError):
                request_json(
                    "GET",
                    "https://fapi.binance.com/fapi/v1/userTrades",
                    params={"symbol": "BTCUSDT", "signature": "must-not-leak"},
                )

        messages = [
            record.getMessage()
            for record in caplog.records  # type: ignore[attr-defined]
            if record.name == "btc_futures_bot.http_client"
            and record.levelno == logging.WARNING
        ]
        assert messages == [
            "exchange_http_rate_limit host=fapi.binance.com "
            "path=/fapi/v1/userTrades status=429 used_weight=1m:2401 "
            "retry_at=1030.000"
        ]
        assert "?" not in messages[0]
        assert "must-not-leak" not in messages[0]
        assert "signature" not in messages[0]
        assert "BTCUSDT" not in messages[0]
    finally:
        clear_rate_limits()


def test_rate_limit_is_not_retried_and_blocks_followup_requests() -> None:
    error = HTTPError(
        "https://example.test/market",
        429,
        "too many requests",
        {"Retry-After": "30"},
        BytesIO(b'{"code":-1003,"msg":"Too many requests"}'),
    )
    clear_rate_limits()
    try:
        with (
            patch("btc_futures_bot.http_client.urlopen", side_effect=error) as mocked_urlopen,
            patch("btc_futures_bot.http_client.time.time", return_value=1_000.0),
            patch("btc_futures_bot.http_client.time.sleep") as mocked_sleep,
        ):
            try:
                request_json("GET", "https://example.test/market")
            except ApiError as raised:
                assert raised.status_code == 429
                assert raised.rate_limited
                assert raised.retry_at == 1_030.0
            else:
                raise AssertionError("429 must be surfaced with retry metadata")

            try:
                request_json("GET", "https://example.test/other")
            except ApiError as blocked:
                assert blocked.rate_limited
                assert blocked.retry_after_seconds == 30.0
            else:
                raise AssertionError("same-host followup request must be blocked locally")

        assert mocked_urlopen.call_count == 1
        mocked_sleep.assert_not_called()
    finally:
        clear_rate_limits()


def test_binance_ban_timestamp_takes_precedence_over_default_backoff() -> None:
    error = HTTPError(
        "https://fapi.binance.com/fapi/v1/klines",
        418,
        "banned",
        {},
        BytesIO(
            b'{"code":-1003,"msg":"IP banned until 1700000123000. Please use websocket."}'
        ),
    )
    clear_rate_limits()
    try:
        with (
            patch("btc_futures_bot.http_client.urlopen", side_effect=error),
            patch("btc_futures_bot.http_client.time.time", return_value=1_700_000_000.0),
        ):
            try:
                request_json("GET", "https://fapi.binance.com/fapi/v1/klines")
            except ApiError as raised:
                assert raised.status_code == 418
                assert raised.retry_at == 1_700_000_123.0
            else:
                raise AssertionError("418 must be surfaced")
    finally:
        clear_rate_limits()


def test_rate_limit_classifier_uses_http_status_or_structured_binance_code() -> None:
    assert is_rate_limit_error(ApiError("teapot ban", status_code=418))
    assert is_rate_limit_error(ApiError("too many requests", status_code=429))
    assert is_rate_limit_error(
        ApiError("Binance request rejected", status_code=400, api_code=-1003)
    )
    assert not is_rate_limit_error(
        ApiError("unrelated bad request", status_code=400, api_code=-1100)
    )


def test_http_400_binance_minus_1003_is_structured_and_blocks_followup_requests() -> None:
    error = HTTPError(
        "https://fapi.binance.com/fapi/v1/account",
        400,
        "bad request",
        {},
        BytesIO(
            b'{"code":-1003,"msg":"Too many requests; IP banned until 1700000123000."}'
        ),
    )
    clear_rate_limits()
    try:
        with (
            patch("btc_futures_bot.http_client.urlopen", side_effect=error) as mocked_urlopen,
            patch("btc_futures_bot.http_client.time.time", return_value=1_700_000_000.0),
        ):
            try:
                request_json("GET", "https://fapi.binance.com/fapi/v1/account")
            except ApiError as raised:
                assert raised.status_code == 400
                assert raised.api_code == -1003
                assert is_rate_limit_error(raised)
                assert raised.rate_limited
                assert raised.retry_at == 1_700_000_123.0
            else:
                raise AssertionError("Binance -1003 must be surfaced as a rate-limit error")

            try:
                request_json("GET", "https://fapi.binance.com/fapi/v1/openOrders")
            except ApiError as blocked:
                assert is_rate_limit_error(blocked)
                assert blocked.rate_limited
                assert blocked.retry_after_seconds == 123.0
            else:
                raise AssertionError("same-host followup request must be blocked locally")

        assert mocked_urlopen.call_count == 1
    finally:
        clear_rate_limits()
