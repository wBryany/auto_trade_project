from __future__ import annotations

from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from btc_futures_bot.http_client import ApiError, clear_rate_limits, request_json


class _Response:
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
