from __future__ import annotations

from unittest.mock import patch

from btc_futures_bot.http_client import ApiError, request_json


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
