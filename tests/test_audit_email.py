from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from btc_futures_bot import audit_email
from btc_futures_bot import main as main_module


_REPORT = """BTC 整点巡检报告
行情分析：1h 下降趋势放缓，最近空单入场偏迟。
策略修改：无，当前样本不足。
调试链接：https://example.invalid/order?signature=report-signature-secret
"""
_SMTP_PASSWORD = "smtp-password-must-not-leak"
_API_SECRET = "exchange-api-secret-must-not-leak"


class _Harness:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        enabled: bool = True,
        ready: bool = True,
        delivery_result: bool = True,
        delivery_error: BaseException | None = None,
    ) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.send_calls: list[tuple[str, str, str, float]] = []
        self.closed = 0
        self.config_path = tmp_path / "config.binance.local.json"
        self.email_mapping = {
            "enabled": enabled,
            "smtp_host": "smtp.example.com",
            "sender": "bot@example.com",
            "recipients": ["owner@example.com"],
            # Deliberately unexpected secret-bearing fields exercise the rule that
            # neither configuration nor exception details are echoed by this CLI.
            "smtp_password": _SMTP_PASSWORD,
        }
        self.raw_config = {
            "mode": "live",
            "active_exchange": "binance",
            "report_dir": str(tmp_path / "reports"),
            "report_timezone": "Asia/Shanghai",
            "email_notifications": self.email_mapping,
            "exchanges": {
                "binance": {"enabled": True, "environment": "production"}
            },
            "credentials": {
                "binance": {"production": {"api_secret": _API_SECRET}}
            },
        }
        self.email_config = SimpleNamespace(
            enabled=enabled,
            timeout_seconds=7.0,
        )

        harness = self

        def fake_load_runtime_dotenv(config_path: str) -> list[str]:
            harness.events.append(("dotenv", config_path))
            return [str(tmp_path / ".env")]

        def fake_load_config(config_path: str) -> dict[str, Any]:
            harness.events.append(("config", config_path))
            return harness.raw_config

        class FakeEmailNotificationConfig:
            @classmethod
            def from_mapping(
                cls,
                raw: dict[str, Any] | None,
                **defaults: Any,
            ) -> SimpleNamespace:
                harness.events.append(("email_config", raw, defaults))
                return harness.email_config

        class FakeEmailNotifier:
            def __init__(self, config: SimpleNamespace) -> None:
                harness.events.append(("notifier", config))

            @property
            def ready(self) -> bool:
                return ready

            def status(self) -> dict[str, Any]:
                return {
                    "enabled": enabled,
                    "ready": ready,
                    "last_error": (
                        f"delivery failed: {_SMTP_PASSWORD}; {_API_SECRET}; {_REPORT}"
                    ),
                }

            def send_strategy_inspection_report(
                self,
                report: str,
                status: str,
                run_id: str,
                timeout: float,
            ) -> bool:
                harness.send_calls.append((report, status, run_id, timeout))
                if not enabled:
                    raise RuntimeError("邮件通知未启用")
                if not ready:
                    raise RuntimeError("邮件通知配置不完整")
                if delivery_error is not None:
                    raise delivery_error
                return delivery_result

            def close(self) -> None:
                harness.closed += 1

        monkeypatch.setattr(main_module, "load_runtime_dotenv", fake_load_runtime_dotenv)
        monkeypatch.setattr(main_module, "load_config", fake_load_config)
        monkeypatch.setattr(
            audit_email,
            "EmailNotificationConfig",
            FakeEmailNotificationConfig,
        )
        monkeypatch.setattr(audit_email, "EmailNotifier", FakeEmailNotifier)


def _arguments(
    harness: _Harness,
    report_path: Path,
    *,
    status: str = "changed",
    run_id: str = "hourly-20260904T010000+0800",
) -> list[str]:
    return [
        "--config",
        str(harness.config_path),
        "--report-file",
        str(report_path),
        "--status",
        status,
        "--run-id",
        run_id,
    ]


def _combined_output(capsys: pytest.CaptureFixture[str]) -> str:
    captured = capsys.readouterr()
    return f"{captured.out}\n{captured.err}"


def _assert_no_sensitive_output(output: str) -> None:
    assert _REPORT not in output
    assert "report-signature-secret" not in output
    assert _SMTP_PASSWORD not in output
    assert _API_SECRET not in output


def test_stdin_report_is_decoded_as_utf8_in_a_child_process() -> None:
    report = "执行情况：服务正常；策略修改：无修改。\n"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from btc_futures_bot.audit_email import _read_report; "
                "sys.stdout.buffer.write(_read_report('-').encode('utf-8'))"
            ),
        ],
        input=report.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert completed.stdout.decode("utf-8") == report


@pytest.mark.parametrize("status", ["no_change", "changed", "failed"])
def test_main_reads_utf8_report_loads_config_and_sends_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
) -> None:
    report_path = tmp_path / "整点巡检报告.md"
    report_path.write_text(_REPORT, encoding="utf-8")
    harness = _Harness(monkeypatch, tmp_path)
    run_id = "hourly-20260904T010000+0800"

    result = audit_email.main(
        _arguments(harness, report_path, status=status, run_id=run_id)
    )

    assert result == 0
    assert harness.events[0] == ("dotenv", str(harness.config_path))
    assert harness.events[1] == ("config", str(harness.config_path))
    config_event = next(event for event in harness.events if event[0] == "email_config")
    assert config_event[1] is harness.email_mapping
    assert config_event[2]["default_timezone"] == "Asia/Shanghai"
    expected_state = (
        tmp_path
        / "reports"
        / "binance-production"
        / "email_notification_state.json"
    )
    assert Path(config_event[2]["default_state_path"]) == expected_state
    assert len(harness.send_calls) == 1
    sent_report, sent_status, sent_run_id, timeout = harness.send_calls[0]
    assert sent_report == _REPORT
    assert sent_status == status
    assert sent_run_id == run_id
    assert timeout >= harness.email_config.timeout_seconds
    assert harness.closed == 1
    _assert_no_sensitive_output(_combined_output(capsys))


def test_main_returns_nonzero_without_sending_when_email_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.md"
    report_path.write_text(_REPORT, encoding="utf-8")
    harness = _Harness(monkeypatch, tmp_path, enabled=False, ready=False)

    result = audit_email.main(_arguments(harness, report_path))

    assert result != 0
    assert harness.send_calls == []
    assert harness.closed == 1
    _assert_no_sensitive_output(_combined_output(capsys))


def test_main_returns_nonzero_without_sending_when_email_is_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.md"
    report_path.write_text(_REPORT, encoding="utf-8")
    harness = _Harness(monkeypatch, tmp_path, ready=False)

    result = audit_email.main(_arguments(harness, report_path))

    assert result != 0
    assert harness.send_calls == []
    assert harness.closed == 1
    _assert_no_sensitive_output(_combined_output(capsys))


def test_main_returns_nonzero_when_delivery_is_not_confirmed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.md"
    report_path.write_text(_REPORT, encoding="utf-8")
    harness = _Harness(monkeypatch, tmp_path, delivery_result=False)

    result = audit_email.main(_arguments(harness, report_path))

    assert result != 0
    assert len(harness.send_calls) == 1
    assert harness.closed == 1
    _assert_no_sensitive_output(_combined_output(capsys))


def test_main_returns_nonzero_and_closes_notifier_when_delivery_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.md"
    report_path.write_text(_REPORT, encoding="utf-8")
    leaked_error = RuntimeError(
        f"SMTP rejected {_SMTP_PASSWORD}; {_API_SECRET}; body={_REPORT}"
    )
    harness = _Harness(monkeypatch, tmp_path, delivery_error=leaked_error)

    result = audit_email.main(_arguments(harness, report_path))

    assert result != 0
    assert len(harness.send_calls) == 1
    assert harness.closed == 1
    _assert_no_sensitive_output(_combined_output(capsys))
