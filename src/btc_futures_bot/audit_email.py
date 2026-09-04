from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .notifications import (
    EmailNotificationConfig,
    EmailNotifier,
    prepare_strategy_inspection_report,
)


_MAX_REPORT_BYTES = 64 * 1024
def _active_exchange(config: dict[str, object]) -> str:
    selected = str(config.get("active_exchange") or "").strip().lower()
    exchanges = config.get("exchanges") or {}
    if selected:
        return selected
    if isinstance(exchanges, dict) and "binance" in exchanges:
        return "binance"
    if isinstance(exchanges, dict) and exchanges:
        return str(next(iter(exchanges))).strip().lower()
    return "binance"


def _read_report(path: str) -> str:
    if path == "-":
        reconfigure = getattr(sys.stdin, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="strict")
        report = sys.stdin.read(_MAX_REPORT_BYTES + 1)
        if len(report.encode("utf-8")) > _MAX_REPORT_BYTES:
            raise ValueError("巡检报告不能超过 64 KiB")
        if not report.strip():
            raise ValueError("巡检报告不能为空")
        return report
    report_path = Path(path)
    if report_path.stat().st_size > _MAX_REPORT_BYTES:
        raise ValueError("巡检报告不能超过 64 KiB")
    report = report_path.read_text(encoding="utf-8")
    if len(report.encode("utf-8")) > _MAX_REPORT_BYTES:
        raise ValueError("巡检报告不能超过 64 KiB")
    if not report.strip():
        raise ValueError("巡检报告不能为空")
    return report


def load_runtime_dotenv(config_path: str) -> list[str]:
    from .main import load_runtime_dotenv as shared_loader

    return shared_loader(config_path)


def load_config(config_path: str) -> dict[str, object]:
    from .main import load_config as shared_loader

    return shared_loader(config_path)


def report_directory(config: dict[str, object], exchange: str) -> Path:
    from .main import report_directory as shared_report_directory

    return shared_report_directory(config, exchange)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="发送 BTC 策略整点巡检报告邮件")
    parser.add_argument("--config", required=True, help="交易引擎配置文件")
    parser.add_argument(
        "--report-file",
        required=True,
        help="UTF-8 巡检报告正文文件；使用 - 从标准输入读取",
    )
    parser.add_argument(
        "--status",
        required=True,
        choices=("no_change", "changed", "failed"),
        help="本次巡检结果",
    )
    parser.add_argument("--run-id", required=True, help="本次巡检的唯一批次标识")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅校验报告完整性，不加载邮件配置也不发送",
    )
    args = parser.parse_args(argv)

    notifier: EmailNotifier | None = None
    stage = "读取报告"
    try:
        report = _read_report(args.report_file)
        stage = "校验报告"
        report = prepare_strategy_inspection_report(report, run_id=args.run_id)
        if args.validate_only:
            print("巡检报告校验通过")
            return 0

        stage = "加载配置"
        load_runtime_dotenv(args.config)
        config = load_config(args.config)
        exchange = _active_exchange(config)
        default_state_path = report_directory(config, exchange) / "email_notification_state.json"
        email_config = EmailNotificationConfig.from_mapping(
            config.get("email_notifications", {}),
            default_timezone=str(config.get("report_timezone", "Asia/Shanghai")),
            default_state_path=str(default_state_path),
        )
        notifier = EmailNotifier(email_config)
        if not email_config.enabled:
            raise RuntimeError("邮件通知未启用")
        if not notifier.ready:
            raise RuntimeError("邮件通知配置不完整")
        stage = "投递邮件"
        delivered = notifier.send_strategy_inspection_report(
            report,
            status=args.status,
            run_id=args.run_id,
            timeout=email_config.timeout_seconds + 5.0,
        )
        if not delivered:
            raise RuntimeError("邮件投递未确认")
    except Exception as error:
        # SMTP and config exceptions can contain credentials or the report body.
        # The scheduled task only needs a safe failure signal and stage here.
        print(
            f"巡检报告邮件发送失败（阶段：{stage}，类型：{type(error).__name__}）",
            file=sys.stderr,
        )
        return 1
    finally:
        if notifier is not None:
            notifier.close()

    print("巡检报告邮件已确认发送")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
