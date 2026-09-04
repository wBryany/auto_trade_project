from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from btc_futures_bot.trade_analysis import (
    DEFAULT_TIMEZONE,
    analyze_trade_csv,
    write_analysis_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a TradeReporter trade_report.csv and emit JSON plus Markdown.",
    )
    parser.add_argument("csv_path", type=Path, help="path to TradeReporter's trade_report.csv")
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"timezone used for entry-hour grouping (default: {DEFAULT_TIMEZONE})",
    )
    parser.add_argument(
        "--json-output",
        "--json",
        dest="json_output",
        type=Path,
        help="JSON output path (default: <csv_stem>_analysis.json)",
    )
    parser.add_argument(
        "--markdown-output",
        "--markdown",
        dest="markdown_output",
        type=Path,
        help="Markdown output path (default: <csv_stem>_analysis.md)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.csv_path
    json_path = args.json_output or source.with_name(f"{source.stem}_analysis.json")
    markdown_path = args.markdown_output or source.with_name(f"{source.stem}_analysis.md")

    report = analyze_trade_csv(source, timezone_name=args.timezone)
    write_analysis_reports(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
        title=f"交易尸检报告：{source.name}",
    )
    print(f"JSON: {json_path.resolve()}")
    print(f"Markdown: {markdown_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
