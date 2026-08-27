import json

from btc_futures_bot.operation_log import OperationLogger


def test_operation_log_sanitizes_credentials_and_filters(tmp_path):
    logger = OperationLogger(tmp_path / "operation_log.jsonl")
    logger.record(
        "strategy_change",
        "update_strategy",
        details={"api_secret": "must-not-be-written", "min_volume_ratio": {"before": 1.3, "after": 1.5}},
        changed_files=["src/btc_futures_bot/strategy.py"],
    )

    raw = (tmp_path / "operation_log.jsonl").read_text(encoding="utf-8")
    assert "must-not-be-written" not in raw
    assert "min_volume_ratio" in raw
    rows = logger.query(event_type="strategy_change", keyword="volume")
    assert len(rows) == 1
    assert rows[0]["details"]["api_secret"] == "[已保存，不记录]"


def test_operation_log_skips_malformed_rows(tmp_path):
    path = tmp_path / "operation_log.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    logger = OperationLogger(path)
    logger.record("engine_start", "start")
    rows = logger.query()
    assert len(rows) == 1
    json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
