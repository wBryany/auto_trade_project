from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from btc_futures_bot.main import load_config
from btc_futures_bot.strategy import MultiTimeframeStrategy, StrategyConfig
from btc_futures_bot.trade_model.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    strategy_config_hash,
)
from btc_futures_bot.trade_model.policy import (
    MANIFEST_VERSION,
    MODEL_TYPE,
    TRAINING_PIPELINE_VERSION,
    build_execution_policy,
    execution_policy_hash,
)
from btc_futures_bot.trade_model.training import (
    DEFAULT_LIGHTGBM_PARAMS,
    ApprovalPolicy,
    BarrierConfig,
    SplitConfig,
    assess_live_approval,
    choose_validation_threshold,
    chronological_purged_split,
    dataclass_json,
    fixed_triple_barrier_live_approval,
    load_candles,
    load_funding_rates,
    predict_samples,
    replay_primary_candidates,
    resolve_replay_history_window,
    sha256_file,
    split_ranges,
    threshold_metrics,
    train_lightgbm_native,
    write_json_atomic,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the configured primary strategy, train a LightGBM meta gate, "
            "and write a native text artifact plus an auditable manifest."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config.binance.model2.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/binance_meta"))
    parser.add_argument("--symbol", help="defaults to the configured Binance symbol")
    parser.add_argument("--one-minute", type=Path)
    parser.add_argument("--five-minute", type=Path)
    parser.add_argument("--one-hour", type=Path)
    parser.add_argument("--funding-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/trade_model_2_0"))
    parser.add_argument("--model-version", help="immutable version recorded in the manifest")
    parser.add_argument("--take-profit-pct", type=float)
    parser.add_argument("--stop-loss-pct", type=float)
    parser.add_argument("--horizon-bars", type=int)
    parser.add_argument("--fee-pct-per-side", type=float)
    parser.add_argument("--slippage-pct-per-side", type=float)
    parser.add_argument("--fallback-funding-rate-per-8h", type=float)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--purge-minutes",
        type=float,
        help="gap removed before each later partition (default: one label horizon)",
    )
    parser.add_argument("--embargo-minutes", type=float, default=5.0)
    parser.add_argument("--min-train-samples", type=int, default=100)
    parser.add_argument("--min-validation-samples", type=int, default=30)
    parser.add_argument("--min-holdout-samples", type=int, default=30)
    parser.add_argument("--min-selected-validation", type=int, default=10)
    parser.add_argument("--min-selected-holdout", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--min-profit-factor", type=float, default=1.0)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--history-limit",
        type=int,
        help=(
            "closed bars supplied per timeframe; defaults to configured "
            "candle_limit minus the newest forming bar"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    raw_config = load_config(str(args.config))
    try:
        history_window = resolve_replay_history_window(
            int(raw_config.get("candle_limit", 300)),
            args.history_limit,
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(f"invalid replay history window: {error}") from error
    strategy_mapping = dict(raw_config.get("strategy") or {})
    strategy = MultiTimeframeStrategy(StrategyConfig(**strategy_mapping))
    exchange = dict((raw_config.get("exchanges") or {}).get("binance") or {})
    symbol = str(args.symbol or exchange.get("symbol") or "BTCUSDT").upper()
    paths = {
        "1m": args.one_minute or args.data_dir / f"{symbol}-1m.jsonl",
        "5m": args.five_minute or args.data_dir / f"{symbol}-5m.jsonl",
        "1h": args.one_hour or args.data_dir / f"{symbol}-1h.jsonl",
    }
    funding_path = args.funding_file or args.data_dir / f"{symbol}-funding.jsonl"
    candles = {timeframe: load_candles(path) for timeframe, path in paths.items()}
    funding_rates = load_funding_rates(funding_path)

    runtime_policy = build_execution_policy(raw_config)
    runtime_barrier = dict(runtime_policy["labeling"]["barrier_config"])
    stop_loss_pct = float(
        args.stop_loss_pct
        if args.stop_loss_pct is not None
        else runtime_barrier["stop_loss_pct"]
    )
    take_profit_pct = float(
        args.take_profit_pct
        if args.take_profit_pct is not None
        else stop_loss_pct * float(strategy.config.take_profit_r)
    )
    horizon_bars = int(
        args.horizon_bars
        if args.horizon_bars is not None
        else runtime_barrier["horizon_bars"]
    )
    barrier_config = BarrierConfig(
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        horizon_bars=horizon_bars,
        fee_pct_per_side=float(
            args.fee_pct_per_side
            if args.fee_pct_per_side is not None
            else runtime_barrier["fee_pct_per_side"]
        ),
        slippage_pct_per_side=float(
            args.slippage_pct_per_side
            if args.slippage_pct_per_side is not None
            else runtime_barrier["slippage_pct_per_side"]
        ),
        fallback_funding_rate_per_8h=float(
            args.fallback_funding_rate_per_8h
            if args.fallback_funding_rate_per_8h is not None
            else runtime_barrier["fallback_funding_rate_per_8h"]
        ),
    )
    execution_policy = build_execution_policy(
        raw_config,
        barrier_config=barrier_config,
        closed_history_limit=history_window.closed_history_limit,
    )
    execution_digest = execution_policy_hash(execution_policy)
    samples, replay_stats = replay_primary_candidates(
        candles,
        strategy,
        barrier_config,
        funding_rates,
        history_limit=history_window.closed_history_limit,
    )
    if len(samples) < 3:
        raise SystemExit(
            f"only {len(samples)} unique labelled candidates were produced; at least 3 are required"
        )
    purge_minutes = (
        float(args.purge_minutes)
        if args.purge_minutes is not None
        else float(horizon_bars)
    )
    split_config = SplitConfig(
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        purge_ms=int(purge_minutes * 60_000),
        embargo_ms=int(args.embargo_minutes * 60_000),
    )
    split = chronological_purged_split(samples, split_config)
    if not split.train or not split.validation or not split.holdout:
        raise SystemExit(
            "purge/embargo left an empty partition; download a longer range or reduce the gaps"
        )

    model_path = args.output_dir / "model.txt"
    booster, model_digest = train_lightgbm_native(
        split.train,
        split.validation,
        model_path,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    # Threshold selection has exactly one input partition: validation. Do not
    # compute holdout predictions until this value is frozen.
    validation_probabilities = predict_samples(booster, split.validation)
    policy = ApprovalPolicy(
        min_train_samples=args.min_train_samples,
        min_validation_samples=args.min_validation_samples,
        min_holdout_samples=args.min_holdout_samples,
        min_selected_validation=args.min_selected_validation,
        min_selected_holdout=args.min_selected_holdout,
        min_coverage=args.min_coverage,
        min_validation_profit_factor=args.min_profit_factor,
        min_holdout_profit_factor=args.min_profit_factor,
    )
    selection = choose_validation_threshold(
        validation_probabilities,
        [sample.net_return for sample in split.validation],
        min_selected=policy.min_selected_validation,
        min_coverage=policy.min_coverage,
    )
    frozen_threshold = selection.threshold

    # Holdout is touched once, only for the final frozen-model report/approval.
    holdout_probabilities = predict_samples(booster, split.holdout)
    holdout = threshold_metrics(
        holdout_probabilities,
        [sample.net_return for sample in split.holdout],
        frozen_threshold,
    )
    train_probabilities = predict_samples(booster, split.train)
    train_metrics = threshold_metrics(
        train_probabilities,
        [sample.net_return for sample in split.train],
        frozen_threshold,
    )
    statistically_qualified, statistical_reasons = assess_live_approval(
        split, selection, holdout, policy
    )
    # Fixed triple-barrier labels are useful for paper filtering experiments,
    # but they do not simulate the engine's dynamic exits and protective order
    # lifecycle. No artifact from this trainer may authorize live enforcement.
    (
        approved,
        live_execution_equivalent,
        fixed_barrier_blockers,
        combined_approval_reasons,
    ) = fixed_triple_barrier_live_approval(statistical_reasons)
    live_approval_blockers = list(fixed_barrier_blockers)
    approval_reasons = list(combined_approval_reasons)

    created_at = datetime.now(timezone.utc)
    model_version = args.model_version or created_at.strftime("meta-%Y%m%dT%H%M%SZ")
    dataset_manifest_path = args.data_dir / "dataset_manifest.json"
    dataset_manifest = _optional_json(dataset_manifest_path)
    input_files = {
        timeframe: _file_record(path, len(candles[timeframe]))
        for timeframe, path in paths.items()
    }
    input_files["funding"] = (
        _file_record(funding_path, len(funding_rates))
        if funding_path.is_file()
        else {"path": str(funding_path.resolve()), "sha256": None, "rows": 0}
    )
    strategy_digest = strategy_config_hash(strategy_mapping)
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "training_pipeline_version": TRAINING_PIPELINE_VERSION,
        "model_type": MODEL_TYPE,
        "model_format": "LightGBM native text",
        "model_version": model_version,
        "model_sha256": model_digest,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "symbol": symbol,
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": FEATURE_SCHEMA_HASH,
        "strategy_config_hash": strategy_digest,
        "execution_policy_hash": execution_digest,
        "execution_policy": execution_policy,
        "implementation_source_sha256": execution_policy[
            "implementation_source_sha256"
        ],
        "strategy_config": strategy_mapping,
        "candidate_generation": {
            **execution_policy["candidate_generation"],
            "history_window": dataclass_json(history_window),
            "replay_stats": dataclass_json(replay_stats),
        },
        "label_config": execution_policy["labeling"]["barrier_config"],
        "label_rules": execution_policy["labeling"]["rules"],
        "split_config": dataclass_json(split_config),
        "split_ranges": split_ranges(split),
        "split_removals": {
            "purged_train": split.purged_train,
            "purged_validation": split.purged_validation,
            "embargoed_validation": split.embargoed_validation,
            "embargoed_holdout": split.embargoed_holdout,
        },
        "input_files": input_files,
        "dataset_manifest": dataset_manifest,
        "config_file": {
            "path": str(args.config.resolve()),
            "sha256": sha256_file(args.config),
        },
        "git_revision": _git_revision(),
        "lightgbm_version": _module_version(booster),
        "training_config": {
            "lightgbm_params": DEFAULT_LIGHTGBM_PARAMS,
            "num_boost_round": args.num_boost_round,
            "early_stopping_rounds": args.early_stopping_rounds,
            "history_limit": args.history_limit,
        },
        "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
        "threshold": frozen_threshold,
        "threshold_selection": {
            "partition": "validation_only",
            **dataclass_json(selection),
        },
        "metrics": {
            "train_at_frozen_threshold": dataclass_json(train_metrics),
            "validation_at_selected_threshold": dataclass_json(selection.metrics),
            "holdout_final_once": dataclass_json(holdout),
        },
        "approval_policy": dataclass_json(policy),
        "statistically_qualified": statistically_qualified,
        "statistical_approval_reasons": list(statistical_reasons),
        "live_execution_equivalent": live_execution_equivalent,
        "live_approval_blockers": live_approval_blockers,
        "approved_for_live": approved,
        "approval_reasons": list(approval_reasons),
    }
    manifest_path = args.output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(f"unique candidates: {len(samples)}")
    print(
        "split after purge/embargo: "
        f"train={len(split.train)} validation={len(split.validation)} holdout={len(split.holdout)}"
    )
    print(
        f"validation threshold={frozen_threshold:.2f} "
        f"expectancy={selection.metrics.net_expectancy} "
        f"profit_factor={selection.metrics.profit_factor} "
        f"coverage={selection.metrics.coverage:.3f}"
    )
    print(
        f"holdout expectancy={holdout.net_expectancy} "
        f"profit_factor={holdout.profit_factor} coverage={holdout.coverage:.3f}"
    )
    print(f"approved_for_live={approved}")
    if approval_reasons:
        print("approval blockers: " + ", ".join(approval_reasons))
    print(f"model -> {model_path}")
    print(f"manifest -> {manifest_path}")
    return 0


def _file_record(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": rows,
    }


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"path": str(path.resolve()), "sha256": sha256_file(path), "valid_json": False}
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "contents": payload,
    }


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _module_version(booster: Any) -> str:
    module_name = booster.__class__.__module__.split(".")[0]
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
