# Model 2.0 minute-scalp experiment

Version: `meta-scalp-20260906-v1`. Deployment: **paper / shadow only**.

This is a newly trained LightGBM artifact, not the old 90-minute model with
an edited hash or reduced threshold. The primary strategy is `scalp_v2`:
closed 1m breakout/EMA reclaim with 5m direction, momentum, turnover and
execution-quality confirmation. Hourly candles are model context, not a
mandatory hourly primary trend gate.

The existing Binance USD-M BTCUSDT public dataset spans 2025-09-01 through
2026-09-04 (530,580 minute bars). Replaying 99 closed bars per timeframe
produced 25,014 unique labelled candidates; after purge/embargo the train,
validation and holdout partitions contain 15,008 / 5,000 / 5,002 candidates.

Labels enter at the next contiguous minute open, with fixed 0.25% SL,
0.375% TP, and a **10-bar** horizon. Each side deducts 0.05% fee and 0.02%
slippage; available funding settlements are included. Threshold selection
uses validation only, with chronological purge and embargo.

No threshold satisfied validation eligibility. The selection fallback was
0.95, accepting **zero** validation and **zero** holdout candidates. Net
expectancy is therefore undefined, not zero or positive. Both
`statistically_qualified` and `approved_for_live` are false. No claim of
statistical advantage or calibrated profit probability is warranted.

The deployed configuration uses `shadow`: the model logs predictions and
its would-reject decisions, but does not veto otherwise risk-approved paper
signals. This tests the minute-scalp primary strategy and collects forward
data; it is **not** an enforced-ML strategy performance test. Enabling
`enforce` with this artifact would predictably suppress entries again.

Limitations:

- Training venue is Binance; runtime market/account context is OKX demo.
- Fixed labels do not reproduce dynamic ATR/structure stops, break-even,
  trailing, adverse exits, the profitable five-minute soft exit or macro
  entry blocks. The engine's ten-minute hard threshold is checked on the
  next closed bar/poll, not guaranteed at exactly 600 seconds.
- Candidates may overlap and are not a capital-constrained portfolio
  backtest. Candidate counts are not executed trade counts.
- Source changes, policy/history/cost changes or model-byte changes fail
  fingerprint validation. `git_revision` identifies the parent checkout at
  training time; `implementation_source_sha256` records the actual worktree
  source used for this artifact.
- The engine explicitly rejects non-paper execution for `scalp_v2`,
  including OKX demo exchange orders. A demo connection is not an order.

Detailed inputs, hashes, parameters, splits, metrics and rejection reasons
are recorded in `manifest.json`. Reproduce with:

```powershell
.venv\Scripts\python.exe scripts/train_meta_model.py --config config.binance.model2.json --data-dir data/binance_meta_12m --output-dir artifacts/trade_model_2_0_scalp --model-version meta-scalp-20260906-v1
```

Keep previous artifacts and report history. Compare forward trades only
after the new deployment boundary and identify the strategy, gate mode and
model version; do not pool the former 90-minute experiment with this one.
