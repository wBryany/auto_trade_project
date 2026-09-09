# Cost-aware scalp research artifact

Version: `meta-scalp-cost-20260907-v1`. Generated on 2026-09-07 from the
cost-aware `scalp_v2` implementation recorded in `manifest.json`.

This is an **unqualified research artifact**, not a profitable-strategy claim.
The Model 2 profile remains `paper` with `trade_model.mode=shadow`: the primary
strategy and independent risk checks control simulated entries; LightGBM scores
are logged, not enforced. No real or exchange-demo orders are authorized by this
artifact. `statistically_qualified=false` and `approved_for_live=false`.

## Reproduction and provenance

```powershell
.venv\Scripts\python.exe scripts/train_meta_model.py --config config.binance.model2.json --data-dir data/binance_meta_12m --output-dir artifacts/trade_model_2_0_scalp_cost --model-version meta-scalp-cost-20260907-v1
```

Training replays 530,580 public Binance BTCUSDT one-minute bars, with matching
five-minute/hourly bars and historical funding, using at most 99 closed bars per
timeframe. The deployment remains OKX BTC-USDT-SWAP: this venue difference is a
research limitation, not evidence of OKX performance. Candidate admission uses
the deployment's effective fee/slippage assumptions, not data-source defaults.
The manifest records input hashes, implementation hashes, effective strategy,
costs, risk policy, labeling, and chronological split boundaries.

The primary cost filter checks six non-overlapping, completed ten-minute
windows. Both the median favorable excursion and the latest excursion must
cover configured round-trip costs plus the minimum net edge. With current
assumptions, this is approximately 0.290203% long / 0.289797% short. These are
historical activity checks, not predicted returns. Funding admission estimates
whole eight-hour intervals and does not model proximity to a settlement.

## Evaluation outcome

- Unique candidates: 1,605. After purge/embargo: 963 train, 319 validation,
  320 holdout.
- Best iteration: 1. No tested threshold met validation eligibility; the
  fail-closed threshold is 0.95, not an optimized trading recommendation.
- Selected validation/holdout trades: 0 / 0; coverage: 0% / 0%.
- Net expectancy and win rate are undefined for those empty selections.
  Reported profit factor 0 is the evaluator's empty-selection sentinel.
- Statistical acceptance failed; see the manifest's complete approval reasons.
- Fixed triple-barrier labels do not reproduce dynamic engine exits, protective
  orders, or position management. They cannot qualify live execution.

Four recorded losing paper entries from 2026-09-06 were separately replayed
against only their pre-entry public OKX candles: this filter rejects all four.
That is a regression check on known incidents, not an independent out-of-sample
profitability test. The filter can reduce entry frequency and still admit losses.

Previous artifacts are retained for audit. Do not edit model bytes, approval
flags, or manifest fingerprints to make a changed implementation load; retrain
and evaluate a new version instead.
