# Trade Model 2.0 model card

Model version: `meta-20250901-20260904-v1`

This is an experimental, paper-only LightGBM meta gate trained from public
Binance USD-M BTCUSDT data spanning 2025-09-01 through 2026-09-04. The frozen
primary strategy produced 3,253 unique labelled candidates from 530,579
decision points.

The validation-only threshold selection chose `0.51`, but it did not satisfy
the minimum coverage/sample policy. At that threshold:

| Partition | Selected / candidates | Coverage | Net expectancy | Profit factor | Win rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 119 / 1,950 | 6.10% | +0.004039 | 50.276 | 98.32% |
| Validation | 8 / 648 | 1.23% | +0.003577 | 5.468 | 87.50% |
| Holdout | 10 / 650 | 1.54% | -0.003054 | 0.266 | 10.00% |

The train/validation-to-holdout collapse is evidence of overfitting. This
artifact has `statistically_qualified=false`, `approved_for_live=false`, and
must not be used for real-money execution. It is retained so the complete
pipeline and the fail-closed paper comparison can be reproduced and audited.

The detailed data hashes, split ranges, costs, label policy, source hashes and
approval blockers are recorded in `manifest.json`.
