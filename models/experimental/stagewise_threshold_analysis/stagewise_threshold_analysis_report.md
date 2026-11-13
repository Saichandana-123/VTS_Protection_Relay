# Stage-wise Threshold Sensitivity Analysis

These are exploratory validation results on cases `case_088`–`case_108`; no threshold was selected for deployment or independently validated.

The synchronized sequential execution order and all existing checkpoints were preserved. Only external sigmoid decision thresholds were changed.

## Aggregate results

| Configuration | Output | Accuracy | Precision | Recall | F1 | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_all_0.5 | SLOW_BLOCK | 90.1092% | 99.9947% | 41.3055% | 58.4618% | 0.0004% | 58.6945% |
| baseline_all_0.5 | FAST_BLOCK | 83.5174% | 100.0000% | 43.0382% | 60.1772% | 0.0000% | 56.9618% |
| baseline_all_0.5 | INDICATION | 78.7522% | 99.9940% | 46.2968% | 63.2905% | 0.0018% | 53.7032% |
| stage1_only_0.6 | SLOW_BLOCK | 97.4343% | 99.9923% | 84.7802% | 91.7601% | 0.0013% | 15.2198% |
| stage1_only_0.6 | FAST_BLOCK | 95.6044% | 100.0000% | 84.8092% | 91.7803% | 0.0000% | 15.1908% |
| stage1_only_0.6 | INDICATION | 86.0773% | 99.9929% | 64.8135% | 78.6485% | 0.0030% | 35.1865% |
| stage2_only_0.6 | SLOW_BLOCK | 90.1092% | 99.9947% | 41.3055% | 58.4618% | 0.0004% | 58.6945% |
| stage2_only_0.6 | FAST_BLOCK | 83.5174% | 100.0000% | 43.0382% | 60.1772% | 0.0000% | 56.9618% |
| stage2_only_0.6 | INDICATION | 78.7522% | 99.9940% | 46.2968% | 63.2905% | 0.0018% | 53.7032% |
| stage3_only_0.6 | SLOW_BLOCK | 90.1092% | 99.9947% | 41.3055% | 58.4618% | 0.0004% | 58.6945% |
| stage3_only_0.6 | FAST_BLOCK | 83.5174% | 100.0000% | 43.0382% | 60.1772% | 0.0000% | 56.9618% |
| stage3_only_0.6 | INDICATION | 78.7522% | 99.9940% | 46.2968% | 63.2905% | 0.0018% | 53.7032% |
| stage4_only_0.6 | SLOW_BLOCK | 90.1092% | 99.9947% | 41.3055% | 58.4618% | 0.0004% | 58.6945% |
| stage4_only_0.6 | FAST_BLOCK | 83.5174% | 100.0000% | 43.0382% | 60.1772% | 0.0000% | 56.9618% |
| stage4_only_0.6 | INDICATION | 78.7522% | 99.9940% | 46.2968% | 63.2905% | 0.0018% | 53.7032% |
| stage5_only_0.6 | SLOW_BLOCK | 90.1092% | 99.9947% | 41.3055% | 58.4618% | 0.0004% | 58.6945% |
| stage5_only_0.6 | FAST_BLOCK | 83.5174% | 100.0000% | 43.0382% | 60.1772% | 0.0000% | 56.9618% |
| stage5_only_0.6 | INDICATION | 78.7522% | 99.9940% | 46.2968% | 63.2905% | 0.0018% | 53.7032% |
| stage6_only_0.6 | SLOW_BLOCK | 90.1092% | 99.9947% | 41.3055% | 58.4618% | 0.0004% | 58.6945% |
| stage6_only_0.6 | FAST_BLOCK | 83.5174% | 100.0000% | 43.0382% | 60.1772% | 0.0000% | 56.9618% |
| stage6_only_0.6 | INDICATION | 78.7522% | 99.9940% | 46.2968% | 63.2905% | 0.0018% | 53.7032% |
| all_0.6 | SLOW_BLOCK | 97.4343% | 99.9923% | 84.7802% | 91.7601% | 0.0013% | 15.2198% |
| all_0.6 | FAST_BLOCK | 95.6044% | 100.0000% | 84.8092% | 91.7803% | 0.0000% | 15.1908% |
| all_0.6 | INDICATION | 86.0773% | 99.9929% | 64.8135% | 78.6485% | 0.0030% | 35.1865% |

## Recall change versus baseline

| Configuration | SLOW_BLOCK | FAST_BLOCK | INDICATION |
|---|---:|---:|---:|
| baseline_all_0.5 | +0.0000% | +0.0000% | +0.0000% |
| stage1_only_0.6 | +43.4748% | +41.7711% | +18.5167% |
| stage2_only_0.6 | +0.0000% | +0.0000% | +0.0000% |
| stage3_only_0.6 | +0.0000% | +0.0000% | +0.0000% |
| stage4_only_0.6 | +0.0000% | +0.0000% | +0.0000% |
| stage5_only_0.6 | +0.0000% | +0.0000% | +0.0000% |
| stage6_only_0.6 | +0.0000% | +0.0000% | +0.0000% |
| all_0.6 | +43.4748% | +41.7711% | +18.5167% |

## Interpretation

A stage produces a meaningful difference only when its one-stage threshold configuration changes the aggregate metrics relative to `baseline_all_0.5`. The per-case metrics are stored in the JSON artifact for all 21 cases.

The `all_0.6` result is the previously observed global-threshold diagnostic configuration. Because configurations were compared on the validation labels, these results must not be reported as unbiased deployment performance.
