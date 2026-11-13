# Sequential Probability and Threshold Diagnostic

## Scope

This was a diagnostic experiment only. No model was retrained, and no source
model, architecture, checkpoint, dataset, timer, or synchronized SR logic was
modified.

The standalone runner was
[`run_threshold_analysis.py`](../../../scripts/training/run_threshold_analysis.py).
It used:

- `stage1_corrected_best.pt`;
- existing `stage2_best.pt` through `stage6_best.pt`;
- the current synchronized `VTSStagedModel` execution order;
- the unchanged 21-case validation split (`case_088`–`case_108`);
- 273,021 validation rows.

The complete machine-readable result is
[`threshold_analysis.json`](threshold_analysis.json).

## Probability capture

The existing evaluator calls `VTSStagedModel._predict()`, which calls
`predict_binary()` and discards sigmoid probabilities. The diagnostic runner
therefore reproduces the same staged equations in a separate script and calls
each stage's `forward()` method directly. It then applies an external
diagnostic threshold to the sigmoid probabilities.

Timers and synchronized SR state are updated in the same order as the current
model:

`Stage 1 -> 240 ms timer -> Stage 2 -> SR1 -> Stage 3 -> SR2 -> Stage 4 ->
tVTS timer -> 20 ms timer -> SR3 -> Stage 5 -> SR4 -> Stage 6`.

The raw sigmoid probabilities captured at threshold 0.5 are saved in
[`sigmoid_probabilities_threshold_0.5.npz`](sigmoid_probabilities_threshold_0.5.npz).
It contains one array for every stage output, each with 273,021 values.

## Earliest divergence and propagation

The first-divergence records vary slightly by threshold because binary
decisions change. At the baseline threshold 0.5, the earliest recurring
divergences were:

| Stage | Case | Row | Time | Signal | Expected | Actual | Probability |
|---|---|---:|---:|---|---:|---:|---:|
| Stage 1 | `case_097` | 1000 | 1.000 s | `1_AND_OUT` | 0 | 1 | 0.925533 |
| Stage 2 | `case_097` | 1000 | 1.000 s | `4_OR_OUT` | 1 | 0 | 0.000031 |
| Stage 3 | `case_098` | 1000 | 1.000 s | `7_AND_OUT` | 0 | 1 | 0.999071 |
| Stage 5 | `case_089` | 10999 | 10.999 s | `5_OR_OUT` | 0 | 1 | 0.999011 |
| Stage 6 | `case_089` | 10999 | 10.999 s | `INDICATION` | 0 | 1 | 0.999363 |

At threshold 0.5, aggregate stage mismatch counts were:

| Stage | Mismatches |
|---|---:|
| Stage 1 | 132,000 |
| Stage 2 | 132,000 |
| Stage 3 | 275,000 |
| Stage 4 | 0 |
| Stage 5 | 215,022 |
| Stage 6 | 130,016 |

The errors are therefore not isolated to Stage 6. Early Stage 1/2 decisions
and later Stage 3/5 decisions propagate into the stateful sequential result.
Stage 4 had no recorded mismatches in this diagnostic.

## Exploratory threshold sweep

A single global threshold was applied to every output of every neural stage.
Thresholds were run at `0.3`, `0.4`, `0.5`, `0.6`, and `0.7`.

These are exploratory validation diagnostics, not unbiased performance
estimates: thresholds were evaluated against the validation labels and were
not selected using a separate calibration set.

### Aggregate results

| Threshold | Output | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR | TP | TN | FP | FN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | `SLOW_BLOCK` | 90.1092% | 99.9947% | 41.3055% | 58.4618% | 99.9996% | 0.0004405% | 58.6945% | 19,003 | 227,014 | 1 | 27,003 |
| 0.3 | `FAST_BLOCK` | 83.5174% | 100.0000% | 43.0382% | 60.1772% | 100.0000% | 0.0000000% | 56.9618% | 34,001 | 194,019 | 0 | 45,001 |
| 0.3 | `INDICATION` | 78.7522% | 99.9940% | 46.2968% | 63.2905% | 99.9982% | 0.0018181% | 53.7032% | 50,008 | 165,002 | 3 | 58,008 |
| 0.4 | `SLOW_BLOCK` | 90.1092% | 99.9947% | 41.3055% | 58.4618% | 99.9996% | 0.0004405% | 58.6945% | 19,003 | 227,014 | 1 | 27,003 |
| 0.4 | `FAST_BLOCK` | 83.5174% | 100.0000% | 43.0382% | 60.1772% | 100.0000% | 0.0000000% | 56.9618% | 34,001 | 194,019 | 0 | 45,001 |
| 0.4 | `INDICATION` | 78.7522% | 99.9940% | 46.2968% | 63.2905% | 99.9982% | 0.0018181% | 53.7032% | 50,008 | 165,002 | 3 | 58,008 |
| 0.5 | `SLOW_BLOCK` | 90.1092% | 99.9947% | 41.3055% | 58.4618% | 99.9996% | 0.0004405% | 58.6945% | 19,003 | 227,014 | 1 | 27,003 |
| 0.5 | `FAST_BLOCK` | 83.5174% | 100.0000% | 43.0382% | 60.1772% | 100.0000% | 0.0000000% | 56.9618% | 34,001 | 194,019 | 0 | 45,001 |
| 0.5 | `INDICATION` | 78.7522% | 99.9940% | 46.2968% | 63.2905% | 99.9982% | 0.0018181% | 53.7032% | 50,008 | 165,002 | 3 | 58,008 |
| 0.6 | `SLOW_BLOCK` | 97.4343% | 100.0000% | 84.7802% | 91.7633% | 99.9987% | 0.0013211% | 15.2198% | 39,004 | 227,012 | 3 | 7,002 |
| 0.6 | `FAST_BLOCK` | 95.6044% | 100.0000% | 84.8092% | 91.7803% | 100.0000% | 0.0000000% | 15.1908% | 67,001 | 194,019 | 0 | 12,001 |
| 0.6 | `INDICATION` | 86.0773% | 99.9929% | 64.8135% | 78.6508% | 99.9970% | 0.0030300% | 35.1865% | 70,009 | 165,000 | 5 | 38,007 |
| 0.7 | `SLOW_BLOCK` | 97.4343% | 100.0000% | 84.7802% | 91.7633% | 99.9987% | 0.0013211% | 15.2198% | 39,004 | 227,012 | 3 | 7,002 |
| 0.7 | `FAST_BLOCK` | 95.6044% | 100.0000% | 84.8092% | 91.7803% | 100.0000% | 0.0000000% | 15.1908% | 67,001 | 194,019 | 0 | 12,001 |
| 0.7 | `INDICATION` | 86.0773% | 99.9929% | 64.8135% | 78.6508% | 99.9970% | 0.0030300% | 35.1865% | 70,009 | 165,000 | 5 | 38,007 |

Thresholds 0.3, 0.4, and 0.5 produced identical binary sequential behavior.
Thresholds 0.6 and 0.7 also produced identical behavior.

## Per-case results

The table gives per-case accuracy as `threshold 0.5 / threshold 0.6` for
`SLOW_BLOCK`, `FAST_BLOCK`, and `INDICATION`, respectively. Thresholds 0.3 and
0.4 match 0.5; threshold 0.7 matches 0.6.

| Case | `SLOW_BLOCK` | `FAST_BLOCK` | `INDICATION` |
|---|---:|---:|---:|
| `case_088` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `case_089` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.9999 / 0.9999 |
| `case_090` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `case_091` | 0.9999 / 0.9999 | 1.0000 / 1.0000 | 0.9999 / 0.9999 |
| `case_092` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `case_093` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `case_094` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.9999 / 0.9999 |
| `case_095` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `case_096` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `case_097` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `case_098` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `case_099` | 0.3847 / 0.9999 | 0.1539 / 1.0000 | 0.3847 / 0.9999 |
| `case_100` | 0.4614 / 0.4614 | 0.0769 / 0.0769 | 0.4614 / 0.4614 |
| `case_101` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.8461 / 0.8461 |
| `case_102` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.2308 / 0.2308 |
| `case_103` | 0.3077 / 0.9999 | 0.1539 / 1.0000 | 0.3077 / 0.9999 |
| `case_104` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.5384 / 0.5384 |
| `case_105` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `case_106` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.3077 / 0.3077 |
| `case_107` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.6923 / 0.6923 |
| `case_108` | 0.7692 / 1.0000 | 0.1539 / 1.0000 | 0.7692 / 1.0000 |

## Interpretation

The 0.6/0.7 exploratory global threshold materially improves recall:

- `SLOW_BLOCK`: 41.3055% to 84.7802%;
- `FAST_BLOCK`: 43.0382% to 84.8092%;
- `INDICATION`: 46.2968% to 64.8135%.

The trade-off is a small increase in false positives:

- `SLOW_BLOCK`: FP 1 to 3;
- `FAST_BLOCK`: FP remains 0;
- `INDICATION`: FP 3 to 5.

Because the sweep used validation labels to show these metrics, the 0.6/0.7
result is **not an unbiased validation estimate** and must not be treated as a
deployment threshold. It is evidence that the saved probabilities contain
useful ranking information and that the 0.5 decision boundary is suppressing
many positive events.

Thresholding is global here. A later controlled experiment should calibrate
stage-specific thresholds using a separate calibration split or training
data, then evaluate once on untouched validation data. It should also inspect
transition timing and state traces, not only aggregate confusion matrices.

## Conclusion

The diagnostic found a promising recall lever without retraining: thresholds
0.6 and 0.7 produced identical exploratory results and substantially reduced
false negatives. However, the result requires independent calibration and
fresh validation before any claim of improved model performance or protection
IED suitability.
