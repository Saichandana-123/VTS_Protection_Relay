# Targeted Class-Weighted BCE Experiment Report

## 1. Objective

This controlled experiment tested whether moderate per-output positive-class
weights could improve recall in the integrated six-stage VTS model.

The experiment preserved:

- the original dataset and case-wise split;
- the existing six neural-network architectures;
- the synchronized `VTSStagedModel`;
- the existing Stage 2 and Stage 4 checkpoints;
- the 21-case sequential validation procedure.

No original checkpoint, dataset, timer logic, SR logic, or training script was
overwritten.

## 2. Experiment implementation

The standalone runner is
[`run_class_weighted_experiment.py`](scripts/training/run_class_weighted_experiment.py).
Experimental files were written under:

`models/experimental/class_weighted/`

The experiment trained only Stage 1, Stage 3, Stage 5, and Stage 6. Stage 2
and Stage 4 used their existing checkpoints because they were not selected for
targeted weighting.

Training retained the original:

- training cases `case_001`–`case_087`;
- validation cases `case_088`–`case_108`;
- 273,021 validation rows;
- architecture definitions;
- batch size 4096;
- Adam optimizer;
- learning rate `1e-3`;
- maximum 12 epochs;
- early stopping patience of 3 epochs.

The loss was `BCEWithLogitsLoss(pos_weight=...)`. The selected moderate
weights were:

| Stage | Output weights |
|---|---|
| Stage 1 | `1_OR_OUT=1.0`, `2_OR_OUT=1.0`, `3_OR_OUT=1.0`, `1_AND_OUT=1.5`, `2_AND_OUT=1.0` |
| Stage 3 | `6_AND_OUT=2.0`, remaining outputs `1.0` |
| Stage 5 | `5_OR_OUT=1.0`, `6_OR_OUT=1.5`, `7_OR_OUT=2.0` |
| Stage 6 | `SLOW_BLOCK=2.0`, `FAST_BLOCK=2.0`, `INDICATION=2.0` |

The complete saved experiment result is
[`class_weighted_experiment.json`](models/experimental/class_weighted/class_weighted_experiment.json).

## 3. Stage-wise validation results

These are unweighted validation accuracies measured after training the
experimental checkpoints. They are not sufficient by themselves to establish
integrated improvement.

| Stage | Experimental validation accuracy | Experimental unweighted validation loss | Epochs |
|---|---:|---:|---:|
| Stage 1 | 91.9420% | 0.112652681 | 6 |
| Stage 2 | 100.0000% | Existing checkpoint; not retrained | Existing checkpoint |
| Stage 3 | 100.0000% | 0.0008950228 | 12 |
| Stage 4 | 100.0000% | Existing checkpoint; not retrained | Existing checkpoint |
| Stage 5 | 100.0000% | 0.0006088106 | 12 |
| Stage 6 | 100.0000% | 0.0008604861 | 12 |

For comparison, the original corrected Stage 1 checkpoint had saved
validation accuracy of approximately 90.3304%. The weighted Stage 1 result
therefore improved its aggregate stage-wise validation accuracy, but this did
not translate into better sequential final outputs.

## 4. Sequential integrated evaluation

The experimental checkpoint set was evaluated through the current synchronized
`VTSStagedModel` on all 21 validation cases. The final metrics are:

| Output | Accuracy | Precision | Recall | F1 | FPR | FNR | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SLOW_BLOCK` | 90.1092% | 99.9947% | 41.3055% | 58.4618% | 0.0004405% | 58.6945% | 19,003 | 227,014 | 1 | 27,003 |
| `FAST_BLOCK` | 83.5174% | 100.0000% | 43.0382% | 60.1772% | 0.0000000% | 56.9618% | 34,001 | 194,019 | 0 | 45,001 |
| `INDICATION` | 78.7522% | 99.9940% | 46.2968% | 63.2905% | 0.0018181% | 53.7032% | 50,008 | 165,002 | 3 | 58,008 |

Specificity is implied by the saved confusion matrices:

- `SLOW_BLOCK`: 99.9995595%
- `FAST_BLOCK`: 100.0000000%
- `INDICATION`: 99.9981819%

### Confusion matrices

#### `SLOW_BLOCK`

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Expected 0 | TN = 227,014 | FP = 1 |
| Expected 1 | FN = 27,003 | TP = 19,003 |

#### `FAST_BLOCK`

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Expected 0 | TN = 194,019 | FP = 0 |
| Expected 1 | FN = 45,001 | TP = 34,001 |

#### `INDICATION`

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Expected 0 | TN = 165,002 | FP = 3 |
| Expected 1 | FN = 58,008 | TP = 50,008 |

## 5. Comparison with post-synchronization baseline

The baseline is
[`integrated_evaluation_post_sync.json`](models/trained/integrated_evaluation_post_sync.json).

| Output | Baseline recall | Experimental recall | Recall change | Baseline FPR | Experimental FPR | Result |
|---|---:|---:|---:|---:|---:|---|
| `SLOW_BLOCK` | 41.3055% | 41.3055% | 0.0000 pp | 0.0004405% | 0.0004405% | No improvement |
| `FAST_BLOCK` | 43.0382% | 43.0382% | 0.0000 pp | 0.0000000% | 0.0000000% | No improvement |
| `INDICATION` | 46.2968% | 46.2968% | 0.0000 pp | 0.0018181% | 0.0018181% | No improvement |

The experimental confusion matrices and mismatch counts are identical to the
post-synchronization baseline:

- `SLOW_BLOCK`: 27,004 mismatches;
- `FAST_BLOCK`: 45,001 mismatches;
- `INDICATION`: 58,011 mismatches.

This means the weighted checkpoints did not change the thresholded sequential
behavior on the evaluated validation cases.

## 6. Per-case performance

The following table reports per-case accuracy for each final output. Each case
contains 13,001 rows.

| Case | `SLOW_BLOCK` accuracy | `FAST_BLOCK` accuracy | `INDICATION` accuracy |
|---|---:|---:|---:|
| `case_088` | 100.0000% | 100.0000% | 100.0000% |
| `case_089` | 100.0000% | 100.0000% | 99.9923% |
| `case_090` | 100.0000% | 100.0000% | 100.0000% |
| `case_091` | 99.9923% | 100.0000% | 99.9923% |
| `case_092` | 100.0000% | 100.0000% | 100.0000% |
| `case_093` | 100.0000% | 100.0000% | 100.0000% |
| `case_094` | 100.0000% | 100.0000% | 99.9923% |
| `case_095` | 100.0000% | 100.0000% | 100.0000% |
| `case_096` | 100.0000% | 100.0000% | 100.0000% |
| `case_097` | 100.0000% | 100.0000% | 100.0000% |
| `case_098` | 100.0000% | 100.0000% | 100.0000% |
| `case_099` | 38.4663% | 15.3911% | 38.4663% |
| `case_100` | 46.1426% | 7.6917% | 46.1426% |
| `case_101` | 100.0000% | 100.0000% | 84.6089% |
| `case_102` | 100.0000% | 100.0000% | 23.0751% |
| `case_103` | 30.7746% | 15.3911% | 30.7746% |
| `case_104` | 100.0000% | 100.0000% | 53.8420% |
| `case_105` | 100.0000% | 100.0000% | 100.0000% |
| `case_106` | 100.0000% | 100.0000% | 30.7669% |
| `case_107` | 100.0000% | 100.0000% | 69.2254% |
| `case_108` | 76.9172% | 15.3911% | 76.9172% |

The same problematic cases remain problematic, especially cases 099, 100,
103, and 108. This supports the conclusion that the issue is not resolved by
moderately increasing positive loss weights.

## 7. Interpretation and trade-offs

The experiment produced an apparently better Stage 1 aggregate validation
accuracy, but no sequential recall improvement. There was also no increase in
false positives because the binary sequential outputs remained unchanged.

The likely explanation is that class weighting changed logits or confidence
margins without changing the final 0.5 decisions on the relevant sequential
inputs. It may also indicate that the dominant errors are caused by the
training/inference distribution and upstream signal contract rather than by
ordinary positive-class underweighting.

The result does **not** justify claiming that class-weighted BCE improves the
integrated model.

## 8. Conclusion

**Decision: reject this class-weighted configuration as an integrated recall
improvement.**

The controlled experiment was completed correctly, but:

- `SLOW_BLOCK` recall remained **41.3055%**;
- `FAST_BLOCK` recall remained **43.0382%**;
- `INDICATION` recall remained **46.2968%**;
- all final confusion matrices remained unchanged;
- the large false-negative problem remains.

No further class-weighted training should be started with the same setup
without first addressing the Stage 1 contract/distribution-shift issue or
performing probability/threshold diagnostics. A useful next experiment would
be a raw-logit sequential threshold sweep, followed by a corrected-contract
diagnostic that records the first divergence in every stage and stateful
component.

## 9. Preserved artifacts

Original checkpoints and reports were preserved. New experimental artifacts:

- [`run_class_weighted_experiment.py`](scripts/training/run_class_weighted_experiment.py)
- [`class_weighted_experiment.json`](models/experimental/class_weighted/class_weighted_experiment.json)
- `stage1_class_weighted_best.pt`
- `stage3_class_weighted_best.pt`
- `stage5_class_weighted_best.pt`
- `stage6_class_weighted_best.pt`
