# BTP Experimental Results and Evaluation

## 1. Executive summary

This report documents the saved training, diagnostic, and integrated-evaluation
results for the virtual replication of the P642 Voltage Transformer
Supervision (VTS) logic. The Simulink model and its generated CSV outputs are
the behavioral reference.

The project has a functioning six-stage hybrid implementation, but the saved
end-to-end sequential evaluation is **not yet suitable as a protection-IED
replication result**. The final-output accuracies on the held-out validation
cases are:

| Output | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `SLOW_BLOCK` | 90.1092% | 99.9947% | 41.3055% | 58.4618% |
| `FAST_BLOCK` | 83.5174% | 100.0000% | 43.0382% | 60.1772% |
| `INDICATION` | 78.7522% | 99.9940% | 46.2968% | 63.2905% |

The dominant error is false negatives: the model usually avoids false
assertions, but misses a substantial proportion of positive protection events.
This is not acceptable for claiming faithful protection behavior.

Teacher-forced analysis is substantially better for later stages, but it is a
diagnostic experiment in which reference intermediate signals replace predicted
upstream signals. It must not be presented as deployment or integrated
performance.

## 2. Source artifacts and evaluation scope

The report was prepared from the following saved artifacts:

- [`training_metrics.json`](models/trained/training_metrics.json)
- [`stage1_corrected_metrics.json`](models/trained/stage1_corrected_metrics.json)
- [`integrated_evaluation.json`](models/trained/integrated_evaluation.json)
- [`integrated_evaluation_corrected_stage1.json`](models/trained/integrated_evaluation_corrected_stage1.json)
- [`integrated_evaluation_post_sync.json`](models/trained/integrated_evaluation_post_sync.json)
- [`integrated_root_cause_analysis.json`](models/trained/integrated_root_cause_analysis.json)
- [`teacher_forced_analysis.json`](models/trained/teacher_forced_analysis.json)
- [`evaluate_integrated_model.py`](scripts/training/evaluate_integrated_model.py)
- [`train_six_stage_network.py`](scripts/training/train_six_stage_network.py)
- [`decisions.md`](decisions.md)

The corrected-stage-1 integrated report uses the same validation split and
downstream checkpoints as the baseline, with `stage1_corrected_best.pt`
substituted for `stage1_best.pt`.

## 3. Dataset and split

### 3.1 Dataset contract

The saved project decisions record a 62-column contract consisting of:

- 23 input columns;
- three final outputs;
- two metadata columns; and
- 34 logged Simulink intermediate signals.

The generated cases use a global 1 ms timestep from 0 through 13 seconds,
giving 13,001 rows per complete case. Case 089 was repaired as a single-file
artifact; the saved decisions state that the other cases were not regenerated.

### 3.2 Case-wise split

The training run used a case-wise split to avoid temporal leakage:

- **Training:** 87 cases, `case_001` through `case_087`.
- **Validation:** 21 cases, `case_088` through `case_108`.
- **Validation samples:** 273,021 rows (21 x 13,001).
- **Training seed:** `20261104`.
- **Recorded device:** CPU.

Validation case IDs:

`case_088`, `case_089`, `case_090`, `case_091`, `case_092`,
`case_093`, `case_094`, `case_095`, `case_096`, `case_097`,
`case_098`, `case_099`, `case_100`, `case_101`, `case_102`,
`case_103`, `case_104`, `case_105`, `case_106`, `case_107`,
`case_108`.

## 4. Training methodology

Each stage was trained separately as a binary-output MLP using:

- `BCEWithLogitsLoss`;
- Adam optimizer;
- learning rate `1e-3`;
- batch size `4096`;
- maximum of 12 epochs;
- early stopping patience of three epochs;
- the fixed seed `20261104`.

The best checkpoint was selected by lowest validation loss. The training
script records aggregate accuracy over all output values in each stage, not
just final-output accuracy.

## 5. Six-stage hybrid architecture

The architecture preserves the relay's staged organization:

1. **Stage 1:** detector/control inputs to `1_*`, `2_*`, and `3_*` signals.
2. **Stage 2:** voltage/current, 240 ms timer, and Stage 1 signals to
   `3_AND`, `4_AND`, `4_OR`, and `5_AND`.
3. **Stage 3:** pole-dead, Stage 1/2, SR1, and voltage-dependent-function
   inputs to `6_AND` through `9_AND`.
4. **Stage 4:** SR2, `6_AND`, 240 ms timer, and `2_AND` to `10_AND`.
5. **Stage 5:** SR3, MCB, `8_AND`, tVTS timer, and 20 ms timer to
   `5_OR` through `7_OR`.
6. **Stage 6:** `5_OR`, `6_OR`, blocking, MCB, and SR4 to
   `SLOW_BLOCK`, `FAST_BLOCK`, and `INDICATION`.

Timers and SR flip-flops remain deterministic stateful components outside the
MLPs. The recorded execution order is:

`stage1 -> timer240 -> stage2 -> SR1 -> stage3 -> SR2 -> stage4 ->
tVTS_timer -> timer20 -> SR3 -> stage5 -> SR4 -> stage6`.

## 6. Stage-wise training results

The values below are the saved best aggregate metrics from
`training_metrics.json`. “Accuracy” is over all target values in the stage.

| Stage | Targets | Epochs | Best train accuracy | Best validation accuracy | Best train loss | Best validation loss |
|---|---|---:|---:|---:|---:|---:|
| Stage 1 | `1_OR_OUT`, `2_OR_OUT`, `3_OR_OUT`, `1_AND_OUT`, `2_AND_OUT` | 7 | 100.0000% | 90.3304% | 0.000508186 | 0.181454441 |
| Stage 2 | `3_AND_OUT`, `4_AND_OUT`, `4_OR_OUT`, `5_AND_OUT` | 12 | 98.7844% | 100.0000% | 0.020110024 | 0.000202057 |
| Stage 3 | `6_AND_OUT`, `7_AND_OUT`, `8_AND_OUT`, `9_AND_OUT` | 12 | 99.9988% | 100.0000% | 0.000730953 | 0.001192244 |
| Stage 4 | `10_AND_OUT` | 12 | 100.0000% | 100.0000% | 0.000003110 | 0.000001202 |
| Stage 5 | `5_OR_OUT`, `6_OR_OUT`, `7_OR_OUT` | 12 | 100.0000% | 100.0000% | 0.000607369 | 0.000664729 |
| Stage 6 | `SLOW_BLOCK`, `FAST_BLOCK`, `INDICATION` | 12 | 100.0000% | 100.0000% | 0.000531769 | 0.000764229 |

These results demonstrate that the individual supervised mappings fit the
saved validation inputs well, except for the lower Stage 1 aggregate
validation accuracy. They do **not** establish that the stateful sequential
system reproduces the Simulink outputs.

## 7. Integrated sequential evaluation

The following is the deployment-relevant saved evaluation. It processes each
validation case row-by-row through the complete staged model, including
predicted intermediate signals, timers, and SR state.

### 7.1 Metrics

Specificity, false-positive rate (FPR), and false-negative rate (FNR) were
calculated directly from the saved confusion-matrix counts:

- Specificity = `TN / (TN + FP)`
- FPR = `FP / (FP + TN)`
- FNR = `FN / (FN + TP)`

| Output | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SLOW_BLOCK` | 90.1092% | 99.9947% | 41.3055% | 58.4618% | 99.9996% | 0.0004% | 58.6945% | 19,003 | 227,014 | 1 | 27,003 |
| `FAST_BLOCK` | 83.5174% | 100.0000% | 43.0382% | 60.1772% | 100.0000% | 0.0000% | 56.9618% | 34,001 | 194,019 | 0 | 45,001 |
| `INDICATION` | 78.7522% | 99.9940% | 46.2968% | 63.2905% | 99.9982% | 0.0018% | 53.7032% | 50,008 | 165,002 | 3 | 58,008 |

The percentages in this table are rounded for readability; the counts are the
saved integer counts and are the basis for the derived rates.

### 7.2 Confusion matrices

Rows are expected/reference labels and columns are predicted labels.

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

## 8. Original versus corrected Stage 1 checkpoint

The corrected Stage 1 work addressed the blocking-polarity/input-contract
issue without changing the dataset or the six-stage architecture.

| Item | Original | Corrected |
|---|---:|---:|
| Checkpoint | `stage1_best.pt` | `stage1_corrected_best.pt` |
| Best validation accuracy | 90.3304141% | 90.3304160% |
| Best validation loss | 0.181454441 | 0.175934970 |
| Integrated `SLOW_BLOCK` accuracy | 90.1091857% | 90.1091857% |
| Integrated `FAST_BLOCK` accuracy | 83.5173851% | 83.5173851% |
| Integrated `INDICATION` accuracy | 78.7521839% | 78.7521839% |
| Integrated mismatch counts | 27,004 / 45,001 / 58,011 | 27,004 / 45,001 / 58,011 |

The corrected checkpoint therefore improved the recorded Stage 1 validation
loss but did not improve the aggregate end-to-end final-output metrics. The
original and corrected integrated reports agree on the final counts.

## 9. Teacher-forced versus sequential inference

### 9.1 Definition

The teacher-forced analysis gives each stage its reference CSV intermediate
inputs rather than passing predicted upstream outputs into the next stage.
Sequential inference uses `VTSStagedModel.step()` and therefore includes
predicted intermediates, timer state, and SR state.

### 9.2 Final-output comparison

| Output | Teacher-forced accuracy | Teacher-forced recall | Sequential accuracy | Sequential recall |
|---|---:|---:|---:|---:|
| `SLOW_BLOCK` | 100.0000% | 100.0000% | 90.1092% | 41.3055% |
| `FAST_BLOCK` | 93.4078% | 77.2183% | 83.5174% | 43.0382% |
| `INDICATION` | 100.0000% | 100.0000% | 78.7522% | 46.2968% |

Teacher-forced `FAST_BLOCK` is not perfect: its saved counts are TP 61,004,
TN 194,019, FP 0, FN 17,998. The teacher-forced `SLOW_BLOCK` and `INDICATION`
outputs have zero false negatives and zero false positives in the saved
analysis.

These teacher-forced values isolate stage mapping behavior under reference
inputs. They are **not** integrated deployment results and do not demonstrate
that the complete model can operate correctly without reference intermediate
signals.

### 9.3 Sequential stage mismatch counts

The root-cause analysis records the following sequential intermediate mismatch
counts across the 273,021 validation rows:

| Stage | Mismatch count |
|---|---:|
| Stage 1 | 121,000 |
| Stage 2 | 121,000 |
| Stage 3 | 264,000 |
| Stage 4 | 0 |
| Stage 5 | 215,022 |
| Stage 6 | 130,016 |

## 10. Root-cause analysis and limitations

The saved root-cause analysis identifies the following issues:

1. **Stage 1 blocking-polarity mismatch:** the mapping difference affected all
   273,021 validation rows. The analysis identified a mismatch between the
   training/inference contract and the `VTS_BLOCKING`/complement semantics.
2. **Error propagation:** sequential inference consumes predicted intermediate
   values, so an early Stage 1 error changes Stage 2, Stage 3, Stage 5, and
   final outputs.
3. **Stateful behavior remains coupled to upstream errors:** timer and SR
   transitions can diverge after upstream predictions diverge. The saved
   analysis explicitly treats this as downstream evidence, not proof that the
   timer implementation itself is wrong.
4. **Large false-negative burden:** the sequential confusion matrices contain
   27,003, 45,001, and 58,008 false negatives for the three outputs.
5. **Class imbalance:** several intermediate targets have zero positive support
   in this validation split. Their 100% accuracy is therefore not evidence of
   positive-event detection.
6. **Limited scenario coverage:** the evaluation is a case-wise holdout over
   21 generated cases. It is not a hardware-in-the-loop, field, noise,
   parameter-variation, or unseen-operating-condition certification.
7. **Reference dependency:** detector-derived Stage 1 signals are taken from
   logged Simulink signals, so the current experiment does not independently
   validate the analog detector implementation.

## 11. Post-synchronization integrated evaluation

The final verification run used the current synchronized
`VTSStagedModel`, `stage1_corrected_best.pt`, and the existing Stage 2-6
checkpoints. It evaluated the same 21 validation cases
(`case_088` through `case_108`) and 273,021 rows. No datasets, thresholds,
architectures, or checkpoints were changed.

### 11.1 Post-synchronization metrics

| Output | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SLOW_BLOCK` | 90.1092% | 99.9947% | 41.3055% | 58.4618% | 99.9996% | 0.0004% | 58.6945% | 19,003 | 227,014 | 1 | 27,003 |
| `FAST_BLOCK` | 83.5174% | 100.0000% | 43.0382% | 60.1772% | 100.0000% | 0.0000% | 56.9618% | 34,001 | 194,019 | 0 | 45,001 |
| `INDICATION` | 78.7522% | 99.9940% | 46.2968% | 63.2905% | 99.9982% | 0.0018% | 53.7032% | 50,008 | 165,002 | 3 | 58,008 |

### 11.2 Comparison with previous evaluations

| Output | Previous baseline accuracy | Corrected Stage 1 accuracy | Post-sync accuracy | Post-sync mismatch count |
|---|---:|---:|---:|---:|
| `SLOW_BLOCK` | 90.1092% | 90.1092% | 90.1092% | 27,004 |
| `FAST_BLOCK` | 83.5174% | 83.5174% | 83.5174% | 45,001 |
| `INDICATION` | 78.7522% | 78.7522% | 78.7522% | 58,011 |

The post-synchronization confusion matrices are identical to the previous
corrected Stage 1 evaluation. Therefore, the synchronization change did not
alter these aggregate final outputs on the saved validation cases. This is a
verification result, not evidence that the sequential model meets a
protection-grade acceptance target.

### 11.3 SR flip-flop synchronization implementation and tests

The SR implementation was updated after the saved model evaluation:

- [`flip_flop.py`](src/timing/flip_flop.py) now provides non-mutating
  `next_state()` and retains reset priority when Set and Reset are both true.
- [`vts_staged_model.py`](src/models/vts_staged_model.py) captures all four
  `(S, R)` pairs before committing SR1, SR2, SR3, or SR4.
- Existing Boolean expressions, signal connections, timer logic, neural
  architectures, checkpoints, and datasets were not changed.
- The downstream stages continue to receive the same logical next-state
  values; state mutation is deferred to the synchronized commit boundary.

Focused tests were added in [`test_timing.py`](tests/test_timing.py) and
[`test_vts_staged_model.py`](tests/test_vts_staged_model.py). The verified
results were:

- Full project test suite: **8 passed**.
- Source/test compilation with `python -m compileall -q src tests`:
  **passed**.
- The integration test demonstrates that all four inputs are captured before
  any SR update, all four states commit to the expected values, and no
  sequential-update dependency is present.
- The SR unit test demonstrates preserved reset priority for simultaneous Set
  and Reset.

This synchronization change is test-verified for the coded discrete
simulation order. The post-synchronization integrated evaluation is saved
separately in `integrated_evaluation_post_sync.json`.

## 12. Interpretation for a protection IED application

The current results show a useful engineering prototype, not a protection-grade
replica. Very high precision and specificity mean the model rarely asserts a
block or indication when the reference is inactive. However, protection logic
must also detect true events reliably. Recall below 47% for all three final
outputs means many reference-positive timesteps are missed in sequential
operation.

The false-negative pattern is especially important: a model that is
conservative by suppressing true block or indication events can appear strong
on accuracy while failing the operational safety objective. The 100% or near
100% stage-wise and teacher-forced scores cannot compensate for the sequential
false-negative behavior.

Accordingly, the current artifact should be treated as an experimental
replication framework and diagnostic baseline. It should not be used as a
replacement for the reference IED logic, protection validation, or safety
certification.

## 13. Conclusion and realistic future improvements

### Conclusion

The project has completed the dataset preparation, staged model training,
deterministic timer/SR integration, saved checkpoint generation, teacher-forced
diagnostics, sequential evaluation, root-cause analysis, and SR synchronization
tests. It has **not** completed faithful end-to-end replication: the saved
sequential validation results remain materially below a credible protection
logic target, primarily because of false negatives and propagated intermediate
state errors.

### Recommended next steps

1. Resolve and independently test the Stage 1 input-polarity contract against
   the Simulink signal connections.
2. Compare every intermediate signal and every state transition at the exact
   first-error timestep, rather than relying only on final-output accuracy.
3. Add balanced positive-event metrics and scenario-level pass/fail criteria;
   do not rely on aggregate accuracy for rare outputs.
4. Expand validation with unseen settings, timing modes, noise, waveform
   variation, and boundary cases.
5. Establish explicit acceptance thresholds for recall, FNR, transition timing,
   and per-case behavior before considering deployment.
6. Only after the deterministic and signal-contract issues are resolved,
   evaluate whether retraining or architecture changes are justified.

## 14. Metrics not recoverable from the saved files

The saved artifacts provide the requested final-output confusion matrices and
allow specificity, FPR, and FNR to be derived exactly. The following were not
available as complete saved metrics in the inspected artifacts:

- per-case confusion matrices for every intermediate stage in one consolidated
  report;
- confidence intervals or statistical significance estimates;
- latency, throughput, memory, or real-time execution measurements;
- hardware-in-the-loop or physical-IED comparison results;
- independent field/noise/generalization results;
- a complete post-fix re-evaluation of timer and SR transition equivalence.

No values above were estimated to fill these gaps. Derived rates were computed
only from saved confusion-matrix counts.
# Experimental Attempts and Performance Improvement

This section documents the experiments conducted to investigate prediction errors and improve the performance of the six-stage neural network-based VTS protection relay model. Not all experiments resulted in improved integrated performance; however, each experiment was used to investigate a specific aspect of the model.

## Experiment 1: Baseline Integrated Evaluation

**Objective:** Establish the end-to-end performance of the integrated six-stage model.

**Methodology:**
- Evaluated the six-stage model using the existing trained checkpoints.
- Used 21 held-out validation cases (case_088 to case_108).
- Evaluated 273,021 time-series samples.
- Preserved sequential execution of neural network stages, deterministic timers and SR flip-flops.

**Results:**

| Output | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| SLOW_BLOCK | 90.11% | 99.99% | 41.31% | 58.46% |
| FAST_BLOCK | 83.52% | 100.00% | 43.04% | 60.18% |
| INDICATION | 78.75% | 99.99% | 46.30% | 63.29% |

**Observation:**
The model achieved high precision but relatively low recall. Most prediction errors were false negatives, indicating that several expected active outputs were not detected.

**Conclusion:**
The baseline evaluation established the need to investigate intermediate-stage prediction errors and their propagation through the integrated model.

## Experiment 2: Stage 1 Polarity Correction

**Objective:** Investigate a possible polarity mismatch in the Stage 1 VTS_BLOCKING signal handling.

**Methodology:**
- Examined the handling of the raw VTS_BLOCKING input.
- Ensured that inversion was applied only within the specific 2_AND gate branch, while retaining the raw signal for other logic.
- Trained a separate corrected Stage 1 checkpoint.
- Preserved the original Stage 1 checkpoint for comparison.

**Results:**
- Corrected Stage 1 validation accuracy: 90.33%.
- Original and corrected checkpoints produced identical binary predictions across 273,021 validation samples.
- The integrated final-output metrics remained unchanged.

**Conclusion:**
The correction clarified the intended signal-handling implementation but did not produce an improvement in end-to-end prediction performance.

## Experiment 3: Synchronized SR Flip-Flop Updates

**Objective:** Ensure that the four SR flip-flops use synchronized same-timestep inputs rather than sequentially influencing one another during state updates.

**Methodology:**
- Implemented a two-phase state-update approach.
- Captured all four flip-flop set/reset input pairs before committing state changes.
- Computed the next states and committed them together.
- Updated the relevant unit and integration tests.

**Verification:**
- All 8 project tests passed.
- Existing trained checkpoints and dataset were preserved.
- Integrated evaluation was repeated using the synchronized implementation.

**Results:**
The final metrics were identical to the baseline evaluation.

**Conclusion:**
The synchronized update mechanism was implemented and tested successfully, but it did not improve the measured integrated performance.

## Experiment 4: Class-Weighted Binary Cross-Entropy Training

**Objective:** Investigate whether class imbalance in intermediate outputs was contributing to low recall.

**Methodology:**
- Implemented a separate experimental training script.
- Applied moderate per-output positive class weights derived from training-set class distributions.
- Retrained only the selected stages using the existing architecture and training/validation split.
- Saved experimental checkpoints separately under `models/experimental/class_weighted/`.
- Evaluated the resulting mixed experimental checkpoint set through the complete sequential model.

**Results:**
- Stage 1 standalone validation accuracy increased to 91.94%.
- Final integrated recall remained unchanged.

| Output | Baseline Recall | Experimental Recall |
|---|---:|---:|
| SLOW_BLOCK | 41.31% | 41.31% |
| FAST_BLOCK | 43.04% | 43.04% |
| INDICATION | 46.30% | 46.30% |

- All final confusion matrices and per-case results remained unchanged.
- All 8 project tests passed.
- Original checkpoints were preserved.

**Conclusion:**
Class weighting improved the standalone validation accuracy of Stage 1 but did not improve end-to-end performance. This indicates that improving individual-stage accuracy alone is insufficient to guarantee better sequential model performance.

## Overall Experimental Findings

The experiments highlighted the following observations:

1. The integrated model exhibits high precision but relatively low recall.
2. False negatives account for most final-output prediction errors.
3. Errors in intermediate stages can propagate through subsequent stages.
4. Improvements in standalone stage metrics do not necessarily translate into improvements in the integrated model.
5. Correcting signal handling and synchronizing flip-flop updates did not change the final evaluation metrics.
6. Class-weighted training did not improve final-output recall.

**Overall conclusion:** The experiments provided insight into the limitations of the current six-stage model and helped narrow down the possible causes of its performance limitations. Further investigation is required before claiming an improvement in end-to-end performance.

## Planned Investigation

A sequential probability and threshold diagnostic is planned to:
- Examine intermediate-stage prediction probabilities.
- Identify the earliest point of divergence between predicted and reference signals.
- Investigate whether threshold selection affects error propagation.
- Evaluate the complete sequential model rather than relying only on standalone stage accuracy.

This investigation is pending and its results will be documented separately.
## Experiment 5: Sequential Probability Analysis and Threshold Exploration

**Objective:**

Investigate whether the low recall of the integrated six-stage model was related to the binary classification threshold and identify the earliest stage responsible for prediction divergence.

### Methodology

- Examined the six-stage model execution and existing evaluation pipeline.
- Captured sigmoid probabilities from the stage outputs before binary thresholding.
- Used the corrected Stage 1 checkpoint and existing trained checkpoints for the remaining stages.
- Preserved the sequential execution order, deterministic timers, and synchronized SR flip-flop updates.
- Evaluated five global classification thresholds: 0.3, 0.4, 0.5, 0.6, and 0.7.
- Evaluated all 21 held-out validation cases, comprising 273,021 samples.
- No model retraining, dataset modification, architecture modification, or checkpoint replacement was performed.

### First-Divergence Analysis

At the baseline threshold of 0.5, the earliest recurring divergence was identified at:

- **Stage:** Stage 1
- **Signal:** `1_AND_OUT`
- **Case:** `case_097`
- **Time:** Approximately 1.000 second

The divergence propagated through subsequent stages:

`Stage 1 → Stage 2 → Stage 3 → Stage 5 → Stage 6`

Stage 4 recorded zero mismatches in this diagnostic.

This observation indicates that errors originating in an earlier stage can propagate through the sequential architecture and affect final outputs.

### Threshold Exploration Results

Thresholds 0.3, 0.4, and 0.5 produced identical final predictions.

Thresholds 0.6 and 0.7 also produced identical predictions, with substantially higher recall.

| Final Output | Recall at 0.5 | Recall at 0.6/0.7 | F1 at 0.6/0.7 |
|---|---:|---:|---:|
| SLOW_BLOCK | 41.31% | 84.78% | 91.76% |
| FAST_BLOCK | 43.04% | 84.81% | 91.78% |
| INDICATION | 46.30% | 64.81% | 78.65% |

The reported false-positive rates at thresholds 0.6/0.7 were:

| Final Output | False Positive Rate |
|---|---:|
| SLOW_BLOCK | 0.0013211% |
| FAST_BLOCK | 0.0000000% |
| INDICATION | 0.0030300% |

### Interpretation and Limitations

The threshold exploration demonstrated that the final predictions were sensitive to the classification threshold. Higher thresholds produced substantially higher recall in this particular diagnostic.

However, these results must be interpreted cautiously:

1. The thresholds were evaluated against validation labels. Therefore, the reported results are exploratory and must not be presented as an unbiased estimate of generalization performance.
2. The results do not establish that the model is ready for practical protection-relay deployment.
3. Threshold selection requires further validation using an independent test dataset.
4. The first-divergence analysis suggests that intermediate-stage errors contribute to downstream prediction errors, but does not establish every underlying cause.

### Conclusion

This experiment provided additional insight into the relationship between classification thresholds, intermediate-stage errors, and final integrated performance.

Although the exploratory threshold results showed substantial recall improvement, further independent validation is necessary before considering the improvement reliable.

The experiment also reinforced the importance of evaluating the complete sequential architecture rather than relying exclusively on individual-stage accuracy.

**Artifacts generated:**
- `models/experimental/threshold_analysis/threshold_analysis_report.md`
- `models/experimental/threshold_analysis/threshold_analysis.json`
- `models/experimental/threshold_analysis/sigmoid_probabilities_threshold_0.5.npz`
- `scripts/training/run_threshold_analysis.py`

---

## Overall Project Findings, Limitations and Future Work

### Overall Findings

The project involved developing a six-stage neural network-based virtual replication of the VTS protection relay logic, with deterministic timer components and synchronized SR flip-flop state updates.

The following findings emerged from the implementation and experiments:

1. The six-stage architecture was successfully implemented and integrated.
2. Individual stages achieved high validation accuracy, although this did not always translate into equivalent end-to-end performance.
3. The baseline integrated evaluation revealed high precision but relatively low recall.
4. Intermediate-stage prediction errors were observed to propagate through subsequent stages.
5. Correcting Stage 1 signal polarity handling and implementing synchronized SR flip-flop updates did not change the final integrated metrics.
6. Class-weighted BCE training improved Stage 1 standalone validation accuracy but did not improve final integrated recall.
7. Exploratory threshold analysis demonstrated that classification thresholds can significantly influence final-output recall.

### Project Limitations

The current implementation has the following limitations:

- The integrated model does not yet demonstrate consistently satisfactory recall across all final outputs.
- The validation results indicate a distribution difference between training and validation data for several intermediate signals.
- Errors in upstream stages can propagate through the sequential architecture.
- The exploratory threshold results have not been independently validated.
- Generalization to unseen operating conditions, noise, and disturbances requires further investigation.
- Hardware-in-the-loop validation and real-time execution performance have not been established.

### Future Work

Potential directions for further research include:

1. Improving the generalization of intermediate-stage neural networks.
2. Investigating alternative architectures and training strategies for handling distribution differences.
3. Conducting systematic threshold calibration using a separate calibration dataset.
4. Evaluating the model on additional operating conditions and unseen test cases.
5. Performing hardware-in-the-loop validation and real-time performance analysis.
6. Investigating the effect of upstream prediction errors on downstream relay decisions.

### Final Conclusion

This project established a modular six-stage neural network-based approach for virtual replication of VTS protection relay logic. The implementation integrates learned logic stages with deterministic timing components and synchronized state updates.

Multiple experiments were conducted to investigate performance limitations, including signal-polarity correction, synchronized flip-flop updates, class-weighted training, and classification-threshold exploration.

Although the experiments did not establish a reliable improvement in final integrated performance, they provided useful insights into error propagation, class-distribution differences, and threshold sensitivity.

The work establishes a foundation for further investigation into neural-network-based virtual replication of protection relay logic, while identifying important limitations that must be addressed before practical deployment.
## Experiment 6: Stage-wise Threshold Sensitivity Analysis

**Objective:**

Identify which individual neural network stage contributes most significantly to the recall improvement observed during the global threshold exploration.

### Methodology

Eight sequential configurations were evaluated across the same 21 validation cases:

1. All six stages at threshold 0.5 (baseline).
2. Stage 1 alone at 0.6.
3. Stage 2 alone at 0.6.
4. Stage 3 alone at 0.6.
5. Stage 4 alone at 0.6.
6. Stage 5 alone at 0.6.
7. Stage 6 alone at 0.6.
8. All six stages at 0.6.

The corrected Stage 1 checkpoint and existing checkpoints for Stages 2–6 were used. The original sequential execution, deterministic timers and synchronized SR flip-flop logic were preserved.

No model retraining, dataset modification or checkpoint replacement was performed.

### Results

| Configuration | SLOW_BLOCK Recall | FAST_BLOCK Recall | INDICATION Recall |
|---|---:|---:|---:|
| All stages at 0.5 | 41.31% | 43.04% | 46.30% |
| Stage 1 only at 0.6 | 84.78% | 84.81% | 64.81% |
| Stages 2–6 individually at 0.6 | Unchanged | Unchanged | Unchanged |
| All stages at 0.6 | 84.78% | 84.81% | 64.81% |

### Observations

- Increasing only the Stage 1 threshold from 0.5 to 0.6 reproduced the recall improvement observed when all six stages were operated at 0.6.
- Changing the threshold of Stages 2–6 individually produced no measurable change in final-output metrics.
- Changing all six thresholds simultaneously did not provide additional improvement over changing Stage 1 alone.
- The baseline configuration exactly reproduced the previously recorded post-synchronization evaluation metrics.
- All eight configurations were evaluated across 21 validation cases.

### Interpretation

The results indicate that Stage 1 is the dominant contributor to the observed threshold sensitivity of the integrated model.

This finding is consistent with the earlier first-divergence analysis, which identified Stage 1 `1_AND_OUT` as the earliest recurring point of prediction divergence.

The results suggest that Stage 1 classification behaviour has a substantial influence on downstream predictions.

However, these experiments use validation labels to examine threshold performance. Consequently, the observed recall improvement is exploratory and does not establish generalization to unseen operating conditions.

### Conclusion

This experiment isolated the contribution of individual stage thresholds and identified Stage 1 as the most influential stage in the observed recall improvement.

It provides a more specific understanding of the integrated model's performance limitations and identifies Stage 1 as a priority for future investigation.

**Artifacts generated:**
- `models/experimental/stagewise_threshold_analysis/stagewise_threshold_analysis_report.md`
- `models/experimental/stagewise_threshold_analysis/stagewise_threshold_analysis.json`
- `models/experimental/stagewise_threshold_analysis/run_stagewise_threshold_analysis.py`

