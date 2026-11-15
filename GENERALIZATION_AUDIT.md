# Generalization Audit

**Audit scope:** Phase 1 only. This audit inspected the existing dataset, split
and training artifacts, model definitions, integrated evaluator, dataset checks,
Simulink workflow files, and existing experiment documentation. No dataset,
checkpoint, implementation file, or existing report was modified.

## 1. Current project status

The project has:

- Six trained neural stages in `models/trained/`.
- A synchronized integrated model in `src/models/vts_staged_model.py`.
- Deterministic timers and four synchronized SR flip-flops.
- Simulink reference assets and a batch workflow under `models/Simulink/`.
- A complete 108-case dataset with intermediate signals and final outputs.
- Validation and exploratory threshold/class-weighted experiment artifacts.

The project does **not** yet have an independent test set generated from cases
outside the original 108 cases. Therefore, current validation metrics do not
demonstrate generalization to genuinely unseen Simulink scenarios.

## 2. Verified data split

The saved split in [`training_metrics.json`](models/trained/training_metrics.json)
contains:

| Split | Cases | Rows | Case IDs |
|---|---:|---:|---|
| Training | 87 | 1,131,087 | `case_001`–`case_087` |
| Validation | 21 | 273,021 | `case_088`–`case_108` |
| Total | 108 | 1,404,108 | `case_001`–`case_108` |

Each case has 13,001 samples. The training script creates the split by sorting
unique case IDs and assigning the final 20% of cases to validation. The saved
artifact confirms the resulting 87/21 case split.

### Overlap checks

- Training and validation case-ID sets are disjoint.
- There are no cases outside `case_001`–`case_108` in the combined training CSV.
- An exact row-value overlap check across all columns except `case_id` found
  **0 distinct sample rows shared between training and validation**.
- This confirms no exact sample duplication across the saved split. It does not
  prove that the underlying scenario distributions are independent.

## 3. Leakage and preprocessing audit

The training scripts read raw columns directly with pandas and convert them to
`float32` tensors. No fitted normalization, standardization, imputation,
feature-selection model, or validation-derived preprocessing artifact was
found.

The split mask is applied using `case_id` before constructing the training and
validation tensors. The saved dataset includes derived detector and intermediate
signals from the Simulink workflow, but these are available for both splits as
part of the generated reference data rather than being fitted from validation
labels.

No conventional preprocessing leakage was identified. Important remaining
limitations are:

- The 21 validation cases come from the same original 108-case generation
  campaign.
- Threshold experiments used validation labels and are therefore exploratory,
  not independent performance estimates.
- The dataset contains multiple related combinations of a fixed scenario set
  and fixed setting grid; distributional independence from future cases is not
  established.

## 4. Checkpoints currently used for inference

The post-synchronization evaluator and threshold diagnostics use:

| Stage | Checkpoint |
|---|---|
| Stage 1 | `stage1_corrected_best.pt` |
| Stage 2 | `stage2_best.pt` |
| Stage 3 | `stage3_best.pt` |
| Stage 4 | `stage4_best.pt` |
| Stage 5 | `stage5_best.pt` |
| Stage 6 | `stage6_best.pt` |

The current saved post-synchronization validation reference is
[`integrated_evaluation_post_sync.json`](models/trained/integrated_evaluation_post_sync.json).
At threshold 0.5 it reports:

| Output | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `SLOW_BLOCK` | 90.1092% | 99.9947% | 41.3055% | 58.4618% |
| `FAST_BLOCK` | 83.5174% | 100.0000% | 43.0382% | 60.1772% |
| `INDICATION` | 78.7522% | 99.9940% | 46.2968% | 63.2905% |

These are validation results only.

## 5. Evaluator capability

[`evaluate_integrated_model.py`](scripts/training/evaluate_integrated_model.py)
loads the six checkpoints and runs the synchronized model sequentially, but it
is currently hard-coded to:

- Read outputs from `data/simulink_dataset/outputs/`.
- Read the validation case list from `training_metrics.json`.
- Write a fixed result artifact under `models/trained/`.

Consequently, the evaluator is **not currently a generic arbitrary-case
command-line evaluator**. It can evaluate a new case after a controlled
standalone wrapper or parameterization is added, but that change should be kept
separate from the existing evaluator and should not alter model behavior.

The integrated model itself exposes `step(inputs)` and can process a compatible
row dictionary sequentially, including timer and SR state reset between cases.

## 6. Independent test data status

No `data/independent_test/` directory or other clearly designated independent
case set exists. The only case-wise generated dataset is `case_001` through
`case_108`.

The repository does contain:

- Existing archive CSVs under `data/archive/`.
- The combined dataset and per-case input/output CSVs.
- A Simulink model and MATLAB workflow under `models/Simulink/`.

The archive files are not identified as independent post-training test data and
must not be treated as such without provenance verification.

## 7. Simulink generation workflow

The established workflow is present:

1. Generate input CSVs and `manifest.csv`.
2. Run [`run_vts_dataset.m`](models/Simulink/run_vts_dataset.m) against
   `P642_VTS_Reference.slx`.
3. Save per-case output CSVs containing inputs, outputs, detectors, intermediate
   signals, and case metadata.
4. Validate sample counts and required signals.

The checked-in Python generator
[`generate_simulink_inputs.py`](src/data_generation/generate_simulink_inputs.py)
creates a fixed 9-scenario by 12-setting input grid; it does not run Simulink.
Its scenario profiles include comments warning that some profiles are initial
proxies requiring confirmation against the relay manual and circuit.

The Simulink workflow is therefore available in the repository, but actual
generation of new independent cases requires MATLAB/Simulink execution and
verification. This audit did not run it.

## 8. Stage mapping consistency

The current model constants and integrated evaluator define the sequential
connections:

`stage1 -> 240 ms timer -> stage2 -> SR1 -> stage3 -> SR2 -> stage4 -> tVTS/20 ms timers -> SR3 -> stage5 -> SR4 -> stage6`.

The current stage output names and final output names are wired consistently in
the integrated model and evaluator:

- Stage 1: five combinational outputs.
- Stage 2: four outputs, including `4_OR_OUT` feeding `5_AND_OUT`.
- Stage 3: four outputs, including timer/SR-driving signals.
- Stage 4: `10_AND_OUT`.
- Stage 5: `5_OR_OUT`, `6_OR_OUT`, `7_OR_OUT`.
- Stage 6: `SLOW_BLOCK`, `FAST_BLOCK`, `INDICATION`.

The following metadata inconsistencies were found:

1. `stage1_corrected_best.pt` metadata names inputs such as `IA_GT` and
   `NOT_VTS_BLOCKING`, while the current `stage1_model.py` constants and
   evaluator use `IA_GT` aliases with `VTS_BLOCKING`; the model implementation
   performs the `2_AND_OUT` blocking inversion internally.
2. `stage5_best.pt` and `stage6_best.pt` metadata use `MCB_VTS`, while the
   current model constants use `MCB_VTS_OPTO`; the evaluator supplies the
   compatibility alias from the dataset's `MCB_VTS` column.
3. The original `stage1_best.pt` metadata uses `*_OK` names, while the current
   inference path intentionally uses the corrected Stage 1 checkpoint.

The checkpoint tensor shapes are loadable by the current models and the saved
post-synchronization evaluation completed, but checkpoint metadata should be
normalized or documented in any future independent-test wrapper. No checkpoint
was changed during this audit.

## 9. Workflow inconsistencies

Several scripts use inconsistent repository-root resolution:

- `train_corrected_stage1.py` and `evaluate_integrated_model.py` resolve the
  repository root with `Path(__file__).resolve().parents[2]`.
- `train_six_stage_network.py` and `combine_simulink_dataset.py` use
  `Path(__file__).resolve().parent`, which resolves to their script directory
  rather than the repository root when run from their stored locations.
- `check_dataset.py` uses paths relative to the current working directory.

The saved artifacts demonstrate that the dataset and training runs were
previously completed, but these scripts should not be used for independent
generation or retraining without first handling the path issue in a separate,
explicitly approved change.

## 10. Exact remaining tasks

1. Confirm MATLAB/Simulink availability and the correct reference model
   version before generating any cases.
2. Define an independent scenario matrix outside the original 9-scenario by
   12-setting combinations.
3. Generate new input cases under a separate directory, without modifying
   `data/simulink_dataset/`.
4. Run the existing Simulink model on those cases and preserve the original
   62-column output schema.
5. Validate every new file for schema, timestamps, finite values, sample count,
   ranges, manifest conditions, and complete logged signals.
6. Add a standalone arbitrary-case evaluator or wrapper that loads the current
   six checkpoints and preserves the existing sequential model behavior.
7. Evaluate independent cases once, without threshold tuning on their labels.
8. Produce per-case, scenario-grouped, stage-propagation, and aggregate
   metrics, then compare them with the saved validation reference.
9. Update the independent-test report and project decision log only after
   measured independent results exist.

## 11. Proposed next steps before Phase 3

Before any generation is started, confirm:

- MATLAB and Simulink are available in the execution environment.
- `models/Simulink/P642_VTS_Reference.slx` is the intended reference model.
- The proposed independent scenario count and parameter ranges are approved.
- Cases will be generated with IDs outside `case_001`–`case_108`.

If MATLAB/Simulink is unavailable, the correct next action is to provide a
precise manual generation procedure and stop; fabricated or Python-only
surrogate outputs would not constitute independent Simulink testing.

## Audit conclusion

The training/validation split is case-disjoint and has no exact cross-split
duplicate sample rows. No ordinary normalization or preprocessing leakage was
found. However, the validation set is not an independent generalization test,
the evaluator is hard-coded to the validation directory/list, and several
workflow/checkpoint metadata inconsistencies require controlled handling.

Independent generalization has therefore **not yet been demonstrated**.


## FiLM status update ? 2026-11-15

The final six-stage FiLM model has since completed training and sequential validation. It used the same 87/21 case split within the original 108-case Simulink campaign, so the audit conclusion remains: no independent generalization test exists. Final FiLM validation metrics at threshold 0.5 are SLOW_BLOCK accuracy/precision/recall/F1 99.9989/99.9935/100/99.9967%; FAST_BLOCK 100/100/100/100%; INDICATION 95.9684/99.9948/89.8145/94.6316%. Source: `models/trained/film/integrated_evaluation.json`.

A two-setting replay changed predictions without retraining, but case-100 Simulink-derived signals were frozen and no changed-setting labels were regenerated. This is a sensitivity demonstration, not unseen-setting accuracy. Analog detector generalization, hardware-in-the-loop, field/noise, real-time, and protection-safety claims remain unverified. See `BTP_Experimental_Results.md` and `results/film_results_summary.md` for complete metrics and interpretation.
