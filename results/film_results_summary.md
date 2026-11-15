# FiLM VTS results summary

Integrated sequential evaluation on 21 case-wise held-out Simulink cases (273,021 rows). Threshold = 0.5; none.

## Training

Training ran on cpu with seed 20261104, 87 training cases, and 21 validation cases. Each of six stages trained for 12 epochs.

| Stage | Validation element accuracy | Validation BCE loss |
|---|---:|---:|
| stage1 | 98.388402% | 0.0354811 |
| stage2 | 100.000000% | 0.000209143 |
| stage3 | 100.000000% | 0.0170315 |
| stage4 | 100.000000% | 4.08718e-07 |
| stage5 | 100.000000% | 1.05895e-05 |
| stage6 | 100.000000% | 7.23225e-06 |

Stage-wise accuracy measures intermediate target bits and is not the end-to-end system score.

## Final outputs

| Output | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SLOW_BLOCK | 99.998901% | 99.993480% | 100.000000% | 99.996740% | 227,012 | 3 | 0 | 46,006 |
| FAST_BLOCK | 100.000000% | 100.000000% | 100.000000% | 100.000000% | 194,019 | 0 | 0 | 79,002 |
| INDICATION | 95.968442% | 99.994846% | 89.814472% | 94.631648% | 165,000 | 5 | 11,002 | 97,014 |

Confusion matrix convention: rows = ground truth `[0, 1]`; columns = prediction `[0, 1]`, so `[[TN, FP], [FN, TP]]`.

## Settings and pipeline check

A fixed FiLM checkpoint produced a different Stage 1 binary output under two setting vectors. In a full 13,001-row `case_100` replay, the same six checkpoints and frozen Simulink-derived signal sequence were run under `[I_SETTING, I2_SETTING, tVTS] = [0.08, 0.05, 1.0]` and `[0.5, 0.1, 2.0]`; SLOW_BLOCK changed on 10,002 rows, FAST_BLOCK on 12,001, and INDICATION on 10,002. No retraining or weight updates occurred. These are counterfactual sensitivity results only: case_100 originally used `[2.0, 0.3, 5.0]`, the signals were not regenerated under the changed settings, and there are no golden-twin labels for those replays.

## Verification

`python -m pytest -q`: **8 passed**. The end-to-end evaluator ran `VTSStagedModel`, whose timers and SR flip-flops remain deterministic components outside the MLPs.

## Limitations to state in the PPT

- The holdout is case-disjoint but remains inside the original 108-case Simulink campaign (cases 088?108); it is not an independent test campaign, and unseen-condition generalization is unproven.
- The Stage 1 model consumes logged Simulink detector signals; independent analog detector validation was not performed.
- The settings replay establishes output sensitivity, not accuracy under newly simulated settings.
- No field, noise, hardware-in-the-loop, real-time performance, or protection-grade safety validation was performed.
- The legacy non-FiLM baseline files in this folder are context only and must not be described as FiLM results.

## Files

- `film_integrated_evaluation.json` ? per-case integrated evaluation.
- `film_integrated_metrics.png` ? final-output metrics plot.
- `film_training_summary.json` and `film_stage_training_curves.png` ? training history.
- `film_pipeline_settings_replay.json` ? full sequence settings replay details.
- `film_setting_sensitivity_stage1.json` ? fixed-input neural-stage sensitivity.
