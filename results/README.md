# VTS FiLM results

The primary result is the **six-stage FiLM sequential evaluation** in `film_integrated_summary.json` and `film_integrated_evaluation.json`. It uses Simulink-derived held-out cases 088?108 (273,021 rows), threshold 0.5, and no threshold tuning. Per-output confusion matrices use `tn`, `fp`, `fn`, `tp`.

## Artifacts

- `film_results_summary.md` ? PPT-ready verified metrics, confusion matrices, setting check, and limitations.
- `film_integrated_summary.json` ? final-output metrics, training split, test status, and limitations.
- `film_integrated_evaluation.json` ? per-case results copied from the existing FiLM evaluator output.
- `film_integrated_metrics.png` ? final-output accuracy, precision, recall, and F1 plot.
- `film_training_summary.json` and `film_stage_training_curves.png` ? stage-wise training/validation history; these are not end-to-end scores.
- `film_pipeline_settings_replay.json` ? same checkpoint set and frozen case_100 sequence under two settings configurations. All three outputs changed; alternate settings have no regenerated Simulink labels.
- `film_setting_sensitivity_stage1.json` ? fixed-input neural-stage sensitivity check, with no optimizer or weight update.
- `verification_summary.json` ? existing pytest result.
- `legacy_nonfilm_integrated_metrics.json` and `legacy_nonfilm_metrics.png` ? older non-FiLM reference only, clearly separate from FiLM results.

## Limitations for presentation

The split is case-disjoint, but it remains within the original 108-case Simulink campaign; no independent scenarios outside that campaign were tested. The changed-setting replay freezes signals from case_100 and does not rerun Simulink, so it establishes setting sensitivity but not counterfactual accuracy. Detector signals are logged Simulink inputs. No independent analog-detector, hardware-in-the-loop, field/noise, or protection-certification validation was performed.


Presentation and viva materials are at repository root: `BTP_FINAL_PRESENTATION_CONTENT.md` and `BTP_VIVA_PREPARATION.md`. Presentation-ready figures are organized under `results/figures/`.
