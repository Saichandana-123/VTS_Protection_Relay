# Final FiLM images

This folder contains curated final-architecture figures for the BTP report/presentation. Numeric plots use final FiLM checkpoints or their saved histories. No legacy non-FiLM-only plot is included.

| File | Use | Source / note |
|---|---|---|
| `01_final_six_stage_film_architecture.png` | Architecture slide | Schematic of six stages, FiLM settings, deterministic timers and SR flip-flops. |
| `02_training_validation_accuracy.png` | Training slide | Saved per-stage histories; element accuracy, not integrated output accuracy. |
| `03_training_validation_loss.png` | Training slide | Saved training/validation BCE histories. |
| `04_final_integrated_metrics.png` | Final results slide | Final FiLM results on cases 088-108; 273,021 rows; threshold 0.5, no tuning. |
| `05_final_confusion_matrices.png` | Final results slide/report | Final FiLM counts; rows actual class, columns predicted class. |
| `06_settings_sensitivity_frozen_input.png` | Adaptability/demo slide | Frozen case-100 sequence under two settings, no retraining. Sensitivity only; changed-setting Simulink labels were not generated. |

Legacy images `legacy_nonfilm_metrics.png` and `nonfilm_vs_film.png` remain outside this folder. Use them only if clearly labeled as legacy non-FiLM context.
