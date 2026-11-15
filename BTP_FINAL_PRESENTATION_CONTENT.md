# BTP Final Presentation Content (Maximum 15 Slides)

## Slide 1 ? Title
**Display:** Virtual Replication of Voltage Transformer Supervision (VTS) Function in Protection IED using Neural Networks. Six-stage MLP + FiLM; deterministic timers and SR flip-flops.
**Visual:** P642 relay photo/icon only if licensed; otherwise a simple relay-to-virtual-model schematic.
**Say:** This is an experimental behavioral replication of selected P642 VTS logic, not a replacement relay.

## Slide 2 ? Problem and motivation
**Display:** Relay logic combines analog-derived detectors, settings, timers, and state. A fixed network does not naturally adapt when settings change.
**Visual:** Inputs ? VTS decisions ? SLOW_BLOCK / FAST_BLOCK / INDICATION.
**Say:** Accurate timing and state matter as much as static logic.

## Slide 3 ? Project objectives
**Display:** Replicate logged Simulink VTS behavior; condition learned logic on relay settings; keep timing/state deterministic; compare final outputs against Simulink labels.
**Visual:** Objective checklist.
**Say:** Ground truth is from the checked-in Simulink reference campaign.

## Slide 4 ? Dataset and split
**Display:** 108 cases ? 13,001 rows = 1,404,108. Train: cases 001?087 (87). Validation: 088?108 (21; 273,021 rows).
**Visual:** Case-wise split bar.
**Say:** Split is by case to avoid temporal row leakage; validation is not an independent campaign.

## Slide 5 ? Hybrid architecture
**Display:** Six MLP stages with FiLM; deterministic 240 ms, 20 ms, and tVTS timers; four SR flip-flops.
**Visual:** `results/figures/six_stage_architecture.png`.
**Say:** Neural stages model combinational groups; explicit state/timing stays interpretable and deterministic.

## Slide 6 ? FiLM intuition and equation
**Display:** `h_cond = gamma(s) ? h + beta(s)`, `s=[I_SETTING,I2_SETTING,tVTS]`. Settings alter hidden features without weight updates.
**Visual:** Settings ? FiLM scale/shift ? MLP hidden features.
**Say:** The output layer of the conditioner starts at zero, giving identity modulation initially (`gamma=1`, `beta=0`).

## Slide 7 ? Six-stage execution
**Display:** S1 ? 240 ms timer ? S2 ? SR1 ? S3 ? SR2 ? S4 ? tVTS + 20 ms timers ? SR3 ? S5 ? SR4 ? S6.
**Visual:** Architecture diagram with state blocks outside MLP boxes.
**Say:** Pipeline calls `VTSStagedModel.step()` once per 1 ms sample and resets state between cases.

## Slide 8 ? Training method
**Display:** CPU; seed 20261104; Adam 1e-3; batch 4096; BCE-with-logits; 12 epochs/stage; best validation-loss checkpoint.
**Visual:** `results/figures/film_training_validation_accuracy.png` and `results/figures/film_training_validation_loss.png`.
**Say:** Stage-wise element accuracy is not integrated output accuracy.

## Slide 9 ? Stage-wise training results
**Display:** Validation accuracy: S1 98.3884%; S2?S6 100%.
**Visual:** Training curve or compact table including validation loss.
**Say:** High intermediate accuracy can still mislead under sequential error propagation and class imbalance.

## Slide 10 ? Evaluation protocol
**Display:** 21 held-out cases; 273,021 rows; sequential stateful inference; threshold 0.5; no threshold tuning.
**Visual:** Golden twin labels vs sequential predictions.
**Say:** Evaluated final output bits against Simulink-derived labels.

## Slide 11 ? Final FiLM metrics
**Display:** SLOW_BLOCK: Acc 99.9989%, Prec 99.9935%, Rec 100%, F1 99.9967%. FAST_BLOCK: 100% for all. INDICATION: Acc 95.9684%, Prec 99.9948%, Rec 89.8145%, F1 94.6316%.
**Visual:** `results/figures/film_integrated_metrics.png`.
**Say:** INDICATION has 11,002 false negatives; accuracy alone hides this.

## Slide 12 ? Confusion matrices
**Display:** SLOW `[[227012,3],[0,46006]]`; FAST `[[194019,0],[0,79002]]`; INDICATION `[[165000,5],[11002,97014]]` (rows true 0/1, columns predicted 0/1).
**Visual:** `results/figures/film_confusion_matrices.png`.
**Say:** Report false negatives and recall, not only accuracy.

## Slide 13 ? Settings sensitivity demonstration
**Display:** Same checkpoint set, same frozen 13,001-row case-100 signal sequence, configs `[0.08,0.05,1.0]` and `[0.5,0.1,2.0]`; changed predictions: SLOW 10,002 rows, FAST 12,001, INDICATION 10,002. No retraining.
**Visual:** `results/figures/film_settings_sensitivity.png`.
**Say:** Case 100 originally used `[2.0,0.3,5.0]`; alternative settings had no regenerated Simulink labels. This proves sensitivity, not changed-setting accuracy.

## Slide 14 ? Previous experiments and learning
**Display:** Polarity correction and SR synchronization did not materially change legacy integrated metrics; class weighting improved a stage metric but not final recall; validation threshold analysis was exploratory. Final FiLM is a separate model/result.
**Visual:** `results/figures/nonfilm_vs_film.png`, clearly label legacy vs final.
**Say:** Do not present old non-FiLM scores as final FiLM scores.

## Slide 15 ? Limitations and conclusion
**Display:** Case-wise holdout within 108-case campaign only; no independent campaign, changed-setting golden-twin labels, analog detector validation, HIL, field/noise or real-time testing.
**Visual:** Next steps: independent Simulink cases ? detector validation ? HIL.
**Say:** Promising prototype result; not protection-grade validation or certification.

