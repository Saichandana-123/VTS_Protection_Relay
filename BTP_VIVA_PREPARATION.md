# BTP Viva Preparation ? VTS Virtual Replication

## Protection and VTS fundamentals

**1. What is a protection IED?**
An Intelligent Electronic Device measures electrical quantities, applies protection/control logic, and issues trip, block, or alarm signals. It is part of a protection scheme; it must behave predictably and safely.

**2. What does VTS do?**
Voltage Transformer Supervision monitors voltage-transformer circuit health and related indications/blocking logic. This project replicates the configured VTS function represented by the P642 Simulink reference and its logged signals. Do not claim to replicate every P642 protection function.

**3. What are the three modeled outputs?**
`SLOW_BLOCK`, `FAST_BLOCK`, and `INDICATION`. The network emits one binary prediction for each output at each time sample.

**4. What is a MiCOM P642?**
A protection IED in the MiCOM product family. Here it is the target relay whose VTS behavior is represented by the supplied Simulink reference. The project is not an approved firmware implementation.

## Model and hybrid design

**5. What is virtual replication?**
A software model that reproduces observed behavior of a physical/embedded function using measured inputs and reference outputs. Here the reference is the Simulink P642 VTS model.

**6. Why use neural networks?**
The project explores learning complex combinations of binary and detector-derived signals from reference data while keeping explicit timing/state blocks deterministic. Neural networks are not automatically safer or more accurate than logic; they require validation.

**7. What is an MLP?**
A feed-forward network of affine layers and nonlinear activations. Each stage maps its declared input signals to intermediate binary logic logits.

**8. Why six stages?**
The staged decomposition follows the reference signal-flow groups, keeps the model modular, and lets intermediate signals feed later stages. The final Stage 6 predicts the three user-visible outputs.

**9. Explain FiLM mathematically.**
Given hidden features `h` and setting vector `s`, a conditioner predicts `gamma(s)` and `beta(s)`. FiLM computes `h' = gamma(s) ? h + beta(s)`. In this implementation `gamma=1+delta_gamma`; the conditioner?s final layer is zero-initialized, so initial modulation is identity.

**10. Why FiLM instead of one fixed MLP?**
A conventional fixed-parameter MLP has no explicit setting-conditioned inference path. FiLM injects settings into hidden representations while holding learned weights fixed. This supports adaptation in inference, but does not prove accuracy for unseen settings.

**11. Why not put timers inside the network?**
Timing semantics are exact, stateful and safety-relevant. Keeping timers deterministic avoids asking a statistical model to approximate elapsed time, pickup, or dropout.

**12. What timers and state are present?**
The staged pipeline contains a 240 ms drop-off timer, a 20 ms timer, a setting-dependent tVTS timer, and four SR flip-flops. They are outside the MLPs.

**13. What does synchronized flip-flop update mean here?**
The implementation computes each next state from the current-step set/reset inputs, then commits state in the intended sample order. It avoids later same-step logic accidentally seeing an already-mutated state.

**14. Describe the execution sequence.**
Stage 1; 240 ms timer; Stage 2; SR1; Stage 3; SR2; Stage 4; tVTS and 20 ms timers; SR3; Stage 5; SR4; Stage 6. State resets between cases.

## Data and training

**15. Where did labels come from?**
The Simulink reference campaign supplies per-case input/output CSVs, including final targets and logged intermediate/detector signals. The Python model is evaluated against those recorded outputs.

**16. How many data cases and how split?**
108 cases, 13,001 samples each. Cases 001?087 (87) train; 088?108 (21) validate, giving 273,021 validation rows. The split sorts case IDs and assigns the last 20% to validation.

**17. Why split by case instead of random rows?**
Adjacent time samples are strongly correlated. A row-wise random split would place near-duplicate temporal points in both partitions and inflate performance.

**18. What are training settings?**
CPU, seed 20261104, Adam learning rate 0.001, batch 4096, BCE-with-logits, 12 epochs per stage, best validation-loss checkpoint.

**19. What do stage-wise accuracies mean?**
They are element accuracies over each stage?s intermediate target bits on validation. S1 was 98.3884%; S2?S6 were 100%. They are not integrated final-output scores and can conceal class imbalance or error propagation.

## Evaluation and metrics

**20. What is the final evaluation protocol?**
Load all six FiLM checkpoints and run each of cases 088?108 sequentially through `VTSStagedModel`, including timer and SR state. Threshold is 0.5 and was not tuned on validation labels.

**21. State the final results.**
SLOW_BLOCK: accuracy 99.9989%, precision 99.9935%, recall 100%, F1 99.9967%. FAST_BLOCK: 100% for all four. INDICATION: accuracy 95.9684%, precision 99.9948%, recall 89.8145%, F1 94.6316%.

**22. Give the confusion counts.**
Rows are ground truth 0/1 and columns prediction 0/1. SLOW: TN 227,012; FP 3; FN 0; TP 46,006. FAST: TN 194,019; FP 0; FN 0; TP 79,002. INDICATION: TN 165,000; FP 5; FN 11,002; TP 97,014.

**23. Define precision, recall and F1.**
Precision = TP/(TP+FP), fraction of positive predictions that are correct. Recall = TP/(TP+FN), fraction of actual positives detected. F1 = harmonic mean `2PR/(P+R)`. Accuracy = (TP+TN)/all samples. Specificity = TN/(TN+FP); FPR=FP/(FP+TN); FNR=FN/(FN+TP).

**24. Why is accuracy alone insufficient?**
If inactive samples dominate, a model can score high accuracy while missing active events. Recall and false negatives matter; INDICATION has 11,002 false negatives despite 95.97% accuracy.

**25. What was the non-FiLM baseline?**
The existing legacy non-FiLM model scored SLOW 90.1092% accuracy/41.3055% recall; FAST 83.5174%/43.0382%; INDICATION 78.7522%/46.2968%. These are historical baseline results, not FiLM results. The comparisons share the saved validation set but should not be overinterpreted as a controlled ablation.

## Experiments and interpretation

**26. What did Stage 1 polarity correction achieve?**
It corrected the `VTS_BLOCKING` polarity contract. Standalone Stage 1 validation changed marginally, but the saved sequential binary outputs on the old model were identical across 273,021 rows.

**27. What did SR synchronization achieve?**
It aligned state commits with the intended simulation order and was covered by tests. The historical non-FiLM aggregate final predictions did not materially change.

**28. Did class weighting fix recall?**
No. It improved an intermediate Stage 1 metric but did not change final integrated recall; it was not adopted as a proven end-to-end improvement.

**29. What did threshold experiments show?**
Validation threshold sweeps showed sensitivity, especially to Stage 1. Since validation labels were used to inspect thresholds, those results are exploratory and must not be represented as unbiased test performance. Final FiLM evaluation used 0.5 without tuning.

**30. What does the settings experiment prove?**
With one loaded checkpoint set, the same frozen 13,001-row case-100 input sequence was replayed under `[0.08,0.05,1.0]` and `[0.5,0.1,2.0]`. Outputs changed on 10,002 SLOW, 12,001 FAST, and 10,002 INDICATION rows without weight updates. It proves sensitivity, not accuracy for changed settings: case 100?s original settings were `[2.0,0.3,5.0]`, signals were frozen, and no new Simulink labels were generated.

## Limitations and future work

**31. Is the validation set independent?**
No. It is case-disjoint but remains within the original 108-case Simulink campaign. No independent scenarios outside that campaign were evaluated.

**32. Were analog detectors validated?**
Not independently. The model consumes logged Simulink detector outputs; this project does not establish analog front-end generalization.

**33. Was the model tested on hardware or real-time?**
No. There was no hardware-in-the-loop, field/noise, protection certification, or real-time performance validation.

**34. Can it replace the relay?**
No. It is a research prototype and behavioral replication framework. High validation scores do not establish safety, unseen-condition performance, or certification.

**35. What is the next work?**
Generate a provenance-controlled independent Simulink campaign including alternative settings; evaluate once with frozen thresholds; validate detector behavior; then conduct timing, HIL, and safety engineering review.
