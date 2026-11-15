# Virtual Replication of Voltage Transformer Supervision (VTS) Function in Protection IED using Neural Networks

A research prototype that reproduces selected VTS logic of a MiCOM P642 relay using a six-stage MLP with Feature-wise Linear Modulation (FiLM). Relay settings condition the neural stages at inference. Timers and synchronized SR flip-flops remain deterministic components outside the neural networks. This project is an experimental replica, not a protection-grade replacement.

## Final architecture

Stages 1?5 estimate intermediate combinational logic; stage 6 emits `SLOW_BLOCK`, `FAST_BLOCK`, and `INDICATION`. FiLM uses `I_SETTING`, `I2_SETTING`, and `tVTS`. `VTSStagedModel` orchestrates stages with the 240 ms drop-off timer, 20 ms timer, setting-dependent `tVTS` timer, and four SR flip-flops.

## Repository map

- `src/models/` ? FiLM conditioner, six stage MLPs, staged pipeline.
- `src/timing/` ? deterministic timer and SR flip-flop components.
- `src/logic/` ? logic/simulator references; final FiLM metrics use `VTSStagedModel`, not the older simplified simulator.
- `data/simulink_dataset/` ? combined training CSV plus per-case Simulink input/output CSVs.
- `models/Simulink/` ? P642 reference model and MATLAB workflow.
- `scripts/training/` ? training and evaluation scripts.
- `models/trained/film/` ? six FiLM checkpoints and metrics.
- `results/` ? summaries, plots, settings sensitivity artifacts; `results/figures/` contains presentation figures.
- `tests/` ? logic, timing, simulator and staged-pipeline tests.

## Environment and commands

Use Python with PyTorch, NumPy, pandas, matplotlib and pytest installed. `requirements.txt` is currently empty; install compatible packages in a project environment before running. From the repository root:

```powershell
python -m pytest -q
python scripts/training/train_six_stage_network.py
python scripts/training/evaluate_film_integrated.py
```

Training rewrites the FiLM checkpoints/metrics; do not run it merely to reproduce evaluation. Current checkpoints: `models/trained/film/stage1_best.pt` through `stage6_best.pt`. Training metrics: `models/trained/film/training_metrics.json`. Sequential metrics: `models/trained/film/integrated_evaluation.json`.

## Data and results

The original Simulink campaign has 108 cases (13,001 rows each): 87 train cases and 21 validation cases (088?108). Validation contains 273,021 rows and is **not an independent test campaign**. The changed-setting replay freezes Simulink-derived signals and does not provide counterfactual accuracy. See `BTP_Experimental_Results.md`, `GENERALIZATION_AUDIT.md`, `INDEPENDENT_TEST_DESIGN.md`, `decisions.md`, and `results/film_results_summary.md`. Figures are in `results/figures/`; source result plots are also in `results/`.
