# Virtual Replication of Protection IED VTS Logic

Virtual replication of Voltage Transformer Supervision (VTS) logic of a Protection IED using a neural-network-based architecture.

## Objective

The objective is to develop a virtual model of the VTS functionality that can reproduce the behavior of the original Protection IED while adapting to changes in relay settings without requiring complete retraining.

## Current Approach

The project is investigating an MLP + FiLM-based architecture with explicit mechanisms for:

- Analog and binary relay inputs
- Relay settings
- Logic operations
- Flip-flop/state behavior
- Timing mechanisms
- VTS outputs

## Project Structure

```text
data/             → Input data and completed datasets
src/              → Python implementation
models/           → Simulink reference model and trained checkpoints
scripts/dataset/  → Dataset preparation and inspection entry points
scripts/training/ → Training and integrated evaluation entry points
scripts/matlab/   → MATLAB utility entry points
docs/             → Signal-flow documentation data
tests/            → Test cases
decisions.md      → Append-only engineering decision log
```

Common commands are run from the project root:

```powershell
python scripts\dataset\check_dataset.py
python scripts\dataset\combine_simulink_dataset.py
python scripts\training\train_six_stage_network.py
python scripts\training\evaluate_integrated_model.py
