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
data/          → Input data and datasets
src/           → Implementation
models/        → Neural network models
logic/         → Relay logic implementation
timing/        → Timing mechanisms
tests/         → Test cases
experiments/   → Experiments and results
notebooks/     → Jupyter notebooks
docs/          → Project documentation
