from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.stage1_model import Stage1MLP
from src.models.stage2_model import Stage2MLP
from src.models.stage3_model import Stage3MLP
from src.models.stage4_model import Stage4MLP
from src.models.stage5_model import Stage5MLP
from src.models.stage6_model import Stage6MLP


DATASET = ROOT / "data" / "simulink_dataset" / "vts_training_dataset.csv"
OUTPUT_DIR = ROOT / "models" / "trained"
SEED = 20261104
EPOCHS = 12
PATIENCE = 3
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3

STAGE_COLUMNS: Dict[str, Tuple[Sequence[str], Sequence[str]]] = {
    "stage1": (
        (
            "IA_OK", "IB_OK", "IC_OK",
            "DELTA_IA_OK", "DELTA_IB_OK", "DELTA_IC_OK",
            "VTS_MANRESET", "VTS_AUTORESET",
            "VAB_OK", "VBC_OK", "VTS_BLOCKING", "ACCELERATE_IND",
        ),
        ("1_OR_OUT", "2_OR_OUT", "3_OR_OUT", "1_AND_OUT", "2_AND_OUT"),
    ),
    "stage2": (
        ("V2", "I2", "240_TIMER_OUT", "1_AND_OUT", "2_OR_OUT"),
        ("3_AND_OUT", "4_AND_OUT", "4_OR_OUT", "5_AND_OUT"),
    ),
    "stage3": (
        (
            "ALL_POLE_DEAD", "1_OR_OUT", "4_OR_OUT", "4_AND_OUT",
            "SR_1_OUT", "3_OR_OUT", "ANY_VOLTAGE_DEPENDENT_FUNCTION",
        ),
        ("6_AND_OUT", "7_AND_OUT", "8_AND_OUT", "9_AND_OUT"),
    ),
    "stage4": (
        ("SR_2_OUT", "6_AND_OUT", "240_TIMER_OUT", "2_AND_OUT"),
        ("10_AND_OUT",),
    ),
    "stage5": (
        ("SR_3_OUT", "MCB_VTS", "8_AND_OUT", "VTS_TIMER_OUT", "20_TIMER_OUT"),
        ("5_OR_OUT", "6_OR_OUT", "7_OR_OUT"),
    ),
    "stage6": (
        ("5_OR_OUT", "6_OR_OUT", "VTS_BLOCKING", "MCB_VTS", "SR_4_OUT"),
        ("SLOW_BLOCK", "FAST_BLOCK", "INDICATION"),
    ),
}

MODELS = {
    "stage1": Stage1MLP,
    "stage2": Stage2MLP,
    "stage3": Stage3MLP,
    "stage4": Stage4MLP,
    "stage5": Stage5MLP,
    "stage6": Stage6MLP,
}


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_stage_data(
    inputs: Sequence[str], targets: Sequence[str], train_cases: set[str],
) -> Tuple[TensorDataset, TensorDataset]:
    columns = ["case_id", *inputs, *targets]
    frame = pd.read_csv(DATASET, usecols=columns)
    frame["case_id"] = frame["case_id"].astype(str)
    train_mask = frame["case_id"].isin(train_cases).to_numpy()
    x = frame[list(inputs)].to_numpy(dtype=np.float32)
    y = frame[list(targets)].to_numpy(dtype=np.float32)
    train = TensorDataset(torch.from_numpy(x[train_mask]), torch.from_numpy(y[train_mask]))
    valid = TensorDataset(torch.from_numpy(x[~train_mask]), torch.from_numpy(y[~train_mask]))
    return train, valid


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total_values = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y_device = y.to(device)
            total_loss += float(loss_fn(logits, y_device)) * y.numel()
            total_correct += int(((torch.sigmoid(logits) >= 0.5) == y_device.bool()).sum())
            total_values += y.numel()
    return total_loss / total_values, total_correct / total_values


def train_stage(
    name: str, inputs: Sequence[str], targets: Sequence[str],
    train_cases: set[str], valid_cases: set[str], device: torch.device,
) -> Dict[str, object]:
    train_data, valid_data = load_stage_data(inputs, targets, train_cases)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    model = MODELS[name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = evaluate(model, train_loader, device)
        valid_loss, valid_accuracy = evaluate(model, valid_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": valid_loss,
            "validation_accuracy": valid_accuracy,
        }
        history.append(record)
        print(
            f"{name} epoch {epoch}: "
            f"train_acc={train_accuracy:.4f} validation_acc={valid_accuracy:.4f}"
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint was produced for {name}.")
    model.load_state_dict(best_state)
    checkpoint_path = OUTPUT_DIR / f"{name}_best.pt"
    torch.save({
        "stage": name,
        "inputs": list(inputs),
        "targets": list(targets),
        "model_state_dict": best_state,
    }, checkpoint_path)
    final_train = evaluate(model, train_loader, device)
    final_valid = evaluate(model, valid_loader, device)
    return {
        "checkpoint": str(checkpoint_path),
        "inputs": list(inputs),
        "targets": list(targets),
        "history": history,
        "best_train_loss": final_train[0],
        "best_train_accuracy": final_train[1],
        "best_validation_loss": final_valid[0],
        "best_validation_accuracy": final_valid[1],
    }


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATASET, usecols=["case_id"])
    cases = sorted(frame["case_id"].astype(str).unique())
    split = max(1, int(len(cases) * 0.2))
    valid_cases = set(cases[-split:])
    train_cases = set(cases[:-split])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; train_cases={len(train_cases)}; validation_cases={len(valid_cases)}")
    metrics: Dict[str, object] = {
        "seed": SEED,
        "device": str(device),
        "train_cases": sorted(train_cases),
        "validation_cases": sorted(valid_cases),
        "stages": {},
    }
    for name, (inputs, targets) in STAGE_COLUMNS.items():
        metrics["stages"][name] = train_stage(
            name, inputs, targets, train_cases, valid_cases, device
        )
    metrics_path = OUTPUT_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
