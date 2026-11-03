from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.stage1_model import STAGE1_INPUTS, STAGE1_OUTPUTS, Stage1MLP


DATASET = ROOT / "data" / "simulink_dataset" / "vts_training_dataset.csv"
TRAINED = ROOT / "models" / "trained"
SEED = 20261104
EPOCHS = 12
PATIENCE = 3
BATCH_SIZE = 4096
DATASET_STAGE1_INPUTS = (
    "IA_OK", "IB_OK", "IC_OK",
    "DELTA_IA_OK", "DELTA_IB_OK", "DELTA_IC_OK",
    "VTS_MANRESET", "VTS_AUTORESET", "VAB_OK", "VBC_OK",
    "VTS_BLOCKING", "ACCELERATE_IND",
)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    training_metrics = json.loads(
        (TRAINED / "training_metrics.json").read_text(encoding="utf-8")
    )
    train_cases = set(training_metrics["train_cases"])
    valid_cases = set(training_metrics["validation_cases"])
    columns = ["case_id", *DATASET_STAGE1_INPUTS, *STAGE1_OUTPUTS]
    frame = pd.read_csv(DATASET, usecols=columns)
    frame["case_id"] = frame["case_id"].astype(str)

    features = frame[list(DATASET_STAGE1_INPUTS)].to_numpy(dtype=np.float32)
    targets = frame[list(STAGE1_OUTPUTS)].to_numpy(dtype=np.float32)
    train_mask = frame["case_id"].isin(train_cases).to_numpy()
    train_data = TensorDataset(
        torch.from_numpy(features[train_mask]),
        torch.from_numpy(targets[train_mask]),
    )
    valid_data = TensorDataset(
        torch.from_numpy(features[~train_mask]),
        torch.from_numpy(targets[~train_mask]),
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)

    model = Stage1MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(batch_x), batch_y).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = torch.cat([model(x) for x, _ in train_loader])
            train_targets = torch.cat([y for _, y in train_loader])
            valid_logits = torch.cat([model(x) for x, _ in valid_loader])
            valid_targets = torch.cat([y for _, y in valid_loader])
        train_loss = float(loss_fn(train_logits, train_targets))
        valid_loss = float(loss_fn(valid_logits, valid_targets))
        train_accuracy = float(
            ((torch.sigmoid(train_logits) >= 0.5) == train_targets.bool()).float().mean()
        )
        valid_accuracy = float(
            ((torch.sigmoid(valid_logits) >= 0.5) == valid_targets.bool()).float().mean()
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": valid_loss,
            "validation_accuracy": valid_accuracy,
        })
        print(
            f"stage1_corrected epoch {epoch}: "
            f"train_acc={train_accuracy:.4f} validation_acc={valid_accuracy:.4f}"
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("Corrected Stage 1 did not produce a checkpoint.")

    checkpoint_path = TRAINED / "stage1_corrected_best.pt"
    torch.save({
        "stage": "stage1_corrected",
        "inputs": STAGE1_INPUTS,
        "targets": STAGE1_OUTPUTS,
        "model_state_dict": best_state,
    }, checkpoint_path)
    metrics_path = TRAINED / "stage1_corrected_metrics.json"
    metrics_path.write_text(json.dumps({
        "checkpoint": str(checkpoint_path),
        "train_cases": sorted(train_cases),
        "validation_cases": sorted(valid_cases),
        "history": history,
        "best_validation_loss": best_loss,
        "best_validation_accuracy": max(
            item["validation_accuracy"] for item in history
            if item["validation_loss"] == min(
                record["validation_loss"] for record in history
            )
        ),
    }, indent=2), encoding="utf-8")
    print(f"checkpoint={checkpoint_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
