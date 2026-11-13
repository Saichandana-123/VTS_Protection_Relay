from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.stage1_model import Stage1MLP
from src.models.stage2_model import Stage2MLP
from src.models.stage3_model import Stage3MLP
from src.models.stage4_model import Stage4MLP
from src.models.stage5_model import Stage5MLP
from src.models.stage6_model import Stage6MLP
from src.models.vts_staged_model import VTSStagedModel

DATASET = ROOT / "data" / "simulink_dataset" / "vts_training_dataset.csv"
OUTPUT_DIR = ROOT / "models" / "experimental" / "class_weighted"
ORIGINAL_OUTPUT_DIR = ROOT / "models" / "trained"
BASELINE_PATH = ORIGINAL_OUTPUT_DIR / "integrated_evaluation_post_sync.json"
SEED = 20261110
EPOCHS = 12
PATIENCE = 3
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3

STAGE_COLUMNS: Dict[str, Tuple[Sequence[str], Sequence[str]]] = {
    "stage1": (
        (
            "IA_OK", "IB_OK", "IC_OK", "DELTA_IA_OK", "DELTA_IB_OK",
            "DELTA_IC_OK", "VTS_MANRESET", "VTS_AUTORESET", "VAB_OK",
            "VBC_OK", "VTS_BLOCKING", "ACCELERATE_IND",
        ),
        ("1_OR_OUT", "2_OR_OUT", "3_OR_OUT", "1_AND_OUT", "2_AND_OUT"),
    ),
    "stage3": (
        (
            "ALL_POLE_DEAD", "1_OR_OUT", "4_OR_OUT", "4_AND_OUT",
            "SR_1_OUT", "3_OR_OUT", "ANY_VOLTAGE_DEPENDENT_FUNCTION",
        ),
        ("6_AND_OUT", "7_AND_OUT", "8_AND_OUT", "9_AND_OUT"),
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

MODEL_CLASSES = {
    "stage1": Stage1MLP,
    "stage2": Stage2MLP,
    "stage3": Stage3MLP,
    "stage4": Stage4MLP,
    "stage5": Stage5MLP,
    "stage6": Stage6MLP,
}

# Moderate, targeted weights. Unrelated outputs remain at weight 1.
POS_WEIGHTS = {
    "stage1": (1.0, 1.0, 1.0, 1.5, 1.0),
    "stage3": (2.0, 1.0, 1.0, 1.0),
    "stage5": (1.0, 1.5, 2.0),
    "stage6": (2.0, 2.0, 2.0),
}


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_data(
    inputs: Sequence[str],
    targets: Sequence[str],
    train_cases: set[str],
) -> Tuple[TensorDataset, TensorDataset]:
    frame = pd.read_csv(DATASET, usecols=["case_id", *inputs, *targets])
    train_mask = frame["case_id"].astype(str).isin(train_cases).to_numpy()
    x = frame[list(inputs)].to_numpy(dtype=np.float32)
    y = frame[list(targets)].to_numpy(dtype=np.float32)
    return (
        TensorDataset(torch.from_numpy(x[train_mask]), torch.from_numpy(y[train_mask])),
        TensorDataset(torch.from_numpy(x[~train_mask]), torch.from_numpy(y[~train_mask])),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_values = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            total_loss += float(loss_fn(logits, y)) * y.numel()
            total_correct += int(((torch.sigmoid(logits) >= 0.5) == y.bool()).sum())
            total_values += y.numel()
    return total_loss / total_values, total_correct / total_values


def train_stage(
    name: str,
    train_cases: set[str],
    validation_cases: set[str],
) -> dict[str, object]:
    inputs, targets = STAGE_COLUMNS[name]
    train_data, validation_data = load_data(inputs, targets, train_cases)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)
    model = MODEL_CLASSES[name]()
    weights = torch.tensor(POS_WEIGHTS[name], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = evaluate(model, train_loader, loss_fn)
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_fn
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss_weighted": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss_weighted": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"{name} epoch {epoch}: train_acc={train_accuracy:.4f} "
            f"validation_acc={validation_accuracy:.4f}"
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
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
        raise RuntimeError(f"No checkpoint was produced for {name}.")

    model.load_state_dict(best_state)
    checkpoint_path = OUTPUT_DIR / f"{name}_class_weighted_best.pt"
    torch.save(
        {
            "stage": name,
            "inputs": list(inputs),
            "targets": list(targets),
            "pos_weight": list(POS_WEIGHTS[name]),
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    unweighted_loss = nn.BCEWithLogitsLoss()
    train_unweighted = evaluate(model, train_loader, unweighted_loss)
    validation_unweighted = evaluate(model, validation_loader, unweighted_loss)
    return {
        "checkpoint": str(checkpoint_path),
        "inputs": list(inputs),
        "targets": list(targets),
        "pos_weight": list(POS_WEIGHTS[name]),
        "history": history,
        "best_weighted_validation_loss": best_loss,
        "unweighted_train_accuracy": train_unweighted[1],
        "unweighted_validation_loss": validation_unweighted[0],
        "unweighted_validation_accuracy": validation_unweighted[1],
    }


def load_checkpoint(model: nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


def load_integrated_models() -> VTSStagedModel:
    model = VTSStagedModel()
    for name in ("stage1", "stage3", "stage5", "stage6"):
        load_checkpoint(
            getattr(model, name),
            OUTPUT_DIR / f"{name}_class_weighted_best.pt",
        )
    for name in ("stage2", "stage4"):
        load_checkpoint(getattr(model, name), ORIGINAL_OUTPUT_DIR / f"{name}_best.pt")
    return model


def evaluate_case(case_id: str) -> dict[str, object]:
    frame = pd.read_csv(
        ROOT / "data" / "simulink_dataset" / "outputs" / f"{case_id}_output.csv"
    )
    model = load_integrated_models()
    model.reset()
    outputs = ("SLOW_BLOCK", "FAST_BLOCK", "INDICATION")
    predictions = {name: [] for name in outputs}
    for row in frame.to_dict(orient="records"):
        result = model.step(row)
        for name in outputs:
            predictions[name].append(int(result[name]))

    case_result = {"case_id": case_id, "rows": len(frame), "outputs": {}}
    for name in outputs:
        expected = frame[name].to_numpy(dtype=int)
        actual = np.asarray(predictions[name])
        tp = int(((expected == 1) & (actual == 1)).sum())
        tn = int(((expected == 0) & (actual == 0)).sum())
        fp = int(((expected == 0) & (actual == 1)).sum())
        fn = int(((expected == 1) & (actual == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fnr = fn / (fn + tp) if fn + tp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        case_result["outputs"][name] = {
            "correct": int((expected == actual).sum()),
            "accuracy": float((expected == actual).mean()),
            "mismatch_count": int((expected != actual).sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }
    return case_result


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    summary = {}
    for name in ("SLOW_BLOCK", "FAST_BLOCK", "INDICATION"):
        total = sum(int(result["rows"]) for result in results)
        confusion = {
            key: sum(
                int(result["outputs"][name]["confusion_matrix"][key])
                for result in results
            )
            for key in ("tn", "fp", "fn", "tp")
        }
        tn, fp, fn, tp = (confusion[key] for key in ("tn", "fp", "fn", "tp"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fnr = fn / (fn + tp) if fn + tp else 0.0
        summary[name] = {
            "correct": total - fn - fp,
            "total": total,
            "accuracy": (total - fn - fp) / total,
            "mismatch_count": fn + fp,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            "specificity": specificity,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "confusion_matrix": confusion,
        }
    return summary


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATASET, usecols=["case_id"])
    cases = sorted(frame["case_id"].astype(str).unique())
    split = max(1, int(len(cases) * 0.2))
    validation_cases = set(cases[-split:])
    train_cases = set(cases[:-split])

    stage_results = {
        name: train_stage(name, train_cases, validation_cases)
        for name in ("stage1", "stage3", "stage5", "stage6")
    }
    case_results = [evaluate_case(case_id) for case_id in sorted(validation_cases)]
    output = {
        "experiment": "targeted_class_weighted_bce",
        "seed": SEED,
        "train_cases": sorted(train_cases),
        "validation_cases": sorted(validation_cases),
        "dataset_rows": len(frame),
        "stages": stage_results,
        "summary": summarize(case_results),
        "cases": case_results,
        "baseline_path": str(BASELINE_PATH),
        "baseline_summary": json.loads(
            BASELINE_PATH.read_text(encoding="utf-8")
        )["summary"],
    }
    result_path = OUTPUT_DIR / "class_weighted_experiment.json"
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"results={result_path}")
    for name, metrics in output["summary"].items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.6f} "
            f"precision={metrics['precision']:.6f} "
            f"recall={metrics['recall']:.6f} "
            f"f1={metrics['f1']:.6f} "
            f"fpr={metrics['false_positive_rate']:.6f}"
        )


if __name__ == "__main__":
    main()
