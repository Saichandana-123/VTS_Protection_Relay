from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.vts_staged_model import VTSStagedModel


MODEL_DIR = ROOT / "models" / "trained" / "film"
DATA_DIR = ROOT / "data" / "simulink_dataset" / "outputs"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
RESULT_PATH = MODEL_DIR / "integrated_evaluation.json"
FINAL_OUTPUTS = ("SLOW_BLOCK", "FAST_BLOCK", "INDICATION")


def load_checkpoints(model: VTSStagedModel) -> None:
    for stage_name in ("stage1", "stage2", "stage3", "stage4", "stage5", "stage6"):
        checkpoint = torch.load(
            MODEL_DIR / f"{stage_name}_best.pt",
            map_location="cpu",
            weights_only=True,
        )
        getattr(model, stage_name).load_state_dict(
            checkpoint["model_state_dict"]
        )
        getattr(model, stage_name).eval()


def evaluate_case(case_id: str) -> dict[str, object]:
    frame = pd.read_csv(DATA_DIR / f"{case_id}_output.csv")
    model = VTSStagedModel(film_conditioned=True)
    load_checkpoints(model)
    model.reset()
    predictions = {name: [] for name in FINAL_OUTPUTS}

    for row in frame.to_dict(orient="records"):
        result = model.step(row)
        for name in FINAL_OUTPUTS:
            predictions[name].append(int(result[name]))

    case_result: dict[str, object] = {
        "case_id": case_id,
        "rows": len(frame),
        "outputs": {},
    }
    for name in FINAL_OUTPUTS:
        expected = frame[name].to_numpy(dtype=int)
        actual = np.asarray(predictions[name], dtype=int)
        tn = int(((expected == 0) & (actual == 0)).sum())
        fp = int(((expected == 0) & (actual == 1)).sum())
        fn = int(((expected == 1) & (actual == 0)).sum())
        tp = int(((expected == 1) & (actual == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        case_result["outputs"][name] = {
            "accuracy": float((actual == expected).mean()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_count": fp,
            "false_negative_count": fn,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }
    return case_result


def main() -> None:
    training_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    validation_cases = training_metrics["validation_cases"]
    cases = [evaluate_case(case_id) for case_id in validation_cases]
    summary = {}
    for name in FINAL_OUTPUTS:
        confusion = {
            key: sum(case["outputs"][name]["confusion_matrix"][key] for case in cases)
            for key in ("tn", "fp", "fn", "tp")
        }
        total = sum(case["rows"] for case in cases)
        correct = confusion["tn"] + confusion["tp"]
        precision = confusion["tp"] / (confusion["tp"] + confusion["fp"]) if confusion["tp"] + confusion["fp"] else 0.0
        recall = confusion["tp"] / (confusion["tp"] + confusion["fn"]) if confusion["tp"] + confusion["fn"] else 0.0
        summary[name] = {
            "accuracy": correct / total,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_positive_count": confusion["fp"],
            "false_negative_count": confusion["fn"],
            "confusion_matrix": confusion,
            "correct": correct,
            "total": total,
        }

    output = {
        "conditioning": {
            "settings": ["I_SETTING", "I2_SETTING", "tVTS"],
            "formula": "gamma(settings) * hidden + beta(settings)",
            "threshold": 0.5,
            "threshold_tuning": "none",
        },
        "validation_cases": validation_cases,
        "summary": summary,
        "cases": cases,
    }
    RESULT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for name, metrics in summary.items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.6f} "
            f"precision={metrics['precision']:.6f} "
            f"recall={metrics['recall']:.6f} f1={metrics['f1']:.6f}"
        )
    print(f"results={RESULT_PATH}")


if __name__ == "__main__":
    main()
