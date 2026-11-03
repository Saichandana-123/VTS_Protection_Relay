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


OUTPUT_DIR = ROOT / "models" / "trained"
DATA_DIR = ROOT / "data" / "simulink_dataset" / "outputs"
METRICS_PATH = OUTPUT_DIR / "training_metrics.json"
BASELINE_PATH = OUTPUT_DIR / "integrated_evaluation.json"
RESULT_PATH = OUTPUT_DIR / "integrated_evaluation_corrected_stage1.json"
FINAL_OUTPUTS = ("SLOW_BLOCK", "FAST_BLOCK", "INDICATION")


def load_checkpoints(model: VTSStagedModel) -> None:
    for stage_name in ("stage1", "stage2", "stage3", "stage4", "stage5", "stage6"):
        filename = (
            "stage1_corrected_best.pt"
            if stage_name == "stage1"
            else f"{stage_name}_best.pt"
        )
        checkpoint = torch.load(
            OUTPUT_DIR / filename,
            map_location="cpu",
            weights_only=True,
        )
        getattr(model, stage_name).load_state_dict(checkpoint["model_state_dict"])
        getattr(model, stage_name).eval()


def evaluate_case(case_id: str) -> dict[str, object]:
    frame = pd.read_csv(DATA_DIR / f"{case_id}_output.csv")
    model = VTSStagedModel()
    load_checkpoints(model)
    model.reset()

    predictions = {name: [] for name in FINAL_OUTPUTS}
    for row in frame.to_dict(orient="records"):
        result = model.step(row)
        for name in FINAL_OUTPUTS:
            predictions[name].append(int(result[name]))

    case_result = {"case_id": case_id, "rows": len(frame), "outputs": {}}
    for name in FINAL_OUTPUTS:
        expected = frame[name].to_numpy(dtype=int)
        actual = predictions[name]
        tp = int(((expected == 1) & (np.asarray(actual) == 1)).sum())
        tn = int(((expected == 0) & (np.asarray(actual) == 0)).sum())
        fp = int(((expected == 0) & (np.asarray(actual) == 1)).sum())
        fn = int(((expected == 1) & (np.asarray(actual) == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        mismatches = [
            {
                "row": index,
                "time": float(frame.iloc[index]["time"]),
                "expected": int(expected[index]),
                "actual": int(actual[index]),
            }
            for index in range(len(frame))
            if int(expected[index]) != int(actual[index])
        ]
        case_result["outputs"][name] = {
            "correct": len(frame) - len(mismatches),
            "accuracy": (len(frame) - len(mismatches)) / len(frame),
            "mismatch_count": len(mismatches),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "mismatches": mismatches,
        }
    return case_result


def main() -> None:
    training_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    validation_cases = training_metrics["validation_cases"]
    results = [evaluate_case(case_id) for case_id in validation_cases]

    summary = {}
    for name in FINAL_OUTPUTS:
        total = sum(result["rows"] for result in results)
        correct = sum(result["outputs"][name]["correct"] for result in results)
        confusion = {
            key: sum(result["outputs"][name]["confusion_matrix"][key] for result in results)
            for key in ("tn", "fp", "fn", "tp")
        }
        precision = confusion["tp"] / (confusion["tp"] + confusion["fp"]) if confusion["tp"] + confusion["fp"] else 0.0
        recall = confusion["tp"] / (confusion["tp"] + confusion["fn"]) if confusion["tp"] + confusion["fn"] else 0.0
        summary[name] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "mismatch_count": total - correct,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "confusion_matrix": confusion,
        }

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    output = {
        "checkpoints": {
            "stage1": "stage1_corrected_best.pt",
            "stage2": "stage2_best.pt",
            "stage3": "stage3_best.pt",
            "stage4": "stage4_best.pt",
            "stage5": "stage5_best.pt",
            "stage6": "stage6_best.pt",
        },
        "validation_cases": validation_cases,
        "summary": summary,
        "baseline_summary": baseline["summary"],
        "cases": results,
    }
    RESULT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for name, metrics in summary.items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.6f} "
            f"mismatches={metrics['mismatch_count']}"
        )
    print(f"results={RESULT_PATH}")


if __name__ == "__main__":
    main()
