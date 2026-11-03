from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.models.vts_staged_model import VTSStagedModel


OUTPUT_DIR = ROOT / "models" / "trained"
DATA_DIR = ROOT / "data" / "simulink_dataset" / "outputs"
METRICS_PATH = OUTPUT_DIR / "training_metrics.json"
RESULT_PATH = OUTPUT_DIR / "integrated_evaluation.json"
FINAL_OUTPUTS = ("SLOW_BLOCK", "FAST_BLOCK", "INDICATION")


def load_checkpoints(model: VTSStagedModel) -> None:
    for stage_name in ("stage1", "stage2", "stage3", "stage4", "stage5", "stage6"):
        checkpoint = torch.load(
            OUTPUT_DIR / f"{stage_name}_best.pt",
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
        summary[name] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "mismatch_count": total - correct,
        }

    output = {
        "validation_cases": validation_cases,
        "summary": summary,
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
