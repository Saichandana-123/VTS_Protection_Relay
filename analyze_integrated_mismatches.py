from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.stage1_model import Stage1MLP
from src.models.stage2_model import Stage2MLP
from src.models.stage3_model import Stage3MLP
from src.models.stage4_model import Stage4MLP
from src.models.stage5_model import Stage5MLP
from src.models.stage6_model import Stage6MLP
from src.models.vts_staged_model import VTSStagedModel


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "models" / "trained"
DATA_DIR = ROOT / "data" / "simulink_dataset" / "outputs"
METRICS_PATH = OUTPUT_DIR / "training_metrics.json"
REPORT_PATH = OUTPUT_DIR / "integrated_root_cause_analysis.json"

STAGE_SPECS = {
    "stage1": (
        Stage1MLP,
        ("IA_OK", "IB_OK", "IC_OK", "DELTA_IA_OK", "DELTA_IB_OK",
         "DELTA_IC_OK", "VTS_MANRESET", "VTS_AUTORESET", "VAB_OK",
         "VBC_OK", "VTS_BLOCKING", "ACCELERATE_IND"),
        ("1_OR_OUT", "2_OR_OUT", "3_OR_OUT", "1_AND_OUT", "2_AND_OUT"),
    ),
    "stage2": (
        Stage2MLP,
        ("V2", "I2", "240_TIMER_OUT", "1_AND_OUT", "2_OR_OUT"),
        ("3_AND_OUT", "4_AND_OUT", "4_OR_OUT", "5_AND_OUT"),
    ),
    "stage3": (
        Stage3MLP,
        ("ALL_POLE_DEAD", "1_OR_OUT", "4_OR_OUT", "4_AND_OUT", "SR_1_OUT",
         "3_OR_OUT", "ANY_VOLTAGE_DEPENDENT_FUNCTION"),
        ("6_AND_OUT", "7_AND_OUT", "8_AND_OUT", "9_AND_OUT"),
    ),
    "stage4": (
        Stage4MLP,
        ("SR_2_OUT", "6_AND_OUT", "240_TIMER_OUT", "2_AND_OUT"),
        ("10_AND_OUT",),
    ),
    "stage5": (
        Stage5MLP,
        ("SR_3_OUT", "MCB_VTS", "8_AND_OUT", "VTS_TIMER_OUT", "20_TIMER_OUT"),
        ("5_OR_OUT", "6_OR_OUT", "7_OR_OUT"),
    ),
    "stage6": (
        Stage6MLP,
        ("5_OR_OUT", "6_OR_OUT", "VTS_BLOCKING", "MCB_VTS", "SR_4_OUT"),
        ("SLOW_BLOCK", "FAST_BLOCK", "INDICATION"),
    ),
}


def load_model(name: str) -> torch.nn.Module:
    model_class, _, _ = STAGE_SPECS[name]
    model = model_class()
    checkpoint = torch.load(
        OUTPUT_DIR / f"{name}_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(model: torch.nn.Module, names: tuple[str, ...], values: dict) -> dict[str, int]:
    tensor = torch.tensor([[float(values[name]) for name in names]], dtype=torch.float32)
    with torch.no_grad():
        result = (torch.sigmoid(model(tensor)) >= 0.5).to(torch.int64)[0].tolist()
    return dict(zip(STAGE_SPECS[next(key for key, spec in STAGE_SPECS.items() if spec[0] is type(model))][2], result))


def stage_prediction(
    name: str, model: torch.nn.Module, values: dict[str, float],
) -> dict[str, int]:
    _, inputs, outputs = STAGE_SPECS[name]
    tensor = torch.tensor([[float(values[column]) for column in inputs]], dtype=torch.float32)
    with torch.no_grad():
        binary = (torch.sigmoid(model(tensor)) >= 0.5).to(torch.int64)[0].tolist()
    return dict(zip(outputs, binary))


def metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    tp = int(((expected == 1) & (actual == 1)).sum())
    tn = int(((expected == 0) & (actual == 0)).sum())
    fp = int(((expected == 0) & (actual == 1)).sum())
    fn = int(((expected == 1) & (actual == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "support_0": int((expected == 0).sum()),
        "support_1": int((expected == 1).sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": float((expected == actual).mean()),
    }


def main() -> None:
    training = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    validation_cases = training["validation_cases"]
    models = {name: load_model(name) for name in STAGE_SPECS}
    stage_counts = Counter()
    stage_first_errors = defaultdict(list)
    end_to_end_counts = Counter()
    transition_records = []
    state_mismatch_counts = Counter()
    all_teacher = defaultdict(lambda: {"expected": [], "actual": []})
    all_inference = defaultdict(lambda: {"expected": [], "actual": []})
    case_summary = []
    mapping_difference = Counter()

    for case_id in validation_cases:
        frame = pd.read_csv(DATA_DIR / f"{case_id}_output.csv")
        model = VTSStagedModel()
        for name, loaded in models.items():
            getattr(model, name).load_state_dict(loaded.state_dict())
        model.reset()
        case_first = {}
        case_end = Counter()
        previous = {}

        for row_index, row in frame.iterrows():
            row_values = row.to_dict()
            reference = {
                name: int(row_values[name])
                for name in row.index
                if name not in ("case_id", "scenario")
            }

            # Training used VTS_BLOCKING for this feature, while inference
            # supplies NOT_VTS_BLOCKING to the Stage 1 architecture.
            if float(row_values["VTS_BLOCKING"]) != 1.0 - float(row_values["VTS_BLOCKING"]):
                mapping_difference["stage1_blocking_polarity_rows"] += 1

            for name in ("stage1", "stage2", "stage3", "stage4", "stage5", "stage6"):
                prediction = stage_prediction(name, models[name], row_values)
                _, _, outputs = STAGE_SPECS[name]
                for output in outputs:
                    all_teacher[output]["expected"].append(reference[output])
                    all_teacher[output]["actual"].append(prediction[output])

            result = model.step(row_values)
            for name in ("stage1", "stage2", "stage3", "stage4", "stage5", "stage6"):
                _, _, outputs = STAGE_SPECS[name]
                for output in outputs:
                    expected_value = reference[output]
                    actual_value = int(result[output])
                    stage_counts[f"{name}_mismatches"] += actual_value != expected_value
                    if actual_value != expected_value and name not in case_first:
                        case_first[name] = {
                            "row": row_index,
                            "time": float(row_values["time"]),
                            "signal": output,
                            "expected": expected_value,
                            "actual": actual_value,
                        }
                    all_inference[output]["expected"].append(expected_value)
                    all_inference[output]["actual"].append(actual_value)

            for output in ("SLOW_BLOCK", "FAST_BLOCK", "INDICATION"):
                if int(result[output]) != reference[output]:
                    end_to_end_counts[output] += 1
                    case_end[output] += 1

            state = {
                name: int(result[name])
                for name in ("240_TIMER_OUT", "20_TIMER_OUT", "tVTS_TIMER_OUT",
                             "SR_1_OUT", "SR_2_OUT", "SR_3_OUT", "SR_4_OUT")
            }
            reference_state = {
                "240_TIMER_OUT": int(row_values["240_TIMER_OUT"]),
                "20_TIMER_OUT": int(row_values["20_TIMER_OUT"]),
                "tVTS_TIMER_OUT": int(row_values["VTS_TIMER_OUT"]),
                "SR_1_OUT": int(row_values["SR_1_OUT"]),
                "SR_2_OUT": int(row_values["SR_2_OUT"]),
                "SR_3_OUT": int(row_values["SR_3_OUT"]),
                "SR_4_OUT": int(row_values["SR_4_OUT"]),
            }
            for name in state:
                if state[name] != reference_state[name]:
                    state_mismatch_counts[name] += 1
            if state != previous:
                transition_records.append({
                    "case_id": case_id,
                    "row": row_index,
                    "time": float(row_values["time"]),
                    "state": state,
                    "reference": reference_state,
                })
                previous = state

        case_summary.append({
            "case_id": case_id,
            "rows": len(frame),
            "final_output_mismatches": dict(case_end),
            "first_stage_errors": case_first,
        })

    teacher_metrics = {
        output: metrics(
            np.asarray(values["expected"]),
            np.asarray(values["actual"]),
        )
        for output, values in all_teacher.items()
    }
    inference_metrics = {
        output: metrics(
            np.asarray(values["expected"]),
            np.asarray(values["actual"]),
        )
        for output, values in all_inference.items()
    }
    report = {
        "validation_cases": validation_cases,
        "mapping_difference": dict(mapping_difference),
        "teacher_forced_stage_metrics": teacher_metrics,
        "sequential_inference_stage_metrics": inference_metrics,
        "case_summary": case_summary,
        "stage_mismatch_counts": dict(stage_counts),
        "state_mismatch_counts": dict(state_mismatch_counts),
        "end_to_end_mismatch_counts": dict(end_to_end_counts),
        "state_transition_records": transition_records,
        "execution_order": [
            "stage1", "timer240", "stage2", "sr1", "stage3", "sr2",
            "stage4", "tvts_timer", "timer20", "sr3", "stage5", "sr4",
            "stage6",
        ],
        "notes": [
            "Teacher-forced predictions use reference CSV values for every stage input independently; no predicted stage output is passed to another teacher-forced stage.",
            "Sequential inference uses VTSStagedModel.step() and therefore uses predicted stage outputs plus deterministic timer and SR state.",
            "This is diagnostic analysis only; no checkpoints or dataset files are written.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={REPORT_PATH}")
    print("teacher_forced:")
    for name in ("SLOW_BLOCK", "FAST_BLOCK", "INDICATION"):
        print(name, teacher_metrics[name])
    print("sequential:")
    for name in ("SLOW_BLOCK", "FAST_BLOCK", "INDICATION"):
        print(name, inference_metrics[name])


if __name__ == "__main__":
    main()
