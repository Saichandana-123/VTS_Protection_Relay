from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.stage1_model import STAGE1_INPUTS, STAGE1_OUTPUTS
from src.models.stage2_model import STAGE2_INPUTS, STAGE2_OUTPUTS
from src.models.stage3_model import STAGE3_INPUTS, STAGE3_OUTPUTS
from src.models.stage4_model import STAGE4_INPUTS, STAGE4_OUTPUTS
from src.models.stage5_model import STAGE5_INPUTS, STAGE5_OUTPUTS
from src.models.stage6_model import STAGE6_INPUTS, STAGE6_OUTPUTS
from src.models.vts_staged_model import VTSStagedModel

MODEL_DIR = ROOT / "models" / "trained"
DATA_DIR = ROOT / "data" / "simulink_dataset" / "outputs"
OUTPUT_DIR = ROOT / "models" / "experimental" / "threshold_analysis"
TRAINING_METRICS = MODEL_DIR / "training_metrics.json"
BASELINE_PATH = MODEL_DIR / "integrated_evaluation_post_sync.json"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
FINAL_OUTPUTS = tuple(STAGE6_OUTPUTS)
STAGES = (
    ("stage1", STAGE1_INPUTS, STAGE1_OUTPUTS),
    ("stage2", STAGE2_INPUTS, STAGE2_OUTPUTS),
    ("stage3", STAGE3_INPUTS, STAGE3_OUTPUTS),
    ("stage4", STAGE4_INPUTS, STAGE4_OUTPUTS),
    ("stage5", STAGE5_INPUTS, STAGE5_OUTPUTS),
    ("stage6", STAGE6_INPUTS, STAGE6_OUTPUTS),
)


def load_models() -> VTSStagedModel:
    model = VTSStagedModel()
    for name, _, _ in STAGES:
        filename = "stage1_corrected_best.pt" if name == "stage1" else f"{name}_best.pt"
        checkpoint = torch.load(
            MODEL_DIR / filename,
            map_location="cpu",
            weights_only=True,
        )
        getattr(model, name).load_state_dict(checkpoint["model_state_dict"])
        getattr(model, name).eval()
    return model


def predict_stage(
    model: torch.nn.Module,
    input_names: tuple[str, ...] | list[str],
    output_names: tuple[str, ...] | list[str],
    values: dict[str, object],
    threshold: float,
) -> tuple[dict[str, int], dict[str, float]]:
    tensor = torch.tensor(
        [[float(values[name]) for name in input_names]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor))[0].tolist()
    binary = [int(probability >= threshold) for probability in probabilities]
    return (
        dict(zip(output_names, binary)),
        dict(zip(output_names, map(float, probabilities))),
    )


def probability_step(
    model: VTSStagedModel,
    inputs: dict[str, object],
    threshold: float,
) -> tuple[dict[str, int], dict[str, float]]:
    if "MCB_VTS_OPTO" not in inputs and "MCB_VTS" in inputs:
        inputs = {**inputs, "MCB_VTS_OPTO": inputs["MCB_VTS"]}

    detector_aliases = {
        "IA_GT": "IA_OK",
        "IB_GT": "IB_OK",
        "IC_GT": "IC_OK",
        "VAB_GT": "VAB_OK",
        "VBC_GT": "VBC_OK",
        "DELTA_IA_GT": "DELTA_IA_OK",
        "DELTA_IB_GT": "DELTA_IB_OK",
        "DELTA_IC_GT": "DELTA_IC_OK",
    }
    inputs = {
        **inputs,
        **{
            alias: inputs[source]
            for alias, source in detector_aliases.items()
            if alias not in inputs and source in inputs
        },
    }
    if "tVTS" not in inputs:
        raise KeyError("Missing setting-dependent timer input: tVTS")
    if model.tvts_timer is None:
        model._create_setting_timer(inputs["tVTS"])

    s1, p1 = predict_stage(model.stage1, STAGE1_INPUTS, STAGE1_OUTPUTS, inputs, threshold)
    timer240_out = model.timer240.update(bool(inputs["ANY_POLE_DEAD"]))
    stage2_inputs = {**inputs, **s1, "240_TIMER_OUT": float(timer240_out)}
    s2, p2 = predict_stage(model.stage2, STAGE2_INPUTS, STAGE2_OUTPUTS, stage2_inputs, threshold)

    flip_flop_inputs = {
        "sr1": {
            "set_signal": bool(s2["5_AND_OUT"]),
            "reset_signal": not bool(s2["4_OR_OUT"]),
        }
    }
    sr1_out = model.sr1.next_state(**flip_flop_inputs["sr1"])
    stage3_inputs = {**inputs, **s1, **s2, "SR_1_OUT": float(sr1_out)}
    s3, p3 = predict_stage(model.stage3, STAGE3_INPUTS, STAGE3_OUTPUTS, stage3_inputs, threshold)

    flip_flop_inputs["sr2"] = {
        "set_signal": bool(s3["9_AND_OUT"]),
        "reset_signal": not bool(inputs["ANY_VOLTAGE_DEPENDENT_FUNCTION"]),
    }
    sr2_out = model.sr2.next_state(**flip_flop_inputs["sr2"])
    stage4_inputs = {
        **inputs,
        **s1,
        **s2,
        **s3,
        "SR_2_OUT": float(sr2_out),
        "240_TIMER_OUT": float(timer240_out),
    }
    s4, p4 = predict_stage(model.stage4, STAGE4_INPUTS, STAGE4_OUTPUTS, stage4_inputs, threshold)

    model.tvts_timer.delay_s = float(inputs["tVTS"])
    tvts_timer_out = model.tvts_timer.update(bool(s3["6_AND_OUT"]))
    if model.timer20 is None:
        model._create_20ms_timer()
    timer20_out = model.timer20.update(bool(s4["10_AND_OUT"]))

    flip_flop_inputs["sr3"] = {
        "set_signal": bool(tvts_timer_out),
        "reset_signal": bool(s3["7_AND_OUT"]),
    }
    sr3_out = model.sr3.next_state(**flip_flop_inputs["sr3"])
    stage5_inputs = {
        **inputs,
        **s1,
        **s2,
        **s3,
        **s4,
        "SR_3_OUT": float(sr3_out),
        "VTS_TIMER_OUT": float(tvts_timer_out),
        "20_TIMER_OUT": float(timer20_out),
    }
    s5, p5 = predict_stage(model.stage5, STAGE5_INPUTS, STAGE5_OUTPUTS, stage5_inputs, threshold)

    flip_flop_inputs["sr4"] = {
        "set_signal": bool(s5["7_OR_OUT"]),
        "reset_signal": bool(s3["7_AND_OUT"]),
    }
    sr4_out = model.sr4.next_state(**flip_flop_inputs["sr4"])
    stage6_inputs = {**inputs, **s5, "SR_4_OUT": float(sr4_out)}

    for name, flip_flop in (
        ("sr1", model.sr1),
        ("sr2", model.sr2),
        ("sr3", model.sr3),
        ("sr4", model.sr4),
    ):
        flip_flop.update(**flip_flop_inputs[name])

    s6, p6 = predict_stage(model.stage6, STAGE6_INPUTS, STAGE6_OUTPUTS, stage6_inputs, threshold)
    return (
        {**s1, **s2, **s3, **s4, **s5, **s6},
        {**p1, **p2, **p3, **p4, **p5, **p6},
    )


def metric(expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    tp = int(((expected == 1) & (actual == 1)).sum())
    tn = int(((expected == 0) & (actual == 0)).sum())
    fp = int(((expected == 0) & (actual == 1)).sum())
    fn = int(((expected == 1) & (actual == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return {
        "accuracy": float((expected == actual).mean()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def run_threshold(
    threshold: float,
    validation_cases: list[str],
    save_probabilities: bool,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model = load_models()
    all_probabilities: dict[str, list[float]] = defaultdict(list)
    stage_mismatches = {name: 0 for name, _, _ in STAGES}
    first_divergence: dict[str, dict[str, object]] = {}
    case_results = []

    for case_id in validation_cases:
        frame = pd.read_csv(DATA_DIR / f"{case_id}_output.csv")
        model.reset()
        predictions = {name: [] for name in FINAL_OUTPUTS}
        case_first = {}

        for row_index, row in enumerate(frame.to_dict(orient="records")):
            actual, probabilities = probability_step(model, row, threshold)
            for name, _, outputs in STAGES:
                for output in outputs:
                    all_probabilities[output].append(probabilities[output])
                    if actual[output] != int(row[output]):
                        stage_mismatches[name] += 1
                        case_first.setdefault(
                            name,
                            {
                                "row": row_index,
                                "time": float(row["time"]),
                                "signal": output,
                                "expected": int(row[output]),
                                "actual": actual[output],
                                "probability": probabilities[output],
                            },
                        )
            for name in FINAL_OUTPUTS:
                predictions[name].append(actual[name])

        case_outputs = {}
        for name in FINAL_OUTPUTS:
            case_outputs[name] = metric(
                frame[name].to_numpy(dtype=int),
                np.asarray(predictions[name], dtype=int),
            )
        case_results.append({"case_id": case_id, "rows": len(frame), "outputs": case_outputs})
        for stage, error in case_first.items():
            first_divergence.setdefault(stage, {"case_id": case_id, **error})

    summary = {}
    for name in FINAL_OUTPUTS:
        confusion = {
            key: sum(case["outputs"][name]["confusion_matrix"][key] for case in case_results)
            for key in ("tn", "fp", "fn", "tp")
        }
        tn, fp, fn, tp = (confusion[key] for key in ("tn", "fp", "fn", "tp"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fnr = fn / (fn + tp) if fn + tp else 0.0
        total = tn + tp + fn + fp
        summary[name] = {
            "correct": tn + tp,
            "total": total,
            "accuracy": (tn + tp) / total,
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

    result = {
        "threshold": threshold,
        "summary": summary,
        "stage_mismatch_counts": stage_mismatches,
        "first_divergence": first_divergence,
        "cases": case_results,
    }
    arrays = {name: np.asarray(values, dtype=np.float32) for name, values in all_probabilities.items()}
    return result, arrays if save_probabilities else {}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validation_cases = json.loads(TRAINING_METRICS.read_text(encoding="utf-8"))["validation_cases"]
    results = {}
    baseline_probabilities = {}
    for threshold in THRESHOLDS:
        result, probabilities = run_threshold(
            threshold,
            validation_cases,
            save_probabilities=threshold == 0.5,
        )
        key = f"{threshold:.1f}"
        results[key] = result
        if probabilities:
            baseline_probabilities = probabilities
        print(
            f"threshold={key} "
            + " ".join(
                f"{name}_recall={result['summary'][name]['recall']:.6f}"
                for name in FINAL_OUTPUTS
            )
        )

    np.savez_compressed(
        OUTPUT_DIR / "sigmoid_probabilities_threshold_0.5.npz",
        **baseline_probabilities,
    )
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    output = {
        "diagnostic": "sequential_probability_threshold_analysis",
        "thresholds": list(THRESHOLDS),
        "threshold_policy": "one global threshold applied to every stage output",
        "validation_cases": validation_cases,
        "probabilities_artifact": "sigmoid_probabilities_threshold_0.5.npz",
        "results": results,
        "post_sync_baseline_summary": baseline["summary"],
        "notes": [
            "Thresholds were exploratory and were not selected using validation labels.",
            "All threshold runs used the current synchronized VTSStagedModel state order.",
            "The 0.5 probability artifact contains raw sigmoid probabilities before binary thresholding.",
        ],
    }
    result_path = OUTPUT_DIR / "threshold_analysis.json"
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
