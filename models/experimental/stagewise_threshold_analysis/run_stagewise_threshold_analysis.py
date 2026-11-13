from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.training.run_threshold_analysis import (  # noqa: E402
    BASELINE_PATH,
    DATA_DIR,
    FINAL_OUTPUTS,
    MODEL_DIR,
    STAGES,
    TRAINING_METRICS,
    load_models,
    metric,
    predict_stage,
)

OUTPUT_DIR = Path(__file__).resolve().parent
STAGE_NAMES = tuple(name for name, _, _ in STAGES)
CONFIGURATIONS = {
    "baseline_all_0.5": {name: 0.5 for name in STAGE_NAMES},
    "stage1_only_0.6": {name: 0.6 if name == "stage1" else 0.5 for name in STAGE_NAMES},
    "stage2_only_0.6": {name: 0.6 if name == "stage2" else 0.5 for name in STAGE_NAMES},
    "stage3_only_0.6": {name: 0.6 if name == "stage3" else 0.5 for name in STAGE_NAMES},
    "stage4_only_0.6": {name: 0.6 if name == "stage4" else 0.5 for name in STAGE_NAMES},
    "stage5_only_0.6": {name: 0.6 if name == "stage5" else 0.5 for name in STAGE_NAMES},
    "stage6_only_0.6": {name: 0.6 if name == "stage6" else 0.5 for name in STAGE_NAMES},
    "all_0.6": {name: 0.6 for name in STAGE_NAMES},
}


def stagewise_step(model, inputs: dict[str, object], thresholds: dict[str, float]):
    if "MCB_VTS_OPTO" not in inputs and "MCB_VTS" in inputs:
        inputs = {**inputs, "MCB_VTS_OPTO": inputs["MCB_VTS"]}
    detector_aliases = {
        "IA_GT": "IA_OK", "IB_GT": "IB_OK", "IC_GT": "IC_OK",
        "VAB_GT": "VAB_OK", "VBC_GT": "VBC_OK",
        "DELTA_IA_GT": "DELTA_IA_OK", "DELTA_IB_GT": "DELTA_IB_OK",
        "DELTA_IC_GT": "DELTA_IC_OK",
    }
    inputs = {
        **inputs,
        **{alias: inputs[source] for alias, source in detector_aliases.items()
           if alias not in inputs and source in inputs},
    }
    if "tVTS" not in inputs:
        raise KeyError("Missing setting-dependent timer input: tVTS")
    if model.tvts_timer is None:
        model._create_setting_timer(inputs["tVTS"])

    def run(name, values):
        _, input_names, output_names = next(item for item in STAGES if item[0] == name)
        return predict_stage(
            getattr(model, name), input_names, output_names, values, thresholds[name]
        )

    s1, _ = run("stage1", inputs)
    timer240_out = model.timer240.update(bool(inputs["ANY_POLE_DEAD"]))
    s2, _ = run("stage2", {**inputs, **s1, "240_TIMER_OUT": float(timer240_out)})
    sr1 = model.sr1.next_state(
        set_signal=bool(s2["5_AND_OUT"]), reset_signal=not bool(s2["4_OR_OUT"])
    )
    s3, _ = run("stage3", {**inputs, **s1, **s2, "SR_1_OUT": float(sr1)})
    sr2 = model.sr2.next_state(
        set_signal=bool(s3["9_AND_OUT"]),
        reset_signal=not bool(inputs["ANY_VOLTAGE_DEPENDENT_FUNCTION"]),
    )
    s4, _ = run(
        "stage4",
        {**inputs, **s1, **s2, **s3, "SR_2_OUT": float(sr2),
         "240_TIMER_OUT": float(timer240_out)},
    )
    model.tvts_timer.delay_s = float(inputs["tVTS"])
    tvts_timer_out = model.tvts_timer.update(bool(s3["6_AND_OUT"]))
    if model.timer20 is None:
        model._create_20ms_timer()
    timer20_out = model.timer20.update(bool(s4["10_AND_OUT"]))
    sr3 = model.sr3.next_state(
        set_signal=bool(tvts_timer_out), reset_signal=bool(s3["7_AND_OUT"])
    )
    s5, _ = run(
        "stage5",
        {**inputs, **s1, **s2, **s3, **s4, "SR_3_OUT": float(sr3),
         "VTS_TIMER_OUT": float(tvts_timer_out), "20_TIMER_OUT": float(timer20_out)},
    )
    sr4 = model.sr4.next_state(
        set_signal=bool(s5["7_OR_OUT"]), reset_signal=bool(s3["7_AND_OUT"])
    )
    stage_inputs = {"sr1": {"set_signal": bool(s2["5_AND_OUT"]),
                            "reset_signal": not bool(s2["4_OR_OUT"])},
                    "sr2": {"set_signal": bool(s3["9_AND_OUT"]),
                            "reset_signal": not bool(inputs["ANY_VOLTAGE_DEPENDENT_FUNCTION"])},
                    "sr3": {"set_signal": bool(tvts_timer_out),
                            "reset_signal": bool(s3["7_AND_OUT"])},
                    "sr4": {"set_signal": bool(s5["7_OR_OUT"]),
                            "reset_signal": bool(s3["7_AND_OUT"])}}
    for name, flip_flop in (("sr1", model.sr1), ("sr2", model.sr2),
                            ("sr3", model.sr3), ("sr4", model.sr4)):
        flip_flop.update(**stage_inputs[name])
    s6, _ = run("stage6", {**inputs, **s5, "SR_4_OUT": float(sr4)})
    return {**s1, **s2, **s3, **s4, **s5, **s6}


def evaluate(config_name: str, thresholds: dict[str, float], cases: list[str]):
    model = load_models()
    case_results = []
    for case_id in cases:
        frame = pd.read_csv(DATA_DIR / f"{case_id}_output.csv")
        model.reset()
        predictions = {name: [] for name in FINAL_OUTPUTS}
        for row in frame.to_dict(orient="records"):
            actual = stagewise_step(model, row, thresholds)
            for name in FINAL_OUTPUTS:
                predictions[name].append(actual[name])
        case_results.append({
            "case_id": case_id,
            "rows": len(frame),
            "outputs": {
                name: metric(frame[name].to_numpy(dtype=int),
                             np.asarray(predictions[name], dtype=int))
                for name in FINAL_OUTPUTS
            },
        })
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
        total = tn + fp + fn + tp
        summary[name] = {
            "correct": tn + tp, "total": total,
            "accuracy": (tn + tp) / total, "mismatch_count": fp + fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall else 0.0,
            "specificity": specificity,
            "false_positive_rate": fpr, "false_negative_rate": fnr,
            "confusion_matrix": confusion,
        }
    return {"configuration": config_name, "thresholds": thresholds,
            "summary": summary, "cases": case_results}


def write_report(output):
    baseline = output["results"]["baseline_all_0.5"]["summary"]
    lines = [
        "# Stage-wise Threshold Sensitivity Analysis",
        "",
        "These are exploratory validation results on cases `case_088`–`case_108`; "
        "no threshold was selected for deployment or independently validated.",
        "",
        "The synchronized sequential execution order and all existing checkpoints were preserved. "
        "Only external sigmoid decision thresholds were changed.",
        "",
        "## Aggregate results",
        "",
        "| Configuration | Output | Accuracy | Precision | Recall | F1 | FPR | FNR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config, result in output["results"].items():
        for name, values in result["summary"].items():
            lines.append(
                f"| {config} | {name} | {values['accuracy']:.4%} | "
                f"{values['precision']:.4%} | {values['recall']:.4%} | "
                f"{values['f1']:.4%} | {values['false_positive_rate']:.4%} | "
                f"{values['false_negative_rate']:.4%} |"
            )
    lines += ["", "## Recall change versus baseline", "",
              "| Configuration | SLOW_BLOCK | FAST_BLOCK | INDICATION |",
              "|---|---:|---:|---:|"]
    for config, result in output["results"].items():
        deltas = [
            result["summary"][name]["recall"] - baseline[name]["recall"]
            for name in FINAL_OUTPUTS
        ]
        lines.append(f"| {config} | {deltas[0]:+.4%} | {deltas[1]:+.4%} | {deltas[2]:+.4%} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "A stage produces a meaningful difference only when its one-stage threshold "
        "configuration changes the aggregate metrics relative to `baseline_all_0.5`. "
        "The per-case metrics are stored in the JSON artifact for all 21 cases.",
        "",
        "The `all_0.6` result is the previously observed global-threshold diagnostic "
        "configuration. Because configurations were compared on the validation labels, "
        "these results must not be reported as unbiased deployment performance.",
    ]
    (OUTPUT_DIR / "stagewise_threshold_analysis_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = json.loads(TRAINING_METRICS.read_text(encoding="utf-8"))["validation_cases"]
    results = {}
    for name, thresholds in CONFIGURATIONS.items():
        results[name] = evaluate(name, thresholds, cases)
        print(name, *(f"{output}={results[name]['summary'][output]['recall']:.6f}"
                       for output in FINAL_OUTPUTS))
    output = {
        "diagnostic": "stagewise_threshold_sensitivity_analysis",
        "validation_cases": cases,
        "threshold_policy": "one stage at 0.6 or all stages at 0.6; remaining stages at 0.5",
        "checkpoints": {
            "stage1": "stage1_corrected_best.pt",
            "stage2": "stage2_best.pt", "stage3": "stage3_best.pt",
            "stage4": "stage4_best.pt", "stage5": "stage5_best.pt",
            "stage6": "stage6_best.pt",
        },
        "results": results,
        "post_sync_baseline_summary": json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["summary"],
        "notes": [
            "All results are exploratory validation results, not independently validated performance.",
            "No model, checkpoint, dataset, architecture, timer, or SR logic was changed.",
        ],
    }
    result_path = OUTPUT_DIR / "stagewise_threshold_analysis.json"
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_report(output)
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
