from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_integrated_mismatches import STAGE_SPECS, load_model, metrics


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "models" / "trained"
DATA_DIR = ROOT / "data" / "simulink_dataset" / "outputs"
METRICS_PATH = OUTPUT_DIR / "training_metrics.json"
REPORT_PATH = OUTPUT_DIR / "teacher_forced_analysis.json"


def main() -> None:
    training = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    models = {name: load_model(name) for name in STAGE_SPECS}
    output = {}

    for stage_name, (_, inputs, targets) in STAGE_SPECS.items():
        expected_parts = []
        actual_parts = []
        for case_id in training["validation_cases"]:
            frame = pd.read_csv(
                DATA_DIR / f"{case_id}_output.csv",
                usecols=[*inputs, *targets],
            )
            x = torch.from_numpy(frame[list(inputs)].to_numpy(dtype=np.float32))
            with torch.no_grad():
                predicted = (
                    torch.sigmoid(models[stage_name](x)) >= 0.5
                ).to(torch.int64).numpy()
            expected_parts.append(frame[list(targets)].to_numpy(dtype=np.int64))
            actual_parts.append(predicted)
        expected = np.concatenate(expected_parts)
        actual = np.concatenate(actual_parts)
        output[stage_name] = {
            target: metrics(expected[:, index], actual[:, index])
            for index, target in enumerate(targets)
        }

    report = {
        "validation_cases": training["validation_cases"],
        "method": "Each stage receives only its reference CSV inputs; no predicted upstream stage output is passed to another stage.",
        "stages": output,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={REPORT_PATH}")
    for stage, values in output.items():
        print(stage, {name: round(item["accuracy"], 6) for name, item in values.items()})


if __name__ == "__main__":
    main()
