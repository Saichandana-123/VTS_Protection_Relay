
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "simulink_dataset" / "outputs"
DESTINATION = ROOT / "data" / "simulink_dataset" / "vts_training_dataset.csv"

files = sorted(OUTPUT_DIR.glob("case_*_output.csv"))

if len(files) != 108:
    raise ValueError(
        f"Expected 108 case output files, found {len(files)}"
    )

frames = []

for file in files:
    df = pd.read_csv(file)

    # Keep each complete waveform identifiable for group-based splitting.
    if "case_id" not in df.columns:
        df["case_id"] = file.stem.replace("_output", "")

    frames.append(df)

dataset = pd.concat(frames, ignore_index=True)

if dataset.isna().any().any():
    raise ValueError("Dataset contains missing values.")

if not {"SLOW_BLOCK", "FAST_BLOCK", "INDICATION"}.issubset(dataset.columns):
    raise ValueError("One or more target columns are missing.")

dataset.to_csv(DESTINATION, index=False)

print("Combined dataset saved:", DESTINATION)
print("Total rows:", len(dataset))
print("Total columns:", len(dataset.columns))
print("Unique cases:", dataset["case_id"].nunique())

print("\nOutput distribution:")
for target in ["SLOW_BLOCK", "FAST_BLOCK", "INDICATION"]:
    print(f"\n{target}:")
    print(dataset[target].value_counts(dropna=False).sort_index())

print("\nScenario counts:")
if "scenario" in dataset.columns:
    print(dataset.groupby("scenario")["case_id"].nunique())
