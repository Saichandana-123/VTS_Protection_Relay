
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
dataset_path = root / "data" / "simulink_dataset" / "vts_training_dataset.csv"
manifest_path = root / "data" / "simulink_dataset" / "inputs" / "manifest.csv"

df = pd.read_csv(dataset_path)
manifest = pd.read_csv(manifest_path)

# Reconstruct case_id from scenario and case ordering in the manifest.
# Each case output contains 13,001 rows.
rows_per_case = 13001

if len(df) != len(manifest) * rows_per_case:
    raise ValueError("Dataset row count does not match manifest case count.")

expected_case_ids = []
expected_scenarios = []

for _, row in manifest.iterrows():
    expected_case_ids.extend([row["case_id"]] * rows_per_case)
    expected_scenarios.extend([row["scenario"]] * rows_per_case)

df["case_id"] = expected_case_ids
df["scenario"] = expected_scenarios

# Add original configuration metadata for every sample.
for col in [
    "VTS_STATUS",
    "VTS_RESET_MODE",
    "VTS_TIME_DELAY",
    "VTS_I_INHIBIT",
    "VTS_I2_INHIBIT",
]:
    values = dict(zip(manifest["case_id"], manifest[col]))
    df[col] = df["case_id"].map(values)

df.to_csv(dataset_path, index=False)

print("Metadata corrected and saved:", dataset_path)
print("Unique cases:", df["case_id"].nunique())
print("Case IDs:", df["case_id"].unique()[:5])
print("\nSettings:")
for col in [
    "VTS_STATUS",
    "VTS_RESET_MODE",
    "VTS_TIME_DELAY",
    "VTS_I_INHIBIT",
    "VTS_I2_INHIBIT",
]:
    print(col, sorted(df[col].unique(), key=str))
