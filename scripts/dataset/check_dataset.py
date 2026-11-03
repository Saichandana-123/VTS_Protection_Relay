
from pathlib import Path
import pandas as pd

folder = Path("data/simulink_dataset/outputs")
files = sorted(folder.glob("case_*_output.csv"))

print("Output files:", len(files))

expected_rows = 13001
bad_files = []

for f in files:
    df = pd.read_csv(f)

    if len(df) != expected_rows:
        bad_files.append((f.name, "wrong row count", len(df)))

    if df.isna().any().any():
        bad_files.append((f.name, "missing values", ""))

    if "time" not in df.columns or abs(df["time"].iloc[-1] - 13.0) > 1e-6:
        bad_files.append((f.name, "wrong time range", ""))

print("Files with issues:", len(bad_files))
for item in bad_files[:20]:
    print(item)

if len(files) == 108 and not bad_files:
    print("PASS: all files have 13001 rows, no missing values, and end at 13 seconds.")

for col in ["SLOW_BLOCK", "FAST_BLOCK", "INDICATION"]:
    total_active = 0
    for f in files:
        df = pd.read_csv(f, usecols=[col])
        total_active += int((df[col] != 0).sum())
    print(f"{col}: {total_active} active samples across all files")
