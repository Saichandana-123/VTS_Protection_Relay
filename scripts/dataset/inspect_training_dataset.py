from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
path = root / "data" / "simulink_dataset" / "vts_training_dataset.csv"

df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nFirst 10 case IDs:")
print(df["case_id"].drop_duplicates().head(10).to_list())

print("\nRows per case:")
print(df.groupby("case_id").size().describe())

print("\nMissing values:", int(df.isna().sum().sum()))

print("\nOutput values:")
for col in ["SLOW_BLOCK", "FAST_BLOCK", "INDICATION"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).sort_index())

print("\nSettings combinations:", df[
    ["VTS_STATUS", "VTS_RESET_MODE", "VTS_TIME_DELAY",
     "VTS_I_INHIBIT", "VTS_I2_INHIBIT"]
].drop_duplicates().shape[0])