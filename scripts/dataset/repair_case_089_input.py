from pathlib import Path
import ast
import csv

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
generator_path = ROOT / "src" / "data_generation" / "generate_simulink_inputs.py"
input_dir = ROOT / "data" / "simulink_dataset" / "inputs"
input_path = input_dir / "case_089.csv"
manifest_path = input_dir / "manifest.csv"

source = generator_path.read_text(encoding="utf-8")
tree = ast.parse(source, filename=str(generator_path))

required_assignments = {"DT", "STOP_TIME", "TIME", "SIGNALS"}
selected_nodes = []

for node in tree.body:
    if isinstance(node, ast.Import):
        imported_modules = {alias.asname or alias.name for alias in node.names}
        if {"np", "numpy"} & imported_modules or {"pd", "pandas"} & imported_modules:
            selected_nodes.append(node)
    elif isinstance(node, ast.ImportFrom):
        selected_nodes.append(node)
    elif isinstance(node, ast.Assign):
        names = {
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        if names & required_assignments:
            selected_nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name == "make_case":
        selected_nodes.append(node)

namespace = {
    "__file__": str(generator_path),
    "__name__": "case_089_input_repair",
}
exec(
    compile(
        ast.Module(body=selected_nodes, type_ignores=[]),
        str(generator_path),
        "exec",
    ),
    namespace,
)

make_case = namespace["make_case"]
expected_manifest_row = (
    "case_089,case_089.csv,three_phase_vt_loss_under_load,"
    "1,Manual,10.0,4.0,0.5,13001"
)

manifest_bytes = manifest_path.read_bytes()
encoding = "utf-8-sig" if manifest_bytes.startswith(b"\xef\xbb\xbf") else "utf-8"
manifest_text = manifest_bytes.decode(encoding)
manifest_lines = manifest_text.splitlines(keepends=True)

matching_indices = [
    index
    for index, line in enumerate(manifest_lines)
    if line.rstrip("\r\n") .startswith("case_089,")
]
if len(matching_indices) != 1:
    raise ValueError(
        f"Expected exactly one case_089 manifest row, found {len(matching_indices)}."
    )

manifest_index = matching_indices[0]
actual_manifest_row = manifest_lines[manifest_index].rstrip("\r\n")
if actual_manifest_row != expected_manifest_row:
    raise ValueError(
        "Existing case_089 manifest row does not match the expected configuration: "
        f"{actual_manifest_row!r}"
    )

scenario = "three_phase_vt_loss_under_load"
settings = {
    "VTS_STATUS": 1,
    "VTS_RESET_MODE": "Manual",
    "VTS_TIME_DELAY": 10.0,
    "VTS_I_INHIBIT": 4.0,
    "VTS_I2_INHIBIT": 0.50,
}

frame = make_case(scenario, settings)
expected_time = np.round(np.arange(0, 13.0 + 0.001 / 2, 0.001), 6)

if frame.shape != (13001, 23):
    raise ValueError(f"Expected a 13001x23 input frame, found {frame.shape}.")
if not np.array_equal(frame["time"].to_numpy(), expected_time):
    raise ValueError("Generated timestamps do not match 0 to 13 seconds at 0.001 s.")

frame.to_csv(input_path, index=False)

line_ending = (
    "\r\n"
    if manifest_lines[manifest_index].endswith("\r\n")
    else "\n"
    if manifest_lines[manifest_index].endswith("\n")
    else ""
)
manifest_lines[manifest_index] = expected_manifest_row + line_ending
manifest_path.write_bytes("".join(manifest_lines).encode(encoding))

print("Repaired case_089.csv and its manifest line.")
print(f"Rows: {len(frame)}; columns: {len(frame.columns)}; time: 0 to 13 seconds")
