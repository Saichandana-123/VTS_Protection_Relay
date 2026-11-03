
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "simulink_dataset" / "inputs"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.001
STOP_TIME = 13.0
TIME = np.round(np.arange(0, STOP_TIME + DT / 2, DT), 6)
# Twelve representative configurations, within the ranges identified
# from the P642 manual. These are not exhaustive.
SETTING_CONFIGS = [
    {"VTS_STATUS": 0, "VTS_RESET_MODE": "Manual", "VTS_TIME_DELAY": 1.0, "VTS_I_INHIBIT": 0.08, "VTS_I2_INHIBIT": 0.05},
    {"VTS_STATUS": 1, "VTS_RESET_MODE": "Auto",   "VTS_TIME_DELAY": 2.0, "VTS_I_INHIBIT": 0.5,  "VTS_I2_INHIBIT": 0.10},
    {"VTS_STATUS": 2, "VTS_RESET_MODE": "Auto",   "VTS_TIME_DELAY": 3.0, "VTS_I_INHIBIT": 1.0,  "VTS_I2_INHIBIT": 0.20},
    {"VTS_STATUS": 2, "VTS_RESET_MODE": "Manual", "VTS_TIME_DELAY": 5.0, "VTS_I_INHIBIT": 2.0,  "VTS_I2_INHIBIT": 0.30},
    {"VTS_STATUS": 1, "VTS_RESET_MODE": "Manual", "VTS_TIME_DELAY": 10.0, "VTS_I_INHIBIT": 4.0, "VTS_I2_INHIBIT": 0.50},
    {"VTS_STATUS": 0, "VTS_RESET_MODE": "Auto",   "VTS_TIME_DELAY": 1.0, "VTS_I_INHIBIT": 8.0,  "VTS_I2_INHIBIT": 0.15},
    {"VTS_STATUS": 2, "VTS_RESET_MODE": "Auto",   "VTS_TIME_DELAY": 2.0, "VTS_I_INHIBIT": 16.0, "VTS_I2_INHIBIT": 0.05},
    {"VTS_STATUS": 1, "VTS_RESET_MODE": "Auto",   "VTS_TIME_DELAY": 5.0, "VTS_I_INHIBIT": 32.0, "VTS_I2_INHIBIT": 0.40},
    {"VTS_STATUS": 2, "VTS_RESET_MODE": "Manual", "VTS_TIME_DELAY": 10.0, "VTS_I_INHIBIT": 0.08, "VTS_I2_INHIBIT": 0.50},
    {"VTS_STATUS": 0, "VTS_RESET_MODE": "Manual", "VTS_TIME_DELAY": 3.0, "VTS_I_INHIBIT": 1.0,  "VTS_I2_INHIBIT": 0.10},
    {"VTS_STATUS": 1, "VTS_RESET_MODE": "Auto",   "VTS_TIME_DELAY": 7.0, "VTS_I_INHIBIT": 2.0,  "VTS_I2_INHIBIT": 0.25},
    {"VTS_STATUS": 2, "VTS_RESET_MODE": "Auto",   "VTS_TIME_DELAY": 8.0, "VTS_I_INHIBIT": 32.0, "VTS_I2_INHIBIT": 0.50},
]

# These names and input order must match the Simulink root Inports.
SIGNALS = [
    "VAB", "VBC", "V2", "I2", "I2_SETTING",
    "delta_IA", "delta_IB", "delta_IC", "In",
    "ANY_POLE_DEAD", "MCB_VTS", "VTS_BLOCKING",
    "VTS_AUTORESET", "VTS_MANRESET",
    "ANY_VOLTAGE_DEPENDENT_FUNCTION", "ACCELERATE_IND",
    "tVTS", "IA", "IB", "IC", "I_SETTING", "ALL_POLE_DEAD",
]

SCENARIOS = [
    "healthy",
    "normal_load_unbalance",
    "three_phase_vt_loss_energization",
    "mcb_vts_operation",
    "negative_sequence_fault",
    "three_phase_fault",
    "two_phase_vt_loss",
    "three_phase_vt_loss_under_load",
    "one_phase_vt_loss",
]


def make_case(scenario, settings):
    """Create one ordered time-series test case.

    Values are initial engineering test profiles, not a claim that
    every profile fully reproduces the physical relay condition.
    Confirm the scenario definitions against the manual and circuit.
    """
    n = len(TIME)
    d = {name: np.zeros(n, dtype=float) for name in SIGNALS}

    # Healthy nominal baseline. Voltages use the 100/120 V VT
    # scale used in the previously identified detector thresholds.
    for name in ("VAB", "VBC"):
        d[name][:] = 110.0

    d["In"][:] = 1.0
    d["IA"][:] = 0.5
    d["IB"][:] = 0.5
    d["IC"][:] = 0.5

    status = settings["VTS_STATUS"]
    reset_mode = settings["VTS_RESET_MODE"]

    d["VTS_BLOCKING"][:] = float(status == 2)
    d["VTS_AUTORESET"][:] = float(reset_mode == "Auto")
    d["VTS_MANRESET"][:] = 0.0

    d["tVTS"][:] = settings["VTS_TIME_DELAY"]
    d["I_SETTING"][:] = settings["VTS_I_INHIBIT"]
    d["I2_SETTING"][:] = settings["VTS_I2_INHIBIT"]

# Keep the disturbance active long enough to test tVTS up to 10 s.
# Start at 1 s and restore at 12 s.
    active = (TIME >= 1.0) & (TIME < 12.0)

    if scenario == "healthy":
        pass

    elif scenario == "normal_load_unbalance":
        d["IA"][active] = 0.65
        d["IB"][active] = 0.45
        d["IC"][active] = 0.55

    elif scenario == "three_phase_vt_loss_energization":
        # Voltage loss during energization; phase currents start low.
        d["VAB"][active] = 5.0
        d["VBC"][active] = 5.0
        d["V2"][active] = 0.0
        for name in ("IA", "IB", "IC"):
            d[name][TIME < 1.0] = 0.05
            d[name][active] = 0.5

    elif scenario == "mcb_vts_operation":
        d["MCB_VTS"][active] = 1.0

    elif scenario == "negative_sequence_fault":
        d["I2"][active] = 0.15
        d["V2"][active] = 15.0

    elif scenario == "three_phase_fault":
        for name in ("IA", "IB", "IC"):
            d[name][active] = 2.5

    elif scenario == "two_phase_vt_loss":
        d["VAB"][active] = 5.0
        d["VBC"][active] = 5.0
        d["V2"][active] = 15.0

    elif scenario == "three_phase_vt_loss_under_load":
        d["VAB"][active] = 5.0
        d["VBC"][active] = 5.0
        for name in ("IA", "IB", "IC"):
            d[name][active] = 0.8

    elif scenario == "one_phase_vt_loss":
        # Initial proxy: one monitored line voltage drops.
        # Verify the intended phase mapping from the P642 circuit.
        d["VAB"][active] = 5.0

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # These features are only used if the scenario definition calls
    # for a superimposed-current event. Keep them at zero otherwise.
    if scenario in {
        "three_phase_fault",
        "negative_sequence_fault",
    }:
        for name in ("delta_IA", "delta_IB", "delta_IC"):
            d[name][active] = 0.2

    # Time-varying control signals remain explicitly represented.
    # The relay's own model handles the timing and state transitions.
    d["ANY_POLE_DEAD"][:] = 0.0
    d["ALL_POLE_DEAD"][:] = 0.0
    d["ANY_VOLTAGE_DEPENDENT_FUNCTION"][:] = 0.0
    d["ACCELERATE_IND"][:] = 0.0

    frame = pd.DataFrame(d, columns=SIGNALS)
    frame.insert(0, "time", TIME)
    return frame


manifest = []
case_number = 0

for scenario, settings in itertools.product(SCENARIOS, SETTING_CONFIGS):
    case_number += 1
    case_id = f"case_{case_number:03d}"
    filename = f"{case_id}.csv"

    frame = make_case(scenario, settings)
    frame.to_csv(INPUT_DIR / filename, index=False)

    manifest.append({
        "case_id": case_id,
        "filename": filename,
        "scenario": scenario,
        **settings,
        "sample_count": len(frame),
    })

manifest_path = INPUT_DIR / "manifest.csv"
pd.DataFrame(manifest).to_csv(manifest_path, index=False)

print(f"Generated {case_number} input cases.")
print(f"Samples per case: {len(TIME)}")
print(f"Total input samples: {case_number * len(TIME):,}")
print(f"Inputs: {INPUT_DIR}")
print(f"Manifest: {manifest_path}")
