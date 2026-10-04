"""
Generate realistic operating scenarios for P642 VTS dataset.

This file creates physically meaningful combinations of:
    - Phase voltages
    - Phase currents
    - Negative sequence quantities
    - Superimposed current
    - Circuit breaker status
    - MCB/VTS input

The scenario label is metadata only.
The actual VTS output will be generated later by the recreated
P642 VTS logic.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SEED = 42
SAMPLES_PER_SCENARIO = 1500

rng = np.random.default_rng(SEED)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "vts_scenario_dataset.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def uniform(low, high):
    return rng.uniform(low, high)


def normal(mean, std, minimum=None, maximum=None):
    value = rng.normal(mean, std)

    if minimum is not None:
        value = max(value, minimum)

    if maximum is not None:
        value = min(value, maximum)

    return value


def base_sample():
    """
    Generate a healthy operating point.
    """

    # Healthy phase voltages
    VAB = normal(110, 3, 100, 120)
    VBC = normal(110, 3, 100, 120)
    VCA = normal(110, 3, 100, 120)

    # Healthy load currents
    IA = normal(80, 15, 10, 150)
    IB = normal(80, 15, 10, 150)
    IC = normal(80, 15, 10, 150)

    # Small normal system unbalance
    V2 = uniform(0, 5)
    I2 = uniform(0, 0.05)

    # Under normal load, superimposed current is approximately zero
    delta_IA = uniform(0, 0.05)
    delta_IB = uniform(0, 0.05)
    delta_IC = uniform(0, 0.05)

    return {
        "VAB": VAB,
        "VBC": VBC,
        "VCA": VCA,

        "IA": IA,
        "IB": IB,
        "IC": IC,

        "V2": V2,
        "I2": I2,

        "delta_IA": delta_IA,
        "delta_IB": delta_IB,
        "delta_IC": delta_IC,

        "CB_CLOSED": 1,
        "ANY_POLE_DEAD": 0,
        "MCB_VTS": 0,

        "scenario": "healthy"
    }


# ============================================================
# Scenario 1: Healthy system
# ============================================================

def healthy():
    return base_sample()


# ============================================================
# Scenario 2: Loss of one phase voltage
# ============================================================

def one_phase_vt_loss():
    sample = base_sample()

    phase = rng.choice(["VAB", "VBC", "VCA"])

    sample[phase] = uniform(0, 10)

    # Loss of one phase creates negative sequence voltage.
    sample["V2"] = uniform(10, 25)

    # No corresponding fault current.
    sample["I2"] = uniform(0, 0.03)

    sample["scenario"] = "one_phase_vt_loss"

    return sample


# ============================================================
# Scenario 3: Loss of two phase voltages
# ============================================================

def two_phase_vt_loss():
    sample = base_sample()

    phases = rng.choice(
        ["VAB", "VBC", "VCA"],
        size=2,
        replace=False
    )

    for phase in phases:
        sample[phase] = uniform(0, 10)

    sample["V2"] = uniform(10, 30)

    # No strong fault current
    sample["I2"] = uniform(0, 0.03)

    sample["scenario"] = "two_phase_vt_loss"

    return sample


# ============================================================
# Scenario 4: Loss of all three phase voltages under load
# ============================================================

def three_phase_vt_loss_under_load():
    sample = base_sample()

    sample["VAB"] = uniform(0, 10)
    sample["VBC"] = uniform(0, 10)
    sample["VCA"] = uniform(0, 10)

    # Important:
    # During pure VT failure under load, current does not
    # disappear along with voltage.
    sample["IA"] = normal(80, 15, 20, 150)
    sample["IB"] = normal(80, 15, 20, 150)
    sample["IC"] = normal(80, 15, 20, 150)

    sample["V2"] = uniform(0, 3)
    sample["I2"] = uniform(0, 0.03)

    # No major change in current
    sample["delta_IA"] = uniform(0, 0.05)
    sample["delta_IB"] = uniform(0, 0.05)
    sample["delta_IC"] = uniform(0, 0.05)

    sample["scenario"] = "three_phase_vt_loss_under_load"

    return sample


# ============================================================
# Scenario 5: Three phase voltage absence during energization
# ============================================================

def three_phase_vt_loss_energization():
    sample = base_sample()

    sample["VAB"] = uniform(0, 10)
    sample["VBC"] = uniform(0, 10)
    sample["VCA"] = uniform(0, 10)

    # Line is being energized
    sample["CB_CLOSED"] = 1
    sample["ANY_POLE_DEAD"] = 0

    # Current can change during energization.
    sample["IA"] = uniform(20, 120)
    sample["IB"] = uniform(20, 120)
    sample["IC"] = uniform(20, 120)

    # No strong fault
    sample["I2"] = uniform(0, 0.05)

    # Current changes are present
    sample["delta_IA"] = uniform(0.1, 1.0)
    sample["delta_IB"] = uniform(0.1, 1.0)
    sample["delta_IC"] = uniform(0.1, 1.0)

    sample["scenario"] = "three_phase_vt_loss_energization"

    return sample


# ============================================================
# Scenario 6: Close-up three-phase fault
# ============================================================

def three_phase_fault():
    sample = base_sample()

    # Voltage collapses
    sample["VAB"] = uniform(0, 10)
    sample["VBC"] = uniform(0, 10)
    sample["VCA"] = uniform(0, 10)

    # Large fault currents
    sample["IA"] = uniform(1200, 3000)
    sample["IB"] = uniform(1200, 3000)
    sample["IC"] = uniform(1200, 3000)

    # Large current change
    sample["delta_IA"] = uniform(0.5, 2.0)
    sample["delta_IB"] = uniform(0.5, 2.0)
    sample["delta_IC"] = uniform(0.5, 2.0)

    # Balanced 3-phase fault -> relatively small I2
    sample["V2"] = uniform(0, 5)
    sample["I2"] = uniform(0, 0.05)

    sample["scenario"] = "three_phase_fault"

    return sample


# ============================================================
# Scenario 7: Negative sequence fault
# ============================================================

def negative_sequence_fault():
    sample = base_sample()

    # Voltage unbalance
    sample["VAB"] = uniform(20, 80)
    sample["VBC"] = uniform(80, 120)
    sample["VCA"] = uniform(20, 80)

    # Negative sequence voltage
    sample["V2"] = uniform(10, 30)

    # Negative sequence current above typical normal imbalance
    sample["I2"] = uniform(0.10, 0.50)

    # Current changes
    sample["delta_IA"] = uniform(0.1, 1.0)
    sample["delta_IB"] = uniform(0.1, 1.0)
    sample["delta_IC"] = uniform(0.1, 1.0)

    sample["scenario"] = "negative_sequence_fault"

    return sample


# ============================================================
# Scenario 8: External MCB/VTS operation
# ============================================================

def mcb_vts_operation():
    sample = base_sample()

    # External MCB/VTS signal asserted
    sample["MCB_VTS"] = 1

    sample["scenario"] = "mcb_vts_operation"

    return sample


# ============================================================
# Scenario 9: Normal load unbalance
# ============================================================

def normal_load_unbalance():
    sample = base_sample()

    # Moderate voltage unbalance
    sample["VAB"] = uniform(95, 120)
    sample["VBC"] = uniform(95, 120)
    sample["VCA"] = uniform(95, 120)

    # Normal current unbalance
    sample["IA"] = uniform(50, 120)
    sample["IB"] = uniform(30, 110)
    sample["IC"] = uniform(50, 130)

    sample["V2"] = uniform(0, 9)
    sample["I2"] = uniform(0.01, 0.08)

    sample["scenario"] = "normal_load_unbalance"

    return sample


# ============================================================
# Scenario collection
# ============================================================

SCENARIOS = [
    healthy,
    one_phase_vt_loss,
    two_phase_vt_loss,
    three_phase_vt_loss_under_load,
    three_phase_vt_loss_energization,
    three_phase_fault,
    negative_sequence_fault,
    mcb_vts_operation,
    normal_load_unbalance,
]


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset():

    rows = []

    for scenario_function in SCENARIOS:

        print(
            f"Generating: {scenario_function.__name__}"
        )

        for _ in range(SAMPLES_PER_SCENARIO):

            row = scenario_function()

            rows.append(row)

    df = pd.DataFrame(rows)

    # Shuffle dataset
    df = df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    # Add sample ID
    df.insert(
        0,
        "sample_id",
        np.arange(len(df))
    )

    return df


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("P642 VTS Realistic Scenario Generator")
    print("=" * 70)

    df = generate_dataset()

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDataset generated successfully!")

    print(f"\nNumber of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")

    print("\nScenario distribution:")
    print(df["scenario"].value_counts())

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print("\nFirst 10 rows:")
    print(df.head(10))


if __name__ == "__main__":
    main()