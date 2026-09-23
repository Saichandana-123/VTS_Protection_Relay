"""
Generate synthetic input data for the MiCOM P642 VTS virtual replication.

This script generates operating inputs within the identified valid ranges.
The generated CSV can later be passed through the recreated VTS logic
to obtain the ground-truth VTS outputs.

Author: Saichandana
Project: Virtual Replication of Protection IED
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SEED = 42
N_SAMPLES = 10000

rng = np.random.default_rng(SEED)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "vts_input_dataset.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def random_uniform(low, high, size):
    """Generate uniformly distributed values."""
    return rng.uniform(low, high, size)


def random_binary(size, probability_one=0.5):
    """Generate binary 0/1 values."""
    return rng.binomial(1, probability_one, size)


# ============================================================
# Generate Analog Inputs
# ============================================================

def generate_analog_inputs(n):
    """
    Generate analog operating quantities.

    Values are generated over broad physically meaningful ranges.
    Exact relay thresholds/settings will be applied later by the
    VTS logic model.
    """

    data = {}

    # --------------------------------------------------------
    # Phase voltages
    # --------------------------------------------------------
    #
    # Nominal VT secondary voltage in the provided settings
    # is around 110 V.
    #
    # We intentionally generate values both below and above
    # the VTS phase-voltage detector levels so that the dataset
    # contains normal and abnormal cases.
    #
    # P642 phase voltage detector:
    # pickup  = 30 V
    # dropout = 10 V
    # --------------------------------------------------------

    data["VAB"] = random_uniform(0, 130, n)
    data["VBC"] = random_uniform(0, 130, n)
    data["VCA"] = random_uniform(0, 130, n)

    # --------------------------------------------------------
    # Negative sequence voltage
    # --------------------------------------------------------

    # V2 threshold is approximately 10 V for VTS detection.
    data["V2"] = random_uniform(0, 30, n)

    # --------------------------------------------------------
    # Phase currents
    # --------------------------------------------------------

    # P642 example configuration has 300 A CT primary.
    # Generate values from 0 to 600 A so that normal,
    # overload and fault-like conditions are represented.
    data["IA"] = random_uniform(0, 600, n)
    data["IB"] = random_uniform(0, 600, n)
    data["IC"] = random_uniform(0, 600, n)

    # --------------------------------------------------------
    # Negative sequence current
    # --------------------------------------------------------

    data["I2"] = random_uniform(0, 150, n)

    # --------------------------------------------------------
    # Superimposed current
    # --------------------------------------------------------

    # Used to distinguish voltage loss caused by a system
    # fault from a VT failure.
    data["delta_I"] = random_uniform(0, 100, n)

    return data


# ============================================================
# Generate Digital Inputs
# ============================================================

def generate_digital_inputs(n):
    """Generate binary/control inputs used by VTS logic."""

    data = {}

    # --------------------------------------------------------
    # External MCB/VTS signal
    # --------------------------------------------------------
    #
    # DDB 874 = MCB/VTS input to VTS logic.
    # --------------------------------------------------------

    data["MCB_VTS"] = random_binary(
        n,
        probability_one=0.10
    )

    # --------------------------------------------------------
    # Manual reset input
    # --------------------------------------------------------

    data["VTS_MANRESET"] = random_binary(
        n,
        probability_one=0.05
    )

    # --------------------------------------------------------
    # Circuit-breaker / pole related status
    # --------------------------------------------------------

    data["CB_CLOSED"] = random_binary(
        n,
        probability_one=0.80
    )

    # --------------------------------------------------------
    # Any Pole Dead indication
    # --------------------------------------------------------

    data["ANY_POLE_DEAD"] = random_binary(
        n,
        probability_one=0.10
    )

    return data


# ============================================================
# Generate Settings
# ============================================================

def generate_settings(n):
    """
    Generate VTS settings.

    These are kept as input features because our final ML model
    must be able to operate when relay settings change.
    """

    data = {}

    # --------------------------------------------------------
    # VTS status
    #
    # 0 = Disabled
    # 1 = Indication
    # 2 = Blocking
    # --------------------------------------------------------

    data["VTS_STATUS"] = rng.choice(
        [0, 1, 2],
        size=n,
        p=[0.05, 0.10, 0.85]
    )

    # --------------------------------------------------------
    # Reset mode
    #
    # 0 = Manual
    # 1 = Auto
    # --------------------------------------------------------

    data["VTS_RESET_MODE"] = rng.choice(
        [0, 1],
        size=n,
        p=[0.7, 0.3]
    )

    # --------------------------------------------------------
    # VTS time delay
    #
    # P642 valid range: 1 to 10 s
    # --------------------------------------------------------

    data["VTS_TIME_DELAY"] = random_uniform(
        1.0,
        10.0,
        n
    )

    # --------------------------------------------------------
    # VTS I> inhibit
    #
    # Valid range from P642 technical data:
    # approximately 0.08 In to 32 In
    #
    # Here represented in per-unit.
    # --------------------------------------------------------

    data["VTS_I_INHIBIT"] = random_uniform(
        0.08,
        32.0,
        n
    )

    # --------------------------------------------------------
    # VTS I2> inhibit
    #
    # Valid range:
    # 0.05 In to 0.50 In
    # --------------------------------------------------------

    data["VTS_I2_INHIBIT"] = random_uniform(
        0.05,
        0.50,
        n
    )

    return data


# ============================================================
# Generate Dataset
# ============================================================

def generate_dataset(n):
    """Generate complete VTS input dataset."""

    data = {}

    # Analog inputs
    data.update(generate_analog_inputs(n))

    # Digital inputs
    data.update(generate_digital_inputs(n))

    # Relay settings
    data.update(generate_settings(n))

    # Sample identifier
    data["sample_id"] = np.arange(n)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Put sample_id first
    columns = ["sample_id"] + [
        col for col in df.columns if col != "sample_id"
    ]

    df = df[columns]

    return df


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("MiCOM P642 VTS Input Dataset Generator")
    print("=" * 60)

    print(f"\nGenerating {N_SAMPLES} samples...")

    df = generate_dataset(N_SAMPLES)

    # Save dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nFirst 5 samples:")
    print(df.head())

    print("\nDataset statistics:")
    print(df.describe().round(3))


if __name__ == "__main__":
    main()