"""
vts_logic.py

Static / combinational ground-truth generator for the VTS
(Voltage Transformer Supervision) logic of a MiCOM P642
protection relay.

This is the FIRST-STAGE model:
    - It is combinational (row-by-row), not a time-domain
      state machine.
    - VTS_TIME_DELAY and the P642's timed slow-block behavior
      are NOT yet implemented. VTS_SLOW_BLOCK is currently set
      as a static placeholder whenever a VTS condition exists
      in Blocking mode.

Run from the project root, e.g.:
    python src/logic/vts_logic.py
"""

import os
import pandas as pd

# ============================================================
# P642 VTS CONSTANTS
# ============================================================
# For 100/120 V VT rating
VOLTAGE_DROPOUT = 10.0     # V
VOLTAGE_PICKUP = 30.0      # V
V2_THRESHOLD = 10.0        # V

# Fixed superimposed current threshold
DELTA_I_THRESHOLD_PU = 0.10  # 0.1 In


# ============================================================
# P642 VTS CONFIGURATION
# ============================================================
# These are the relay settings used for this ground-truth
# generation model.
#
# VTS Status:
#   0 = Disabled
#   1 = Indication
#   2 = Blocking
#
# VTS Reset Mode:
#   0 = Manual
#   1 = Auto
VTS_STATUS = 2
VTS_RESET_MODE = 0
VTS_TIME_DELAY = 5.0      # seconds
VTS_I_INHIBIT = 10.0      # pu
VTS_I2_INHIBIT = 0.05     # pu


# ============================================================
# LEVEL DETECTORS
# ============================================================
def voltage_level_detected(v, previous_state=False):
    """
    P642 phase-phase voltage detector.

    Pickup = 95 V
    Drop-off = 70 V

    Hysteresis:
        If currently OFF, voltage must rise to 95 V to pick up.
        If currently ON, voltage must fall below 70 V to drop out.
    """
    if previous_state:
        return v >= 70.0
    else:
        return v >= 95.0


def phase_voltage_present(v):
    """
    Simplified phase-voltage detector.

    P642 phase-voltage detector:
        Pickup   = 30 V
        Drop-off = 10 V

    For ground-truth classification, voltage is considered
    present when it is above the dropout boundary.
    """
    return v >= VOLTAGE_DROPOUT


def v2_detected(v2, previous_state=False):
    """
    P642 negative-sequence voltage detector.

    Pickup = 10 V
    Drop-off = 95% of pickup = 9.5 V.
    """
    pickup = V2_THRESHOLD
    dropout = 0.95 * V2_THRESHOLD

    if previous_state:
        return v2 >= dropout
    else:
        return v2 >= pickup


def i2_detected(i2_pu, i2_setting_pu, previous_state=False):
    """
    P642 negative-sequence current detector.

    Pickup = 100% of configured setting.
    Drop-off = 95% of configured setting.
    """
    pickup = i2_setting_pu
    dropout = 0.95 * i2_setting_pu

    if previous_state:
        return i2_pu >= dropout
    else:
        return i2_pu >= pickup


def current_inhibit_detected(ia, ib, ic, i_inhibit_pu, nominal_current_a):
    """
    VTS I> inhibit.

    The relay setting is expressed in per-unit. Measured phase
    currents are converted using the configured nominal CT
    current.

    IMPORTANT:
        nominal_current_a is a model parameter and is NOT taken
        from the P642 manual. It must eventually be replaced by
        the actual CT nominal current for the relay
        configuration.
    """
    threshold_a = i_inhibit_pu * nominal_current_a
    return (
        ia >= threshold_a
        or ib >= threshold_a
        or ic >= threshold_a
    )


def superimposed_current_detected(delta_i):
    """
    Superimposed current detector.

    P642 fixed threshold = 0.1 In.

    In this first static model, delta_i represents the maximum
    of delta_IA, delta_IB and delta_IC.
    """
    return delta_i >= DELTA_I_THRESHOLD_PU


# ============================================================
# VTS COMBINATIONAL LOGIC
# ============================================================
def evaluate_vts_condition(row, nominal_current_a=100.0):
    """
    Evaluate whether a VTS condition exists for one input row.

    This is the first static/combinational implementation of
    the P642 VTS logic. It is NOT yet the final time-domain
    relay state machine.

    Parameters
    ----------
    row : pandas.Series
        One row from the scenario dataset.
    nominal_current_a : float
        CT nominal current used to convert the VTS I> inhibit
        setting from pu to amperes.
        Default = 100 A as a placeholder model parameter.
    """
    # --------------------------------------------------------
    # Extract inputs
    # --------------------------------------------------------
    vab = row["VAB"]
    vbc = row["VBC"]
    v2 = row["V2"]

    ia = row["IA"]
    ib = row["IB"]
    ic = row["IC"]
    i2 = row["I2"]

    delta_IA = row["delta_IA"]
    delta_IB = row["delta_IB"]
    delta_IC = row["delta_IC"]

    # Any phase with a superimposed-current operation
    delta_i = max(delta_IA, delta_IB, delta_IC)

    mcb_vts = row["MCB_VTS"]
    any_pole_dead = row["ANY_POLE_DEAD"]

    # VTS settings are added to the dataframe by
    # generate_ground_truth().
    vts_i_inhibit = row["VTS_I_INHIBIT"]
    vts_i2_inhibit = row["VTS_I2_INHIBIT"]

    # --------------------------------------------------------
    # 1. External MCB/VTS input
    # --------------------------------------------------------
    if mcb_vts == 1:
        return True, "MCB_VTS"

    # --------------------------------------------------------
    # 2. Detect voltage/current conditions
    # --------------------------------------------------------
    v2_high = v2_detected(v2)
    i2_high = i2_detected(i2, vts_i2_inhibit)

    # Count how many phase-phase voltage inputs are lost.
    voltage_loss_count = sum(
        [
            not phase_voltage_present(vab),
            not phase_voltage_present(vbc),
            not phase_voltage_present(vca),
        ]
    )
    all_voltage_lost = voltage_loss_count == 3

    # --------------------------------------------------------
    # 3. Loss of one/two phase voltages
    #
    # Simplified representation of the P642 negative-sequence
    # voltage/current discrimination:
    #
    #   V2 high
    #   AND I2 below VTS I2 inhibit
    #   AND one/two voltages lost
    #
    #   => VT failure indication/condition
    # --------------------------------------------------------
    if v2_high and not i2_high:
        if 1 <= voltage_loss_count < 3:
            return True, "ONE_TWO_PHASE_VOLTAGE_LOSS"

    # --------------------------------------------------------
    # 4. Loss of all three phase voltages under load
    #
    # If all voltages disappear without a corresponding
    # superimposed current change, classify it as VT failure.
    #
    # NOTE:
    #   The actual P642 uses temporal detector/counting
    #   behavior. That timing is not yet modeled here.
    # --------------------------------------------------------
    if all_voltage_lost:
        delta_i_active = superimposed_current_detected(delta_i)
        if not delta_i_active:
            return True, "THREE_PHASE_VOLTAGE_LOSS_UNDER_LOAD"

    # --------------------------------------------------------
    # 5. Three-phase voltage loss during line energization
    #
    # Absence of all three voltages can represent:
    #   a) VT failure
    #   b) close-up three-phase fault
    #
    # The VTS I> inhibit prevents VTS operation for the
    # high-current fault case.
    #
    # Simplified condition:
    #   all voltage lost
    #   AND line is not pole-dead
    #   AND phase current below I> inhibit
    #   => VTS
    # --------------------------------------------------------
    if all_voltage_lost and not any_pole_dead:
        current_inhibit = current_inhibit_detected(
            ia, ib, ic, vts_i_inhibit, nominal_current_a
        )
        if not current_inhibit:
            return True, "THREE_PHASE_VOLTAGE_LOSS_ENERGIZATION"

    # --------------------------------------------------------
    # No VTS condition
    # --------------------------------------------------------
    return False, "NO_VTS"


# ============================================================
# DATASET PROCESSING
# ============================================================
def generate_ground_truth(input_csv, output_csv, nominal_current_a=100.0):
    """
    Read the scenario dataset, apply the P642 VTS logic, and
    generate a labeled ground-truth dataset.
    """
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(
            f"Input CSV not found: '{input_csv}'\n"
            "Make sure you are running this script from the "
            "project root, e.g.:\n"
            "    python src/logic/vts_logic.py"
        )

    df = pd.read_csv(input_csv)

    # --------------------------------------------------------
    # Add P642 relay configuration to every sample
    # --------------------------------------------------------
    df["VTS_STATUS"] = VTS_STATUS
    df["VTS_RESET_MODE"] = VTS_RESET_MODE
    df["VTS_TIME_DELAY"] = VTS_TIME_DELAY
    df["VTS_I_INHIBIT"] = VTS_I_INHIBIT
    df["VTS_I2_INHIBIT"] = VTS_I2_INHIBIT

    # --------------------------------------------------------
    # Evaluate VTS condition
    # --------------------------------------------------------
    vts_conditions = []
    reasons = []
    for _, row in df.iterrows():
        condition, reason = evaluate_vts_condition(
            row, nominal_current_a=nominal_current_a
        )
        vts_conditions.append(int(condition))
        reasons.append(reason)

    df["VTS_CONDITION"] = vts_conditions
    df["VTS_REASON"] = reasons

    # --------------------------------------------------------
    # VTS outputs
    # --------------------------------------------------------
    df["VTS_INDICATION"] = 0
    df["VTS_FAST_BLOCK"] = 0
    df["VTS_SLOW_BLOCK"] = 0

    for idx, row in df.iterrows():
        vts_condition = row["VTS_CONDITION"]
        status = row["VTS_STATUS"]

        if vts_condition == 1:
            # ------------------------------------------------
            # Indication mode
            # ------------------------------------------------
            if status == 1:
                df.at[idx, "VTS_INDICATION"] = 1

            # ------------------------------------------------
            # Blocking mode
            # ------------------------------------------------
            elif status == 2:
                df.at[idx, "VTS_INDICATION"] = 1
                df.at[idx, "VTS_FAST_BLOCK"] = 1
                # IMPORTANT:
                # This is a placeholder static representation.
                # The actual P642 slow block is time-delayed.
                # A separate time-domain state machine will be
                # implemented later.
                df.at[idx, "VTS_SLOW_BLOCK"] = 1

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------
    print("=" * 60)
    print("P642 VTS GROUND TRUTH GENERATED")
    print("=" * 60)
    print(f"Input : {input_csv}")
    print(f"Output: {output_csv}")

    print("\nP642 configuration used:")
    print(f"  VTS Status  : {VTS_STATUS}")
    print(f"  Reset Mode  : {VTS_RESET_MODE}")
    print(f"  Time Delay  : {VTS_TIME_DELAY} s")
    print(f"  I> Inhibit  : {VTS_I_INHIBIT} pu")
    print(f"  I2> Inhibit : {VTS_I2_INHIBIT} pu")
    print(f"  CT nominal  : {nominal_current_a} A (placeholder)")

    print("\nVTS condition counts:")
    print(df["VTS_CONDITION"].value_counts())

    print("\nVTS reasons:")
    print(df["VTS_REASON"].value_counts())

    print("\nScenario-wise VTS condition:")
    print(pd.crosstab(df["scenario"], df["VTS_CONDITION"]))

    print("\nOutput columns added:")
    print("  VTS_STATUS")
    print("  VTS_RESET_MODE")
    print("  VTS_TIME_DELAY")
    print("  VTS_I_INHIBIT")
    print("  VTS_I2_INHIBIT")
    print("  VTS_CONDITION")
    print("  VTS_REASON")
    print("  VTS_INDICATION")
    print("  VTS_FAST_BLOCK")
    print("  VTS_SLOW_BLOCK")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    input_file = "data/raw/vts_scenario_dataset.csv"
    output_file = "data/raw/vts_ground_truth_dataset.csv"

    generate_ground_truth(
        input_csv=input_file,
        output_csv=output_file,
        # IMPORTANT:
        # Placeholder model parameter.
        # Replace with the actual CT nominal current later.
        nominal_current_a=100.0,
    )