import pandas as pd


INPUT_FILE = "data/raw/vts_ground_truth_dataset.csv"


def main():

    df = pd.read_csv(INPUT_FILE)

    print("=" * 60)
    print("VTS GROUND TRUTH VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Expected VTS behavior for each scenario
    # --------------------------------------------------------

    expected = {
        "healthy": 0,
        "mcb_vts_operation": 1,
        "negative_sequence_fault": 0,
        "normal_load_unbalance": 0,
        "one_phase_vt_loss": 1,
        "two_phase_vt_loss": 1,
        "three_phase_vt_loss_under_load": 1,
        "three_phase_vt_loss_energization": 1,
        "three_phase_fault": 0,
    }

    passed = True

    for scenario, expected_value in expected.items():

        scenario_data = df[df["scenario"] == scenario]

        actual_values = scenario_data["VTS_CONDITION"].unique()

        print(f"\nScenario: {scenario}")
        print(f"Expected: {expected_value}")
        print(f"Actual  : {actual_values}")

        # Every sample in this scenario should have
        # the expected VTS condition.
        if len(actual_values) != 1 or actual_values[0] != expected_value:
            print("❌ FAIL")
            passed = False
        else:
            print("✅ PASS")

    # --------------------------------------------------------
    # Check dataset size
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET CHECK")
    print("=" * 60)

    print(f"Total samples: {len(df)}")

    if len(df) == 13500:
        print("✅ Correct dataset size")
    else:
        print("❌ Incorrect dataset size")
        passed = False

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    if passed:
        print("✅ ALL VTS VALIDATION TESTS PASSED")
    else:
        print("❌ SOME VTS VALIDATION TESTS FAILED")

    print("=" * 60)


if __name__ == "__main__":
    main()