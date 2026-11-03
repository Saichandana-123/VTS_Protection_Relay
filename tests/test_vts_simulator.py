
from src.logic.vts_simulator import simulate_delayed_latch


def test_delayed_latch_activates_after_delay():
    time_s = [0.000, 0.001, 0.002, 0.003, 0.004]
    condition = [False, True, True, True, True]

    result = simulate_delayed_latch(
        condition=condition,
        time_s=time_s,
        delay_s=0.003,
    )

    assert result.delayed_condition == [
        False, False, False, True, True
    ]
    assert result.stored_state == [
        False, False, False, True, True
    ]


def test_reset_clears_stored_state():
    time_s = [0.000, 0.001, 0.002, 0.003, 0.004]
    condition = [True, True, True, True, True]
    reset = [False, False, False, True, False]

    result = simulate_delayed_latch(
        condition=condition,
        time_s=time_s,
        delay_s=0.002,
        reset=reset,
    )

    assert result.stored_state[2] is True
    assert result.stored_state[3] is False
    assert result.stored_state[4] is True
