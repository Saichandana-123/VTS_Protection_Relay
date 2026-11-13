
from src.timing.timers import PickupDelay, DropoffDelay
from src.timing.flip_flop import SRFlipFlop


def test_pickup_delay():
    timer = PickupDelay(delay_s=0.003, sample_time_s=0.001)

    assert timer.update(True) is False
    assert timer.update(True) is False
    assert timer.update(True) is True
    assert timer.update(False) is False


def test_dropoff_delay():
    timer = DropoffDelay(delay_s=0.003, sample_time_s=0.001)

    assert timer.update(True) is True
    assert timer.update(False) is True
    assert timer.update(False) is True
    assert timer.update(False) is False


def test_sr_flip_flop():
    ff = SRFlipFlop()

    assert ff.update(True, False) is True
    assert ff.update(False, False) is True
    assert ff.update(False, True) is False


def test_sr_flip_flop_next_state_does_not_mutate_before_commit():
    flip_flops = [SRFlipFlop() for _ in range(4)]
    inputs = [
        {"set_signal": True, "reset_signal": False},
        {"set_signal": False, "reset_signal": False},
        {"set_signal": True, "reset_signal": True},
        {"set_signal": False, "reset_signal": True},
    ]

    next_states = [
        flip_flop.next_state(**signals)
        for flip_flop, signals in zip(flip_flops, inputs)
    ]

    assert [flip_flop.q for flip_flop in flip_flops] == [False] * 4
    assert next_states == [True, False, False, False]

    for flip_flop, signals in zip(flip_flops, inputs):
        flip_flop.update(**signals)

    assert [flip_flop.q for flip_flop in flip_flops] == next_states


def test_sr_flip_flop_reset_priority_is_preserved():
    flip_flop = SRFlipFlop(initial_state=True)

    assert flip_flop.next_state(True, True) is False
    assert flip_flop.q is True
    assert flip_flop.update(True, True) is False
