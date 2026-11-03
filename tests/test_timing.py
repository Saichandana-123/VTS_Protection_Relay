
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
