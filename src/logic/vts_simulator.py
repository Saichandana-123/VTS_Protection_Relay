
"""Time-domain components for the P642 VTS hybrid model.

This first version provides a reusable stateful simulation harness.
It is not yet a complete implementation of every branch in P642
manual Figure 59; the exact intermediate conditions and reset paths
must be mapped from the reference diagram.
"""

from dataclasses import dataclass
from typing import Sequence

from src.timing.timers import PickupDelay
from src.timing.flip_flop import SRFlipFlop


@dataclass
class VTSRunResult:
    """Time-series results for a single simulated condition."""

    condition: list[bool]
    delayed_condition: list[bool]
    stored_state: list[bool]


def simulate_delayed_latch(
    condition: Sequence[bool],
    time_s: Sequence[float],
    delay_s: float,
    reset: Sequence[bool] | None = None,
) -> VTSRunResult:
    """Simulate a pickup delay followed by an SR state element.

    Parameters
    ----------
    condition:
        Boolean condition at each sample.
    time_s:
        Strictly increasing simulation times in seconds.
    delay_s:
        Required continuous pickup time in seconds.
    reset:
        Optional reset signal at each sample.

    Notes
    -----
    This is a generic timing/state prototype, not the complete
    P642 Figure 59 logic. The actual relay has multiple branches
    and specific reset conditions that must be mapped separately.
    """

    if len(condition) != len(time_s):
        raise ValueError("condition and time_s must have equal lengths.")

    if reset is not None and len(reset) != len(condition):
        raise ValueError("reset and condition must have equal lengths.")

    if not time_s:
        return VTSRunResult([], [], [])

    if any(time_s[i] <= time_s[i - 1] for i in range(1, len(time_s))):
        raise ValueError("time_s must be strictly increasing.")

    if delay_s < 0:
        raise ValueError("delay_s must be non-negative.")

    # This prototype assumes a uniform sample interval.
    if len(time_s) > 1:
        sample_time_s = time_s[1] - time_s[0]
        for i in range(2, len(time_s)):
            interval = time_s[i] - time_s[i - 1]
            if abs(interval - sample_time_s) > 1e-9:
                raise ValueError(
                    "This prototype requires uniformly spaced samples."
                )
    else:
        sample_time_s = 0.001

    timer = PickupDelay(delay_s, sample_time_s)
    state = SRFlipFlop()

    delayed_values = []
    state_values = []

    for i, signal in enumerate(condition):
        delayed = timer.update(bool(signal))
        reset_now = bool(reset[i]) if reset is not None else False

        stored = state.update(
            set_signal=delayed,
            reset_signal=reset_now,
        )

        delayed_values.append(delayed)
        state_values.append(stored)

    return VTSRunResult(
        condition=[bool(x) for x in condition],
        delayed_condition=delayed_values,
        stored_state=state_values,
    )
