
"""Deterministic SR flip-flop component for the P642 VTS hybrid model."""


class SRFlipFlop:
    """Stores state until reset.

    Reset has priority if set and reset are both True.
    This priority must be checked against the specific reference logic.
    """

    def __init__(self, initial_state: bool = False):
        self.q = bool(initial_state)

    def reset(self):
        self.q = False

    def update(self, set_signal: bool, reset_signal: bool) -> bool:
        if reset_signal:
            self.q = False
        elif set_signal:
            self.q = True

        return self.q
