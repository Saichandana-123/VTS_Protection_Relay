
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

    def next_state(self, set_signal: bool, reset_signal: bool) -> bool:
        """Return the state for these inputs without changing the state."""

        if reset_signal:
            return False
        if set_signal:
            return True
        return self.q

    def update(self, set_signal: bool, reset_signal: bool) -> bool:
        self.q = self.next_state(set_signal, reset_signal)
        return self.q
