
"""Deterministic timing components for the P642 VTS hybrid model."""


class PickupDelay:
    """Output turns ON after the input remains ON for the delay."""

    def __init__(self, delay_s: float, sample_time_s: float = 0.001):
        if delay_s < 0 or sample_time_s <= 0:
            raise ValueError("Delay must be >= 0 and sample time must be > 0.")

        self.delay_s = delay_s
        self.sample_time_s = sample_time_s
        self.elapsed_s = 0.0
        self.output = False

    def reset(self):
        self.elapsed_s = 0.0
        self.output = False

    def update(self, signal: bool) -> bool:
        if not signal:
            self.reset()
            return self.output

        self.elapsed_s += self.sample_time_s

        if self.elapsed_s + 1e-12 >= self.delay_s:
            self.output = True

        return self.output


class DropoffDelay:
    """Output turns ON immediately; turns OFF after input turns OFF
    continuously for the configured delay.
    """

    def __init__(self, delay_s: float, sample_time_s: float = 0.001):
        if delay_s < 0 or sample_time_s <= 0:
            raise ValueError("Delay must be >= 0 and sample time must be > 0.")

        self.delay_s = delay_s
        self.sample_time_s = sample_time_s
        self.off_elapsed_s = 0.0
        self.output = False

    def reset(self):
        self.off_elapsed_s = 0.0
        self.output = False

    def update(self, signal: bool) -> bool:
        if signal:
            self.output = True
            self.off_elapsed_s = 0.0
            return self.output

        if self.output:
            self.off_elapsed_s += self.sample_time_s

            if self.off_elapsed_s + 1e-12 >= self.delay_s:
                self.output = False
                self.off_elapsed_s = 0.0

        return self.output
