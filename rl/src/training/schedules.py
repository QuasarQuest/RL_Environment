"""Learning-rate and other value schedules for SB3 training."""
from __future__ import annotations


def linear_schedule(start: float, end: float):
    """Return f(progress_remaining) → float that linearly interpolates start → end.

    progress_remaining goes from 1.0 (start of training) to 0.0 (end).
    When start == end the schedule is constant and avoids the mul entirely.
    """
    if start == end:
        return lambda _: float(start)
    return lambda p: float(end + p * (start - end))
