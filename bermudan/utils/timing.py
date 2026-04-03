"""Wall-clock timing utilities."""

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class TimerResult:
    """Stores elapsed wall-clock time in seconds."""

    elapsed: float = 0.0


@contextmanager
def timer():
    """Context manager that measures wall-clock time.

    Usage
    -----
    >>> result = TimerResult()
    >>> with timer() as t:
    ...     expensive_computation()
    >>> print(t.elapsed)
    """
    t = TimerResult()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start
