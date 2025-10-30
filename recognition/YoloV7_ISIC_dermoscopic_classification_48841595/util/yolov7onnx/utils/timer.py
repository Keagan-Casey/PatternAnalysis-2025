import time
from typing import Optional


class Timer:
    """A simple timer class to measure the time elapsed between two points in the code.

    The Timer class can be used as a context manager, which will start the timer upon entering
    the context and stop the timer upon exiting the context. The elapsed time can then be
    accessed using the `elapsed` attribute.

    Examples:
        >>> with Timer() as t:
        ...     time.sleep(1)  # Simulate some work
        >>> print(f"Elapsed time: {t.elapsed:.2f} seconds")
        Elapsed time: 1.00 seconds
    """

    def __init__(self):
        """Initializes a Timer instance."""
        self.start_t: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        """Starts the timer by recording the current time.

        Returns:
            Timer: The Timer instance itself.
        """
        self.start_t = time.perf_counter()
        return self

    def __exit__(self, *args):
        """Stops the timer by calculating the elapsed time."""
        if self.start_t is None:
            raise ValueError("Timer was never started")
        self.elapsed = time.perf_counter() - self.start_t
