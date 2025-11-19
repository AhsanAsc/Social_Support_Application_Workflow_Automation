import time
from contextlib import contextmanager
from typing import Iterator, Callable


@contextmanager
def timing_metric(name: str) -> Iterator[None]:
    """
    Simple timing context manager.
    Later: integrate with Prometheus / OpenTelemetry.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        # TODO: export to metrics backend
        print(f"[METRIC] {name} took {duration:.3f}s")
