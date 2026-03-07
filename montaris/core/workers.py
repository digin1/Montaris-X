"""Singleton ThreadPoolExecutor for CPU-bound work that releases the GIL.

numpy, scipy, and PIL all release the GIL during heavy C-level computation,
so threads achieve true parallelism for these operations without
serialization overhead (shared memory).
"""

import os
from concurrent.futures import ThreadPoolExecutor

_pool = None
_pool_size = 0


def get_pool():
    """Return the shared ThreadPoolExecutor, creating it lazily."""
    global _pool, _pool_size
    if _pool is None:
        _pool_size = max(2, min(os.cpu_count() or 2, 8))
        _pool = ThreadPoolExecutor(max_workers=_pool_size)
    return _pool


def worker_count():
    """Return the number of workers in the pool."""
    get_pool()  # ensure initialized
    return _pool_size


def shutdown_pool():
    """Shut down the pool cleanly. Safe to call multiple times."""
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False)
        _pool = None
