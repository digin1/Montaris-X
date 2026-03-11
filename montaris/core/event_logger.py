"""Structured event logger for diagnostics export.

Collects timing and operational data in a fixed-size ring buffer.
Export as JSON for offline analysis.
"""

import collections
import os
import platform
import time
from contextlib import contextmanager


class EventLogger:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, max_events=5000):
        self._events = collections.deque(maxlen=max_events)
        self._session_start = time.time()

    def log(self, category, name, duration_ms=None, **metadata):
        """Append a structured event.

        Categories: io, render, tool, memory, compress, undo
        """
        entry = {
            'ts': time.time(),
            'cat': category,
            'name': name,
        }
        if duration_ms is not None:
            entry['dur_ms'] = round(duration_ms, 2)
        if metadata:
            entry['meta'] = metadata
        self._events.append(entry)

    @contextmanager
    def timed(self, category, name, **metadata):
        """Context manager that auto-logs duration."""
        t0 = time.perf_counter()
        yield
        self.log(category, name, (time.perf_counter() - t0) * 1000, **metadata)

    def export_json(self, app=None):
        """Return full diagnostics dict with session summary + events."""
        now = time.time()
        session = {
            'start_time': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(self._session_start)),
            'duration_s': round(now - self._session_start, 1),
            'platform': platform.platform(),
            'python': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'total_events': len(self._events),
        }

        # Memory info (Linux /proc/self/status, fallback to resource module)
        try:
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            # maxrss is in KB on Linux
            session['peak_rss_mb'] = round(rusage.ru_maxrss / 1024, 1)
        except Exception:
            pass
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        session['current_rss_mb'] = round(int(line.split()[1]) / 1024, 1)
                        break
        except Exception:
            pass

        # App-specific info
        if app is not None:
            try:
                ls = app.layer_stack
                if ls.image_layer is not None:
                    h, w = ls.image_layer.shape[:2]
                    session['image_dimensions'] = [w, h]
                session['total_rois'] = len(ls.roi_layers)
            except Exception:
                pass

        return {
            'version': 1,
            'session': session,
            'events': list(self._events),
        }
