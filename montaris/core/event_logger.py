"""Structured event logger for diagnostics export.

Collects timing and operational data in a fixed-size ring buffer.
Export as JSON for offline analysis.
"""

import collections
import os
import platform
import time
from contextlib import contextmanager


def _get_rss_mb():
    """Return current process RSS in MB (cross-platform via psutil)."""
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return 0.0


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
        self._hwm_mb = 0.0  # all-time high-water mark

    def log(self, category, name, duration_ms=None, **metadata):
        """Append a structured event.

        Categories: io, render, tool, memory, compress, undo, transform
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

    def log_mem(self, label, **extra):
        """Log a memory snapshot with a descriptive label."""
        mb = _get_rss_mb()
        if mb > self._hwm_mb:
            self._hwm_mb = mb
        self.log("memory", label, rss_mb=mb, hwm_mb=self._hwm_mb, **extra)
        return mb

    @contextmanager
    def timed(self, category, name, **metadata):
        """Context manager that auto-logs duration."""
        t0 = time.perf_counter()
        yield
        self.log(category, name, (time.perf_counter() - t0) * 1000, **metadata)

    @contextmanager
    def timed_mem(self, category, name, **metadata):
        """Context manager that logs duration AND memory before/after."""
        mem_before = _get_rss_mb()
        t0 = time.perf_counter()
        yield
        dur = (time.perf_counter() - t0) * 1000
        mem_after = _get_rss_mb()
        if mem_after > self._hwm_mb:
            self._hwm_mb = mem_after
        self.log(category, name, dur,
                 mem_before_mb=mem_before, mem_after_mb=mem_after,
                 mem_delta_mb=round(mem_after - mem_before, 1),
                 **metadata)

    def export_json(self, app=None):
        """Return full diagnostics dict with session summary + events."""
        now = time.time()
        current_rss = _get_rss_mb()
        session = {
            'start_time': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(self._session_start)),
            'duration_s': round(now - self._session_start, 1),
            'platform': platform.platform(),
            'python': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'total_events': len(self._events),
            'current_rss_mb': current_rss,
            'hwm_rss_mb': self._hwm_mb,
        }

        # ROI memory breakdown
        if app is not None:
            try:
                ls = app.layer_stack
                if ls.image_layer is not None:
                    h, w = ls.image_layer.shape[:2]
                    session['image_dimensions'] = [w, h]
                rois = ls.roi_layers
                session['total_rois'] = len(rois)
                compressed = sum(1 for r in rois if r.is_compressed)
                decompressed = len(rois) - compressed
                session['rois_compressed'] = compressed
                session['rois_decompressed'] = decompressed
                # Estimate memory: decompressed masks
                mask_bytes = sum(
                    r._mask.nbytes for r in rois
                    if r._mask is not None
                )
                session['decompressed_mask_mb'] = round(mask_bytes / (1024 * 1024), 1)
                # RLE data size
                rle_bytes = sum(
                    len(r._rle_data) for r in rois
                    if r._rle_data is not None
                )
                session['rle_data_mb'] = round(rle_bytes / (1024 * 1024), 2)
            except Exception:
                pass

        # Memory timeline: extract memory events for quick overview
        mem_events = [e for e in self._events if e['cat'] == 'memory']
        mem_timeline = []
        for e in mem_events:
            entry = {'ts': round(e['ts'] - self._session_start, 2), 'label': e['name']}
            if 'meta' in e:
                entry.update(e['meta'])
            elif 'rss_mb' in e:
                entry['rss_mb'] = e['rss_mb']
            mem_timeline.append(entry)

        # Transform operations summary
        transform_events = [e for e in self._events if e['cat'] == 'transform']

        return {
            'version': 2,
            'session': session,
            'memory_timeline': mem_timeline,
            'transform_ops': transform_events,
            'events': list(self._events),
        }
