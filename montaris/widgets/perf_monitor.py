import time
import psutil
import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import QTimer
from montaris import theme as _theme


class PerfMonitor(QWidget):
    """Performance monitor showing FPS, render time, memory usage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        _lbl_ss = _theme.perf_label_style()

        self.fps_label = QLabel("FPS: -")
        self.fps_label.setStyleSheet(_lbl_ss)
        layout.addWidget(self.fps_label)

        self.render_label = QLabel("Render: - ms")
        self.render_label.setStyleSheet(_lbl_ss)
        layout.addWidget(self.render_label)

        self.memory_label = QLabel("Memory: - MB")
        self.memory_label.setStyleSheet(_lbl_ss)
        layout.addWidget(self.memory_label)

        self.tile_cache_label = QLabel("Tile Cache: -")
        self.tile_cache_label.setStyleSheet(_lbl_ss)
        layout.addWidget(self.tile_cache_label)

        from montaris.core.workers import worker_count
        cpu_count = os.cpu_count() or 1
        pool_size = worker_count()
        self.cpu_label = QLabel(f"CPU Cores: {cpu_count} (pool: {pool_size})")
        self.cpu_label.setStyleSheet(_lbl_ss)
        layout.addWidget(self.cpu_label)

        self.accel_label = QLabel("Accel: off")
        self.accel_label.setStyleSheet(_lbl_ss)
        layout.addWidget(self.accel_label)

        self._labels = [self.fps_label, self.render_label, self.memory_label,
                        self.tile_cache_label, self.cpu_label, self.accel_label]
        layout.addStretch()

        self._frame_times = []
        self._render_times = []
        self._tile_cache_info = ""
        self._peak_mem_mb = 0.0       # high-water mark (persists until reset)
        self._interval_peak_mb = 0.0  # peak within current 1s interval

        self._process = psutil.Process(os.getpid())

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_stats)
        self._timer.start(1000)

    def _sample_mem(self):
        """Sample current RSS and update peak trackers. Returns MB."""
        try:
            mb = self._process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
        if mb > self._interval_peak_mb:
            self._interval_peak_mb = mb
        if mb > self._peak_mem_mb:
            self._peak_mem_mb = mb
        return mb

    def record_frame(self):
        self._frame_times.append(time.time())

    def record_render_time(self, ms):
        self._render_times.append(ms)
        # Sample memory at each render — catches spikes between timer ticks
        self._sample_mem()

    def set_tile_cache_info(self, info):
        self._tile_cache_info = info

    def reset_peak_memory(self):
        """Reset the persistent high-water mark."""
        self._peak_mem_mb = self._sample_mem()

    def _update_stats(self):
        now = time.time()
        # FPS: count frames in last second
        self._frame_times = [t for t in self._frame_times if now - t < 1.0]
        fps = len(self._frame_times)
        self.fps_label.setText(f"FPS: {fps}")

        # Render: last + avg
        if self._render_times:
            last = self._render_times[-1]
            avg = sum(self._render_times) / len(self._render_times)
            mx = max(self._render_times)
            if len(self._render_times) == 1:
                self.render_label.setText(f"Render: {last:.1f} ms")
            else:
                self.render_label.setText(
                    f"Render: {last:.1f} ms (avg {avg:.1f}, max {mx:.1f})")
            self._render_times.clear()
        else:
            self.render_label.setText("Render: idle")

        # Memory: current + interval peak + all-time peak
        mem_mb = self._sample_mem()
        interval_peak = self._interval_peak_mb
        self.memory_label.setText(
            f"Memory: {mem_mb:.0f} MB  "
            f"(peak: {interval_peak:.0f}, HWM: {self._peak_mem_mb:.0f})")
        # Reset interval peak for next tick
        self._interval_peak_mb = mem_mb

        self.tile_cache_label.setText(f"Tile Cache: {self._tile_cache_info or '-'}")

        # Acceleration status
        try:
            from montaris.core.accel import is_enabled, get_mode, HAS_CUDA
            if is_enabled():
                mode = get_mode()
                if mode == "cuda" and HAS_CUDA:
                    try:
                        from numba import cuda
                        dev_name = cuda.get_current_device().name
                        self.accel_label.setText(f"Accel: CUDA ({dev_name})")
                    except Exception:
                        self.accel_label.setText("Accel: CUDA")
                elif mode == "jit":
                    self.accel_label.setText("Accel: Numba JIT")
                else:
                    self.accel_label.setText("Accel: off")
            else:
                self.accel_label.setText("Accel: off")
        except ImportError:
            self.accel_label.setText("Accel: off")

    def refresh_theme(self):
        ss = _theme.perf_label_style()
        for lbl in self._labels:
            lbl.setStyleSheet(ss)
