import time
import psutil
import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import QTimer


class PerfMonitor(QWidget):
    """Performance monitor showing FPS, render time, memory usage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.fps_label = QLabel("FPS: -")
        layout.addWidget(self.fps_label)

        self.render_label = QLabel("Render: - ms")
        layout.addWidget(self.render_label)

        self.memory_label = QLabel("Memory: - MB")
        layout.addWidget(self.memory_label)

        self.tile_cache_label = QLabel("Tile Cache: -")
        layout.addWidget(self.tile_cache_label)

        cpu_count = os.cpu_count() or 1
        self.cpu_label = QLabel(f"CPU Cores: {cpu_count} (using 1)")
        layout.addWidget(self.cpu_label)

        layout.addStretch()

        self._frame_times = []
        self._render_times = []
        self._tile_cache_info = ""

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_stats)
        self._timer.start(1000)

    def record_frame(self):
        self._frame_times.append(time.time())

    def record_render_time(self, ms):
        self._render_times.append(ms)

    def set_tile_cache_info(self, info):
        self._tile_cache_info = info

    def _update_stats(self):
        now = time.time()
        # FPS: count frames in last second
        self._frame_times = [t for t in self._frame_times if now - t < 1.0]
        fps = len(self._frame_times)
        self.fps_label.setText(f"FPS: {fps}")

        # Render: average of recent render times
        if self._render_times:
            avg = sum(self._render_times) / len(self._render_times)
            self.render_label.setText(f"Render: {avg:.1f} ms")
            self._render_times.clear()
        else:
            self.render_label.setText("Render: idle")

        # Memory
        try:
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            self.memory_label.setText(f"Memory: {mem_mb:.0f} MB")
        except Exception:
            self.memory_label.setText("Memory: N/A")

        self.tile_cache_label.setText(f"Tile Cache: {self._tile_cache_info or '-'}")
