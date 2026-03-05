import numpy as np
import pytest
from montaris.widgets.minimap import MiniMap
from montaris.widgets.perf_monitor import PerfMonitor
from montaris.widgets.debug_console import DebugConsole
from PySide6.QtCore import QRectF


class TestMiniMap:
    def test_create(self, qapp):
        mm = MiniMap()
        assert mm.width() == 200
        assert mm.height() == 200

    def test_set_image(self, qapp):
        mm = MiniMap()
        data = np.random.randint(0, 255, (100, 150), dtype=np.uint8)
        mm.set_image(data)
        assert mm._thumbnail is not None

    def test_set_rgb_image(self, qapp):
        mm = MiniMap()
        data = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        mm.set_image(data)
        assert mm._thumbnail is not None

    def test_update_viewport(self, qapp):
        mm = MiniMap()
        data = np.random.randint(0, 255, (100, 150), dtype=np.uint8)
        mm.set_image(data)
        mm.update_viewport(
            QRectF(10, 10, 50, 50),
            QRectF(0, 0, 150, 100),
        )

    def test_set_none_image(self, qapp):
        mm = MiniMap()
        mm.set_image(None)
        assert mm._thumbnail is None


class TestPerfMonitor:
    def test_create(self, qapp):
        pm = PerfMonitor()
        assert pm.fps_label.text() == "FPS: -"

    def test_record_frame(self, qapp):
        pm = PerfMonitor()
        pm.record_frame()
        pm._update_stats()

    def test_record_render_time(self, qapp):
        pm = PerfMonitor()
        pm.record_render_time(16.5)
        pm._update_stats()
        assert "16.5" in pm.render_label.text()

    def test_tile_cache_info(self, qapp):
        pm = PerfMonitor()
        pm.set_tile_cache_info("50/500 tiles")
        pm._update_stats()
        assert "50/500" in pm.tile_cache_label.text()


class TestDebugConsole:
    def test_create(self, qapp):
        dc = DebugConsole()
        assert dc.eval_input is not None

    def test_log(self, qapp):
        dc = DebugConsole()
        dc.log("Test message")
        assert "Test message" in dc.log_output.toPlainText()

    def test_eval(self, qapp):
        dc = DebugConsole()
        dc.eval_input.setText("1 + 1")
        dc._on_eval()
        text = dc.log_output.toPlainText()
        assert "2" in text
