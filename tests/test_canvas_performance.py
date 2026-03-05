"""Tests for canvas rendering, overlays, and dirty-region tracking."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from montaris.layers import LayerStack, ImageLayer, ROILayer
from montaris.canvas import ImageCanvas
from montaris.app import MontarisApp, apply_dark_theme


# ── Ensure a QApplication exists ─────────────────────────────────────
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setApplicationName("Montaris-X-Test")
        apply_dark_theme(app)
    yield app


# =====================================================================
# Canvas image display
# =====================================================================


class TestCanvasRefresh:
    """Verify that refresh_image works with single-pixmap display."""

    @pytest.fixture
    def canvas(self, qapp):
        stack = LayerStack()
        c = ImageCanvas(stack)
        c.resize(800, 600)
        c.show()
        yield c
        c.close()

    def test_refresh_with_small_image(self, canvas):
        data = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
        canvas.layer_stack.set_image(ImageLayer("small", data))
        canvas.refresh_image()
        assert canvas._image_item is not None

    def test_refresh_with_medium_image(self, canvas):
        data = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        canvas.layer_stack.set_image(ImageLayer("medium", data))
        canvas.refresh_image()
        assert canvas._image_item is not None

    def test_refresh_with_rgb_image(self, canvas):
        data = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        canvas.layer_stack.set_image(ImageLayer("rgb", data))
        canvas.refresh_image()
        assert canvas._image_item is not None

    def test_refresh_replaces_old_image(self, canvas):
        data1 = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
        canvas.layer_stack.set_image(ImageLayer("img1", data1))
        canvas.refresh_image()
        old_item = canvas._image_item

        data2 = np.random.randint(0, 255, (200, 250), dtype=np.uint8)
        canvas.layer_stack.set_image(ImageLayer("img2", data2))
        canvas.refresh_image()
        assert canvas._image_item is not old_item

    def test_fit_to_window(self, canvas):
        data = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
        canvas.layer_stack.set_image(ImageLayer("small", data))
        canvas.refresh_image()
        canvas.fit_to_window()
        assert canvas._image_item is not None

    def test_reset_zoom(self, canvas):
        data = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
        canvas.layer_stack.set_image(ImageLayer("small", data))
        canvas.refresh_image()
        canvas.reset_zoom()
        assert canvas._image_item is not None


class TestCanvasOverlays:
    """Verify overlay rendering works."""

    @pytest.fixture
    def canvas_with_image(self, qapp):
        stack = LayerStack()
        c = ImageCanvas(stack)
        c.resize(800, 600)
        c.show()
        data = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
        stack.set_image(ImageLayer("test", data))
        c.refresh_image()
        roi = ROILayer("ROI 1", 120, 100)
        stack.add_roi(roi)
        c.refresh_overlays()
        yield c
        c.close()

    def test_refresh_overlays(self, canvas_with_image):
        c = canvas_with_image
        assert len(c._roi_items) == 1

    def test_refresh_overlays_after_hide(self, canvas_with_image):
        c = canvas_with_image
        roi = c.layer_stack.roi_layers[0]
        roi.visible = False
        c.refresh_overlays()
        assert len(c._roi_items) == 0


# =====================================================================
# Dirty region tracking on ROILayer
# =====================================================================


class TestDirtyRegion:
    def test_initial_state_clean(self):
        roi = ROILayer("test", 100, 80)
        assert roi.dirty_rect is None

    def test_mark_dirty_single(self):
        roi = ROILayer("test", 100, 80)
        roi.mark_dirty((10, 20, 30, 40))
        assert roi.dirty_rect == (10, 20, 30, 40)

    def test_mark_dirty_union(self):
        roi = ROILayer("test", 100, 80)
        roi.mark_dirty((10, 10, 20, 20))
        roi.mark_dirty((50, 50, 20, 20))
        x, y, w, h = roi.dirty_rect
        assert x == 10
        assert y == 10
        assert x + w == 70
        assert y + h == 70

    def test_clear_dirty(self):
        roi = ROILayer("test", 100, 80)
        roi.mark_dirty((10, 20, 30, 40))
        roi.clear_dirty()
        assert roi.dirty_rect is None


# =====================================================================
# Small-image regression
# =====================================================================


class TestSmallImageRegression:
    """Ensure the full app flow still works with small images."""

    @pytest.fixture
    def app(self, qapp):
        window = MontarisApp()
        window.show()
        yield window
        window.close()

    def test_full_app_flow(self, app):
        data = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
        app.layer_stack.set_image(ImageLayer("small", data))
        app.canvas.refresh_image()
        app.canvas.fit_to_window()
        assert app.canvas._image_item is not None
        # Add ROI
        roi = ROILayer("ROI 1", 120, 100)
        app.layer_stack.add_roi(roi)
        app.canvas.set_active_layer(roi)
        app.canvas.refresh_overlays()
        assert len(app.canvas._roi_items) == 1

    def test_tile_pyramid_via_image_layer(self):
        data = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
        layer = ImageLayer("test", data)
        p = layer.tile_pyramid
        assert p.num_levels >= 1
        tile = p.get_tile(0, 0, 0)
        assert tile is not None
