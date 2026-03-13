"""Tests for MiniMap widget (montaris/widgets/minimap.py).

Covers: set_image, _rebuild_thumbnail, update_viewport, _pan_to,
mouse click, and resizeEvent debounced rebuild.
"""

import numpy as np
import pytest
from unittest.mock import patch
from PySide6.QtCore import Qt, QRectF, QPointF, QEvent
from PySide6.QtGui import QImage, QResizeEvent
from PySide6.QtWidgets import QApplication

from montaris.widgets.minimap import MiniMap, MINIMAP_SIZE


class TestMiniMapSetImage:
    @pytest.fixture
    def minimap(self, qapp):
        m = MiniMap()
        m.resize(200, MINIMAP_SIZE)
        yield m
        m.close()
        m.deleteLater()
        QApplication.processEvents()

    def test_set_image_none_clears_thumbnail(self, minimap):
        # First set an image, then clear it
        data = np.random.randint(0, 255, (100, 150), dtype=np.uint8)
        minimap.set_image(data)
        assert minimap._thumbnail is not None
        minimap.set_image(None)
        assert minimap._thumbnail is None

    def test_set_image_none_from_init(self, minimap):
        """set_image(None) on a fresh MiniMap should not crash."""
        minimap.set_image(None)
        assert minimap._thumbnail is None

    def test_set_image_stores_data(self, minimap):
        data = np.random.randint(0, 255, (80, 120), dtype=np.uint8)
        minimap.set_image(data)
        assert minimap._image_data is data


class TestRebuildThumbnail:
    @pytest.fixture
    def minimap(self, qapp):
        m = MiniMap()
        m.resize(200, MINIMAP_SIZE)
        yield m
        m.close()
        m.deleteLater()
        QApplication.processEvents()

    def test_grayscale_thumbnail_aspect_ratio(self, minimap):
        """Grayscale image: thumbnail maintains aspect ratio."""
        data = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        minimap.set_image(data)
        thumb = minimap._thumbnail
        assert thumb is not None
        tw = thumb.width()
        th = thumb.height()
        # Original aspect is 2:1 (200/100), thumbnail should approximate this
        thumb_aspect = tw / max(th, 1)
        original_aspect = 200 / 100
        assert abs(thumb_aspect - original_aspect) < 0.5

    def test_grayscale_uint16_thumbnail(self, minimap):
        """uint16 grayscale data is normalized to uint8 for thumbnail."""
        data = np.random.randint(0, 65535, (80, 80), dtype=np.uint16)
        minimap.set_image(data)
        assert minimap._thumbnail is not None

    def test_rgb_thumbnail(self, minimap):
        """RGB data produces a valid thumbnail."""
        data = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        minimap.set_image(data)
        assert minimap._thumbnail is not None
        tw = minimap._thumbnail.width()
        th = minimap._thumbnail.height()
        assert tw > 0 and th > 0

    def test_tall_image_aspect(self, minimap):
        """Tall image (aspect < 1): thumbnail height fills widget."""
        data = np.random.randint(0, 255, (400, 100), dtype=np.uint8)
        minimap.set_image(data)
        thumb = minimap._thumbnail
        assert thumb is not None
        # For tall images, th should be close to map height
        assert thumb.height() <= minimap.height()

    def test_no_data_noop(self, minimap):
        """_rebuild_thumbnail with no data should do nothing."""
        minimap._rebuild_thumbnail()
        assert minimap._thumbnail is None

    def test_constant_image(self, minimap):
        """Constant-value image doesn't crash (max == min edge case)."""
        data = np.full((64, 64), 42, dtype=np.uint8)
        minimap.set_image(data)
        assert minimap._thumbnail is not None

    def test_single_channel_3d(self, minimap):
        """(H, W, 1) array should produce a grayscale thumbnail."""
        data = np.random.randint(0, 255, (50, 80, 1), dtype=np.uint8)
        minimap.set_image(data)
        assert minimap._thumbnail is not None


class TestUpdateViewport:
    @pytest.fixture
    def minimap(self, qapp):
        m = MiniMap()
        m.resize(200, MINIMAP_SIZE)
        data = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        m.set_image(data)
        yield m
        m.close()
        m.deleteLater()
        QApplication.processEvents()

    def test_scale_computation(self, minimap):
        viewport = QRectF(50, 50, 100, 100)
        scene = QRectF(0, 0, 200, 200)
        minimap.update_viewport(viewport, scene)
        # scale_x = thumbnail_width / scene_width
        tw = minimap._thumbnail.width()
        th = minimap._thumbnail.height()
        expected_sx = tw / 200.0
        expected_sy = th / 200.0
        assert minimap._scale_x == pytest.approx(expected_sx)
        assert minimap._scale_y == pytest.approx(expected_sy)

    def test_viewport_stored(self, minimap):
        viewport = QRectF(10, 20, 80, 60)
        scene = QRectF(0, 0, 200, 200)
        minimap.update_viewport(viewport, scene)
        assert minimap._viewport_rect == viewport
        assert minimap._scene_rect == scene

    def test_zero_scene_no_crash(self, minimap):
        """Zero-size scene rect should not divide by zero."""
        viewport = QRectF(0, 0, 50, 50)
        scene = QRectF(0, 0, 0, 0)
        # Should not raise
        minimap.update_viewport(viewport, scene)


class TestPanTo:
    @pytest.fixture
    def minimap(self, qapp):
        m = MiniMap()
        m.resize(200, MINIMAP_SIZE)
        data = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        m.set_image(data)
        m.update_viewport(QRectF(0, 0, 100, 100), QRectF(0, 0, 200, 200))
        yield m
        m.close()
        m.deleteLater()
        QApplication.processEvents()

    def test_pan_requested_signal(self, minimap):
        received = []
        minimap.pan_requested.connect(lambda x, y: received.append((x, y)))
        # Click at center of widget
        cx = minimap.width() / 2.0
        cy = minimap.height() / 2.0
        minimap._pan_to(QPointF(cx, cy))
        assert len(received) == 1
        scene_x, scene_y = received[0]
        # Should map back to approximately the center of the scene
        assert isinstance(scene_x, float)
        assert isinstance(scene_y, float)

    def test_pan_to_no_thumbnail(self, minimap):
        """_pan_to with no thumbnail should be a no-op."""
        minimap._thumbnail = None
        received = []
        minimap.pan_requested.connect(lambda x, y: received.append((x, y)))
        minimap._pan_to(QPointF(50, 50))
        assert len(received) == 0

    def test_pan_to_null_scene_rect(self, minimap):
        """_pan_to with null scene rect should be a no-op."""
        minimap._scene_rect = QRectF()
        received = []
        minimap.pan_requested.connect(lambda x, y: received.append((x, y)))
        minimap._pan_to(QPointF(50, 50))
        assert len(received) == 0

    def test_pan_coords_mapping(self, minimap):
        """Verify scene coordinate mapping is reasonable."""
        received = []
        minimap.pan_requested.connect(lambda x, y: received.append((x, y)))
        # Click at top-left of thumbnail area
        x_off = (minimap.width() - minimap._thumbnail.width()) // 2
        y_off = (minimap.height() - minimap._thumbnail.height()) // 2
        minimap._pan_to(QPointF(float(x_off), float(y_off)))
        assert len(received) == 1
        # At top-left of thumbnail: should map to scene origin (0, 0)
        sx, sy = received[0]
        assert sx == pytest.approx(0.0, abs=5.0)
        assert sy == pytest.approx(0.0, abs=5.0)


class TestMouseClick:
    @pytest.fixture
    def minimap(self, qapp):
        m = MiniMap()
        m.resize(200, MINIMAP_SIZE)
        data = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        m.set_image(data)
        m.update_viewport(QRectF(0, 0, 100, 100), QRectF(0, 0, 200, 200))
        yield m
        m.close()
        m.deleteLater()
        QApplication.processEvents()

    def test_mouse_press_emits_pan(self, minimap):
        received = []
        minimap.pan_requested.connect(lambda x, y: received.append((x, y)))
        # Simulate mouse press using QTest
        from PySide6.QtTest import QTest
        QTest.mouseClick(minimap, Qt.LeftButton, pos=QPointF(100, 100).toPoint())
        assert len(received) >= 1


class TestResizeEvent:
    @pytest.fixture
    def minimap(self, qapp):
        m = MiniMap()
        m.resize(200, MINIMAP_SIZE)
        data = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        m.set_image(data)
        yield m
        m.close()
        m.deleteLater()
        QApplication.processEvents()

    def _trigger_resize(self, minimap, new_w, new_h):
        """Directly invoke resizeEvent to work in offscreen mode."""
        from PySide6.QtCore import QSize
        old_size = minimap.size()
        minimap.resize(new_w, new_h)
        event = QResizeEvent(QSize(new_w, new_h), old_size)
        minimap.resizeEvent(event)

    def test_resize_creates_timer(self, minimap):
        """resizeEvent should create a debounce timer."""
        self._trigger_resize(minimap, 250, MINIMAP_SIZE)
        assert hasattr(minimap, '_resize_timer')
        assert minimap._resize_timer.isSingleShot()

    def test_resize_triggers_rebuild(self, minimap):
        """After debounce timer fires, thumbnail is rebuilt."""
        self._trigger_resize(minimap, 300, MINIMAP_SIZE)
        # Force the timer to fire immediately
        if hasattr(minimap, '_resize_timer') and minimap._resize_timer.isActive():
            minimap._resize_timer.stop()
            minimap._rebuild_thumbnail()
        new_thumb = minimap._thumbnail
        assert new_thumb is not None

    def test_debounce_interval(self, minimap):
        """Timer interval should be 100ms as defined in source."""
        self._trigger_resize(minimap, 250, MINIMAP_SIZE)
        assert minimap._resize_timer.interval() == 100
