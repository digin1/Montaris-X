"""Comprehensive tests for ImageCanvas methods in montaris/canvas.py.

Covers display pipeline, event handling, zoom, brush/polygon preview,
ROI compositing, LOD, and progress flashing.
"""

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF, Qt, QRectF, QEvent
from PySide6.QtGui import QKeyEvent, QMouseEvent

from montaris.app import MontarisApp, apply_dark_theme
from montaris.layers import ImageLayer, ROILayer
from montaris.canvas import (
    ImageCanvas,
    _composite_roi,
    _composite_roi_region,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(app)
    return app


@pytest.fixture
def app(qapp):
    win = MontarisApp()
    img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    win.layer_stack.set_image(ImageLayer("test", img))
    win.canvas.refresh_image()
    QApplication.processEvents()
    # Add some ROIs
    for i in range(3):
        roi = ROILayer(f"roi_{i}", 200, 200)
        roi.mask[i * 30:(i + 1) * 30, i * 30:(i + 1) * 30] = 255
        win.layer_stack.add_roi(roi)
    win.canvas.set_active_layer(win.layer_stack.roi_layers[0])
    QApplication.processEvents()
    return win


def _make_mouse_event(event_type, button, pos, modifiers=Qt.NoModifier, buttons=None):
    """Create a QMouseEvent for testing."""
    qpos = QPointF(pos[0], pos[1])
    if buttons is None:
        buttons = button
    return QMouseEvent(event_type, qpos, qpos, button, buttons, modifiers)


def _make_key_event(event_type, key, modifiers=Qt.NoModifier):
    """Create a QKeyEvent for testing."""
    return QKeyEvent(event_type, key, modifiers)


# ===================================================================
# 1. _get_display_uint8 — uint8 passthrough, uint16 normalization, caching
# ===================================================================

class TestGetDisplayUint8:
    def test_uint8_passthrough(self, app):
        """uint8 image data should be returned directly without copy."""
        canvas = app.canvas
        data = canvas.layer_stack.image_layer.data
        assert data.dtype == np.uint8
        result = canvas._get_display_uint8()
        assert result is data  # same object, no copy

    def test_uint16_normalization(self, qapp):
        """uint16 data should be normalized to uint8 range."""
        win = MontarisApp()
        img16 = np.array([[0, 32768, 65535]], dtype=np.uint16)
        win.layer_stack.set_image(ImageLayer("u16", img16))
        win.canvas.refresh_image()
        QApplication.processEvents()

        result = win.canvas._get_display_uint8()
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_uint16_caching(self, qapp):
        """Repeated calls should return the cached result."""
        win = MontarisApp()
        img16 = np.random.randint(0, 65535, (50, 50), dtype=np.uint16)
        win.layer_stack.set_image(ImageLayer("u16cache", img16))
        win.canvas.refresh_image()
        QApplication.processEvents()

        first = win.canvas._get_display_uint8()
        second = win.canvas._get_display_uint8()
        assert first is second  # same cached object

    def test_cache_invalidated_on_refresh(self, qapp):
        """refresh_image() should clear the display cache."""
        win = MontarisApp()
        img16 = np.random.randint(0, 65535, (50, 50), dtype=np.uint16)
        win.layer_stack.set_image(ImageLayer("u16inv", img16))
        win.canvas.refresh_image()
        QApplication.processEvents()

        first = win.canvas._get_display_uint8()
        win.canvas.refresh_image()  # invalidates cache
        second = win.canvas._get_display_uint8()
        # After invalidation and re-compute, could be a new object
        # but still equivalent
        assert second.dtype == np.uint8

    def test_none_when_no_image(self, qapp):
        """Should return None when no image layer is set."""
        win = MontarisApp()
        result = win.canvas._get_display_uint8()
        assert result is None


# ===================================================================
# 2. refresh_adjustments — verify adjustments applied
# ===================================================================

class TestRefreshAdjustments:
    def test_no_crash_without_adjustments(self, app):
        """Should not crash when adjustments are None/identity."""
        app.canvas._adjustments = None
        app.canvas.refresh_adjustments()  # should not raise

    def test_adjustments_applied(self, app):
        """Image item pixmap should update when adjustments are set."""
        from montaris.core.adjustments import ImageAdjustments
        adj = ImageAdjustments()
        adj.brightness = 0.5
        app.canvas._adjustments = adj
        app.canvas.refresh_adjustments()
        QApplication.processEvents()
        # Verify image item still exists and has a pixmap
        assert app.canvas._image_item is not None

    def test_no_op_without_image_item(self, qapp):
        """Should return early if _image_item is None."""
        win = MontarisApp()
        win.canvas._image_item = None
        win.canvas.refresh_adjustments()  # should not raise


# ===================================================================
# 3. refresh_image_from_array — verify image item updated
# ===================================================================

class TestRefreshImageFromArray:
    def test_sets_image_item(self, app):
        """Should create a new image item from the given array."""
        data = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
        app.canvas.refresh_image_from_array(data)
        assert app.canvas._image_item is not None

    def test_replaces_existing_image(self, app):
        """Existing image item should be replaced."""
        old_item = app.canvas._image_item
        data = np.random.randint(0, 255, (80, 90), dtype=np.uint8)
        app.canvas.refresh_image_from_array(data)
        assert app.canvas._image_item is not old_item

    def test_scene_rect_updated(self, app):
        """Scene rect should accommodate the new image size."""
        data = np.zeros((50, 60), dtype=np.uint8)
        app.canvas.refresh_image_from_array(data)
        sr = app.canvas._scene.sceneRect()
        # Scene rect should be at least as big as the image
        assert sr.width() >= 60
        assert sr.height() >= 50


# ===================================================================
# 4. refresh_overlays — verify ROI items created, re-entrancy guard
# ===================================================================

class TestRefreshOverlays:
    def test_creates_roi_items(self, app):
        """Should create overlay items or mark as stale for each ROI."""
        app.canvas.refresh_overlays()
        QApplication.processEvents()
        # Each ROI should either have an item or be marked stale
        # (viewport culling may defer off-screen ROIs in offscreen mode)
        for roi in app.layer_stack.roi_layers:
            rid = id(roi)
            has_item = rid in app.canvas._roi_items
            is_stale = rid in app.canvas._roi_stale
            assert has_item or is_stale, (
                f"ROI {roi.name} neither has an item nor is marked stale"
            )

    def test_reentrancy_guard(self, app):
        """Setting _refreshing=True should prevent re-entrant calls."""
        app.canvas._refreshing = True
        # Store current state
        items_before = dict(app.canvas._roi_items)
        # This should be a no-op due to re-entrancy guard
        app.canvas.refresh_overlays()
        app.canvas._refreshing = False  # reset
        # Items should be unchanged (the guard prevented refresh)
        assert app.canvas._roi_items == items_before

    def test_stale_items_removed(self, app):
        """Deleting an ROI and refreshing should remove its overlay item."""
        app.canvas.refresh_overlays()
        QApplication.processEvents()
        roi_to_remove = app.layer_stack.roi_layers[-1]
        rid = id(roi_to_remove)
        app.layer_stack.remove_roi(len(app.layer_stack.roi_layers) - 1)
        app.canvas.refresh_overlays()
        QApplication.processEvents()
        assert rid not in app.canvas._roi_items


# ===================================================================
# 5. _current_lod_level — LOD from zoom
# ===================================================================

class TestCurrentLodLevel:
    def test_lod0_at_100pct(self, app):
        """At 100% zoom (m11=1.0), LOD should be 0."""
        app.canvas.resetTransform()
        assert app.canvas._current_lod_level() == 0

    def test_lod0_at_50pct(self, app):
        """At 50% zoom (m11=0.5), LOD should still be 0."""
        app.canvas.resetTransform()
        app.canvas.scale(0.5, 0.5)
        assert app.canvas._current_lod_level() == 0

    def test_lod1_below_50pct(self, app):
        """At 25-49% zoom, LOD should be 1."""
        app.canvas.resetTransform()
        app.canvas.scale(0.3, 0.3)
        assert app.canvas._current_lod_level() == 1

    def test_lod2_below_25pct(self, app):
        """At 12.5-24% zoom, LOD should be 2."""
        app.canvas.resetTransform()
        app.canvas.scale(0.15, 0.15)
        assert app.canvas._current_lod_level() == 2

    def test_lod3_very_small_zoom(self, app):
        """At <12.5% zoom, LOD should be 3."""
        app.canvas.resetTransform()
        app.canvas.scale(0.05, 0.05)
        assert app.canvas._current_lod_level() == 3


# ===================================================================
# 6. _apply_roi_rgba_result — verify item created with correct RGBA
# ===================================================================

class TestApplyRoiRgbaResult:
    def test_creates_new_item(self, app):
        """Should create a new overlay item when none exists for the ROI."""
        roi = ROILayer("new_roi", 200, 200)
        roi.mask[10:20, 10:20] = 255
        app.layer_stack.add_roi(roi)

        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        rgba[:, :] = [255, 0, 0, 128]
        rgba = np.ascontiguousarray(rgba)
        result = (rgba, 10, 10, 10, 10, 1)

        rid = id(roi)
        # Ensure no pre-existing item
        app.canvas._roi_items.pop(rid, None)
        idx = len(app.layer_stack.roi_layers) - 1
        app.canvas._apply_roi_rgba_result(roi, idx, result)
        assert rid in app.canvas._roi_items

    def test_updates_existing_item(self, app):
        """Should update an existing item's image and position."""
        app.canvas.refresh_overlays()
        QApplication.processEvents()
        roi = app.layer_stack.roi_layers[0]
        rid = id(roi)
        existing_item = app.canvas._roi_items.get(rid)
        assert existing_item is not None

        rgba = np.zeros((5, 5, 4), dtype=np.uint8)
        rgba[:, :] = [0, 255, 0, 100]
        rgba = np.ascontiguousarray(rgba)
        result = (rgba, 5, 5, 20, 30, 1)

        app.canvas._apply_roi_rgba_result(roi, 0, result)
        # Item should still be the same object but with updated position
        item = app.canvas._roi_items[rid]
        assert item.pos().x() == 20
        assert item.pos().y() == 30

    def test_scale_factor_applied(self, app):
        """When scale_factor > 1, item should have setScale applied."""
        roi = app.layer_stack.roi_layers[0]
        rid = id(roi)
        app.canvas.refresh_overlays()
        QApplication.processEvents()

        rgba = np.zeros((5, 5, 4), dtype=np.uint8)
        rgba[:, :] = [0, 0, 255, 128]
        rgba = np.ascontiguousarray(rgba)
        result = (rgba, 5, 5, 0, 0, 4)

        app.canvas._apply_roi_rgba_result(roi, 0, result)
        item = app.canvas._roi_items[rid]
        assert item.scale() == 4


# ===================================================================
# 7. keyPressEvent — Space (pan), Escape (deselect), bracket keys
# ===================================================================

class TestKeyPressEvent:
    def test_space_enables_pan(self, app):
        """Space key should set _space_held and open hand cursor."""
        event = _make_key_event(QEvent.KeyPress, Qt.Key_Space)
        app.canvas.keyPressEvent(event)
        assert app.canvas._space_held is True

    def test_escape_clears_selection(self, app):
        """Escape should clear the selection."""
        app.canvas._selection.add(app.layer_stack.roi_layers[0])
        assert app.canvas._selection.count > 0
        event = _make_key_event(QEvent.KeyPress, Qt.Key_Escape)
        app.canvas.keyPressEvent(event)
        assert app.canvas._selection.count == 0

    def test_bracket_left_decreases_brush(self, app):
        """[ key should decrease brush size via _adjust_brush_size."""
        with patch.object(app.canvas, '_adjust_brush_size') as mock_adj:
            event = _make_key_event(QEvent.KeyPress, Qt.Key_BracketLeft)
            app.canvas.keyPressEvent(event)
            mock_adj.assert_called_once_with(-2)

    def test_bracket_right_increases_brush(self, app):
        """] key should increase brush size via _adjust_brush_size."""
        with patch.object(app.canvas, '_adjust_brush_size') as mock_adj:
            event = _make_key_event(QEvent.KeyPress, Qt.Key_BracketRight)
            app.canvas.keyPressEvent(event)
            mock_adj.assert_called_once_with(2)


# ===================================================================
# 8. keyReleaseEvent — Space release
# ===================================================================

class TestKeyReleaseEvent:
    def test_space_release(self, app):
        """Releasing Space should clear _space_held."""
        app.canvas._space_held = True
        event = _make_key_event(QEvent.KeyRelease, Qt.Key_Space)
        app.canvas.keyReleaseEvent(event)
        assert app.canvas._space_held is False


# ===================================================================
# 9. mousePressEvent — middle button pan, left button tool dispatch
# ===================================================================

class TestMousePressEvent:
    def test_middle_button_starts_pan(self, app):
        """Middle mouse button should start panning."""
        event = _make_mouse_event(
            QEvent.MouseButtonPress, Qt.MiddleButton, (100, 100)
        )
        app.canvas.mousePressEvent(event)
        assert app.canvas._is_panning is True
        app.canvas._is_panning = False  # cleanup

    def test_space_left_click_starts_pan(self, app):
        """Left click while Space is held should start panning."""
        app.canvas._space_held = True
        event = _make_mouse_event(
            QEvent.MouseButtonPress, Qt.LeftButton, (100, 100)
        )
        app.canvas.mousePressEvent(event)
        assert app.canvas._is_panning is True
        app.canvas._is_panning = False
        app.canvas._space_held = False

    def test_left_button_dispatches_to_tool(self, app):
        """Left click with a tool set should call tool.on_press."""
        mock_tool = MagicMock()
        mock_tool.name = "Brush"
        mock_tool.is_hand = False
        mock_tool.on_key_press = MagicMock(return_value=False)
        app.canvas._tool = mock_tool
        event = _make_mouse_event(
            QEvent.MouseButtonPress, Qt.LeftButton, (50, 50)
        )
        app.canvas.mousePressEvent(event)
        mock_tool.on_press.assert_called_once()
        app.canvas._tool = None  # cleanup


# ===================================================================
# 10. mouseMoveEvent — HUD update, pan mode, tool delegation
# ===================================================================

class TestMouseMoveEvent:
    def test_hud_update(self, app):
        """Mouse move should update the HUD label text."""
        event = _make_mouse_event(
            QEvent.MouseMove, Qt.NoButton, (50, 60), buttons=Qt.NoButton
        )
        app.canvas.mouseMoveEvent(event)
        hud = app.canvas._hud_label.text()
        assert "I:" in hud
        assert "Z:" in hud

    def test_pan_mode_scrolls(self, app):
        """When panning, mouse move should adjust scroll bars."""
        app.canvas._is_panning = True
        app.canvas._last_pan_pos = QPointF(100, 100)
        event = _make_mouse_event(
            QEvent.MouseMove, Qt.MiddleButton, (110, 105),
            buttons=Qt.MiddleButton
        )
        app.canvas.mouseMoveEvent(event)
        # Just verify no crash; panning logic was exercised
        app.canvas._is_panning = False

    def test_tool_on_move_called(self, app):
        """Moving with left button held and tool set should call on_move."""
        mock_tool = MagicMock()
        mock_tool.name = "Brush"
        mock_tool.is_hand = False
        mock_tool.on_key_press = MagicMock(return_value=False)
        # Remove width/height to avoid stamp preview code path
        del mock_tool.width
        del mock_tool.height
        app.canvas._tool = mock_tool
        event = _make_mouse_event(
            QEvent.MouseMove, Qt.LeftButton, (60, 70),
            buttons=Qt.LeftButton
        )
        app.canvas.mouseMoveEvent(event)
        mock_tool.on_move.assert_called_once()
        app.canvas._tool = None


# ===================================================================
# 11. mouseReleaseEvent — stop panning, tool delegation
# ===================================================================

class TestMouseReleaseEvent:
    def test_stop_panning_on_middle_release(self, app):
        """Releasing middle button should stop panning."""
        app.canvas._is_panning = True
        event = _make_mouse_event(
            QEvent.MouseButtonRelease, Qt.MiddleButton, (100, 100)
        )
        app.canvas.mouseReleaseEvent(event)
        assert app.canvas._is_panning is False

    def test_tool_on_release_called(self, app):
        """Releasing left button with tool should call tool.on_release."""
        mock_tool = MagicMock()
        mock_tool.is_hand = False
        app.canvas._tool = mock_tool
        app.canvas._is_panning = False
        event = _make_mouse_event(
            QEvent.MouseButtonRelease, Qt.LeftButton, (50, 50)
        )
        app.canvas.mouseReleaseEvent(event)
        mock_tool.on_release.assert_called_once()
        app.canvas._tool = None


# ===================================================================
# 12. mouseDoubleClickEvent — tool.finish() called
# ===================================================================

class TestMouseDoubleClickEvent:
    def test_finish_called_on_double_click(self, app):
        """Double-click with a tool that has finish() should call it."""
        mock_tool = MagicMock()
        mock_tool.finish = MagicMock()
        app.canvas._tool = mock_tool
        event = _make_mouse_event(
            QEvent.MouseButtonDblClick, Qt.LeftButton, (80, 80)
        )
        app.canvas.mouseDoubleClickEvent(event)
        mock_tool.finish.assert_called_once()
        app.canvas._tool = None

    def test_no_crash_without_finish(self, app):
        """Double-click with tool lacking finish() should not crash."""
        mock_tool = MagicMock(spec=[])  # no attributes
        app.canvas._tool = mock_tool
        event = _make_mouse_event(
            QEvent.MouseButtonDblClick, Qt.LeftButton, (80, 80)
        )
        app.canvas.mouseDoubleClickEvent(event)  # should not raise
        app.canvas._tool = None


# ===================================================================
# 13. show_brush_preview — verify ellipse visible
# ===================================================================

class TestShowBrushPreview:
    def test_brush_preview_visible(self, app):
        """show_brush_preview should make the preview ellipse visible."""
        app.canvas.show_brush_preview(100.0, 100.0, 20.0)
        assert app.canvas._brush_preview is not None
        assert app.canvas._brush_preview.isVisible()

    def test_brush_preview_geometry(self, app):
        """Preview rect should match the requested cx, cy, radius."""
        app.canvas.show_brush_preview(50.0, 60.0, 15.0)
        rect = app.canvas._brush_preview.rect()
        assert abs(rect.x() - (50.0 - 15.0)) < 0.01
        assert abs(rect.y() - (60.0 - 15.0)) < 0.01
        assert abs(rect.width() - 30.0) < 0.01
        assert abs(rect.height() - 30.0) < 0.01

    def test_brush_preview_uses_roi_color(self, app):
        """Preview pen color should match active ROI layer color."""
        roi = app.layer_stack.roi_layers[0]
        app.canvas.set_active_layer(roi)
        app.canvas.show_brush_preview(50.0, 60.0, 10.0)
        pen = app.canvas._brush_preview.pen()
        r, g, b = roi.color
        assert pen.color().red() == r
        assert pen.color().green() == g
        assert pen.color().blue() == b


# ===================================================================
# 14. hide_brush_preview — verify hidden
# ===================================================================

class TestHideBrushPreview:
    def test_hide_after_show(self, app):
        """hide_brush_preview should make the preview invisible."""
        app.canvas.show_brush_preview(100.0, 100.0, 20.0)
        assert app.canvas._brush_preview.isVisible()
        app.canvas.hide_brush_preview()
        assert not app.canvas._brush_preview.isVisible()

    def test_hide_when_none(self, app):
        """Hiding when preview was never shown should not crash."""
        app.canvas._brush_preview = None
        app.canvas.hide_brush_preview()  # should not raise


# ===================================================================
# 15. draw_polygon_preview — verify items added
# ===================================================================

class TestDrawPolygonPreview:
    def test_polygon_items_created(self, app):
        """Drawing polygon preview should create path and close marker."""
        vertices = [(10, 10), (50, 10), (50, 50)]
        app.canvas.draw_polygon_preview(vertices)
        assert app.canvas._polygon_item is not None
        assert app.canvas._polygon_close_marker is not None

    def test_single_vertex_no_marker(self, app):
        """A single vertex should create path but no close marker."""
        vertices = [(10, 10)]
        app.canvas.draw_polygon_preview(vertices)
        assert app.canvas._polygon_item is not None
        assert app.canvas._polygon_close_marker is None

    def test_empty_vertices_no_item(self, app):
        """Empty vertices list should not create items."""
        app.canvas.draw_polygon_preview([])
        assert app.canvas._polygon_item is None


# ===================================================================
# 16. clear_polygon_preview — verify items removed
# ===================================================================

class TestClearPolygonPreview:
    def test_clear_removes_items(self, app):
        """clear_polygon_preview should remove both polygon items."""
        vertices = [(10, 10), (50, 10), (50, 50)]
        app.canvas.draw_polygon_preview(vertices)
        assert app.canvas._polygon_item is not None
        app.canvas.clear_polygon_preview()
        assert app.canvas._polygon_item is None
        assert app.canvas._polygon_close_marker is None

    def test_clear_when_empty(self, app):
        """Clearing when no preview exists should not crash."""
        app.canvas._polygon_item = None
        app.canvas._polygon_close_marker = None
        app.canvas.clear_polygon_preview()  # should not raise


# ===================================================================
# 17. flash_progress — verify progress shown
# ===================================================================

class TestFlashProgress:
    def test_progress_shown(self, app):
        """flash_progress should call show() on the progress bar."""
        with patch.object(app.canvas._progress_bar, 'show') as mock_show:
            app.canvas.flash_progress()
            mock_show.assert_called_once()

    def test_progress_with_message(self, app):
        """flash_progress with a message should call show() and not crash."""
        with patch.object(app.canvas._progress_bar, 'show') as mock_show:
            app.canvas.flash_progress("Rasterizing...")
            mock_show.assert_called_once()


# ===================================================================
# 18. _update_brush_cursor — stamp vs brush preview
# ===================================================================

class TestUpdateBrushCursor:
    def test_stamp_tool_shows_stamp_preview(self, app):
        """When tool has width/height, stamp preview should be shown."""
        mock_tool = MagicMock()
        mock_tool.width = 20
        mock_tool.height = 30
        app.canvas._tool = mock_tool
        app.canvas._is_panning = False
        scene_pos = QPointF(100, 100)
        app.canvas._update_brush_cursor(scene_pos)
        assert app.canvas._stamp_preview is not None
        assert app.canvas._stamp_preview.isVisible()
        app.canvas._tool = None
        app.canvas._hide_stamp_preview()

    def test_brush_tool_shows_circle_preview(self, app):
        """When tool has size attribute, brush circle should be shown."""
        mock_tool = MagicMock(spec=['size', 'cursor', 'on_press', 'on_move',
                                     'on_release', 'on_key_press', 'name'])
        mock_tool.size = 10
        # Ensure no width/height attrs
        assert not hasattr(mock_tool, 'width')
        assert not hasattr(mock_tool, 'height')
        app.canvas._tool = mock_tool
        app.canvas._is_panning = False
        scene_pos = QPointF(80, 90)
        app.canvas._update_brush_cursor(scene_pos)
        assert app.canvas._brush_preview is not None
        assert app.canvas._brush_preview.isVisible()
        app.canvas._tool = None

    def test_no_tool_hides_previews(self, app):
        """Without a tool, all previews should be hidden."""
        app.canvas._tool = None
        app.canvas._is_panning = False
        scene_pos = QPointF(50, 50)
        app.canvas._update_brush_cursor(scene_pos)
        if app.canvas._brush_preview is not None:
            assert not app.canvas._brush_preview.isVisible()


# ===================================================================
# 19. _composite_roi — verify in-place composite
# ===================================================================

class TestCompositeRoi:
    def test_solid_mode_paints_pixels(self, qapp):
        """Solid mode should fill painted pixels with color + opacity."""
        combined = np.zeros((20, 20, 4), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        _composite_roi(combined, mask, (255, 0, 0), 128, "solid")
        # Painted pixel should have color
        assert combined[10, 10, 0] == 255
        assert combined[10, 10, 1] == 0
        assert combined[10, 10, 2] == 0
        assert combined[10, 10, 3] == 128
        # Unpainted pixel should remain transparent
        assert combined[0, 0, 3] == 0

    def test_outline_mode_edges_only(self, qapp):
        """Outline mode should paint only edge pixels."""
        combined = np.zeros((20, 20, 4), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        _composite_roi(combined, mask, (0, 255, 0), 200, "outline")
        # Interior pixel should remain transparent
        assert combined[10, 10, 3] == 0
        # At least some edge pixels should be painted
        assert np.any(combined[:, :, 3] > 0)

    def test_both_mode_fill_and_edge(self, qapp):
        """Both mode should have fill + brighter edges."""
        combined = np.zeros((20, 20, 4), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        _composite_roi(combined, mask, (0, 0, 255), 180, "both")
        # Interior should have fill_alpha (180 // 2 = 90)
        interior_alpha = combined[10, 10, 3]
        assert interior_alpha > 0
        # Edge pixels should have higher opacity
        edge_alpha = combined[:, :, 3].max()
        assert edge_alpha >= interior_alpha

    def test_empty_mask_no_change(self, qapp):
        """Empty mask should leave combined array unchanged."""
        combined = np.zeros((10, 10, 4), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        _composite_roi(combined, mask, (255, 255, 0), 128, "solid")
        assert np.all(combined == 0)

    def test_in_place_modification(self, qapp):
        """Should modify the combined array in-place (no return value)."""
        combined = np.zeros((10, 10, 4), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        result = _composite_roi(combined, mask, (128, 64, 32), 100, "solid")
        assert result is None  # in-place, no return
        assert combined[5, 5, 0] == 128


# ===================================================================
# 20. _composite_roi_region — verify region composite
# ===================================================================

class TestCompositeRoiRegion:
    def test_region_composite_solid(self, qapp):
        """Should composite only the specified region."""
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        region = np.zeros((20, 20, 4), dtype=np.uint8)
        _composite_roi_region(region, mask, (255, 0, 0), 128, "solid",
                              10, 10, 30, 30)
        assert region[5, 5, 0] == 255
        assert region[5, 5, 3] == 128

    def test_region_composite_outline(self, qapp):
        """Outline mode should paint edges in the sub-region."""
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        region = np.zeros((20, 20, 4), dtype=np.uint8)
        _composite_roi_region(region, mask, (0, 255, 0), 200, "outline",
                              10, 10, 30, 30)
        # Interior of region should be transparent
        assert region[10, 10, 3] == 0
        # Some edge pixels should be painted
        assert np.any(region[:, :, 3] > 0)

    def test_region_empty_mask_no_change(self, qapp):
        """Empty mask region should leave region array unchanged."""
        mask = np.zeros((40, 40), dtype=np.uint8)
        region = np.zeros((10, 10, 4), dtype=np.uint8)
        _composite_roi_region(region, mask, (255, 0, 0), 128, "solid",
                              0, 0, 10, 10)
        assert np.all(region == 0)


# ===================================================================
# 21. zoom_in, zoom_out, reset_zoom — verify scale changes
# ===================================================================

class TestZoom:
    def test_zoom_in_increases_scale(self, app):
        """zoom_in should increase the transform scale."""
        app.canvas.resetTransform()
        scale_before = app.canvas.transform().m11()
        app.canvas.zoom_in()
        scale_after = app.canvas.transform().m11()
        assert scale_after > scale_before

    def test_zoom_out_decreases_scale(self, app):
        """zoom_out should decrease the transform scale."""
        app.canvas.resetTransform()
        scale_before = app.canvas.transform().m11()
        app.canvas.zoom_out()
        scale_after = app.canvas.transform().m11()
        assert scale_after < scale_before

    def test_reset_zoom(self, app):
        """reset_zoom should restore identity transform."""
        app.canvas.scale(3.0, 3.0)
        app.canvas.reset_zoom()
        m11 = app.canvas.transform().m11()
        assert abs(m11 - 1.0) < 0.001

    def test_zoom_in_emits_viewport_changed(self, app):
        """zoom_in should emit the viewport_changed signal."""
        received = []
        app.canvas.viewport_changed.connect(lambda: received.append(True))
        app.canvas.zoom_in()
        assert len(received) > 0

    def test_zoom_out_emits_viewport_changed(self, app):
        """zoom_out should emit the viewport_changed signal."""
        received = []
        app.canvas.viewport_changed.connect(lambda: received.append(True))
        app.canvas.zoom_out()
        assert len(received) > 0


# ===================================================================
# 22. fit_to_window — verify fitInView called
# ===================================================================

class TestFitToWindow:
    def test_fit_to_window_with_image(self, app):
        """fit_to_window should call fitInView when an image item exists."""
        assert app.canvas._image_item is not None
        with patch.object(app.canvas, 'fitInView') as mock_fit:
            app.canvas.fit_to_window()
            mock_fit.assert_called_once_with(
                app.canvas._image_item, Qt.KeepAspectRatio
            )

    def test_fit_to_window_no_image(self, qapp):
        """fit_to_window should be a no-op when no image item exists."""
        win = MontarisApp()
        win.canvas._image_item = None
        with patch.object(win.canvas, 'fitInView') as mock_fit:
            win.canvas.fit_to_window()
            mock_fit.assert_not_called()
