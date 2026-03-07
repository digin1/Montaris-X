"""Tests for Phase 4: Drawing Tool Enhancements."""
import pytest
import numpy as np
from PySide6.QtCore import QPointF

from montaris.layers import ROILayer


class TestBrushSize:
    def test_large_brush_size(self, qapp, app_with_image):
        app = app_with_image
        app.tool_panel._select_tool('Brush')
        app.tool_panel.size_slider.setValue(2000)
        assert app.tool_panel._current_tool.size == 2000

    def test_slider_range(self, qapp, app_with_image):
        assert app_with_image.tool_panel.size_slider.maximum() == 2000


class TestBucketFillTolerance:
    def test_tolerance_attribute(self, qapp, app_with_image):
        app = app_with_image
        app.tool_panel._select_tool('Bucket Fill')
        tool = app.tool_panel._current_tool
        assert hasattr(tool, 'tolerance')
        assert tool.tolerance == 0

    def test_tolerance_slider(self, qapp, app_with_image):
        app = app_with_image
        app.tool_panel._select_tool('Bucket Fill')
        app.tool_panel.tolerance_slider.setValue(50)
        assert app.tool_panel._current_tool.tolerance == 50

    def test_tolerance_visible_for_bucket(self, qapp, app_with_image):
        app = app_with_image
        app.tool_panel._select_tool('Bucket Fill')
        assert app.tool_panel.tolerance_slider.isVisible()
        app.tool_panel._select_tool('Brush')
        assert not app.tool_panel.tolerance_slider.isVisible()


class TestStampWidthHeight:
    def test_stamp_has_wh(self, qapp, app_with_image):
        app = app_with_image
        app.tool_panel._select_tool('Stamp')
        tool = app.tool_panel._current_tool
        assert hasattr(tool, 'width')
        assert hasattr(tool, 'height')
        assert tool.width == 20
        assert tool.height == 20

    def test_stamp_rectangular(self, qapp, app_with_image):
        app = app_with_image
        app.tool_panel._select_tool('Stamp')
        tool = app.tool_panel._current_tool
        tool.width = 30
        tool.height = 10
        roi = app.layer_stack.roi_layers[0]
        tool.on_press(QPointF(60, 50), roi, app.canvas)
        tool.on_release(QPointF(60, 50), roi, app.canvas)
        # Should have stamped a rectangle
        assert np.any(roi.mask > 0)


class TestDeselectTool:
    def test_deselect_button_exists(self, qapp, app_with_image):
        assert app_with_image.tool_panel.deselect_btn is not None

    def test_deselect_clears_tool(self, qapp, app_with_image):
        app = app_with_image
        app.tool_panel._select_tool('Brush')
        app.tool_panel._deselect_tool()
        assert app.tool_panel._current_tool is None


class TestAutoOverlapBrush:
    def test_auto_overlap_clears_other_layers(self, qapp, app_with_image):
        app = app_with_image
        app._auto_overlap = True
        # Add second ROI with some content
        h, w = app.layer_stack.image_layer.shape[:2]
        roi2 = ROILayer("ROI 2", w, h)
        roi2.mask[5:10, 5:10] = 255
        app.layer_stack.add_roi(roi2)
        app.canvas.refresh_overlays()

        # Paint on ROI 1 over roi2's area
        roi1 = app.layer_stack.roi_layers[0]
        app.canvas.set_active_layer(roi1)
        app.tool_panel._select_tool('Brush')
        tool = app.tool_panel._current_tool
        tool.size = 20
        tool.on_press(QPointF(7, 7), roi1, app.canvas)
        tool.on_release(QPointF(7, 7), roi1, app.canvas)

        # roi2 should have lost pixels where roi1 painted
        overlap = (roi1.mask > 0) & (roi2.mask > 0)
        assert not np.any(overlap)
        app._auto_overlap = False


class TestStampPreview:
    def test_stamp_preview_created(self, qapp, app_with_image):
        app = app_with_image
        app.tool_panel._select_tool('Stamp')
        # Initially hidden or None
        assert app.canvas._stamp_preview is None or not app.canvas._stamp_preview.isVisible()


class TestBrushColorPreview:
    def test_brush_preview_uses_roi_color(self, qapp, app_with_image):
        app = app_with_image
        roi = app.layer_stack.roi_layers[0]
        roi.color = (255, 0, 0)
        app.canvas.set_active_layer(roi)
        app.canvas.show_brush_preview(50, 50, 10)
        assert app.canvas._brush_preview is not None
        pen_color = app.canvas._brush_preview.pen().color()
        assert pen_color.red() == 255
