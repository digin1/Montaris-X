import os
import tempfile
import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from montaris.layers import ImageLayer, ROILayer, LayerStack
from montaris.tools.brush import BrushTool
from montaris.tools.eraser import EraserTool
from montaris.tools.polygon import PolygonTool
from montaris.core.undo import UndoStack, UndoCommand
from montaris.io.roi_io import save_roi_set, load_roi_set
from montaris.io.image_io import load_image
from tests.helpers import simulate_brush_stroke


class TestImageLoad:
    def test_load_png(self, tmp_path):
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8))
        path = tmp_path / "test.png"
        img.save(str(path))
        data = load_image(str(path))
        assert data.shape == (50, 60, 3)

    def test_load_grayscale(self, tmp_path):
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (50, 60), dtype=np.uint8))
        path = tmp_path / "test.png"
        img.save(str(path))
        data = load_image(str(path))
        assert data.shape == (50, 60)

    def test_load_tiff(self, tmp_path):
        import tifffile
        arr = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        path = tmp_path / "test.tif"
        tifffile.imwrite(str(path), arr)
        data = load_image(str(path))
        assert data.shape == (50, 60)


class TestLayerStack:
    def test_set_image(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        assert layer_stack.image_layer is image_layer

    def test_add_remove_roi(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi = ROILayer("test", 120, 100)
        layer_stack.add_roi(roi)
        assert len(layer_stack.roi_layers) == 1
        layer_stack.remove_roi(0)
        assert len(layer_stack.roi_layers) == 0

    def test_roi_colors_cycle(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        rois = []
        for i in range(3):
            roi = ROILayer(f"roi_{i}", 120, 100)
            layer_stack.add_roi(roi)
            rois.append(roi)
        assert rois[0].color != rois[1].color


class TestBrushTool:
    def test_paint_pixels(self, app_with_image):
        app = app_with_image
        tool = BrushTool(app)
        tool.size = 10
        layer = app.layer_stack.roi_layers[0]
        assert layer.mask.sum() == 0
        simulate_brush_stroke(tool, layer, app.canvas, [(50, 50), (55, 50), (60, 50)])
        assert layer.mask.sum() > 0

    def test_undo_after_paint(self, app_with_image):
        app = app_with_image
        tool = BrushTool(app)
        tool.size = 10
        layer = app.layer_stack.roi_layers[0]
        simulate_brush_stroke(tool, layer, app.canvas, [(50, 50)])
        assert layer.mask.sum() > 0
        app.undo_stack.undo()
        assert layer.mask.sum() == 0


class TestEraserTool:
    def test_erase_pixels(self, app_with_image):
        app = app_with_image
        layer = app.layer_stack.roi_layers[0]
        layer.mask[40:60, 40:60] = 255
        before = layer.mask.sum()
        tool = EraserTool(app)
        tool.size = 10
        simulate_brush_stroke(tool, layer, app.canvas, [(50, 50)])
        assert layer.mask.sum() < before


class TestPolygonTool:
    def test_fill_polygon(self, app_with_image):
        app = app_with_image
        tool = PolygonTool(app)
        layer = app.layer_stack.roi_layers[0]
        tool.on_press(QPointF(10, 10), layer, app.canvas)
        tool.on_press(QPointF(50, 10), layer, app.canvas)
        tool.on_press(QPointF(50, 50), layer, app.canvas)
        tool.on_press(QPointF(10, 50), layer, app.canvas)
        tool.finish()
        assert layer.mask.sum() > 0


class TestUndoRedo:
    def test_undo_redo_stack(self):
        stack = UndoStack()
        roi = ROILayer("test", 100, 100)
        roi.mask[10:20, 10:20] = 255
        old = np.zeros((10, 10), dtype=np.uint8)
        new = np.full((10, 10), 255, dtype=np.uint8)
        cmd = UndoCommand(roi, (10, 20, 10, 20), old, new)
        stack.push(cmd)
        assert stack.can_undo
        stack.undo()
        assert roi.mask[15, 15] == 0
        stack.redo()
        assert roi.mask[15, 15] == 255


class TestROISaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        roi1 = ROILayer("ROI A", 100, 80, color=(255, 0, 0))
        roi1.mask[10:30, 10:30] = 255
        roi2 = ROILayer("ROI B", 100, 80, color=(0, 255, 0))
        roi2.mask[40:60, 40:60] = 128
        path = tmp_path / "test_rois.npz"
        save_roi_set(str(path), [roi1, roi2])
        loaded = load_roi_set(str(path))
        assert len(loaded) == 2
        assert loaded[0].name == "ROI A"
        assert loaded[1].name == "ROI B"
        assert np.array_equal(loaded[0].mask, roi1.mask)
        assert np.array_equal(loaded[1].mask, roi2.mask)


class TestCanvasBasic:
    def test_refresh_image(self, app_with_image):
        app = app_with_image
        assert app.canvas._image_item is not None

    def test_refresh_overlays(self, app_with_image):
        app = app_with_image
        app.canvas.refresh_overlays()
        assert len(app.canvas._roi_items) >= 0

    def test_fit_to_window(self, app_with_image):
        app = app_with_image
        app.canvas.fit_to_window()

    def test_reset_zoom(self, app_with_image):
        app = app_with_image
        app.canvas.reset_zoom()
