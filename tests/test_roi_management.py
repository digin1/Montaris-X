import os
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from montaris.layers import LayerStack, ImageLayer, ROILayer
from montaris.canvas import mask_to_qimage, mask_to_outline_qimage


class TestMergeROIs:
    def test_merge_two_rois(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi1 = ROILayer("ROI 1", 120, 100)
        roi1.mask[10:30, 10:30] = 255
        roi2 = ROILayer("ROI 2", 120, 100)
        roi2.mask[40:60, 40:60] = 200
        layer_stack.add_roi(roi1)
        layer_stack.add_roi(roi2)
        assert len(layer_stack.roi_layers) == 2

        layer_stack.merge_rois([0, 1])
        assert len(layer_stack.roi_layers) == 1
        merged = layer_stack.roi_layers[0]
        # Contains union of both painted regions
        assert merged.mask[15, 15] == 255
        assert merged.mask[50, 50] == 200
        # Unpainted area still zero
        assert merged.mask[0, 0] == 0

    def test_merge_three_rois(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi1 = ROILayer("ROI 1", 120, 100)
        roi1.mask[0:10, 0:10] = 100
        roi2 = ROILayer("ROI 2", 120, 100)
        roi2.mask[20:30, 20:30] = 150
        roi3 = ROILayer("ROI 3", 120, 100)
        roi3.mask[50:60, 50:60] = 200
        layer_stack.add_roi(roi1)
        layer_stack.add_roi(roi2)
        layer_stack.add_roi(roi3)

        layer_stack.merge_rois([0, 1, 2])
        assert len(layer_stack.roi_layers) == 1
        merged = layer_stack.roi_layers[0]
        assert merged.mask[5, 5] == 100
        assert merged.mask[25, 25] == 150
        assert merged.mask[55, 55] == 200

    def test_merge_single_roi_noop(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi1 = ROILayer("ROI 1", 120, 100)
        layer_stack.add_roi(roi1)
        layer_stack.merge_rois([0])
        assert len(layer_stack.roi_layers) == 1

    def test_merge_overlapping_uses_maximum(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi1 = ROILayer("ROI 1", 120, 100)
        roi1.mask[10:20, 10:20] = 100
        roi2 = ROILayer("ROI 2", 120, 100)
        roi2.mask[10:20, 10:20] = 200
        layer_stack.add_roi(roi1)
        layer_stack.add_roi(roi2)

        layer_stack.merge_rois([0, 1])
        merged = layer_stack.roi_layers[0]
        # np.maximum should pick 200
        assert merged.mask[15, 15] == 200


class TestDuplicateROI:
    def test_duplicate(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi = ROILayer("ROI 1", 120, 100)
        roi.mask[10:30, 10:30] = 255
        roi.opacity = 200
        layer_stack.add_roi(roi)

        layer_stack.duplicate_roi(0)
        assert len(layer_stack.roi_layers) == 2
        original = layer_stack.roi_layers[0]
        copy = layer_stack.roi_layers[1]

        # Same mask data
        assert np.array_equal(original.mask, copy.mask)
        # Same opacity
        assert copy.opacity == 200
        # Name indicates copy
        assert "(copy)" in copy.name
        # Separate objects — modifying copy doesn't affect original
        copy.mask[0, 0] = 128
        assert original.mask[0, 0] != 128

    def test_duplicate_preserves_fill_mode(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi = ROILayer("ROI 1", 120, 100)
        roi.fill_mode = "outline"
        layer_stack.add_roi(roi)

        layer_stack.duplicate_roi(0)
        copy = layer_stack.roi_layers[1]
        assert copy.fill_mode == "outline"

    def test_duplicate_invalid_index(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi = ROILayer("ROI 1", 120, 100)
        layer_stack.add_roi(roi)
        layer_stack.duplicate_roi(5)
        assert len(layer_stack.roi_layers) == 1

    def test_duplicate_inserts_after(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi1 = ROILayer("ROI 1", 120, 100)
        roi2 = ROILayer("ROI 2", 120, 100)
        layer_stack.add_roi(roi1)
        layer_stack.add_roi(roi2)

        layer_stack.duplicate_roi(0)
        assert len(layer_stack.roi_layers) == 3
        assert layer_stack.roi_layers[0].name == "ROI 1"
        assert "(copy)" in layer_stack.roi_layers[1].name
        assert layer_stack.roi_layers[2].name == "ROI 2"


class TestReorderROI:
    def test_reorder(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi1 = ROILayer("A", 120, 100)
        roi2 = ROILayer("B", 120, 100)
        roi3 = ROILayer("C", 120, 100)
        layer_stack.add_roi(roi1)
        layer_stack.add_roi(roi2)
        layer_stack.add_roi(roi3)

        # Move first to last
        layer_stack.reorder_roi(0, 2)
        assert layer_stack.roi_layers[0].name == "B"
        assert layer_stack.roi_layers[1].name == "C"
        assert layer_stack.roi_layers[2].name == "A"

    def test_reorder_same_position(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi1 = ROILayer("A", 120, 100)
        roi2 = ROILayer("B", 120, 100)
        layer_stack.add_roi(roi1)
        layer_stack.add_roi(roi2)

        layer_stack.reorder_roi(0, 0)
        assert layer_stack.roi_layers[0].name == "A"
        assert layer_stack.roi_layers[1].name == "B"

    def test_reorder_invalid_index(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi1 = ROILayer("A", 120, 100)
        layer_stack.add_roi(roi1)
        # Invalid from_idx — should be a no-op
        layer_stack.reorder_roi(5, 0)
        assert len(layer_stack.roi_layers) == 1
        assert layer_stack.roi_layers[0].name == "A"


class TestInsertROI:
    def test_insert_at_position(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi1 = ROILayer("A", 120, 100)
        roi2 = ROILayer("B", 120, 100)
        layer_stack.add_roi(roi1)
        layer_stack.add_roi(roi2)

        new_roi = ROILayer("X", 120, 100)
        layer_stack.insert_roi(1, new_roi)

        assert len(layer_stack.roi_layers) == 3
        assert layer_stack.roi_layers[0].name == "A"
        assert layer_stack.roi_layers[1].name == "X"
        assert layer_stack.roi_layers[2].name == "B"

    def test_insert_at_start(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi1 = ROILayer("A", 120, 100)
        layer_stack.add_roi(roi1)

        new_roi = ROILayer("X", 120, 100)
        layer_stack.insert_roi(0, new_roi)

        assert len(layer_stack.roi_layers) == 2
        assert layer_stack.roi_layers[0].name == "X"
        assert layer_stack.roi_layers[1].name == "A"

    def test_insert_at_end(self, layer_stack, image_layer):
        layer_stack.set_image(image_layer)
        roi1 = ROILayer("A", 120, 100)
        layer_stack.add_roi(roi1)

        new_roi = ROILayer("X", 120, 100)
        layer_stack.insert_roi(1, new_roi)

        assert len(layer_stack.roi_layers) == 2
        assert layer_stack.roi_layers[0].name == "A"
        assert layer_stack.roi_layers[1].name == "X"


class TestOutlineMode:
    def test_outline_rendering(self):
        """A filled square's outline should have only edge pixels."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        color = (255, 0, 0)

        qimg = mask_to_outline_qimage(mask, color, 200)
        assert qimg.width() == 20
        assert qimg.height() == 20

        # Convert back to numpy to verify
        ptr = qimg.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(20, 20, 4).copy()

        # Interior pixel (not on edge) should be transparent
        assert arr[10, 10, 3] == 0

        # Edge pixel (on boundary of the square) should be colored
        assert arr[5, 5, 3] == 200  # top-left corner
        assert arr[5, 10, 3] == 200  # top edge
        assert arr[14, 10, 3] == 200  # bottom edge
        assert arr[10, 5, 3] == 200  # left edge
        assert arr[10, 14, 3] == 200  # right edge

    def test_outline_empty_mask(self):
        """An empty mask should produce an all-transparent image."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        color = (0, 255, 0)
        qimg = mask_to_outline_qimage(mask, color)
        ptr = qimg.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(10, 10, 4).copy()
        assert arr[:, :, 3].sum() == 0

    def test_solid_rendering_unchanged(self):
        """Solid rendering should still fill the interior."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        color = (255, 0, 0)
        qimg = mask_to_qimage(mask, color, 128)
        ptr = qimg.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(20, 20, 4).copy()
        # Interior pixel should be filled
        assert arr[10, 10, 3] == 128
        assert arr[10, 10, 0] == 255


class TestFillMode:
    def test_fill_mode_default(self):
        roi = ROILayer("test", 100, 80)
        assert roi.fill_mode == "solid"

    def test_fill_mode_set_outline(self):
        roi = ROILayer("test", 100, 80)
        roi.fill_mode = "outline"
        assert roi.fill_mode == "outline"

    def test_fill_mode_set_solid(self):
        roi = ROILayer("test", 100, 80)
        roi.fill_mode = "outline"
        roi.fill_mode = "solid"
        assert roi.fill_mode == "solid"
