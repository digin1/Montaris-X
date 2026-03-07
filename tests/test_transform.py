import numpy as np
import pytest
from PySide6.QtCore import QPointF
from montaris.core.roi_transform import (
    get_mask_bbox, compute_handles, apply_affine_to_mask,
    make_scale_matrix, make_rotation_matrix, make_translation_matrix,
)
from montaris.layers import ROILayer
from montaris.tools.move import MoveTool


class TestMaskBbox:
    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert get_mask_bbox(mask) is None

    def test_bbox(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 255
        bbox = get_mask_bbox(mask)
        assert bbox == (20, 40, 30, 60)


class TestComputeHandles:
    def test_eight_plus_rotate(self):
        handles = compute_handles((10, 50, 20, 80))
        assert len(handles) == 9
        types = {h.handle_type for h in handles}
        assert 'rotate' in types
        assert 'tl' in types
        assert 'br' in types


class TestAffineTransform:
    def test_identity(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:50, 30:50] = 255
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        result = apply_affine_to_mask(mask, M)
        assert np.array_equal(result, mask)

    def test_translation(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        M = make_translation_matrix(10, 5)
        result = apply_affine_to_mask(mask, M)
        assert result[15, 20] == 255  # shifted by (10, 5)
        assert result[10, 10] == 0  # original position cleared

    def test_scale_up(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        M = make_scale_matrix(2.0, 2.0, 50, 50)
        result = apply_affine_to_mask(mask, M)
        # Scaled up = more pixels
        assert result.sum() > mask.sum()

    def test_scale_down(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        M = make_scale_matrix(0.5, 0.5, 50, 50)
        result = apply_affine_to_mask(mask, M)
        assert result.sum() < mask.sum()

    def test_rotation(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 45:55] = 255  # Vertical bar
        M = make_rotation_matrix(np.pi / 2, 50, 50)  # 90 degrees
        result = apply_affine_to_mask(mask, M)
        # After 90 degree rotation, vertical bar becomes horizontal
        assert result.sum() > 0
        # Check that pixels moved
        assert not np.array_equal(result, mask)


class TestMoveTool:
    def test_move_basic(self, app_with_image):
        app = app_with_image
        tool = MoveTool(app)
        layer = app.layer_stack.roi_layers[0]
        layer.mask[20:40, 20:40] = 255
        before_sum = layer.mask.sum()

        tool.on_press(QPointF(30, 30), layer, app.canvas)
        tool.on_move(QPointF(40, 40), layer, app.canvas)
        tool.on_release(QPointF(40, 40), layer, app.canvas)

        # Offset-based move: mask unchanged, offset shifted
        assert layer.offset_x == 10 and layer.offset_y == 10
        assert layer.mask.sum() == before_sum  # No pixel loss
        # After flattening, mask at new position
        layer.flatten_offset()
        assert layer.mask[30:50, 30:50].sum() > 0
        assert layer.mask[20:30, 20:30].sum() < before_sum
