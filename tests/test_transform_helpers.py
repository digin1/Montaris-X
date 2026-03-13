"""Tests for transform helper / pure functions in roi_transform and TransformTool."""

import math
import numpy as np
import pytest

from montaris.core.roi_transform import (
    compute_handles,
    apply_affine_to_mask,
    make_scale_matrix,
    make_rotation_matrix,
    make_translation_matrix,
    TransformHandle,
)


# ---------------------------------------------------------------------------
# compute_handles
# ---------------------------------------------------------------------------

class TestComputeHandles:
    def test_returns_nine_handles(self):
        """compute_handles should return exactly 9 handles (8 scale + 1 rotate)."""
        bbox = (10, 50, 20, 80)  # y1, y2, x1, x2
        handles = compute_handles(bbox)
        assert len(handles) == 9

    def test_handle_types(self):
        """All expected handle types must be present."""
        bbox = (10, 50, 20, 80)
        handles = compute_handles(bbox)
        types = {h.handle_type for h in handles}
        expected = {'tl', 'tr', 'bl', 'br', 'tm', 'bm', 'ml', 'mr', 'rotate'}
        assert types == expected

    def test_corner_positions(self):
        """Corner handles should sit at the bbox corners."""
        y1, y2, x1, x2 = 0, 100, 0, 200
        handles = compute_handles((y1, y2, x1, x2))
        by_type = {h.handle_type: h for h in handles}

        assert (by_type['tl'].x, by_type['tl'].y) == (x1, y1)
        assert (by_type['tr'].x, by_type['tr'].y) == (x2, y1)
        assert (by_type['bl'].x, by_type['bl'].y) == (x1, y2)
        assert (by_type['br'].x, by_type['br'].y) == (x2, y2)

    def test_midpoint_positions(self):
        """Midpoint handles should be at the edge midpoints."""
        y1, y2, x1, x2 = 0, 100, 0, 200
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        handles = compute_handles((y1, y2, x1, x2))
        by_type = {h.handle_type: h for h in handles}

        assert (by_type['tm'].x, by_type['tm'].y) == (cx, y1)
        assert (by_type['bm'].x, by_type['bm'].y) == (cx, y2)
        assert (by_type['ml'].x, by_type['ml'].y) == (x1, cy)
        assert (by_type['mr'].x, by_type['mr'].y) == (x2, cy)

    def test_rotate_handle_above_top(self):
        """The rotate handle sits 20 pixels above the top-center."""
        y1, y2, x1, x2 = 10, 50, 20, 80
        cx = (x1 + x2) / 2
        handles = compute_handles((y1, y2, x1, x2))
        by_type = {h.handle_type: h for h in handles}

        assert by_type['rotate'].x == cx
        assert by_type['rotate'].y == y1 - 20


# ---------------------------------------------------------------------------
# apply_affine_to_mask — identity, translation, scale
# ---------------------------------------------------------------------------

class TestApplyAffineToMask:
    def _rect_mask(self, h=100, w=100, y1=30, y2=50, x1=30, x2=50):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    def test_identity_preserves_mask(self):
        """Identity matrix should leave the mask unchanged."""
        mask = self._rect_mask()
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        result = apply_affine_to_mask(mask, M)
        assert np.array_equal(result, mask)

    def test_translation(self):
        """Translating a mask should shift non-zero pixels."""
        mask = self._rect_mask(y1=10, y2=20, x1=10, x2=20)
        tx, ty = 15, 10
        M = make_translation_matrix(tx, ty)
        result = apply_affine_to_mask(mask, M)

        # Shifted pixel should be set
        assert result[10 + ty, 10 + tx] == 255
        # Original pixel should be clear (no longer in the source rect)
        assert result[10, 10] == 0

    def test_scale_doubles_area(self):
        """Scaling by 2x around the mask center should roughly double the area."""
        mask = self._rect_mask(h=200, w=200, y1=80, y2=100, x1=80, x2=100)
        cx, cy = 90.0, 90.0
        M = make_scale_matrix(2.0, 2.0, cx, cy)
        result = apply_affine_to_mask(mask, M, output_shape=(200, 200))

        src_count = int(np.count_nonzero(mask))
        dst_count = int(np.count_nonzero(result))
        # Scaled area should be roughly 4x (2x in each dimension); allow
        # some tolerance due to rasterisation at boundaries.
        assert dst_count > src_count * 2


# ---------------------------------------------------------------------------
# make_*_matrix helpers
# ---------------------------------------------------------------------------

class TestMatrixHelpers:
    def test_translation_matrix_shape(self):
        M = make_translation_matrix(5, 10)
        assert M.shape == (2, 3)
        assert M[0, 2] == 5
        assert M[1, 2] == 10

    def test_scale_matrix_at_origin(self):
        M = make_scale_matrix(2.0, 3.0)
        assert M.shape == (2, 3)
        assert M[0, 0] == 2.0
        assert M[1, 1] == 3.0
        # Translation component should be 0 when center is origin
        assert M[0, 2] == 0.0
        assert M[1, 2] == 0.0

    def test_scale_matrix_centered(self):
        """Scaling centered at (50, 50) by 2x should leave (50,50) fixed."""
        M = make_scale_matrix(2.0, 2.0, cx=50, cy=50)
        # Applying to (50, 50): result = sx*50 + tx = 2*50 + (50 - 2*50) = 50
        result_x = M[0, 0] * 50 + M[0, 1] * 50 + M[0, 2]
        result_y = M[1, 0] * 50 + M[1, 1] * 50 + M[1, 2]
        assert abs(result_x - 50) < 1e-9
        assert abs(result_y - 50) < 1e-9

    def test_rotation_matrix_shape(self):
        M = make_rotation_matrix(math.pi / 4)
        assert M.shape == (2, 3)

    def test_rotation_360_is_identity(self):
        """A full 360-degree rotation around the origin is identity."""
        M = make_rotation_matrix(2 * math.pi)
        np.testing.assert_allclose(M, [[1, 0, 0], [0, 1, 0]], atol=1e-12)


# ---------------------------------------------------------------------------
# TransformTool._rotate_point — tested via the instance method
# ---------------------------------------------------------------------------

class TestRotatePoint:
    """Test _rotate_point which is an instance method on TransformTool.

    We instantiate it with a mock app just to reach the helper method.
    """

    @pytest.fixture(autouse=True)
    def _setup_tool(self):
        """Import TransformTool and create one with a mocked app."""
        # Guard: if PySide6 is not available, skip the whole class.
        pytest.importorskip("PySide6")
        from unittest.mock import MagicMock
        from montaris.tools.transform import TransformTool

        mock_app = MagicMock()
        self.tool = TransformTool(mock_app)

    def test_zero_angle_same_point(self):
        """Rotating by 0 should return the same point."""
        x, y = self.tool._rotate_point(10, 20, 0, 0, 0)
        assert abs(x - 10) < 1e-9
        assert abs(y - 20) < 1e-9

    def test_90_degrees(self):
        """Rotating (1, 0) by 90 degrees around origin gives (0, 1)."""
        x, y = self.tool._rotate_point(1, 0, 0, 0, math.pi / 2)
        assert abs(x - 0) < 1e-9
        assert abs(y - 1) < 1e-9

    def test_180_degrees_mirrors(self):
        """Rotating (5, 3) by 180 degrees around origin gives (-5, -3)."""
        x, y = self.tool._rotate_point(5, 3, 0, 0, math.pi)
        assert abs(x - (-5)) < 1e-9
        assert abs(y - (-3)) < 1e-9

    def test_360_degrees_returns_same(self):
        """Full rotation should return the original point."""
        x, y = self.tool._rotate_point(7, 11, 5, 5, 2 * math.pi)
        assert abs(x - 7) < 1e-9
        assert abs(y - 11) < 1e-9

    def test_around_nonzero_center(self):
        """Rotating (10, 5) by 90 deg around (5, 5) gives (5, 10)."""
        x, y = self.tool._rotate_point(10, 5, 5, 5, math.pi / 2)
        assert abs(x - 5) < 1e-9
        assert abs(y - 10) < 1e-9
