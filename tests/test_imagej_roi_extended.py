"""Extended tests for montaris.io.imagej_roi — roundtrips, parsing, edge cases."""

import numpy as np
import pytest

from montaris.io.imagej_roi import (
    read_imagej_roi,
    write_imagej_roi_bytes,
    mask_to_imagej_roi,
    imagej_roi_to_mask,
    _build_roi_bytes,
    scale_roi_dict,
    ROI_RECT,
    ROI_OVAL,
    ROI_FREEHAND,
)


# ---------------------------------------------------------------------------
# 1. Mask -> ROI -> bytes -> ROI -> mask roundtrip
# ---------------------------------------------------------------------------

class TestMaskRoundtrip:
    def test_rect_mask_roundtrip(self):
        """Rectangular mask survives full roundtrip."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:70] = 255
        roi = mask_to_imagej_roi(mask, "rect_test")
        assert roi is not None

        raw = write_imagej_roi_bytes(roi)
        loaded = read_imagej_roi(raw)
        recovered = imagej_roi_to_mask(loaded, 100, 100)

        # Interior must be filled
        assert recovered[30:55, 35:65].all()
        # Far exterior must be zero
        assert recovered[0:15, :].sum() == 0

    def test_square_mask_roundtrip(self):
        """Small square mask survives roundtrip."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        roi = mask_to_imagej_roi(mask, "square")
        raw = write_imagej_roi_bytes(roi)
        loaded = read_imagej_roi(raw)
        recovered = imagej_roi_to_mask(loaded, 50, 50)

        assert recovered[20:35, 20:35].all()
        assert recovered[0:5, :].sum() == 0

    def test_roundtrip_preserves_bbox(self):
        """Bounding box coordinates survive write/read."""
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[15:65, 25:95] = 255
        roi = mask_to_imagej_roi(mask, "bbox_check")
        raw = write_imagej_roi_bytes(roi)
        loaded = read_imagej_roi(raw)

        assert loaded['top'] == roi['top']
        assert loaded['left'] == roi['left']
        assert loaded['bottom'] == roi['bottom']
        assert loaded['right'] == roi['right']


# ---------------------------------------------------------------------------
# 2. Ellipse ROI parsing
# ---------------------------------------------------------------------------

class TestEllipseROI:
    def test_oval_rasterization_centered(self):
        """Oval ROI rasterizes to a filled ellipse."""
        roi = {
            'type': ROI_OVAL,
            'top': 20,
            'left': 30,
            'bottom': 80,
            'right': 90,
            'x_coords': None,
            'y_coords': None,
            'paths': None,
        }
        mask = imagej_roi_to_mask(roi, 120, 100)
        # Center of ellipse should be filled
        cy, cx = 50, 60
        assert mask[cy, cx] == 255

    def test_oval_exterior_empty(self):
        """Pixels outside the oval bounding box should be zero."""
        roi = {
            'type': ROI_OVAL,
            'top': 30,
            'left': 30,
            'bottom': 70,
            'right': 70,
            'x_coords': None,
            'y_coords': None,
            'paths': None,
        }
        mask = imagej_roi_to_mask(roi, 100, 100)
        assert mask[0:25, :].sum() == 0
        assert mask[:, 0:25].sum() == 0
        assert mask[75:, :].sum() == 0
        assert mask[:, 75:].sum() == 0

    def test_oval_corners_empty(self):
        """Corners of the bounding box should NOT be filled (it's an ellipse)."""
        roi = {
            'type': ROI_OVAL,
            'top': 10,
            'left': 10,
            'bottom': 90,
            'right': 90,
            'x_coords': None,
            'y_coords': None,
            'paths': None,
        }
        mask = imagej_roi_to_mask(roi, 100, 100)
        # Top-left corner of bbox should be empty
        assert mask[10, 10] == 0
        # Top-right corner
        assert mask[10, 89] == 0


# ---------------------------------------------------------------------------
# 3. Shape ROI (composite) with move/line/close segments
# ---------------------------------------------------------------------------

class TestCompositeROI:
    def _make_composite_roi(self):
        """Create a composite ROI with a single square path."""
        return {
            'type': ROI_RECT,
            'top': 10,
            'left': 10,
            'bottom': 50,
            'right': 50,
            'x_coords': None,
            'y_coords': None,
            'paths': [
                [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]
            ],
            'name': 'composite_square',
        }

    def test_composite_write_read(self):
        """Composite ROI survives write/read cycle."""
        roi = self._make_composite_roi()
        raw = write_imagej_roi_bytes(roi)
        loaded = read_imagej_roi(raw)
        assert loaded['paths'] is not None
        assert len(loaded['paths']) >= 1

    def test_composite_rasterization(self):
        """Composite ROI rasterizes to filled polygon."""
        roi = self._make_composite_roi()
        mask = imagej_roi_to_mask(roi, 60, 60)
        # Interior should be filled
        assert mask[25:45, 25:45].any()

    def test_composite_roundtrip(self):
        """Composite ROI survives full write -> read -> rasterize roundtrip."""
        roi = self._make_composite_roi()
        raw = write_imagej_roi_bytes(roi)
        loaded = read_imagej_roi(raw)
        mask = imagej_roi_to_mask(loaded, 60, 60)
        assert mask[30, 30] > 0

    def test_multi_path_composite(self):
        """Composite with two separate paths (XOR fill)."""
        roi = {
            'type': ROI_RECT,
            'top': 5,
            'left': 5,
            'bottom': 50,
            'right': 95,
            'x_coords': None,
            'y_coords': None,
            'paths': [
                [(5.0, 5.0), (45.0, 5.0), (45.0, 50.0), (5.0, 50.0)],
                [(50.0, 5.0), (95.0, 5.0), (95.0, 50.0), (50.0, 50.0)],
            ],
            'name': 'two_rects',
        }
        mask = imagej_roi_to_mask(roi, 100, 60)
        # Both regions should have painted pixels
        assert mask[20:40, 10:40].any()
        assert mask[20:40, 55:90].any()


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_pixel_mask(self):
        """Single pixel mask should produce a valid ROI (or None for too-small contour)."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 10] = 255
        roi = mask_to_imagej_roi(mask, "single_px")
        # find_contours may or may not find a contour for a single pixel
        # but it should not crash
        if roi is not None:
            recovered = imagej_roi_to_mask(roi, 20, 20)
            assert recovered.shape == (20, 20)

    def test_full_mask(self):
        """Full mask should produce a valid ROI."""
        mask = np.ones((30, 30), dtype=np.uint8) * 255
        roi = mask_to_imagej_roi(mask, "full")
        assert roi is not None
        recovered = imagej_roi_to_mask(roi, 30, 30)
        # Most pixels should be filled
        assert recovered.sum() > 0

    def test_l_shaped_mask(self):
        """L-shaped mask should roundtrip correctly."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[5:40, 5:20] = 255   # vertical bar
        mask[25:40, 5:40] = 255  # horizontal bar
        roi = mask_to_imagej_roi(mask, "L_shape")
        assert roi is not None

        raw = write_imagej_roi_bytes(roi)
        loaded = read_imagej_roi(raw)
        recovered = imagej_roi_to_mask(loaded, 50, 50)

        # Center of vertical bar
        assert recovered[15, 12] > 0
        # Center of horizontal bar
        assert recovered[32, 30] > 0

    def test_empty_mask_returns_none(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        assert mask_to_imagej_roi(mask) is None

    def test_too_short_data_raises(self):
        with pytest.raises(ValueError, match="too short"):
            read_imagej_roi(b'\x00' * 10)

    def test_bad_magic_raises(self):
        with pytest.raises(ValueError, match="Invalid ROI magic"):
            read_imagej_roi(b'\x00' * 64)


# ---------------------------------------------------------------------------
# 5. Coordinate scaling for non-zero bbox offsets
# ---------------------------------------------------------------------------

class TestCoordinateScaling:
    def test_bbox_offset_freehand(self):
        """Freehand ROI with non-zero bbox offset scales correctly."""
        roi = {
            'type': ROI_FREEHAND,
            'top': 50,
            'left': 60,
            'bottom': 100,
            'right': 120,
            'x_coords': np.array([60, 120, 120, 60], dtype=np.int32),
            'y_coords': np.array([50, 50, 100, 100], dtype=np.int32),
            'paths': None,
            'name': 'offset',
        }
        scaled = scale_roi_dict(roi, 0.5, 0.5)
        assert scaled['top'] == 25
        assert scaled['left'] == 30
        assert scaled['bottom'] == 50
        assert scaled['right'] == 60
        np.testing.assert_array_equal(scaled['x_coords'], [30, 60, 60, 30])
        np.testing.assert_array_equal(scaled['y_coords'], [25, 25, 50, 50])

    def test_bbox_offset_composite(self):
        """Composite ROI path coordinates scale with non-zero offset."""
        roi = {
            'type': ROI_RECT,
            'top': 100,
            'left': 200,
            'bottom': 300,
            'right': 400,
            'x_coords': None,
            'y_coords': None,
            'paths': [[(200.0, 100.0), (400.0, 100.0),
                        (400.0, 300.0), (200.0, 300.0)]],
            'name': 'offset_comp',
        }
        scaled = scale_roi_dict(roi, 0.25, 0.25)
        assert scaled['top'] == 25
        assert scaled['left'] == 50
        path = scaled['paths'][0]
        assert path[0] == (50.0, 25.0)
        assert path[1] == (100.0, 25.0)

    def test_upscale_rasterizes_larger(self):
        """Scaling up a ROI and rasterizing in a larger canvas gives bigger region."""
        roi = {
            'type': ROI_RECT,
            'top': 10,
            'left': 10,
            'bottom': 30,
            'right': 30,
            'x_coords': None,
            'y_coords': None,
            'paths': None,
        }
        mask_small = imagej_roi_to_mask(roi, 40, 40)
        scaled = scale_roi_dict(roi, 2.0, 2.0)
        mask_big = imagej_roi_to_mask(scaled, 80, 80)

        # Scaled mask should have ~4x the filled area
        count_small = np.count_nonzero(mask_small)
        count_big = np.count_nonzero(mask_big)
        assert count_big == count_small * 4
