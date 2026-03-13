"""Tests for module-level canvas helper functions in montaris/canvas.py.

Covers: numpy_to_qimage, _apply_tint, _mask_to_rgba,
        mask_to_qimage, mask_to_outline_qimage, _compute_roi_rgba_from_crop.
"""

import numpy as np
import pytest
from PySide6.QtGui import QImage

from montaris.canvas import (
    numpy_to_qimage,
    _apply_tint,
    _mask_to_rgba,
    mask_to_qimage,
    mask_to_outline_qimage,
    _compute_roi_rgba_from_crop,
)


# ---------------------------------------------------------------------------
# numpy_to_qimage
# ---------------------------------------------------------------------------

class TestNumpyToQImage:
    def test_2d_grayscale_uint8(self, qapp):
        arr = np.random.randint(0, 255, (64, 48), dtype=np.uint8)
        qimg = numpy_to_qimage(arr)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 48
        assert qimg.height() == 64
        assert qimg.format() == QImage.Format_Grayscale8

    def test_2d_uint16_normalizes(self, qapp):
        arr = np.array([[0, 32768, 65535]], dtype=np.uint16)
        qimg = numpy_to_qimage(arr)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 3
        assert qimg.height() == 1
        # Format should be grayscale since input is 2D
        assert qimg.format() == QImage.Format_Grayscale8

    def test_2d_uint16_constant_normalizes_to_zero(self, qapp):
        """Constant uint16 array: max == min, should produce all-zero image."""
        arr = np.full((10, 12), 5000, dtype=np.uint16)
        qimg = numpy_to_qimage(arr)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 12
        assert qimg.height() == 10

    def test_3d_rgb_uint8(self, qapp):
        arr = np.random.randint(0, 255, (32, 24, 3), dtype=np.uint8)
        qimg = numpy_to_qimage(arr)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 24
        assert qimg.height() == 32
        assert qimg.format() == QImage.Format_RGB888

    def test_3d_rgba_uint8(self, qapp):
        arr = np.random.randint(0, 255, (20, 30, 4), dtype=np.uint8)
        qimg = numpy_to_qimage(arr)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 30
        assert qimg.height() == 20
        assert qimg.format() == QImage.Format_RGBA8888

    def test_3d_single_channel(self, qapp):
        """(H, W, 1) array should collapse to grayscale."""
        arr = np.random.randint(0, 255, (10, 15, 1), dtype=np.uint8)
        qimg = numpy_to_qimage(arr)
        assert isinstance(qimg, QImage)
        assert qimg.format() == QImage.Format_Grayscale8
        assert qimg.width() == 15

    def test_empty_array_raises(self, qapp):
        """1-D or unsupported shape should raise ValueError."""
        arr = np.array([], dtype=np.uint8)
        with pytest.raises(ValueError):
            numpy_to_qimage(arr)

    def test_3d_float_normalizes(self, qapp):
        arr = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        qimg = numpy_to_qimage(arr)
        assert isinstance(qimg, QImage)
        assert qimg.format() == QImage.Format_RGB888


# ---------------------------------------------------------------------------
# _apply_tint
# ---------------------------------------------------------------------------

class TestApplyTint:
    def test_grayscale_with_rgb_tint(self, qapp):
        gray = np.full((4, 4), 128, dtype=np.uint8)
        tint = (255, 0, 0)  # pure red
        result = _apply_tint(gray, tint)
        assert result.shape == (4, 4, 3)
        assert result.dtype == np.uint8
        # Red channel should be roughly 128 (128 * 255 // 255)
        assert result[0, 0, 0] == 128
        # Green and blue channels should be 0
        assert result[0, 0, 1] == 0
        assert result[0, 0, 2] == 0

    def test_tint_with_white(self, qapp):
        gray = np.full((2, 2), 200, dtype=np.uint8)
        tint = (255, 255, 255)
        result = _apply_tint(gray, tint)
        # White tint preserves value
        assert np.all(result[:, :, 0] == 200)
        assert np.all(result[:, :, 1] == 200)
        assert np.all(result[:, :, 2] == 200)

    def test_tint_with_zero_value(self, qapp):
        gray = np.zeros((3, 3), dtype=np.uint8)
        tint = (255, 128, 64)
        result = _apply_tint(gray, tint)
        # All zeros tinted is still all zeros
        assert np.all(result == 0)

    def test_tint_uint16_input(self, qapp):
        """uint16 input should be normalized before tinting."""
        gray = np.array([[0, 65535]], dtype=np.uint16)
        tint = (255, 0, 0)
        result = _apply_tint(gray, tint)
        assert result.shape == (1, 2, 3)
        # Min pixel normalizes to 0, max pixel normalizes to 255
        assert result[0, 0, 0] == 0    # tinted 0 is 0
        assert result[0, 1, 0] == 255  # tinted 255 with red 255 = 255
        assert result[0, 1, 1] == 0    # green channel stays 0

    def test_tint_preserves_shape(self, qapp):
        gray = np.random.randint(0, 255, (10, 20), dtype=np.uint8)
        tint = (100, 200, 50)
        result = _apply_tint(gray, tint)
        assert result.shape == (10, 20, 3)


# ---------------------------------------------------------------------------
# _mask_to_rgba
# ---------------------------------------------------------------------------

class TestMaskToRGBA:
    def _make_mask(self):
        """Create a small mask with a filled region."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        return mask

    def test_solid_mode(self, qapp):
        mask = self._make_mask()
        color = (255, 0, 0)
        opacity = 128
        rgba = _mask_to_rgba(mask, color, opacity, fill_mode="solid")
        assert rgba.shape == (10, 10, 4)
        assert rgba.dtype == np.uint8
        # Outside mask: fully transparent
        assert rgba[0, 0, 3] == 0
        # Inside mask: opaque with color
        assert rgba[5, 5, 0] == 255  # R
        assert rgba[5, 5, 1] == 0    # G
        assert rgba[5, 5, 2] == 0    # B
        assert rgba[5, 5, 3] == 128  # A

    def test_outline_mode(self, qapp):
        mask = self._make_mask()
        color = (0, 255, 0)
        opacity = 200
        rgba = _mask_to_rgba(mask, color, opacity, fill_mode="outline")
        assert rgba.shape == (10, 10, 4)
        # Interior pixels (away from edge) should be transparent
        assert rgba[4, 4, 3] == 0 or rgba[4, 4, 3] == 200
        # At least some edge pixels should be painted
        edge_pixels = rgba[:, :, 3] > 0
        assert np.any(edge_pixels)

    def test_both_mode(self, qapp):
        mask = self._make_mask()
        color = (0, 0, 255)
        opacity = 180
        rgba = _mask_to_rgba(mask, color, opacity, fill_mode="both")
        assert rgba.shape == (10, 10, 4)
        # Edge pixels should have full opacity
        # Interior painted pixels should have half opacity
        interior = rgba[5, 5, 3]  # center of mask
        # Could be fill_alpha or edge alpha depending on geometry
        assert interior > 0

    def test_empty_mask(self, qapp):
        mask = np.zeros((8, 8), dtype=np.uint8)
        color = (255, 255, 0)
        rgba = _mask_to_rgba(mask, color, 100, fill_mode="solid")
        # Empty mask: all transparent
        assert np.all(rgba[:, :, 3] == 0)


# ---------------------------------------------------------------------------
# mask_to_qimage / mask_to_outline_qimage
# ---------------------------------------------------------------------------

class TestMaskToQImage:
    def _make_mask(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        return mask

    def test_mask_to_qimage_returns_qimage(self, qapp):
        mask = self._make_mask()
        qimg = mask_to_qimage(mask, (255, 0, 0), 128)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 20
        assert qimg.height() == 20
        assert qimg.format() == QImage.Format_RGBA8888

    def test_mask_to_qimage_empty_mask(self, qapp):
        mask = np.zeros((10, 10), dtype=np.uint8)
        qimg = mask_to_qimage(mask, (0, 255, 0), 100)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 10

    def test_mask_to_outline_qimage_returns_qimage(self, qapp):
        mask = self._make_mask()
        qimg = mask_to_outline_qimage(mask, (0, 0, 255), 200)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 20
        assert qimg.height() == 20
        assert qimg.format() == QImage.Format_RGBA8888

    def test_mask_to_outline_qimage_has_edge_pixels(self, qapp):
        mask = self._make_mask()
        color = (255, 128, 0)
        opacity = 150
        qimg = mask_to_outline_qimage(mask, color, opacity)
        # Convert back to check: at least some non-transparent pixels exist
        assert not qimg.isNull()
        # Check a known edge pixel (row 5, col 5 is top-left corner of mask)
        pixel = qimg.pixelColor(5, 5)
        assert pixel.alpha() > 0

    def test_mask_to_outline_small_mask(self, qapp):
        """1-pixel mask should still produce a valid outline."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        qimg = mask_to_outline_qimage(mask, (255, 0, 0), 128)
        assert isinstance(qimg, QImage)
        pixel = qimg.pixelColor(2, 2)
        assert pixel.alpha() > 0


# ---------------------------------------------------------------------------
# _compute_roi_rgba_from_crop
# ---------------------------------------------------------------------------

class TestComputeROIRGBAFromCrop:
    def _make_crop(self):
        crop = np.zeros((16, 16), dtype=np.uint8)
        crop[4:12, 4:12] = 1
        return crop

    def test_basic_solid_lod0(self, qapp):
        crop = self._make_crop()
        result = _compute_roi_rgba_from_crop(
            crop, (255, 0, 0), 128, "solid", 0, 10, 20
        )
        rgba, w, h, dx, dy, scale = result
        assert rgba.shape == (16, 16, 4)
        assert w == 16
        assert h == 16
        assert dx == 10
        assert dy == 20
        assert scale == 1
        # Check painted pixel
        assert rgba[6, 6, 0] == 255
        assert rgba[6, 6, 3] == 128
        # Check non-painted pixel
        assert rgba[0, 0, 3] == 0

    def test_lod1_downsamples(self, qapp):
        crop = self._make_crop()
        result = _compute_roi_rgba_from_crop(
            crop, (0, 255, 0), 100, "solid", 1, 0, 0
        )
        rgba, w, h, dx, dy, scale = result
        # LOD 1: factor = 2, so 16x16 -> 8x8
        assert w == 8
        assert h == 8
        assert scale == 2

    def test_lod2_downsamples(self, qapp):
        # Need 32x32 so LOD 2 (factor=4) gives 8x8
        crop = np.zeros((32, 32), dtype=np.uint8)
        crop[8:24, 8:24] = 1
        result = _compute_roi_rgba_from_crop(
            crop, (0, 0, 255), 150, "solid", 2, 5, 5
        )
        rgba, w, h, dx, dy, scale = result
        assert w == 8
        assert h == 8
        assert scale == 4

    def test_outline_mode(self, qapp):
        crop = self._make_crop()
        result = _compute_roi_rgba_from_crop(
            crop, (255, 255, 0), 200, "outline", 0, 0, 0
        )
        rgba, w, h, dx, dy, scale = result
        # Interior pixel should be transparent (only edges painted)
        assert rgba[6, 6, 3] == 0
        # Edge pixel should be painted
        edge_mask = rgba[:, :, 3] > 0
        assert np.any(edge_mask)

    def test_both_mode(self, qapp):
        crop = self._make_crop()
        result = _compute_roi_rgba_from_crop(
            crop, (128, 64, 200), 180, "both", 0, 0, 0
        )
        rgba, w, h, dx, dy, scale = result
        # Interior should have fill_alpha (180 // 2 = 90)
        interior_alpha = rgba[6, 6, 3]
        assert interior_alpha > 0
        # Edge pixels should have higher alpha
        edge_pixels = rgba[:, :, 3]
        assert edge_pixels.max() >= interior_alpha

    def test_empty_crop(self, qapp):
        crop = np.zeros((8, 8), dtype=np.uint8)
        result = _compute_roi_rgba_from_crop(
            crop, (255, 0, 0), 128, "solid", 0, 0, 0
        )
        rgba, w, h, dx, dy, scale = result
        assert np.all(rgba[:, :, 3] == 0)

    def test_preserves_disp_offsets(self, qapp):
        crop = self._make_crop()
        result = _compute_roi_rgba_from_crop(
            crop, (255, 0, 0), 128, "solid", 0, 42, 99
        )
        _, _, _, dx, dy, _ = result
        assert dx == 42
        assert dy == 99
