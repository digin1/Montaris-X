"""Extended tests for montaris.io.image_io functions.

Covers load_image, load_image_stack, _split_tiff_channels, and _pil_to_array
using temporary files and synthetic data.
"""
import numpy as np
import pytest
from PIL import Image

from montaris.io.image_io import (
    load_image,
    load_image_stack,
    _split_tiff_channels,
    _pil_to_array,
)


# ---------------------------------------------------------------------------
# 1. load_image with a temp PNG file (grayscale)
# ---------------------------------------------------------------------------

class TestLoadImageGrayscale:
    def test_grayscale_png(self, tmp_path):
        arr = np.arange(100, dtype=np.uint8).reshape(10, 10)
        img = Image.fromarray(arr, mode='L')
        path = tmp_path / "gray.png"
        img.save(str(path))

        result = load_image(path)
        assert result.dtype == np.uint8
        assert result.ndim == 2
        np.testing.assert_array_equal(result, arr)

    def test_grayscale_shape(self, tmp_path):
        arr = np.zeros((25, 40), dtype=np.uint8)
        img = Image.fromarray(arr, mode='L')
        path = tmp_path / "gray_shape.png"
        img.save(str(path))

        result = load_image(path)
        assert result.shape == (25, 40)


# ---------------------------------------------------------------------------
# 2. load_image with a temp RGB PNG
# ---------------------------------------------------------------------------

class TestLoadImageRGB:
    def test_rgb_png(self, tmp_path):
        arr = np.zeros((20, 30, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # red channel
        img = Image.fromarray(arr, mode='RGB')
        path = tmp_path / "rgb.png"
        img.save(str(path))

        result = load_image(path)
        assert result.ndim == 3
        assert result.shape == (20, 30, 3)
        # Red channel should be 255
        assert np.all(result[:, :, 0] == 255)
        assert np.all(result[:, :, 1] == 0)
        assert np.all(result[:, :, 2] == 0)

    def test_rgb_png_dtype(self, tmp_path):
        arr = np.full((10, 10, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr, mode='RGB')
        path = tmp_path / "rgb_dtype.png"
        img.save(str(path))

        result = load_image(path)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# 3. load_image_stack with single-channel image -> returns 1-element list
# ---------------------------------------------------------------------------

class TestLoadImageStackSingleChannel:
    def test_single_channel_png(self, tmp_path):
        arr = np.full((15, 15), 42, dtype=np.uint8)
        img = Image.fromarray(arr, mode='L')
        path = tmp_path / "single.png"
        img.save(str(path))

        result = load_image_stack(path)
        assert isinstance(result, list)
        assert len(result) == 1
        name, data = result[0]
        assert name == "single"
        np.testing.assert_array_equal(data, arr)

    def test_single_channel_rgb_png(self, tmp_path):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode='RGB')
        path = tmp_path / "rgb_stack.png"
        img.save(str(path))

        result = load_image_stack(path)
        assert len(result) == 1
        name, data = result[0]
        assert name == "rgb_stack"
        assert data.shape == (10, 10, 3)


# ---------------------------------------------------------------------------
# 4. _split_tiff_channels with 2D array -> returns single entry
# ---------------------------------------------------------------------------

class TestSplitTiffChannels2D:
    def test_2d_grayscale(self):
        data = np.zeros((50, 60), dtype=np.uint8)
        result = _split_tiff_channels(data, "mystem")
        assert len(result) == 1
        name, arr = result[0]
        assert name == "mystem"
        assert arr.shape == (50, 60)
        np.testing.assert_array_equal(arr, data)

    def test_2d_preserves_values(self):
        data = np.arange(12, dtype=np.uint16).reshape(3, 4)
        result = _split_tiff_channels(data, "test")
        assert len(result) == 1
        np.testing.assert_array_equal(result[0][1], data)


# ---------------------------------------------------------------------------
# 5. _split_tiff_channels with 3D channels-first RGB (3, H, W) -> moveaxis
# ---------------------------------------------------------------------------

class TestSplitTiffChannelsFirst:
    def test_channels_first_rgb(self):
        # (3, 100, 200) — channels-first
        data = np.zeros((3, 100, 200), dtype=np.uint8)
        data[0] = 10  # R
        data[1] = 20  # G
        data[2] = 30  # B
        result = _split_tiff_channels(data, "cfirst")
        assert len(result) == 1
        name, arr = result[0]
        assert name == "cfirst"
        assert arr.shape == (100, 200, 3)
        assert np.all(arr[:, :, 0] == 10)
        assert np.all(arr[:, :, 1] == 20)
        assert np.all(arr[:, :, 2] == 30)

    def test_channels_first_rgba(self):
        data = np.zeros((4, 50, 80), dtype=np.uint8)
        result = _split_tiff_channels(data, "rgba")
        assert len(result) == 1
        assert result[0][1].shape == (50, 80, 4)


# ---------------------------------------------------------------------------
# 6. _split_tiff_channels with 3D channels-last RGB (H, W, 3) -> pass through
# ---------------------------------------------------------------------------

class TestSplitTiffChannelsLast:
    def test_channels_last_rgb(self):
        data = np.zeros((100, 200, 3), dtype=np.uint8)
        data[:, :, 2] = 255
        result = _split_tiff_channels(data, "clast")
        assert len(result) == 1
        name, arr = result[0]
        assert name == "clast"
        assert arr.shape == (100, 200, 3)
        np.testing.assert_array_equal(arr, data)

    def test_channels_last_rgba(self):
        data = np.zeros((80, 90, 4), dtype=np.uint8)
        result = _split_tiff_channels(data, "clast_rgba")
        assert len(result) == 1
        assert result[0][1].shape == (80, 90, 4)
        np.testing.assert_array_equal(result[0][1], data)


# ---------------------------------------------------------------------------
# 7. _split_tiff_channels with 3D single-channel (1, H, W) -> squeeze
# ---------------------------------------------------------------------------

class TestSplitTiffSingleChannel:
    def test_single_channel_squeezed(self):
        data = np.full((1, 64, 64), 42, dtype=np.uint8)
        result = _split_tiff_channels(data, "single")
        assert len(result) == 1
        name, arr = result[0]
        assert name == "single"
        assert arr.ndim == 2
        assert arr.shape == (64, 64)
        assert np.all(arr == 42)


# ---------------------------------------------------------------------------
# 8. _split_tiff_channels with 3D multi-channel (5, H, W) -> split into 5
# ---------------------------------------------------------------------------

class TestSplitTiffMultiChannel:
    def test_five_channels_split(self):
        data = np.zeros((5, 30, 40), dtype=np.uint8)
        for i in range(5):
            data[i] = i * 50
        result = _split_tiff_channels(data, "multi")
        assert len(result) == 5
        for i, (name, arr) in enumerate(result):
            assert name == f"multi_ch{i}"
            assert arr.ndim == 2
            assert arr.shape == (30, 40)
            assert np.all(arr == i * 50)

    def test_seven_channels_split(self):
        data = np.zeros((7, 20, 25), dtype=np.uint16)
        result = _split_tiff_channels(data, "stem")
        assert len(result) == 7
        assert all(name == f"stem_ch{i}" for i, (name, _) in enumerate(result))

    def test_two_channels_not_rgb(self):
        """shape[0]=2 is not RGB/RGBA, and shape[2]!=3/4, so should split."""
        data = np.zeros((2, 50, 60), dtype=np.uint8)
        data[0] = 100
        data[1] = 200
        result = _split_tiff_channels(data, "two")
        assert len(result) == 2
        np.testing.assert_array_equal(result[0][1], np.full((50, 60), 100, dtype=np.uint8))
        np.testing.assert_array_equal(result[1][1], np.full((50, 60), 200, dtype=np.uint8))


# ---------------------------------------------------------------------------
# 9. _split_tiff_channels with 4D array -> recursive split
# ---------------------------------------------------------------------------

class TestSplitTiffChannels4D:
    def test_4d_simple(self):
        # (2, 3, 20, 30) — 2 pages, each with 3 channels-first
        # shape[0]=3 with shape[0]<shape[1] triggers moveaxis
        data = np.zeros((2, 3, 20, 30), dtype=np.uint8)
        result = _split_tiff_channels(data, "vol")
        # Each page recurse: (3, 20, 30) -> channels-first -> 1 entry per page
        assert len(result) == 2
        for i, (name, arr) in enumerate(result):
            assert name == f"vol_p{i}"
            assert arr.shape == (20, 30, 3)

    def test_4d_multi_channel_pages(self):
        # (3, 5, 16, 16) — 3 pages, each (5, 16, 16) -> 5 channels each
        data = np.zeros((3, 5, 16, 16), dtype=np.uint8)
        result = _split_tiff_channels(data, "stack")
        # Each page is (5, 16, 16) -> 5 channels
        assert len(result) == 15
        assert result[0][0] == "stack_p0_ch0"
        assert result[4][0] == "stack_p0_ch4"
        assert result[5][0] == "stack_p1_ch0"

    def test_4d_grayscale_pages(self):
        # (4, 1, 32, 32) — 4 pages, each single-channel
        data = np.zeros((4, 1, 32, 32), dtype=np.uint8)
        result = _split_tiff_channels(data, "gs")
        assert len(result) == 4
        for i, (name, arr) in enumerate(result):
            assert name == f"gs_p{i}"
            assert arr.ndim == 2
            assert arr.shape == (32, 32)


# ---------------------------------------------------------------------------
# 10. _pil_to_array with 'L' mode image
# ---------------------------------------------------------------------------

class TestPilToArrayL:
    def test_l_mode(self):
        arr = np.arange(20, dtype=np.uint8).reshape(4, 5)
        img = Image.fromarray(arr, mode='L')
        result = _pil_to_array(img)
        assert result.dtype == np.uint8
        assert result.ndim == 2
        np.testing.assert_array_equal(result, arr)

    def test_l_mode_shape(self):
        img = Image.new('L', (30, 20), color=128)
        result = _pil_to_array(img)
        # PIL size is (width, height), numpy shape is (height, width)
        assert result.shape == (20, 30)
        assert np.all(result == 128)


# ---------------------------------------------------------------------------
# 11. _pil_to_array with 'P' palette mode image
# ---------------------------------------------------------------------------

class TestPilToArrayPalette:
    def test_p_mode_no_transparency(self):
        """P mode without transparency converts to RGB."""
        img = Image.new('P', (10, 10))
        # Put a simple palette
        palette = list(range(256)) * 3
        img.putpalette(palette)
        result = _pil_to_array(img)
        assert result.ndim == 3
        assert result.shape[2] == 3  # RGB

    def test_p_mode_with_transparency(self):
        """P mode with transparency info converts to RGBA."""
        img = Image.new('P', (8, 8))
        palette = list(range(256)) * 3
        img.putpalette(palette)
        img.info['transparency'] = 0
        result = _pil_to_array(img)
        assert result.ndim == 3
        assert result.shape[2] == 4  # RGBA

    def test_p_mode_dtype(self):
        img = Image.new('P', (5, 5))
        palette = [0] * 768
        img.putpalette(palette)
        result = _pil_to_array(img)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# 12. _pil_to_array with 'RGBA' mode image
# ---------------------------------------------------------------------------

class TestPilToArrayRGBA:
    def test_rgba_mode(self):
        arr = np.zeros((15, 20, 4), dtype=np.uint8)
        arr[:, :, 0] = 255  # R
        arr[:, :, 3] = 128  # A
        img = Image.fromarray(arr, mode='RGBA')
        result = _pil_to_array(img)
        assert result.shape == (15, 20, 4)
        np.testing.assert_array_equal(result, arr)

    def test_rgba_preserves_alpha(self):
        img = Image.new('RGBA', (10, 10), color=(100, 150, 200, 50))
        result = _pil_to_array(img)
        assert result.shape == (10, 10, 4)
        assert np.all(result[:, :, 0] == 100)
        assert np.all(result[:, :, 1] == 150)
        assert np.all(result[:, :, 2] == 200)
        assert np.all(result[:, :, 3] == 50)

    def test_rgba_dtype(self):
        img = Image.new('RGBA', (5, 5), color=(0, 0, 0, 255))
        result = _pil_to_array(img)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Additional edge cases for _pil_to_array
# ---------------------------------------------------------------------------

class TestPilToArrayEdgeCases:
    def test_1bit_mode_converts_to_l(self):
        """'1' mode (1-bit) should be converted to 'L' then to array."""
        img = Image.new('1', (10, 10), color=1)
        result = _pil_to_array(img)
        assert result.ndim == 2
        assert result.dtype == np.uint8
        assert np.all(result == 255)

    def test_cmyk_converts_to_rgb(self):
        """CMYK mode should be converted to RGB."""
        img = Image.new('CMYK', (8, 8), color=(0, 0, 0, 0))
        result = _pil_to_array(img)
        assert result.ndim == 3
        assert result.shape[2] == 3
