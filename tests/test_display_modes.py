import numpy as np
import pytest
from montaris.core.display_modes import DisplayMode, DisplayCompositor


class TestDisplayCompositor:
    def test_first_channel(self):
        comp = DisplayCompositor()
        ch = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        result = comp.compose([ch], DisplayMode.FIRST_CHANNEL)
        assert result.shape == (50, 60, 3)
        assert result.dtype == np.uint8
        assert np.array_equal(result[:, :, 0], ch)

    def test_grayscale_single(self):
        comp = DisplayCompositor()
        ch = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        result = comp.compose([ch], DisplayMode.GRAYSCALE)
        assert result.shape == (50, 60, 3)

    def test_grayscale_multi(self):
        comp = DisplayCompositor()
        ch1 = np.full((50, 60), 100, dtype=np.uint8)
        ch2 = np.full((50, 60), 200, dtype=np.uint8)
        result = comp.compose([ch1, ch2], DisplayMode.GRAYSCALE)
        assert result.shape == (50, 60, 3)

    def test_max_projection(self):
        comp = DisplayCompositor()
        ch1 = np.full((50, 60), 50, dtype=np.uint8)
        ch2 = np.full((50, 60), 200, dtype=np.uint8)
        result = comp.compose([ch1, ch2], DisplayMode.MAX_PROJECTION)
        assert result.shape == (50, 60, 3)
        # Max should be close to 200/255 * 255 = 200
        assert result[:, :, 0].mean() > 150

    def test_composite_rgb(self):
        comp = DisplayCompositor()
        ch_r = np.full((50, 60), 200, dtype=np.uint8)
        ch_g = np.full((50, 60), 100, dtype=np.uint8)
        ch_b = np.full((50, 60), 50, dtype=np.uint8)
        result = comp.compose([ch_r, ch_g, ch_b], DisplayMode.COMPOSITE_RGB)
        assert result.shape == (50, 60, 3)
        assert result[:, :, 0].mean() > result[:, :, 2].mean()

    def test_false_color(self):
        comp = DisplayCompositor()
        ch1 = np.full((50, 60), 200, dtype=np.uint8)
        ch2 = np.full((50, 60), 100, dtype=np.uint8)
        result = comp.compose([ch1, ch2], DisplayMode.FALSE_COLOR)
        assert result.shape == (50, 60, 3)

    def test_empty_channels(self):
        comp = DisplayCompositor()
        result = comp.compose([], DisplayMode.FIRST_CHANNEL)
        assert result.shape == (1, 1, 3)

    def test_custom_lut(self):
        comp = DisplayCompositor()
        comp.set_lut(0, [0.0, 1.0, 0.0])  # Green
        ch = np.full((50, 60), 255, dtype=np.uint8)
        result = comp.compose([ch], DisplayMode.FALSE_COLOR)
        assert result[:, :, 1].mean() > result[:, :, 0].mean()  # Green > Red
