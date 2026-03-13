"""Extended tests for montaris.core.display_modes covering Compositor output,
single/multi-channel display, empty inputs, and all DisplayMode variants."""

import numpy as np
import pytest
from montaris.core.display_modes import DisplayMode, DisplayCompositor, FALSE_COLOR_LUTS


class TestComposeRGBOutput:
    """Verify compose() always returns (H, W, 3) uint8 arrays."""

    def test_single_channel_returns_rgb_shape(self):
        comp = DisplayCompositor()
        ch = np.full((32, 48), 128, dtype=np.uint8)
        result = comp.compose([ch], DisplayMode.FALSE_COLOR)
        assert result.shape == (32, 48, 3)
        assert result.dtype == np.uint8

    def test_multi_channel_returns_rgb_shape(self):
        comp = DisplayCompositor()
        chs = [np.full((20, 30), v, dtype=np.uint8) for v in (50, 100, 200)]
        result = comp.compose(chs, DisplayMode.FALSE_COLOR)
        assert result.shape == (20, 30, 3)
        assert result.dtype == np.uint8


class TestSingleChannelDisplay:
    """Single-channel tinted grayscale output."""

    def test_single_channel_false_color_tints_red(self):
        """Default LUT for channel 0 is red, so G and B should be ~0."""
        comp = DisplayCompositor()
        ch = np.full((10, 10), 255, dtype=np.uint8)
        result = comp.compose([ch], DisplayMode.FALSE_COLOR)
        assert result[:, :, 0].mean() == pytest.approx(255, abs=1)
        assert result[:, :, 1].mean() == pytest.approx(0, abs=1)
        assert result[:, :, 2].mean() == pytest.approx(0, abs=1)

    def test_single_channel_grayscale_all_equal(self):
        """Grayscale mode: R == G == B for a single channel."""
        comp = DisplayCompositor()
        ch = np.full((10, 10), 180, dtype=np.uint8)
        result = comp.compose([ch], DisplayMode.GRAYSCALE)
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result[:, :, 1], result[:, :, 2])


class TestMultiChannelComposite:
    """Multi-channel blending / composite tests."""

    def test_two_channel_false_color_blend(self):
        """Two channels with default LUTs (red + green) should produce yellow-ish."""
        comp = DisplayCompositor()
        ch1 = np.full((10, 10), 255, dtype=np.uint8)
        ch2 = np.full((10, 10), 255, dtype=np.uint8)
        result = comp.compose([ch1, ch2], DisplayMode.FALSE_COLOR)
        # Red channel should be ~255, Green channel should be ~255
        assert result[:, :, 0].mean() == pytest.approx(255, abs=1)
        assert result[:, :, 1].mean() == pytest.approx(255, abs=1)
        assert result[:, :, 2].mean() == pytest.approx(0, abs=1)

    def test_composite_rgb_maps_channels_to_planes(self):
        """COMPOSITE_RGB should map ch0->R, ch1->G, ch2->B."""
        comp = DisplayCompositor()
        ch_r = np.full((10, 10), 255, dtype=np.uint8)
        ch_g = np.full((10, 10), 0, dtype=np.uint8)
        ch_b = np.full((10, 10), 128, dtype=np.uint8)
        result = comp.compose([ch_r, ch_g, ch_b], DisplayMode.COMPOSITE_RGB)
        assert result[:, :, 0].mean() == pytest.approx(255, abs=1)
        assert result[:, :, 1].mean() == pytest.approx(0, abs=1)
        assert result[:, :, 2].mean() == pytest.approx(128, abs=1)


class TestEmptyActiveIndices:
    """Empty channel list should produce a black 1x1 image."""

    def test_empty_list_returns_black(self):
        comp = DisplayCompositor()
        result = comp.compose([])
        assert result.shape == (1, 1, 3)
        assert result.sum() == 0

    def test_empty_list_with_each_mode(self):
        comp = DisplayCompositor()
        for mode in DisplayMode:
            result = comp.compose([], mode)
            assert result.shape == (1, 1, 3)
            assert result.sum() == 0


class TestDisplayModes:
    """Cover FALSE_COLOR, GRAYSCALE, COMPOSITE_RGB, MAX_PROJECTION, FIRST_CHANNEL."""

    def test_false_color_custom_lut_green(self):
        comp = DisplayCompositor()
        comp.set_lut(0, [0.0, 1.0, 0.0])
        ch = np.full((8, 8), 200, dtype=np.uint8)
        result = comp.compose([ch], DisplayMode.FALSE_COLOR)
        # Green should dominate
        assert result[:, :, 1].mean() > result[:, :, 0].mean()
        assert result[:, :, 1].mean() > result[:, :, 2].mean()

    def test_grayscale_multi_averages(self):
        comp = DisplayCompositor()
        ch1 = np.full((10, 10), 0, dtype=np.uint8)
        ch2 = np.full((10, 10), 255, dtype=np.uint8)
        result = comp.compose([ch1, ch2], DisplayMode.GRAYSCALE)
        # Average of 0 and 1.0 normalised = 0.5 -> ~127-128
        mean_val = result[:, :, 0].mean()
        assert 120 < mean_val < 135

    def test_max_projection_picks_brighter(self):
        comp = DisplayCompositor()
        ch_dim = np.full((10, 10), 50, dtype=np.uint8)
        ch_bright = np.full((10, 10), 250, dtype=np.uint8)
        result = comp.compose([ch_dim, ch_bright], DisplayMode.MAX_PROJECTION)
        assert result[:, :, 0].mean() > 240

    def test_first_channel_ignores_others(self):
        comp = DisplayCompositor()
        ch1 = np.full((10, 10), 100, dtype=np.uint8)
        ch2 = np.full((10, 10), 200, dtype=np.uint8)
        result = comp.compose([ch1, ch2], DisplayMode.FIRST_CHANNEL)
        assert np.array_equal(result[:, :, 0], ch1)

    def test_composite_rgb_fourth_channel_distributed(self):
        """Channels beyond 3 are distributed evenly across R, G, B."""
        comp = DisplayCompositor()
        chs = [np.full((8, 8), 0, dtype=np.uint8) for _ in range(4)]
        chs[3] = np.full((8, 8), 255, dtype=np.uint8)  # 4th channel -> +1/3 each
        result = comp.compose(chs, DisplayMode.COMPOSITE_RGB)
        # All three planes should get roughly equal contribution from ch4
        r_mean = result[:, :, 0].mean()
        g_mean = result[:, :, 1].mean()
        b_mean = result[:, :, 2].mean()
        assert abs(r_mean - g_mean) < 5
        assert abs(g_mean - b_mean) < 5
        assert r_mean > 50  # non-trivial contribution


class TestNormalization:
    """Test _normalize with different dtypes."""

    def test_uint16_normalisation(self):
        comp = DisplayCompositor()
        ch = np.full((5, 5), 65535, dtype=np.uint16)
        result = comp.compose([ch], DisplayMode.FIRST_CHANNEL)
        assert result[:, :, 0].mean() == pytest.approx(255, abs=1)

    def test_float_normalisation(self):
        comp = DisplayCompositor()
        ch = np.full((5, 5), 0.5, dtype=np.float32)
        result = comp.compose([ch], DisplayMode.FIRST_CHANNEL)
        # float range [0.5, 0.5] -> all same -> half gray
        # _normalize: mx == mn, mx > 0 -> 0.5
        expected = int(0.5 * 255)
        assert abs(result[:, :, 0].mean() - expected) < 2

    def test_float_all_zero(self):
        comp = DisplayCompositor()
        ch = np.zeros((5, 5), dtype=np.float32)
        result = comp.compose([ch], DisplayMode.FIRST_CHANNEL)
        assert result[:, :, 0].mean() == pytest.approx(0, abs=1)
