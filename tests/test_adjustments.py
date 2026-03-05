import numpy as np
import pytest
from montaris.core.adjustments import ImageAdjustments


class TestImageAdjustments:
    def test_identity(self):
        adj = ImageAdjustments()
        assert adj.is_identity()
        img = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        result = adj.apply(img)
        assert np.array_equal(result, img)

    def test_brightness_up(self):
        adj = ImageAdjustments(brightness=0.2)
        img = np.full((50, 60), 100, dtype=np.uint8)
        result = adj.apply(img)
        assert result.mean() > 100

    def test_brightness_down(self):
        adj = ImageAdjustments(brightness=-0.2)
        img = np.full((50, 60), 100, dtype=np.uint8)
        result = adj.apply(img)
        assert result.mean() < 100

    def test_contrast_increase(self):
        adj = ImageAdjustments(contrast=2.0)
        img = np.array([[80, 180]], dtype=np.uint8)
        result = adj.apply(img)
        # Higher contrast = more spread
        diff_before = 180 - 80
        diff_after = int(result[0, 1]) - int(result[0, 0])
        assert diff_after >= diff_before

    def test_exposure_increase(self):
        adj = ImageAdjustments(exposure=1.0)
        img = np.full((50, 60), 100, dtype=np.uint8)
        result = adj.apply(img)
        assert result.mean() > 100

    def test_gamma(self):
        adj = ImageAdjustments(gamma=2.0)
        img = np.full((50, 60), 100, dtype=np.uint8)
        result = adj.apply(img)
        # Gamma > 1 brightens dark areas
        assert result.dtype == np.uint8

    def test_reset(self):
        adj = ImageAdjustments(brightness=0.5, contrast=2.0)
        adj.reset()
        assert adj.is_identity()

    def test_apply_uint16(self):
        adj = ImageAdjustments(brightness=0.1)
        img = np.full((50, 60), 30000, dtype=np.uint16)
        result = adj.apply(img)
        assert result.dtype == np.uint8
        assert result.mean() > 0

    def test_smart_auto(self):
        img = np.random.randint(50, 100, (50, 60), dtype=np.uint8)
        adj = ImageAdjustments.smart_auto(img)
        assert not adj.is_identity()
        # Auto should increase contrast for narrow range image
        assert adj.contrast > 1.0

    def test_quick_boost(self):
        img = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        adj = ImageAdjustments.quick_boost(img)
        assert adj.contrast > 1.0
        assert adj.brightness > 0.0

    def test_clipping(self):
        adj = ImageAdjustments(brightness=2.0)
        img = np.full((50, 60), 200, dtype=np.uint8)
        result = adj.apply(img)
        assert result.max() <= 255
        assert result.min() >= 0
