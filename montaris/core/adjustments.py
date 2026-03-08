import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ImageAdjustments:
    """Image adjustment parameters.

    Manual sliders use a data-adaptive contrast model:
    - Contrast pivots around image mean, brightness is proportional
    - Exposure multiplies, gamma applies power curve

    Smart Auto uses direct window/level (display_min/max) for scientific images.
    """
    brightness: float = 0.0     # -1.0 to 1.0
    contrast: float = 0.0       # -1.0 to 1.0 (0 = identity)
    exposure: float = 0.0       # -2.0 to 2.0
    gamma: float = 1.0          # 0.1 to 5.0
    _pivot: float = 0.5         # data-adaptive contrast pivot (0-1 normalized)
    # Direct window/level override (set by Smart Auto, bypasses B/C/E)
    _window_min: float = -1.0   # -1 = disabled (use B/C/E model)
    _window_max: float = -1.0

    def is_identity(self):
        return (self.brightness == 0.0 and self.contrast == 0.0
                and self.exposure == 0.0 and self.gamma == 1.0
                and self._window_min < 0)

    def reset(self):
        self.brightness = 0.0
        self.contrast = 0.0
        self.exposure = 0.0
        self.gamma = 1.0
        self._window_min = -1.0
        self._window_max = -1.0

    def set_pivot(self, image_data):
        """Set contrast pivot from image data (call when image changes)."""
        if image_data is None:
            self._pivot = 0.5
            return
        sample = image_data.ravel()
        if len(sample) > 100000:
            sample = sample[::len(sample) // 100000]
        vals = sample.astype(np.float64)
        if image_data.dtype == np.uint8:
            vals /= 255.0
        elif image_data.dtype == np.uint16:
            mn, mx = float(image_data.min()), float(image_data.max())
            if mx > mn:
                vals = (vals - mn) / (mx - mn)
            else:
                vals /= 65535.0
        else:
            mn, mx = vals.min(), vals.max()
            if mx > mn:
                vals = (vals - mn) / (mx - mn)
        self._pivot = float(np.clip(vals.mean(), 0.01, 0.99))

    def _build_lut(self):
        """Build a 256-entry uint8 lookup table."""
        x = np.arange(256, dtype=np.float64) / 255.0

        if self._window_min >= 0:
            # Direct window/level mode (Smart Auto / Quick Boost)
            wmin = self._window_min
            wmax = self._window_max
            rng = max(wmax - wmin, 1.0 / 255.0)

            # Apply any manual slider adjustments ON TOP of the window
            # Exposure
            if self.exposure != 0.0:
                x = x * (2.0 ** self.exposure)
            # Brightness shifts the window
            b = self.brightness / 2.0
            if b < 0.0:
                x = x * (1.0 + b)
            elif b > 0.0:
                x = x + (1.0 - x) * b
            # Contrast adjusts window width
            if self.contrast != 0.0:
                slope = math.exp(self.contrast * 2.0)
                center = (wmin + wmax) / 2.0
                wmin = center - (center - wmin) / slope
                wmax = center + (wmax - center) / slope
                rng = max(wmax - wmin, 1.0 / 255.0)

            # Window/level mapping
            x = (x - wmin) / rng

            # Gamma
            if self.gamma != 1.0:
                np.clip(x, 0, 1, out=x)
                np.power(x, 1.0 / self.gamma, out=x)
        else:
            # Manual slider mode (no window/level)
            if self.exposure != 0.0:
                x = x * (2.0 ** self.exposure)

            b = self.brightness / 2.0
            if b < 0.0:
                x = x * (1.0 + b)
            elif b > 0.0:
                x = x + (1.0 - x) * b

            if self.contrast != 0.0:
                slope = math.exp(self.contrast * 2.0)
                pivot = self._pivot
                x = (x - pivot) * slope + pivot

            if self.gamma != 1.0:
                np.clip(x, 0, 1, out=x)
                np.power(x, 1.0 / self.gamma, out=x)

        np.clip(x * 255, 0, 255, out=x)
        return x.astype(np.uint8)

    def apply(self, image):
        """Apply adjustments. For uint8: pure LUT indexing (fast)."""
        if self.is_identity():
            if image.dtype == np.uint8:
                return image
            return np.clip(image, 0, 255).astype(np.uint8)

        lut = self._build_lut()

        if image.dtype == np.uint8:
            return lut[image]

        # Non-uint8: normalize to uint8 first, then apply LUT
        img = image.astype(np.float32)
        if image.dtype == np.uint16:
            mn, mx = float(image.min()), float(image.max())
            if mx > mn:
                img = (img - mn) * (255.0 / (mx - mn))
        else:
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) * (255.0 / (mx - mn))
        np.clip(img, 0, 255, out=img)
        return lut[img.astype(np.uint8)]

    @staticmethod
    def smart_auto(image):
        """Auto window/level for microscopy images.

        Designed for fluorescence/brightfield brain imaging:
        - Background (dark/empty areas) → pure black
        - Tissue signal → stretched to fill display range
        - Uses percentile analysis to find the signal floor/ceiling
        """
        adj = ImageAdjustments()
        adj.set_pivot(image)

        # Normalize to 0-1
        img = image.astype(np.float32)
        if image.dtype == np.uint8:
            img /= 255.0
        elif image.dtype == np.uint16:
            mn, mx = float(image.min()), float(image.max())
            if mx > mn:
                img = (img - mn) / (mx - mn)
            else:
                img /= 65535.0
        else:
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn)

        # Subsample for speed
        flat = img.ravel()
        if len(flat) > 500000:
            flat = flat[::len(flat) // 500000]

        # Find signal boundaries
        p01 = np.percentile(flat, 1)
        p25 = np.percentile(flat, 25)
        p50 = np.percentile(flat, 50)
        p75 = np.percentile(flat, 75)
        p99 = np.percentile(flat, 99)

        if p50 < 0.2:
            # Dark-dominant (fluorescence microscopy):
            # Background is the dominant mode. Signal floor = p75 neighborhood
            # Use the "elbow" between background and signal
            display_min = p50
            display_max = max(p99, display_min + 0.05)
        elif p50 > 0.8:
            # Bright-dominant (brightfield):
            display_min = p01
            display_max = p25
        else:
            # Balanced
            display_min = p01
            display_max = p99

        # Set direct window/level (values are in 0-1, map to 0-255 for LUT)
        adj._window_min = display_min
        adj._window_max = display_max

        # Gentle gamma boost for dark microscopy images
        mean_signal = float(flat[flat >= display_min].mean()) if (flat >= display_min).any() else 0.5
        if mean_signal < display_max * 0.4:
            adj.gamma = 1.3

        return adj

    @staticmethod
    def quick_boost(image):
        """Quick brightness/contrast boost for dim images."""
        adj = ImageAdjustments()
        adj.set_pivot(image)
        adj.contrast = 0.15
        adj.brightness = 0.15
        adj.gamma = 1.3
        return adj
