import numpy as np
from dataclasses import dataclass, field


@dataclass
class ImageAdjustments:
    """Image adjustment parameters."""
    brightness: float = 0.0     # -1.0 to 1.0
    contrast: float = 1.0       # 0.0 to 3.0
    exposure: float = 0.0       # -2.0 to 2.0
    gamma: float = 1.0          # 0.1 to 5.0

    def is_identity(self):
        return (self.brightness == 0.0 and self.contrast == 1.0
                and self.exposure == 0.0 and self.gamma == 1.0)

    def reset(self):
        self.brightness = 0.0
        self.contrast = 1.0
        self.exposure = 0.0
        self.gamma = 1.0

    def _build_lut(self):
        """Build a 256-entry uint8 lookup table (ImageJ-style).

        O(256) instead of O(W*H) — maps each input intensity to its
        adjusted output via the same math as the old per-pixel apply().
        """
        x = np.arange(256, dtype=np.float32) / 255.0

        if self.exposure != 0.0:
            x = x * (2.0 ** self.exposure)
        if self.contrast != 1.0:
            x = (x - 0.5) * self.contrast + 0.5
        if self.brightness != 0.0:
            x = x + self.brightness
        if self.gamma != 1.0:
            np.clip(x, 0, 1, out=x)
            np.power(x, 1.0 / self.gamma, out=x)

        np.clip(x * 255, 0, 255, out=x)
        return x.astype(np.uint8)

    def apply(self, image):
        """Apply adjustments to a numpy image (uint8 or float).
        Returns uint8 image.

        For uint8 input: uses a 256-entry LUT for O(256) computation.
        For other dtypes: normalizes to 0-255 uint8 first, then applies LUT.
        """
        if self.is_identity():
            if image.dtype == np.uint8:
                return image
            return np.clip(image, 0, 255).astype(np.uint8)

        lut = self._build_lut()

        if image.dtype == np.uint8:
            return lut[image]

        # Normalize non-uint8 to uint8, then apply LUT
        img = image.astype(np.float32)
        if image.dtype == np.uint16:
            img = img * (255.0 / 65535.0)
        else:
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) * (255.0 / (mx - mn))
        np.clip(img, 0, 255, out=img)
        return lut[img.astype(np.uint8)]

    @staticmethod
    def smart_auto(image):
        """Compute auto-adjustments for an image. Returns ImageAdjustments."""
        adj = ImageAdjustments()
        img = image.astype(np.float32)
        if image.dtype == np.uint8:
            img /= 255.0
        elif image.dtype == np.uint16:
            img /= 65535.0

        # Use percentiles for robust min/max
        p_low = np.percentile(img, 1)
        p_high = np.percentile(img, 99)

        if p_high > p_low:
            # Set contrast to stretch the range
            current_range = p_high - p_low
            adj.contrast = min(3.0, 1.0 / current_range)

            # Set brightness to center the image
            mid = (p_low + p_high) / 2.0
            adj.brightness = 0.5 - mid * adj.contrast

        mean_val = img.mean()
        if mean_val < 0.3:
            adj.gamma = min(2.5, 0.5 / max(mean_val, 0.01))
        elif mean_val > 0.7:
            adj.gamma = max(0.3, 0.5 / max(mean_val, 0.01))

        return adj

    @staticmethod
    def quick_boost(image):
        """Quick brightness/contrast boost. Returns ImageAdjustments."""
        adj = ImageAdjustments()
        adj.contrast = 1.3
        adj.brightness = 0.05
        return adj
