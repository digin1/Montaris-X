import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ImageAdjustments:
    """Image adjustment parameters.

    Uses a data-adaptive window/level model inspired by ImageJ:
    - The contrast pivot is set to the image's actual mean intensity
      (not a fixed 128), so contrast changes stretch the actual data range
    - Brightness shifts the output proportionally (GIMP model)
    - Exposure multiplies input (photography standard)
    - Gamma applies power curve

    This works well for scientific images with arbitrary intensity ranges
    (dark, narrow-range, 16-bit, etc.)
    """
    brightness: float = 0.0     # -1.0 to 1.0
    contrast: float = 0.0       # -1.0 to 1.0 (0 = identity)
    exposure: float = 0.0       # -2.0 to 2.0
    gamma: float = 1.0          # 0.1 to 5.0
    _pivot: float = 0.5         # data-adaptive contrast pivot (0-1 normalized)

    def is_identity(self):
        return (self.brightness == 0.0 and self.contrast == 0.0
                and self.exposure == 0.0 and self.gamma == 1.0)

    def reset(self):
        self.brightness = 0.0
        self.contrast = 0.0
        self.exposure = 0.0
        self.gamma = 1.0

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
            vals /= 65535.0
        else:
            mn, mx = vals.min(), vals.max()
            if mx > mn:
                vals = (vals - mn) / (mx - mn)
        self._pivot = float(np.clip(vals.mean(), 0.01, 0.99))

    def _build_lut(self):
        """Build a 256-entry uint8 lookup table.

        Contrast pivots around the actual image center (_pivot),
        not a fixed 0.5, so it works correctly on dark/bright images.

        Uses GIMP-style proportional brightness and exp() contrast slope.
        """
        x = np.arange(256, dtype=np.float64) / 255.0

        # Exposure: scale input (photography standard: multiply by 2^exposure)
        if self.exposure != 0.0:
            x = x * (2.0 ** self.exposure)

        # Brightness — GIMP proportional model
        b = self.brightness / 2.0
        if b < 0.0:
            x = x * (1.0 + b)
        elif b > 0.0:
            x = x + (1.0 - x) * b

        # Contrast — pivot around actual image center, exp() slope
        # exp() is smooth, bounded, and gives gentle response near 0
        # At contrast=0: slope=1 (identity)
        # At contrast=1: slope=exp(2)≈7.4 (strong but not infinite)
        # At contrast=-1: slope=exp(-2)≈0.14 (flatten)
        if self.contrast != 0.0:
            slope = math.exp(self.contrast * 2.0)
            pivot = self._pivot
            x = (x - pivot) * slope + pivot

        # Gamma
        if self.gamma != 1.0:
            np.clip(x, 0, 1, out=x)
            np.power(x, 1.0 / self.gamma, out=x)

        np.clip(x * 255, 0, 255, out=x)
        return x.astype(np.uint8)

    def apply(self, image):
        """Apply adjustments to a numpy image (uint8 or float).
        Returns uint8 image.

        For uint8 input: uses a 256-entry LUT for O(256) computation.
        For other dtypes: normalizes to uint8 first, then applies LUT.
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
        adj.set_pivot(image)
        img = image.astype(np.float32)
        if image.dtype == np.uint8:
            img /= 255.0
        elif image.dtype == np.uint16:
            img /= 65535.0

        p_low = np.percentile(img, 1)
        p_high = np.percentile(img, 99)

        if p_high > p_low:
            current_range = p_high - p_low
            # Map desired contrast to exp() model: slope = 1/range = exp(2c)
            # c = ln(1/range) / 2
            adj.contrast = np.clip(math.log(1.0 / max(current_range, 0.01)) / 2.0, -1.0, 1.0)

            mid = (p_low + p_high) / 2.0
            adj.brightness = np.clip((0.5 - mid) * 2.0, -1.0, 1.0)

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
        adj.set_pivot(image)
        adj.contrast = 0.15
        adj.brightness = 0.1
        return adj
