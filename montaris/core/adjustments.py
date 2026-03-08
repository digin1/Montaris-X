import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ImageAdjustments:
    """Image adjustment parameters.

    Brightness/contrast follow GIMP's model (gegl brightness-contrast):
    - Brightness is proportional (not a flat offset) for natural feel
    - Contrast uses tan() curve: gentle near center, strong at extremes
    """
    brightness: float = 0.0     # -1.0 to 1.0
    contrast: float = 0.0       # -1.0 to 1.0 (0 = identity)
    exposure: float = 0.0       # -2.0 to 2.0
    gamma: float = 1.0          # 0.1 to 5.0

    def is_identity(self):
        return (self.brightness == 0.0 and self.contrast == 0.0
                and self.exposure == 0.0 and self.gamma == 1.0)

    def reset(self):
        self.brightness = 0.0
        self.contrast = 0.0
        self.exposure = 0.0
        self.gamma = 1.0

    def _build_lut(self):
        """Build a 256-entry uint8 lookup table using GIMP's formulas.

        Brightness: proportional (GIMP gimp_operation_brightness_contrast_map)
        Contrast: tan((contrast+1) * pi/4) slope (GIMP slant formula)
        """
        x = np.arange(256, dtype=np.float64) / 255.0

        # Exposure: multiply by 2^exposure (photography standard)
        if self.exposure != 0.0:
            x = x * (2.0 ** self.exposure)

        # Brightness — GIMP model (proportional, not flat offset)
        # GIMP halves the UI value internally: brightness = config / 2.0
        b = self.brightness / 2.0
        if b < 0.0:
            x = x * (1.0 + b)
        elif b > 0.0:
            x = x + (1.0 - x) * b

        # Contrast — GIMP tan() curve
        # slant = tan((contrast + 1) * pi/4)
        # At contrast=0: slant=tan(pi/4)=1.0 (identity)
        # Clamp to avoid tan(pi/2) singularity
        c = np.clip(self.contrast, -1.0, 0.9999)
        slant = math.tan((c + 1.0) * math.pi / 4.0)
        x = (x - 0.5) * slant + 0.5

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
        img = image.astype(np.float32)
        if image.dtype == np.uint8:
            img /= 255.0
        elif image.dtype == np.uint16:
            img /= 65535.0

        # Use percentiles for robust min/max
        p_low = np.percentile(img, 1)
        p_high = np.percentile(img, 99)

        if p_high > p_low:
            current_range = p_high - p_low
            # Map desired contrast stretch to GIMP's tan-based contrast
            # We want slant = 1/current_range, so contrast = 4/pi * atan(slant) - 1
            desired_slant = min(4.0, 1.0 / current_range)
            adj.contrast = np.clip(4.0 / math.pi * math.atan(desired_slant) - 1.0, -1.0, 0.99)

            # Brightness: shift midpoint toward 0.5
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
        adj.contrast = 0.15
        adj.brightness = 0.1
        return adj
