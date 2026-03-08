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

        Designed for fluorescence/brightfield brain imaging (Nikon etc.):
        - Finds ALL noise peaks (background + autofluorescence)
        - Sets display_min at the valley after the last noise peak
        - Background AND autofluorescence → black
        - Only true signal (puncta, staining) → visible

        Uses smoothed histogram to find the transition from noise to signal:
        the last local minimum before the signal tail begins.
        """
        adj = ImageAdjustments()
        adj.set_pivot(image)

        # Work in uint8 space for histogram analysis
        if image.dtype == np.uint8:
            img8 = image
        elif image.dtype == np.uint16:
            mn, mx = float(image.min()), float(image.max())
            if mx > mn:
                img8 = ((image.astype(np.float32) - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)
            else:
                img8 = np.zeros_like(image, dtype=np.uint8)
        else:
            img = image.astype(np.float32)
            mn, mx = img.min(), img.max()
            if mx > mn:
                img8 = ((img - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)
            else:
                img8 = np.zeros(image.shape, dtype=np.uint8)

        # Build histogram
        hist = np.bincount(img8.ravel(), minlength=256).astype(np.float64)
        total = img8.size

        # Find the noise/signal boundary using smoothed histogram.
        # For fluorescence microscopy: background (0) + autofluorescence
        # (dim peak on tissue) should both be black. Only actual signal visible.
        #
        # Approach: smooth histogram, find the last significant peak
        # in the lower intensity range, then set display_min at its
        # descending edge (where it drops to 10% of that peak's height).
        kernel = np.ones(3) / 3.0
        smoothed = np.convolve(hist, kernel, mode='same')

        # Find the last significant local maximum in the lower half
        # (background peak, autofluorescence peak, etc.)
        bg_peak = int(np.argmax(hist))
        last_peak_val = bg_peak
        last_peak_height = smoothed[bg_peak]

        # Search for peaks up to the 98th percentile
        p98_bin = 0
        cumsum = 0.0
        for i in range(256):
            cumsum += hist[i]
            if cumsum / total >= 0.98:
                p98_bin = i
                break

        search_limit = min(p98_bin, 128)
        for i in range(bg_peak + 2, search_limit):
            if (smoothed[i] >= smoothed[i - 1] and
                    smoothed[i] >= smoothed[i + 1] and
                    smoothed[i] > total * 0.001):  # peak must have >0.1% of pixels
                last_peak_val = i
                last_peak_height = smoothed[i]

        # Walk right from the last peak until the histogram drops to
        # 10% of that peak's height — that's the noise/signal boundary
        threshold = last_peak_height * 0.10
        noise_end = last_peak_val
        for i in range(last_peak_val + 1, min(last_peak_val + 50, 255)):
            if smoothed[i] < threshold:
                noise_end = i
                break
            noise_end = i

        display_min = noise_end / 255.0

        # display_max = p99 of signal pixels (above the noise floor)
        signal_mask = img8 > noise_end
        if signal_mask.any():
            signal_pixels = img8[signal_mask]
            display_max = float(np.percentile(signal_pixels, 99)) / 255.0
        else:
            display_max = 1.0

        # Ensure minimum window width
        if display_max - display_min < 0.02:
            display_max = min(1.0, display_min + 0.1)

        adj._window_min = display_min
        adj._window_max = display_max

        # Fixed adjustments on top of window/level
        adj.brightness = 0.02
        adj.contrast = -0.02
        adj.exposure = 0.02
        adj.gamma = 3.28

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
