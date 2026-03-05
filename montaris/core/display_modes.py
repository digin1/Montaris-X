import numpy as np
from enum import Enum


class DisplayMode(Enum):
    COMPOSITE_RGB = "composite_rgb"
    FALSE_COLOR = "false_color"
    MAX_PROJECTION = "max_projection"
    FIRST_CHANNEL = "first_channel"
    GRAYSCALE = "grayscale"


# Default false-color LUTs: Red, Green, Blue, Cyan, Magenta, Yellow
FALSE_COLOR_LUTS = [
    np.array([1.0, 0.0, 0.0]),  # Red
    np.array([0.0, 1.0, 0.0]),  # Green
    np.array([0.0, 0.0, 1.0]),  # Blue
    np.array([0.0, 1.0, 1.0]),  # Cyan
    np.array([1.0, 0.0, 1.0]),  # Magenta
    np.array([1.0, 1.0, 0.0]),  # Yellow
]


class DisplayCompositor:
    """Composites multiple image channels into a display image."""

    def __init__(self):
        self.mode = DisplayMode.COMPOSITE_RGB
        self.channel_luts = {}  # channel_index -> np.array([r, g, b]) 0-1

    def set_lut(self, channel_idx, rgb):
        self.channel_luts[channel_idx] = np.array(rgb, dtype=np.float32)

    def compose(self, channels, mode=None):
        """Compose list of 2D numpy arrays into an RGB uint8 image.

        Args:
            channels: list of 2D numpy arrays (grayscale channel data)
            mode: override display mode, or use self.mode

        Returns:
            np.ndarray of shape (H, W, 3) dtype uint8
        """
        if not channels:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        mode = mode or self.mode
        h, w = channels[0].shape[:2]

        if mode == DisplayMode.FIRST_CHANNEL:
            return self._to_rgb(channels[0])

        if mode == DisplayMode.GRAYSCALE:
            if len(channels) == 1:
                return self._to_rgb(channels[0])
            # Average all channels
            stacked = np.stack([self._normalize(c) for c in channels], axis=0)
            avg = stacked.mean(axis=0)
            return self._to_rgb_from_float(avg)

        if mode == DisplayMode.MAX_PROJECTION:
            stacked = np.stack([self._normalize(c) for c in channels], axis=0)
            projected = stacked.max(axis=0)
            return self._to_rgb_from_float(projected)

        if mode == DisplayMode.COMPOSITE_RGB:
            result = np.zeros((h, w, 3), dtype=np.float32)
            for i, ch in enumerate(channels):
                norm = self._normalize(ch)
                if i < 3:
                    result[:, :, i] = norm
                else:
                    # Extra channels added to all
                    result += norm[:, :, np.newaxis] / 3.0
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            return result

        if mode == DisplayMode.FALSE_COLOR:
            result = np.zeros((h, w, 3), dtype=np.float32)
            for i, ch in enumerate(channels):
                norm = self._normalize(ch)
                lut = self.channel_luts.get(i, FALSE_COLOR_LUTS[i % len(FALSE_COLOR_LUTS)])
                result += norm[:, :, np.newaxis] * lut[np.newaxis, np.newaxis, :]
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            return result

        return self._to_rgb(channels[0])

    def _normalize(self, data):
        """Normalize data to 0-1 float based on dtype range."""
        if data.dtype == np.uint8:
            return data.astype(np.float32) / 255.0
        elif data.dtype == np.uint16:
            return data.astype(np.float32) / 65535.0
        else:
            data = data.astype(np.float32)
            mn, mx = data.min(), data.max()
            if mx > mn:
                return (data - mn) / (mx - mn)
            if mx > 0:
                return np.full_like(data, 0.5)
            return np.zeros_like(data)

    def _to_rgb(self, data):
        """Convert single channel to RGB uint8."""
        if data.dtype != np.uint8:
            data = (self._normalize(data) * 255).astype(np.uint8)
        h, w = data.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = data
        rgb[:, :, 1] = data
        rgb[:, :, 2] = data
        return rgb

    def _to_rgb_from_float(self, data):
        """Convert 0-1 float to RGB uint8."""
        gray = np.clip(data * 255, 0, 255).astype(np.uint8)
        h, w = gray.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = gray
        rgb[:, :, 1] = gray
        rgb[:, :, 2] = gray
        return rgb
