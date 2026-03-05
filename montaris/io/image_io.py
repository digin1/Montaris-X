import os
import numpy as np
from pathlib import Path

# Files larger than this threshold are loaded via memory-mapping when
# the TIFF format supports it.  100 MB.
_MEMMAP_THRESHOLD = 100 * 1024 * 1024


def load_image(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in ('.tif', '.tiff'):
        import tifffile

        file_size = os.path.getsize(path)

        if file_size > _MEMMAP_THRESHOLD:
            # Try memory-mapped loading for large files
            try:
                data = tifffile.memmap(str(path))
                data = _normalise_tiff(data)
                return data
            except Exception:
                pass  # Fall through to regular imread

        data = tifffile.imread(str(path))
        data = _normalise_tiff(data)
        return data
    else:
        from PIL import Image
        img = Image.open(str(path))
        return np.array(img)


def _normalise_tiff(data):
    """Apply common TIFF normalisations (multi-page, channel order, squeeze)."""
    # Multi-page TIFF: take first page
    if data.ndim > 3:
        data = data[0]
    # Channels-first to channels-last
    if data.ndim == 3 and data.shape[0] in (1, 3, 4) and data.shape[0] < data.shape[1]:
        data = np.moveaxis(data, 0, -1)
    # Squeeze single channel
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    return data
