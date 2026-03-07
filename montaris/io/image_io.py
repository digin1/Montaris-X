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
        return _pil_to_array(img)


def _pil_to_array(img):
    """Convert a PIL Image to a well-typed numpy array.

    Handles palette, 1-bit, and CMYK modes by converting to RGB/L first.
    """
    if img.mode in ('1',):
        img = img.convert('L')
    elif img.mode in ('P', 'PA'):
        img = img.convert('RGBA' if 'A' in img.mode or 'transparency' in img.info else 'RGB')
    elif img.mode in ('CMYK', 'YCbCr', 'LAB', 'I', 'F'):
        img = img.convert('RGB')
    return np.array(img)


def load_image_stack(path):
    """Load an image file, returning a list of (name, array) for each channel/page.

    For multi-channel/page TIFFs, each channel becomes a separate entry.
    For standard images (PNG, JPG, single-page TIFF), returns a single entry.
    """
    path = Path(path)
    stem = path.stem
    suffix = path.suffix.lower()

    if suffix in ('.tif', '.tiff'):
        import tifffile
        data = tifffile.imread(str(path))
        return _split_tiff_channels(data, stem)
    else:
        from PIL import Image
        img = Image.open(str(path))
        return [(stem, _pil_to_array(img))]


def _split_tiff_channels(data, stem):
    """Split a TIFF array into individual 2D images.

    Returns list of (name, 2d_or_rgb_array) tuples.
    """
    # 2D grayscale
    if data.ndim == 2:
        return [(stem, data)]

    # 3D: could be (H, W, C_rgb), (C, H, W), or (N_pages, H, W)
    if data.ndim == 3:
        # Channels-first RGB/RGBA: small first dim that looks like color
        if data.shape[0] in (3, 4) and data.shape[0] < data.shape[1]:
            return [(stem, np.moveaxis(data, 0, -1))]
        # Channels-last RGB/RGBA
        if data.shape[2] in (3, 4):
            return [(stem, data)]
        # Single channel channels-first
        if data.shape[0] == 1:
            return [(stem, data[0])]
        # Multi-channel/page: split along first axis
        return [(f"{stem}_ch{i}", data[i]) for i in range(data.shape[0])]

    # 4D+: e.g. (N, C, H, W) or (T, Z, H, W)
    # Flatten to list of 2D/RGB images
    if data.ndim == 4:
        results = []
        for i in range(data.shape[0]):
            sub = data[i]
            # Each sub is 3D — recurse
            results.extend(_split_tiff_channels(sub, f"{stem}_p{i}"))
        return results

    # 5D+: take first slice and recurse
    if data.ndim > 4:
        return _split_tiff_channels(data[0], stem)

    return [(stem, data)]


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
