import os
import numpy as np
from pathlib import Path

# Files larger than this threshold are loaded via memory-mapping when
# the TIFF format supports it.  100 MB.
_MEMMAP_THRESHOLD = 100 * 1024 * 1024


def probe_tiff(path):
    """Inspect a TIFF without loading the full array.

    Returns a dict with keys: is_zstack (bool), shape (tuple), dtype (str),
    axes (str — tifffile series axes, e.g. 'ZYX'), n_slices (int), pages (int).
    Returns None for non-TIFF files.
    """
    p = Path(path)
    if p.suffix.lower() not in ('.tif', '.tiff'):
        return None
    import tifffile
    with tifffile.TiffFile(str(p)) as tf:
        if not tf.series:
            return None
        s = tf.series[0]
        axes = s.axes or ''
        shape = tuple(s.shape)
        dtype = str(s.dtype)
        pages = len(tf.pages)
        # Z-stack detection priority:
        #   1. Explicit Z axis in the series
        #   2. ImageJ metadata `slices > 1`
        #   3. Any non-YX axis that actually iterates (shape > 1) — covers
        #      `tifffile.imwrite(arr_3d)` which comes back as SYX/QYX with no
        #      Z letter but is clearly a stack. Skip when the non-YX length
        #      is one of the RGB(A) sample counts and the image looks like
        #      a single color page (1 page, S axis with 3/4 channels) so we
        #      don't misread a plain RGB TIFF as a 3-slice volume.
        n_slices = 1
        is_zstack = False
        if 'Z' in axes:
            is_zstack = True
            n_slices = shape[axes.index('Z')]
        else:
            md = tf.imagej_metadata or {}
            slices = md.get('slices', 1) or 1
            if slices > 1:
                is_zstack = True
                n_slices = int(slices)
            elif 'Y' in axes and 'X' in axes:
                other = [i for i, a in enumerate(axes) if a not in 'YX']
                extent = int(np.prod([shape[i] for i in other])) if other else 1
                looks_like_rgb = (
                    pages == 1
                    and len(other) == 1
                    and shape[other[0]] in (3, 4)
                    and axes[other[0]] == 'S'
                )
                if extent > 1 and not looks_like_rgb:
                    is_zstack = True
                    n_slices = extent
        return {
            'is_zstack': is_zstack,
            'shape': shape,
            'dtype': dtype,
            'axes': axes,
            'n_slices': n_slices,
            'pages': pages,
        }


def load_volume(path):
    """Load a TIFF as a 3D volume (Z, H, W), squeezing trailing singletons.

    Returns (volume, axes) where axes is the tifffile series axes string.
    Raises ValueError if the file is not a 3D stack.

    For multi-channel / multi-timepoint volumes (e.g. ``CZYX``, ``TZYX``),
    this returns only the first channel/timepoint. Use
    :func:`load_volume_channels` to recover every channel.
    """
    channels = load_volume_channels(path)
    _suffix, vol = channels[0]
    return vol, 'ZYX'


def load_volume_channels(path):
    """Load a TIFF as a list of ``(suffix, ZYX_volume)`` channels.

    Splits any non-ZYX axes (``C``, ``T``, ``S`` used as a slice index,
    etc.) into separate entries so callers don't silently drop channels on
    ``CZYX`` / ``TZYX`` stacks. The suffix is empty for single-channel
    stacks and a short axis-coded tag (e.g. ``"_C0"``) otherwise.

    Raises ValueError if the file isn't a 3D stack.
    """
    import tifffile
    with tifffile.TiffFile(str(path)) as tf:
        s = tf.series[0]
        axes = s.axes or ''
        data = s.asarray()

    if 'Y' not in axes or 'X' not in axes:
        if data.ndim == 3:
            # No letters at all — assume the three axes are ZYX in order.
            return [('', data)]
        raise ValueError(f"Not a 3D stack: shape={data.shape}, axes={axes!r}")

    # If the series has no explicit Z, promote the single non-YX axis to Z.
    # That covers tifffile's default 'SYX'/'QYX' for plain 3D arrays.
    if 'Z' not in axes:
        other = [i for i, a in enumerate(axes) if a not in 'YX']
        if len(other) == 1:
            axes = ''.join('Z' if i == other[0] else a for i, a in enumerate(axes))
        elif data.ndim == 3:
            # Degenerate: YX only (2D image). Can't build a volume.
            raise ValueError(f"Not a 3D stack: shape={data.shape}, axes={axes!r}")

    zi, yi, xi = axes.index('Z'), axes.index('Y'), axes.index('X')
    other_idxs = [i for i in range(data.ndim) if i not in (zi, yi, xi)]
    # Move extra channel-like axes to the front, then ZYX trails.
    perm = other_idxs + [zi, yi, xi]
    data = np.transpose(data, perm)
    other_axes = ''.join(axes[i] for i in other_idxs)
    other_shape = tuple(data.shape[i] for i in range(len(other_idxs)))

    if not other_idxs:
        return [('', data)]
    from itertools import product
    results = []
    for idx in product(*(range(s) for s in other_shape)):
        sub = data[idx]
        suffix = '_' + '_'.join(f"{other_axes[i]}{v}" for i, v in enumerate(idx))
        results.append((suffix, sub))
    return results


def max_projection(volume, axis=0):
    """Max-intensity projection along the given axis (default Z)."""
    return np.max(volume, axis=axis)


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
