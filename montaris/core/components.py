"""Connected component analysis using scipy.ndimage."""

import numpy as np
from scipy import ndimage


def get_component_at(mask, x, y):
    """Return boolean mask of the connected component at (x, y)."""
    h, w = mask.shape
    ix, iy = int(x), int(y)
    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return None
    if mask[iy, ix] == 0:
        return None

    labels, _ = ndimage.label(mask > 0)
    target = labels[iy, ix]
    if target == 0:
        return None
    return labels == target


def label_connected_components(mask):
    """Label connected components in a binary mask.

    Returns:
        (label_array, n_components) where label_array has integer labels 1..n
    """
    return ndimage.label(mask > 0)


def get_component_bbox(label_array, label_id):
    """Get bounding box for a specific label. Returns (y1, y2, x1, x2) or None."""
    ys, xs = np.where(label_array == label_id)
    if len(ys) == 0:
        return None
    return (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
