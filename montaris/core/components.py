"""Connected component analysis using scipy.ndimage."""

import numpy as np
from scipy import ndimage


def get_component_at(mask, x, y, bbox=None):
    """Return boolean mask of the connected component at (x, y).

    Uses binary_propagation from the seed point — only visits the target
    component, not the entire mask.

    If *bbox* is provided as (y1, y2, x1, x2), operates on the cropped
    region and embeds the result back into a full-size mask. This avoids
    allocating/propagating over the entire image when the ROI is small.
    """
    h, w = mask.shape
    ix, iy = int(x), int(y)
    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return None
    if mask[iy, ix] == 0:
        return None

    if bbox is not None:
        by1, by2, bx1, bx2 = bbox
        # Ensure click is within bbox (should always be true)
        if by1 <= iy < by2 and bx1 <= ix < bx2:
            crop = mask[by1:by2, bx1:bx2]
            ly, lx = iy - by1, ix - bx1
            seed_crop = np.zeros(crop.shape, dtype=bool)
            seed_crop[ly, lx] = True
            comp_crop = ndimage.binary_propagation(seed_crop, mask=(crop > 0))
            # Embed back into full-size mask
            result = np.zeros((h, w), dtype=bool)
            result[by1:by2, bx1:bx2] = comp_crop
            return result

    seed = np.zeros((h, w), dtype=bool)
    seed[iy, ix] = True
    return ndimage.binary_propagation(seed, mask=(mask > 0))


def label_connected_components(mask, structure=None) -> tuple[np.ndarray, int]:
    """Label connected components in a binary mask.

    ``structure`` controls connectivity. ``None`` (default) is scipy's
    cross structure → 4-connected (orthogonal neighbours only). Pass
    ``np.ones((3, 3), bool)`` for 8-connected (orthogonal + diagonal).
    The 2D bucket-fill tool uses 8-connected so a diagonal chain of
    same-value pixels counts as one component, matching the prior
    ``binary_propagation`` behaviour and napari's labels-layer fill.

    Skips the ``mask > 0`` materialisation when the caller already
    passes a bool array — saves a 71 MB allocation on the user's
    71 M-pixel masks (Codex review M1).

    Returns:
        (label_array, n_components) where label_array has integer
        labels 1..n.
    """
    if mask.dtype != np.bool_:
        mask = mask > 0
    result = ndimage.label(mask, structure=structure)
    return result[0], int(result[1])  # type: ignore[index]


def get_component_bbox(label_array, label_id):
    """Get bounding box for a specific label. Returns (y1, y2, x1, x2) or None."""
    ys, xs = np.where(label_array == label_id)
    if len(ys) == 0:
        return None
    return (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
