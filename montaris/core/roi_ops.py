import numpy as np


def fix_overlaps(roi_layers, priority="later_wins"):
    """Fix overlapping pixels between ROI layers.

    Args:
        roi_layers: list of ROILayer objects
        priority: "later_wins" — later layers take precedence
                  "earlier_wins" — earlier layers take precedence

    Returns:
        list of modified ROILayer objects (modified in-place)
    """
    if len(roi_layers) < 2:
        return roi_layers

    # Flatten offsets before overlap operations
    for roi in roi_layers:
        if hasattr(roi, 'flatten_offset'):
            roi.flatten_offset()

    h, w = roi_layers[0].shape

    if priority == "later_wins":
        # Later layers overwrite earlier ones
        # Walk from last to first, track claimed pixels
        claimed = np.zeros((h, w), dtype=bool)
        for roi in reversed(roi_layers):
            current = roi.mask > 0
            # Remove pixels claimed by later layers
            roi.mask[claimed & current] = 0
            roi.invalidate_bbox()
            claimed |= current
    else:
        # Earlier layers keep priority
        claimed = np.zeros((h, w), dtype=bool)
        for roi in roi_layers:
            current = roi.mask > 0
            roi.mask[claimed & current] = 0
            roi.invalidate_bbox()
            claimed |= current

    return roi_layers


def compute_overlap_map(roi_layers):
    """Compute a map of pixel overlap counts.

    Returns:
        2D array where each pixel value = number of ROIs claiming it
    """
    if not roi_layers:
        return np.zeros((1, 1), dtype=np.int32)

    h, w = roi_layers[0].shape
    overlap = np.zeros((h, w), dtype=np.int32)
    for roi in roi_layers:
        overlap += (roi.mask > 0).astype(np.int32)
    return overlap


def auto_fit_rois(roi_layers, image_w, image_h):
    """Shift or resize ROI masks that extend outside the image bounds.

    Returns:
        Number of ROIs that were modified.
    """
    count = 0
    for roi in roi_layers:
        h, w = roi.shape
        if h != image_h or w != image_w:
            # Resize mask to match image dimensions
            from PIL import Image
            img = Image.fromarray(roi.mask)
            img = img.resize((image_w, image_h), Image.NEAREST)
            roi.mask = np.array(img)
            roi.invalidate_bbox()
            count += 1
            continue
        # Check if content is out of bounds (shouldn't happen if sizes match)
        ys, xs = np.where(roi.mask > 0)
        if len(ys) == 0:
            continue
        # Compute shift needed
        dy = 0
        dx = 0
        if ys.min() < 0:
            dy = -ys.min()
        if xs.min() < 0:
            dx = -xs.min()
        if ys.max() >= image_h:
            dy = image_h - 1 - ys.max()
        if xs.max() >= image_w:
            dx = image_w - 1 - xs.max()
        if dy != 0 or dx != 0:
            new_mask = np.zeros_like(roi.mask)
            new_ys = np.clip(ys + dy, 0, image_h - 1)
            new_xs = np.clip(xs + dx, 0, image_w - 1)
            new_mask[new_ys, new_xs] = roi.mask[ys, xs]
            roi.mask = new_mask
            roi.invalidate_bbox()
            count += 1
    return count


def find_overlapping_pairs(roi_layers):
    """Find pairs of ROIs that have overlapping pixels.

    Returns:
        list of (index_a, index_b) tuples
    """
    pairs = []
    for i in range(len(roi_layers)):
        for j in range(i + 1, len(roi_layers)):
            mask_i = roi_layers[i].mask > 0
            mask_j = roi_layers[j].mask > 0
            if np.any(mask_i & mask_j):
                pairs.append((i, j))
    return pairs
