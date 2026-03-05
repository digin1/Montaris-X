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

    h, w = roi_layers[0].mask.shape

    if priority == "later_wins":
        # Later layers overwrite earlier ones
        # Walk from last to first, track claimed pixels
        claimed = np.zeros((h, w), dtype=bool)
        for roi in reversed(roi_layers):
            current = roi.mask > 0
            # Remove pixels claimed by later layers
            roi.mask[claimed & current] = 0
            claimed |= current
    else:
        # Earlier layers keep priority
        claimed = np.zeros((h, w), dtype=bool)
        for roi in roi_layers:
            current = roi.mask > 0
            roi.mask[claimed & current] = 0
            claimed |= current

    return roi_layers


def compute_overlap_map(roi_layers):
    """Compute a map of pixel overlap counts.

    Returns:
        2D array where each pixel value = number of ROIs claiming it
    """
    if not roi_layers:
        return np.zeros((1, 1), dtype=np.int32)

    h, w = roi_layers[0].mask.shape
    overlap = np.zeros((h, w), dtype=np.int32)
    for roi in roi_layers:
        overlap += (roi.mask > 0).astype(np.int32)
    return overlap


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
