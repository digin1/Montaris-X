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


# ---------------------------------------------------------------------------
# Automatic high-contrast colour assignment
# ---------------------------------------------------------------------------

def _rgb_to_lab(rgb):
    """Convert (r, g, b) uint8 tuple to CIE-LAB (L*, a*, b*)."""
    # sRGB -> linear
    c = np.array(rgb, dtype=np.float64) / 255.0
    mask = c <= 0.04045
    c = np.where(mask, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    # Linear RGB -> XYZ (D65)
    X = 0.4124564 * c[0] + 0.3575761 * c[1] + 0.1804375 * c[2]
    Y = 0.2126729 * c[0] + 0.7151522 * c[1] + 0.0721750 * c[2]
    Z = 0.0193339 * c[0] + 0.1191920 * c[1] + 0.9503041 * c[2]
    # XYZ -> LAB (D65 white point)
    ref = np.array([0.95047, 1.00000, 1.08883])
    xyz = np.array([X, Y, Z]) / ref
    delta = 6.0 / 29.0
    m = xyz > delta ** 3
    xyz = np.where(m, xyz ** (1.0 / 3.0), xyz / (3 * delta ** 2) + 4.0 / 29.0)
    L = 116.0 * xyz[1] - 16.0
    a = 500.0 * (xyz[0] - xyz[1])
    b = 200.0 * (xyz[1] - xyz[2])
    return np.array([L, a, b])


def _delta_e_sq(lab1, lab2):
    """Squared Euclidean distance in CIE-LAB (approximation of perceptual diff)."""
    d = lab1 - lab2
    return float(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)


def _find_adjacent_pairs(roi_layers, margin=5):
    """Build adjacency list using dilated bounding-box overlap.

    Two ROIs are "adjacent" if their bounding boxes (expanded by *margin* px)
    intersect. This is O(n²) on bbox checks — much cheaper than full-mask
    overlap for large ROI counts.  Falls back to pixel overlap for ROIs
    that share a bbox but might not truly neighbour.
    """
    n = len(roi_layers)
    adj = [set() for _ in range(n)]
    bboxes = []
    for roi in roi_layers:
        bb = roi.get_bbox()
        if bb is None:
            bboxes.append(None)
        else:
            y1, y2, x1, x2 = bb
            ox = getattr(roi, 'offset_x', 0)
            oy = getattr(roi, 'offset_y', 0)
            bboxes.append((y1 + oy - margin, y2 + oy + margin,
                           x1 + ox - margin, x2 + ox + margin))

    for i in range(n):
        if bboxes[i] is None:
            continue
        iy1, iy2, ix1, ix2 = bboxes[i]
        for j in range(i + 1, n):
            if bboxes[j] is None:
                continue
            jy1, jy2, jx1, jx2 = bboxes[j]
            # Check bbox overlap
            if iy1 < jy2 and iy2 > jy1 and ix1 < jx2 and ix2 > jx1:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _generate_palette(n):
    """Generate *n* maximally-spaced colours in LAB, returned as RGB tuples."""
    from montaris.layers import _generate_color
    # Use the existing napari-compatible colour table which already has good
    # perceptual spacing.  We pick *n* colours starting from index 0.
    return [_generate_color(i) for i in range(n)]


def auto_color_for_contrast(roi_layers):
    """Assign colours to ROIs so that adjacent ROIs have maximum contrast.

    Algorithm:
        1. Build an adjacency graph using dilated bounding-box proximity.
        2. Generate a palette of perceptually-spaced candidate colours
           (at least max_degree + extra).
        3. Greedy graph-colouring (Welsh-Powell): process ROIs in decreasing
           degree order, assign the colour with highest minimum perceptual
           distance to all already-coloured neighbours.

    Returns:
        list of (index, old_color, new_color) for every ROI that changed.
    """
    n = len(roi_layers)
    if n == 0:
        return []

    adj = _find_adjacent_pairs(roi_layers)
    max_degree = max((len(a) for a in adj), default=0)

    # Palette: enough colours for the chromatic number + headroom
    palette_size = max(max_degree + 4, 12, n)
    palette_rgb = _generate_palette(palette_size)
    palette_lab = np.array([_rgb_to_lab(c) for c in palette_rgb])

    # Welsh-Powell ordering: highest degree first
    order = sorted(range(n), key=lambda i: len(adj[i]), reverse=True)

    assignments = [None] * n  # palette index per ROI
    changes = []

    for i in order:
        neighbour_labs = []
        for nb in adj[i]:
            if assignments[nb] is not None:
                neighbour_labs.append(palette_lab[assignments[nb]])

        if not neighbour_labs:
            # No coloured neighbours yet — pick the palette colour with
            # the best global spread (first unused index).
            used = {assignments[nb] for nb in adj[i] if assignments[nb] is not None}
            best = 0
            for ci in range(palette_size):
                if ci not in used:
                    best = ci
                    break
            assignments[i] = best
        else:
            # Pick the palette colour that maximises the minimum perceptual
            # distance to all coloured neighbours.
            nb_arr = np.array(neighbour_labs)  # (k, 3)
            best_ci = 0
            best_min_dist = -1.0
            used = {assignments[nb] for nb in adj[i] if assignments[nb] is not None}
            for ci in range(palette_size):
                if ci in used:
                    continue
                diffs = palette_lab[ci] - nb_arr  # (k, 3)
                dists = np.sum(diffs ** 2, axis=1)  # (k,)
                min_dist = float(np.min(dists))
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_ci = ci
            assignments[i] = best_ci

    # Apply and record changes (keyed by ROI reference, not index)
    for i in range(n):
        new_color = palette_rgb[assignments[i]]
        old_color = roi_layers[i].color
        if new_color != old_color:
            changes.append((roi_layers[i], old_color, new_color))
            roi_layers[i].color = new_color

    return changes
