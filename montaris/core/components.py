"""Connected component analysis — pure numpy, no scipy dependency."""

import collections
import numpy as np


def get_component_at(mask, x, y):
    """BFS from (x, y) on mask>0 pixels, return boolean mask of the component."""
    h, w = mask.shape
    ix, iy = int(x), int(y)
    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return None
    if mask[iy, ix] == 0:
        return None

    visited = np.zeros((h, w), dtype=bool)
    queue = collections.deque()
    queue.append((ix, iy))
    visited[iy, ix] = True

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and mask[ny, nx] > 0:
                visited[ny, nx] = True
                queue.append((nx, ny))

    return visited


def label_connected_components(mask):
    """Label connected components in a binary mask.

    Returns:
        (label_array, n_components) where label_array has integer labels 1..n
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    binary = mask > 0

    for y in range(h):
        for x in range(w):
            if binary[y, x] and labels[y, x] == 0:
                current_label += 1
                queue = collections.deque()
                queue.append((x, y))
                labels[y, x] = current_label
                while queue:
                    cx, cy = queue.popleft()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < w and 0 <= ny < h
                                and binary[ny, nx] and labels[ny, nx] == 0):
                            labels[ny, nx] = current_label
                            queue.append((nx, ny))

    return labels, current_label


def get_component_bbox(label_array, label_id):
    """Get bounding box for a specific label. Returns (y1, y2, x1, x2) or None."""
    ys, xs = np.where(label_array == label_id)
    if len(ys) == 0:
        return None
    return (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
