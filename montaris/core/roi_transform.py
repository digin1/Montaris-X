import numpy as np
from dataclasses import dataclass


@dataclass
class TransformHandle:
    """A handle point for interactive transformation."""
    x: float
    y: float
    handle_type: str  # 'tl', 'tr', 'bl', 'br', 'tm', 'bm', 'ml', 'mr', 'rotate'


def get_mask_bbox(mask):
    """Get bounding box of non-zero mask pixels. Returns (y1, y2, x1, x2) or None."""
    rows = np.any(mask > 0, axis=1)
    if not np.any(rows):
        return None
    cols = np.any(mask > 0, axis=0)
    y1, y2 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    x1, x2 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    return (y1, y2, x1, x2)


def compute_handles(bbox):
    """Compute 8 scale handles + 1 rotation handle from a bounding box."""
    y1, y2, x1, x2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    handles = [
        TransformHandle(x1, y1, 'tl'),
        TransformHandle(x2, y1, 'tr'),
        TransformHandle(x1, y2, 'bl'),
        TransformHandle(x2, y2, 'br'),
        TransformHandle(cx, y1, 'tm'),
        TransformHandle(cx, y2, 'bm'),
        TransformHandle(x1, cy, 'ml'),
        TransformHandle(x2, cy, 'mr'),
        TransformHandle(cx, y1 - 20, 'rotate'),
    ]
    return handles


def apply_affine_to_mask(mask, matrix, output_shape=None):
    """Apply a 2x3 affine transformation matrix to a binary mask.

    Only processes the bounding box region of the source mask and its
    forward-transformed output region, avoiding full-image coordinate grids.

    Args:
        mask: 2D uint8 array
        matrix: 2x3 affine matrix [[a, b, tx], [c, d, ty]]
        output_shape: (height, width) of output, defaults to same as input

    Returns:
        Transformed mask as uint8 array
    """
    if output_shape is None:
        output_shape = mask.shape

    h_out, w_out = output_shape
    result = np.zeros((h_out, w_out), dtype=np.uint8)
    _apply_affine_bbox(mask, matrix, result, h_out, w_out)
    return result


def apply_affine_inplace(dest, snap, matrix):
    """Transform snap and write result into dest in-place.

    Zeros only the source bbox region and writes to the dest bbox region,
    avoiding full-array allocation or zeroing.
    """
    h, w = dest.shape
    src_bbox = get_mask_bbox(snap)
    if src_bbox is None:
        return
    sy1, sy2, sx1, sx2 = src_bbox
    # Zero only the source region
    dest[sy1:sy2, sx1:sx2] = 0
    _apply_affine_bbox(snap, matrix, dest, h, w, src_bbox)


def _apply_affine_bbox(mask, matrix, result, h_out, w_out, bbox=None):
    """Core bbox-optimized affine transform (writes into result)."""
    if bbox is None:
        bbox = get_mask_bbox(mask)
    if bbox is None:
        return

    sy1, sy2, sx1, sx2 = bbox

    M = np.array([
        [matrix[0, 0], matrix[0, 1], matrix[0, 2]],
        [matrix[1, 0], matrix[1, 1], matrix[1, 2]],
        [0, 0, 1],
    ], dtype=np.float64)

    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return

    # Forward-transform source bbox corners to find output region
    corners = np.array([
        [sx1, sy1, 1], [sx2, sy1, 1],
        [sx1, sy2, 1], [sx2, sy2, 1],
    ], dtype=np.float64).T
    dst_corners = M @ corners
    dx1 = max(0, int(np.floor(dst_corners[0].min())))
    dx2 = min(w_out, int(np.ceil(dst_corners[0].max())))
    dy1 = max(0, int(np.floor(dst_corners[1].min())))
    dy2 = min(h_out, int(np.ceil(dst_corners[1].max())))

    if dx2 <= dx1 or dy2 <= dy1:
        return

    # Create coordinate grid only for the output region
    rh, rw = dy2 - dy1, dx2 - dx1
    yy, xx = np.mgrid[dy1:dy2, dx1:dx2]
    coords = np.stack([xx.ravel(), yy.ravel(), np.ones(rh * rw)], axis=0)

    # Inverse transform to source coordinates
    src = M_inv @ coords
    src_xi = np.round(src[0]).astype(np.int64).reshape(rh, rw)
    src_yi = np.round(src[1]).astype(np.int64).reshape(rh, rw)

    # Valid mask
    h_in, w_in = mask.shape
    valid = (src_xi >= 0) & (src_xi < w_in) & (src_yi >= 0) & (src_yi < h_in)

    result[dy1:dy2, dx1:dx2][valid] = mask[src_yi[valid], src_xi[valid]]


def make_scale_matrix(sx, sy, cx=0, cy=0):
    """Create affine scale matrix centered at (cx, cy)."""
    M = np.array([
        [sx, 0, cx - sx * cx],
        [0, sy, cy - sy * cy],
    ], dtype=np.float64)
    return M


def make_rotation_matrix(angle_rad, cx=0, cy=0):
    """Create affine rotation matrix centered at (cx, cy)."""
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    M = np.array([
        [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy],
        [sin_a, cos_a, cy - sin_a * cx - cos_a * cy],
    ], dtype=np.float64)
    return M


def make_translation_matrix(tx, ty):
    """Create affine translation matrix."""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
    ], dtype=np.float64)
