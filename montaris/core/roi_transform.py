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
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)


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

    # Build inverse transform for backward mapping
    # Forward: [x', y'] = M @ [x, y, 1]
    # We need inverse: given (x', y'), find (x, y)
    M = np.array([
        [matrix[0, 0], matrix[0, 1], matrix[0, 2]],
        [matrix[1, 0], matrix[1, 1], matrix[1, 2]],
        [0, 0, 1],
    ], dtype=np.float64)

    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return result

    # Create coordinate grid for output
    yy, xx = np.mgrid[0:h_out, 0:w_out]
    coords = np.stack([xx.ravel(), yy.ravel(), np.ones(h_out * w_out)], axis=0)

    # Transform back to source
    src = M_inv @ coords
    src_x = src[0].reshape(h_out, w_out)
    src_y = src[1].reshape(h_out, w_out)

    # Round to nearest neighbor
    src_xi = np.round(src_x).astype(np.int64)
    src_yi = np.round(src_y).astype(np.int64)

    # Valid mask
    h_in, w_in = mask.shape
    valid = (src_xi >= 0) & (src_xi < w_in) & (src_yi >= 0) & (src_yi < h_in)

    result[valid] = mask[src_yi[valid], src_xi[valid]]
    return result


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
