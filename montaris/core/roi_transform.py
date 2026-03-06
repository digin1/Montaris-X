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

    Uses Pillow's C-accelerated affine transform on the bbox crop only.

    Returns:
        (src_bbox, dst_bbox) tuple, or (None, None) if no pixels to transform.
    """
    from PIL import Image

    h, w = dest.shape
    src_bbox = get_mask_bbox(snap)
    if src_bbox is None:
        return None, None
    sy1, sy2, sx1, sx2 = src_bbox
    dest[sy1:sy2, sx1:sx2] = 0

    crop = snap[sy1:sy2, sx1:sx2]

    M3 = np.array([
        [matrix[0, 0], matrix[0, 1], matrix[0, 2]],
        [matrix[1, 0], matrix[1, 1], matrix[1, 2]],
        [0, 0, 1],
    ], dtype=np.float64)

    try:
        M_inv = np.linalg.inv(M3)
    except np.linalg.LinAlgError:
        return src_bbox, None

    # Forward-transform source bbox corners to find output region
    corners = np.array([
        [sx1, sy1, 1], [sx2, sy1, 1],
        [sx1, sy2, 1], [sx2, sy2, 1],
    ], dtype=np.float64).T
    dst_corners = M3 @ corners
    ox1 = max(0, int(np.floor(dst_corners[0].min())))
    ox2 = min(w, int(np.ceil(dst_corners[0].max())))
    oy1 = max(0, int(np.floor(dst_corners[1].min())))
    oy2 = min(h, int(np.ceil(dst_corners[1].max())))

    if ox2 <= ox1 or oy2 <= oy1:
        return src_bbox, None

    out_w, out_h = ox2 - ox1, oy2 - oy1
    dst_bbox = (oy1, oy2, ox1, ox2)

    # Pillow inverse affine: map output pixel (ox, oy) → crop pixel (cx, cy)
    pil_data = (
        M_inv[0, 0], M_inv[0, 1],
        M_inv[0, 0] * ox1 + M_inv[0, 1] * oy1 + M_inv[0, 2] - sx1,
        M_inv[1, 0], M_inv[1, 1],
        M_inv[1, 0] * ox1 + M_inv[1, 1] * oy1 + M_inv[1, 2] - sy1,
    )

    pil_img = Image.fromarray(crop)
    result_img = pil_img.transform(
        (out_w, out_h), Image.AFFINE, pil_data, resample=Image.NEAREST,
    )
    result_arr = np.asarray(result_img)

    out_region = dest[oy1:oy2, ox1:ox2]
    nz = result_arr > 0
    out_region[nz] = result_arr[nz]

    return src_bbox, dst_bbox


def _apply_affine_bbox(mask, matrix, result, h_out, w_out, bbox=None):
    """Core bbox-optimized affine transform (writes into result).

    Uses scipy.ndimage.affine_transform for C-accelerated interpolation.
    """
    from scipy.ndimage import affine_transform as scipy_affine

    if bbox is None:
        bbox = get_mask_bbox(mask)
    if bbox is None:
        return

    sy1, sy2, sx1, sx2 = bbox

    # Build 3x3 forward transform matrix
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

    # Use only the bbox crop as scipy input (much smaller than full mask).
    # scipy affine_transform uses (row, col) convention.
    # M_inv maps output (x, y) -> input (x, y).
    # Offset into crop: subtract (sy1, sx1) from full-mask coordinates.
    scipy_matrix = np.array([
        [M_inv[1, 1], M_inv[1, 0]],
        [M_inv[0, 1], M_inv[0, 0]],
    ], dtype=np.float64)
    scipy_offset = np.array([
        M_inv[1, 0] * dx1 + M_inv[1, 1] * dy1 + M_inv[1, 2] - sy1,
        M_inv[0, 0] * dx1 + M_inv[0, 1] * dy1 + M_inv[0, 2] - sx1,
    ], dtype=np.float64)

    crop = np.ascontiguousarray(mask[sy1:sy2, sx1:sx2])
    rh, rw = dy2 - dy1, dx2 - dx1
    region = scipy_affine(
        crop, scipy_matrix, offset=scipy_offset,
        output_shape=(rh, rw), order=0, mode='constant', cval=0,
    )
    # Merge into result (don't overwrite existing non-zero pixels with zero)
    out_region = result[dy1:dy2, dx1:dx2]
    mask_nz = region > 0
    out_region[mask_nz] = region[mask_nz]


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
