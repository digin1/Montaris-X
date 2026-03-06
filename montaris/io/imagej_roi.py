"""Read/write ImageJ binary ROI format.

ImageJ ROI format specification:
- 4-byte magic: "Iout"
- 2-byte version
- 1-byte type (0=polygon, 1=rect, 2=oval, 3=line, 7=freehand, 10=point)
- Coordinates as 16-bit signed integers (or 32-bit float with subpixel flag)
"""

import struct
import io
import numpy as np
from pathlib import Path

# ROI types
ROI_POLYGON = 0
ROI_RECT = 1
ROI_OVAL = 2
ROI_LINE = 3
ROI_FREEHAND = 7
ROI_TRACED = 8
ROI_POINT = 10

MAGIC = b'Iout'
VERSION = 228


def read_imagej_roi(path_or_bytes):
    """Read a single ImageJ .roi file.

    Args:
        path_or_bytes: file path string/Path or bytes

    Returns:
        dict with keys: 'type', 'top', 'left', 'bottom', 'right',
                        'x_coords', 'y_coords', 'name'
    """
    if isinstance(path_or_bytes, (str, Path)):
        with open(path_or_bytes, 'rb') as f:
            data = f.read()
    else:
        data = path_or_bytes

    if len(data) < 64:
        raise ValueError("ROI data too short")

    buf = io.BytesIO(data)

    magic = buf.read(4)
    if magic != MAGIC:
        raise ValueError(f"Invalid ROI magic: {magic}")

    version = struct.unpack('>H', buf.read(2))[0]
    roi_type = struct.unpack('B', buf.read(1))[0]
    buf.read(1)  # padding

    top = struct.unpack('>h', buf.read(2))[0]
    left = struct.unpack('>h', buf.read(2))[0]
    bottom = struct.unpack('>h', buf.read(2))[0]
    right = struct.unpack('>h', buf.read(2))[0]

    n_coords = struct.unpack('>H', buf.read(2))[0]

    # Check for composite ROI (shape_roi_size at offset 36)
    buf.seek(36)
    shape_roi_size = struct.unpack('>i', buf.read(4))[0]
    is_composite = shape_roi_size > 0

    # Skip to offset 64 where coordinates/shape data start
    buf.seek(64)

    x_coords = None
    y_coords = None
    paths = None

    if is_composite:
        # Composite ROI: read float path segments
        shape_array = []
        for _ in range(shape_roi_size):
            val = struct.unpack('>f', buf.read(4))[0]
            shape_array.append(val)

        paths = []
        current_path = None
        i = 0
        while i < len(shape_array):
            seg_type = int(shape_array[i])
            if seg_type == 0:  # SEG_MOVETO
                if current_path is not None:
                    paths.append(current_path)
                current_path = []
                current_path.append((shape_array[i + 1], shape_array[i + 2]))
                i += 3
            elif seg_type == 1:  # SEG_LINETO
                if current_path is not None:
                    current_path.append((shape_array[i + 1], shape_array[i + 2]))
                i += 3
            elif seg_type == 4:  # SEG_CLOSE
                if current_path is not None:
                    paths.append(current_path)
                    current_path = None
                i += 1
            else:
                i += 1
        if current_path is not None:
            paths.append(current_path)

    elif roi_type in (ROI_POLYGON, ROI_FREEHAND, ROI_TRACED) and n_coords > 0:
        x_coords = np.array(
            [struct.unpack('>h', buf.read(2))[0] for _ in range(n_coords)],
            dtype=np.int32
        )
        y_coords = np.array(
            [struct.unpack('>h', buf.read(2))[0] for _ in range(n_coords)],
            dtype=np.int32
        )
        # Coordinates are relative to (left, top)
        x_coords += left
        y_coords += top

    result = {
        'type': roi_type,
        'top': top,
        'left': left,
        'bottom': bottom,
        'right': right,
        'n_coords': n_coords,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'paths': paths,
        'name': '',
    }
    return result


def write_imagej_roi(roi_dict, path):
    """Write a single ImageJ .roi file.

    Args:
        roi_dict: dict with 'type', 'top', 'left', 'bottom', 'right',
                  optionally 'x_coords', 'y_coords'
        path: output file path
    """
    buf = io.BytesIO()

    roi_type = roi_dict.get('type', ROI_RECT)
    top = roi_dict['top']
    left = roi_dict['left']
    bottom = roi_dict['bottom']
    right = roi_dict['right']
    x_coords = roi_dict.get('x_coords')
    y_coords = roi_dict.get('y_coords')
    n_coords = len(x_coords) if x_coords is not None else 0

    # Header
    buf.write(MAGIC)
    buf.write(struct.pack('>H', VERSION))
    buf.write(struct.pack('B', roi_type))
    buf.write(b'\x00')  # padding

    buf.write(struct.pack('>h', top))
    buf.write(struct.pack('>h', left))
    buf.write(struct.pack('>h', bottom))
    buf.write(struct.pack('>h', right))
    buf.write(struct.pack('>H', n_coords))

    # Pad to 64 bytes
    current = buf.tell()
    buf.write(b'\x00' * (64 - current))

    # Coordinates (relative to top-left)
    if roi_type in (ROI_POLYGON, ROI_FREEHAND, ROI_TRACED) and n_coords > 0:
        for x in x_coords:
            buf.write(struct.pack('>h', int(x) - left))
        for y in y_coords:
            buf.write(struct.pack('>h', int(y) - top))

    with open(path, 'wb') as f:
        f.write(buf.getvalue())


def write_imagej_roi_bytes(roi_dict):
    """Write a single ImageJ ROI to bytes (for ZIP export)."""
    buf = io.BytesIO()

    roi_type = roi_dict.get('type', ROI_RECT)
    top = roi_dict['top']
    left = roi_dict['left']
    bottom = roi_dict['bottom']
    right = roi_dict['right']
    x_coords = roi_dict.get('x_coords')
    y_coords = roi_dict.get('y_coords')
    n_coords = len(x_coords) if x_coords is not None else 0

    buf.write(MAGIC)
    buf.write(struct.pack('>H', VERSION))
    buf.write(struct.pack('B', roi_type))
    buf.write(b'\x00')

    buf.write(struct.pack('>h', top))
    buf.write(struct.pack('>h', left))
    buf.write(struct.pack('>h', bottom))
    buf.write(struct.pack('>h', right))
    buf.write(struct.pack('>H', n_coords))

    current = buf.tell()
    buf.write(b'\x00' * (64 - current))

    if roi_type in (ROI_POLYGON, ROI_FREEHAND, ROI_TRACED) and n_coords > 0:
        for x in x_coords:
            buf.write(struct.pack('>h', int(x) - left))
        for y in y_coords:
            buf.write(struct.pack('>h', int(y) - top))

    return buf.getvalue()


def mask_to_imagej_roi(mask, name=""):
    """Convert a binary mask to an ImageJ ROI dict (freehand contour).

    Uses contour tracing to extract the boundary.
    Returns a dict suitable for write_imagej_roi.
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None

    top = int(ys.min())
    bottom = int(ys.max()) + 1
    left = int(xs.min())
    right = int(xs.max()) + 1

    # Simple contour: find boundary pixels
    boundary_x = []
    boundary_y = []
    h, w = mask.shape

    # Edge detection: pixel is boundary if any 4-neighbor is 0 or out-of-bounds
    padded = np.pad(mask, 1, mode='constant', constant_values=0)
    for y in range(top, bottom):
        for x in range(left, right):
            if mask[y, x] > 0:
                py, px = y + 1, x + 1  # padded coords
                if (padded[py - 1, px] == 0 or padded[py + 1, px] == 0 or
                        padded[py, px - 1] == 0 or padded[py, px + 1] == 0):
                    boundary_x.append(x)
                    boundary_y.append(y)

    if not boundary_x:
        return None

    # Sort boundary points by angle from centroid for proper contour ordering
    cx = np.mean(boundary_x)
    cy = np.mean(boundary_y)
    angles = np.arctan2(
        np.array(boundary_y) - cy,
        np.array(boundary_x) - cx
    )
    order = np.argsort(angles)
    boundary_x = np.array(boundary_x)[order]
    boundary_y = np.array(boundary_y)[order]

    return {
        'type': ROI_FREEHAND,
        'top': top,
        'left': left,
        'bottom': bottom,
        'right': right,
        'x_coords': boundary_x.astype(np.int32),
        'y_coords': boundary_y.astype(np.int32),
        'name': name,
    }


def imagej_roi_to_mask(roi_dict, width, height):
    """Convert an ImageJ ROI dict to a binary mask.

    Args:
        roi_dict: dict from read_imagej_roi
        width, height: mask dimensions

    Returns:
        uint8 mask array
    """
    from PIL import Image, ImageDraw

    mask = np.zeros((height, width), dtype=np.uint8)

    # Composite ROI: multiple sub-paths
    paths = roi_dict.get('paths')
    if paths:
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        for path in paths:
            if len(path) >= 3:
                verts = [(int(round(x)), int(round(y))) for x, y in path]
                draw.polygon(verts, fill=255)
        mask = np.asarray(img).copy()
        return mask

    roi_type = roi_dict['type']

    if roi_type == ROI_RECT:
        top = max(0, roi_dict['top'])
        left = max(0, roi_dict['left'])
        bottom = min(height, roi_dict['bottom'])
        right = min(width, roi_dict['right'])
        mask[top:bottom, left:right] = 255

    elif roi_type == ROI_OVAL:
        top = roi_dict['top']
        left = roi_dict['left']
        bottom = roi_dict['bottom']
        right = roi_dict['right']
        cy = (top + bottom) / 2.0
        cx = (left + right) / 2.0
        ry = (bottom - top) / 2.0
        rx = (right - left) / 2.0
        if rx > 0 and ry > 0:
            # Use bbox-only ogrid
            by1 = max(0, int(cy - ry))
            by2 = min(height, int(cy + ry) + 1)
            bx1 = max(0, int(cx - rx))
            bx2 = min(width, int(cx + rx) + 1)
            if by1 < by2 and bx1 < bx2:
                yy, xx = np.ogrid[by1:by2, bx1:bx2]
                ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
                mask[by1:by2, bx1:bx2][ellipse] = 255

    elif roi_type in (ROI_POLYGON, ROI_FREEHAND, ROI_TRACED):
        x_coords = roi_dict.get('x_coords')
        y_coords = roi_dict.get('y_coords')
        if x_coords is not None and y_coords is not None and len(x_coords) >= 3:
            verts = list(zip(x_coords.tolist(), y_coords.tolist()))
            img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img).polygon(verts, fill=255)
            mask = np.asarray(img).copy()

    return mask


