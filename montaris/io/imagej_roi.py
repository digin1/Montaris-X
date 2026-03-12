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
ROI_COMPOSITE = 9
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


def _build_roi_bytes(roi_dict):
    """Build binary ROI data from a roi_dict. Supports both simple and composite ROIs."""
    buf = io.BytesIO()

    roi_type = roi_dict.get('type', ROI_RECT)
    top = roi_dict['top']
    left = roi_dict['left']
    bottom = roi_dict['bottom']
    right = roi_dict['right']
    paths = roi_dict.get('paths')
    x_coords = roi_dict.get('x_coords')
    y_coords = roi_dict.get('y_coords')
    n_coords = len(x_coords) if x_coords is not None else 0

    # Build shape array for composite ROIs
    shape_floats = []
    if paths:
        for path in paths:
            if len(path) < 3:
                continue
            shape_floats.append(0.0)  # SEG_MOVETO
            shape_floats.append(float(path[0][0]))
            shape_floats.append(float(path[0][1]))
            for x, y in path[1:]:
                shape_floats.append(1.0)  # SEG_LINETO
                shape_floats.append(float(x))
                shape_floats.append(float(y))
            shape_floats.append(4.0)  # SEG_CLOSE

    shape_roi_size = len(shape_floats)
    # ImageJ requires type=rect for composite (shape) ROIs
    wire_type = ROI_RECT if shape_roi_size > 0 else roi_type

    # Header (offset 0)
    buf.write(MAGIC)
    buf.write(struct.pack('>H', VERSION))
    buf.write(struct.pack('B', wire_type))
    buf.write(b'\x00')  # padding

    # Bounding box + n_coords (offset 8)
    buf.write(struct.pack('>h', top))
    buf.write(struct.pack('>h', left))
    buf.write(struct.pack('>h', bottom))
    buf.write(struct.pack('>h', right))
    buf.write(struct.pack('>H', n_coords))

    # Pad from current position to offset 36
    current = buf.tell()  # should be 18
    buf.write(b'\x00' * (36 - current))

    # shape_roi_size at offset 36
    buf.write(struct.pack('>i', shape_roi_size))

    # Pad to 64 bytes
    current = buf.tell()  # should be 40
    buf.write(b'\x00' * (64 - current))

    # Data section (offset 64)
    if shape_roi_size > 0:
        for val in shape_floats:
            buf.write(struct.pack('>f', val))
    elif roi_type in (ROI_POLYGON, ROI_FREEHAND, ROI_TRACED) and n_coords > 0:
        for x in x_coords:
            buf.write(struct.pack('>h', int(x) - left))
        for y in y_coords:
            buf.write(struct.pack('>h', int(y) - top))

    return buf.getvalue()


def write_imagej_roi(roi_dict, path):
    """Write a single ImageJ .roi file."""
    with open(path, 'wb') as f:
        f.write(_build_roi_bytes(roi_dict))


def write_imagej_roi_bytes(roi_dict):
    """Write a single ImageJ ROI to bytes (for ZIP export)."""
    return _build_roi_bytes(roi_dict)


def mask_to_imagej_roi(mask, name="", bbox=None):
    """Convert a binary mask to an ImageJ ROI dict.

    Uses skimage.measure.find_contours for proper contour tracing.
    Single contour → freehand ROI with x/y coords.
    Multiple contours (holes) → composite ROI with paths.

    Args:
        mask: 2D uint8 array
        name: ROI name string
        bbox: optional (top, bottom, left, right) tuple to skip full-mask scan
    """
    from skimage.measure import find_contours

    if bbox is not None:
        top, bottom, left, right = bbox
    else:
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        top = int(ys.min())
        bottom = int(ys.max()) + 1
        left = int(xs.min())
        right = int(xs.max()) + 1

    submask = mask[top:bottom, left:right]
    if not submask.any():
        return None

    # Pad with zeros so find_contours detects edges at submask boundary
    padded = np.pad(submask, 1, mode='constant', constant_values=0)
    contours = find_contours(padded, 0.5)
    if not contours:
        return None

    if len(contours) == 1:
        # Single contour → simple freehand ROI with integer coords
        c = contours[0]
        if np.allclose(c[0], c[-1]) and len(c) > 1:
            c = c[:-1]
        # find_contours returns (row, col) = (y, x)
        # Subtract 1 for padding, then offset back to full-mask space
        x_coords = np.round(c[:, 1] - 1).astype(np.int32) + left
        y_coords = np.round(c[:, 0] - 1).astype(np.int32) + top
        return {
            'type': ROI_FREEHAND,
            'top': top,
            'left': left,
            'bottom': bottom,
            'right': right,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'paths': None,
            'name': name,
        }
    else:
        # Multiple contours → composite ROI (written as rect+shape data) with float paths
        paths = []
        for c in contours:
            if np.allclose(c[0], c[-1]) and len(c) > 1:
                c = c[:-1]
            # Subtract 1 for padding, then offset back to full-mask space
            path = [(float(pt[1]) - 1 + left, float(pt[0]) - 1 + top) for pt in c]
            paths.append(path)
        return {
            'type': ROI_RECT,
            'top': top,
            'left': left,
            'bottom': bottom,
            'right': right,
            'x_coords': None,
            'y_coords': None,
            'paths': paths,
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
    mask = np.zeros((height, width), dtype=np.uint8)

    # Composite ROI: multiple sub-paths with even-odd fill (XOR)
    paths = roi_dict.get('paths')
    if paths:
        from PIL import Image, ImageDraw
        for path in paths:
            if len(path) >= 3:
                xs = [p[0] for p in path]
                ys = [p[1] for p in path]
                x0 = max(0, int(min(xs)))
                y0 = max(0, int(min(ys)))
                x1 = min(width, int(max(xs)) + 2)
                y1 = min(height, int(max(ys)) + 2)
                bw, bh = x1 - x0, y1 - y0
                xy = [(round(x) - x0, round(y) - y0) for x, y in path]
                sub_img = Image.new('L', (bw, bh), 0)
                draw = ImageDraw.Draw(sub_img)
                draw.polygon(xy, fill=255)
                sub = np.array(sub_img)
                mask[y0:y1, x0:x1] ^= sub
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
            from PIL import Image, ImageDraw
            x0 = max(0, int(x_coords.min()))
            y0 = max(0, int(y_coords.min()))
            x1 = min(width, int(x_coords.max()) + 2)
            y1 = min(height, int(y_coords.max()) + 2)
            bw, bh = x1 - x0, y1 - y0
            xy = list(zip((x_coords - x0).tolist(), (y_coords - y0).tolist()))
            img = Image.new('L', (bw, bh), 0)
            draw = ImageDraw.Draw(img)
            draw.polygon(xy, fill=255)
            mask[y0:y1, x0:x1] = np.array(img)

    return mask


def scale_roi_dict(roi_dict, sx, sy):
    """Scale all coordinates in an ImageJ ROI dict. Returns a new dict."""
    d = dict(roi_dict)
    d['top'] = int(d['top'] * sy)
    d['bottom'] = int(d['bottom'] * sy)
    d['left'] = int(d['left'] * sx)
    d['right'] = int(d['right'] * sx)
    if d.get('x_coords') is not None:
        d['x_coords'] = (d['x_coords'] * sx).astype(np.int32)
        d['y_coords'] = (d['y_coords'] * sy).astype(np.int32)
    if d.get('paths'):
        d['paths'] = [[(x * sx, y * sy) for x, y in p] for p in d['paths']]
    return d


