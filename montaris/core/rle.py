import numpy as np


def rle_encode(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    """Encode uint8 mask to RLE bytes. Returns (data, shape)."""
    flat = mask.ravel()
    shape = mask.shape
    if len(flat) == 0:
        return b'', shape
    diffs = np.diff(flat.astype(np.int16))
    change_idx = np.flatnonzero(diffs)
    starts = np.concatenate(([0], change_idx + 1))
    lengths = np.diff(np.concatenate((starts, [len(flat)])))
    values = flat[starts]
    # Pack as value:uint8, length:uint32 pairs
    pairs = np.empty(len(values), dtype=[('v', 'u1'), ('n', '<u4')])
    pairs['v'] = values
    pairs['n'] = lengths
    return pairs.tobytes(), shape


def rle_decode(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Decode RLE bytes back to uint8 mask."""
    if not data:
        return np.zeros(shape, dtype=np.uint8)
    dt = np.dtype([('v', 'u1'), ('n', '<u4')])
    pairs = np.frombuffer(data, dtype=dt)
    return np.repeat(pairs['v'], pairs['n']).reshape(shape)


def rle_decode_crop(data: bytes, shape: tuple[int, int],
                    bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Decode RLE only within bbox (y1, y2, x1, x2). Returns crop array.

    Avoids allocating the full mask — only materialises pixels inside bbox.
    """
    y1, y2, x1, x2 = bbox
    crop_h, crop_w = y2 - y1, x2 - x1
    if not data or crop_h <= 0 or crop_w <= 0:
        return np.zeros((crop_h, crop_w), dtype=np.uint8)

    h, w = shape
    dt = np.dtype([('v', 'u1'), ('n', '<u4')])
    pairs = np.frombuffer(data, dtype=dt)
    values = pairs['v']
    lengths = pairs['n'].astype(np.int64)

    # Cumulative end positions of each run
    ends = np.cumsum(lengths)
    starts = ends - lengths

    # Filter to non-zero runs that overlap the bbox rows
    flat_y1 = np.int64(y1) * w
    flat_y2 = np.int64(y2) * w
    keep = (values > 0) & (ends > flat_y1) & (starts < flat_y2)
    if not keep.any():
        return np.zeros((crop_h, crop_w), dtype=np.uint8)

    v_sel = values[keep]
    s_sel = np.maximum(starts[keep], flat_y1)
    e_sel = np.minimum(ends[keep], flat_y2)

    crop = np.zeros((crop_h, crop_w), dtype=np.uint8)

    for v, s, e in zip(v_sel, s_sel, e_sel):
        pos = np.arange(s, e, dtype=np.int64)
        rows = pos // w
        cols = pos % w
        col_ok = (cols >= x1) & (cols < x2)
        if col_ok.any():
            crop[rows[col_ok] - y1, cols[col_ok] - x1] = v

    return crop
