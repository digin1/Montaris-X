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
