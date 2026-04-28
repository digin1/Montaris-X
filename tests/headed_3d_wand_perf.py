"""Headed measurement of the 3D wand path on the GQR183 file (851 MB
intensity, 348M voxels). Breaks the wand into discrete steps and times
each so we can identify the bottleneck.

The wand currently does:
  1. ``self._active_channel_volume()``         — return ref to vol (no copy)
  2. ``int(vol[seed_tuple])``                  — single voxel
  3. ``np.minimum(vol, seed_val)``             — full 851 MB uint16 COPY
  4. ``skimage.flood(clipped, seed, tol)``     — flood-fill, allocates mask
  5. ``mask &= (doc.labels_3d == 0)``          — full 348M-voxel mask scan
  6. ``int(mask.sum())``                       — full O(n) reduce
  7. ``np.where(mask)`` for the bbox + crop    — O(n) index gather

Run on a real display:
    DISPLAY=:1 LD_LIBRARY_PATH="" /usr/bin/python3 tests/headed_3d_wand_perf.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

INTENSITY_PATH = os.path.join(
    REPO,
    "GQR183_s02_eGFP_cellfill_PSD93SiRHalo_PSD93JF552Halo_eGFPfill_bottom_"
    "100x_100nmstep_10mthick_CrotexVIS_27072025-1.tif",
)


def _t(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    flag = "🚨" if dt * 1000 > 500 else ("⚠ " if dt * 1000 > 100 else "  ")
    print(f"  {flag} [{dt*1000:8.1f} ms]  {label}")
    return out


def main():
    if not os.path.exists(INTENSITY_PATH):
        print("FATAL: GQR TIFF missing")
        return 2

    print("Loading intensity TIFF...")
    import tifffile
    vol = tifffile.imread(INTENSITY_PATH)
    print(f"  vol {vol.shape} {vol.dtype}  {vol.nbytes / 1024**2:.0f} MB  "
          f"max={int(vol.max())}")

    # Pick a seed at the brightest voxel so flood will actually grow.
    seed_idx = int(np.argmax(vol))
    z, y, x = np.unravel_index(seed_idx, vol.shape)
    seed_tuple = (int(z), int(y), int(x))
    print(f"  seed @ {seed_tuple}, value={int(vol[seed_tuple])}")

    # Synthesize a labels_3d (background only) the same shape as vol —
    # mirrors what _run_wand has on a fresh ROI.
    labels_3d = np.zeros_like(vol, dtype=np.uint16)
    print(f"  labels_3d {labels_3d.shape} {labels_3d.dtype}")

    print("\n=== Wand step-by-step (tolerance=40, GQR seed) ===")
    seed_val = int(vol[seed_tuple])
    tol = 40

    clipped = _t(
        "Step 3: np.minimum(vol, seed_val)  — full 851 MB COPY",
        lambda: np.minimum(vol, seed_val),
    )

    from skimage.segmentation import flood
    mask = _t(
        "Step 4: skimage.flood(clipped, seed, tol)  — flood-fill",
        lambda: flood(clipped, seed_tuple, tolerance=tol),
    )
    print(f"     flood selected {int(mask.sum()):,} voxels "
          f"({mask.sum() / mask.size * 100:.1f}% of volume)")

    bg_mask = _t(
        "Step 5a: doc.labels_3d == 0  — 348M-voxel mask alloc",
        lambda: labels_3d == 0,
    )

    _t(
        "Step 5b: mask &= bg_mask  — in-place AND",
        lambda: mask.__iand__(bg_mask),
    )

    vox = _t(
        "Step 6: int(mask.sum())  — full O(n) reduce",
        lambda: int(mask.sum()),
    )
    print(f"     final wand selection: {vox:,} voxels")

    if vox > 0:
        zs_ys_xs = _t(
            "Step 7: np.where(mask)  — index gather",
            lambda: np.where(mask),
        )
        zs, ys, xs = zs_ys_xs
        _t(
            "Step 7b: bbox extraction (min/max per axis)",
            lambda: (int(zs.min()), int(zs.max()) + 1,
                     int(ys.min()), int(ys.max()) + 1,
                     int(xs.min()), int(xs.max()) + 1),
        )

    # Now also measure with a TIGHT tolerance so flood is small.
    print("\n=== Wand with tolerance=5 (tight selection) ===")
    clipped2 = _t(
        "Step 3: np.minimum(vol, seed_val)",
        lambda: np.minimum(vol, seed_val),
    )
    mask2 = _t(
        "Step 4: skimage.flood (tol=5)",
        lambda: flood(clipped2, seed_tuple, tolerance=5),
    )
    print(f"     flood selected {int(mask2.sum()):,} voxels")

    print("\n=== Total wand cost (old: np.where for bbox) ===")
    t0 = time.perf_counter()
    clipped3 = np.minimum(vol, seed_val)
    mask3 = flood(clipped3, seed_tuple, tolerance=tol)
    mask3 &= (labels_3d == 0)
    vox3 = int(mask3.sum())
    if vox3:
        zs, ys, xs = np.where(mask3)
        _ = (int(zs.min()), int(zs.max())+1, int(ys.min()),
             int(ys.max())+1, int(xs.min()), int(xs.max())+1)
    total_old = time.perf_counter() - t0
    print(f"  🚨 OLD bbox-via-where total: {total_old*1000:.1f} ms")

    print("\n=== Total wand cost (new: any-axis bbox) ===")
    t0 = time.perf_counter()
    clipped4 = np.minimum(vol, seed_val)
    mask4 = flood(clipped4, seed_tuple, tolerance=tol)
    mask4 &= (labels_3d == 0)
    vox4 = int(mask4.sum())
    if vox4:
        zs_idx = np.flatnonzero(mask4.any(axis=(1, 2)))
        ys_idx = np.flatnonzero(mask4.any(axis=(0, 2)))
        xs_idx = np.flatnonzero(mask4.any(axis=(0, 1)))
        _ = (int(zs_idx[0]), int(zs_idx[-1])+1,
             int(ys_idx[0]), int(ys_idx[-1])+1,
             int(xs_idx[0]), int(xs_idx[-1])+1)
    total_new = time.perf_counter() - t0
    print(f"  🚨 NEW any-axis-bbox total: {total_new*1000:.1f} ms")
    print(f"\n  Saved per wand click: {(total_old - total_new) * 1000:.0f} ms "
          f"({100 * (1 - total_new/total_old):.0f}% faster)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
