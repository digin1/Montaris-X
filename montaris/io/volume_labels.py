"""Save/load for 3D ROI label volumes.

Format:

- A multi-page uint16 TIFF containing the ``labels_3d`` array. Napari reads
  this as a Labels layer out-of-the-box; Fiji sees it as an integer stack.
- A sidecar JSON file (same stem, ``.labels.json``) recording per-id display
  attributes: name, color (RGB tuple), opacity, visible, fill_mode. Readers
  that don't know about the sidecar still get correctly-labeled voxels.

The sidecar is optional on load; when missing we regenerate names and a
palette from the existing ``ROI_COLORS`` so imports from napari/Fiji Just
Work.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from montaris.layers import ROI_COLORS


def _sidecar_path(tiff_path: str | Path) -> Path:
    """Return the JSON sidecar path that pairs with a labels TIFF."""
    p = Path(tiff_path)
    # Strip .tif / .tiff if present so "x.tif" → "x.labels.json", and
    # bare "x" → "x.labels.json". Case-insensitive suffix match.
    if p.suffix.lower() in (".tif", ".tiff"):
        return p.with_suffix(".labels.json")
    return p.with_name(p.name + ".labels.json")


def save_volume_labels(tiff_path: str | Path, labels_3d: np.ndarray,
                        labels_meta: dict[int, dict[str, Any]]) -> tuple[Path, Path]:
    """Write ``labels_3d`` as an unsigned integer multi-page TIFF plus sidecar.

    Returns the two output paths. The on-disk dtype is the smallest
    unsigned type that can hold every ID (uint8 for ≤255, uint16 for
    ≤65535, uint32 otherwise). uint16 is napari/Fiji's happy path but
    sessions with >65k ROIs promote automatically instead of silently
    wrapping.
    """
    import tifffile

    path = Path(tiff_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    max_id = max(int(labels_3d.max()) if labels_3d.size else 0,
                 int(max(labels_meta.keys()) if labels_meta else 0))
    if max_id <= np.iinfo(np.uint8).max:
        out_dtype = np.uint16  # uint8 labels are noisy in napari; use uint16 for ≤255
    elif max_id <= np.iinfo(np.uint16).max:
        out_dtype = np.uint16
    else:
        out_dtype = np.uint32
    out = labels_3d.astype(out_dtype, copy=False)
    tifffile.imwrite(str(path), out, photometric="minisblack")

    sidecar = _sidecar_path(path)
    payload = {
        "schema": "montaris.volume_labels.v1",
        "labels": {
            str(lid): {
                "name": meta.get("name", f"3D ROI {lid}"),
                "color": list(meta.get("color", ROI_COLORS[0])),
                "opacity": int(meta.get("opacity", 128)),
                "visible": bool(meta.get("visible", True)),
                "fill_mode": meta.get("fill_mode", "solid"),
            }
            for lid, meta in labels_meta.items()
        },
    }
    sidecar.write_text(json.dumps(payload, indent=2))
    return path, sidecar


def load_volume_labels(tiff_path: str | Path) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    """Read a labels TIFF + its sidecar, returning ``(labels_3d, labels_meta)``.

    When the sidecar is missing we fabricate metadata from the TIFF's
    actual unique IDs, assigning names ``3D ROI <n>`` and cycling through
    ``ROI_COLORS`` for the palette.
    """
    import tifffile

    path = Path(tiff_path)
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        # Single-slice "volume" — accept it and promote to (1, Y, X) so the
        # rest of the pipeline can treat it uniformly.
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D TIFF for labels, got shape {arr.shape}")

    # Reject anything we can't faithfully represent as unsigned integer labels.
    # Booleans collapse multi-class intent; floats/complex aren't label data.
    if arr.dtype.kind not in ("u", "i"):
        raise ValueError(
            f"Labels TIFF must be an integer dtype, got {arr.dtype}. "
            f"Booleans, floats, and complex arrays are not valid label data."
        )
    if arr.dtype.kind == "i":
        min_val = int(arr.min()) if arr.size else 0
        if min_val < 0:
            raise ValueError(
                f"Labels TIFF has negative values (min={min_val}); IDs must "
                f"be non-negative. Re-export without signed-int casting."
            )
    max_val = int(arr.max()) if arr.size else 0
    # Pick the narrowest unsigned dtype that fits without truncation. Most
    # crucially, do NOT collapse uint32/int32 into uint16 — a napari label
    # TIFF can hold IDs beyond 65535.
    if max_val <= np.iinfo(np.uint8).max:
        target = np.uint8
    elif max_val <= np.iinfo(np.uint16).max:
        target = np.uint16
    elif max_val <= np.iinfo(np.uint32).max:
        target = np.uint32
    else:
        raise ValueError(
            f"Labels TIFF contains an ID ({max_val}) that exceeds uint32 "
            f"range; Montaris-X only supports up to {np.iinfo(np.uint32).max}."
        )
    arr = arr.astype(target, copy=False)

    sidecar = _sidecar_path(path)
    meta: dict[int, dict[str, Any]] = {}
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text())
            for k, v in payload.get("labels", {}).items():
                lid = int(k)
                meta[lid] = {
                    "name": str(v.get("name", f"3D ROI {lid}")),
                    "color": tuple(v.get("color", ROI_COLORS[lid % len(ROI_COLORS)])),
                    "opacity": int(v.get("opacity", 128)),
                    "visible": bool(v.get("visible", True)),
                    "fill_mode": str(v.get("fill_mode", "solid")),
                }
        except (ValueError, KeyError, TypeError):
            # Sidecar corrupted — fall through to auto-meta below.
            meta = {}

    if not meta:
        ids = np.unique(arr)
        for lid in ids:
            if lid == 0:
                continue
            lid = int(lid)
            meta[lid] = {
                "name": f"3D ROI {lid}",
                "color": ROI_COLORS[(lid - 1) % len(ROI_COLORS)],
                "opacity": 128,
                "visible": True,
                "fill_mode": "solid",
            }

    return arr, meta
