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
    """Write ``labels_3d`` as a uint16 multi-page TIFF plus sidecar JSON.

    Returns the two output paths. Promotes to uint16 on write so napari,
    which assumes multi-class Labels layers are integer-typed, can load
    without complaining even when only a handful of IDs are in use.
    """
    import tifffile

    path = Path(tiff_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # uint16 is the interoperable default — napari and Fiji both read it as
    # a Labels layer without prompting. uint8 would save bytes but breaks
    # once more than 255 ROIs are in the session.
    out = labels_3d.astype(np.uint16, copy=False)
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

    # uint16 is the on-disk type; downcast only if every ID fits in uint8.
    if arr.dtype not in (np.uint8, np.uint16, np.uint32, np.int32, np.int64):
        arr = arr.astype(np.uint16, copy=False)
    if arr.dtype != np.uint8 and arr.max() <= np.iinfo(np.uint8).max:
        arr = arr.astype(np.uint8, copy=False)
    else:
        arr = arr.astype(np.uint16, copy=False)

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
