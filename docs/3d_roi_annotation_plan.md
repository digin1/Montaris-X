# 3D ROI Annotation Plan

Add napari-style 3D ROI annotation to Montaris-X: a magic-wand / bucket-fill
tool that operates on the z-stack volume in the 3D view and produces ROIs
that round-trip with the existing 2D ROI pipeline.

## Goals

- Click a voxel in the 3D view → flood-fill a connected region (with a
  tolerance slider) → that region becomes a named, colored 3D ROI.
- Reuse the existing `LayerPanel` (rename / color / visibility / delete).
- Sliding through Z in 2D shows the 3D ROI's cross-section as a normal
  overlay on the 2D canvas.
- Export as a single napari/Fiji-compatible multi-page `uint16` TIFF.

## Storage model: hybrid labels volume + per-ROI wrappers

One labels volume per document, wrapped per-ROI for the panel.

```python
# montaris/layers.py
class MontageDocument:
    ...
    labels_3d: np.ndarray | None = None       # (Z, Y, X) uint16, shape == volume_data.shape
    labels_meta: dict[int, dict] | None = None  # {id: {"name", "color", "visible", "opacity"}}
```

- `labels_3d` is lazy-allocated on first 3D annotation (0 = background).
- Dtype starts as `uint8`; promoted to `uint16` when ID #256 is created.
- `VolumeROILayer` is a thin wrapper exposed to `LayerPanel`. It looks like
  `ROILayer` but all storage lives in `labels_3d` + `labels_meta`:

```python
class VolumeROILayer:
    is_roi = True
    is_volume = True   # lets the panel show a little badge

    def __init__(self, doc, label_id):
        self._doc = doc
        self._label_id = label_id

    @property
    def name(self):  return self._doc.labels_meta[self._label_id]["name"]
    @name.setter
    def name(self, v): self._doc.labels_meta[self._label_id]["name"] = v

    # color, opacity, visible: same pattern

    @property
    def mask(self):           # full 3D mask (lazy; only for export/paint)
        return (self._doc.labels_3d == self._label_id)

    def mask_slice(self, z):  # 2D bridge
        return (self._doc.labels_3d[z] == self._label_id)
```

LayerPanel does not need to care whether an item is `ROILayer` or
`VolumeROILayer` — it sees the same properties.

## UI changes

### View3DPanel (`montaris/widgets/view_3d.py`)

Add a small toolbar above the render controls:

- **Tool mode**: `Navigate | Fill | Paint | Erase` (QComboBox or button group).
- **Tolerance**: QSlider 0–255, only enabled for Fill. Flood accepts voxels
  where `|value − seed_value| ≤ tolerance` on the active channel.
- **Active channel**: QComboBox listing the loaded channels (the fill reads
  intensity from this one volume).
- **Active label**: the panel's currently-selected ROI (auto-picked from
  `LayerPanel.selected_roi`; falls back to "create new").
- **Brush radius**: QSpinBox, only for Paint/Erase.

Add a second `scene.visuals.Volume` layered over the intensity one,
showing `labels_3d` through a categorical colormap. Opacity slider controls
only the overlay.

### Interaction

- Navigate mode: current behavior (orbit/pan/zoom).
- Fill mode: left-click → cast a ray into the scene → find first voxel
  above an intensity threshold (or the max along the ray) → use that as
  seed → run `scipy.ndimage.label` / `flood_fill` on the active channel
  with tolerance → `labels_3d[mask] = active_label_id` → refresh overlay.
- Paint mode: left-drag paints a spherical brush along the ray's first
  hit; on each mouse move, stamp the sphere into `labels_3d`.
- Erase mode: same as paint but writes 0.

Undo/redo: push a sparse diff (`(z, y, x, old, new)` of changed voxels)
onto the existing command stack. For fill ops, record the bounding box
and pre-state of that sub-volume rather than every voxel.

## Fill algorithm (magic wand)

```python
from scipy.ndimage import binary_dilation
import numpy as np

def flood_3d(volume, seed_zyx, tolerance, max_voxels=50_000_000):
    """Connected-component flood fill with intensity tolerance.

    Uses a BFS on a boolean mask: voxels within `tolerance` of the seed
    value, 6-connected. Returns a (Z,Y,X) bool mask.
    """
    seed_val = float(volume[seed_zyx])
    lo = seed_val - tolerance
    hi = seed_val + tolerance
    candidate = (volume >= lo) & (volume <= hi)

    # Connected component containing the seed
    from scipy.ndimage import label
    lab, _ = label(candidate, structure=np.ones((3, 3, 3), dtype=bool))
    seed_comp = lab[seed_zyx]
    if seed_comp == 0:
        return np.zeros_like(volume, dtype=bool)
    return lab == seed_comp
```

Performance: for a 2048×2048×100 stack, `scipy.ndimage.label` is a few
seconds. Acceptable for a click-and-wait tool. Show `busy_cursor()` during
the operation.

## 2D bridge

In `ImageLayer._redraw_overlay` (or wherever 2D ROIs are composited),
extend the iteration over `ROILayer`s to also iterate `VolumeROILayer`s
and pull `layer.mask_slice(current_z)`. From the panel's side nothing
changes; only the compositor learns about the new type.

## Export / import

- **Save**: write `labels_3d` as a multi-page `uint16` TIFF using
  `tifffile.imwrite(path, labels_3d, photometric="minisblack")`. Sidecar
  JSON stores `labels_meta` (name, color, opacity per ID). Napari reads
  the TIFF as a Labels layer and ignores the JSON.
- **Load**: detect `uint8`/`uint16` single-channel stacks with discrete
  values → offer "Import as 3D Labels". Reconstruct `labels_meta` from
  sidecar JSON if present, otherwise auto-assign names and palette.
- **Flatten to 2D ROIs**: a menu action that, for each label ID, slices
  through Z and produces one `ROILayer` per slice (named
  `"{label_name}_z{z:03d}"`). This is the export path that keeps the
  legacy `roi_io.save_roi_set` / ImageJ ZIP format working.

## Phased rollout

1. **Phase 1 — data model**: add `labels_3d` + `labels_meta` to
   `MontageDocument`. Add `VolumeROILayer` wrapper. Make `LayerPanel`
   tolerant of it (should be no-op since it duck-types to `ROILayer`).

2. **Phase 2 — overlay render**: in `View3DPanel`, allocate and display
   an empty labels overlay (second `scene.visuals.Volume`). Verify it
   appears above the intensity volume and respects visibility/opacity
   controls. No interaction yet.

3. **Phase 3 — fill tool**: add the toolbar + `flood_3d` implementation +
   ray-pick → seed voxel. Bind click in Fill mode. Undo support.

4. **Phase 4 — 2D bridge**: extend the 2D compositor to show
   `labels_3d[z]` slices. Round-trip: create a 3D ROI, return to 2D,
   scroll Z, confirm slices show up.

5. **Phase 5 — paint / erase brush**: add spherical brush with radius.

6. **Phase 6 — export**: uint16 TIFF + sidecar JSON. Import side. Add a
   "Flatten to 2D ROIs" command.

## Open questions

- Anisotropic voxels: if the z-step is 100 nm and xy-pixel is 65 nm, the
  flood fill's 6-connectivity is isotropic but the *visual* result isn't.
  Probably good enough; revisit if users complain.
- Maximum ID count: `uint16` caps at 65 535 ROIs. Fine for any realistic
  microscopy session.
- RAM: one `uint16` volume for a 2048×2048×100 stack is ~800 MB. Lazy
  allocation keeps non-annotating users at zero cost; for large stacks
  we can add a "compressed on save" path later using RLE like the 2D
  ROIs already do.
