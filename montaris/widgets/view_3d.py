"""3D volume-rendering dialog built on vispy.

Renders each z-stack from the active MontageDocuments as a separate Volume
visual in a shared ArcballCamera scene. Multi-channel stacks are composited
additively with per-channel colormaps (cyan / magenta / yellow by default,
matching common multi-channel microscopy LUTs).

The approach — uploading the volume as a 3D texture and ray-marching in a
fragment shader — is the same one napari uses, which is why we rely on the
same underlying engine (vispy). vispy is BSD-3-Clause; code here is original.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QSlider, QMessageBox, QWidget, QFrame, QSpinBox,
    QColorDialog, QButtonGroup, QApplication, QProgressBar,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics, QKeySequence, QShortcut, QColor


VISPY_AVAILABLE = True
try:
    from vispy import scene
    from vispy.color import Colormap
    from vispy.scene.cameras.perspective import PerspectiveCamera as _PerspectiveCamera

    class PanArcballCamera(scene.cameras.ArcballCamera):
        # Widest scale_factor observed. Pan speed is boosted at deep zoom
        # relative to this reference so drags at high magnification still
        # traverse a useful fraction of the scene without feeling floaty at
        # the initial fit-to-view.
        _pan_sf_ref: float = 0.0
        """ArcballCamera with right-drag remapped from zoom to pan.

        vispy's default bindings (right=zoom, Shift+left=pan) feel wrong for users
        coming from napari/ImageJ/Photoshop, where right-drag pans. Mouse wheel
        still zooms; left-drag still rotates.
        """

        def viewbox_mouse_event(self, event):
            if event.handled or not self.interactive:
                return

            # Intercept wheel zoom so we can pivot around the cursor instead
            # of always around self.center.
            if event.type == 'mouse_wheel':
                self._zoom_to_cursor(event)
                event.handled = True
                return

            # PerspectiveCamera handles other events (including trackpad
            # gesture_zoom) — skip Base3D's orbit bindings.
            _PerspectiveCamera.viewbox_mouse_event(self, event)

            if event.type == 'mouse_release':
                self._event_value = None
                return
            if event.type == 'mouse_press':
                event.handled = True
                return
            if event.type != 'mouse_move':
                return
            if event.press_event is None:
                return
            if 1 in event.buttons and 2 in event.buttons:
                return

            modifiers = event.mouse_event.modifiers
            p1 = event.mouse_event.press_event.pos
            p2 = event.mouse_event.pos

            if 1 in event.buttons and not modifiers:
                self._update_rotation(event)
            elif 2 in event.buttons and not modifiers:
                # Pan — logic borrowed from Base3DRotationCamera's Shift+left branch.
                norm = np.mean(self._viewbox.size)
                if self._event_value is None or len(self._event_value) == 2:
                    self._event_value = self.center
                # Keep pan 1:1 with the cursor at the fit-to-view zoom, but
                # accelerate as the user zooms in so a single drag still
                # covers a useful fraction of the scene. Cap at 4× to keep
                # things controllable at extreme zoom.
                self._pan_sf_ref = max(self._pan_sf_ref, self._scale_factor)
                ratio = self._pan_sf_ref / max(self._scale_factor, 1e-6)
                gain = min(4.0, max(1.0, ratio ** 0.5))
                dist = (p1 - p2) / norm * self._scale_factor * gain
                dist[1] *= -1
                dx, dy, dz = self._dist_to_trans(dist)
                ff = self._flip_factors
                up, forward, right = self._get_dim_vectors()
                dx, dy, dz = right * dx + forward * dy + up * dz
                dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]
                c = self._event_value
                self.center = c[0] + dx, c[1] + dy, c[2] + dz

        def _cursor_world_pos(self, pos):
            """Map a viewbox pixel position to the world point on the pivot plane.

            For perspective cameras, a single unprojected point is ambiguous:
            any point along the view ray projects to the same pixel. We want
            the point at the depth the user *perceives* — the plane through
            ``self.center`` perpendicular to the view direction.

            We unproject at two NDC z-levels (near and far) to build the view
            ray, then intersect it with that pivot plane.
            """
            try:
                tr = self._viewbox.scene.node_transform(self._viewbox)
                p_near = tr.imap(np.array([pos[0], pos[1], -1, 1], dtype=float))
                p_far = tr.imap(np.array([pos[0], pos[1], 1, 1], dtype=float))
                w_n = p_near[3] if p_near[3] else 1.0
                w_f = p_far[3] if p_far[3] else 1.0
                p_near = p_near[:3] / w_n
                p_far = p_far[:3] / w_f
                ray_d = p_far - p_near
                nrm = np.linalg.norm(ray_d)
                if nrm < 1e-9:
                    return np.array(self.center)
                ray_d /= nrm
                center = np.array(self.center)
                # Use ray_d as plane normal — an approximation of camera forward
                # that's exact at screen center and very close to it elsewhere
                # for typical FOVs.
                t = float(np.dot(center - p_near, ray_d))
                return p_near + ray_d * t
            except Exception:
                vb_w, vb_h = self._viewbox.size
                ndc_x = (pos[0] - vb_w / 2.0) / (vb_w / 2.0)
                ndc_y = -(pos[1] - vb_h / 2.0) / (vb_h / 2.0)
                half_h = self._scale_factor / 2.0
                half_w = half_h * (vb_w / max(vb_h, 1))
                up, forward, right = self._get_dim_vectors()
                ff = self._flip_factors
                off = (right * ndc_x * half_w * ff[0]
                       + up * ndc_y * half_h * ff[1])
                return np.array(self.center) + off

        def _zoom_to_cursor(self, event):
            """Scale the view around the mouse pointer, not around self.center."""
            delta = event.delta[1]
            if delta == 0:
                return
            # 1.5 per tick — cursor-anchored zoom hides a lot of the visible
            # motion behind the anchor point, so we need a larger coefficient
            # than vispy's 1.1 default to feel responsive. 1.5 gives a ~33%
            # zoom per notch, which matches typical 3D DCC apps (Blender, Maya).
            scale = 1.5 ** -delta

            cursor_world = self._cursor_world_pos(event.pos)

            old_sf = self._scale_factor
            new_sf = old_sf * scale
            self._scale_factor = new_sf
            if self._distance is not None:
                self._distance *= scale

            # For orthographic-like projections the world point under the
            # cursor stays put if we shift self.center by (1 - scale) of the
            # vector from the old center to the cursor. For the 45° perspective
            # ArcballCamera this is an approximation, but a good one — the
            # cursor visually anchors during zoom, which is what users expect.
            c = np.array(self.center)
            self.center = tuple(c + (cursor_world - c) * (1.0 - scale))

            self.view_changed()
except Exception:  # noqa: BLE001 — vispy is optional; report failure at call site
    VISPY_AVAILABLE = False


# Conservative fallback used only if the live GL context won't answer a
# GL_MAX_3D_TEXTURE_SIZE query. The real cap is detected per-dialog at runtime
# (see _detect_max_3d_dim) so the code auto-adapts to the active GPU across
# Wayland/X11/macOS/Windows without hardcoded assumptions.
_MAX_3D_DIM_FALLBACK = 2048

import os


# Raw GL enum values — avoids depending on vispy exposing them by name.
_GL_VENDOR = 0x1F00
_GL_RENDERER = 0x1F01
_GL_MAX_3D_TEXTURE_SIZE = 0x8073
# NVIDIA's NVX_gpu_memory_info extension — returns kilobytes.
_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX = 0x9049
_GPU_MEMORY_INFO_TOTAL_AVAILABLE_VIDMEM_NVX = 0x9048
# AMD's ATI_meminfo extension — returns a 4-tuple of kilobytes per pool
# (total free, largest free block, total aux free, largest aux free). We
# care about the texture pool's total free memory for labels + intensity
# volume uploads.
_TEXTURE_FREE_MEMORY_ATI = 0x87FC


def _query_gl(canvas, enum, fallback):
    """Pull a GL state value from the canvas's live context via vispy's wrapper."""
    try:
        from vispy.gloo import gl
        return gl.glGetParameter(enum)
    except Exception:
        return fallback


def _coerce_gl_int(value, fallback=0):
    """Normalise a GL state return into an int.

    vispy's ``glGetParameter`` wraps ``glGetIntegerv``, which can yield a
    tuple/list when the driver reports multiple components for the same
    enum (common for NVIDIA's NVX_gpu_memory_info in some driver builds,
    and for ``GL_MAX_3D_TEXTURE_SIZE`` on wrappers that always return
    fixed-size arrays). Treating such a return as opaque and dropping
    through to the heuristic silently caps the VRAM budget at 1 GB on a
    24 GB card — the user sees visibly-worse downsample than they should.
    """
    if isinstance(value, bool):  # bool is a subclass of int; skip
        return fallback
    if isinstance(value, (int, float)):
        return int(value)
    # Bytes/str iterate per-byte/per-char and would return the ordinal of
    # the first character — not an integer count. Reject them outright.
    if isinstance(value, (bytes, bytearray, str)):
        return fallback
    try:
        if hasattr(value, "__len__") and len(value) == 0:
            return fallback
        for x in value:
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                return int(x)
    except TypeError:
        pass
    return fallback


def _detect_max_3d_dim(canvas):
    try:
        raw = _query_gl(canvas, _GL_MAX_3D_TEXTURE_SIZE, _MAX_3D_DIM_FALLBACK)
        cap = _coerce_gl_int(raw, _MAX_3D_DIM_FALLBACK)
        return cap if cap > 0 else _MAX_3D_DIM_FALLBACK
    except Exception:
        return _MAX_3D_DIM_FALLBACK


def _gpu_description(canvas):
    vendor = _query_gl(canvas, _GL_VENDOR, "")
    renderer = _query_gl(canvas, _GL_RENDERER, "")
    # vispy may return bytes on some backends.
    if isinstance(vendor, bytes):
        vendor = vendor.decode(errors="replace")
    if isinstance(renderer, bytes):
        renderer = renderer.decode(errors="replace")
    return f"{vendor} — {renderer}".strip(" —")


def _detect_vram_budget_bytes(canvas, gpu_desc):
    """Return a safe per-upload VRAM budget in bytes.

    Precedence:
    1. MONTARIS_GPU_BUDGET_MB environment override (explicit user control).
    2. NVIDIA NVX_gpu_memory_info query — real free VRAM, halved for headroom.
    3. AMD ATI_meminfo texture-pool query — real free VRAM, halved.
    4. Heuristic based on GL_RENDERER string:
       - Software (llvmpipe/swr) → 64 MB (CPU path, keep tiny)
       - Intel / integrated      → 256 MB (shares system RAM, stay modest)
       - Apple Silicon           → 1024 MB (unified memory, but don't hog)
       - NVIDIA / AMD discrete   → 4096 MB (modern workstation baseline;
                                             OOM retry adds stride if wrong)

    OpenGL has no portable free-VRAM query, so the heuristic is the honest
    best-effort for non-NVIDIA GPUs.
    """
    override = os.environ.get("MONTARIS_GPU_BUDGET_MB")
    if override:
        try:
            return max(16, int(override)) * 1024 * 1024
        except ValueError:
            pass

    # NVX extension path — only exists on NVIDIA drivers. The raw return
    # can be an int OR a multi-element tuple/array depending on the GL
    # wrapper / driver build; coerce before using it.
    try:
        raw = _query_gl(canvas, _GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, 0)
        kb = _coerce_gl_int(raw, 0)
        if kb > 0:
            # Half of free VRAM leaves room for other textures + framebuffers.
            return kb * 1024 // 2
    except Exception:
        pass

    # ATI_meminfo extension path — AMD cards. TEXTURE_FREE_MEMORY_ATI
    # returns a 4-tuple of kilobytes; element [0] is the texture pool's
    # total free memory. Other vendors silently return no-data (GL_INVALID_ENUM
    # or a zero tuple) so this call is safe to attempt unconditionally.
    try:
        raw = _query_gl(canvas, _TEXTURE_FREE_MEMORY_ATI, 0)
        kb = _coerce_gl_int(raw, 0)
        if kb > 0:
            return kb * 1024 // 2
    except Exception:
        pass

    desc = (gpu_desc or "").lower()
    if "llvmpipe" in desc or "swr" in desc or "software" in desc:
        return 64 * 1024 * 1024
    if "intel" in desc:
        return 256 * 1024 * 1024
    if "apple" in desc or "metal" in desc:
        return 1024 * 1024 * 1024
    # NVIDIA/AMD heuristic fallback. The NVX path should have hit on
    # NVIDIA; if it didn't (driver variant, non-NVIDIA discrete), 4 GB
    # matches the low end of modern discrete cards and keeps the
    # downsample from collapsing to a 1 GB cap on users with 12-24 GB
    # cards like the RTX 4090.
    if "nvidia" in desc or "geforce" in desc or "rtx" in desc or "quadro" in desc \
       or "amd" in desc or "radeon" in desc:
        return 4 * 1024 * 1024 * 1024
    # Unknown GPU — be conservative.
    return 256 * 1024 * 1024


def _fit_to_budget(shapes, budget_bytes, bytes_per_voxel=1):
    """Return a stride factor that keeps the total upload within budget.

    *shapes* is a list of (Z, Y, X) tuples, one per channel. We apply the same
    stride to all channels so their axes stay aligned for compositing.
    """
    if not shapes or budget_bytes <= 0:
        return 1
    total = sum(s[0] * s[1] * s[2] for s in shapes) * bytes_per_voxel
    if total <= budget_bytes:
        return 1
    # Each stride factor f reduces voxels by f^3. Solve for ceil(cubert(ratio)).
    import math
    ratio = total / budget_bytes
    return max(1, int(math.ceil(ratio ** (1 / 3))))


# Default per-channel tints, cycled across N channels. Cyan/magenta/yellow is a
# balanced, colorblind-safe triplet widely used for multi-channel fluorescence.
_DEFAULT_TINTS = [
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 0.0, 1.0),  # magenta
    (1.0, 1.0, 0.0),  # yellow
    (0.2, 1.0, 0.4),  # green
    (1.0, 0.4, 0.2),  # orange
    (0.4, 0.6, 1.0),  # light blue
]

# Render methods we expose. vispy's 'iso' mode ships but has shader issues on
# some GL drivers (silent black output); 'attenuated_mip' gives a similar
# 3D-feel MIP that works reliably, so we use that instead.
_RENDER_METHODS = ['mip', 'attenuated_mip', 'translucent', 'additive', 'average']


class _ToolComboShim:
    """Duck-typed ``QComboBox``-like adapter over ``View3DPanel._tool_buttons``.

    Keeps the pre-existing API (``currentText`` / ``setCurrentText`` /
    ``count`` / ``itemText`` / ``addItems``) working after the tool picker
    was converted from a dropdown to an exclusive toggle-button row. That
    means tests and scripts that drive tool changes via the combo still
    work untouched.
    """

    def __init__(self, panel):
        self._panel = panel

    def currentText(self):
        return self._panel._current_tool_label()

    def setCurrentText(self, label):
        self._panel._set_current_tool(label)

    def count(self):
        return len(self._panel._tool_buttons)

    def itemText(self, idx):
        labels = list(self._panel._tool_buttons.keys())
        if 0 <= idx < len(labels):
            return labels[idx]
        return ""

    def addItems(self, items):
        # The button row is fixed at construction time; extra items would
        # require re-laying out the row. Raise loudly so silent test drift
        # surfaces rather than producing a combo-vs-buttons mismatch.
        raise NotImplementedError(
            "Tool list is fixed; extend _tool_buttons in __init__ instead."
        )


def _tinted_colormap(rgb):
    """Build a black→colour colormap with alpha ramping from 0 to 1.

    Used so dark background voxels don't occlude other channels during
    additive/translucent rendering.
    """
    r, g, b = rgb
    colors = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [r, g, b, 1.0],
    ], dtype=np.float32)
    return Colormap(colors)


def _percentile_clim(volume, lo=1.0, hi=99.5):
    """Estimate display limits from a subsample (fast on large uint16 stacks)."""
    flat = volume.ravel()
    if flat.size > 1_000_000:
        step = flat.size // 1_000_000
        flat = flat[::step]
    a, b = np.percentile(flat, [lo, hi])
    if b <= a:
        b = a + 1
    return float(a), float(b)


def _downsample(volume, factor):
    if factor <= 1:
        return volume
    return volume[::factor, ::factor, ::factor]


def _max_pool_labels(labels, block, id_map=None, out=None):
    """Downsample an integer labels volume by block-max instead of stride.

    Stride-decimation on sparse integer labels collapses thin structures
    (e.g. 1-voxel-wide dendrites) into disconnected dots — the "skeleton"
    artefact. Block max-pool keeps any voxel whose block contained a nonzero
    label, which is the right behaviour for overlays where the user just
    needs to see where ROIs are rather than exact voxel counts.

    ``block`` is a ``(dz, dy, dx)`` tuple of per-axis pooling factors.

    ``id_map`` (optional) is a 1-D lookup array indexed by label id. When
    given, each sub-view is remapped through it before being folded into
    the block max. This is how partial visibility is handled: invisible
    labels map to 0 so max-pool won't pick them over a visible but
    smaller-id neighbour. Without this remap, a block containing ROI 3
    (visible) and ROI 150 (hidden) pools to 150; the cmap gives alpha=0
    to 150; ROI 3's voxels in that block disappear — the "skeleton" you
    see when only some ROIs are toggled on.

    Output shape matches ``ceil(S/f)`` per axis — the same shape the
    intensity pipeline's stride produces via ``_fit_to_gpu`` — so the
    STTransform scale in :meth:`_rebuild_labels_overlay` keeps the overlay
    voxel-for-voxel aligned with the intensity volume. Trailing-edge
    voxels that don't fill a full block still contribute: we accumulate
    by phase-offset strided views into a pre-allocated output, so cells
    whose block straddles the array edge just get max over the present
    neighbours without any padding or trim.
    """
    dz, dy, dx = (max(1, int(b)) for b in block)
    no_pool = (dz, dy, dx) == (1, 1, 1)
    if no_pool and id_map is None:
        return labels
    # Defensive: a short id_map would IndexError on advanced indexing below.
    # Callers should size from ``max(labels.max(), max_meta_id)``; if they
    # slipped, extend the LUT with identity entries so indexing is safe.
    if id_map is not None:
        lbl_max = int(labels.max()) if labels.size else 0
        if lbl_max >= id_map.size:
            extended = np.arange(lbl_max + 1, dtype=id_map.dtype)
            extended[:id_map.size] = id_map
            id_map = extended
    if no_pool:
        # No spatial reduction — just remap ids. The cmap already handles
        # alpha=0 for invisible labels, so callers usually skip the id_map
        # here, but support it for parity with the pooled path.
        return id_map[labels]
    Z, Y, X = labels.shape
    oz = (Z + dz - 1) // dz
    oy = (Y + dy - 1) // dy
    ox = (X + dx - 1) // dx
    # When ``id_map`` is given, the output values are LUT outputs, not
    # raw labels — allocate ``out`` in the LUT dtype so a uint8 dense
    # palette doesn't get force-promoted to the labels' uint16/uint32
    # via ``np.maximum(uint16, uint8) → uint16``. Halves GPU upload
    # bytes for sparse uint16 label files (the common case).
    out_dtype = id_map.dtype if id_map is not None else labels.dtype
    if out is None:
        out = np.zeros((oz, oy, ox), dtype=out_dtype)
    else:
        if out.shape != (oz, oy, ox) or out.dtype != out_dtype:
            out = np.zeros((oz, oy, ox), dtype=out_dtype)
        else:
            out.fill(0)
    for iz in range(dz):
        sz = Z - iz
        if sz <= 0:
            continue
        nz = (sz + dz - 1) // dz
        for iy in range(dy):
            sy = Y - iy
            if sy <= 0:
                continue
            ny = (sy + dy - 1) // dy
            for ix in range(dx):
                sx = X - ix
                if sx <= 0:
                    continue
                nx = (sx + dx - 1) // dx
                sub = labels[iz::dz, iy::dy, ix::dx]
                if id_map is not None:
                    sub = id_map[sub]
                np.maximum(out[:nz, :ny, :nx], sub,
                           out=out[:nz, :ny, :nx])
    return out


def _fit_to_gpu(volume, max_dim=_MAX_3D_DIM_FALLBACK):
    """Stride-downsample axes exceeding the driver's 3D-texture limit.

    Returns (volume_or_view, per_axis_stride_factors). If every axis already
    fits, the original array is returned unchanged.
    """
    factors = [max(1, (s + max_dim - 1) // max_dim) for s in volume.shape]
    if all(f == 1 for f in factors):
        return volume, factors
    fz, fy, fx = factors
    return volume[::fz, ::fy, ::fx], factors


def _ray_aabb(bmin, bmax, origin, direction):
    """Ray vs. axis-aligned bounding box slab test.

    Returns ``(t_enter, t_exit)`` — the parametric ``t`` values where the
    ray first enters and finally leaves the box. If the ray misses the box,
    the returned values satisfy ``t_exit < t_enter`` and the caller should
    skip. ``origin`` and ``direction`` must be 3-vectors; ``direction`` need
    not be unit but behaves best when normalized.
    """
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    bmin = np.asarray(bmin, dtype=float)
    bmax = np.asarray(bmax, dtype=float)
    # Avoid /0 for rays parallel to an axis.
    eps = 1e-12
    inv = 1.0 / np.where(np.abs(d) < eps, eps, d)
    t1 = (bmin - o) * inv
    t2 = (bmax - o) * inv
    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)
    return float(np.max(t_min)), float(np.min(t_max))


def _to_uint8(volume, clim):
    """Normalise to uint8 using clim.

    vispy picks a GL pixel format from the numpy dtype; uint16 falls back to
    GL_LUMINANCE + GL_BYTE on some drivers, which silently clips to ±127 and
    looks black. Pre-converting to uint8 sidesteps that.
    """
    lo, hi = float(clim[0]), float(clim[1])
    if hi <= lo:
        hi = lo + 1
    v = volume.astype(np.float32)
    v -= lo
    v *= 255.0 / (hi - lo)
    np.clip(v, 0, 255, out=v)
    return v.astype(np.uint8)


class View3DPanel(QWidget):
    """Embed a vispy SceneCanvas and its controls as a reusable QWidget.

    Designed to be dropped into a QStackedWidget (or any parent layout) so
    the 3D view can live inside the main window. ``release_gl()`` must be
    called before discarding the panel so volume textures are freed.
    """

    # Emitted after a fill/paint operation successfully writes voxels into
    # ``doc.labels_3d``. Payload is (document, label_id). The main app listens
    # and mirrors the new label into its LayerStack so the 2D LayerPanel picks
    # it up.
    label_added = Signal(object, int)
    # Emitted after a stroke/fill with a ready-to-push undo command. App-
    # level code connects this to its UndoStack.push() so Ctrl+Z can reverse
    # 3D paint, erase, and fill operations.
    undo_pushed = Signal(object)

    def __init__(self, parent=None, channels=None, documents=None):
        """channels: list of (name, volume_ndarray, tint_rgb_or_None).

        documents: optional list of MontageDocument. When provided, the
        first document with volume_data becomes the "primary" doc and its
        labels_3d / labels_meta back a labels overlay rendered on top of
        the intensity volumes.
        """
        super().__init__(parent)
        self._channels_raw = list(channels or [])
        self._documents = list(documents or [])
        self._primary_doc = next(
            (d for d in self._documents if getattr(d, 'volume_data', None) is not None),
            None,
        )
        self._volumes = []        # list of scene.visuals.Volume (intensity)
        self._labels_volume = None  # scene.visuals.Volume for labels_3d overlay
        self._labels_visible = True
        # Dense-palette LUT cached after each upload so ``_on_drag_refresh_tick``
        # can remap raw ``labels_3d`` voxels into palette slots before its
        # sub-region texture upload. The texture stores slots, NOT raw ids,
        # so writing a raw value (5, 663, …) into it would sample a wrong
        # cmap entry and bypass the slot-0 visibility filter — exactly the
        # bug the dense-palette fix removed from the full-rebuild path.
        self._labels_texture_lut = None
        self._labels_palette_size = 0
        self._labels_opacity = 0.6
        # Cache for the pooled labels array — reused across visibility toggles
        # to avoid re-allocating ~0.5 GB per refresh. Keyed by (shape, dtype,
        # block); invalidated when any of those change (downsample slider,
        # import, etc.). Preallocation also lets us pass ``out=`` into the
        # pooler so each call just overwrites the buffer.
        self._pooled_labels_buf = None
        self._pooled_labels_key = None  # (shape, dtype, block)
        # Reentrancy guard for _rebuild_labels_overlay. show_progress flushes
        # paint events which in turn can process queued signals (visibility
        # toggles, opacity edits) that re-enter refresh_labels. Even with
        # ExcludeUserInputEvents the guard is cheap belt-and-suspenders.
        self._rebuilding_labels = False
        # Last total downsample applied to intensity volumes, kept as
        # per-axis ``(dz, dy, dx)`` so non-cubic stacks that only exceed the
        # driver's 3D-texture limit on one axis render at correct scale.
        # ``_last_total_ds`` retains the scalar max for display/heuristics.
        self._last_total_ds_axes = (1, 1, 1)
        self._last_total_ds = 1
        # Annotation tool state. ``_tool_mode`` is one of navigate/fill/paint/
        # erase. ``_fill_channel_idx`` selects which intensity channel the flood
        # fill reads from (each channel is usually a different wavelength and
        # they don't share intensities). Paint/erase operate on ``labels_3d``
        # directly and don't need a source channel.
        self._tool_mode = 'navigate'
        self._fill_channel_idx = 0
        self._fill_tolerance = 40  # 0..255 in uint8 space (auto-scaled per channel)
        # Brush stroke state. ``_brush_radius`` is in voxels; the stamp is a
        # true 3D sphere in voxel space (anisotropy is ignored — good enough
        # at typical microscopy step ratios). The drag-* fields are reset at
        # the end of every stroke.
        self._brush_radius = 5
        self._drag_active = False
        self._drag_mode = None         # 'paint' or 'erase' during a stroke
        self._drag_label_id = None     # reserved on mouse_press for paint
        self._drag_dirty = False       # set by _stamp_brush, consumed by timer
        # Bbox of voxels touched since the last overlay refresh: accumulated
        # across _stamp_brush calls and consumed by the drag tick for a
        # sub-region texture upload (instead of recreating the whole volume
        # visual every frame — the difference is seconds vs. milliseconds on
        # a 50M+ voxel stack). Format: (z0, z1, y0, y1, x0, x1) or None.
        self._dirty_bbox = None
        # Stroke-wide state for undo. The render-throttle ``_dirty_bbox``
        # gets cleared every tick, which would miss voxels touched across
        # non-contiguous drags. ``_stroke_bbox`` / ``_stroke_before`` accumulate
        # for the whole stroke and are consumed by ``_finish_drag`` to build
        # the VolumeStrokeUndoCommand. ``_stroke_dtype`` pins the before-patch
        # dtype so later dtype promotions don't invalidate the snapshot.
        self._stroke_bbox = None
        self._stroke_before = None
        self._stroke_dtype = None
        # When non-None and the id exists in the active doc, paint strokes
        # extend that ROI instead of allocating a new one. MontarisApp pushes
        # this from the LayerPanel's selection_changed signal.
        self._active_volume_roi_id = None
        self._drag_extends_existing = False
        self._canvas = None
        self._view = None
        self._downsample_factor = 1

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Canvas. Passing `parent=self` here is important — without it vispy
        # creates the native QWidget as a top-level window first and then
        # reparents it, which on Wayland destroys and recreates the surface
        # (visible as the main window closing and reopening).
        self._canvas = scene.SceneCanvas(
            keys='interactive', show=False, bgcolor='black', parent=self,
        )
        root.addWidget(self._canvas.native, 1)

        # Slim progress bar (matches 2D canvas style) for slow refreshes
        # (partial-visibility re-pool) and label-TIFF imports. Indeterminate
        # by default — the work is mostly a single numpy pool + GL upload,
        # so there's no useful percentage to report.
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: transparent; border: none; }"
            "QProgressBar::chunk { background: #00b4ff; }"
        )
        self._progress_bar.hide()
        root.addWidget(self._progress_bar)

        self._view = self._canvas.central_widget.add_view()
        self._view.camera = PanArcballCamera(fov=0)

        # Detect the real 3D-texture cap and GPU label once the context is live.
        # Runs here (not at module load) because the value is tied to the
        # specific GL context vispy bound to.
        self._max_3d_dim = _detect_max_3d_dim(self._canvas)
        self._gpu_desc = _gpu_description(self._canvas)
        self._vram_budget = _detect_vram_budget_bytes(self._canvas, self._gpu_desc)
        # Extra stride applied after a failed upload — grows only if the
        # initial estimate wasn't aggressive enough and the driver throws OOM.
        self._oom_retry_factor = 1
        self._last_reason = ""  # 'GPU' | 'memory' | 'GPU+memory' | 'memory×N'

        # Controls row
        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("Render:"))
        self._mode_combo = QComboBox()
        # Pretty-print internal names: 'attenuated_mip' → 'Attenuated MIP'.
        self._mode_labels = {m: m.replace('_', ' ').title().replace('Mip', 'MIP')
                             for m in _RENDER_METHODS}
        self._mode_combo.addItems([self._mode_labels[m] for m in _RENDER_METHODS])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        controls.addWidget(self._mode_combo)

        controls.addWidget(self._sep())

        controls.addWidget(QLabel("Quality:"))
        self._ds_slider = QSlider(Qt.Horizontal)
        self._ds_slider.setRange(1, 4)  # 1x, 2x, 3x, 4x downsample
        self._ds_slider.setValue(1)
        self._ds_slider.setFixedWidth(120)
        self._ds_slider.setTickPosition(QSlider.TicksBelow)
        self._ds_slider.setTickInterval(1)
        self._ds_label = QLabel("full")
        self._ds_slider.valueChanged.connect(self._on_downsample_changed)
        controls.addWidget(self._ds_slider)
        controls.addWidget(self._ds_label)

        controls.addWidget(self._sep())

        reset_btn = QPushButton("Reset view")
        reset_btn.clicked.connect(self._reset_view)
        controls.addWidget(reset_btn)

        controls.addStretch(1)

        # GPU indicator — helpful for diagnosing PRIME/offload situations on
        # hybrid-GPU laptops where the app may silently pick the iGPU.
        self._gpu_badge = QLabel(self._short_gpu_label())
        pref = os.environ.get("MONTARIS_GPU_PREFERENCE", "").strip()
        pref_line = f"\nLauncher preference: {pref}" if pref else ""
        self._gpu_badge.setToolTip(
            f"Active GL context\n{self._gpu_desc}\n"
            f"GL_MAX_3D_TEXTURE_SIZE = {self._max_3d_dim}\n"
            f"VRAM budget per upload ≈ {self._vram_budget // (1024*1024)} MB\n"
            f"(override with MONTARIS_GPU_BUDGET_MB env var)"
            f"{pref_line}\n"
            f"Use --list-gpus / --gpu-index N / --no-gpu at launch to change."
        )
        self._gpu_badge.setStyleSheet("color: #888; padding: 0 6px;")
        controls.addWidget(self._gpu_badge)

        root.addLayout(controls)

        # Per-channel visibility row. Channel names are often 100+-char TIFF
        # filenames; without elision each QCheckBox forces the dialog's
        # minimum width to grow past typical screens (3 channels × ~800 px).
        # A 16×16 swatch button sits next to each checkbox: click to recolor
        # that channel live (swatch background = current tint).
        self._channel_toggles = []
        self._channel_swatches = []
        ch_row = QHBoxLayout()
        ch_row.setSpacing(12)
        for idx, (name, _vol, _tint) in enumerate(self._channels_raw):
            swatch = QPushButton()
            swatch.setFixedSize(16, 16)
            swatch.setToolTip(f"Change colour for {name}")
            swatch.clicked.connect(lambda _=False, i=idx: self._pick_channel_tint(i))
            ch_row.addWidget(swatch)
            self._channel_swatches.append(swatch)

            cb = QCheckBox()
            fm = QFontMetrics(cb.font())
            cb.setText(fm.elidedText(name, Qt.ElideMiddle, 180))
            cb.setToolTip(name)
            cb.setChecked(True)
            cb.toggled.connect(lambda on, i=idx: self._set_visible(i, on))
            ch_row.addWidget(cb)
            self._channel_toggles.append(cb)

            rgb = self._tint_for(idx, self._channels_raw[idx][2])
            self._refresh_channel_tint_ui(idx, rgb)
        ch_row.addStretch(1)
        if self._channel_toggles:
            root.addLayout(ch_row)

        # Labels overlay controls. Only shown when a primary document is
        # attached — the 2D-only callers (legacy open_view_3d, headed tests)
        # pass no documents, and in that case we skip this row entirely.
        self._labels_cb = None
        self._labels_opacity_slider = None
        if self._primary_doc is not None:
            labels_row = QHBoxLayout()
            labels_row.setSpacing(8)
            self._labels_cb = QCheckBox("3D ROIs")
            self._labels_cb.setChecked(self._labels_visible)
            self._labels_cb.setToolTip(
                "Show the 3D ROI labels volume on top of the intensity channels."
            )
            self._labels_cb.toggled.connect(self._on_labels_visible_toggled)
            labels_row.addWidget(self._labels_cb)

            labels_row.addWidget(QLabel("Opacity:"))
            self._labels_opacity_slider = QSlider(Qt.Horizontal)
            self._labels_opacity_slider.setRange(0, 100)
            self._labels_opacity_slider.setValue(int(self._labels_opacity * 100))
            self._labels_opacity_slider.setFixedWidth(120)
            self._labels_opacity_slider.valueChanged.connect(
                self._on_labels_opacity_changed
            )
            labels_row.addWidget(self._labels_opacity_slider)
            labels_row.addStretch(1)
            root.addLayout(labels_row)

            # Annotation tools. Fill and Wand are intentionally separate —
            # Fill is napari-style label relabel (no intensity, no tolerance);
            # Wand is intensity-based 3D flood into the selected ROI using
            # Channel + Tolerance. Paint/Erase are spherical brushes.
            # Presented as an exclusive toggle-button row (not a dropdown)
            # so all tools are one click away and the active tool is visible.
            tool_row = QHBoxLayout()
            tool_row.setSpacing(8)
            tool_row.addWidget(QLabel("Tool:"))
            self._tool_buttons: dict[str, QPushButton] = {}
            self._tool_group = QButtonGroup(self)
            self._tool_group.setExclusive(True)
            for label, shortcut in (
                ("Navigate", "V"), ("Fill", "F"), ("Wand", "W"),
                ("Paint", "P"), ("Erase", "E"),
            ):
                btn = QPushButton(f"{label} ({shortcut})")
                btn.setCheckable(True)
                btn.setToolTip(self._TOOL_HINTS.get(label.lower(), label))
                btn.clicked.connect(
                    lambda _=False, lbl=label: self._set_current_tool(lbl)
                )
                self._tool_group.addButton(btn)
                tool_row.addWidget(btn)
                self._tool_buttons[label] = btn
            self._tool_buttons["Navigate"].setChecked(True)
            # Backward-compat shim: earlier code (and plenty of tests) drove
            # tool changes through ``_tool_combo.setCurrentText(...)`` /
            # ``currentText()``. Expose the same duck-typed API so the
            # button-row swap stays a drop-in replacement.
            self._tool_combo = _ToolComboShim(self)

            tool_row.addWidget(self._sep())

            # Wand group — Channel + Tolerance. Disabled unless Wand is active.
            self._wand_channel_label = QLabel("Wand Channel:")
            tool_row.addWidget(self._wand_channel_label)
            self._fill_channel_combo = QComboBox()
            for idx, (name, _vol, _tint) in enumerate(self._channels_raw):
                fm = QFontMetrics(self._fill_channel_combo.font())
                self._fill_channel_combo.addItem(
                    fm.elidedText(name, Qt.ElideMiddle, 160), idx
                )
            self._fill_channel_combo.setCurrentIndex(0)
            self._fill_channel_combo.currentIndexChanged.connect(
                self._on_fill_channel_changed
            )
            tool_row.addWidget(self._fill_channel_combo)

            self._wand_tol_label_prefix = QLabel("Tolerance:")
            tool_row.addWidget(self._wand_tol_label_prefix)
            self._tol_slider = QSlider(Qt.Horizontal)
            # Tolerance is an absolute 0..255 intensity delta (not a percent).
            # Max 255 lets the wand reach dim boundaries of a bright dendrite
            # without asking users to understand the uint8 intensity space.
            self._tol_slider.setRange(0, 255)
            self._tol_slider.setValue(40)
            self._tol_slider.setFixedWidth(140)
            self._tol_slider.valueChanged.connect(self._on_tolerance_changed)
            tool_row.addWidget(self._tol_slider)
            self._tol_label = QLabel("40")
            tool_row.addWidget(self._tol_label)

            tool_row.addWidget(self._sep())

            # Brush group — Size. Disabled unless Paint/Erase is active.
            self._brush_size_label = QLabel("Brush Size:")
            tool_row.addWidget(self._brush_size_label)
            self._brush_spin = QSpinBox()
            self._brush_spin.setRange(1, 50)
            self._brush_spin.setValue(self._brush_radius)
            self._brush_spin.setSuffix(" vx")
            self._brush_spin.valueChanged.connect(self._on_brush_radius_changed)
            tool_row.addWidget(self._brush_spin)

            tool_row.addStretch(1)
            root.addLayout(tool_row)

            # Camera controls row — available while in Paint/Fill/Wand/Erase
            # so the user can re-frame without switching to Navigate. Hold
            # Space to temporarily orbit/pan with the mouse (released =
            # resume annotation).
            cam_row = QHBoxLayout()
            cam_row.setSpacing(6)
            cam_row.addWidget(QLabel("Camera:"))
            self._cam_reset_btn = QPushButton("Reset View")
            self._cam_reset_btn.setToolTip("Fit the volume to the view (R)")
            self._cam_reset_btn.clicked.connect(self._reset_view)
            cam_row.addWidget(self._cam_reset_btn)
            self._cam_zoom_in_btn = QPushButton("Zoom +")
            self._cam_zoom_in_btn.setToolTip("Zoom in (=)")
            self._cam_zoom_in_btn.clicked.connect(lambda: self._camera_zoom(0.8))
            cam_row.addWidget(self._cam_zoom_in_btn)
            self._cam_zoom_out_btn = QPushButton("Zoom −")
            self._cam_zoom_out_btn.setToolTip("Zoom out (-)")
            self._cam_zoom_out_btn.clicked.connect(lambda: self._camera_zoom(1.25))
            cam_row.addWidget(self._cam_zoom_out_btn)
            self._cam_rot_left_btn = QPushButton("⟲")
            self._cam_rot_left_btn.setToolTip("Rotate left 15° (Shift+Left)")
            self._cam_rot_left_btn.clicked.connect(lambda: self._camera_rotate(-15, 0))
            cam_row.addWidget(self._cam_rot_left_btn)
            self._cam_rot_right_btn = QPushButton("⟳")
            self._cam_rot_right_btn.setToolTip("Rotate right 15° (Shift+Right)")
            self._cam_rot_right_btn.clicked.connect(lambda: self._camera_rotate(15, 0))
            cam_row.addWidget(self._cam_rot_right_btn)
            self._cam_rot_up_btn = QPushButton("⤒")
            self._cam_rot_up_btn.setToolTip("Tilt up 15° (Shift+Up)")
            self._cam_rot_up_btn.clicked.connect(lambda: self._camera_rotate(0, -15))
            cam_row.addWidget(self._cam_rot_up_btn)
            self._cam_rot_down_btn = QPushButton("⤓")
            self._cam_rot_down_btn.setToolTip("Tilt down 15° (Shift+Down)")
            self._cam_rot_down_btn.clicked.connect(lambda: self._camera_rotate(0, 15))
            cam_row.addWidget(self._cam_rot_down_btn)
            cam_row.addStretch(1)
            # Cursor readout — shows voxel coordinates + intensity of the
            # brightest voxel along the ray under the cursor. Updates from
            # _on_canvas_mouse_move when not dragging, so the user can tell
            # what seed the Wand is about to use before they click.
            self._cursor_readout_label = QLabel("")
            self._cursor_readout_label.setStyleSheet(
                "color: #aaa; font-family: monospace;"
            )
            cam_row.addWidget(self._cursor_readout_label)
            root.addLayout(cam_row)

            # One-line hint under the tool row so it's obvious what the
            # selected tool will do. Updated by _on_tool_changed.
            self._tool_hint_label = QLabel()
            self._tool_hint_label.setStyleSheet(
                "color: #888; font-style: italic; padding-left: 36px;"
            )
            root.addWidget(self._tool_hint_label)

            # Set initial enabled/disabled + hint to match the default tool.
            self._apply_tool_ui_state()

            # Hook up annotation mouse handlers. Vispy SceneCanvas emits
            # mouse events independently of Qt's mousePressEvent so we attach
            # via its events system. Each handler short-circuits unless the
            # active tool needs it.
            self._canvas.events.mouse_press.connect(self._on_canvas_mouse_press)
            self._canvas.events.mouse_move.connect(self._on_canvas_mouse_move)
            self._canvas.events.mouse_release.connect(self._on_canvas_mouse_release)

            # Tool-switch keyboard shortcuts. Single-key when focus is in
            # the 3D panel; match the dropdown items so we keep one source
            # of truth. Space held = temporary Navigate (handled in
            # keyPressEvent / keyReleaseEvent below).
            self._install_tool_shortcuts()
            # Space is handled via keyPressEvent so we can detect release;
            # track the prior tool so we can restore it.
            self._space_prev_tool = None
            self.setFocusPolicy(Qt.StrongFocus)

            # Throttle overlay refreshes during paint/erase drags. Each
            # stamp marks ``_drag_dirty``; this timer coalesces dirty flags
            # into one rebuild at ~30 fps while a stroke is active. Without
            # this, large volumes rebuild + re-upload the labels Volume on
            # every mouse_move and the drag stutters into the hundreds of ms.
            self._drag_refresh_timer = QTimer(self)
            self._drag_refresh_timer.setInterval(33)
            self._drag_refresh_timer.timeout.connect(self._on_drag_refresh_tick)

        # Defer the first GPU upload by one event-loop tick so the panel
        # can lay out and paint before we block the thread with texture
        # uploads. Without this, embedding the panel into a QStackedWidget
        # freezes the main window for the duration of the upload (hundreds
        # of ms to several seconds) — long enough for the Wayland
        # compositor to paint the window blank and for clicks on other
        # docks to queue up and feel like a freeze.
        QTimer.singleShot(0, self._rebuild_volumes)

    def _sep(self):
        s = QFrame()
        s.setFrameShape(QFrame.VLine)
        s.setFixedWidth(1)
        s.setFixedHeight(22)
        s.setStyleSheet("color: #555;")
        return s

    def _short_gpu_label(self):
        """Compact GPU name for the toolbar badge. Full info is in the tooltip."""
        desc = (self._gpu_desc or "").lower()
        if "nvidia" in desc or "geforce" in desc or "quadro" in desc or "rtx" in desc:
            tag = "NVIDIA"
        elif "intel" in desc:
            tag = "Intel"
        elif "amd" in desc or "radeon" in desc:
            tag = "AMD"
        elif "apple" in desc or "metal" in desc:
            tag = "Apple"
        elif "llvmpipe" in desc or "software" in desc or "swr" in desc:
            tag = "Software"
        else:
            tag = "GPU"
        return f"{tag} · 3D ≤ {self._max_3d_dim}"

    def _tint_for(self, idx, preferred):
        if preferred:
            return preferred
        return _DEFAULT_TINTS[idx % len(_DEFAULT_TINTS)]

    def _pick_channel_tint(self, idx):
        """Open a colour picker for channel ``idx`` and apply the choice."""
        if not (0 <= idx < len(self._channels_raw)):
            return
        current = self._tint_for(idx, self._channels_raw[idx][2])
        initial = QColor(
            int(current[0] * 255), int(current[1] * 255), int(current[2] * 255)
        )
        color = QColorDialog.getColor(
            initial, self, "Channel Colour",
            options=QColorDialog.DontUseNativeDialog,
        )
        if not color.isValid():
            return
        rgb = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)
        name, vol, _old = self._channels_raw[idx]
        self._channels_raw[idx] = (name, vol, rgb)
        # Live-update the GPU colormap — vispy rebuilds the fragment shader.
        if 0 <= idx < len(self._volumes):
            try:
                self._volumes[idx].cmap = _tinted_colormap(rgb)
            except Exception:
                # If the visual can't accept a live cmap swap on this driver,
                # rebuild the volume stack so the new tint takes effect.
                self._rebuild_volumes()
        self._refresh_channel_tint_ui(idx, rgb)

    def _refresh_channel_tint_ui(self, idx, rgb):
        """Sync the swatch background and checkbox text colour to ``rgb``."""
        r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        if 0 <= idx < len(self._channel_swatches):
            self._channel_swatches[idx].setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #555;"
            )
        if 0 <= idx < len(self._channel_toggles):
            self._channel_toggles[idx].setStyleSheet(f"color: rgb({r},{g},{b});")

    def _on_mode_changed(self, text):
        method = next((m for m, lbl in self._mode_labels.items() if lbl == text), 'mip')
        for vol in self._volumes:
            vol.method = method

    def _on_downsample_changed(self, val):
        self._downsample_factor = int(val)
        self._rebuild_volumes()

    def _set_visible(self, idx, on):
        if 0 <= idx < len(self._volumes):
            self._volumes[idx].visible = on

    def _current_tool_label(self):
        """Return the label (e.g. ``"Navigate"``) of the currently checked tool."""
        for label, btn in self._tool_buttons.items():
            if btn.isChecked():
                return label
        return "Navigate"

    def _set_current_tool(self, label):
        """Check the button for ``label`` and run the tool-change handler.

        Safe to call with a label the panel doesn't know — unrecognised
        labels are no-ops so a stale keyboard binding can't crash us.
        """
        btn = self._tool_buttons.get(label)
        if btn is None:
            return
        btn.setChecked(True)
        self._on_tool_changed(label)

    def _on_tool_changed(self, text):
        self._tool_mode = text.lower()
        # Disable camera interaction in any annotation mode so left-drag
        # doesn't rotate while the user is trying to click or paint.
        if self._view is not None and self._view.camera is not None:
            self._view.camera.interactive = (self._tool_mode == 'navigate')
        self._apply_tool_ui_state()
        # A tool switch mid-drag (via keyboard shortcut etc.) should abandon
        # the stroke cleanly so we don't paint into the new mode's state.
        # rollback=True drops a freshly-reserved paint id (and its voxels)
        # so it doesn't appear in the LayerPanel as an orphan.
        if self._drag_active:
            self._finish_drag(emit=False, rollback=True)

    _TOOL_HINTS = {
        'navigate': "Navigate (V) — left-drag rotates, right-drag pans, "
                    "scroll zooms. Hold Space in any tool for the same.",
        'fill': "Fill (F) — click a labelled region to recolor its connected "
                "component into the selected ROI. Hold Space to rotate.",
        'wand': "Wand (W) — click a dendrite to flood fill by image intensity. "
                "Tune Tolerance; Hold Space to rotate without switching tools.",
        'paint': "Paint (P) — sphere brush writes into the selected ROI. "
                 "Hold Space to rotate; set size with Brush Size.",
        'erase': "Erase (E) — sphere brush clears voxels from all ROIs. "
                 "Hold Space to rotate.",
    }

    # Per-tool cursor. Uses Qt stock cursors so the mapping is consistent
    # with platform conventions (open-hand = "grab to move the scene",
    # pointing-hand = "click this", cross = "precision point"). Paint and
    # Erase share the crosshair — the tool hint plus the Brush Size readout
    # distinguish them without needing a custom pixmap.
    _TOOL_CURSORS = {
        'navigate': Qt.OpenHandCursor,
        'fill': Qt.PointingHandCursor,
        'wand': Qt.CrossCursor,
        'paint': Qt.CrossCursor,
        'erase': Qt.CrossCursor,
    }

    def _apply_tool_ui_state(self):
        """Enable/disable per-tool controls and refresh the hint label.

        Called on init and whenever the tool changes. Keeping it in one
        place means the enabled state is always consistent with the hint.
        """
        wand_on = self._tool_mode == 'wand'
        brush_on = self._tool_mode in ('paint', 'erase')
        # Wand-only controls
        self._wand_channel_label.setEnabled(wand_on)
        self._fill_channel_combo.setEnabled(wand_on)
        self._wand_tol_label_prefix.setEnabled(wand_on)
        self._tol_slider.setEnabled(wand_on)
        self._tol_label.setEnabled(wand_on)
        # Brush-only controls
        self._brush_size_label.setEnabled(brush_on)
        self._brush_spin.setEnabled(brush_on)
        # One-line hint
        self._tool_hint_label.setText(self._TOOL_HINTS.get(self._tool_mode, ""))
        # Tool cursor — set here (not in _on_tool_changed) so the initial
        # panel state picks up the right cursor too, not just tool switches.
        cursor = self._TOOL_CURSORS.get(self._tool_mode)
        if self._canvas is not None:
            if cursor is not None:
                self._canvas.native.setCursor(cursor)
            else:
                self._canvas.native.unsetCursor()

    def _on_fill_channel_changed(self, idx):
        self._fill_channel_idx = int(idx)

    def _on_tolerance_changed(self, val):
        self._fill_tolerance = int(val)
        self._tol_label.setText(str(val))

    def _on_brush_radius_changed(self, val):
        self._brush_radius = int(val)

    def _on_labels_visible_toggled(self, on):
        self._labels_visible = bool(on)
        if self._labels_volume is not None:
            self._labels_volume.visible = self._labels_visible

    def _on_labels_opacity_changed(self, val):
        self._labels_opacity = val / 100.0
        # Rebuilding is the cheapest way to restyle the Colormap's alpha
        # channel; labels volumes are much smaller than intensity data so
        # the upload cost is negligible.
        self._rebuild_labels_overlay()

    def refresh_labels(self):
        """Public hook used by paint/fill tools after they mutate labels_3d."""
        self._rebuild_labels_overlay()

    def show_progress(self):
        """Show the indeterminate progress bar and flush paint events so it
        renders before the caller blocks on a long operation (pool + GL
        upload, TIFF import). ``ExcludeUserInputEvents`` is critical:
        without it, a rapid-fire user click reenters through Qt's queued
        events and kicks off a nested ``_rebuild_labels_overlay`` while
        the outer one still holds stale ``labels`` / ``max_id`` locals —
        the outer call then overwrites the freshly-uploaded state on
        return. This excludes mouse/keyboard, so paint+expose flush but
        no new work is started. Cheap to call speculatively — callers
        without heavy work to do can skip this and the bar never shows.
        """
        from PySide6.QtCore import QEventLoop
        self._progress_bar.setRange(0, 0)  # indeterminate / marquee
        self._progress_bar.show()
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

    def hide_progress(self):
        """Hide the progress bar."""
        self._progress_bar.hide()

    def _on_canvas_mouse_press(self, event):
        """Route mouse press to the active annotation tool.

        In Navigate mode this is a no-op — the camera's own bindings
        handle rotation/pan/zoom. In Fill mode a left-click runs a
        ray-pick → seed → flood fill. In Paint/Erase a left-press starts
        a stroke: reserve a new label id (paint only), stamp the first
        sphere immediately, then keep stamping on every mouse_move until
        mouse_release.
        """
        if event.button != 1:  # only left button starts an annotation
            return
        if self._primary_doc is None or not self._volumes:
            return
        if self._tool_mode == 'fill':
            seed = self._ray_pick_seed(event.pos)
            if seed is None:
                return
            self._run_fill(seed)
            return
        if self._tool_mode == 'wand':
            seed = self._ray_pick_seed(event.pos)
            if seed is None:
                return
            self._run_wand(seed)
            return
        if self._tool_mode in ('paint', 'erase'):
            # Paint/erase are allowed to start anywhere, including over
            # empty voxels — users expect to pre-seed an empty region or
            # clean up a stray voxel. The ray pick still needs a valid
            # intersection with the volume's AABB though.
            seed = self._ray_pick_seed(event.pos, require_bright=False)
            if seed is None:
                return
            doc = self._primary_doc
            if doc.labels_3d is None:
                doc.ensure_labels_3d()
            # Paint requires a LayerPanel-selected 3D ROI. Painting with
            # nothing selected used to silently auto-reserve a label id,
            # which surprised users — strokes spawned new rows in the
            # sidebar instead of extending the ROI they meant to edit.
            # Mirrors the Fill behavior: ask the user to Add/select an
            # ROI first.
            if self._tool_mode == 'paint':
                active = self._active_volume_roi_id
                if active is None or active not in doc.labels_meta:
                    QMessageBox.information(
                        self, "No 3D ROI selected",
                        "Paint needs a 3D ROI to write into. Click '+' in "
                        "the Layers panel to add one (or select an "
                        "existing 3D ROI), then try again.",
                    )
                    return
            self._drag_active = True
            self._drag_mode = self._tool_mode
            self._drag_extends_existing = False
            if self._tool_mode == 'paint':
                # Extend the LayerPanel-selected 3D ROI. ``active`` was
                # validated above, so it's always present in labels_meta.
                self._drag_label_id = int(self._active_volume_roi_id)
                self._drag_extends_existing = True
                if self._drag_label_id > np.iinfo(doc.labels_3d.dtype).max:
                    doc.promote_labels_dtype(np.uint16)
            else:
                self._drag_label_id = 0  # erase writes background
            self._stamp_brush(seed)
            self._drag_refresh_timer.start()

    def _on_canvas_mouse_move(self, event):
        """Continue an in-progress paint/erase stroke; update cursor readout."""
        if self._drag_active:
            if self._primary_doc is None or not self._volumes:
                return
            seed = self._ray_pick_seed(event.pos, require_bright=False)
            if seed is None:
                return
            self._stamp_brush(seed)
            return
        # Not dragging — refresh the voxel-under-cursor readout so the user
        # can see what seed the Wand (or Fill/Paint) is about to hit.
        self._update_cursor_readout(event.pos)

    def _update_cursor_readout(self, pos):
        """Show (z, y, x) + active-channel intensity for the voxel under ``pos``."""
        if not self._volumes or self._primary_doc is None:
            self._cursor_readout_label.setText("")
            return
        seed = self._ray_pick_seed(pos, require_bright=False)
        if seed is None:
            self._cursor_readout_label.setText("")
            return
        z, y, x = seed
        vol = self._active_channel_volume()
        val = int(vol[z, y, x]) if vol is not None else None
        if val is None:
            self._cursor_readout_label.setText(f"(z={z}, y={y}, x={x})")
        else:
            self._cursor_readout_label.setText(
                f"(z={z}, y={y}, x={x})  intensity={val}"
            )

    def _on_canvas_mouse_release(self, event):
        """Finalize the current stroke on left-release."""
        if not self._drag_active:
            return
        if event.button != 1:
            return
        # Only emit label_added for a freshly-reserved paint id. Extending an
        # existing ROI reuses an id that already has a VolumeROILayer wrapper,
        # so emitting again would create a duplicate row in the LayerPanel.
        emit = (self._drag_mode == 'paint'
                and self._drag_label_id
                and not self._drag_extends_existing)
        self._finish_drag(emit=bool(emit))

    def _finish_drag(self, emit, rollback=False):
        """Tear down drag state and refresh overlays once.

        ``emit`` is True for a completed paint stroke that allocated a new
        id (signals a new ROI to MontarisApp). False for erase strokes,
        existing-ROI extensions, and aborted drags.

        ``rollback`` is True when the stroke was aborted mid-flight (e.g.,
        tool switched) and we need to drop a freshly-reserved paint id so
        it doesn't leak into the LayerPanel as an orphan.
        """
        self._drag_refresh_timer.stop()
        doc = self._primary_doc
        lid = self._drag_label_id
        mode = self._drag_mode
        extended = self._drag_extends_existing
        stroke_bbox = self._stroke_bbox
        stroke_before = self._stroke_before
        self._drag_active = False
        self._drag_mode = None
        self._drag_label_id = None
        self._drag_extends_existing = False
        self._drag_dirty = False
        self._dirty_bbox = None
        self._stroke_bbox = None
        self._stroke_before = None
        self._stroke_dtype = None
        # Rollback only applies to freshly-reserved paint ids — erase writes
        # 0 and extensions reuse an id owned by an existing VolumeROILayer.
        if (rollback and doc is not None and mode == 'paint'
                and lid and not extended):
            doc.release_label_id(int(lid))
            self.refresh_labels()
            return
        self.refresh_labels()
        # Emit label_added FIRST so MontarisApp can create the VolumeROILayer
        # wrapper synchronously. We then bundle the wrapper's Add undo with
        # the stroke's voxel undo into one Ctrl+Z.
        if emit and doc is not None and lid:
            self.label_added.emit(doc, lid)
        if (doc is not None and stroke_bbox is not None
                and stroke_before is not None
                and doc.labels_3d is not None):
            from montaris.core.undo import (
                VolumeStrokeUndoCommand,
                AddVolumeROIUndoCommand,
            )
            from montaris.core.multi_undo import CompoundUndoCommand
            z0, z1, y0, y1, x0, x1 = stroke_bbox
            after = np.ascontiguousarray(doc.labels_3d[z0:z1, y0:y1, x0:x1])
            stroke_cmd = VolumeStrokeUndoCommand(
                doc, stroke_bbox, stroke_before, after,
            )
            if emit and lid and not extended:
                wrapper = self._find_wrapper_for(doc, int(lid))
                if wrapper is None:
                    self.undo_pushed.emit(stroke_cmd)
                else:
                    add_cmd = AddVolumeROIUndoCommand(
                        self._layer_stack(), doc, int(lid), wrapper,
                    )
                    self.undo_pushed.emit(
                        CompoundUndoCommand([add_cmd, stroke_cmd]),
                    )
            else:
                self.undo_pushed.emit(stroke_cmd)

    def _layer_stack(self):
        """Return the host app's LayerStack, or None if not hosted."""
        win = self.window()
        return getattr(win, 'layer_stack', None) if win is not None else None

    def _find_wrapper_for(self, doc, lid):
        """Return the VolumeROILayer wrapping ``lid`` in this doc, else None."""
        ls = self._layer_stack()
        if ls is None:
            return None
        for roi in ls.roi_layers:
            if (getattr(roi, 'is_volume', False)
                    and getattr(roi, '_doc', None) is doc
                    and getattr(roi, '_label_id', None) == lid):
                return roi
        return None

    def set_active_volume_roi_id(self, lid):
        """Tell the panel which 3D ROI id a subsequent paint stroke should
        extend. ``None`` clears it so the next stroke allocates a new id.
        """
        self._active_volume_roi_id = int(lid) if lid else None

    def _on_drag_refresh_tick(self):
        """Coalesced overlay repaint while a stroke is in progress.

        Does a sub-region texture upload when possible (existing labels
        visual + no downsample) instead of rebuilding the Volume visual
        from scratch. Full rebuild = seconds on 50M-voxel stacks because
        it recreates the entire 3D texture; sub-region upload is ms.
        """
        if not self._drag_dirty:
            return
        self._drag_dirty = False
        bbox = self._dirty_bbox
        self._dirty_bbox = None
        doc = self._primary_doc
        labels = doc.labels_3d if doc is not None else None
        tex_lut = self._labels_texture_lut
        # Fast path: live visual, cached LUT, no downsample, bbox
        # available. Upload just the touched subvolume via
        # Texture3D.set_data(offset=...). The texture stores DENSE
        # palette slots — raw ``labels`` values must be remapped through
        # ``tex_lut`` before upload (mirrors the dense-palette upload
        # in ``_do_rebuild_labels_overlay``); writing raw ids would
        # sample wrong cmap entries with ``clim=(0, palette_size)``.
        if (self._labels_volume is not None
                and labels is not None
                and tex_lut is not None
                and bbox is not None
                and tuple(int(a) for a in self._last_total_ds_axes) == (1, 1, 1)):
            z0, z1, y0, y1, x0, x1 = bbox
            raw_sub = labels[z0:z1, y0:y1, x0:x1]
            # Stale-id defence: out-of-range raw ids → slot 0
            # (background). Mask via ``np.where`` rather than extending
            # the LUT — a stale uint32 id (partial sidecars) would
            # otherwise drive a multi-GB ``np.zeros`` allocation here.
            if raw_sub.size and int(raw_sub.max()) >= tex_lut.size:
                raw_sub = np.where(raw_sub < tex_lut.size, raw_sub, 0)
            sub = np.ascontiguousarray(tex_lut[raw_sub])
            try:
                # vispy's Volume data is (Z, Y, X); Texture3D.set_data's
                # offset follows the same axis order as the stored data.
                self._labels_volume._texture.set_data(
                    sub, offset=(z0, y0, x0),
                )
                self._labels_volume.update()
                return
            except Exception:
                # Fallthrough to full rebuild on any upload failure.
                pass
        self._rebuild_labels_overlay()

    def _extend_stroke_before(self, labels_3d, stamp_bbox):
        """Grow ``_stroke_before`` to cover ``stamp_bbox`` with pre-stroke data.

        Called from ``_stamp_brush`` before any write. When the bbox is
        already large enough, no-op — the snapshot already holds the pre-
        stroke values for every voxel the stamp is about to touch. When
        the bbox expands, the newly-included region is seeded from
        ``labels_3d`` (still pre-stroke there) and the overlap region is
        copied from the old snapshot (which holds the original values).
        """
        nz0, nz1, ny0, ny1, nx0, nx1 = stamp_bbox
        if self._stroke_bbox is None:
            self._stroke_bbox = (nz0, nz1, ny0, ny1, nx0, nx1)
            # .copy() (not ascontiguousarray) — a trivially-sized slice can
            # alias the parent buffer, so later writes to labels_3d would
            # leak into the snapshot and break undo.
            self._stroke_before = labels_3d[nz0:nz1, ny0:ny1, nx0:nx1].copy()
            self._stroke_dtype = labels_3d.dtype
            return
        oz0, oz1, oy0, oy1, ox0, ox1 = self._stroke_bbox
        # No expansion needed — stamp fits inside the existing stroke bbox.
        if nz0 >= oz0 and nz1 <= oz1 and ny0 >= oy0 and ny1 <= oy1 \
                and nx0 >= ox0 and nx1 <= ox1:
            return
        uz0, uz1 = min(oz0, nz0), max(oz1, nz1)
        uy0, uy1 = min(oy0, ny0), max(oy1, ny1)
        ux0, ux1 = min(ox0, nx0), max(ox1, nx1)
        shape = (uz1 - uz0, uy1 - uy0, ux1 - ux0)
        # Seed every voxel from labels_3d — the regions outside the old
        # stroke bbox are still pre-stroke there. .copy() again — we may
        # be writing into this snapshot after further stamps, and we
        # cannot alias the live labels volume.
        new_before = labels_3d[uz0:uz1, uy0:uy1, ux0:ux1].copy()
        # Overwrite the overlap region with the saved pre-stroke values so
        # voxels the stroke has already mutated retain their original state.
        dz0, dz1 = oz0 - uz0, oz1 - uz0
        dy0, dy1 = oy0 - uy0, oy1 - uy0
        dx0, dx1 = ox0 - ux0, ox1 - ux0
        new_before[dz0:dz1, dy0:dy1, dx0:dx1] = self._stroke_before
        self._stroke_before = new_before
        self._stroke_bbox = (uz0, uz1, uy0, uy1, ux0, ux1)

    def _stamp_brush(self, seed_zyx):
        """Stamp a filled sphere of radius ``self._brush_radius`` into
        ``labels_3d`` centered at ``seed_zyx``. Paint uses the drag's
        reserved label id; erase writes 0 (background). Marks
        ``_drag_dirty`` so the throttle timer issues a single sub-region
        upload at the next tick, and accumulates ``_dirty_bbox``.
        """
        doc = self._primary_doc
        if doc is None or doc.labels_3d is None:
            return
        r = self._brush_radius
        cz, cy, cx = seed_zyx
        Z, Y, X = doc.labels_3d.shape
        z0, z1 = max(0, cz - r), min(Z, cz + r + 1)
        y0, y1 = max(0, cy - r), min(Y, cy + r + 1)
        x0, x1 = max(0, cx - r), min(X, cx + r + 1)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            return
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        inside = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        val = int(self._drag_label_id) if self._drag_mode == 'paint' else 0
        # Snapshot pre-stroke voxels the first time each region is touched,
        # BEFORE we write into labels_3d. The snapshot grows as the stroke
        # extends its bbox so a later Ctrl+Z can restore the full stroke.
        self._extend_stroke_before(doc.labels_3d, (z0, z1, y0, y1, x0, x1))
        region = doc.labels_3d[z0:z1, y0:y1, x0:x1]
        region[inside] = val
        # Grow the dirty bbox so the tick uploads the smallest subvolume
        # that covers everything since the last refresh.
        if self._dirty_bbox is None:
            self._dirty_bbox = (z0, z1, y0, y1, x0, x1)
        else:
            pz0, pz1, py0, py1, px0, px1 = self._dirty_bbox
            self._dirty_bbox = (
                min(pz0, z0), max(pz1, z1),
                min(py0, y0), max(py1, y1),
                min(px0, x0), max(px1, x1),
            )
        self._drag_dirty = True

    def _ray_pick_seed(self, pos, require_bright=True):
        """Cast a ray through canvas pixel ``pos`` and return the brightest
        voxel along it, as ``(z, y, x)`` indices into the active channel.

        Returns ``None`` if the ray misses the volume. When ``require_bright``
        is True (Fill mode) a ray whose brightest sample is 0 also returns
        ``None`` — there's nothing to seed from. Paint/erase pass False so
        strokes can start over empty regions (pre-seeding an ROI or cleaning
        up a stray voxel).
        """
        if self._view is None or self._view.camera is None:
            return None
        volume = self._active_channel_volume()
        if volume is None:
            return None
        # Build a ray in world space via the viewbox's scene transform.
        try:
            tr = self._view.scene.node_transform(self._view)
            p_near = tr.imap(np.array([pos[0], pos[1], -1, 1], dtype=float))
            p_far = tr.imap(np.array([pos[0], pos[1], 1, 1], dtype=float))
            w_n = p_near[3] if p_near[3] else 1.0
            w_f = p_far[3] if p_far[3] else 1.0
            o = p_near[:3] / w_n
            f = p_far[:3] / w_f
        except Exception:
            return None
        d = f - o
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-9:
            return None
        d = d / nrm

        # Vispy's Volume visual places voxel (z, y, x) at world coords
        # (x, y, z) — x and z get swapped on the way in. Sample world-space
        # points along the ray, round to voxel indices on the fly.
        Z, Y, X = volume.shape
        # AABB in world space for Volume defaults: origin=(0,0,0), extent=
        # (X, Y, Z). Compute ray vs AABB slab intersection to get t bounds.
        t_enter, t_exit = _ray_aabb((0.0, 0.0, 0.0), (float(X), float(Y), float(Z)), o, d)
        if t_exit <= t_enter:
            return None
        # March in ~half-voxel steps for seed-finding. Half-voxel is fine
        # even at aggressive tilts; smaller costs more, larger risks
        # skipping over thin bright structures.
        step = 0.5
        n_samples = max(2, int((t_exit - t_enter) / step) + 1)
        ts = np.linspace(t_enter, t_exit, n_samples)
        pts = o + np.outer(ts, d)  # (n, 3), world xyz
        ix = np.clip(pts[:, 0].astype(np.int64), 0, X - 1)
        iy = np.clip(pts[:, 1].astype(np.int64), 0, Y - 1)
        iz = np.clip(pts[:, 2].astype(np.int64), 0, Z - 1)
        samples = volume[iz, iy, ix]
        best = int(np.argmax(samples))
        if require_bright and samples[best] <= 0:
            return None
        return (int(iz[best]), int(iy[best]), int(ix[best]))

    def _active_channel_volume(self):
        """Return the raw ndarray for the channel chosen in the Fill combo.

        Uses the un-downsampled original from ``_channels_raw`` so flood
        fills operate at full resolution regardless of quality slider.
        """
        idx = self._fill_channel_idx
        if not (0 <= idx < len(self._channels_raw)):
            return None
        _, vol, _ = self._channels_raw[idx]
        return None if vol is None else np.asarray(vol)

    def _run_fill(self, seed_zyx):
        """napari-style label-flood from ``seed_zyx``.

        Reads the label id under the seed (``old_label``) and replaces its
        connected component in ``labels_3d`` with the target id. Target is
        the LayerPanel-selected 3D ROI when one is active, else a fresh id.

        This is the napari model: Fill operates on ``labels_3d``, not on
        image intensity. It relabels already-painted regions — it does not
        segment raw voxels. Users who need initial segmentation should use
        Paint first (or a future intensity-wand tool).
        """
        doc = self._primary_doc
        if doc is None:
            return
        if doc.labels_3d is None:
            doc.ensure_labels_3d()

        old_label = int(doc.labels_3d[tuple(seed_zyx)])

        active = self._active_volume_roi_id
        if active is not None and active in doc.labels_meta:
            new_label = int(active)
            extends_existing = True
        else:
            # napari allows fill-on-background with `selected_label` set; we
            # refuse the no-selection + background combo because it would
            # relabel the (huge) background connected component into a
            # brand-new ROI. Guide the user instead.
            if old_label == 0:
                QMessageBox.information(
                    self, "Nothing to fill",
                    "Fill extends an existing 3D ROI into its connected "
                    "component. Paint a region first, or select a 3D ROI "
                    "in the Layers panel before clicking.",
                )
                return
            new_label = doc.reserve_label_id()
            extends_existing = False

        # napari early-out: clicking on a voxel already of the target id.
        if old_label == new_label:
            return

        # Connected component of `old_label` containing the seed. scipy's
        # default structure is 6-connectivity on 3D — same as napari.
        from scipy.ndimage import label as cc_label
        matches = (doc.labels_3d == old_label)
        components, _ = cc_label(matches)
        comp_id = int(components[tuple(seed_zyx)])
        if comp_id == 0:
            return  # impossible — seed matches by construction
        mask = (components == comp_id)

        # Runaway guard for the "fill the background by accident" footgun:
        # if the connected component is half the volume, it's almost
        # certainly background. Refuse and roll back an unused reservation.
        frac = mask.sum() / mask.size
        if frac >= 0.5:
            QMessageBox.warning(
                self, "Fill aborted",
                f"Connected region covers {frac*100:.0f}% of the volume — "
                f"probably seeded on background. Refusing.",
            )
            if not extends_existing:
                doc.release_label_id(new_label)
            return

        if new_label > np.iinfo(doc.labels_3d.dtype).max:
            doc.promote_labels_dtype(np.uint16)
        doc.labels_3d[mask] = new_label

        self.refresh_labels()
        if not extends_existing:
            self.label_added.emit(doc, new_label)

        # Compact undo entry for the fill: bit-packed component mask + two
        # label ids. Bundled with an Add undo when this reserved a fresh id
        # so Ctrl+Z reverses both the new ROI and its voxels in one step.
        from montaris.core.undo import (
            VolumeFillUndoCommand,
            AddVolumeROIUndoCommand,
        )
        from montaris.core.multi_undo import CompoundUndoCommand
        zs, ys, xs = np.where(mask)
        if zs.size == 0:
            return
        bbox = (int(zs.min()), int(zs.max()) + 1,
                int(ys.min()), int(ys.max()) + 1,
                int(xs.min()), int(xs.max()) + 1)
        z0, z1, y0, y1, x0, x1 = bbox
        mask_crop = mask[z0:z1, y0:y1, x0:x1]
        fill_cmd = VolumeFillUndoCommand(doc, bbox, mask_crop, old_label, new_label)
        if extends_existing:
            self.undo_pushed.emit(fill_cmd)
        else:
            wrapper = self._find_wrapper_for(doc, int(new_label))
            if wrapper is None:
                self.undo_pushed.emit(fill_cmd)
            else:
                add_cmd = AddVolumeROIUndoCommand(
                    self._layer_stack(), doc, int(new_label), wrapper,
                )
                self.undo_pushed.emit(CompoundUndoCommand([add_cmd, fill_cmd]))

    def _run_wand(self, seed_zyx):
        """Intensity-based 3D flood fill from ``seed_zyx``.

        Grows a region through voxels whose intensity is within
        ``self._fill_tolerance`` of the seed value on the Wand Channel,
        then writes the selected ROI's label id into those voxels —
        but only where they are currently background (label 0). Other
        ROIs are left untouched.

        The tolerance is **asymmetric**: voxels as dim as ``seed - tol``
        are accepted, while voxels brighter than the seed are accepted
        without a ceiling. Rationale: when the seed lands on the bright
        core of a dendrite (via ray-pick's brightest-along-ray), the
        boundaries fade to dimmer intensities. A symmetric window clips
        off those edges; a downward-only window reaches them without
        leaking upward into unrelated bright structures (which would
        have to be *connected and brighter* than the core — rare).
        """
        doc = self._primary_doc
        if doc is None:
            return
        if doc.labels_3d is None:
            doc.ensure_labels_3d()

        active = self._active_volume_roi_id
        if active is None or active not in doc.labels_meta:
            QMessageBox.information(
                self, "No 3D ROI selected",
                "Wand needs a 3D ROI to write into. Click '+' in the "
                "Layers panel to add one (or select an existing 3D ROI), "
                "then try again.",
            )
            return
        new_label = int(active)

        vol = self._active_channel_volume()
        if vol is None or vol.shape != doc.labels_3d.shape:
            return

        from skimage.segmentation import flood
        tol = max(0, int(self._fill_tolerance))
        seed_tuple = tuple(int(v) for v in seed_zyx)
        seed_val = int(vol[seed_tuple])
        # Clip values above seed down to seed — this turns skimage's
        # symmetric |val - seed| <= tol into a one-sided "val >= seed - tol"
        # without allocating a full uint16 copy of the volume.
        clipped = np.minimum(vol, seed_val)
        try:
            mask = flood(clipped, seed_tuple, tolerance=tol)
        except Exception:
            return

        # Don't clobber other ROIs — wand only claims background voxels.
        mask &= (doc.labels_3d == 0)

        vox = int(mask.sum())
        if vox == 0:
            return
        frac = vox / mask.size
        if frac >= 0.5:
            QMessageBox.warning(
                self, "Wand aborted",
                f"Intensity-matched region covers {frac*100:.0f}% of the "
                f"volume — Tolerance is probably too high. Lower it and "
                f"try again.",
            )
            return

        if new_label > np.iinfo(doc.labels_3d.dtype).max:
            doc.promote_labels_dtype(np.uint16)
        doc.labels_3d[mask] = new_label
        self.refresh_labels()

        from montaris.core.undo import VolumeFillUndoCommand
        zs, ys, xs = np.where(mask)
        bbox = (int(zs.min()), int(zs.max()) + 1,
                int(ys.min()), int(ys.max()) + 1,
                int(xs.min()), int(xs.max()) + 1)
        z0, z1, y0, y1, x0, x1 = bbox
        mask_crop = mask[z0:z1, y0:y1, x0:x1]
        self.undo_pushed.emit(
            VolumeFillUndoCommand(doc, bbox, mask_crop, 0, new_label)
        )

    def _reset_view(self):
        if self._volumes:
            self._view.camera.set_range()
            self._view.camera.view_changed()

    def _camera_zoom(self, factor):
        """Multiply the camera's scale_factor by ``factor`` (< 1 zooms in).

        Works regardless of tool mode — the camera's own mouse bindings
        would otherwise be disabled during Paint/Fill/Wand/Erase.
        """
        cam = self._view.camera if self._view is not None else None
        if cam is None or not hasattr(cam, '_scale_factor'):
            return
        cam._scale_factor = max(1e-6, float(cam._scale_factor) * float(factor))
        cam.view_changed()

    def _camera_rotate(self, d_azimuth, d_elevation):
        """Nudge the camera by ``d_azimuth``/``d_elevation`` degrees.

        Matches the discrete rotate buttons in the camera row. Works for
        both turntable-style cameras (azimuth/elevation) and arcball-style
        (quaternion rotation around scene axes).
        """
        cam = self._view.camera if self._view is not None else None
        if cam is None:
            return
        try:
            if hasattr(cam, 'azimuth') and hasattr(cam, 'elevation'):
                cam.azimuth = float(cam.azimuth) + float(d_azimuth)
                cam.elevation = float(cam.elevation) + float(d_elevation)
                cam.view_changed()
                return
            # Arcball: compose a rotation quaternion around world Y (azimuth)
            # and world X (elevation), multiply into the existing orientation.
            if hasattr(cam, '_quaternion'):
                import math
                from vispy.util.quaternion import Quaternion
                q = cam._quaternion
                if d_azimuth:
                    q = Quaternion.create_from_axis_angle(
                        math.radians(float(d_azimuth)), 0.0, 1.0, 0.0
                    ) * q
                if d_elevation:
                    q = Quaternion.create_from_axis_angle(
                        math.radians(float(d_elevation)), 1.0, 0.0, 0.0
                    ) * q
                cam._quaternion = q
                cam.view_changed()
        except Exception:
            pass

    def _install_tool_shortcuts(self):
        """Wire single-key shortcuts for tool switching (V/F/W/P/E/R/=/-)."""
        mapping = [
            ('v', "Navigate"),
            ('f', "Fill"),
            ('w', "Wand"),
            ('p', "Paint"),
            ('e', "Erase"),
        ]
        for key, label in mapping:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(
                lambda lbl=label: self._set_current_tool(lbl)
            )
        # Discrete camera shortcuts
        sc_reset = QShortcut(QKeySequence("r"), self)
        sc_reset.setContext(Qt.WidgetWithChildrenShortcut)
        sc_reset.activated.connect(self._reset_view)
        sc_in = QShortcut(QKeySequence("="), self)
        sc_in.setContext(Qt.WidgetWithChildrenShortcut)
        sc_in.activated.connect(lambda: self._camera_zoom(0.8))
        sc_out = QShortcut(QKeySequence("-"), self)
        sc_out.setContext(Qt.WidgetWithChildrenShortcut)
        sc_out.activated.connect(lambda: self._camera_zoom(1.25))
        sc_rot_l = QShortcut(QKeySequence("Shift+Left"), self)
        sc_rot_l.setContext(Qt.WidgetWithChildrenShortcut)
        sc_rot_l.activated.connect(lambda: self._camera_rotate(-15, 0))
        sc_rot_r = QShortcut(QKeySequence("Shift+Right"), self)
        sc_rot_r.setContext(Qt.WidgetWithChildrenShortcut)
        sc_rot_r.activated.connect(lambda: self._camera_rotate(15, 0))
        sc_rot_u = QShortcut(QKeySequence("Shift+Up"), self)
        sc_rot_u.setContext(Qt.WidgetWithChildrenShortcut)
        sc_rot_u.activated.connect(lambda: self._camera_rotate(0, -15))
        sc_rot_d = QShortcut(QKeySequence("Shift+Down"), self)
        sc_rot_d.setContext(Qt.WidgetWithChildrenShortcut)
        sc_rot_d.activated.connect(lambda: self._camera_rotate(0, 15))

    def keyPressEvent(self, event):
        """Space = temporary Navigate while held.

        Snapshots the current tool on keydown, switches to Navigate so
        the camera's mouse bindings come alive, and the release handler
        restores the prior tool. Autorepeat is ignored so repeated
        keydowns during a long hold don't overwrite the snapshot.
        """
        if (event.key() == Qt.Key_Space
                and not event.isAutoRepeat()
                and self._tool_mode != 'navigate'):
            self._space_prev_tool = self._current_tool_label()
            self._set_current_tool("Navigate")
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if (event.key() == Qt.Key_Space
                and not event.isAutoRepeat()
                and self._space_prev_tool is not None):
            self._set_current_tool(self._space_prev_tool)
            self._space_prev_tool = None
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _rebuild_volumes(self):
        # Remove any existing volumes
        for vol in self._volumes:
            vol.parent = None
        self._volumes = []

        label = self._mode_combo.currentText()
        method = next((m for m, lbl in self._mode_labels.items() if lbl == label), 'mip')

        # -- Pass 1: per-channel prep (user stride + per-axis dim cap) --
        # prepped holds per-channel (idx, tint, clim, arr, per_axis_gpu_factors).
        # Per-axis factors matter for non-cubic stacks: a (500, 3000, 4097)
        # stack only needs striding on Y/X, so collapsing to the scalar max
        # would also downsample Z and misalign overlays against the world.
        prepped = []
        max_gpu_axes = [1, 1, 1]
        for idx, (name, volume, tint) in enumerate(self._channels_raw):
            if volume is None:
                continue
            arr = np.asarray(volume)
            clim = _percentile_clim(arr)
            arr = _downsample(arr, self._downsample_factor)
            arr, gpu_factors = _fit_to_gpu(arr, self._max_3d_dim)
            max_gpu_axes = [max(a, b) for a, b in zip(max_gpu_axes, gpu_factors)]
            prepped.append((idx, tint, clim, arr, tuple(gpu_factors)))

        # -- Pass 2: budget fit across all channels together --
        shapes = [p[3].shape for p in prepped]
        budget_factor = _fit_to_budget(shapes, self._vram_budget, bytes_per_voxel=1)
        # OOM retries only add stride — never relax it — until the upload succeeds.
        total_extra = budget_factor * self._oom_retry_factor
        if total_extra > 1:
            prepped = [(idx, tint, clim, _downsample(arr, total_extra), gf)
                       for idx, tint, clim, arr, gf in prepped]

        # -- Pass 3: upload, catching OOM so we can retry with more stride --
        from vispy.visuals.transforms import STTransform
        budget_total = budget_factor * self._oom_retry_factor
        try:
            for idx, tint, clim, arr, gpu_ax in prepped:
                arr = _to_uint8(arr, clim)
                rgb = self._tint_for(idx, tint)
                cmap = _tinted_colormap(rgb)
                vol = scene.visuals.Volume(
                    arr,
                    parent=self._view.scene,
                    method=method,
                    cmap=cmap,
                    clim=(0, 255),
                )
                # Scale back to full-resolution world coordinates, per axis,
                # so ray-picks, labels overlays, and paint tools (all of
                # which work in full-res voxel space) line up with what the
                # user sees. Array order is (Z, Y, X) but vispy's STTransform
                # takes scale in (X, Y, Z) order.
                dz = self._downsample_factor * gpu_ax[0] * budget_total
                dy = self._downsample_factor * gpu_ax[1] * budget_total
                dx = self._downsample_factor * gpu_ax[2] * budget_total
                if (dz, dy, dx) != (1, 1, 1):
                    vol.transform = STTransform(scale=(dx, dy, dz))
                if idx < len(self._channel_toggles):
                    vol.visible = self._channel_toggles[idx].isChecked()
                self._volumes.append(vol)
        except (RuntimeError, MemoryError) as e:
            # Likely GL_OUT_OF_MEMORY. Halve once more (cap at 8× so we don't
            # silently grind down to 1×1×1 on a broken driver) and recurse.
            msg = str(e).lower()
            looks_like_oom = ("memory" in msg or "out of" in msg or "gl_out_of_memory" in msg
                             or self._oom_retry_factor < 8)
            if looks_like_oom and self._oom_retry_factor < 8:
                for v in self._volumes:
                    v.parent = None
                self._volumes = []
                self._oom_retry_factor *= 2
                self._rebuild_volumes()
                return
            raise

        if self._volumes:
            self._view.camera.set_range()

        # Label: slider × GPU-cap × budget × OOM-retry, with a reason tag.
        # Per-axis total used by the labels overlay; display uses max(axes).
        total_axes = tuple(
            self._downsample_factor * gpu_ax * budget_factor * self._oom_retry_factor
            for gpu_ax in max_gpu_axes
        )
        total = max(total_axes)
        reasons = []
        if any(a > 1 for a in max_gpu_axes):
            reasons.append("GPU")
        if budget_factor > 1 or self._oom_retry_factor > 1:
            reasons.append("memory")
        self._last_reason = "+".join(reasons)
        self._last_total_ds_axes = tuple(int(a) for a in total_axes)
        self._last_total_ds = int(total)
        if total == 1:
            self._ds_label.setText("full")
        elif reasons:
            self._ds_label.setText(f"1/{total} ({self._last_reason})")
        else:
            self._ds_label.setText(f"1/{total}")

        # Labels overlay is rebuilt in lock-step with intensity volumes so
        # it picks up the same downsample stride.
        self._rebuild_labels_overlay()

    def _build_labels_colormap_and_lut(self, labels_meta, max_id, dtype=None):
        """Build (texture_lut, dense_cmap, palette_size).

        ``texture_lut[raw_id]`` → dense palette index in ``[0, palette_size]``.
        Slot 0 is transparent (background). Visible labels occupy slots
        1..palette_size, sorted by raw id. Hidden / opacity=0 labels map
        to slot 0 so they render transparent without consuming a palette
        slot.

        Why we need this: vispy's ``Colormap`` is uploaded to a fixed
        1024-pixel 1D GPU LUT (``LUT_len = 1024``). When raw label ids
        span a wide sparse range (e.g. neuron skeletons with ids
        0..663), vispy resamples the user-supplied colormap from N
        entries to 1024 — and for any N that doesn't divide 1024
        evenly, voxel-value → texel mapping shifts by ``value/N``. A
        voxel with id 5 ends up sampling the texel that contains color
        slot 4, etc., scrambling the rendering and dropping voxels
        whose mapped slot happens to be alpha=0. napari avoids this in
        ``_accelerated_cmap.zero_preserving_modulo`` by pre-mapping
        labels to a small dense palette before GPU upload; we do the
        same here, but keyed by the actual ROI ids so each ROI keeps
        its user-assigned colour.

        ``dtype`` is accepted for back-compat but ignored — the LUT
        output dtype is chosen below to match ``palette_size``, since
        the dense slots fit dramatically smaller than raw ids (e.g.
        a 195-label sparse uint16 file remaps to uint8, halving the
        GPU upload bytes).
        """
        del dtype  # see docstring
        sorted_ids = sorted(int(lid) for lid in labels_meta.keys()
                            if int(lid) > 0 and int(lid) <= max_id)
        # Palette = visible-AND-non-transparent labels only; hidden ones
        # are filtered into slot 0 via the LUT.
        visible_ids: list[int] = []
        meta_for_id: dict[int, dict] = {}
        for lid in sorted_ids:
            meta = labels_meta[lid]
            hidden = (not meta.get('visible', True)
                      or int(meta.get('opacity', 128)) <= 0)
            if hidden:
                continue
            visible_ids.append(lid)
            meta_for_id[lid] = meta
        palette_size = len(visible_ids)

        # Output dtype = smallest unsigned int that fits ``palette_size``
        # (= the largest slot index ever stored in the LUT). Halves GPU
        # upload bytes when ``palette_size ≤ 255`` (the common case for
        # sparse uint16/uint32 label files: ~200 ROIs map to uint8).
        # ``texture_lut[labels]`` returns LUT dtype regardless of
        # ``labels.dtype``, so the eventual ``Texture3D`` upload uses
        # this narrower dtype too.
        if palette_size <= np.iinfo(np.uint8).max:
            out_dtype = np.uint8
        elif palette_size <= np.iinfo(np.uint16).max:
            out_dtype = np.uint16
        else:
            out_dtype = np.uint32

        lut_size = int(max_id) + 1
        texture_lut = np.zeros(lut_size, dtype=out_dtype)
        # ``+ 1`` for the background slot at index 0.
        cmap_colors = np.zeros((palette_size + 1, 4), dtype=np.float32)

        for slot, lid in enumerate(visible_ids, start=1):
            texture_lut[lid] = slot
            meta = meta_for_id[lid]
            r, g, b = meta.get('color', (255, 255, 255))
            if max(r, g, b) > 1.0:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            base_alpha = meta.get('opacity', 128) / 255.0
            cmap_colors[slot] = (r, g, b, base_alpha * self._labels_opacity)

        # ``+ 2`` controls for the (palette_size + 1)-entry cmap with
        # zero-order interpolation (one step boundary per entry).
        controls = np.linspace(0.0, 1.0, palette_size + 2, dtype=np.float32)
        cmap = Colormap(colors=cmap_colors, controls=controls,
                        interpolation='zero')
        return texture_lut, cmap, palette_size

    def _rebuild_labels_overlay(self):
        """(Re)create — or in-place update — the labels overlay visual.

        Full rebuild happens the first time, on dtype/shape changes
        (downsample slider, re-import with different volume), or when no
        labels exist. Visibility / opacity / colour changes reuse the
        existing ``scene.visuals.Volume`` and just re-upload the texture
        data + swap the colormap — this avoids recreating the whole vispy
        visual, which costs hundreds of milliseconds on large stacks.

        The pooled output buffer is cached too, so the repeated pool that
        happens on every toggle doesn't re-allocate ~0.5 GB per call.
        """
        # Reentrancy guard: show_progress() pumps paint events and a
        # queued signal (e.g. another visibility toggle) could otherwise
        # reenter us while `labels`/`max_id` locals are still stale.
        # Skipping reentrant calls is safe — the triggering signal either
        # beat us to the same state or will fire a fresh refresh on its
        # own emit path. We never lose a refresh: every external trigger
        # calls refresh_labels again after its own state is committed.
        if self._rebuilding_labels:
            return
        self._rebuilding_labels = True
        try:
            self._do_rebuild_labels_overlay()
        finally:
            self._rebuilding_labels = False
            # Belt-and-suspenders: guarantee the progress bar is hidden
            # even if _do_rebuild_labels_overlay's own finally missed a
            # path (e.g. an exception thrown before its try started).
            self.hide_progress()

    def _do_rebuild_labels_overlay(self):
        """Actual rebuild body. Never call directly — always route through
        ``_rebuild_labels_overlay`` so the reentrancy guard holds.
        """
        if self._view is None:
            if self._labels_volume is not None:
                self._labels_volume.parent = None
                self._labels_volume = None
            return
        doc = self._primary_doc
        if doc is None or doc.labels_3d is None or not doc.labels_meta:
            if self._labels_volume is not None:
                self._labels_volume.parent = None
                self._labels_volume = None
            return
        labels = doc.labels_3d
        max_id = int(max(doc.labels_meta.keys()))
        if max_id <= 0:
            if self._labels_volume is not None:
                self._labels_volume.parent = None
                self._labels_volume = None
            return
        # All-hidden fast-path: nothing will render anyway. Skip the pool
        # and the upload entirely and just toggle the existing visual off.
        # LayerPanel's "hide all" sets every VolumeROILayer.visible=False,
        # which hits this path — the slowest case in the old code becomes
        # the fastest.
        all_hidden = all(
            (not m.get('visible', True))
            or int(m.get('opacity', 128)) <= 0
            for m in doc.labels_meta.values()
        )
        if all_hidden:
            if self._labels_volume is not None:
                self._labels_volume.visible = False
                self._labels_volume.update()
            return
        # Build the texture LUT + dense cmap. ``texture_lut`` maps raw
        # label ids to dense palette slots (1..palette_size, 0 =
        # transparent / hidden). Dense slots dodge vispy's LUT_len=1024
        # resampling that scrambled colours when raw ids spanned a wide
        # sparse range (the neuron_*_branch_labels_full.tif case — 195
        # ids in [0..663], renders showed wrong colours / missing
        # skeletons). The LUT also folds in visibility/opacity filtering
        # — hidden labels map to slot 0 so max-pool naturally prefers
        # visible neighbours, no separate ``id_map`` needed.
        # The LUT output dtype is now picked inside the builder based
        # on ``palette_size`` (uint8 ≤255, uint16 ≤65535, else uint32),
        # NOT the input labels' dtype — so a 200-ROI sparse uint16
        # file uploads to the GPU as uint8, halving texture bandwidth.
        # ``dtype`` argument is back-compat only; ignored.
        texture_lut, cmap, palette_size = self._build_labels_colormap_and_lut(
            doc.labels_meta, max_id, dtype=labels.dtype,
        )
        if palette_size == 0:
            # No visible labels survived the LUT build (every entry was
            # hidden) — same UX as all_hidden above; just toggle off.
            if self._labels_volume is not None:
                self._labels_volume.visible = False
                self._labels_volume.update()
            return

        # Stale-id defence: ``labels_3d`` can carry voxel ids beyond
        # ``max(labels_meta)`` after a partial sidecar import (see
        # ``montaris/io/volume_labels.py``, which accepts uint32 ids).
        # Without a meta entry those ids have no colour, so render them
        # as background (slot 0). Mask via ``np.where`` rather than
        # extending the LUT — a stale 3-billion uint32 id would drive a
        # multi-GB ``np.zeros`` allocation under extension. Masking
        # caps memory at one ``labels``-sized working copy.
        if labels.size:
            lbl_max_raw = int(labels.max())
            if lbl_max_raw >= texture_lut.size:
                labels = np.where(labels < texture_lut.size, labels, 0)

        dz, dy, dx = (max(1, int(a)) for a in self._last_total_ds_axes)
        block = (dz, dy, dx)
        if block != (1, 1, 1):
            # Max-pool per (dz, dy, dx) block rather than stride-decimate.
            # Stride decimation skips voxels, which destroys sparse thin
            # structures (e.g. 1-voxel-wide dendrites collapse to a dot
            # grid — the "skeleton" effect). Max-pool preserves any
            # block that contains a nonzero label, matching napari's
            # behaviour. ``id_map=texture_lut`` doubles as the palette
            # remap and the visibility filter (hidden ids → slot 0 so
            # max-pool prefers visible neighbours).
            self.show_progress()
            key = (labels.shape, labels.dtype, block)
            reuse = (self._pooled_labels_key == key
                     and self._pooled_labels_buf is not None)
            out_buf = self._pooled_labels_buf if reuse else None
            labels = _max_pool_labels(labels, block, id_map=texture_lut,
                                      out=out_buf)
            self._pooled_labels_buf = labels
            self._pooled_labels_key = key
        else:
            # No pooling — route the LUT through ``_max_pool_labels`` so
            # the bounds-extension guard (extends short LUT to identity
            # for ids beyond ``max_id``) is shared with the pool path.
            # Direct ``texture_lut[labels]`` would IndexError on stale
            # imports where labels_3d carries ids absent from
            # labels_meta (partial sidecars; see volume_labels.py).
            labels = _max_pool_labels(labels, (1, 1, 1), id_map=texture_lut)
        try:
            labels_contig = np.ascontiguousarray(labels)

            # Fast path: reuse the existing Volume visual when shape/dtype
            # match what's already on the GPU. Swap the data + cmap in
            # place so visibility toggles don't pay the cost of rebuilding
            # the vispy visual (shader compile + scene graph churn). Use
            # VolumeVisual.set_data rather than poking _texture directly —
            # the visual caches the data for later re-uploads (e.g. when
            # clim widens), so bypassing it would leave stale voxels
            # around. ``copy=False`` skips the internal memcpy since
            # labels_contig is already contiguous and we won't mutate it.
            # ``clim`` matches the dense palette range, NOT raw max_id.
            # Voxel value `slot` (1..palette_size) maps to LUT texcoord
            # ``slot / palette_size`` which falls cleanly inside the
            # palette_size+1 cmap controls — no resampling drift.
            clim_max = float(palette_size)
            if (self._labels_volume is not None
                    and tuple(getattr(self, '_labels_tex_shape', ())) == labels_contig.shape
                    and getattr(self, '_labels_tex_dtype', None) == labels_contig.dtype):
                try:
                    self._labels_volume.set_data(
                        labels_contig, clim=(0, clim_max), copy=False,
                    )
                    self._labels_volume.cmap = cmap
                    self._labels_volume.visible = self._labels_visible
                    self._labels_volume.update()
                    self._labels_texture_lut = texture_lut
                    self._labels_palette_size = palette_size
                    return
                except Exception:
                    # Fall through to full rebuild on any in-place failure.
                    pass

            # Slow path: full rebuild (first render, shape change, fallback).
            if self._labels_volume is not None:
                self._labels_volume.parent = None
                self._labels_volume = None
            try:
                # Hand the integer array straight through — vispy's Volume
                # accepts uint8/uint16 and uploads them verbatim.
                vol = scene.visuals.Volume(
                    labels_contig,
                    parent=self._view.scene,
                    method='translucent',
                    cmap=cmap,
                    clim=(0, clim_max),
                )
            except Exception as e:  # noqa: BLE001 — GL upload can fail on edge dtypes
                print(f"[view_3d] labels overlay failed: {e}")
                return
            vol.visible = self._labels_visible
            vol.order = 10  # drawn after intensity so labels composite on top
            if block != (1, 1, 1):
                from vispy.visuals.transforms import STTransform
                vol.transform = STTransform(scale=(dx, dy, dz))
            self._labels_volume = vol
            self._labels_tex_shape = labels_contig.shape
            self._labels_tex_dtype = labels_contig.dtype
            self._labels_texture_lut = texture_lut
            self._labels_palette_size = palette_size
        finally:
            # Hide is cheap and a no-op when the bar wasn't shown, so just
            # always call it here instead of tracking whether show_progress
            # ran.
            self.hide_progress()

    def capture_state(self):
        """Snapshot camera pose so it can be restored after a rebuild."""
        if self._view is None or self._view.camera is None:
            return None
        cam = self._view.camera
        try:
            quat = cam._quaternion
            qx, qy, qz, qw = quat.x, quat.y, quat.z, quat.w
        except Exception:
            qx = qy = qz = qw = None
        return {
            'center': tuple(cam.center),
            'scale_factor': float(cam._scale_factor),
            'quaternion': (qx, qy, qz, qw),
            'pan_sf_ref': float(getattr(cam, '_pan_sf_ref', 0.0)),
            'mode': self._mode_combo.currentText(),
            'downsample': int(self._ds_slider.value()),
            'visible': [cb.isChecked() for cb in self._channel_toggles],
        }

    def apply_state(self, state):
        """Restore a state dict produced by :meth:`capture_state`."""
        if not state or self._view is None or self._view.camera is None:
            return
        if 'mode' in state:
            idx = self._mode_combo.findText(state['mode'])
            if idx >= 0:
                self._mode_combo.blockSignals(True)
                self._mode_combo.setCurrentIndex(idx)
                self._mode_combo.blockSignals(False)
                # Apply mode to visuals without triggering rebuild
                method = next((m for m, lbl in self._mode_labels.items()
                               if lbl == state['mode']), 'mip')
                for vol in self._volumes:
                    vol.method = method
        if 'downsample' in state and state['downsample'] != self._downsample_factor:
            self._ds_slider.blockSignals(True)
            self._ds_slider.setValue(state['downsample'])
            self._ds_slider.blockSignals(False)
            self._downsample_factor = int(state['downsample'])
            self._rebuild_volumes()
        for i, on in enumerate(state.get('visible', [])):
            if i < len(self._channel_toggles):
                self._channel_toggles[i].blockSignals(True)
                self._channel_toggles[i].setChecked(on)
                self._channel_toggles[i].blockSignals(False)
                self._set_visible(i, on)
        cam = self._view.camera
        try:
            cam.center = state['center']
            cam._scale_factor = float(state['scale_factor'])
            cam._pan_sf_ref = float(state.get('pan_sf_ref', 0.0))
            q = state.get('quaternion')
            if q and all(v is not None for v in q):
                from vispy.util.quaternion import Quaternion
                cam._quaternion = Quaternion(q[3], q[0], q[1], q[2])
            cam.view_changed()
        except Exception:
            pass

    def release_gl(self):
        """Free GPU resources held by this panel. Safe to call repeatedly."""
        for vol in self._volumes:
            vol.parent = None
        self._volumes = []
        if self._labels_volume is not None:
            self._labels_volume.parent = None
            self._labels_volume = None
        if self._canvas is not None:
            try:
                self._canvas.close()
            except Exception:
                pass
            self._canvas = None
        self._view = None

    def closeEvent(self, event):
        self.release_gl()
        super().closeEvent(event)


def channels_from_documents(documents):
    """Build the ``channels`` list View3DPanel expects from MontageDocuments.

    Skips documents without a ``volume_data`` attribute. Returns an empty list
    if no volumes are available.

    In z-stack ``'synced'`` mode the app produces one MontageDocument per
    channel that may share the same ``volume_data`` ndarray (when a single
    file supplies multiple views of the same stack). Dedupe by ``id(vol)``
    so each distinct volume array is rendered once, using the first doc's
    name/tint.
    """
    channels = []
    seen = set()
    for doc in documents:
        vol = getattr(doc, 'volume_data', None)
        if vol is None:
            continue
        key = id(vol)
        if key in seen:
            continue
        seen.add(key)
        tint = doc.tint_color
        if tint is not None and max(tint) > 1.0:
            tint = tuple(c / 255.0 for c in tint)
        channels.append((doc.name, vol, tint))
    return channels


def open_view_3d(parent, documents):
    """Legacy helper: open the 3D viewer as a standalone top-level window.

    Kept for the headed smoke script and tests. The main app now embeds
    :class:`View3DPanel` inside its central stack instead of calling this.
    """
    if not VISPY_AVAILABLE:
        QMessageBox.information(
            parent, "3D view unavailable",
            "The 3D viewer needs the optional 'vispy' package.\n\n"
            "Install it with:\n    pip install vispy\n\n"
            "Then restart Montaris-X.",
        )
        return False
    channels = channels_from_documents(documents)
    if not channels:
        QMessageBox.information(
            parent, "No 3D data",
            "No z-stack volumes loaded. Open one or more z-stack TIFFs first.",
        )
        return False
    panel = View3DPanel(None, channels=channels, documents=documents)
    panel.setWindowTitle("3D View")
    panel.setWindowFlags(
        Qt.Window
        | Qt.WindowSystemMenuHint
        | Qt.WindowTitleHint
        | Qt.WindowMinimizeButtonHint
        | Qt.WindowMaximizeButtonHint
        | Qt.WindowCloseButtonHint
    )
    panel.resize(960, 720)
    panel.show()
    return True
