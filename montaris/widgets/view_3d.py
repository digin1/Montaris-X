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
    QCheckBox, QSlider, QMessageBox, QWidget, QFrame,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics


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


def _query_gl(canvas, enum, fallback):
    """Pull a GL state value from the canvas's live context via vispy's wrapper."""
    try:
        from vispy.gloo import gl
        return gl.glGetParameter(enum)
    except Exception:
        return fallback


def _detect_max_3d_dim(canvas):
    try:
        return int(_query_gl(canvas, _GL_MAX_3D_TEXTURE_SIZE, _MAX_3D_DIM_FALLBACK))
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
    3. Heuristic based on GL_RENDERER string:
       - Software (llvmpipe/swr) → 64 MB (CPU path, keep tiny)
       - Intel / integrated      → 256 MB (shares system RAM, stay modest)
       - Apple Silicon           → 1024 MB (unified memory, but don't hog)
       - NVIDIA / AMD discrete   → 1024 MB (assume ≥2 GB card, leave headroom)

    OpenGL has no portable free-VRAM query, so the heuristic is the honest
    best-effort for non-NVIDIA GPUs.
    """
    override = os.environ.get("MONTARIS_GPU_BUDGET_MB")
    if override:
        try:
            return max(16, int(override)) * 1024 * 1024
        except ValueError:
            pass

    # NVX extension path — only exists on NVIDIA drivers.
    try:
        kb = _query_gl(canvas, _GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, 0)
        if isinstance(kb, (int, float)) and kb > 0:
            # Half of free VRAM leaves room for other textures + framebuffers.
            return int(kb) * 1024 // 2
    except Exception:
        pass

    desc = (gpu_desc or "").lower()
    if "llvmpipe" in desc or "swr" in desc or "software" in desc:
        return 64 * 1024 * 1024
    if "intel" in desc:
        return 256 * 1024 * 1024
    if "apple" in desc or "metal" in desc:
        return 1024 * 1024 * 1024
    if "nvidia" in desc or "geforce" in desc or "rtx" in desc or "quadro" in desc \
       or "amd" in desc or "radeon" in desc:
        return 1024 * 1024 * 1024
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
        self._labels_opacity = 0.6
        # Last total downsample applied to intensity volumes; labels overlay
        # re-uses the same factor so voxels line up.
        self._last_total_ds = 1
        # Annotation tool state. ``_tool_mode`` is 'navigate' or 'fill' for
        # Phase 3; paint/erase land in Phase 5. ``_fill_channel_idx`` selects
        # which intensity channel the flood fill reads from (each channel is
        # usually a different wavelength and they don't share intensities).
        self._tool_mode = 'navigate'
        self._fill_channel_idx = 0
        self._fill_tolerance = 20  # 0..255 in uint8 space (auto-scaled per channel)
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
        self._channel_toggles = []
        ch_row = QHBoxLayout()
        ch_row.setSpacing(12)
        for idx, (name, _vol, _tint) in enumerate(self._channels_raw):
            cb = QCheckBox()
            fm = QFontMetrics(cb.font())
            cb.setText(fm.elidedText(name, Qt.ElideMiddle, 180))
            cb.setToolTip(name)
            cb.setChecked(True)
            rgb = self._tint_for(idx, self._channels_raw[idx][2])
            cb.setStyleSheet(f"color: rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)});")
            cb.toggled.connect(lambda on, i=idx: self._set_visible(i, on))
            ch_row.addWidget(cb)
            self._channel_toggles.append(cb)
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

            # Annotation tool row. Currently just Navigate + Fill (Phase 3).
            tool_row = QHBoxLayout()
            tool_row.setSpacing(8)
            tool_row.addWidget(QLabel("Tool:"))
            self._tool_combo = QComboBox()
            self._tool_combo.addItems(["Navigate", "Fill"])
            self._tool_combo.currentTextChanged.connect(self._on_tool_changed)
            tool_row.addWidget(self._tool_combo)

            tool_row.addWidget(self._sep())

            tool_row.addWidget(QLabel("Channel:"))
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

            tool_row.addWidget(self._sep())

            tool_row.addWidget(QLabel("Tolerance:"))
            self._tol_slider = QSlider(Qt.Horizontal)
            self._tol_slider.setRange(0, 100)
            self._tol_slider.setValue(20)
            self._tol_slider.setFixedWidth(120)
            self._tol_slider.valueChanged.connect(self._on_tolerance_changed)
            tool_row.addWidget(self._tol_slider)
            self._tol_label = QLabel("20")
            tool_row.addWidget(self._tol_label)

            tool_row.addStretch(1)
            root.addLayout(tool_row)

            # Hook up the click-to-fill mouse handler. Vispy SceneCanvas
            # emits mouse events independently of Qt's mousePressEvent so we
            # attach via its events system. The handler itself short-circuits
            # unless tool mode is 'fill'.
            self._canvas.events.mouse_press.connect(self._on_canvas_mouse_press)

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

    def _on_tool_changed(self, text):
        self._tool_mode = text.lower()
        # Disable camera interaction in Fill mode so left-drag doesn't rotate
        # while the user is trying to click-to-fill.
        if self._view is not None and self._view.camera is not None:
            self._view.camera.interactive = (self._tool_mode == 'navigate')
        # Cursor hint
        if self._tool_mode == 'fill':
            self._canvas.native.setCursor(Qt.CrossCursor)
        else:
            self._canvas.native.unsetCursor()

    def _on_fill_channel_changed(self, idx):
        self._fill_channel_idx = int(idx)

    def _on_tolerance_changed(self, val):
        self._fill_tolerance = int(val)
        self._tol_label.setText(str(val))

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

    def _on_canvas_mouse_press(self, event):
        """Route mouse press to the active annotation tool.

        In Navigate mode this is a no-op — the camera's own bindings
        handle rotation/pan/zoom. In Fill mode, a left-click runs a
        ray-pick → seed → flood fill sequence on the active channel.
        """
        if self._tool_mode != 'fill':
            return
        if event.button != 1:  # only left button triggers fill
            return
        if self._primary_doc is None or not self._volumes:
            return
        # event.pos is in canvas pixel space; for vispy ViewBox with the
        # camera we configured, that equals viewbox pixel space.
        seed = self._ray_pick_seed(event.pos)
        if seed is None:
            return
        self._run_fill(seed)

    def _ray_pick_seed(self, pos):
        """Cast a ray through canvas pixel ``pos`` and return the brightest
        voxel along it, as ``(z, y, x)`` indices into the active channel.

        Returns ``None`` if the ray misses the volume or the volume is empty.
        The active channel is the one selected in ``_fill_channel_combo``.
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
        if samples[best] <= 0:
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
        """Run an intensity-tolerance flood fill from ``seed_zyx`` and write
        the resulting mask into ``primary_doc.labels_3d`` under a fresh
        label ID. Emits ``label_added(doc, lid)`` on success.
        """
        doc = self._primary_doc
        volume = self._active_channel_volume()
        if doc is None or volume is None:
            return
        # Flood tolerance: slider is 0..100 in percentage of the channel's
        # observed dynamic range. 20 on a [0, 1000] volume → ±200 intensity.
        vmin = float(volume.min())
        vmax = float(volume.max())
        drange = max(1.0, vmax - vmin)
        tol = (self._fill_tolerance / 100.0) * drange

        try:
            from skimage.segmentation import flood
        except Exception as e:  # noqa: BLE001 — surface to the user
            QMessageBox.warning(
                self, "Fill unavailable",
                f"The Fill tool needs scikit-image installed:\n{e}",
            )
            return
        # skimage.flood works on 3D arrays; returns bool mask.
        mask = flood(volume, tuple(seed_zyx), tolerance=tol, connectivity=1)
        if not mask.any():
            return

        # Allocate the labels volume on first fill (same shape as volume_data,
        # which is what _primary_doc uses); reserve a label ID and stamp.
        if doc.labels_3d is None:
            doc.ensure_labels_3d()
        lid = doc.reserve_label_id()
        if lid > np.iinfo(doc.labels_3d.dtype).max:
            doc.promote_labels_dtype(np.uint16)
        # Write voxels. Voxels already owned by another label get overwritten
        # — napari's behavior is the same (whichever label you fill last wins).
        doc.labels_3d[mask] = lid

        self.refresh_labels()
        self.label_added.emit(doc, lid)

    def _reset_view(self):
        if self._volumes:
            self._view.camera.set_range()
            self._view.camera.view_changed()

    def _rebuild_volumes(self):
        # Remove any existing volumes
        for vol in self._volumes:
            vol.parent = None
        self._volumes = []

        label = self._mode_combo.currentText()
        method = next((m for m, lbl in self._mode_labels.items() if lbl == label), 'mip')

        # -- Pass 1: per-channel prep (user stride + per-axis dim cap) --
        prepped = []  # list of (idx, tint, clim, arr_after_gpu_cap, gpu_factor_max)
        max_gpu_factor = 1
        for idx, (name, volume, tint) in enumerate(self._channels_raw):
            if volume is None:
                continue
            arr = np.asarray(volume)
            clim = _percentile_clim(arr)
            arr = _downsample(arr, self._downsample_factor)
            arr, gpu_factors = _fit_to_gpu(arr, self._max_3d_dim)
            max_gpu_factor = max(max_gpu_factor, max(gpu_factors))
            prepped.append((idx, tint, clim, arr, max(gpu_factors)))

        # -- Pass 2: budget fit across all channels together --
        shapes = [p[3].shape for p in prepped]
        budget_factor = _fit_to_budget(shapes, self._vram_budget, bytes_per_voxel=1)
        # OOM retries only add stride — never relax it — until the upload succeeds.
        total_extra = budget_factor * self._oom_retry_factor
        if total_extra > 1:
            prepped = [(idx, tint, clim, _downsample(arr, total_extra), gf)
                       for idx, tint, clim, arr, gf in prepped]

        # -- Pass 3: upload, catching OOM so we can retry with more stride --
        try:
            for idx, tint, clim, arr, _gf in prepped:
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
        total = self._downsample_factor * max_gpu_factor * budget_factor * self._oom_retry_factor
        reasons = []
        if max_gpu_factor > 1:
            reasons.append("GPU")
        if budget_factor > 1 or self._oom_retry_factor > 1:
            reasons.append("memory")
        self._last_reason = "+".join(reasons)
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

    def _build_labels_colormap(self, labels_meta, max_id):
        """Build a discrete Colormap: 0 → transparent, N → meta[N] color.

        Uses ``interpolation='zero'`` so rendering is nearest-neighbor; any
        lerp between slots would produce ghost labels on the boundary.
        """
        n_slots = int(max_id) + 1
        colors = np.zeros((n_slots, 4), dtype=np.float32)
        for lid, meta in labels_meta.items():
            lid = int(lid)
            if lid <= 0 or lid >= n_slots:
                continue
            r, g, b = meta.get('color', (255, 255, 255))
            if max(r, g, b) > 1.0:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            base_alpha = meta.get('opacity', 128) / 255.0
            visible = meta.get('visible', True)
            a = (base_alpha * self._labels_opacity) if visible else 0.0
            colors[lid] = (r, g, b, a)
        # Zero-order interpolation needs N+1 control points (one per step
        # boundary) for N color entries. Linear interp would blur between
        # labels and create ghost colors at the boundary voxels, so we stay
        # with step.
        controls = np.linspace(0.0, 1.0, n_slots + 1, dtype=np.float32)
        return Colormap(colors=colors, controls=controls, interpolation='zero')

    def _rebuild_labels_overlay(self):
        """(Re)create the labels overlay visual from ``primary_doc.labels_3d``.

        Safe to call when no labels exist — tears down any stale overlay and
        returns. Called at the end of every intensity rebuild so downsample
        changes propagate, and from ``refresh_labels()`` after paint ops.
        """
        # Always drop the old visual first — it may be out of date.
        if self._labels_volume is not None:
            self._labels_volume.parent = None
            self._labels_volume = None
        if self._view is None:
            return
        doc = self._primary_doc
        if doc is None or doc.labels_3d is None or not doc.labels_meta:
            return
        labels = doc.labels_3d
        ds = max(1, int(self._last_total_ds))
        if ds > 1:
            labels = labels[::ds, ::ds, ::ds]
        max_id = max(doc.labels_meta.keys())
        if max_id <= 0:
            return
        cmap = self._build_labels_colormap(doc.labels_meta, max_id)
        try:
            vol = scene.visuals.Volume(
                labels.astype(np.float32, copy=False),
                parent=self._view.scene,
                method='translucent',
                cmap=cmap,
                clim=(0, float(max_id)),
            )
        except Exception as e:  # noqa: BLE001 — GL upload can fail on edge dtypes
            print(f"[view_3d] labels overlay failed: {e}")
            return
        vol.visible = self._labels_visible
        # Render labels after intensity so they composite on top. Larger
        # order = drawn later.
        vol.order = 10
        self._labels_volume = vol

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
    """
    channels = []
    for doc in documents:
        vol = getattr(doc, 'volume_data', None)
        if vol is None:
            continue
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
