import sys
import os
import time
from datetime import datetime
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QFileDialog,
    QStatusBar, QMessageBox, QProgressDialog, QDialog, QVBoxLayout, QTextEdit,
    QDialogButtonBox, QToolBar, QLabel, QSlider, QSpinBox, QHBoxLayout, QWidget, QPushButton,
    QComboBox, QInputDialog, QColorDialog, QFrame,
)
from PySide6.QtCore import Qt, QSettings, QRectF, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QPalette, QColor, QTransform, QShortcut, QIcon

from montaris.canvas import ImageCanvas
from montaris.layers import LayerStack, ImageLayer, ROILayer, ROI_COLORS, generate_unique_roi_name, MontageDocument
from montaris.widgets.layer_panel import LayerPanel
from montaris.widgets.tool_panel import ToolPanel
from montaris.widgets.properties_panel import PropertiesPanel
from montaris.widgets.display_panel import DisplayPanel
from montaris.widgets.adjustments_panel import AdjustmentsPanel
from montaris.widgets.minimap import MiniMap
from montaris.widgets.perf_monitor import PerfMonitor
from montaris.widgets.debug_console import DebugConsole
from montaris.core.undo import UndoStack
from montaris.core.adjustments import ImageAdjustments
from montaris.core.display_modes import DisplayCompositor
from montaris.widgets.toast import ToastManager
from montaris.io.image_io import load_image, load_image_stack
from montaris.io.roi_io import save_roi_set, load_roi_set
from montaris.core.busy import busy_cursor, should_process_events
from montaris import theme as _theme
from montaris.widgets import AnimatedButton

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:
    _HAS_QTA = False


def _qta_icon(name):
    """Return a QtAwesome icon with theme-appropriate color, or None."""
    if _HAS_QTA:
        color = '#dcdcdc' if _theme.is_dark() else '#333'
        return qta.icon(name, color=color)
    return None


def apply_dark_theme(app):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText, QColor(255, 50, 50))
    palette.setColor(QPalette.Link, QColor(80, 140, 220))
    palette.setColor(QPalette.Highlight, QColor(80, 140, 220))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)


def apply_light_theme(app):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(30, 30, 30))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(30, 30, 30))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(30, 30, 30))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 100, 200))
    palette.setColor(QPalette.Highlight, QColor(42, 100, 200))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)


def apply_system_theme(app):
    app.setStyle("WindowsVista")
    app.setPalette(app.style().standardPalette())


def _sanitize_roi_filename(name):
    """Sanitize an ROI name for use as a filename component.

    Replaces all characters that are illegal in Windows filenames:
    ``/ \\ : * ? " < > |``
    """
    import re
    return re.sub(r'[/\\:*?"<>|]', '_', name)


def _save_session_from_snapshots(session_dir, snapshots, meta):
    """Write ROI snapshots to a session folder (runs in background thread).

    Snapshots contain bbox-cropped masks ('crop' + 'bbox') to avoid holding
    full-resolution masks in memory.
    """
    import json as _json
    from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi

    os.makedirs(session_dir, exist_ok=True)

    # Remove stale .roi files from previous save so deleted ROIs don't linger
    for existing in os.listdir(session_dir):
        if existing.lower().endswith('.roi'):
            try:
                os.remove(os.path.join(session_dir, existing))
            except OSError:
                pass

    # Use indexed filenames to avoid collisions from sanitization
    roi_files = []
    for i, snap in enumerate(snapshots):
        name = snap['name']
        safe_name = _sanitize_roi_filename(name)
        filename = f"{i:04d}_{safe_name}.roi"
        # Pass crop directly — mask_to_imagej_roi slices mask[top:bottom, left:right]
        # so pass bbox=(0, h, 0, w) and let it use the crop as-is,
        # then fix up the coordinates with the real offset.
        bbox = snap['bbox']  # (y1, y2, x1, x2)
        crop = snap.get('crop')
        if crop is not None:
            # Crop-based: run contour tracing on the small crop, then offset
            roi_dict = mask_to_imagej_roi(crop, name, bbox=(0, crop.shape[0], 0, crop.shape[1]))
        else:
            # Full-mask fallback (used by tests / legacy callers)
            roi_dict = mask_to_imagej_roi(snap['mask'], name, bbox=bbox)
        if roi_dict is None:
            roi_files.append(None)
            continue
        # Shift coordinates from crop-local to full-mask space (crop path only)
        if crop is not None:
            y_off, _, x_off, _ = bbox
            roi_dict['top'] += y_off
            roi_dict['bottom'] += y_off
            roi_dict['left'] += x_off
            roi_dict['right'] += x_off
            if roi_dict.get('x_coords') is not None:
                roi_dict['x_coords'] = roi_dict['x_coords'] + x_off
                roi_dict['y_coords'] = roi_dict['y_coords'] + y_off
            if roi_dict.get('paths'):
                roi_dict['paths'] = [
                    [(x + x_off, y + y_off) for x, y in p]
                    for p in roi_dict['paths']
                ]
        write_imagej_roi(roi_dict, os.path.join(session_dir, filename))
        roi_files.append(filename)

    meta['roi_files'] = roi_files

    with open(os.path.join(session_dir, 'session.json'), 'w') as f:
        _json.dump(meta, f, indent=2)


class MontarisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Montaris-X")
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(1400, avail.width() - 20)
            h = min(900, avail.height() - 20)
            self.resize(w, h)
        else:
            self.resize(1400, 900)
        _logo = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        if os.path.exists(_logo):
            self.setWindowIcon(QIcon(_logo))

        self.layer_stack = LayerStack()
        self.undo_stack = UndoStack()
        self.active_tool = None
        self._compositor = DisplayCompositor()
        self._adjustments = ImageAdjustments()
        self._auto_overlap = False
        self._downsample_factor = 1
        self._student_session = False
        self._roi_import_path = None  # track where ROI ZIP was loaded from
        self._documents = []
        self._active_doc_index = -1
        self._composite_mode = False
        self._session_dir = None  # current session folder path (reused on save)

        self._setup_canvas()
        self._setup_panels()
        self._setup_menus()
        self._setup_statusbar()

        self.toast = ToastManager(self)
        self._setup_toolbar()
        self.setAcceptDrops(True)

        # Alternative redo shortcut (Ctrl+Shift+Z)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.redo)
        # Tool shortcuts (H.15)
        QShortcut(QKeySequence("Ctrl+B"), self, lambda: self.tool_panel._select_tool('Brush'))
        QShortcut(QKeySequence("Ctrl+U"), self, lambda: self.tool_panel._select_tool('Bucket Fill'))

        self.settings = QSettings("Montaris", "Montaris-X")
        self._restore_state()

        # Set narrower right sidebar width
        QTimer.singleShot(0, self._apply_dock_widths)

        # Activate default tool after all setup (signals, statusbar, toolbar)
        self.tool_panel.activate_default_tool()

    # -- Drag-and-drop support ------------------------------------------
    _DROP_IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        image_paths = []
        zip_paths = []
        for url in urls:
            path = url.toLocalFile()
            if not path:
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext in self._DROP_IMAGE_EXTS:
                image_paths.append(path)
            elif ext == '.zip':
                zip_paths.append(path)

        # Load images first so ROI import has a canvas to attach to
        if image_paths:
            self.open_image(image_paths)

        for zp in zip_paths:
            self.import_roi_zip(zp)

        if not image_paths and not zip_paths:
            self.toast.show("Unsupported file type", "warning")

    # ------------------------------------------------------------------

    def _setup_canvas(self):
        self.canvas = ImageCanvas(self.layer_stack, self)
        self.canvas._adjustments = self._adjustments
        self.setCentralWidget(self.canvas)

    def _setup_panels(self):
        # Layer panel
        self.layer_panel = LayerPanel(self.layer_stack, self)
        layer_dock = QDockWidget("Layers", self)
        layer_dock.setObjectName("LayersDock")
        layer_dock.setWidget(self.layer_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, layer_dock)
        self._layer_dock = layer_dock

        # Tool panel — header is inside ToolPanel, hide dock title bar
        self.tool_panel = ToolPanel(self, self)
        tool_dock = QDockWidget("Tools", self)
        tool_dock.setObjectName("ToolsDock")
        tool_dock.setTitleBarWidget(QWidget())  # hide dock title
        tool_dock.setWidget(self.tool_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, tool_dock)
        self._tool_dock = tool_dock

        # Styled section title bar for right-side docks
        self._themed_labels = []  # track for theme refresh

        def _section_title(text):
            lbl = QLabel(f"  {text}")
            lbl.setFixedHeight(26)
            lbl.setStyleSheet(_theme.section_header_style())
            self._themed_labels.append(lbl)
            return lbl

        # Combined right-side panel with collapsible sections
        self.properties_panel = PropertiesPanel(self, self)
        self.display_panel = DisplayPanel(self)
        self.adjustments_panel = AdjustmentsPanel(self)

        self.display_panel.mode_changed.connect(self._on_display_mode_changed)
        self.display_panel.channels_changed.connect(self._on_channels_changed)
        self.display_panel.composite_toggled.connect(self._on_composite_toggled)
        self.adjustments_panel.adjustments_changed.connect(self._on_adjustments_changed)

        self._collapsible_headers = []  # track for theme refresh
        right_combined = QWidget()
        right_lay = QVBoxLayout(right_combined)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        for title, panel, collapsed in [
            ("ROI Properties", self.properties_panel, False),
            ("Display Settings", self.display_panel, True),
            ("Image Adjustments", self.adjustments_panel, False),
        ]:
            header = QPushButton(f"\u25BC  {title}")
            header.setFixedHeight(26)
            header.setStyleSheet(_theme.collapsible_header_style())
            panel.setVisible(not collapsed)
            if collapsed:
                header.setText(f"\u25B6  {title}")

            def _make_toggle(h, p, t):
                def toggle():
                    vis = not p.isVisible()
                    p.setVisible(vis)
                    h.setText(f"\u25BC  {t}" if vis else f"\u25B6  {t}")
                return toggle

            header.clicked.connect(_make_toggle(header, panel, title))
            self._collapsible_headers.append(header)
            right_lay.addWidget(header)
            right_lay.addWidget(panel)

        right_lay.addStretch()

        combined_dock = QDockWidget("Details", self)
        combined_dock.setObjectName("DetailsDock")
        combined_dock.setTitleBarWidget(_section_title("Details"))
        combined_dock.setWidget(right_combined)
        self.addDockWidget(Qt.RightDockWidgetArea, combined_dock)
        self._details_dock = combined_dock

        # Minimap (Phase 6)
        self.minimap = MiniMap(self)
        minimap_dock = QDockWidget("Mini Map", self)
        minimap_dock.setObjectName("MiniMapDock")
        minimap_dock.setWidget(self.minimap)
        self.addDockWidget(Qt.LeftDockWidgetArea, minimap_dock)
        self.minimap.pan_requested.connect(self._on_minimap_pan)
        self._minimap_dock = minimap_dock

        # Performance monitor (Phase 6)
        self.perf_monitor = PerfMonitor(self)
        perf_dock = QDockWidget("Performance", self)
        perf_dock.setObjectName("PerformanceDock")
        perf_dock.setWidget(self.perf_monitor)
        self.addDockWidget(Qt.LeftDockWidgetArea, perf_dock)
        perf_dock.setVisible(False)
        self._perf_dock = perf_dock

        # Debug console (Phase 6)
        self.debug_console = DebugConsole(self, self)
        debug_dock = QDockWidget("Debug Console", self)
        debug_dock.setObjectName("DebugConsoleDock")
        debug_dock.setWidget(self.debug_console)
        self.addDockWidget(Qt.BottomDockWidgetArea, debug_dock)
        debug_dock.setVisible(False)
        self._debug_dock = debug_dock

        # Wire selection model
        self.layer_panel.set_selection_model(self.canvas._selection)
        self.canvas._selection.changed.connect(self._on_selection_count_changed)

        # Collapsed left toolbar (hidden by default)
        self._left_toolbar = QToolBar("Toolbox", self)
        self._left_toolbar.setObjectName("LeftToolBar")
        self._left_toolbar.setOrientation(Qt.Vertical)
        self._left_toolbar.setMovable(False)
        from PySide6.QtCore import QSize
        self._left_toolbar.setIconSize(QSize(28, 28))
        self._left_toolbar.setStyleSheet(
            "QToolBar { spacing: 2px; padding: 2px; }"
            "QToolButton { font-size: 18px; min-width: 36px; min-height: 32px; }"
        )
        self.addToolBar(Qt.LeftToolBarArea, self._left_toolbar)
        self._left_toolbar.setVisible(False)
        self._left_collapsed = False

        from montaris.widgets.tool_panel import TOOL_ICONS, _QTA_ICONS, _tool_icon
        _expand_ico = _qta_icon('fa6s.angles-right')
        expand_left_act = QAction("Expand") if not _expand_ico else QAction(_expand_ico, "", self)
        expand_left_act.setToolTip("Expand Toolbox (Ctrl+[)")
        expand_left_act.triggered.connect(self._toggle_left_sidebar)
        self._left_toolbar.addAction(expand_left_act)
        self._left_toolbar.addSeparator()

        self._left_tool_actions = {}
        from montaris.tools import TOOL_REGISTRY
        for name, (module, cls_name, shortcut, category) in TOOL_REGISTRY.items():
            qicon = _tool_icon(name)
            if qicon:
                act = QAction(qicon, "", self)
            else:
                icon = TOOL_ICONS.get(name, shortcut)
                act = QAction(f"{icon}", self)
            act.setToolTip(f"{name} [{shortcut}]")
            act.setCheckable(True)
            act.triggered.connect(lambda checked, t=name: self.tool_panel._select_tool(t))
            self._left_toolbar.addAction(act)
            self._left_tool_actions[name] = act

        # Collapsed right toolbar (hidden by default)
        self._right_toolbar = QToolBar("Layers & Properties", self)
        self._right_toolbar.setObjectName("RightToolBar")
        self._right_toolbar.setOrientation(Qt.Vertical)
        self._right_toolbar.setMovable(False)
        self._right_toolbar.setIconSize(QSize(28, 28))
        self._right_toolbar.setStyleSheet(
            "QToolBar { spacing: 2px; padding: 2px; }"
            "QToolButton { font-size: 18px; min-width: 36px; min-height: 32px; }"
        )
        self.addToolBar(Qt.RightToolBarArea, self._right_toolbar)
        self._right_toolbar.setVisible(False)
        self._right_collapsed = False

        _expand_left_ico = _qta_icon('fa6s.angles-left')
        expand_right_act = QAction("Expand") if not _expand_left_ico else QAction(_expand_left_ico, "", self)
        expand_right_act.setToolTip("Expand Layers & Properties (Ctrl+])")
        expand_right_act.triggered.connect(self._toggle_right_sidebar)
        self._right_toolbar.addAction(expand_right_act)

        # Full-width clickable header bar for right sidebar
        _collapse_right_ico = _qta_icon('fa6s.angles-right')
        if _collapse_right_ico:
            self._right_collapse_btn = QPushButton(_collapse_right_ico, " Layers && Properties")
        else:
            self._right_collapse_btn = QPushButton("Layers && Properties  \u25b6")
        self._right_collapse_btn.setFixedHeight(26)
        self._right_collapse_btn.setToolTip("Collapse sidebar (Ctrl+])")
        self._right_collapse_btn.setStyleSheet(_theme.collapse_btn_style())
        self._right_collapse_btn.clicked.connect(self._toggle_right_sidebar)
        right_header = QWidget()
        right_header_lay = QHBoxLayout(right_header)
        right_header_lay.setContentsMargins(2, 2, 2, 0)
        right_header_lay.addWidget(self._right_collapse_btn)
        self._layer_dock.setTitleBarWidget(right_header)

        # Connections
        self.tool_panel.collapse_requested.connect(self._toggle_left_sidebar)
        self.tool_panel.tool_changed.connect(self._on_tool_changed)
        self.tool_panel.open_montage_requested.connect(self.open_image)
        self.tool_panel.import_roi_zip_requested.connect(self.import_roi_zip)
        self.tool_panel.export_roi_zip_requested.connect(self.export_all_rois_zip)
        self.tool_panel.load_instructions_requested.connect(self.load_instructions_file)
        self.tool_panel.view_instructions_requested.connect(self._view_instructions)
        self.canvas.viewport_changed.connect(self._update_minimap_viewport)
        self.layer_panel.selection_changed.connect(self._on_layer_selected)
        self.layer_panel.visibility_changed.connect(self.canvas.refresh_overlays)
        self.layer_panel.roi_added.connect(self._on_roi_added)
        self.layer_panel.roi_removed.connect(self._on_roi_removed)
        self.layer_panel.all_cleared.connect(self._on_all_cleared)

    def _setup_menus(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")

        open_act = QAction("&Open Image(s)...", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self.open_image)
        file_menu.addAction(open_act)

        close_img_act = QAction("&Close Image(s)", self)
        close_img_act.setShortcut(QKeySequence("Ctrl+W"))
        close_img_act.triggered.connect(self.close_image)
        file_menu.addAction(close_img_act)

        file_menu.addSeparator()

        load_roi_act = QAction("Load &ROI Set (.npz)...", self)
        load_roi_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        load_roi_act.triggered.connect(self.load_rois)
        file_menu.addAction(load_roi_act)

        save_roi_act = QAction("&Save ROI Set (.npz)...", self)
        save_roi_act.setShortcut(QKeySequence.Save)
        save_roi_act.triggered.connect(self.save_rois)
        file_menu.addAction(save_roi_act)

        file_menu.addSeparator()

        # Import submenu
        import_menu = file_menu.addMenu("Import")

        import_ij_act = QAction("ROI from .roi File(s)...", self)
        import_ij_act.triggered.connect(self.import_imagej_roi)
        import_menu.addAction(import_ij_act)

        import_png_act = QAction("ROI from PNG Mask(s)...", self)
        import_png_act.triggered.connect(self.import_png_masks)
        import_menu.addAction(import_png_act)

        import_zip_act = QAction("ROI from ZIP (.roi + .png)...", self)
        import_zip_act.triggered.connect(self.import_roi_zip)
        import_menu.addAction(import_zip_act)

        # Export submenu
        export_menu = file_menu.addMenu("Export")

        export_ij_single_act = QAction("Active ROI as .roi...", self)
        export_ij_single_act.triggered.connect(self.export_active_imagej_roi)
        export_menu.addAction(export_ij_single_act)

        export_ij_act = QAction("All ROIs as .roi Files...", self)
        export_ij_act.triggered.connect(self.export_imagej_rois)
        export_menu.addAction(export_ij_act)

        export_act = QAction("All ROIs as PNG Mask(s)...", self)
        export_act.setShortcut(QKeySequence("Ctrl+E"))
        export_act.triggered.connect(self.export_roi_png)
        export_menu.addAction(export_act)

        export_menu.addSeparator()

        export_zip_act = QAction("All ROIs as ZIP (.roi)...", self)
        export_zip_act.triggered.connect(self.export_all_rois_zip)
        export_menu.addAction(export_zip_act)

        batch_export_act = QAction("Batch Export (choose format)...", self)
        batch_export_act.triggered.connect(self.batch_export_rois)
        export_menu.addAction(batch_export_act)

        file_menu.addSeparator()

        # Load instructions
        load_instr_act = QAction("Load Instructions...", self)
        load_instr_act.triggered.connect(self.load_instructions_file)
        file_menu.addAction(load_instr_act)

        view_instr_act = QAction("View Instructions...", self)
        view_instr_act.triggered.connect(self._view_instructions)
        file_menu.addAction(view_instr_act)

        file_menu.addSeparator()

        restore_session_act = QAction("Restore from Session...", self)
        restore_session_act.triggered.connect(self.restore_from_session)
        file_menu.addAction(restore_session_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # Edit
        edit_menu = menubar.addMenu("&Edit")

        undo_act = QAction("&Undo", self)
        undo_act.setShortcut(QKeySequence.Undo)
        undo_act.triggered.connect(self.undo)
        edit_menu.addAction(undo_act)

        redo_act = QAction("&Redo", self)
        redo_act.setShortcut(QKeySequence.Redo)
        redo_act.triggered.connect(self.redo)
        edit_menu.addAction(redo_act)

        # Per-ROI undo (Phase 4E)
        layer_undo_act = QAction("Layer &Undo", self)
        layer_undo_act.setShortcut(QKeySequence("Ctrl+Alt+Z"))
        layer_undo_act.triggered.connect(self.layer_undo)
        edit_menu.addAction(layer_undo_act)

        layer_redo_act = QAction("Layer R&edo", self)
        layer_redo_act.setShortcut(QKeySequence("Ctrl+Alt+Y"))
        layer_redo_act.triggered.connect(self.layer_redo)
        edit_menu.addAction(layer_redo_act)

        edit_menu.addSeparator()

        clear_act = QAction("&Delete Active ROI", self)
        clear_act.setShortcut(QKeySequence.Delete)
        clear_act.triggered.connect(self.clear_active_roi)
        edit_menu.addAction(clear_act)

        # Fix overlaps (Phase 4D)
        edit_menu.addSeparator()
        fix_overlaps_act = QAction("Fix &Overlaps (Later Wins)", self)
        fix_overlaps_act.triggered.connect(lambda: self.fix_overlaps("later_wins"))
        edit_menu.addAction(fix_overlaps_act)

        fix_overlaps_early_act = QAction("Fix Overlaps (&Earlier Wins)", self)
        fix_overlaps_early_act.triggered.connect(lambda: self.fix_overlaps("earlier_wins"))
        edit_menu.addAction(fix_overlaps_early_act)

        self._auto_overlap_act = QAction("Auto Overlap Fix", self)
        self._auto_overlap_act.setCheckable(True)
        self._auto_overlap_act.setChecked(False)
        self._auto_overlap_act.toggled.connect(self._toggle_auto_overlap)
        edit_menu.addAction(self._auto_overlap_act)

        edit_menu.addSeparator()

        select_all_act = QAction("Select &All ROIs", self)
        select_all_act.setShortcut(QKeySequence("Ctrl+A"))
        select_all_act.triggered.connect(self._select_all_rois)
        edit_menu.addAction(select_all_act)

        edit_menu.addSeparator()
        auto_fit_act = QAction("Auto-&Fit OOB ROIs", self)
        auto_fit_act.triggered.connect(self._auto_fit_rois)
        edit_menu.addAction(auto_fit_act)

        # View
        view_menu = menubar.addMenu("&View")

        fit_act = QAction("&Fit to Window", self)
        fit_act.setShortcut(QKeySequence("Ctrl+0"))
        fit_act.triggered.connect(self.canvas.fit_to_window)
        view_menu.addAction(fit_act)

        reset_act = QAction("&Reset Zoom (1:1)", self)
        reset_act.setShortcut(QKeySequence("Ctrl+1"))
        reset_act.triggered.connect(self.canvas.reset_zoom)
        view_menu.addAction(reset_act)

        zoom_in_act = QAction("Zoom &In", self)
        zoom_in_act.setShortcut(QKeySequence("Ctrl+="))
        zoom_in_act.triggered.connect(self.canvas.zoom_in)
        view_menu.addAction(zoom_in_act)

        zoom_out_act = QAction("Zoom &Out", self)
        zoom_out_act.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_act.triggered.connect(self.canvas.zoom_out)
        view_menu.addAction(zoom_out_act)

        view_menu.addSeparator()

        # Flip/Rotate (Phase 2D)
        flip_h_act = QAction("Flip &Horizontal", self)
        flip_h_act.setShortcut(QKeySequence("H"))
        flip_h_act.triggered.connect(self.flip_horizontal)
        view_menu.addAction(flip_h_act)

        rotate_act = QAction("Rotate &90 CW", self)
        rotate_act.setShortcut(QKeySequence("Ctrl+R"))
        rotate_act.triggered.connect(self.rotate_90)
        view_menu.addAction(rotate_act)

        view_menu.addSeparator()

        # Flip/Rotate on load (E.21, E.22)
        view_menu.addSeparator()
        self._flip_on_load_act = QAction("Flip on Load", self)
        self._flip_on_load_act.setCheckable(True)
        view_menu.addAction(self._flip_on_load_act)

        self._rotate_on_load_act = QAction("Rotate on Load", self)
        self._rotate_on_load_act.setCheckable(True)
        view_menu.addAction(self._rotate_on_load_act)

        view_menu.addSeparator()

        # Sidebar toggles
        collapse_left_act = QAction("Collapse &Left Sidebar", self)
        collapse_left_act.setShortcut(QKeySequence("Ctrl+["))
        collapse_left_act.triggered.connect(self._toggle_left_sidebar)
        view_menu.addAction(collapse_left_act)

        collapse_right_act = QAction("Collapse &Right Sidebar", self)
        collapse_right_act.setShortcut(QKeySequence("Ctrl+]"))
        collapse_right_act.triggered.connect(self._toggle_right_sidebar)
        view_menu.addAction(collapse_right_act)

        view_menu.addSeparator()

        # JIT acceleration toggle
        self._accel_act = QAction("&JIT Acceleration (Numba)", self)
        self._accel_act.setCheckable(True)
        self._accel_act.triggered.connect(self._toggle_accel)
        view_menu.addAction(self._accel_act)

        view_menu.addSeparator()

        # Dock toggles
        view_menu.addAction(self._details_dock.toggleViewAction())
        view_menu.addAction(self._minimap_dock.toggleViewAction())
        view_menu.addAction(self._perf_dock.toggleViewAction())
        view_menu.addAction(self._debug_dock.toggleViewAction())

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")

        # Theme submenu
        _theme_ico = _qta_icon('fa6s.palette')
        theme_menu = settings_menu.addMenu("Theme")
        if _theme_ico:
            theme_menu.setIcon(_theme_ico)
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        _theme_icons = {"dark": 'fa6s.moon', "light": 'fa6s.sun', "system": 'fa6s.desktop'}
        for key, label in [("dark", "Dark"), ("light", "Light"), ("system", "System")]:
            ico = _qta_icon(_theme_icons[key])
            act = QAction(ico, label, self) if ico else QAction(label, self)
            act.setCheckable(True)
            act.setData(key)
            self._theme_group.addAction(act)
            theme_menu.addAction(act)
        self._theme_group.triggered.connect(self._on_theme_changed)

        settings_menu.addSeparator()

        _student_ico = _qta_icon('fa6s.user-graduate')
        self._student_session_act = QAction("Student Session", self) if not _student_ico else QAction(_student_ico, "Student Session", self)
        self._student_session_act.setCheckable(True)
        self._student_session_act.setChecked(False)
        self._student_session_act.setToolTip(
            "Export ROIs to the same folder they were loaded from, prefixed with adj_")
        self._student_session_act.toggled.connect(self._on_student_session_toggled)
        settings_menu.addAction(self._student_session_act)

        _savep_ico = _qta_icon('fa6s.floppy-disk')
        self._save_progress_act = QAction(_savep_ico, "Save Progress", self) if _savep_ico else QAction("Save Progress", self)
        self._save_progress_act.setCheckable(True)
        self._save_progress_act.setChecked(False)
        self._save_progress_act.setToolTip("Enable/disable session save progress")
        self._save_progress_act.toggled.connect(self._on_save_progress_toggled)
        settings_menu.addAction(self._save_progress_act)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        guide_act = QAction("User &Guide", self)
        guide_act.setShortcut(QKeySequence("F1"))
        guide_act.triggered.connect(self._show_help)
        help_menu.addAction(guide_act)
        diag_act = QAction("Export &Diagnostics...", self)
        diag_act.triggered.connect(self._export_diagnostics)
        help_menu.addAction(diag_act)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.canvas.cursor_moved.connect(self._update_cursor_info)
        # Student Session indicator
        self._student_label = QLabel("")
        self._student_label.setStyleSheet(_theme.student_label_style())
        self._student_label.setVisible(False)
        self.statusbar.addPermanentWidget(self._student_label)
        # Tool status widget (G.14)
        self._tool_status_label = QLabel("Tool: Hand")
        self._tool_status_label.setStyleSheet(_theme.status_label_style())
        self.statusbar.addPermanentWidget(self._tool_status_label)

        # Move tool hint — shown in status bar only when Move is active
        self._move_hint = QLabel(
            "Hint: Drag outside to move all selected ROIs | "
            "Drag a component to move independently | "
            "Ctrl+click to multi-select"
        )
        self._move_hint.setStyleSheet(_theme.hint_style())
        self._move_hint.setVisible(False)
        self.statusbar.addPermanentWidget(self._move_hint)

    def _setup_toolbar(self):
        """Add main toolbar with brush size and opacity controls (G.19, G.20)."""
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setObjectName("MainToolbar")
        toolbar.setMovable(True)
        toolbar.setStyleSheet(
            "QToolBar { background: transparent; border: none; spacing: 0px; }"
        )
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # Single container with items centred
        tb_widget = QWidget()
        tb_widget.setStyleSheet("background: transparent;")
        tb_lay = QHBoxLayout(tb_widget)
        tb_lay.setContentsMargins(4, 4, 4, 4)
        tb_lay.setSpacing(6)
        tb_lay.addStretch(1)

        # Toolbar button style
        _tb_btn_style = _theme.toolbar_btn_style()
        self._themed_tb_btns = []  # track for theme refresh
        self._toolbar_groups = []  # track group frames for theme refresh
        self._toolbar_seps = []   # track separators for theme refresh

        # -- Group 1: Undo / Redo --
        _undo_ico = _qta_icon('fa6s.rotate-left')
        undo_btn = AnimatedButton(_undo_ico, " Undo") if _undo_ico else AnimatedButton("\u21A9  Undo")
        undo_btn.setToolTip("Undo (Ctrl+Z)")
        undo_btn.setStyleSheet(_tb_btn_style)
        undo_btn.clicked.connect(self.undo)
        _redo_ico = _qta_icon('fa6s.rotate-right')
        redo_btn = AnimatedButton(_redo_ico, " Redo") if _redo_ico else AnimatedButton("Redo  \u21AA")
        redo_btn.setToolTip("Redo (Ctrl+Shift+Z)")
        redo_btn.setStyleSheet(_tb_btn_style)
        redo_btn.clicked.connect(self.redo)
        self._themed_tb_btns.extend([undo_btn, redo_btn])
        tb_lay.addWidget(undo_btn, 0, Qt.AlignVCenter)
        tb_lay.addWidget(self._toolbar_sep(), 0, Qt.AlignVCenter)
        tb_lay.addWidget(redo_btn, 0, Qt.AlignVCenter)
        tb_lay.addWidget(self._toolbar_sep(), 0, Qt.AlignVCenter)

        # -- Group 2: Brush size --
        self._tb_size_slider = QSlider(Qt.Horizontal)
        self._tb_size_slider.setRange(1, 2000)
        self._tb_size_slider.setValue(100)
        self._tb_size_slider.setFixedWidth(120)
        self._tb_size_slider.setStyleSheet(_theme.slider_style())
        self._tb_size_spin = QSpinBox()
        self._tb_size_spin.setRange(1, 2000)
        self._tb_size_spin.setValue(100)
        self._tb_size_spin.setStyleSheet(_theme.spinbox_style())
        tb_lay.addWidget(
            self._toolbar_group(QLabel("Brush Size"), self._tb_size_slider, self._tb_size_spin),
            0, Qt.AlignVCenter,
        )
        tb_lay.addWidget(self._toolbar_sep(), 0, Qt.AlignVCenter)

        # Bidirectional sync between toolbar and tool_panel
        tp_slider = self.tool_panel.size_slider
        self._tb_size_slider.valueChanged.connect(tp_slider.setValue)
        tp_slider.valueChanged.connect(self._tb_size_slider.setValue)
        self._tb_size_spin.valueChanged.connect(self._tb_size_slider.setValue)
        self._tb_size_slider.valueChanged.connect(self._tb_size_spin.setValue)

        # -- Group 3: Global opacity --
        self._global_opacity_slider = QSlider(Qt.Horizontal)
        self._global_opacity_slider.setRange(0, 100)
        self._global_opacity_slider.setValue(100)
        self._global_opacity_slider.setFixedWidth(120)
        self._global_opacity_slider.setStyleSheet(_theme.slider_style())
        self._global_opacity_spin = QSpinBox()
        self._global_opacity_spin.setRange(0, 100)
        self._global_opacity_spin.setValue(100)
        self._global_opacity_spin.setSuffix("%")
        self._global_opacity_spin.setStyleSheet(_theme.spinbox_style())
        tb_lay.addWidget(
            self._toolbar_group(QLabel("Global Opacity"), self._global_opacity_slider, self._global_opacity_spin),
            0, Qt.AlignVCenter,
        )
        tb_lay.addWidget(self._toolbar_sep(), 0, Qt.AlignVCenter)

        self._global_opacity_slider.valueChanged.connect(self._global_opacity_spin.setValue)
        self._global_opacity_spin.valueChanged.connect(self._global_opacity_slider.setValue)
        self._global_opacity_slider.valueChanged.connect(self._on_global_opacity_changed)

        # -- Group 4: Document switcher --
        self._doc_combo = QComboBox()
        self._doc_combo.setFixedWidth(150)
        self._doc_combo.setStyleSheet(_theme.combobox_style())
        self._doc_combo.currentIndexChanged.connect(self._switch_to_document)

        self._tint_btn = AnimatedButton("Tint")
        self._tint_btn.setFixedWidth(50)
        self._tint_btn.setStyleSheet(_tb_btn_style)
        self._tint_btn.setToolTip("Set display tint color for current channel (right-click to clear)")
        self._tint_btn.clicked.connect(self._pick_tint_color)
        self._tint_btn.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tint_btn.customContextMenuRequested.connect(self._clear_tint_color)
        self._themed_tb_btns.append(self._tint_btn)
        tb_lay.addWidget(
            self._toolbar_group(QLabel("Montage"), self._doc_combo, self._tint_btn),
            0, Qt.AlignVCenter,
        )
        tb_lay.addWidget(self._toolbar_sep(), 0, Qt.AlignVCenter)

        # -- Group 5: Save Progress --
        _save_ico = _qta_icon('fa6s.floppy-disk')
        self._save_progress_btn = AnimatedButton(_save_ico, " Save Progress") if _save_ico else AnimatedButton("\u2B07  Save Progress")
        self._save_progress_btn.setToolTip("Ctrl+Shift+S")
        self._save_progress_btn.setStyleSheet(_tb_btn_style)
        self._save_progress_btn.clicked.connect(self.save_session_progress)
        self._themed_tb_btns.append(self._save_progress_btn)
        self._save_progress_btn.setVisible(False)
        self._save_progress_shortcut = QAction(self)
        self._save_progress_shortcut.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._save_progress_shortcut.triggered.connect(self.save_session_progress)
        self._save_progress_shortcut.setEnabled(False)
        self.addAction(self._save_progress_shortcut)
        tb_lay.addWidget(self._save_progress_btn, 0, Qt.AlignVCenter)
        tb_lay.addStretch(1)

        toolbar.addWidget(tb_widget)

    def _toolbar_group(self, *widgets):
        """Wrap widgets in a styled QFrame for visual grouping."""
        frame = QFrame()
        frame.setStyleSheet(_theme.toolbar_group_style())
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        for w in widgets:
            lay.addWidget(w)
        self._toolbar_groups.append(frame)
        return frame

    def _toolbar_sep(self):
        """Create a vertical pipe separator for the toolbar."""
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedWidth(2)
        sep.setFixedHeight(24)
        sep.setStyleSheet(
            f"color: {'#555' if _theme.is_dark() else '#bbb'};"
        )
        self._toolbar_seps.append(sep)
        return sep

    def _update_cursor_info(self, x, y, value):
        roi_info = ""
        if self.canvas._active_layer and getattr(self.canvas._active_layer, 'is_roi', False):
            mask = self.canvas._active_layer.mask
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                roi_info = f"  ROI: {'yes' if mask[y, x] > 0 else 'no'}"
        self.statusbar.showMessage(f"X: {x}  Y: {y}  Value: {value}{roi_info}")

    def _toggle_left_sidebar(self):
        self._left_collapsed = not getattr(self, '_left_collapsed', False)
        left_docks = [self._tool_dock, self._minimap_dock]
        if self._left_collapsed:
            # Hide docks, show icon toolbar
            for d in left_docks:
                d.setVisible(False)
            self._left_toolbar.setVisible(True)
        else:
            # Show docks, hide icon toolbar
            self._left_toolbar.setVisible(False)
            for d in left_docks:
                d.setVisible(True)

    def _toggle_right_sidebar(self):
        self._right_collapsed = not getattr(self, '_right_collapsed', False)
        right_docks = [self._layer_dock, self._details_dock]
        if self._right_collapsed:
            for d in right_docks:
                d.setVisible(False)
            self._right_toolbar.setVisible(True)
        else:
            self._right_toolbar.setVisible(False)
            for d in right_docks:
                d.setVisible(True)

    def _on_global_opacity_changed(self, value):
        """Update global opacity factor from toolbar slider."""
        self.layer_stack._global_opacity_factor = value / 100.0
        self.canvas.refresh_overlays_lut_only()

    def _toggle_accel(self, checked):
        """Toggle Numba JIT acceleration."""
        try:
            from montaris.core.accel import set_enabled, get_mode, HAS_NUMBA
            if checked and not HAS_NUMBA:
                self._accel_act.setChecked(False)
                QMessageBox.warning(self, "Numba Not Available",
                    "JIT acceleration requires the numba package.\n"
                    "Install with: pip install numba")
                return
            set_enabled(checked)
            mode = get_mode()
            self.settings.setValue("use_jit_accel", checked)
            if checked:
                self.toast.show(f"JIT acceleration enabled — {mode}", "success")
            else:
                self.toast.show("JIT acceleration disabled (numpy fallback)", "success")
        except ImportError:
            self._accel_act.setChecked(False)
            self.toast.show("Acceleration module not available", "error")
        # Re-render with new backend
        if self.layer_stack.image_layer is not None:
            self.canvas.refresh_overlays()

    def _on_tool_changed(self, tool):
        self.active_tool = tool
        self.canvas.set_tool(tool)
        # Update tool status (G.14)
        tool_name = getattr(tool, 'name', 'None') if tool else 'None'
        roi_info = ""
        if self.canvas._active_layer and hasattr(self.canvas._active_layer, 'name'):
            roi_info = f"  |  {self.canvas._active_layer.name}"
        self._tool_status_label.setText(f"Tool: {tool_name}{roi_info}")
        self._move_hint.setVisible(tool_name in ('Move (selected)', 'Move All'))
        # Sync collapsed toolbar actions
        for name, act in self._left_tool_actions.items():
            act.setChecked(name == tool_name)

    def _on_layer_selected(self, layer):
        self.canvas.set_active_layer(layer)
        self.properties_panel.set_layer(layer)
        # Update tool status (G.14)
        tool_name = getattr(self.active_tool, 'name', 'None') if self.active_tool else 'None'
        roi_info = f"  |  {layer.name}" if layer and hasattr(layer, 'name') else ""
        self._tool_status_label.setText(f"Tool: {tool_name}{roi_info}")

    _NON_DRAWING_TOOLS = {'Hand', 'Select ROI', 'Transform (selected)', 'Transform All',
                          'Move (selected)', 'Move All'}

    def _on_roi_added(self):
        if self.layer_stack.image_layer is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return
        h, w = self.layer_stack.image_layer.shape[:2]
        base = f"ROI {len(self.layer_stack.roi_layers) + 1}"
        name = generate_unique_roi_name(base, self.layer_stack.roi_layers)
        roi = ROILayer(name, w, h)
        self.layer_stack.add_roi(roi)
        self.canvas.refresh_overlays()
        self.layer_panel.refresh()

        # Auto-select the new ROI
        last_row = self.layer_panel.list_widget.count() - 1
        self.layer_panel.list_widget.setCurrentRow(last_row)
        # Switch to Brush only if current tool isn't a drawing tool
        current = getattr(self.canvas._tool, 'name', None)
        if current is None or current in self._NON_DRAWING_TOOLS:
            self.tool_panel._select_tool('Brush')

    def _on_roi_removed(self, index):
        removed = self.layer_stack.get_roi(index)
        self.layer_stack.remove_roi(index)
        # Clean stale reference from selection
        if removed and self.canvas._selection.contains(removed):
            self.canvas._selection.remove(removed)
        self.canvas.set_active_layer(None)
        self.canvas.refresh_overlays()
        self.layer_panel.refresh()
        self.properties_panel.set_layer(None)

    def _on_all_cleared(self):
        self.canvas._active_layer = None
        self.canvas._selection._layers.clear()
        self.canvas.refresh_overlays()
        self.layer_panel.refresh()
        self.properties_panel.set_layer(None)

    # -- Display mode callbacks (Phase 2) --

    def _on_display_mode_changed(self, mode):
        self._compositor.mode = mode
        if self._composite_mode:
            self._refresh_composite()
        else:
            self.canvas.refresh_image()

    def _on_channels_changed(self, active_indices):
        if self._composite_mode:
            self._refresh_composite()

    def _on_composite_toggled(self, checked):
        self._composite_mode = checked
        if checked:
            self._refresh_composite()
        else:
            # Restore single-channel view
            self.canvas.set_tint_color(self._get_active_tint())
            self.canvas.refresh_image()

    def _refresh_composite(self):
        """Compose all active channel documents into a single RGB image."""
        if not self._documents:
            return
        active = self.display_panel.get_active_channels()
        if not active:
            return
        ch_arrays = []
        for idx, i in enumerate(active):
            if i >= len(self._documents):
                continue
            doc = self._documents[i]
            data = doc.image_layer.data
            # Set LUT from tint color
            tint = doc.tint_color
            if tint is not None:
                self._compositor.set_lut(idx, (tint[0] / 255.0, tint[1] / 255.0, tint[2] / 255.0))
            else:
                self._compositor.set_lut(idx, (1.0, 1.0, 1.0))
            # Convert to 2D for compositor
            if data.ndim == 2:
                ch_arrays.append(data)
            else:
                ch_arrays.append(data.mean(axis=2).astype(data.dtype))
        if not ch_arrays:
            return
        composite = self._compositor.compose(ch_arrays)
        self.canvas.set_tint_color(None)
        self.canvas.refresh_image_from_array(composite)

    def _on_adjustments_changed(self, adjustments):
        self._adjustments = adjustments
        self.canvas._adjustments = adjustments
        # Debounce: coalesce rapid slider ticks into a single update
        if not hasattr(self, '_adj_timer'):
            self._adj_timer = QTimer(self)
            self._adj_timer.setSingleShot(True)
            self._adj_timer.setInterval(16)  # ~60fps cap
            self._adj_timer.timeout.connect(self._apply_adjustments)
        self._adj_timer.start()

    def _apply_adjustments(self):
        if self._composite_mode:
            self._refresh_composite()
        else:
            self.canvas.refresh_adjustments()

    def _on_minimap_pan(self, scene_x, scene_y):
        self.canvas.centerOn(scene_x, scene_y)

    def _update_minimap_viewport(self):
        viewport_rect = self.canvas.mapToScene(self.canvas.viewport().rect()).boundingRect()
        # Use image bounds, not sceneRect (which includes Qt padding)
        if self.canvas._image_item:
            image_rect = self.canvas._image_item.boundingRect()
        else:
            image_rect = self.canvas.sceneRect()
        self.minimap.update_viewport(viewport_rect, image_rect)

    # -- View transforms (Phase 2D) --

    def flip_horizontal(self):
        t = self.canvas.transform()
        t *= QTransform(-1, 0, 0, 1, 0, 0)
        self.canvas.setTransform(t)

    def rotate_90(self):
        self.canvas.rotate(90)

    # -- Fix overlaps (Phase 4D) --

    def fix_overlaps(self, priority="later_wins"):
        from montaris.core.roi_ops import fix_overlaps
        with busy_cursor("Fixing overlaps...", self, log_as="tool.fix_overlaps"):
            fix_overlaps(self.layer_stack.roi_layers, priority)
            self.canvas.refresh_overlays()
        self.statusbar.showMessage(f"Fixed overlaps ({priority})")

    def _toggle_auto_overlap(self, checked):
        self._auto_overlap = checked

    def _on_theme_changed(self, action):
        key = action.data()
        app = QApplication.instance()
        if key == "dark":
            apply_dark_theme(app)
        elif key == "light":
            apply_light_theme(app)
        elif key == "system":
            apply_system_theme(app)
        self._refresh_themed_styles()

    def _refresh_themed_styles(self):
        """Re-apply all hardcoded CSS after a theme switch."""
        for lbl in self._themed_labels:
            lbl.setStyleSheet(_theme.section_header_style())
        for btn in self._themed_tb_btns:
            btn.setStyleSheet(_theme.toolbar_btn_style())
        _grp_ss = _theme.toolbar_group_style()
        for grp in self._toolbar_groups:
            grp.setStyleSheet(_grp_ss)
        _sep_c = '#555' if _theme.is_dark() else '#bbb'
        for sep in self._toolbar_seps:
            sep.setStyleSheet(f"color: {_sep_c};")
        self._right_collapse_btn.setStyleSheet(_theme.collapse_btn_style())
        self._move_hint.setStyleSheet(_theme.hint_style())
        # Refresh icon colors across the app
        if _HAS_QTA:
            from montaris.widgets.tool_panel import _QTA_ICONS
            color = '#dcdcdc' if _theme.is_dark() else '#333'
            # Collapsed sidebar tool actions
            for name, act in self._left_tool_actions.items():
                if name in _QTA_ICONS:
                    act.setIcon(qta.icon(_QTA_ICONS[name], color=color))
            # Collapse / expand header buttons
            self._right_collapse_btn.setIcon(qta.icon('fa6s.angles-right', color=color))
        # Refresh child panels and canvas
        if hasattr(self, 'canvas'):
            self.canvas.refresh_theme()
        if hasattr(self, 'tool_panel'):
            self.tool_panel.refresh_theme()
        if hasattr(self, 'layer_panel'):
            self.layer_panel.refresh_theme()
        if hasattr(self, 'adjustments_panel'):
            self.adjustments_panel.refresh_theme()
        if hasattr(self, 'properties_panel'):
            self.properties_panel.refresh_theme()
        if hasattr(self, 'display_panel'):
            self.display_panel.refresh_theme()
        # Refresh toolbar sliders/spinboxes/combos
        _slider_ss = _theme.slider_style()
        _spin_ss = _theme.spinbox_style()
        self._tb_size_slider.setStyleSheet(_slider_ss)
        self._tb_size_spin.setStyleSheet(_spin_ss)
        self._global_opacity_slider.setStyleSheet(_slider_ss)
        self._global_opacity_spin.setStyleSheet(_spin_ss)
        self._doc_combo.setStyleSheet(_theme.combobox_style())
        _ch_ss = _theme.collapsible_header_style()
        for h in self._collapsible_headers:
            h.setStyleSheet(_ch_ss)
        # Refresh status bar labels
        self._student_label.setStyleSheet(_theme.student_label_style())
        self._tool_status_label.setStyleSheet(_theme.status_label_style())
        # Refresh peripheral widgets
        if hasattr(self, 'perf_monitor'):
            self.perf_monitor.refresh_theme()
        if hasattr(self, 'debug_console'):
            self.debug_console.refresh_theme()
        if hasattr(self, 'minimap'):
            self.minimap.refresh_theme()

    def _on_student_session_toggled(self, checked):
        self._student_session = checked
        self._student_label.setVisible(checked)
        self._student_label.setText("Student Session" if checked else "")
        if checked:
            self.toast.show("Student Session ON — exports save to import folder with adj_ prefix", "success")
        else:
            self.toast.show("Student Session OFF", "info")

    def _select_all_rois(self):
        visible = [r for r in self.layer_stack.roi_layers if r.visible]
        self.canvas._selection.select_all(visible)

    def _on_selection_count_changed(self, layers):
        count = len(layers)
        if count == 0:
            self.statusbar.showMessage("Selection cleared")
        elif count == 1:
            self.statusbar.showMessage(f"Selected: {layers[0].name}")
        else:
            self.statusbar.showMessage(f"Selected {count} ROIs")

    # -- Per-ROI undo (Phase 4E) --

    def layer_undo(self):
        layer = self.canvas._active_layer
        if layer and hasattr(layer, 'undo_stack'):
            if layer.undo_stack.undo():
                self.canvas.refresh_overlays()
                self.properties_panel.set_layer(layer)

    def layer_redo(self):
        layer = self.canvas._active_layer
        if layer and hasattr(layer, 'undo_stack'):
            if layer.undo_stack.redo():
                self.canvas.refresh_overlays()
                self.properties_panel.set_layer(layer)

    # -- File operations --

    def _last_dir(self):
        """Return the last used file dialog directory."""
        return self.settings.value("last_dir", "")

    def _update_last_dir(self, path):
        """Store the directory of *path* for next file dialog."""
        if path:
            d = os.path.dirname(path if isinstance(path, str) else path[0])
            if d:
                self.settings.setValue("last_dir", d)

    def open_image(self, paths=None):
        if not isinstance(paths, (list, tuple)):
            paths = None
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Open Image(s)", self._last_dir(),
                "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)",
            )
            if not paths:
                return
        self._initial_session_saved = False
        self._session_dir = None
        self._update_last_dir(paths)
        # Ask downsample once for the batch
        first_data = load_image(paths[0])
        ds_factor = 1
        if first_data.shape[0] * first_data.shape[1] > 4_000_000:
            items = ["1x (Original)", "2x", "4x", "8x"]
            item, ok = QInputDialog.getItem(
                self, "Downsample", "Image is large. Choose downsample factor:",
                items, 0, False,
            )
            if ok and item != items[0]:
                ds_factor = int(item[0])
        skipped = []
        with busy_cursor("Loading image...", self, log_as="io.open_image"):
            for path in paths:
                try:
                    # Split multi-channel TIFFs into individual images
                    channels = load_image_stack(path)
                    for name, data in channels:
                        self._load_single_channel(name, data, ds_factor, skipped, image_path=path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load {os.path.basename(path)}:\n{e}")
        if skipped:
            QMessageBox.warning(
                self, "Dimension Mismatch",
                f"Skipped {len(skipped)} image(s) with different dimensions:\n"
                + "\n".join(skipped),
            )
        n = len(self._documents)
        if n > 1:
            self.toast.show(f"Loaded {n} images in stack", "success")

        # Auto-load instructions file from the same folder
        folder = os.path.dirname(paths[0])
        self._auto_load_instructions(folder)

        # Auto-detect ROI ZIP in the same folder
        self._auto_detect_roi_zip(folder)

        # Auto-detect previous sessions
        self._auto_detect_session(folder)

    def _load_single_channel(self, name, data, ds_factor, skipped, image_path=None):
        """Load one image/channel into the image stack."""
        if self._flip_on_load_act.isChecked():
            data = np.flip(data, axis=1).copy()
        if self._rotate_on_load_act.isChecked():
            data = np.rot90(data, k=-1).copy()
        original_shape = data.shape
        if ds_factor > 1:
            data = data[::ds_factor, ::ds_factor]

        # Validate dimensions match existing stack
        cur = self.layer_stack.image_layer
        if cur is not None and data.shape[:2] != cur.data.shape[:2]:
            skipped.append(f"{name} ({data.shape[1]}x{data.shape[0]})")
            return

        self._save_current_document()
        new_layer = ImageLayer(name, data)

        if cur is not None:
            # Same dimensions — add to stack, keep ROIs
            self.layer_stack.image_layer = new_layer
            self.canvas.refresh_image()
        else:
            # First image — set image, preserve any existing ROIs
            self._downsample_factor = ds_factor
            has_rois = bool(self.layer_stack.roi_layers)
            if has_rois:
                self.layer_stack.image_layer = new_layer
                self.layer_stack.changed.emit()
            else:
                self.canvas._selection.clear()
                self.canvas._active_layer = None
                self.layer_stack.set_image(new_layer)
            self.canvas.refresh_image()
            self.canvas.fit_to_window()
            self.layer_panel.refresh()

        self.undo_stack.clear()
        self.minimap.set_image(data)
        self._update_minimap_viewport()
        self.adjustments_panel.set_image_data(data)

        doc = MontageDocument(
            name=name,
            image_layer=new_layer,
            downsample_factor=ds_factor,
            original_shape=original_shape,
            image_path=image_path,
        )
        self._documents.append(doc)
        self._active_doc_index = len(self._documents) - 1
        self._doc_combo.blockSignals(True)
        self._doc_combo.addItem(doc.name)
        self._doc_combo.setCurrentIndex(self._active_doc_index)
        self._doc_combo.blockSignals(False)

        self._update_display_channels()
        self.toast.show(f"Loaded: {name}  {data.shape}", "success")

    def _update_display_channels(self):
        """Update display panel channel list from loaded documents."""
        names = [doc.name for doc in self._documents]
        self.display_panel.set_channels(names)

    def close_image(self):
        """Close the current image, prompting user about ROIs."""
        if self.layer_stack.image_layer is None:
            return
        self._initial_session_saved = False
        self._session_dir = None

        clear_rois = False
        if self.layer_stack.roi_layers:
            from montaris.widgets.alert_modal import AlertModal
            result = AlertModal.confirm(
                self, "Close Image(s)",
                "You have ROIs. What would you like to do?",
                ["Save ROIs && Clear All", "Keep ROIs", "Cancel"],
            )
            if result == "Cancel" or result is None:
                return
            if result == "Save ROIs && Clear All":
                self.export_all_rois_zip()
                clear_rois = True

        # Clear image
        self.layer_stack.image_layer = None
        self.canvas.refresh_image()
        self.minimap.set_image(None)
        self.adjustments_panel.set_image_data(None)
        self._documents.clear()
        self._active_doc_index = -1
        self._doc_combo.blockSignals(True)
        self._doc_combo.clear()
        self._doc_combo.blockSignals(False)
        self._composite_mode = False
        self.display_panel.composite_cb.setChecked(False)
        self._update_display_channels()
        self.undo_stack.clear()

        if clear_rois:
            self.canvas._selection.clear()
            self.layer_stack.roi_layers.clear()
            self.layer_stack._color_index = 0
            self.canvas._active_layer = None
            self.properties_panel.set_layer(None)

        self.canvas.refresh_overlays()
        self.layer_panel.refresh()
        self.setWindowTitle("Montaris-X")
        self.statusbar.showMessage("Image closed" + (" (ROIs kept)" if not clear_rois else ""))

    def load_rois(self):
        if self.layer_stack.image_layer is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ROI Set", self._last_dir(),
            "ROI Files (*.npz);;All Files (*)",
        )
        if path:
            self._update_last_dir(path)
            try:
                with busy_cursor("Loading ROIs...", self, log_as="io.load_rois"):
                    rois = load_roi_set(path)
                    n = len(rois)
                    progress = None
                    if n > 5:
                        progress = QProgressDialog("Loading ROIs…", None, 0, n + 1, self)
                        progress.setWindowModality(Qt.WindowModal)
                        progress.setMinimumDuration(0)
                        progress.show()
                        QApplication.processEvents()
                    for i, roi in enumerate(rois):
                        self.layer_stack.add_roi(roi)
                        if progress:
                            progress.setValue(i + 1)
                    self._auto_fit_rois()
                    if progress:
                        progress.setLabelText("Rendering ROIs…")
                        progress.setValue(n)
                    self.canvas.refresh_overlays()
                    self.layer_panel.refresh()
                    if progress:
                        progress.close()
                self.statusbar.showMessage(f"Loaded {len(rois)} ROIs from {path}")
                QTimer.singleShot(0, lambda: self.layer_stack.compress_inactive(self.canvas._active_layer))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load ROIs:\n{e}")

    def save_rois(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to save.")
            return
        self._flatten_roi_offsets()
        resolution = self._ask_export_resolution()
        if resolution is None:
            return
        upscale = resolution == "original"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI Set", os.path.join(self._last_dir(), "rois.npz"),
            "NumPy Archive (*.npz);;All Files (*)",
        )
        if path:
            self._update_last_dir(path)
            try:
                n = len(self.layer_stack.roi_layers)
                progress = QProgressDialog("Saving ROIs (compressing)...", None, 0, n, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                _last_pe = [time.monotonic()]

                def _on_progress(i):
                    progress.setValue(i)
                    _last_pe[0] = should_process_events(_last_pe[0])

                mt = self._upscale_mask_if_needed if upscale else None
                with busy_cursor("Saving ROIs...", self, log_as="io.save_rois"):
                    save_roi_set(path, self.layer_stack.roi_layers, progress_callback=_on_progress, mask_transform=mt)
                progress.close()
                self.toast.show(f"Saved {len(self.layer_stack.roi_layers)} ROI(s)", "success")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save ROIs:\n{e}")

    def _flatten_roi_offsets(self):
        """Flatten all layer offsets before export/save operations."""
        with busy_cursor("Flattening offsets...", self, log_as="tool.flatten_offsets"):
            _last_pe = time.monotonic()
            for roi in self.layer_stack.roi_layers:
                roi.flatten_offset()
                _last_pe = should_process_events(_last_pe)

    # ── Session save / restore ──────────────────────────────────────

    def save_session_progress(self):
        """Save all ROIs as ImageJ .roi files in a timestamped session folder."""
        if not self.layer_stack.roi_layers:
            self.toast.show("No ROIs to save", "warning")
            return
        if self.layer_stack.image_layer is None:
            self.toast.show("No image loaded", "warning")
            return

        # Determine image directory
        doc = self._documents[self._active_doc_index] if self._documents else None
        image_path = doc.image_path if doc and doc.image_path else None
        if image_path:
            image_dir = os.path.dirname(image_path)
            stem = os.path.splitext(os.path.basename(image_path))[0]
        else:
            image_dir = self._last_dir()
            stem = doc.name if doc else "untitled"
            if not image_dir:
                self.toast.show("No directory to save session", "warning")
                return

        # Flatten offsets before snapshotting
        self._flatten_roi_offsets()

        # Snapshot ROI data on main thread — crop to bbox to avoid
        # decompressing full masks (which would spike memory for 100+ ROIs)
        from montaris.io.session import build_session_folder_name, get_base_stem
        snapshots = []
        for roi in self.layer_stack.roi_layers:
            bbox = roi.get_bbox()
            if bbox is None:
                continue  # skip empty masks
            # get_mask_crop works on compressed ROIs without full decompression
            crop = roi.get_mask_crop(bbox).copy()
            snapshots.append({
                'crop': crop,
                'name': roi.name,
                'color': list(roi.color),
                'opacity': roi.opacity,
                'bbox': bbox,
            })

        # Recompress ROIs that were decompressed during snapshotting
        QTimer.singleShot(0, lambda: self.layer_stack.compress_inactive(self.canvas._active_layer))

        if not snapshots:
            self.toast.show("All ROIs are empty — nothing to save", "warning")
            return

        if self._session_dir and os.path.isdir(self._session_dir):
            session_dir = self._session_dir
        else:
            folder_name = build_session_folder_name(stem)
            session_dir = os.path.join(image_dir, folder_name)
            self._session_dir = session_dir
        ds_factor = doc.downsample_factor if doc else 1
        original_shape = doc.original_shape if doc else None
        img = self.layer_stack.image_layer
        canvas_shape = list(img.data.shape[:2]) if img else None
        channel_names = [d.name for d in self._documents]
        base_stem = get_base_stem(stem)

        meta = {
            'version': 1,
            'timestamp': datetime.now().isoformat(),
            'image_stem': base_stem,
            'image_path': image_path or '',
            'downsample_factor': ds_factor,
            'original_shape': list(original_shape) if original_shape else None,
            'canvas_shape': canvas_shape,
            'channel_names': channel_names,
            'roi_count': len(snapshots),
            'roi_names': [s['name'] for s in snapshots],
            'roi_colors': [s['color'] for s in snapshots],
            'roi_opacities': [s['opacity'] for s in snapshots],
        }

        self.toast.show("Saving session...", "info")

        from montaris.core.workers import get_pool
        future = get_pool().submit(
            _save_session_from_snapshots, session_dir, snapshots, meta
        )

        n_saved = len(snapshots)

        def _poll_save():
            if not future.done():
                QTimer.singleShot(100, _poll_save)
                return
            try:
                future.result()
                self.toast.show(f"Session saved ({n_saved} ROIs)", "success")
            except Exception as exc:
                self.toast.show(f"Session save failed: {exc}", "error")

        QTimer.singleShot(100, _poll_save)

    def restore_from_session(self):
        """Restore ROIs from a previous session folder."""
        if self.layer_stack.image_layer is None:
            self.toast.show("Load an image first", "warning")
            return

        doc = self._documents[self._active_doc_index] if self._documents else None
        image_path = doc.image_path if doc and doc.image_path else None
        if image_path:
            image_dir = os.path.dirname(image_path)
            stem = os.path.splitext(os.path.basename(image_path))[0]
        else:
            image_dir = self._last_dir()
            stem = doc.name if doc else "untitled"

        if not image_dir:
            self.toast.show("No directory to search for sessions", "warning")
            return

        from montaris.io.session import find_sessions
        sessions = find_sessions(image_dir, stem)
        if not sessions:
            self.toast.show("No sessions found", "info")
            return

        # Build selection list
        items = []
        for folder, meta in sessions:
            ts = meta.get('timestamp', '?')
            count = meta.get('roi_count', '?')
            ds = meta.get('downsample_factor', 1)
            items.append(f"{ts}  —  {count} ROIs  (ds={ds}x)")

        item, ok = QInputDialog.getItem(
            self, "Restore from Session",
            f"Found {len(sessions)} session(s). Select one:",
            items, 0, False,
        )
        if not ok:
            return

        idx = items.index(item)
        folder, meta = sessions[idx]

        # Ask replace or keep
        action = self._ask_replace_or_keep()
        if action is None:
            return
        if action == "replace":
            self.canvas._selection.clear()
            self.layer_stack.roi_layers.clear()
            self.layer_stack._color_index = 0
            self.canvas._active_layer = None

        self._session_dir = folder
        self._restore_session_rois(folder, meta)

    def _restore_session_rois(self, folder, meta):
        """Import ROIs from a session folder."""
        from montaris.io.imagej_roi import read_imagej_roi, imagej_roi_to_mask, scale_roi_dict

        roi_names = meta.get('roi_names', [])
        roi_colors = meta.get('roi_colors', [])
        roi_opacities = meta.get('roi_opacities', [])
        roi_files = meta.get('roi_files')  # indexed filenames (v1.1+), or None
        session_ds = meta.get('downsample_factor', 1)

        doc = self._documents[self._active_doc_index] if self._documents else None
        current_ds = doc.downsample_factor if doc else 1
        img = self.layer_stack.image_layer
        img_h, img_w = img.data.shape[:2]

        # Calculate scale factor if ds factors differ
        need_scale = session_ds != current_ds
        if need_scale:
            scale = session_ds / current_ds  # e.g. session=2, current=1 → scale=2 (upscale)
        else:
            scale = 1.0

        progress = QProgressDialog("Restoring ROIs...", "Cancel", 0, len(roi_names), self)
        progress.setWindowTitle("Restore Session")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        insert_at = len(self.layer_stack.roi_layers)
        count = 0

        for i, name in enumerate(roi_names):
            if progress.wasCanceled():
                break

            # Resolve filename: prefer indexed roi_files, fall back to name-based
            if roi_files and i < len(roi_files):
                filename = roi_files[i]
                if filename is None:
                    continue  # was empty mask at save time
            else:
                safe_name = _sanitize_roi_filename(name)
                filename = f"{safe_name}.roi"

            roi_path = os.path.join(folder, filename)
            if not os.path.isfile(roi_path):
                continue

            try:
                roi_data = read_imagej_roi(roi_path)
                if need_scale:
                    roi_data = scale_roi_dict(roi_data, scale, scale)
                mask = imagej_roi_to_mask(roi_data, img_w, img_h)
                if not mask.any():
                    continue

                color = tuple(roi_colors[i]) if i < len(roi_colors) else ROI_COLORS[count % len(ROI_COLORS)]
                opacity = roi_opacities[i] if i < len(roi_opacities) else 128
                roi_name = generate_unique_roi_name(name, self.layer_stack.roi_layers)

                from montaris.core.rle import rle_encode
                rle_bytes, rle_shape = rle_encode(mask)
                del mask
                roi = ROILayer.__new__(ROILayer)
                roi.name = roi_name
                roi._mask = None
                roi._rle_data = rle_bytes
                roi._mask_shape = rle_shape
                roi.color = color
                roi.opacity = opacity
                roi.visible = True
                roi.fill_mode = "solid"
                roi._dirty_rect = None
                roi.offset_x = 0
                roi.offset_y = 0
                roi._cached_bbox = None
                roi._bbox_valid = False
                self.layer_stack.insert_roi(insert_at + count, roi)
                count += 1
            except Exception:
                continue  # skip corrupt .roi files

            progress.setValue(i + 1)
            QApplication.processEvents()

        progress.close()
        self.canvas.refresh_overlays()
        self.layer_panel.refresh()

        msg = f"Restored {count} ROI(s)"
        if need_scale:
            msg += f" (scaled from ds={session_ds}x to ds={current_ds}x)"
        self.toast.show(msg, "success")
        QTimer.singleShot(0, lambda: self.layer_stack.compress_inactive(self.canvas._active_layer))

    def _on_save_progress_toggled(self, checked):
        self._save_progress_btn.setVisible(checked)
        self._save_progress_shortcut.setEnabled(checked)

    def _auto_save_initial_session(self):
        """Auto-save session after ROI import (called once per image load)."""
        if not self._save_progress_act.isChecked():
            return
        if not self.layer_stack.roi_layers:
            return
        if getattr(self, '_initial_session_saved', False):
            return
        self._initial_session_saved = True
        self.save_session_progress()

    def _auto_detect_session(self, folder):
        """Prompt user if sessions are found for the current image."""
        if not self._documents:
            return
        # Skip if ROIs were already loaded (e.g. from auto-detect ZIP)
        if self.layer_stack.roi_layers:
            return
        doc = self._documents[self._active_doc_index]
        stem = doc.name

        from montaris.io.session import find_sessions
        sessions = find_sessions(folder, stem)
        if not sessions:
            return

        newest_folder, newest_meta = sessions[0]
        ts = newest_meta.get('timestamp', '?')
        count = newest_meta.get('roi_count', '?')

        reply = QMessageBox.question(
            self, "Session Found",
            f"Found {len(sessions)} session(s) for this image.\n"
            f"Most recent: {ts} ({count} ROIs)\n\nRestore most recent?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._session_dir = newest_folder
            self._restore_session_rois(newest_folder, newest_meta)

    def export_roi_png(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to export.")
            return
        self._flatten_roi_offsets()
        resolution = self._ask_export_resolution()
        if resolution is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ROI(s) as PNG", os.path.join(self._last_dir(), "roi_export.png"),
            "PNG (*.png);;All Files (*)",
        )
        if path:
            self._update_last_dir(path)
            self.export_roi_png_to(path, upscale=(resolution == "original"))

    def _upscale_mask_if_needed(self, mask):
        """Upscale mask if image was downsampled (A.14)."""
        if self._downsample_factor <= 1:
            return mask
        f = self._downsample_factor
        # Find original_shape from current document
        original_shape = None
        if 0 <= self._active_doc_index < len(self._documents):
            original_shape = self._documents[self._active_doc_index].original_shape
        upscaled = np.repeat(np.repeat(mask, f, axis=0), f, axis=1)
        if original_shape is not None:
            oh, ow = original_shape[:2]
            upscaled = upscaled[:oh, :ow]
        return upscaled

    def export_roi_png_to(self, path, upscale=True):
        try:
            from PIL import Image
            h, w = self.layer_stack.image_layer.shape[:2]

            rois = self.layer_stack.roi_layers
            n = len(rois)
            progress = QProgressDialog("Exporting PNG masks...", "Cancel", 0, n, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)

            base, ext = os.path.splitext(path)
            if not ext:
                ext = '.png'

            def _export_single_png(mask, out_path):
                img = Image.fromarray(mask)
                img.save(out_path)

            with busy_cursor("Exporting PNG masks...", self, log_as="io.export_png"):
                # Build jobs
                jobs = []
                for i, roi in enumerate(rois):
                    mask = self._get_export_mask(roi, upscale)
                    if n == 1:
                        out_path = base + ext
                    else:
                        safe_name = roi.name.replace("/", "_").replace("\\", "_")
                        out_path = f"{base}_{safe_name}{ext}"
                    jobs.append((mask, out_path))

                _last_pe = time.monotonic()
                if n > 3:
                    from montaris.core.workers import get_pool
                    futures = [get_pool().submit(_export_single_png, m, p) for m, p in jobs]
                    for i, fut in enumerate(futures):
                        if progress.wasCanceled():
                            return
                        fut.result()
                        progress.setValue(i + 1)
                        _last_pe = should_process_events(_last_pe)
                else:
                    for i, (mask, out_path) in enumerate(jobs):
                        if progress.wasCanceled():
                            return
                        _export_single_png(mask, out_path)
                        progress.setValue(i + 1)
                        _last_pe = should_process_events(_last_pe)

            progress.close()
            self.toast.show(f"Exported {n} PNG mask(s)", "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")

    # -- ImageJ ROI (Phase 7) --

    def import_imagej_roi(self):
        if self.layer_stack.image_layer is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import ImageJ ROI(s)", self._last_dir(),
            "ImageJ ROI (*.roi);;All Files (*)",
        )
        if not paths:
            return
        self._update_last_dir(paths)
        try:
            from montaris.io.imagej_roi import read_imagej_roi, imagej_roi_to_mask, scale_roi_dict
            from montaris.core.rle import rle_encode
            h, w = self.layer_stack.image_layer.shape[:2]
            ds = self._downsample_factor
            n = len(paths)
            # Insert after currently selected ROI, or append at end
            active = self.canvas._active_layer
            if active and active in self.layer_stack.roi_layers:
                insert_at = self.layer_stack.roi_layers.index(active) + 1
            else:
                insert_at = len(self.layer_stack.roi_layers)
            if n > 1:
                progress = QProgressDialog("Importing ImageJ ROIs…", "Cancel", 0, n + 1, self)
                progress.setWindowModality(Qt.WindowModal)
            else:
                progress = None

            def _decode_and_compress(path, tw, th, do_scale, sx, sy):
                """Read .roi file, rasterize mask, and RLE-compress immediately."""
                roi_dict = read_imagej_roi(path)
                if do_scale:
                    roi_dict = scale_roi_dict(roi_dict, sx, sy)
                mask = imagej_roi_to_mask(roi_dict, tw, th)
                t, b = roi_dict['top'], roi_dict['bottom']
                l, r = roi_dict['left'], roi_dict['right']
                t = max(0, t); l = max(0, l)
                b = min(th, b + 1); r = min(tw, r + 1)
                bbox = (t, b, l, r) if b > t and r > l else None
                rle_bytes, rle_shape = rle_encode(mask)
                return rle_bytes, rle_shape, bbox

            def _init_roi(name, rle_bytes, rle_shape, bbox):
                """Create ROILayer pre-compressed to minimize peak memory."""
                roi = ROILayer.__new__(ROILayer)
                roi.name = name
                roi._mask = None
                roi._rle_data = rle_bytes
                roi._mask_shape = rle_shape
                roi.color = ROI_COLORS[0]
                roi.opacity = 128
                roi.visible = True
                roi.fill_mode = "solid"
                roi._dirty_rect = None
                roi.offset_x = 0
                roi.offset_y = 0
                if bbox is not None:
                    roi._cached_bbox = bbox
                    roi._bbox_valid = True
                else:
                    roi._cached_bbox = None
                    roi._bbox_valid = False
                return roi

            need_scale = ds > 1
            sx = 1.0 / ds if need_scale else 1.0
            sy = 1.0 / ds if need_scale else 1.0
            count = 0

            with busy_cursor("Importing ImageJ ROIs...", self, log_as="io.import_imagej"):
                if n > 3:
                    from montaris.core.workers import get_pool, worker_count
                    pool = get_pool()
                    batch_sz = worker_count()
                    for batch_start in range(0, n, batch_sz):
                        batch = paths[batch_start:batch_start + batch_sz]
                        futures = [
                            (os.path.splitext(os.path.basename(p))[0],
                             pool.submit(_decode_and_compress, p, w, h, need_scale, sx, sy))
                            for p in batch
                        ]
                        for i, (name, fut) in enumerate(futures):
                            if progress and progress.wasCanceled():
                                break
                            rle_bytes, rle_shape, bbox = fut.result()
                            roi = _init_roi(name, rle_bytes, rle_shape, bbox)
                            self.layer_stack.insert_roi(insert_at + count, roi)
                            count += 1
                            if progress:
                                progress.setValue(batch_start + i + 1)
                                QApplication.processEvents()
                else:
                    for i, path in enumerate(paths):
                        if progress and progress.wasCanceled():
                            break
                        rle_bytes, rle_shape, bbox = _decode_and_compress(
                            path, w, h, need_scale, sx, sy)
                        roi = _init_roi(
                            os.path.splitext(os.path.basename(path))[0],
                            rle_bytes, rle_shape, bbox)
                        self.layer_stack.insert_roi(insert_at + count, roi)
                        count += 1
                        if progress:
                            progress.setValue(i + 1)
                if progress:
                    progress.setLabelText("Rendering ROIs…")
                    progress.setValue(n)
                self.canvas.refresh_overlays()
                self.layer_panel.refresh()
            if progress:
                progress.close()
            self.toast.show(f"Imported {count} ImageJ ROI(s)", "success")
            QTimer.singleShot(0, lambda: self.layer_stack.compress_inactive(self.canvas._active_layer))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import:\n{e}")

    def export_active_imagej_roi(self):
        roi = self.canvas._active_layer
        if not roi or not getattr(roi, 'is_roi', False):
            QMessageBox.information(self, "Info", "No active ROI selected.")
            return
        self._flatten_roi_offsets()
        resolution = self._ask_export_resolution()
        if resolution is None:
            return
        upscale = resolution == "original"
        safe_name = roi.name.replace("/", "_").replace("\\", "_")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Active ROI as .roi", os.path.join(self._last_dir(), f"{safe_name}.roi"),
            "ImageJ ROI (*.roi);;All Files (*)",
        )
        if not path:
            return
        try:
            from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi
            with busy_cursor("Exporting ImageJ ROI...", self, log_as="io.export_imagej_single"):
                mask = self._get_export_mask(roi, upscale)
                bbox = self._get_export_bbox(roi, upscale)
                roi_dict = mask_to_imagej_roi(mask, roi.name, bbox=bbox)
                if roi_dict:
                    write_imagej_roi(roi_dict, path)
            if roi_dict:
                self.toast.show(f"Exported {roi.name} as .roi", "success")
            else:
                QMessageBox.warning(self, "Warning", "ROI mask is empty, nothing to export.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")

    def export_imagej_rois(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to export.")
            return
        self._flatten_roi_offsets()
        resolution = self._ask_export_resolution()
        if resolution is None:
            return
        upscale = resolution == "original"
        dir_path = QFileDialog.getExistingDirectory(self, "Export ImageJ ROIs to Directory")
        if not dir_path:
            return
        try:
            from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi
            n = len(self.layer_stack.roi_layers)
            progress = QProgressDialog("Exporting ImageJ ROIs...", "Cancel", 0, n, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)

            def _export_single_imagej(mask, name, dp, bbox=None):
                roi_dict = mask_to_imagej_roi(mask, name, bbox=bbox)
                if roi_dict:
                    safe_name = name.replace("/", "_").replace("\\", "_")
                    write_imagej_roi(roi_dict, os.path.join(dp, f"{safe_name}.roi"))
                    return True
                return False

            with busy_cursor("Exporting ImageJ ROIs...", self, log_as="io.export_imagej_bulk"):
                _last_pe = time.monotonic()
                if upscale:
                    # Serial path: upscaled masks are large, process one at a time
                    count = 0
                    for i, roi in enumerate(self.layer_stack.roi_layers):
                        if progress.wasCanceled():
                            break
                        mask = self._get_export_mask(roi, True)
                        bbox = self._get_export_bbox(roi, True)
                        if _export_single_imagej(mask, roi.name, dir_path, bbox):
                            count += 1
                        del mask
                        progress.setValue(i + 1)
                        _last_pe = should_process_events(_last_pe)
                elif n > 3:
                    from montaris.core.workers import get_pool
                    futures = []
                    for roi in self.layer_stack.roi_layers:
                        fut = get_pool().submit(_export_single_imagej, roi.mask, roi.name, dir_path, roi.get_bbox())
                        futures.append(fut)
                    count = 0
                    for i, fut in enumerate(futures):
                        if progress.wasCanceled():
                            break
                        if fut.result():
                            count += 1
                        progress.setValue(i + 1)
                        _last_pe = should_process_events(_last_pe)
                else:
                    count = 0
                    for i, roi in enumerate(self.layer_stack.roi_layers):
                        if progress.wasCanceled():
                            break
                        if _export_single_imagej(roi.mask, roi.name, dir_path, roi.get_bbox()):
                            count += 1
                        progress.setValue(i + 1)
                        _last_pe = should_process_events(_last_pe)
            progress.close()
            self.toast.show(f"Exported {count} ImageJ ROI(s)", "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")

    def batch_export_rois(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to export.")
            return
        self._flatten_roi_offsets()
        resolution = self._ask_export_resolution()
        if resolution is None:
            return
        upscale = resolution == "original"
        path, filt = QFileDialog.getSaveFileName(
            self, "Batch Export ROIs", os.path.join(self._last_dir(), "rois"),
            "NumPy Archive (*.npz);;ImageJ ROI Directory;;PNG (*.png)",
        )
        if not path:
            return
        try:
            with busy_cursor("Batch exporting...", self, log_as="io.batch_export"):
                n = len(self.layer_stack.roi_layers)
                if filt.startswith("NumPy"):
                    progress = QProgressDialog("Saving ROIs (compressing)...", None, 0, n, self)
                    progress.setWindowModality(Qt.WindowModal)
                    progress.setMinimumDuration(0)
                    _last_pe = [time.monotonic()]

                    def _on_progress(i):
                        progress.setValue(i)
                        _last_pe[0] = should_process_events(_last_pe[0])

                    mt = self._upscale_mask_if_needed if upscale else None
                    save_roi_set(path, self.layer_stack.roi_layers, progress_callback=_on_progress, mask_transform=mt)
                    progress.close()
                elif filt.startswith("ImageJ"):
                    from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi
                    os.makedirs(path, exist_ok=True)
                    progress = QProgressDialog("Exporting ImageJ ROIs...", "Cancel", 0, n, self)
                    progress.setWindowModality(Qt.WindowModal)
                    progress.setMinimumDuration(0)
                    cancelled = False
                    _last_pe = time.monotonic()
                    for i, roi in enumerate(self.layer_stack.roi_layers):
                        if progress.wasCanceled():
                            cancelled = True
                            break
                        mask = self._get_export_mask(roi, upscale)
                        bbox = self._get_export_bbox(roi, upscale)
                        roi_dict = mask_to_imagej_roi(mask, roi.name, bbox=bbox)
                        if roi_dict:
                            safe_name = roi.name.replace("/", "_").replace("\\", "_")
                            write_imagej_roi(roi_dict, os.path.join(path, f"{safe_name}.roi"))
                        del mask
                        progress.setValue(i + 1)
                        _last_pe = should_process_events(_last_pe)
                    progress.close()
                    if cancelled:
                        self.toast.show("Export cancelled", "warning")
                        return
                elif filt.startswith("PNG"):
                    self.export_roi_png_to(path, upscale=upscale)
            self.toast.show(f"Batch exported to {os.path.basename(path)}", "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to batch export:\n{e}")

    def _auto_load_instructions(self, folder):
        """Auto-load .txt files containing 'instructions' in their name from folder."""
        try:
            matches = sorted(
                f for f in os.listdir(folder)
                if f.lower().endswith('.txt') and 'instruction' in f.lower()
            )
            if not matches:
                return
            parts = []
            for fname in matches:
                with open(os.path.join(folder, fname), 'r') as f:
                    parts.append(f"=== {fname} ===\n{f.read()}")
            self._last_instructions_text = "\n\n".join(parts) if len(parts) > 1 else parts[0].split("\n", 1)[1]
            names = ", ".join(matches)
            self.toast.show(f"Auto-loaded instructions: {names}", "success")
        except OSError:
            pass

    def _auto_detect_roi_zip(self, folder):
        """Prompt user if exactly one ROI .zip file is found in folder or subfolders."""
        try:
            zips = []
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith('.zip'):
                        zips.append(os.path.join(root, f))
            if len(zips) != 1:
                return
            zip_path = zips[0]
            display = os.path.relpath(zip_path, folder)
            reply = QMessageBox.question(
                self, "ROI ZIP detected",
                f"ROI ZIP detected:\n{display}\n\nImport?\n\n"
                f"Note: Auto-import only works when there is exactly one ZIP in the folder tree.",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.import_roi_zip(zip_path)
        except OSError:
            pass

    def load_instructions_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Instructions", self._last_dir(),
            "Text (*.txt);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, 'r') as f:
                self._last_instructions_text = f.read()
            self.toast.show(f"Loaded instructions: {os.path.basename(path)}", "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load instructions:\n{e}")

    # -- Replace / Keep dialog for imports --

    def _ask_replace_or_keep(self):
        """Ask whether to replace existing ROIs or keep and add. Returns 'replace', 'keep', or None."""
        if not self.layer_stack.roi_layers:
            return "keep"
        from montaris.widgets.alert_modal import AlertModal
        result = AlertModal.confirm(
            self, "Existing ROIs",
            "There are existing ROIs. What would you like to do?",
            ["Replace All", "Keep & Add New", "Cancel"],
        )
        if result == "Replace All":
            return "replace"
        elif result == "Keep & Add New":
            return "keep"
        return None

    def _ask_export_resolution(self):
        """Ask whether to export at original or current resolution.

        Returns 'original', 'current', or None (cancelled).
        Skips dialog silently when ds_factor == 1.
        """
        ds = self._downsample_factor
        if ds <= 1:
            return "current"
        h, w = self.layer_stack.image_layer.shape[:2]
        orig_w, orig_h = w * ds, h * ds
        original_shape = None
        if 0 <= self._active_doc_index < len(self._documents):
            original_shape = self._documents[self._active_doc_index].original_shape
        if original_shape:
            orig_h, orig_w = original_shape[:2]
        from montaris.widgets.alert_modal import AlertModal
        result = AlertModal.confirm(
            self, "Export Resolution",
            f"Image was downsampled {ds}x. Export at which resolution?",
            [f"Original ({orig_w}\u00d7{orig_h})", f"Current ({w}\u00d7{h})", "Cancel"],
        )
        if result and result.startswith("Original"):
            return "original"
        elif result and result.startswith("Current"):
            return "current"
        return None

    def _get_export_mask(self, roi, upscale):
        """Get mask, optionally upscaled to original resolution."""
        if not upscale or self._downsample_factor <= 1:
            return roi.mask
        return self._upscale_mask_if_needed(roi.mask)

    def _get_export_bbox(self, roi, upscale):
        """Get bbox, optionally scaled to original resolution."""
        bbox = roi.get_bbox()
        if not upscale or self._downsample_factor <= 1 or bbox is None:
            return bbox
        f = self._downsample_factor
        t, b, l, r = bbox
        scaled = (t * f, b * f, l * f, r * f)
        original_shape = None
        if 0 <= self._active_doc_index < len(self._documents):
            original_shape = self._documents[self._active_doc_index].original_shape
        if original_shape:
            oh, ow = original_shape[:2]
            scaled = (scaled[0], min(scaled[1], oh), scaled[2], min(scaled[3], ow))
        return scaled

    # -- Import PNG masks --

    def import_png_masks(self):
        if self.layer_stack.image_layer is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import PNG Masks", self._last_dir(),
            "PNG Masks (*.png);;All Files (*)",
        )
        if not paths:
            return
        self._update_last_dir(paths)
        action = self._ask_replace_or_keep()
        if action is None:
            return
        if action == "replace":
            self.layer_stack.roi_layers.clear()
            self.layer_stack._color_index = 0
            self.canvas._selection.clear()
        try:
            from PIL import Image
            h, w = self.layer_stack.image_layer.shape[:2]
            n = len(paths)
            # Insert after currently selected ROI, or append at end
            active = self.canvas._active_layer
            if active and active in self.layer_stack.roi_layers:
                insert_at = self.layer_stack.roi_layers.index(active) + 1
            else:
                insert_at = len(self.layer_stack.roi_layers)
            if n > 1:
                progress = QProgressDialog("Importing PNG masks...", "Cancel", 0, n, self)
                progress.setWindowModality(Qt.WindowModal)
            else:
                progress = None

            def _decode_png_compressed(p, tw, th):
                from montaris.core.rle import rle_encode
                img = Image.open(p).convert('L')
                arr = np.array(img)
                if arr.shape != (th, tw):
                    arr = np.array(img.resize((tw, th), Image.NEAREST))
                mask = (arr > 0).astype(np.uint8) * 255
                rle_bytes, rle_shape = rle_encode(mask)
                return rle_bytes, rle_shape

            def _init_roi(name, rle_bytes, rle_shape):
                roi = ROILayer.__new__(ROILayer)
                roi.name = name
                roi._mask = None
                roi._rle_data = rle_bytes
                roi._mask_shape = rle_shape
                roi.color = ROI_COLORS[0]
                roi.opacity = 128
                roi.visible = True
                roi.fill_mode = "solid"
                roi._dirty_rect = None
                roi.offset_x = 0
                roi.offset_y = 0
                roi._cached_bbox = None
                roi._bbox_valid = False
                return roi

            count = 0
            with busy_cursor("Importing PNG masks...", self, log_as="io.import_png"):
                if n > 3:
                    from montaris.core.workers import get_pool, worker_count
                    pool = get_pool()
                    batch_sz = worker_count()
                    for batch_start in range(0, n, batch_sz):
                        batch = paths[batch_start:batch_start + batch_sz]
                        futures = [
                            (os.path.splitext(os.path.basename(p))[0],
                             pool.submit(_decode_png_compressed, p, w, h))
                            for p in batch
                        ]
                        for i, (name, fut) in enumerate(futures):
                            if progress and progress.wasCanceled():
                                break
                            rle_bytes, rle_shape = fut.result()
                            roi = _init_roi(name, rle_bytes, rle_shape)
                            self.layer_stack.insert_roi(insert_at + count, roi)
                            count += 1
                            if progress:
                                progress.setValue(batch_start + i + 1)
                                QApplication.processEvents()
                else:
                    for i, path in enumerate(paths):
                        if progress and progress.wasCanceled():
                            break
                        rle_bytes, rle_shape = _decode_png_compressed(path, w, h)
                        name = os.path.splitext(os.path.basename(path))[0]
                        roi = _init_roi(name, rle_bytes, rle_shape)
                        self.layer_stack.insert_roi(insert_at + count, roi)
                        count += 1
                        if progress:
                            progress.setValue(i + 1)

            if progress:
                progress.close()
            self.canvas.refresh_overlays()
            self.layer_panel.refresh()
            self.toast.show(f"Imported {len(paths)} PNG mask(s)", "success")
            QTimer.singleShot(0, lambda: self.layer_stack.compress_inactive(self.canvas._active_layer))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import PNG masks:\n{e}")

    # -- Import ROI ZIP --

    def import_roi_zip(self, path=None):
        if self.layer_stack.image_layer is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Import ROI ZIP", self._last_dir(),
                "ZIP Archive (*.zip);;All Files (*)",
            )
            if not path:
                return
            self._update_last_dir(path)
        self._roi_import_path = path  # remember source for Student Session export
        action = self._ask_replace_or_keep()
        if action is None:
            return
        if action == "replace":
            self.layer_stack.roi_layers.clear()
            self.layer_stack._color_index = 0
            self.canvas._selection.clear()
        _t0 = time.perf_counter()
        try:
            import zipfile
            from montaris.io.imagej_roi import read_imagej_roi, imagej_roi_to_mask
            from PIL import Image
            import io as _io
            img_h, img_w = self.layer_stack.image_layer.shape[:2]

            # First pass: scan .roi files to find max extent
            max_w, max_h = img_w, img_h
            roi_entries = []
            with zipfile.ZipFile(path, 'r') as zf:
                names = zf.namelist()
                for idx, name in enumerate(names):
                    lower = name.lower()
                    data = zf.read(name)
                    base = os.path.splitext(os.path.basename(name))[0]
                    if lower.endswith('.roi'):
                        roi_dict = read_imagej_roi(data)
                        max_w = max(max_w, roi_dict['right'])
                        max_h = max(max_h, roi_dict['bottom'])
                        if roi_dict.get('x_coords') is not None:
                            max_w = max(max_w, int(roi_dict['x_coords'].max()) + 1)
                            max_h = max(max_h, int(roi_dict['y_coords'].max()) + 1)
                        if roi_dict.get('paths'):
                            for sub_path in roi_dict['paths']:
                                for x, y in sub_path:
                                    max_w = max(max_w, int(x) + 1)
                                    max_h = max(max_h, int(y) + 1)
                        roi_entries.append(('roi', base, roi_dict))
                    elif lower.endswith('.png'):
                        roi_entries.append(('png', base, data))

            # Scale ROI coordinates to fit image if needed
            w, h = img_w, img_h
            ds = self._downsample_factor
            if ds > 1:
                need_scale = True
                scale_x = 1.0 / ds
                scale_y = 1.0 / ds
            else:
                need_scale = max_w > img_w or max_h > img_h
                if need_scale:
                    scale_x = img_w / max(max_w, 1)
                    scale_y = img_h / max(max_h, 1)

            # Insert after currently selected ROI, or append at end
            active = self.canvas._active_layer
            if active and active in self.layer_stack.roi_layers:
                insert_at = self.layer_stack.roi_layers.index(active) + 1
            else:
                insert_at = len(self.layer_stack.roi_layers)

            QApplication.setOverrideCursor(Qt.WaitCursor)
            count = 0
            total = len(roi_entries)
            progress = QProgressDialog(f"Importing {total} ROI(s)...", "Cancel", 0, total, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            QApplication.processEvents()

            def _decode_roi_entry(entry_type, payload, tw, th, sx, sy, do_scale):
                from montaris.core.rle import rle_encode
                if entry_type == 'roi':
                    roi_dict = payload
                    if do_scale:
                        from montaris.io.imagej_roi import scale_roi_dict
                        roi_dict = scale_roi_dict(roi_dict, sx, sy)
                    mask = imagej_roi_to_mask(roi_dict, tw, th)
                    # Pre-compute bbox from roi_dict to avoid full-mask scan.
                    # Pad bottom/right by +1 because rasterization uses +2
                    # beyond coordinate max, so mask content can extend 1px
                    # past the ImageJ header bbox.
                    t, b = roi_dict['top'], roi_dict['bottom']
                    l, r = roi_dict['left'], roi_dict['right']
                    t = max(0, t); l = max(0, l)
                    b = min(th, b + 1); r = min(tw, r + 1)
                    bbox = (t, b, l, r) if b > t and r > l else None
                    # RLE-compress immediately to avoid holding full mask in memory
                    rle_bytes, rle_shape = rle_encode(mask)
                    return rle_bytes, rle_shape, bbox
                else:  # png
                    img = Image.open(_io.BytesIO(payload)).convert('L')
                    arr = np.array(img)
                    if arr.shape != (th, tw):
                        arr = np.array(img.resize((tw, th), Image.NEAREST))
                    mask = (arr > 0).astype(np.uint8) * 255
                    rle_bytes, rle_shape = rle_encode(mask)
                    return rle_bytes, rle_shape, None

            sx = scale_x if need_scale else 1.0
            sy = scale_y if need_scale else 1.0

            def _init_roi(name, rle_bytes, rle_shape, bbox):
                """Create ROILayer pre-compressed to minimize peak memory."""
                roi = ROILayer.__new__(ROILayer)
                roi.name = name
                roi._mask = None
                roi._rle_data = rle_bytes
                roi._mask_shape = rle_shape
                roi.color = ROI_COLORS[0]
                roi.opacity = 128
                roi.visible = True
                roi.fill_mode = "solid"
                roi._dirty_rect = None
                roi.offset_x = 0
                roi.offset_y = 0
                # Pre-cache bbox to avoid full-mask scan
                if bbox is not None:
                    roi._cached_bbox = bbox
                    roi._bbox_valid = True
                else:
                    roi._cached_bbox = None
                    roi._bbox_valid = False
                return roi

            if total > 3:
                from montaris.core.workers import get_pool, worker_count
                pool = get_pool()
                batch_sz = worker_count()
                for batch_start in range(0, total, batch_sz):
                    batch = roi_entries[batch_start:batch_start + batch_sz]
                    futures = [
                        (base, pool.submit(_decode_roi_entry, et, pl, w, h, sx, sy, need_scale))
                        for et, base, pl in batch
                    ]
                    for i, (base, fut) in enumerate(futures):
                        if progress.wasCanceled():
                            break
                        rle_bytes, rle_shape, bbox = fut.result()
                        roi = _init_roi(base, rle_bytes, rle_shape, bbox)
                        self.layer_stack.insert_roi(insert_at + count, roi)
                        count += 1
                        progress.setValue(batch_start + i + 1)
                        QApplication.processEvents()
            else:
                for i, (entry_type, base, payload) in enumerate(roi_entries):
                    if progress.wasCanceled():
                        break
                    rle_bytes, rle_shape, bbox = _decode_roi_entry(entry_type, payload, w, h, sx, sy, need_scale)
                    roi = _init_roi(base, rle_bytes, rle_shape, bbox)
                    self.layer_stack.insert_roi(insert_at + count, roi)
                    count += 1
                    progress.setValue(i + 1)
                    QApplication.processEvents()

            progress.setLabelText("Rendering ROIs…")
            progress.setRange(0, 0)  # indeterminate
            QApplication.processEvents()
            self.canvas.refresh_overlays()
            self.layer_panel.refresh()
            progress.close()
            QApplication.restoreOverrideCursor()
            msg = f"Imported {count} ROI(s) from ZIP"
            if need_scale:
                msg += f" (scaled from {max_w}\u00d7{max_h} to {img_w}\u00d7{img_h})"
            self.toast.show(msg, "success")
            from montaris.core.event_logger import EventLogger
            EventLogger.instance().log("io", "import_roi_zip",
                duration_ms=(time.perf_counter() - _t0) * 1000, count=count)
            QTimer.singleShot(0, lambda: self.layer_stack.compress_inactive(self.canvas._active_layer))
            QTimer.singleShot(500, self._auto_save_initial_session)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Failed to import ZIP:\n{e}")

    # -- Export all ROIs as ZIP --

    def export_all_rois_zip(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to export.")
            return
        self._flatten_roi_offsets()
        resolution = self._ask_export_resolution()
        if resolution is None:
            return
        upscale = resolution == "original"
        # Student Session: auto-save to import directory with adj_ prefix
        if self._student_session and self._roi_import_path:
            import_dir = os.path.dirname(self._roi_import_path)
            import_name = os.path.basename(self._roi_import_path)
            path = os.path.join(import_dir, f"adj_{import_name}")
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export All ROIs as ZIP", os.path.join(self._last_dir(), "rois.zip"),
                "ZIP Archive (*.zip);;All Files (*)",
            )
            if not path:
                return
        try:
            import zipfile
            from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi_bytes
            n = len(self.layer_stack.roi_layers)
            progress = QProgressDialog("Exporting ROIs to ZIP...", "Cancel", 0, n, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)

            def _compute_roi_bytes(mask, name, bbox=None):
                roi_dict = mask_to_imagej_roi(mask, name, bbox=bbox)
                if roi_dict:
                    safe = name.replace("/", "_").replace("\\", "_")
                    return (f"{safe}.roi", write_imagej_roi_bytes(roi_dict))
                return None

            # Compute in parallel, write to zipfile sequentially (not thread-safe)
            with busy_cursor("Exporting ROIs to ZIP...", self, log_as="io.export_zip"):
                _last_pe = time.monotonic()
                if upscale:
                    # Serial: upscaled masks are large, compute+write one at a time
                    results = []
                    cancelled = False
                    for i, roi in enumerate(self.layer_stack.roi_layers):
                        if progress.wasCanceled():
                            cancelled = True
                            break
                        mask = self._get_export_mask(roi, True)
                        bbox = self._get_export_bbox(roi, True)
                        results.append(_compute_roi_bytes(mask, roi.name, bbox))
                        del mask
                        progress.setValue(i + 1)
                        _last_pe = should_process_events(_last_pe)
                elif n > 3:
                    from montaris.core.workers import get_pool
                    futures = [
                        get_pool().submit(_compute_roi_bytes, roi.mask, roi.name, roi.get_bbox())
                        for roi in self.layer_stack.roi_layers
                    ]
                    results = []
                    cancelled = False
                    for i, fut in enumerate(futures):
                        if progress.wasCanceled():
                            cancelled = True
                            break
                        results.append(fut.result())
                        progress.setValue(i + 1)
                        _last_pe = should_process_events(_last_pe)
                else:
                    results = []
                    cancelled = False
                    for i, roi in enumerate(self.layer_stack.roi_layers):
                        if progress.wasCanceled():
                            cancelled = True
                            break
                        results.append(_compute_roi_bytes(roi.mask, roi.name, roi.get_bbox()))
                        progress.setValue(i + 1)
                        _last_pe = should_process_events(_last_pe)

            if not cancelled:
                progress.setLabelText("Writing ZIP file...")
                progress.setValue(0)
                progress.setMaximum(len(results))
                _last_pe = time.monotonic()
                with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for j, entry in enumerate(results):
                        if entry is not None:
                            zf.writestr(entry[0], entry[1])
                        progress.setValue(j + 1)
                        _last_pe = should_process_events(_last_pe)

            progress.close()
            if cancelled:
                if os.path.exists(path):
                    os.remove(path)
                self.toast.show("Export cancelled", "warning")
            else:
                self.toast.show(f"Exported {n} ROI(s) to ZIP", "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export ZIP:\n{e}")

    # -- Auto-fit OOB ROIs --

    # -- Multi-document support (A.10-A.14) --

    def _save_current_document(self):
        """Save current app state into the active MontageDocument."""
        if self._active_doc_index < 0 or self._active_doc_index >= len(self._documents):
            return
        doc = self._documents[self._active_doc_index]
        doc.image_layer = self.layer_stack.image_layer
        doc.adjustments = {
            'brightness': self._adjustments.brightness,
            'contrast': self._adjustments.contrast,
            'exposure': self._adjustments.exposure,
            'gamma': self._adjustments.gamma,
        }
        doc.downsample_factor = self._downsample_factor

    def _switch_to_document(self, index):
        """Switch to a different MontageDocument (shared ROIs, swap image only)."""
        if index < 0 or index >= len(self._documents):
            return
        if index == self._active_doc_index:
            return
        # Save current
        self._save_current_document()
        # Load new — only swap image, ROIs stay shared
        doc = self._documents[index]
        self._active_doc_index = index
        self._downsample_factor = doc.downsample_factor
        self.layer_stack.image_layer = doc.image_layer
        # Restore adjustments before refresh so the image renders correctly
        from montaris.core.adjustments import ImageAdjustments
        self._adjustments = ImageAdjustments(**doc.adjustments)
        self.canvas._adjustments = self._adjustments
        if hasattr(self.adjustments_panel, '_adjustments'):
            self.adjustments_panel._adjustments = self._adjustments
            self.adjustments_panel._sync_sliders()
        self.canvas.set_tint_color(doc.tint_color)
        self.canvas.refresh_image()
        self.canvas.refresh_overlays()
        self.undo_stack.clear()
        if doc.image_layer:
            self.minimap.set_image(doc.image_layer.data)
            self.adjustments_panel.set_image_data(doc.image_layer.data)
        self._update_tint_btn()
        self.toast.show(f"Switched to: {doc.name}", "info")

    def _get_active_tint(self):
        """Return tint_color of the active document, or None."""
        if 0 <= self._active_doc_index < len(self._documents):
            return self._documents[self._active_doc_index].tint_color
        return None

    def _pick_tint_color(self):
        if self._active_doc_index < 0 or self._active_doc_index >= len(self._documents):
            return
        doc = self._documents[self._active_doc_index]
        initial = QColor(*(doc.tint_color or (255, 255, 255)))
        color = QColorDialog.getColor(initial, self, "Channel Tint Color")
        if color.isValid():
            doc.tint_color = (color.red(), color.green(), color.blue())
            self._update_tint_btn()
            if self._composite_mode:
                self._refresh_composite()
            else:
                self.canvas.set_tint_color(doc.tint_color)
                self.canvas.refresh_image()

    def _clear_tint_color(self):
        if self._active_doc_index < 0 or self._active_doc_index >= len(self._documents):
            return
        doc = self._documents[self._active_doc_index]
        doc.tint_color = None
        self._update_tint_btn()
        if self._composite_mode:
            self._refresh_composite()
        else:
            self.canvas.set_tint_color(None)
            self.canvas.refresh_image()

    def _update_tint_btn(self):
        tint = self._get_active_tint()
        if tint:
            r, g, b = tint
            text_color = "white" if (r * 0.299 + g * 0.587 + b * 0.114) < 128 else "black"
            self._tint_btn.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); color: {text_color};"
            )
        else:
            self._tint_btn.setStyleSheet(_theme.toolbar_btn_style())

    def _auto_fit_rois(self):
        if self.layer_stack.image_layer is None:
            return
        from montaris.core.roi_ops import auto_fit_rois
        h, w = self.layer_stack.image_layer.shape[:2]
        count = auto_fit_rois(self.layer_stack.roi_layers, w, h)
        if count > 0:
            self.canvas.refresh_overlays()
            self.toast.show(f"Auto-fitted {count} ROI(s)", "success")

    # -- View instructions --

    def _show_help(self):
        from montaris.widgets.help_modal import HelpModal
        dlg = HelpModal(self)
        dlg.exec()

    def _export_diagnostics(self):
        import json
        from datetime import datetime
        from montaris.core.event_logger import EventLogger
        default_name = f"montaris_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Diagnostics", os.path.join(self._last_dir(), default_name),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        data = EventLogger.instance().export_json(app=self)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        self.toast.show(f"Exported {data['session']['total_events']} events", "success")

    def _view_instructions(self):
        text = getattr(self, '_last_instructions_text', None)
        if text is None:
            QMessageBox.information(self, "Info", "No instructions loaded yet.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Instructions")
        dlg.resize(800, 600)
        layout = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(text)
        layout.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        layout.addWidget(bb)
        dlg.exec()

    def clear_active_roi(self):
        """Remove selected ROI(s), or the active ROI if none selected."""
        try:
            selected = self.canvas._selection.layers
            if not selected:
                layer = self.canvas._active_layer
                if layer and getattr(layer, 'is_roi', False) and layer in self.layer_stack.roi_layers:
                    selected = [layer]
            if not selected:
                return
            # Get indices to remove (reverse order to avoid shifting)
            indices = []
            for layer in selected:
                if layer in self.layer_stack.roi_layers:
                    indices.append(self.layer_stack.roi_layers.index(layer))
            if not indices:
                return
            first_idx = min(indices)
            # Block signals during batch removal to prevent cascading updates
            # on stale layer references
            self.canvas._active_layer = None
            self.canvas._selection._layers.clear()  # silent clear, no signal
            # Batch remove
            for idx in sorted(indices, reverse=True):
                if 0 <= idx < len(self.layer_stack.roi_layers):
                    self.layer_stack.roi_layers.pop(idx)
            # Now emit signals after state is consistent
            self.layer_stack.changed.emit()
            # Select adjacent ROI if available
            if self.layer_stack.roi_layers:
                new_idx = min(first_idx, len(self.layer_stack.roi_layers) - 1)
                self.canvas.set_active_layer(self.layer_stack.roi_layers[new_idx])
            self.canvas.refresh_overlays()
            self.layer_panel.refresh()
            self.properties_panel.set_layer(self.canvas._active_layer)
        except Exception:
            import traceback
            traceback.print_exc()

    def undo(self):
        # If polygon tool is mid-drawing, undo last vertex instead
        tool = self.canvas._tool
        if (tool is not None and getattr(tool, 'name', None) == 'Polygon'
                and getattr(tool, '_vertices', None)):
            tool._vertices.pop()
            if tool._vertices:
                self.canvas.draw_polygon_preview(tool._vertices)
            else:
                self.canvas.clear_polygon_preview()
            return
        cmd = self.undo_stack.undo()
        if cmd:
            from montaris.core.event_logger import EventLogger
            EventLogger.instance().log("undo", "undo", bytes=getattr(cmd, 'byte_size', 0))
            self._refresh_affected_layers(cmd)
            self._auto_select_roi_from_command(cmd)
            self.layer_panel.refresh()

    def redo(self):
        cmd = self.undo_stack.redo()
        if cmd:
            from montaris.core.event_logger import EventLogger
            EventLogger.instance().log("undo", "redo", bytes=getattr(cmd, 'byte_size', 0))
            self._refresh_affected_layers(cmd)
            self._auto_select_roi_from_command(cmd)
            self.layer_panel.refresh()

    def _refresh_affected_layers(self, cmd):
        """Refresh only the layers affected by an undo/redo command."""
        layers = set()
        if hasattr(cmd, 'commands'):
            # CompoundUndoCommand
            for sub in cmd.commands:
                if hasattr(sub, 'roi_layer') and sub.roi_layer is not None:
                    layers.add(sub.roi_layer)
        elif hasattr(cmd, '_entries'):
            # SnapshotUndoCommand (entries are 4-tuples: layer, bbox, old, new)
            for entry in cmd._entries:
                layers.add(entry[0])
        elif hasattr(cmd, 'roi_layer') and cmd.roi_layer is not None:
            layers.add(cmd.roi_layer)

        if layers:
            for layer in layers:
                self.canvas.refresh_active_overlay(layer)
        else:
            self.canvas.refresh_overlays()

    def _auto_select_roi_from_command(self, cmd):
        """Switch active layer to the ROI affected by an undo/redo command."""
        roi = getattr(cmd, 'roi_layer', None)
        if roi is None:
            if self.canvas._active_layer:
                self.properties_panel.set_layer(self.canvas._active_layer)
            return
        # Find index and switch
        try:
            idx = self.layer_stack.roi_layers.index(roi)
        except ValueError:
            if self.canvas._active_layer:
                self.properties_panel.set_layer(self.canvas._active_layer)
            return
        self.canvas.set_active_layer(roi)
        self.properties_panel.set_layer(roi)
        # Update layer panel selection (row 0 is image, roi rows start at 1)
        row = idx + (1 if self.layer_stack.image_layer else 0)
        self.layer_panel._updating = True
        self.layer_panel.list_widget.setCurrentRow(row)
        self.layer_panel._updating = False

    def _restore_state(self):
        geom = self.settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)
            # Clamp restored geometry to available screen to avoid Qt warnings
            screen = self.screen() or QApplication.primaryScreen()
            if screen:
                avail = screen.availableGeometry()
                fg = self.frameGeometry()
                if fg.width() > avail.width() or fg.height() > avail.height():
                    w = min(fg.width(), avail.width())
                    h = min(fg.height(), avail.height())
                    self.resize(w, h)
                # Also ensure window isn't positioned off-screen
                pos = self.pos()
                x = max(avail.x(), min(pos.x(), avail.right() - self.width()))
                y = max(avail.y(), min(pos.y(), avail.bottom() - self.height()))
                if pos.x() != x or pos.y() != y:
                    self.move(x, y)
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
        # restoreState may make toolbars visible from saved state.
        # Force: docks expanded, toolbars hidden on launch.
        self._left_toolbar.setVisible(False)
        self._left_collapsed = False
        for d in [self._tool_dock, self._minimap_dock]:
            d.setVisible(True)

        self._right_toolbar.setVisible(False)
        self._right_collapsed = False
        for d in [self._layer_dock, self._details_dock]:
            d.setVisible(True)

        # Restore JIT acceleration setting
        try:
            from montaris.core.accel import set_enabled, HAS_NUMBA, is_enabled
            use_jit = self.settings.value("use_jit_accel", True, type=bool)
            set_enabled(use_jit and HAS_NUMBA)
            self._accel_act.setChecked(is_enabled())
        except ImportError:
            self._accel_act.setChecked(False)

        # Restore Settings menu preferences
        self._student_session_act.setChecked(
            self.settings.value("student_session", False, type=bool))
        self._save_progress_act.setChecked(
            self.settings.value("save_progress", False, type=bool))

        # Restore theme selection (check radio button; theme itself applied at startup)
        saved_theme = self.settings.value("theme", "dark")
        for act in self._theme_group.actions():
            if act.data() == saved_theme:
                act.setChecked(True)
                break

    def _apply_dock_widths(self):
        """Set compact right sidebar width after layout is ready."""
        right_docks = [self._layer_dock, self._details_dock]
        self.resizeDocks(right_docks, [220] * len(right_docks), Qt.Horizontal)

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("student_session", self._student_session_act.isChecked())
        self.settings.setValue("save_progress", self._save_progress_act.isChecked())
        checked_theme = self._theme_group.checkedAction()
        self.settings.setValue("theme", checked_theme.data() if checked_theme else "dark")
        from montaris.core.workers import shutdown_pool
        shutdown_pool()
        super().closeEvent(event)


def _install_crash_handler(dump_dir):
    """Install a global exception hook that writes crash dumps to *dump_dir*.

    Captures:
    - Python exceptions (sys.excepthook)
    - C-level segfaults (faulthandler)
    - Unhandled exceptions in Qt slots (sys.unraisablehook)
    """
    import traceback as _tb
    import faulthandler as _fh
    os.makedirs(dump_dir, exist_ok=True)

    # Persistent file for faulthandler (segfaults)
    _fh_path = os.path.join(dump_dir, "segfault.log")
    _fh_file = open(_fh_path, 'w')
    _fh.enable(file=_fh_file, all_threads=True)

    _original_hook = sys.excepthook

    def _crash_hook(exc_type, exc_value, exc_tb):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        dump_path = os.path.join(dump_dir, f"crash_{ts}.log")
        try:
            lines = _tb.format_exception(exc_type, exc_value, exc_tb)
            with open(dump_path, 'w') as f:
                f.write(f"Montaris-X crash dump — {datetime.now().isoformat()}\n")
                f.write(f"Python {sys.version}\n")
                f.write(f"Platform: {sys.platform}\n\n")
                f.write("".join(lines))
            print(f"\n[CRASH] Dump written to: {dump_path}", file=sys.stderr)
            print("".join(lines), file=sys.stderr)
        except Exception:
            pass
        _original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _crash_hook

    # Catch exceptions swallowed by Qt slots (shows as "unraisable")
    _original_unraisable = getattr(sys, 'unraisablehook', None)

    def _unraisable_hook(unraisable):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        dump_path = os.path.join(dump_dir, f"unraisable_{ts}.log")
        try:
            lines = _tb.format_exception(
                type(unraisable.exc_value), unraisable.exc_value,
                unraisable.exc_traceback,
            )
            with open(dump_path, 'w') as f:
                f.write(f"Montaris-X unraisable exception — {datetime.now().isoformat()}\n")
                f.write(f"Object: {unraisable.object}\n")
                f.write(f"Message: {unraisable.err_msg}\n\n")
                f.write("".join(lines))
            print(f"\n[UNRAISABLE] Dump written to: {dump_path}", file=sys.stderr)
            print("".join(lines), file=sys.stderr)
        except Exception:
            pass
        if _original_unraisable:
            _original_unraisable(unraisable)

    sys.unraisablehook = _unraisable_hook


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="montaris",
        description="Montaris-X — PySide6 ROI editor for scientific images",
    )
    parser.add_argument(
        "--crash-dump", metavar="DIR",
        help="Enable crash dump logging to DIR (creates dir if needed)",
    )
    parser.add_argument(
        "--crash-dump-default", action="store_true",
        help="Enable crash dump logging to ~/.montaris/crash_dumps/",
    )
    args, qt_args = parser.parse_known_args()

    # Windows: set AppUserModelID so taskbar uses our icon, not Python's
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('montaris.montaris-x')
        except Exception:
            pass

    dump_dir = args.crash_dump
    if not dump_dir and args.crash_dump_default:
        dump_dir = os.path.join(os.path.expanduser("~"), ".montaris", "crash_dumps")
    if dump_dir:
        _install_crash_handler(dump_dir)
        print(f"[Montaris-X] Crash dumps enabled → {dump_dir}", file=sys.stderr)

    app = QApplication([sys.argv[0]] + qt_args)
    app.setApplicationName("Montaris-X")
    app.setOrganizationName("Montaris")

    _logo = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(_logo):
        app.setWindowIcon(QIcon(_logo))

    _settings = QSettings("Montaris", "Montaris-X")
    _theme = _settings.value("theme", "dark")
    if _theme == "light":
        apply_light_theme(app)
    elif _theme == "system":
        apply_system_theme(app)
    else:
        apply_dark_theme(app)
    window = MontarisApp()
    window.show()

    # Background warmup of JIT kernels
    try:
        from montaris.core.accel import warmup, HAS_NUMBA
        if HAS_NUMBA:
            from montaris.core.workers import get_pool
            get_pool().submit(warmup)
    except ImportError:
        pass

    sys.exit(app.exec())
