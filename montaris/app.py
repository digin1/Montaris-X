import sys
import os
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QFileDialog,
    QStatusBar, QMessageBox, QProgressDialog, QDialog, QVBoxLayout, QTextEdit,
    QDialogButtonBox, QToolBar, QLabel, QSlider, QSpinBox, QHBoxLayout, QWidget, QPushButton,
    QComboBox, QInputDialog, QColorDialog,
)
from PySide6.QtCore import Qt, QSettings, QRectF, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPalette, QColor, QTransform, QShortcut, QIcon

from montaris.canvas import ImageCanvas
from montaris.layers import LayerStack, ImageLayer, ROILayer, generate_unique_roi_name, MontageDocument
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
        self._documents = []
        self._active_doc_index = -1
        self._composite_mode = False

        self._setup_canvas()
        self._setup_panels()
        self._setup_menus()
        self._setup_statusbar()

        self.toast = ToastManager(self)
        self._setup_toolbar()

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

        # Tool panel
        self.tool_panel = ToolPanel(self, self)
        tool_dock = QDockWidget("Tools", self)
        tool_dock.setObjectName("ToolsDock")
        tool_dock.setWidget(self.tool_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, tool_dock)
        self._tool_dock = tool_dock

        # Properties panel
        self.properties_panel = PropertiesPanel(self, self)
        props_dock = QDockWidget("Properties", self)
        props_dock.setObjectName("PropertiesDock")
        props_dock.setWidget(self.properties_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, props_dock)
        self._props_dock = props_dock

        # Display panel (Phase 2)
        self.display_panel = DisplayPanel(self)
        display_dock = QDockWidget("Display", self)
        display_dock.setObjectName("DisplayDock")
        display_dock.setWidget(self.display_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, display_dock)
        self.display_panel.mode_changed.connect(self._on_display_mode_changed)
        self.display_panel.channels_changed.connect(self._on_channels_changed)
        self.display_panel.composite_toggled.connect(self._on_composite_toggled)
        self._display_dock = display_dock

        # Adjustments panel (Phase 2)
        self.adjustments_panel = AdjustmentsPanel(self)
        adj_dock = QDockWidget("Adjustments", self)
        adj_dock.setObjectName("AdjustmentsDock")
        adj_dock.setWidget(self.adjustments_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, adj_dock)
        self.adjustments_panel.adjustments_changed.connect(self._on_adjustments_changed)
        self._adj_dock = adj_dock

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
        self._left_toolbar = QToolBar("Tools", self)
        self._left_toolbar.setObjectName("LeftToolBar")
        self._left_toolbar.setOrientation(Qt.Vertical)
        self._left_toolbar.setMovable(False)
        self.addToolBar(Qt.LeftToolBarArea, self._left_toolbar)
        self._left_toolbar.setVisible(False)
        self._left_collapsed = False

        from montaris.widgets.tool_panel import TOOL_ICONS
        expand_left_act = QAction("\u25b6", self)  # ▶
        expand_left_act.setToolTip("Expand sidebar (Ctrl+[)")
        expand_left_act.triggered.connect(self._toggle_left_sidebar)
        self._left_toolbar.addAction(expand_left_act)
        self._left_toolbar.addSeparator()

        self._left_tool_actions = {}
        from montaris.tools import TOOL_REGISTRY
        for name, (module, cls_name, shortcut, category) in TOOL_REGISTRY.items():
            icon = TOOL_ICONS.get(name, shortcut)
            act = QAction(f"{icon}", self)
            act.setToolTip(f"{name} [{shortcut}]")
            act.setCheckable(True)
            act.triggered.connect(lambda checked, t=name: self.tool_panel._select_tool(t))
            self._left_toolbar.addAction(act)
            self._left_tool_actions[name] = act

        # Collapsed right toolbar (hidden by default)
        self._right_toolbar = QToolBar("Panels", self)
        self._right_toolbar.setObjectName("RightToolBar")
        self._right_toolbar.setOrientation(Qt.Vertical)
        self._right_toolbar.setMovable(False)
        self.addToolBar(Qt.RightToolBarArea, self._right_toolbar)
        self._right_toolbar.setVisible(False)
        self._right_collapsed = False

        expand_right_act = QAction("\u25c0", self)  # ◀
        expand_right_act.setToolTip("Expand sidebar (Ctrl+])")
        expand_right_act.triggered.connect(self._toggle_right_sidebar)
        self._right_toolbar.addAction(expand_right_act)

        # Collapse button inside right sidebar (added to layer dock title)
        self._right_collapse_btn = QPushButton("\u25b6")  # ▶
        self._right_collapse_btn.setFixedSize(22, 22)
        self._right_collapse_btn.setToolTip("Collapse sidebar (Ctrl+])")
        self._right_collapse_btn.setStyleSheet(
            "QPushButton { border: none; font-size: 12px; padding: 2px; }"
        )
        self._right_collapse_btn.clicked.connect(self._toggle_right_sidebar)
        collapse_widget = QWidget()
        collapse_layout = QHBoxLayout(collapse_widget)
        collapse_layout.setContentsMargins(0, 0, 0, 0)
        collapse_layout.addWidget(self._right_collapse_btn)
        collapse_layout.addStretch()
        self._layer_dock.setTitleBarWidget(collapse_widget)

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

        # Dock toggles
        view_menu.addAction(self._display_dock.toggleViewAction())
        view_menu.addAction(self._adj_dock.toggleViewAction())
        view_menu.addAction(self._minimap_dock.toggleViewAction())
        view_menu.addAction(self._perf_dock.toggleViewAction())
        view_menu.addAction(self._debug_dock.toggleViewAction())

        # Help menu
        help_menu = menubar.addMenu("&Help")
        guide_act = QAction("User &Guide", self)
        guide_act.setShortcut(QKeySequence("F1"))
        guide_act.triggered.connect(self._show_help)
        help_menu.addAction(guide_act)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.canvas.cursor_moved.connect(self._update_cursor_info)
        # Tool status widget (G.14)
        self._tool_status_label = QLabel("Tool: Hand")
        self._tool_status_label.setStyleSheet("font-size: 11px; padding: 0 8px;")
        self.statusbar.addPermanentWidget(self._tool_status_label)

        # Move tool hint — shown in status bar only when Move is active
        self._move_hint = QLabel(
            "Hint: Drag outside to move all selected ROIs | "
            "Drag a component to move independently | "
            "Ctrl+click to multi-select"
        )
        self._move_hint.setStyleSheet("color: #888; font-size: 11px; padding: 0 8px;")
        self._move_hint.setVisible(False)
        self.statusbar.addPermanentWidget(self._move_hint)

    def _setup_toolbar(self):
        """Add main toolbar with brush size and opacity controls (G.19, G.20)."""
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setObjectName("MainToolbar")
        toolbar.setMovable(True)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # Undo / Redo buttons (shortcuts already in Edit menu)
        undo_btn = QAction("⟲ Undo", self)
        undo_btn.setToolTip("Undo (Ctrl+Z)")
        undo_btn.triggered.connect(self.undo)
        toolbar.addAction(undo_btn)

        redo_btn = QAction("⟳ Redo", self)
        redo_btn.setToolTip("Redo (Ctrl+Shift+Z)")
        redo_btn.triggered.connect(self.redo)
        toolbar.addAction(redo_btn)

        toolbar.addSeparator()

        # Brush size in toolbar — synced with tool_panel
        toolbar.addWidget(QLabel(" Brush Size: "))
        tb_size_slider = QSlider(Qt.Horizontal)
        tb_size_slider.setRange(1, 2000)
        tb_size_slider.setValue(100)
        tb_size_slider.setFixedWidth(120)
        toolbar.addWidget(tb_size_slider)

        tb_size_spin = QSpinBox()
        tb_size_spin.setRange(1, 2000)
        tb_size_spin.setValue(100)
        toolbar.addWidget(tb_size_spin)

        # Bidirectional sync between toolbar and tool_panel
        tp_slider = self.tool_panel.size_slider
        tb_size_slider.valueChanged.connect(tp_slider.setValue)
        tp_slider.valueChanged.connect(tb_size_slider.setValue)
        tb_size_spin.valueChanged.connect(tb_size_slider.setValue)
        tb_size_slider.valueChanged.connect(tb_size_spin.setValue)

        toolbar.addSeparator()

        # ROI opacity in toolbar — synced with properties_panel
        toolbar.addWidget(QLabel(" Opacity: "))
        tb_opacity_slider = QSlider(Qt.Horizontal)
        tb_opacity_slider.setRange(0, 255)
        tb_opacity_slider.setValue(128)
        tb_opacity_slider.setFixedWidth(120)
        toolbar.addWidget(tb_opacity_slider)

        tb_opacity_spin = QSpinBox()
        tb_opacity_spin.setRange(0, 255)
        tb_opacity_spin.setValue(128)
        toolbar.addWidget(tb_opacity_spin)

        pp_slider = self.properties_panel.opacity_slider
        tb_opacity_slider.valueChanged.connect(pp_slider.setValue)
        pp_slider.valueChanged.connect(tb_opacity_slider.setValue)
        tb_opacity_spin.valueChanged.connect(tb_opacity_slider.setValue)
        tb_opacity_slider.valueChanged.connect(tb_opacity_spin.setValue)

        toolbar.addSeparator()

        # Document switcher (A.11)
        toolbar.addWidget(QLabel(" Montage: "))
        self._doc_combo = QComboBox()
        self._doc_combo.setMinimumWidth(150)
        self._doc_combo.currentIndexChanged.connect(self._switch_to_document)
        toolbar.addWidget(self._doc_combo)

        self._tint_btn = QPushButton("Tint")
        self._tint_btn.setFixedWidth(50)
        self._tint_btn.setToolTip("Set display tint color for current channel (right-click to clear)")
        self._tint_btn.clicked.connect(self._pick_tint_color)
        self._tint_btn.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tint_btn.customContextMenuRequested.connect(self._clear_tint_color)
        toolbar.addWidget(self._tint_btn)

    def _update_cursor_info(self, x, y, value):
        roi_info = ""
        if self.canvas._active_layer and hasattr(self.canvas._active_layer, 'mask'):
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
        right_docks = [
            self._layer_dock, self._props_dock,
            self._display_dock, self._adj_dock,
        ]
        if self._right_collapsed:
            for d in right_docks:
                d.setVisible(False)
            self._right_toolbar.setVisible(True)
        else:
            self._right_toolbar.setVisible(False)
            for d in right_docks:
                d.setVisible(True)

    def _on_tool_changed(self, tool):
        self.active_tool = tool
        self.canvas.set_tool(tool)
        # Update tool status (G.14)
        tool_name = getattr(tool, 'name', 'None') if tool else 'None'
        roi_info = ""
        if self.canvas._active_layer and hasattr(self.canvas._active_layer, 'name'):
            roi_info = f"  |  {self.canvas._active_layer.name}"
        self._tool_status_label.setText(f"Tool: {tool_name}{roi_info}")
        self._move_hint.setVisible(tool_name == 'Move')
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

    _NON_DRAWING_TOOLS = {'Hand', 'Select', 'Transform', 'Move'}

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
        self.canvas._selection.clear()
        self.canvas.set_active_layer(None)
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
        fix_overlaps(self.layer_stack.roi_layers, priority)
        self.canvas.refresh_overlays()
        self.statusbar.showMessage(f"Fixed overlaps ({priority})")

    def _toggle_auto_overlap(self, checked):
        self._auto_overlap = checked

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

    def open_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Image(s)", "",
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if not paths:
            return
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
        for path in paths:
            try:
                # Split multi-channel TIFFs into individual images
                channels = load_image_stack(path)
                for name, data in channels:
                    self._load_single_channel(name, data, ds_factor, skipped)
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
        self._auto_load_instructions(os.path.dirname(paths[0]))

    def _load_single_channel(self, name, data, ds_factor, skipped):
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
            self, "Load ROI Set", "",
            "ROI Files (*.npz);;All Files (*)",
        )
        if path:
            try:
                rois = load_roi_set(path)
                for roi in rois:
                    self.layer_stack.add_roi(roi)
                self._auto_fit_rois()
                self.canvas.refresh_overlays()
                self.layer_panel.refresh()
                self.statusbar.showMessage(f"Loaded {len(rois)} ROIs from {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load ROIs:\n{e}")

    def save_rois(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to save.")
            return
        self._flatten_roi_offsets()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI Set", "rois.npz",
            "NumPy Archive (*.npz);;All Files (*)",
        )
        if path:
            try:
                save_roi_set(path, self.layer_stack.roi_layers)
                self.toast.show(f"Saved {len(self.layer_stack.roi_layers)} ROI(s)", "success")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save ROIs:\n{e}")

    def _flatten_roi_offsets(self):
        """Flatten all layer offsets before export/save operations."""
        for roi in self.layer_stack.roi_layers:
            roi.flatten_offset()

    def export_roi_png(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to export.")
            return
        self._flatten_roi_offsets()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ROI(s) as PNG", "roi_export.png",
            "PNG (*.png);;All Files (*)",
        )
        if path:
            self.export_roi_png_to(path)

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

    def export_roi_png_to(self, path):
        try:
            from PIL import Image
            h, w = self.layer_stack.image_layer.shape[:2]

            rois = self.layer_stack.roi_layers
            n = len(rois)
            if n > 1:
                progress = QProgressDialog("Exporting PNG masks...", "Cancel", 0, n, self)
                progress.setWindowModality(Qt.WindowModal)
            else:
                progress = None

            base, ext = os.path.splitext(path)
            if not ext:
                ext = '.png'

            def _export_single_png(mask, out_path):
                img = Image.fromarray(mask)
                img.save(out_path)

            # Build jobs
            jobs = []
            for i, roi in enumerate(rois):
                mask = self._upscale_mask_if_needed(roi.mask)
                if n == 1:
                    out_path = base + ext
                else:
                    safe_name = roi.name.replace("/", "_").replace("\\", "_")
                    out_path = f"{base}_{safe_name}{ext}"
                jobs.append((mask, out_path))

            if n > 3:
                from montaris.core.workers import get_pool
                futures = [get_pool().submit(_export_single_png, m, p) for m, p in jobs]
                for i, fut in enumerate(futures):
                    if progress and progress.wasCanceled():
                        return
                    fut.result()
                    if progress:
                        progress.setValue(i + 1)
            else:
                for i, (mask, out_path) in enumerate(jobs):
                    if progress and progress.wasCanceled():
                        return
                    _export_single_png(mask, out_path)
                    if progress:
                        progress.setValue(i + 1)

            if progress:
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
            self, "Import ImageJ ROI(s)", "",
            "ImageJ ROI (*.roi);;All Files (*)",
        )
        if not paths:
            return
        try:
            from montaris.io.imagej_roi import read_imagej_roi, imagej_roi_to_mask
            h, w = self.layer_stack.image_layer.shape[:2]
            n = len(paths)
            # Insert after currently selected ROI, or append at end
            active = self.canvas._active_layer
            if active and active in self.layer_stack.roi_layers:
                insert_at = self.layer_stack.roi_layers.index(active) + 1
            else:
                insert_at = len(self.layer_stack.roi_layers)
            if n > 1:
                progress = QProgressDialog("Importing ImageJ ROIs...", "Cancel", 0, n, self)
                progress.setWindowModality(Qt.WindowModal)
            else:
                progress = None
            for i, path in enumerate(paths):
                if progress and progress.wasCanceled():
                    break
                roi_dict = read_imagej_roi(path)
                mask = imagej_roi_to_mask(roi_dict, w, h)
                name = os.path.splitext(os.path.basename(path))[0]
                roi = ROILayer(name, w, h)
                roi.mask = mask
                self.layer_stack.insert_roi(insert_at + i, roi)
                if progress:
                    progress.setValue(i + 1)
            if progress:
                progress.close()
            self.canvas.refresh_overlays()
            self.layer_panel.refresh()
            self.toast.show(f"Imported {len(paths)} ImageJ ROI(s)", "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import:\n{e}")

    def export_active_imagej_roi(self):
        roi = self.canvas._active_layer
        if not roi or not hasattr(roi, 'mask'):
            QMessageBox.information(self, "Info", "No active ROI selected.")
            return
        self._flatten_roi_offsets()
        safe_name = roi.name.replace("/", "_").replace("\\", "_")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Active ROI as .roi", f"{safe_name}.roi",
            "ImageJ ROI (*.roi);;All Files (*)",
        )
        if not path:
            return
        try:
            from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi
            roi_dict = mask_to_imagej_roi(roi.mask, roi.name)
            if roi_dict:
                write_imagej_roi(roi_dict, path)
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
        dir_path = QFileDialog.getExistingDirectory(self, "Export ImageJ ROIs to Directory")
        if not dir_path:
            return
        try:
            from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi
            n = len(self.layer_stack.roi_layers)
            if n > 1:
                progress = QProgressDialog("Exporting ImageJ ROIs...", "Cancel", 0, n, self)
                progress.setWindowModality(Qt.WindowModal)
            else:
                progress = None

            def _export_single_imagej(mask, name, dp):
                roi_dict = mask_to_imagej_roi(mask, name)
                if roi_dict:
                    safe_name = name.replace("/", "_").replace("\\", "_")
                    write_imagej_roi(roi_dict, os.path.join(dp, f"{safe_name}.roi"))
                    return True
                return False

            if n > 3:
                from montaris.core.workers import get_pool
                futures = []
                for roi in self.layer_stack.roi_layers:
                    fut = get_pool().submit(_export_single_imagej, roi.mask, roi.name, dir_path)
                    futures.append(fut)
                count = 0
                for i, fut in enumerate(futures):
                    if progress and progress.wasCanceled():
                        break
                    if fut.result():
                        count += 1
                    if progress:
                        progress.setValue(i + 1)
            else:
                count = 0
                for i, roi in enumerate(self.layer_stack.roi_layers):
                    if progress and progress.wasCanceled():
                        break
                    if _export_single_imagej(roi.mask, roi.name, dir_path):
                        count += 1
                    if progress:
                        progress.setValue(i + 1)
            if progress:
                progress.close()
            self.toast.show(f"Exported {count} ImageJ ROI(s)", "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")

    def batch_export_rois(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to export.")
            return
        self._flatten_roi_offsets()
        path, filt = QFileDialog.getSaveFileName(
            self, "Batch Export ROIs", "rois",
            "NumPy Archive (*.npz);;ImageJ ROI Directory;;PNG (*.png)",
        )
        if not path:
            return
        try:
            n = len(self.layer_stack.roi_layers)
            if filt.startswith("NumPy"):
                save_roi_set(path, self.layer_stack.roi_layers)
            elif filt.startswith("ImageJ"):
                from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi
                os.makedirs(path, exist_ok=True)
                progress = QProgressDialog("Exporting ImageJ ROIs...", "Cancel", 0, n, self)
                progress.setWindowModality(Qt.WindowModal)
                cancelled = False
                for i, roi in enumerate(self.layer_stack.roi_layers):
                    if progress.wasCanceled():
                        cancelled = True
                        break
                    roi_dict = mask_to_imagej_roi(roi.mask, roi.name)
                    if roi_dict:
                        safe_name = roi.name.replace("/", "_").replace("\\", "_")
                        write_imagej_roi(roi_dict, os.path.join(path, f"{safe_name}.roi"))
                    progress.setValue(i + 1)
                progress.close()
                if cancelled:
                    self.toast.show("Export cancelled", "warning")
                    return
            elif filt.startswith("PNG"):
                self.export_roi_png_to(path)
            self.toast.show(f"Batch exported to {os.path.basename(path)}", "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to batch export:\n{e}")

    def _auto_load_instructions(self, folder):
        """Auto-load a .txt file containing 'instructions' in its name from folder."""
        try:
            for fname in os.listdir(folder):
                if fname.lower().endswith('.txt') and 'instruction' in fname.lower():
                    path = os.path.join(folder, fname)
                    with open(path, 'r') as f:
                        self._last_instructions_text = f.read()
                    self.toast.show(f"Auto-loaded instructions: {fname}", "success")
                    return
        except OSError:
            pass

    def load_instructions_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Instructions", "",
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

    # -- Import PNG masks --

    def import_png_masks(self):
        if self.layer_stack.image_layer is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import PNG Masks", "",
            "PNG Masks (*.png);;All Files (*)",
        )
        if not paths:
            return
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

            def _decode_png_mask(p, tw, th):
                img = Image.open(p).convert('L')
                arr = np.array(img)
                if arr.shape != (th, tw):
                    arr = np.array(img.resize((tw, th), Image.NEAREST))
                return (arr > 0).astype(np.uint8) * 255

            if n > 3:
                from montaris.core.workers import get_pool
                futures = [(p, get_pool().submit(_decode_png_mask, p, w, h)) for p in paths]
                for i, (p, fut) in enumerate(futures):
                    if progress and progress.wasCanceled():
                        break
                    mask = fut.result()
                    name = os.path.splitext(os.path.basename(p))[0]
                    roi = ROILayer(name, w, h)
                    roi.mask = mask
                    self.layer_stack.insert_roi(insert_at + i, roi)
                    if progress:
                        progress.setValue(i + 1)
            else:
                for i, path in enumerate(paths):
                    if progress and progress.wasCanceled():
                        break
                    mask = _decode_png_mask(path, w, h)
                    name = os.path.splitext(os.path.basename(path))[0]
                    roi = ROILayer(name, w, h)
                    roi.mask = mask
                    self.layer_stack.insert_roi(insert_at + i, roi)
                    if progress:
                        progress.setValue(i + 1)

            if progress:
                progress.close()
            self.canvas.refresh_overlays()
            self.layer_panel.refresh()
            self.toast.show(f"Imported {len(paths)} PNG mask(s)", "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import PNG masks:\n{e}")

    # -- Import ROI ZIP --

    def import_roi_zip(self):
        if self.layer_stack.image_layer is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import ROI ZIP", "",
            "ZIP Archive (*.zip);;All Files (*)",
        )
        if not path:
            return
        action = self._ask_replace_or_keep()
        if action is None:
            return
        if action == "replace":
            self.layer_stack.roi_layers.clear()
            self.layer_stack._color_index = 0
            self.canvas._selection.clear()
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
            need_scale = max_w > img_w or max_h > img_h
            if need_scale:
                scale_x = img_w / max_w
                scale_y = img_h / max_h

            # Insert after currently selected ROI, or append at end
            active = self.canvas._active_layer
            if active and active in self.layer_stack.roi_layers:
                insert_at = self.layer_stack.roi_layers.index(active) + 1
            else:
                insert_at = len(self.layer_stack.roi_layers)

            count = 0
            total = len(roi_entries)
            progress = QProgressDialog(f"Importing {total} ROI(s)...", "Cancel", 0, total, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            QApplication.processEvents()

            def _decode_roi_entry(entry_type, payload, tw, th, sx, sy, do_scale):
                if entry_type == 'roi':
                    roi_dict = payload
                    if do_scale:
                        roi_dict = dict(roi_dict)
                        roi_dict['top'] = int(roi_dict['top'] * sy)
                        roi_dict['bottom'] = int(roi_dict['bottom'] * sy)
                        roi_dict['left'] = int(roi_dict['left'] * sx)
                        roi_dict['right'] = int(roi_dict['right'] * sx)
                        if roi_dict.get('x_coords') is not None:
                            roi_dict['x_coords'] = (roi_dict['x_coords'] * sx).astype(np.int32)
                            roi_dict['y_coords'] = (roi_dict['y_coords'] * sy).astype(np.int32)
                        if roi_dict.get('paths'):
                            roi_dict['paths'] = [
                                [(x * sx, y * sy) for x, y in p]
                                for p in roi_dict['paths']
                            ]
                    return imagej_roi_to_mask(roi_dict, tw, th)
                else:  # png
                    img = Image.open(_io.BytesIO(payload)).convert('L')
                    arr = np.array(img)
                    if arr.shape != (th, tw):
                        arr = np.array(img.resize((tw, th), Image.NEAREST))
                    return (arr > 0).astype(np.uint8) * 255

            sx = scale_x if need_scale else 1.0
            sy = scale_y if need_scale else 1.0

            if total > 3:
                from montaris.core.workers import get_pool
                futures = [
                    (base, get_pool().submit(_decode_roi_entry, et, pl, w, h, sx, sy, need_scale))
                    for et, base, pl in roi_entries
                ]
                for i, (base, fut) in enumerate(futures):
                    if progress.wasCanceled():
                        break
                    mask = fut.result()
                    roi = ROILayer(base, w, h)
                    roi.mask = mask
                    self.layer_stack.insert_roi(insert_at + count, roi)
                    count += 1
                    progress.setValue(i + 1)
                    QApplication.processEvents()
            else:
                for i, (entry_type, base, payload) in enumerate(roi_entries):
                    if progress.wasCanceled():
                        break
                    mask = _decode_roi_entry(entry_type, payload, w, h, sx, sy, need_scale)
                    roi = ROILayer(base, w, h)
                    roi.mask = mask
                    self.layer_stack.insert_roi(insert_at + count, roi)
                    count += 1
                    progress.setValue(i + 1)
                    QApplication.processEvents()

            progress.close()
            self.canvas.refresh_overlays()
            self.layer_panel.refresh()
            msg = f"Imported {count} ROI(s) from ZIP"
            if need_scale:
                msg += f" (scaled from {max_w}\u00d7{max_h} to {img_w}\u00d7{img_h})"
            self.toast.show(msg, "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import ZIP:\n{e}")

    # -- Export all ROIs as ZIP --

    def export_all_rois_zip(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to export.")
            return
        self._flatten_roi_offsets()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export All ROIs as ZIP", "rois.zip",
            "ZIP Archive (*.zip);;All Files (*)",
        )
        if not path:
            return
        try:
            import zipfile
            from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi_bytes
            n = len(self.layer_stack.roi_layers)
            if n > 1:
                progress = QProgressDialog("Exporting ROIs to ZIP...", "Cancel", 0, n, self)
                progress.setWindowModality(Qt.WindowModal)
            else:
                progress = None

            def _compute_roi_bytes(mask, name):
                roi_dict = mask_to_imagej_roi(mask, name)
                if roi_dict:
                    safe = name.replace("/", "_").replace("\\", "_")
                    return (f"{safe}.roi", write_imagej_roi_bytes(roi_dict))
                return None

            # Compute in parallel, write to zipfile sequentially (not thread-safe)
            if n > 3:
                from montaris.core.workers import get_pool
                futures = [
                    get_pool().submit(_compute_roi_bytes, roi.mask, roi.name)
                    for roi in self.layer_stack.roi_layers
                ]
                results = []
                cancelled = False
                for i, fut in enumerate(futures):
                    if progress and progress.wasCanceled():
                        cancelled = True
                        break
                    results.append(fut.result())
                    if progress:
                        progress.setValue(i + 1)
            else:
                results = []
                cancelled = False
                for i, roi in enumerate(self.layer_stack.roi_layers):
                    if progress and progress.wasCanceled():
                        cancelled = True
                        break
                    results.append(_compute_roi_bytes(roi.mask, roi.name))
                    if progress:
                        progress.setValue(i + 1)

            if not cancelled:
                with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for entry in results:
                        if entry is not None:
                            zf.writestr(entry[0], entry[1])

            if progress:
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
            self._tint_btn.setStyleSheet("")

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
        selected = self.canvas._selection.layers
        if not selected:
            layer = self.canvas._active_layer
            if layer and hasattr(layer, 'mask') and layer in self.layer_stack.roi_layers:
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
        self.canvas._active_layer = None
        self.canvas._selection.clear()
        # Batch remove without emitting signals per removal
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.layer_stack.roi_layers):
                self.layer_stack.roi_layers.pop(idx)
        self.layer_stack.changed.emit()
        # Select adjacent ROI if available
        if self.layer_stack.roi_layers:
            new_idx = min(first_idx, len(self.layer_stack.roi_layers) - 1)
            self.canvas.set_active_layer(self.layer_stack.roi_layers[new_idx])
        self.canvas.refresh_overlays()
        self.layer_panel.refresh()
        self.properties_panel.set_layer(self.canvas._active_layer)

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
            self._refresh_affected_layers(cmd)
            self._auto_select_roi_from_command(cmd)

    def redo(self):
        cmd = self.undo_stack.redo()
        if cmd:
            self._refresh_affected_layers(cmd)
            self._auto_select_roi_from_command(cmd)

    def _refresh_affected_layers(self, cmd):
        """Refresh only the layers affected by an undo/redo command."""
        layers = set()
        if hasattr(cmd, 'commands'):
            # CompoundUndoCommand
            for sub in cmd.commands:
                if hasattr(sub, 'roi_layer') and sub.roi_layer is not None:
                    layers.add(sub.roi_layer)
        elif hasattr(cmd, '_entries'):
            # SnapshotUndoCommand
            for layer, _, _ in cmd._entries:
                layers.add(layer)
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
        for d in [self._layer_dock, self._props_dock,
                  self._display_dock, self._adj_dock]:
            d.setVisible(True)

    def _apply_dock_widths(self):
        """Set compact right sidebar width after layout is ready."""
        right_docks = [self._layer_dock, self._props_dock,
                       self._display_dock, self._adj_dock]
        self.resizeDocks(right_docks, [220] * len(right_docks), Qt.Horizontal)

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        from montaris.core.workers import shutdown_pool
        shutdown_pool()
        super().closeEvent(event)


def main():
    # Windows: set AppUserModelID so taskbar uses our icon, not Python's
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('montaris.montaris-x')
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("Montaris-X")
    app.setOrganizationName("Montaris")

    _logo = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(_logo):
        app.setWindowIcon(QIcon(_logo))

    apply_dark_theme(app)
    window = MontarisApp()
    window.show()
    sys.exit(app.exec())
