import sys
import os
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QFileDialog,
    QStatusBar, QMessageBox, QProgressDialog, QDialog, QVBoxLayout, QTextEdit,
    QDialogButtonBox, QToolBar, QLabel, QSlider, QSpinBox, QHBoxLayout, QWidget,
    QComboBox, QInputDialog,
)
from PySide6.QtCore import Qt, QSettings, QRectF
from PySide6.QtGui import QAction, QKeySequence, QPalette, QColor, QTransform, QShortcut

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
from montaris.io.image_io import load_image
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
        self.resize(1400, 900)

        self.layer_stack = LayerStack()
        self.undo_stack = UndoStack()
        self.active_tool = None
        self._compositor = DisplayCompositor()
        self._adjustments = ImageAdjustments()
        self._auto_overlap = False
        self._downsample_factor = 1
        self._documents = []
        self._active_doc_index = -1

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

    def _setup_canvas(self):
        self.canvas = ImageCanvas(self.layer_stack, self)
        self.setCentralWidget(self.canvas)

    def _setup_panels(self):
        # Layer panel
        self.layer_panel = LayerPanel(self.layer_stack, self)
        layer_dock = QDockWidget("Layers", self)
        layer_dock.setWidget(self.layer_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, layer_dock)

        # Tool panel
        self.tool_panel = ToolPanel(self, self)
        tool_dock = QDockWidget("Tools", self)
        tool_dock.setWidget(self.tool_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, tool_dock)

        # Properties panel
        self.properties_panel = PropertiesPanel(self, self)
        props_dock = QDockWidget("Properties", self)
        props_dock.setWidget(self.properties_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, props_dock)

        # Display panel (Phase 2)
        self.display_panel = DisplayPanel(self)
        display_dock = QDockWidget("Display", self)
        display_dock.setWidget(self.display_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, display_dock)
        self.display_panel.mode_changed.connect(self._on_display_mode_changed)
        self.display_panel.channels_changed.connect(self._on_channels_changed)
        self._display_dock = display_dock

        # Adjustments panel (Phase 2)
        self.adjustments_panel = AdjustmentsPanel(self)
        adj_dock = QDockWidget("Adjustments", self)
        adj_dock.setWidget(self.adjustments_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, adj_dock)
        self.adjustments_panel.adjustments_changed.connect(self._on_adjustments_changed)
        self._adj_dock = adj_dock

        # Minimap (Phase 6)
        self.minimap = MiniMap(self)
        minimap_dock = QDockWidget("Mini Map", self)
        minimap_dock.setWidget(self.minimap)
        self.addDockWidget(Qt.LeftDockWidgetArea, minimap_dock)
        self.minimap.pan_requested.connect(self._on_minimap_pan)
        self._minimap_dock = minimap_dock

        # Performance monitor (Phase 6)
        self.perf_monitor = PerfMonitor(self)
        perf_dock = QDockWidget("Performance", self)
        perf_dock.setWidget(self.perf_monitor)
        self.addDockWidget(Qt.LeftDockWidgetArea, perf_dock)
        perf_dock.setVisible(False)
        self._perf_dock = perf_dock

        # Debug console (Phase 6)
        self.debug_console = DebugConsole(self, self)
        debug_dock = QDockWidget("Debug Console", self)
        debug_dock.setWidget(self.debug_console)
        self.addDockWidget(Qt.BottomDockWidgetArea, debug_dock)
        debug_dock.setVisible(False)
        self._debug_dock = debug_dock

        # Wire selection model
        self.layer_panel.set_selection_model(self.canvas._selection)
        self.canvas._selection.changed.connect(self._on_selection_count_changed)

        # Connections
        self.tool_panel.tool_changed.connect(self._on_tool_changed)
        self.layer_panel.selection_changed.connect(self._on_layer_selected)
        self.layer_panel.visibility_changed.connect(self.canvas.refresh_overlays)
        self.layer_panel.roi_added.connect(self._on_roi_added)
        self.layer_panel.roi_removed.connect(self._on_roi_removed)

    def _setup_menus(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")

        open_act = QAction("&Open Image...", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self.open_image)
        file_menu.addAction(open_act)

        file_menu.addSeparator()

        load_roi_act = QAction("Load &ROI Set...", self)
        load_roi_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        load_roi_act.triggered.connect(self.load_rois)
        file_menu.addAction(load_roi_act)

        save_roi_act = QAction("&Save ROI Set...", self)
        save_roi_act.setShortcut(QKeySequence.Save)
        save_roi_act.triggered.connect(self.save_rois)
        file_menu.addAction(save_roi_act)

        file_menu.addSeparator()

        # Import/Export ImageJ ROI (Phase 7)
        import_ij_act = QAction("Import ImageJ ROI...", self)
        import_ij_act.triggered.connect(self.import_imagej_roi)
        file_menu.addAction(import_ij_act)

        export_ij_act = QAction("Export ImageJ ROIs...", self)
        export_ij_act.triggered.connect(self.export_imagej_rois)
        file_menu.addAction(export_ij_act)

        file_menu.addSeparator()

        export_act = QAction("&Export ROI as PNG...", self)
        export_act.setShortcut(QKeySequence("Ctrl+E"))
        export_act.triggered.connect(self.export_roi_png)
        file_menu.addAction(export_act)

        # Batch export (Phase 7)
        batch_export_act = QAction("Batch Export All ROIs...", self)
        batch_export_act.triggered.connect(self.batch_export_rois)
        file_menu.addAction(batch_export_act)

        # Import PNG masks
        import_png_act = QAction("Import PNG Mask(s)...", self)
        import_png_act.triggered.connect(self.import_png_masks)
        file_menu.addAction(import_png_act)

        # Import ROI ZIP
        import_zip_act = QAction("Import ROI ZIP...", self)
        import_zip_act.triggered.connect(self.import_roi_zip)
        file_menu.addAction(import_zip_act)

        # Export all as ZIP
        export_zip_act = QAction("Export All ROIs as ZIP...", self)
        export_zip_act.triggered.connect(self.export_all_rois_zip)
        file_menu.addAction(export_zip_act)

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

        clear_act = QAction("&Clear Active ROI", self)
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

    def _setup_toolbar(self):
        """Add main toolbar with brush size and opacity controls (G.19, G.20)."""
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(True)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # Brush size in toolbar — synced with tool_panel
        toolbar.addWidget(QLabel(" Size: "))
        tb_size_slider = QSlider(Qt.Horizontal)
        tb_size_slider.setRange(1, 500)
        tb_size_slider.setValue(10)
        tb_size_slider.setFixedWidth(120)
        toolbar.addWidget(tb_size_slider)

        tb_size_spin = QSpinBox()
        tb_size_spin.setRange(1, 500)
        tb_size_spin.setValue(10)
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

        pp_slider = self.properties_panel.opacity_slider
        tb_opacity_slider.valueChanged.connect(pp_slider.setValue)
        pp_slider.valueChanged.connect(tb_opacity_slider.setValue)

        toolbar.addSeparator()

        # Document switcher (A.11)
        toolbar.addWidget(QLabel(" Montage: "))
        self._doc_combo = QComboBox()
        self._doc_combo.setMinimumWidth(150)
        self._doc_combo.currentIndexChanged.connect(self._switch_to_document)
        toolbar.addWidget(self._doc_combo)

    def _update_cursor_info(self, x, y, value):
        roi_info = ""
        if self.canvas._active_layer and hasattr(self.canvas._active_layer, 'mask'):
            mask = self.canvas._active_layer.mask
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                roi_info = f"  ROI: {'yes' if mask[y, x] > 0 else 'no'}"
        self.statusbar.showMessage(f"X: {x}  Y: {y}  Value: {value}{roi_info}")

    def _on_tool_changed(self, tool):
        self.active_tool = tool
        self.canvas.set_tool(tool)
        # Update tool status (G.14)
        tool_name = getattr(tool, 'name', 'None') if tool else 'None'
        roi_info = ""
        if self.canvas._active_layer and hasattr(self.canvas._active_layer, 'name'):
            roi_info = f"  |  {self.canvas._active_layer.name}"
        self._tool_status_label.setText(f"Tool: {tool_name}{roi_info}")

    def _on_layer_selected(self, layer):
        self.canvas.set_active_layer(layer)
        self.properties_panel.set_layer(layer)
        # Update tool status (G.14)
        tool_name = getattr(self.active_tool, 'name', 'None') if self.active_tool else 'None'
        roi_info = f"  |  {layer.name}" if layer and hasattr(layer, 'name') else ""
        self._tool_status_label.setText(f"Tool: {tool_name}{roi_info}")

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

    # -- Display mode callbacks (Phase 2) --

    def _on_display_mode_changed(self, mode):
        self._compositor.mode = mode
        self.canvas.refresh_image()

    def _on_channels_changed(self, active_indices):
        pass  # Channel toggling handled via display panel

    def _on_adjustments_changed(self, adjustments):
        self._adjustments = adjustments
        self.canvas.refresh_image()

    def _on_minimap_pan(self, scene_x, scene_y):
        self.canvas.centerOn(scene_x, scene_y)

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
        self.canvas._selection.select_all(self.layer_stack.roi_layers)

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
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if path:
            try:
                data = load_image(path)
                # Apply flip/rotate on load (E.21, E.22)
                if self._flip_on_load_act.isChecked():
                    data = np.flip(data, axis=1).copy()
                if self._rotate_on_load_act.isChecked():
                    data = np.rot90(data, k=-1).copy()
                original_shape = data.shape

                # Downsample dialog (A.13)
                ds_factor = 1
                if data.shape[0] * data.shape[1] > 4_000_000:
                    items = ["1x (Original)", "2x", "4x", "8x"]
                    item, ok = QInputDialog.getItem(
                        self, "Downsample", "Image is large. Choose downsample factor:",
                        items, 0, False,
                    )
                    if ok and item != items[0]:
                        ds_factor = int(item[0])
                        data = data[::ds_factor, ::ds_factor]

                # Save current document state
                self._save_current_document()

                self._downsample_factor = ds_factor
                self.canvas._selection.clear()
                self.layer_stack.set_image(ImageLayer(os.path.basename(path), data))
                self.canvas.refresh_image()
                self.canvas.fit_to_window()
                self.layer_panel.refresh()
                self.undo_stack.clear()
                self.minimap.set_image(data)
                self.adjustments_panel.set_image_data(data)

                # Create montage document (A.10)
                doc = MontageDocument(
                    name=os.path.basename(path),
                    image_layer=self.layer_stack.image_layer,
                    roi_layers=self.layer_stack.roi_layers,
                    downsample_factor=ds_factor,
                    original_shape=original_shape,
                )
                self._documents.append(doc)
                self._active_doc_index = len(self._documents) - 1
                self._doc_combo.blockSignals(True)
                self._doc_combo.addItem(doc.name)
                self._doc_combo.setCurrentIndex(self._active_doc_index)
                self._doc_combo.blockSignals(False)

                self.toast.show(
                    f"Loaded: {os.path.basename(path)}  {data.shape}", "success"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

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
                self.canvas.refresh_overlays()
                self.layer_panel.refresh()
                self.statusbar.showMessage(f"Loaded {len(rois)} ROIs from {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load ROIs:\n{e}")

    def save_rois(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI Set", "rois.npz",
            "NumPy Archive (*.npz);;All Files (*)",
        )
        if path:
            try:
                save_roi_set(path, self.layer_stack.roi_layers)
                self.statusbar.showMessage(
                    f"Saved {len(self.layer_stack.roi_layers)} ROIs to {path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save ROIs:\n{e}")

    def export_roi_png(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ROI as PNG", "roi_export.png",
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
            img_layer = self.layer_stack.image_layer
            h, w = img_layer.shape[:2]
            composite = np.zeros((h, w, 4), dtype=np.uint8)

            if img_layer.data.ndim == 2:
                gray = img_layer.data
                if gray.dtype != np.uint8:
                    mn, mx = float(gray.min()), float(gray.max())
                    if mx > mn:
                        gray = ((gray.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
                    else:
                        gray = np.zeros_like(gray, dtype=np.uint8)
                composite[:, :, 0] = gray
                composite[:, :, 1] = gray
                composite[:, :, 2] = gray
                composite[:, :, 3] = 255
            else:
                c = min(3, img_layer.data.shape[2])
                bg = img_layer.data[:, :, :c]
                if bg.dtype != np.uint8:
                    mn, mx = float(bg.min()), float(bg.max())
                    if mx > mn:
                        bg = ((bg.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
                composite[:, :, :c] = bg
                composite[:, :, 3] = 255

            for roi in self.layer_stack.roi_layers:
                if not roi.visible:
                    continue
                mask = roi.mask > 0
                alpha = roi.opacity / 255.0
                r, g, b = roi.color
                for c_idx, c_val in enumerate([r, g, b]):
                    composite[:, :, c_idx][mask] = (
                        composite[:, :, c_idx][mask] * (1 - alpha) + c_val * alpha
                    ).astype(np.uint8)

            img = Image.fromarray(composite)
            img.save(path)
            self.statusbar.showMessage(f"Exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")

    # -- ImageJ ROI (Phase 7) --

    def import_imagej_roi(self):
        if self.layer_stack.image_layer is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import ImageJ ROI", "",
            "ImageJ ROI (*.roi);;All Files (*)",
        )
        if not paths:
            return
        try:
            from montaris.io.imagej_roi import read_imagej_roi, imagej_roi_to_mask
            h, w = self.layer_stack.image_layer.shape[:2]
            for path in paths:
                roi_dict = read_imagej_roi(path)
                mask = imagej_roi_to_mask(roi_dict, w, h)
                name = os.path.splitext(os.path.basename(path))[0]
                roi = ROILayer(name, w, h)
                roi.mask = mask
                self.layer_stack.add_roi(roi)
            self.canvas.refresh_overlays()
            self.layer_panel.refresh()
            self.statusbar.showMessage(f"Imported {len(paths)} ImageJ ROI(s)")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import:\n{e}")

    def export_imagej_rois(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to export.")
            return
        dir_path = QFileDialog.getExistingDirectory(self, "Export ImageJ ROIs to Directory")
        if not dir_path:
            return
        try:
            from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi
            count = 0
            for roi in self.layer_stack.roi_layers:
                roi_dict = mask_to_imagej_roi(roi.mask, roi.name)
                if roi_dict:
                    safe_name = roi.name.replace("/", "_").replace("\\", "_")
                    write_imagej_roi(roi_dict, os.path.join(dir_path, f"{safe_name}.roi"))
                    count += 1
            self.statusbar.showMessage(f"Exported {count} ImageJ ROI(s) to {dir_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")

    def batch_export_rois(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to export.")
            return
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

    def load_instructions_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Instructions", "",
            "JSON (*.json);;Text (*.txt);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, 'r') as f:
                self._last_instructions_text = f.read()
            from montaris.io.instructions import load_instructions, apply_instructions
            instructions = load_instructions(path)
            log = apply_instructions(self, instructions)
            self.layer_panel.refresh()
            for msg in log:
                self.debug_console.log(msg)
            self.toast.show(f"Applied instructions from {os.path.basename(path)}", "success")
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
            for path in paths:
                img = Image.open(path).convert('L')
                arr = np.array(img)
                if arr.shape != (h, w):
                    img_resized = img.resize((w, h), Image.NEAREST)
                    arr = np.array(img_resized)
                mask = (arr > 0).astype(np.uint8) * 255
                name = os.path.splitext(os.path.basename(path))[0]
                roi = ROILayer(name, w, h)
                roi.mask = mask
                self.layer_stack.add_roi(roi)
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
            h, w = self.layer_stack.image_layer.shape[:2]
            count = 0
            with zipfile.ZipFile(path, 'r') as zf:
                for name in zf.namelist():
                    lower = name.lower()
                    data = zf.read(name)
                    base = os.path.splitext(os.path.basename(name))[0]
                    if lower.endswith('.roi'):
                        roi_dict = read_imagej_roi(data)
                        mask = imagej_roi_to_mask(roi_dict, w, h)
                        roi = ROILayer(base, w, h)
                        roi.mask = mask
                        self.layer_stack.add_roi(roi)
                        count += 1
                    elif lower.endswith('.png'):
                        img = Image.open(_io.BytesIO(data)).convert('L')
                        arr = np.array(img)
                        if arr.shape != (h, w):
                            img = img.resize((w, h), Image.NEAREST)
                            arr = np.array(img)
                        mask = (arr > 0).astype(np.uint8) * 255
                        roi = ROILayer(base, w, h)
                        roi.mask = mask
                        self.layer_stack.add_roi(roi)
                        count += 1
            self.canvas.refresh_overlays()
            self.layer_panel.refresh()
            self.toast.show(f"Imported {count} ROI(s) from ZIP", "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import ZIP:\n{e}")

    # -- Export all ROIs as ZIP --

    def export_all_rois_zip(self):
        if not self.layer_stack.roi_layers:
            QMessageBox.information(self, "Info", "No ROIs to export.")
            return
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
            progress = QProgressDialog("Exporting ROIs...", "Cancel", 0, n, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for i, roi in enumerate(self.layer_stack.roi_layers):
                    if progress.wasCanceled():
                        break
                    roi_dict = mask_to_imagej_roi(roi.mask, roi.name)
                    if roi_dict:
                        safe_name = roi.name.replace("/", "_").replace("\\", "_")
                        data = write_imagej_roi_bytes(roi_dict)
                        zf.writestr(f"{safe_name}.roi", data)
                    progress.setValue(i + 1)
            progress.close()
            if progress.wasCanceled():
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
        doc.roi_layers = list(self.layer_stack.roi_layers)
        doc.color_index = self.layer_stack._color_index
        doc.adjustments = {
            'brightness': self._adjustments.brightness,
            'contrast': self._adjustments.contrast,
            'exposure': self._adjustments.exposure,
            'gamma': self._adjustments.gamma,
        }
        doc.downsample_factor = self._downsample_factor

    def _switch_to_document(self, index):
        """Switch to a different MontageDocument."""
        if index < 0 or index >= len(self._documents):
            return
        if index == self._active_doc_index:
            return
        # Save current
        self._save_current_document()
        # Load new
        doc = self._documents[index]
        self._active_doc_index = index
        self._downsample_factor = doc.downsample_factor
        self.canvas._selection.clear()
        self.layer_stack.image_layer = doc.image_layer
        self.layer_stack.roi_layers = doc.roi_layers
        self.layer_stack._color_index = doc.color_index
        self.layer_stack.changed.emit()
        self.canvas.refresh_image()
        self.canvas.refresh_overlays()
        self.canvas.fit_to_window()
        self.layer_panel.refresh()
        self.undo_stack.clear()
        if doc.image_layer:
            self.minimap.set_image(doc.image_layer.data)
            self.adjustments_panel.set_image_data(doc.image_layer.data)
        # Restore adjustments (A.12)
        from montaris.core.adjustments import ImageAdjustments
        self._adjustments = ImageAdjustments(**doc.adjustments)
        if hasattr(self.adjustments_panel, '_adjustments'):
            self.adjustments_panel._adjustments = self._adjustments
            self.adjustments_panel._sync_sliders()
        self.toast.show(f"Switched to: {doc.name}", "info")

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
        dlg.resize(600, 400)
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
        layer = self.canvas._active_layer
        if layer and hasattr(layer, 'mask'):
            from montaris.core.undo import UndoCommand
            snapshot = layer.mask.copy()
            layer.mask[:] = 0
            if snapshot.any():
                ys, xs = np.where(snapshot > 0)
                y1, y2 = ys.min(), ys.max() + 1
                x1, x2 = xs.min(), xs.max() + 1
                cmd = UndoCommand(
                    layer, (y1, y2, x1, x2),
                    snapshot[y1:y2, x1:x2],
                    layer.mask[y1:y2, x1:x2],
                )
                self.undo_stack.push(cmd)
            self.canvas.refresh_overlays()
            self.properties_panel.set_layer(layer)

    def undo(self):
        cmd = self.undo_stack.undo()
        if cmd:
            self.canvas.refresh_overlays()
            self.canvas._update_selection_highlights()
            self._auto_select_roi_from_command(cmd)

    def redo(self):
        cmd = self.undo_stack.redo()
        if cmd:
            self.canvas.refresh_overlays()
            self.canvas._update_selection_highlights()
            self._auto_select_roi_from_command(cmd)

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
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Montaris-X")
    app.setOrganizationName("Montaris")
    apply_dark_theme(app)
    window = MontarisApp()
    window.show()
    sys.exit(app.exec())
