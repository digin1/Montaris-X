import random
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QColorDialog, QInputDialog, QMenu, QAbstractItemView,
    QMessageBox, QSlider, QCheckBox, QDialog, QGridLayout,
)
from PySide6.QtCore import Signal, Qt, QItemSelectionModel
from PySide6.QtGui import QColor, QIcon, QPixmap, QAction, QPainter

from montaris.layers import ROILayer, ROI_COLORS, generate_unique_roi_name


# ---------------------------------------------------------------------------
# Color palette dialog (B.12)
# ---------------------------------------------------------------------------

class ColorPaletteDialog(QDialog):
    def __init__(self, current_color, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Color")
        self._selected = None
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        for i, c in enumerate(ROI_COLORS):
            btn = QPushButton()
            btn.setFixedSize(28, 28)
            btn.setStyleSheet(f"background: rgb({c[0]},{c[1]},{c[2]}); border: 1px solid #555;")
            btn.clicked.connect(lambda checked, color=c: self._pick(color))
            grid.addWidget(btn, i // 5, i % 5)
        layout.addLayout(grid)

        custom_btn = QPushButton("Custom...")
        custom_btn.clicked.connect(self._custom)
        layout.addWidget(custom_btn)

        self._current = current_color

    def _pick(self, color):
        self._selected = color
        self.accept()

    def _custom(self):
        color = QColorDialog.getColor(QColor(*self._current), self)
        if color.isValid():
            self._selected = (color.red(), color.green(), color.blue())
            self.accept()

    @property
    def selected_color(self):
        return self._selected


# ---------------------------------------------------------------------------
# ROI navigation bar (G.13)
# ---------------------------------------------------------------------------

class ROINavBar(QWidget):
    roi_clicked = Signal(int)  # emits roi index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(20)
        self._segments = []  # list of (fraction, color, roi_index)

    def set_segments(self, segments):
        self._segments = segments
        self.update()

    def paintEvent(self, event):
        if not self._segments:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w = self.width()
        x = 0
        for frac, color, _ in self._segments:
            seg_w = max(1, int(frac * w))
            p.fillRect(int(x), 0, seg_w, self.height(), QColor(*color))
            x += seg_w
        p.end()

    def mousePressEvent(self, event):
        if not self._segments:
            return
        click_x = event.position().x()
        w = self.width()
        x = 0
        for frac, color, idx in self._segments:
            seg_w = max(1, int(frac * w))
            if x <= click_x < x + seg_w:
                self.roi_clicked.emit(idx)
                return
            x += seg_w


# ---------------------------------------------------------------------------
# ROI binning helpers
# ---------------------------------------------------------------------------

def _voronoi_bin(mask, n_bins):
    """Split an ROI mask into n_bins region-aware sub-regions using Voronoi
    tessellation with Lloyd's relaxation for balanced area.

    Seeds are initialized from distance-transform peaks (maximally interior
    points), then iteratively moved to region centroids so that each region
    converges to roughly equal area.

    Returns list of (sub_mask, y_offset, x_offset) tuples, where sub_mask
    is a cropped uint8 array and offsets locate it within the original mask.
    """
    from scipy.ndimage import distance_transform_edt

    binary = mask > 0
    if not binary.any():
        return []

    # Crop to bounding box for efficiency
    ys, xs = np.where(binary)
    by1, by2 = int(ys.min()), int(ys.max()) + 1
    bx1, bx2 = int(xs.min()), int(xs.max()) + 1
    crop = binary[by1:by2, bx1:bx2]

    # Distance transform — peaks are maximally interior points
    dist = distance_transform_edt(crop)

    # Suppress radius: scale with bounding box diagonal / sqrt(n_bins)
    diag = np.sqrt(crop.shape[0] ** 2 + crop.shape[1] ** 2)
    suppress_radius = max(3, int(diag / np.sqrt(max(n_bins, 1)) * 0.5))

    # Find initial seed points via iterative peak suppression
    seeds = []
    dist_work = dist.copy()
    for _ in range(n_bins):
        peak_idx = np.unravel_index(np.argmax(dist_work), dist_work.shape)
        if dist_work[peak_idx] <= 0:
            break
        seeds.append(list(peak_idx))
        py, px = peak_idx
        ch, cw = dist_work.shape
        r = suppress_radius
        y_lo, y_hi = max(0, py - r), min(ch, py + r + 1)
        x_lo, x_hi = max(0, px - r), min(cw, px + r + 1)
        yy, xx = np.ogrid[y_lo:y_hi, x_lo:x_hi]
        circ = (yy - py) ** 2 + (xx - px) ** 2 <= r * r
        dist_work[y_lo:y_hi, x_lo:x_hi][circ] = 0

    if len(seeds) < 2:
        sub = mask[by1:by2, bx1:bx2].copy()
        return [(sub, by1, bx1)]

    # Pixel coordinates of all mask points (computed once)
    mask_ys, mask_xs = np.where(crop)
    mask_ys_f = mask_ys.astype(np.float64)
    mask_xs_f = mask_xs.astype(np.float64)
    n_pixels = len(mask_ys)

    # Lloyd's relaxation: assign → recompute centroids → repeat
    seeds_arr = np.array(seeds, dtype=np.float64)  # (K, 2)
    for _iteration in range(15):
        # Assign each pixel to nearest seed
        best_label = np.zeros(n_pixels, dtype=np.int32)
        best_dist = np.full(n_pixels, np.finfo(np.float64).max)
        for i in range(len(seeds_arr)):
            d = (mask_ys_f - seeds_arr[i, 0]) ** 2 + (mask_xs_f - seeds_arr[i, 1]) ** 2
            closer = d < best_dist
            best_dist[closer] = d[closer]
            best_label[closer] = i

        # Move seeds to region centroids
        converged = True
        for i in range(len(seeds_arr)):
            members = best_label == i
            if not members.any():
                continue
            cy = mask_ys_f[members].mean()
            cx = mask_xs_f[members].mean()
            # Ensure centroid stays inside the mask
            if not crop[int(round(cy)), int(round(cx))]:
                # Snap to nearest mask pixel to centroid
                dists = (mask_ys_f - cy) ** 2 + (mask_xs_f - cx) ** 2
                nearest = np.argmin(dists)
                cy, cx = mask_ys_f[nearest], mask_xs_f[nearest]
            if abs(seeds_arr[i, 0] - cy) > 0.5 or abs(seeds_arr[i, 1] - cx) > 0.5:
                converged = False
            seeds_arr[i] = [cy, cx]
        if converged:
            break

    # Final assignment with converged seeds
    best_label = np.zeros(n_pixels, dtype=np.int32)
    best_dist = np.full(n_pixels, np.finfo(np.float64).max)
    for i in range(len(seeds_arr)):
        d = (mask_ys_f - seeds_arr[i, 0]) ** 2 + (mask_xs_f - seeds_arr[i, 1]) ** 2
        closer = d < best_dist
        best_dist[closer] = d[closer]
        best_label[closer] = i + 1

    labels = np.zeros(crop.shape, dtype=np.int32)
    labels[mask_ys, mask_xs] = best_label

    # Extract each labeled region as a cropped sub-mask
    results = []
    for i in range(1, len(seeds_arr) + 1):
        region = labels == i
        if not region.any():
            continue
        rys, rxs = np.where(region)
        ry1, ry2 = int(rys.min()), int(rys.max()) + 1
        rx1, rx2 = int(rxs.min()), int(rxs.max()) + 1
        sub = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)
        sub[region[ry1:ry2, rx1:rx2]] = 255
        results.append((sub, by1 + ry1, bx1 + rx1))

    return results


def _grid_bin(mask, cols, rows):
    """Split an ROI mask into a cols x rows grid.

    Returns list of (sub_mask, y_offset, x_offset) tuples for non-empty cells.
    """
    h, w = mask.shape
    bx = (w + cols - 1) // cols
    by = (h + rows - 1) // rows
    results = []

    for gy in range(rows):
        for gx in range(cols):
            y0 = gy * by
            y1 = min(y0 + by, h)
            x0 = gx * bx
            x1 = min(x0 + bx, w)
            cell = mask[y0:y1, x0:x1]
            if cell.max() == 0:
                continue
            results.append((cell.copy(), y0, x0))

    return results


# ---------------------------------------------------------------------------
# Layer panel
# ---------------------------------------------------------------------------

class LayerPanel(QWidget):
    selection_changed = Signal(object)
    visibility_changed = Signal()
    roi_added = Signal()
    roi_removed = Signal(int)
    all_cleared = Signal()

    def __init__(self, layer_stack, parent=None):
        super().__init__(parent)
        self.layer_stack = layer_stack
        self._updating = False
        self._selection_model = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header with ROI count (G.22)
        self.header = QLabel("Layers")
        self.header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self.header)

        # Toggle all visibility (B.14)
        self.show_all_cb = QCheckBox("Show All")
        self.show_all_cb.setChecked(True)
        self.show_all_cb.toggled.connect(self._toggle_all_visibility)
        layout.addWidget(self.show_all_cb)

        # Nav bar (G.13)
        self.nav_bar = ROINavBar(self)
        self.nav_bar.roi_clicked.connect(self._on_nav_bar_clicked)
        layout.addWidget(self.nav_bar)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        self.list_widget.itemSelectionChanged.connect(self._on_multi_selection_changed)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self.list_widget)

        # Button row 1
        btn_layout = QHBoxLayout()

        self.add_btn = QPushButton("+")
        self.add_btn.setFixedWidth(30)
        self.add_btn.setToolTip("Add ROI layer")
        self.add_btn.clicked.connect(lambda: self.roi_added.emit())
        btn_layout.addWidget(self.add_btn)

        self.remove_btn = QPushButton("-")
        self.remove_btn.setFixedWidth(30)
        self.remove_btn.setToolTip("Remove selected ROI")
        self.remove_btn.clicked.connect(self._remove_selected)
        btn_layout.addWidget(self.remove_btn)

        # Clear All (B.7)
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.setToolTip("Remove all ROIs")
        self.clear_all_btn.clicked.connect(self._clear_all)
        btn_layout.addWidget(self.clear_all_btn)

        self.dup_btn = QPushButton("Dup")
        self.dup_btn.setToolTip("Duplicate selected ROI")
        self.dup_btn.clicked.connect(self._duplicate_selected)
        btn_layout.addWidget(self.dup_btn)

        self.merge_btn = QPushButton("Merge")
        self.merge_btn.setToolTip("Merge selected ROIs")
        self.merge_btn.clicked.connect(self._merge_selected)
        btn_layout.addWidget(self.merge_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Button row 2
        btn_layout2 = QHBoxLayout()

        self.rename_btn = QPushButton("Rename")
        self.rename_btn.clicked.connect(self._rename_selected)
        btn_layout2.addWidget(self.rename_btn)

        self.color_btn = QPushButton("Color")
        self.color_btn.clicked.connect(self._change_color)
        btn_layout2.addWidget(self.color_btn)

        # Random color (B.11)
        self.random_color_btn = QPushButton("Random")
        self.random_color_btn.setToolTip("Random color")
        self.random_color_btn.clicked.connect(self._random_color)
        btn_layout2.addWidget(self.random_color_btn)

        btn_layout2.addStretch()
        layout.addLayout(btn_layout2)

        # Global opacity slider (B.30)
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Global Opacity:"))
        self.global_opacity_slider = QSlider(Qt.Horizontal)
        self.global_opacity_slider.setRange(0, 100)
        self.global_opacity_slider.setValue(100)
        self.global_opacity_slider.valueChanged.connect(self._on_global_opacity_changed)
        opacity_layout.addWidget(self.global_opacity_slider)
        layout.addLayout(opacity_layout)

    def refresh(self):
        self._updating = True
        self.list_widget.clear()

        roi_layers = self.layer_stack.roi_layers

        # Header with count (G.22)
        self.header.setText(f"Layers ({len(roi_layers)} ROIs)")

        if self.layer_stack.image_layer:
            item = QListWidgetItem(self.layer_stack.image_layer.name)
            item.setData(Qt.UserRole, ("image", 0))
            item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable & ~Qt.ItemIsDragEnabled)
            self.list_widget.addItem(item)

        px_counts = []
        for i, roi in enumerate(roi_layers):
            icon = self._color_icon(roi.color)
            bbox = roi.get_bbox()
            if bbox is None:
                px_count = 0
            else:
                y1, y2, x1, x2 = bbox
                px_count = int(np.count_nonzero(roi.mask[y1:y2, x1:x2]))
            px_counts.append(px_count)
            display_name = f"{i + 1}. {roi.name}"
            item = QListWidgetItem(display_name)
            item.setIcon(icon)
            item.setData(Qt.UserRole, ("roi", i))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            item.setCheckState(Qt.Checked if roi.visible else Qt.Unchecked)
            self.list_widget.addItem(item)

        # Update nav bar (G.13) — reuse precomputed pixel counts
        self._update_nav_bar(px_counts)

        self._updating = False

    def _update_nav_bar(self, px_counts=None):
        roi_layers = self.layer_stack.roi_layers
        if px_counts is None:
            px_counts = []
            for roi in roi_layers:
                bbox = roi.get_bbox()
                if bbox is None:
                    px_counts.append(0)
                else:
                    y1, y2, x1, x2 = bbox
                    px_counts.append(int(np.count_nonzero(roi.mask[y1:y2, x1:x2])))
        total = sum(max(1, c) for c in px_counts) if px_counts else 1
        segments = []
        for i, roi in enumerate(roi_layers):
            frac = max(1, px_counts[i]) / total
            segments.append((frac, roi.color, i))
        self.nav_bar.set_segments(segments)

    def _on_nav_bar_clicked(self, roi_index):
        # Select the clicked ROI (offset by 1 for image row)
        row = roi_index + (1 if self.layer_stack.image_layer else 0)
        self.list_widget.setCurrentRow(row)

    def set_selection_model(self, model):
        """Connect to a SelectionModel for bidirectional sync."""
        self._selection_model = model
        model.changed.connect(self._on_external_selection_changed)

    def _on_multi_selection_changed(self):
        """Push list-widget multi-selection into SelectionModel."""
        if self._updating or self._selection_model is None:
            return
        layers = []
        for item in self.list_widget.selectedItems():
            data = item.data(Qt.UserRole)
            if data and data[0] == "roi":
                layer = self.layer_stack.get_roi(data[1])
                if layer:
                    layers.append(layer)
        self._updating = True
        self._selection_model.set(layers)
        self._updating = False

    def _on_external_selection_changed(self, layers):
        """Update list widget selection when SelectionModel changes externally."""
        if self._updating:
            return
        self._updating = True
        self.list_widget.clearSelection()
        primary_item = None
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            data = item.data(Qt.UserRole)
            if data and data[0] == "roi":
                roi = self.layer_stack.get_roi(data[1])
                if roi in layers:
                    item.setSelected(True)
                    if roi is (layers[0] if layers else None):
                        primary_item = item
        # Set current item without clearing multi-selection
        if primary_item is not None:
            self.list_widget.setCurrentItem(
                primary_item, QItemSelectionModel.Current
            )
        self._updating = False
        # Emit selection_changed for primary so app syncs active_layer
        if layers:
            self.selection_changed.emit(layers[0])

    def _color_icon(self, color):
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(*color))
        return QIcon(pixmap)

    def _on_selection_changed(self, row):
        if row < 0 or self._updating:
            return
        item = self.list_widget.item(row)
        if item is None:
            return
        data = item.data(Qt.UserRole)
        if data and data[0] == "roi":
            layer = self.layer_stack.get_roi(data[1])
            self.selection_changed.emit(layer)

    def _on_item_changed(self, item):
        if self._updating:
            return
        data = item.data(Qt.UserRole)
        if data and data[0] == "roi":
            roi = self.layer_stack.get_roi(data[1])
            if roi:
                # Check visibility toggle
                new_visible = item.checkState() == Qt.Checked
                if roi.visible != new_visible:
                    roi.visible = new_visible
                    self.visibility_changed.emit()
                    return
                # Inline rename (B.9): detect text change
                raw_text = item.text()
                # Strip index prefix for actual name
                # Format: "1. Name"
                name_part = raw_text
                if ". " in name_part:
                    name_part = name_part.split(". ", 1)[1]
                if name_part and name_part != roi.name:
                    # Name uniqueness (B.10)
                    validated = generate_unique_roi_name(
                        name_part, self.layer_stack.roi_layers
                    )
                    roi.name = validated
                    self.refresh()

    def _get_selected_roi_indices(self):
        """Return sorted list of ROI indices from selected items."""
        indices = []
        for item in self.list_widget.selectedItems():
            data = item.data(Qt.UserRole)
            if data and data[0] == "roi":
                indices.append(data[1])
        return sorted(indices)

    def _on_rows_moved(self):
        """Handle drag-drop reorder by syncing list widget order back to layer_stack."""
        if self._updating:
            return
        # Read the new order from the list widget
        new_order = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            data = item.data(Qt.UserRole)
            if data and data[0] == "roi":
                new_order.append(data[1])

        if not new_order:
            return

        # Reorder the roi_layers list to match the new widget order
        old_layers = list(self.layer_stack.roi_layers)
        new_layers = []
        for idx in new_order:
            if 0 <= idx < len(old_layers):
                new_layers.append(old_layers[idx])

        # Only update if we got all layers
        if len(new_layers) == len(old_layers):
            self.layer_stack.roi_layers[:] = new_layers
            self.layer_stack.changed.emit()

        # Refresh to fix UserRole data indices
        self.refresh()

    def _show_context_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item:
            return
        data = item.data(Qt.UserRole)
        if not data or data[0] != "roi":
            return

        menu = QMenu(self)
        selected_indices = self._get_selected_roi_indices()

        dup_action = QAction("Duplicate", self)
        dup_action.triggered.connect(self._duplicate_selected)
        menu.addAction(dup_action)

        if len(selected_indices) >= 2:
            merge_action = QAction("Merge Selected", self)
            merge_action.triggered.connect(self._merge_selected)
            menu.addAction(merge_action)

        menu.addSeparator()

        insert_above_menu = menu.addMenu("Insert Above")
        ins_above_blank = QAction("Empty ROI", self)
        ins_above_blank.triggered.connect(lambda: self._insert_roi_at(data[1]))
        insert_above_menu.addAction(ins_above_blank)
        ins_above_roi = QAction("From .roi File(s)...", self)
        ins_above_roi.triggered.connect(lambda: self._insert_roi_from_file(data[1], "roi"))
        insert_above_menu.addAction(ins_above_roi)
        ins_above_png = QAction("From PNG Mask(s)...", self)
        ins_above_png.triggered.connect(lambda: self._insert_roi_from_file(data[1], "png"))
        insert_above_menu.addAction(ins_above_png)

        insert_below_menu = menu.addMenu("Insert Below")
        ins_below_blank = QAction("Empty ROI", self)
        ins_below_blank.triggered.connect(lambda: self._insert_roi_at(data[1] + 1))
        insert_below_menu.addAction(ins_below_blank)
        ins_below_roi = QAction("From .roi File(s)...", self)
        ins_below_roi.triggered.connect(lambda: self._insert_roi_from_file(data[1] + 1, "roi"))
        insert_below_menu.addAction(ins_below_roi)
        ins_below_png = QAction("From PNG Mask(s)...", self)
        ins_below_png.triggered.connect(lambda: self._insert_roi_from_file(data[1] + 1, "png"))
        insert_below_menu.addAction(ins_below_png)

        # Move to position (B.22)
        move_to_action = QAction("Move To Position...", self)
        move_to_action.triggered.connect(lambda: self._move_roi_to(data[1]))
        menu.addAction(move_to_action)

        menu.addSeparator()

        if len(selected_indices) >= 2:
            export_ij_action = QAction(f"Export {len(selected_indices)} ROIs as .roi", self)
            export_ij_action.triggered.connect(lambda: self._export_selected_rois("imagej"))
            menu.addAction(export_ij_action)

            export_png_action = QAction(f"Export {len(selected_indices)} ROIs as PNG", self)
            export_png_action.triggered.connect(lambda: self._export_selected_rois("png"))
            menu.addAction(export_png_action)
        else:
            export_ij_action = QAction("Export as ImageJ .roi", self)
            export_ij_action.triggered.connect(lambda: self._export_single_roi(data[1], "imagej"))
            menu.addAction(export_ij_action)

            export_png_action = QAction("Export as PNG", self)
            export_png_action.triggered.connect(lambda: self._export_single_roi(data[1], "png"))
            menu.addAction(export_png_action)

        menu.addSeparator()

        bin_action = QAction("Bin ROI...", self)
        bin_action.triggered.connect(lambda: self._bin_roi(data[1]))
        menu.addAction(bin_action)

        menu.addSeparator()

        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(self._remove_selected)
        menu.addAction(delete_action)

        menu.exec(self.list_widget.mapToGlobal(pos))

    def _remove_selected(self):
        indices = self._get_selected_roi_indices()
        if not indices:
            # Fallback to current row
            row = self.list_widget.currentRow()
            item = self.list_widget.item(row)
            if item:
                data = item.data(Qt.UserRole)
                if data and data[0] == "roi":
                    indices = [data[1]]
        if not indices:
            return
        # Delete confirmation (B.8)
        count = len(indices)
        msg = f"Delete {count} ROI(s)?" if count > 1 else "Delete this ROI?"
        reply = QMessageBox.question(self, "Confirm Delete", msg,
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        # Batch remove without per-item signal/refresh
        for idx in sorted(indices, reverse=True):
            removed = self.layer_stack.get_roi(idx)
            if removed:
                if 0 <= idx < len(self.layer_stack.roi_layers):
                    self.layer_stack.roi_layers.pop(idx)
        self.layer_stack.changed.emit()
        app = self.window()
        if hasattr(app, 'canvas'):
            app.canvas._selection.clear()
            app.canvas.set_active_layer(None)
            app.canvas.refresh_overlays()
        self.refresh()
        if hasattr(app, 'properties_panel'):
            app.properties_panel.set_layer(None)

    def _clear_all(self):
        """Remove all ROIs (B.7)."""
        if not self.layer_stack.roi_layers:
            return
        reply = QMessageBox.question(
            self, "Clear All", f"Remove all {len(self.layer_stack.roi_layers)} ROIs?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.layer_stack.roi_layers.clear()
        self.layer_stack._color_index = 0
        self.all_cleared.emit()

    def _duplicate_selected(self):
        row = self.list_widget.currentRow()
        item = self.list_widget.item(row)
        if item:
            data = item.data(Qt.UserRole)
            if data and data[0] == "roi":
                self.layer_stack.duplicate_roi(data[1])
                self.refresh()
                self.visibility_changed.emit()

    def _merge_selected(self):
        indices = self._get_selected_roi_indices()
        if len(indices) >= 2:
            self.layer_stack.merge_rois(indices)
            self.refresh()
            self.visibility_changed.emit()

    def _insert_roi_at(self, index):
        img = self.layer_stack.image_layer
        if img is None:
            return
        h, w = img.shape[:2]
        color = self.layer_stack.next_color()
        base = f"ROI {len(self.layer_stack.roi_layers) + 1}"
        name = generate_unique_roi_name(base, self.layer_stack.roi_layers)
        roi = ROILayer(name, w, h, color)
        self.layer_stack.insert_roi(index, roi)

    def _insert_roi_from_file(self, index, fmt):
        import os
        from PySide6.QtWidgets import QFileDialog
        img = self.layer_stack.image_layer
        if img is None:
            return
        h, w = img.shape[:2]
        if fmt == "roi":
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Import ImageJ ROI(s)", "",
                "ImageJ ROI (*.roi);;All Files (*)",
            )
            if not paths:
                return
            from montaris.io.imagej_roi import read_imagej_roi, imagej_roi_to_mask
            for i, path in enumerate(paths):
                roi_dict = read_imagej_roi(path)
                mask = imagej_roi_to_mask(roi_dict, w, h)
                name = os.path.splitext(os.path.basename(path))[0]
                roi = ROILayer(name, w, h)
                roi.mask = mask
                self.layer_stack.insert_roi(index + i, roi)
        elif fmt == "png":
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Import PNG Mask(s)", "",
                "PNG (*.png);;All Files (*)",
            )
            if not paths:
                return
            from PIL import Image
            for i, path in enumerate(paths):
                pil_img = Image.open(path).convert('L')
                arr = np.array(pil_img)
                if arr.shape != (h, w):
                    arr = np.array(pil_img.resize((w, h), Image.NEAREST))
                mask = (arr > 0).astype(np.uint8) * 255
                name = os.path.splitext(os.path.basename(path))[0]
                roi = ROILayer(name, w, h)
                roi.mask = mask
                self.layer_stack.insert_roi(index + i, roi)
        self.refresh()
        self.visibility_changed.emit()
        app = self.window()
        if hasattr(app, 'canvas'):
            app.canvas.refresh_overlays()

    def _rename_selected(self):
        row = self.list_widget.currentRow()
        item = self.list_widget.item(row)
        if item:
            data = item.data(Qt.UserRole)
            if data and data[0] == "roi":
                roi = self.layer_stack.get_roi(data[1])
                if roi:
                    name, ok = QInputDialog.getText(
                        self, "Rename ROI", "Name:", text=roi.name
                    )
                    if ok and name:
                        validated = generate_unique_roi_name(
                            name, self.layer_stack.roi_layers
                        )
                        roi.name = validated
                        self.refresh()

    def _move_roi_to(self, current_index):
        """Move ROI to a specific position (B.22)."""
        n = len(self.layer_stack.roi_layers)
        pos, ok = QInputDialog.getInt(
            self, "Move ROI To", f"Position (1-{n}):",
            current_index + 1, 1, n,
        )
        if ok:
            target = pos - 1
            if target != current_index:
                self.layer_stack.reorder_roi(current_index, target)
                self.refresh()
                self.visibility_changed.emit()

    def _export_single_roi(self, roi_index, fmt):
        self._export_selected_rois(fmt, override_indices=[roi_index])

    def _export_selected_rois(self, fmt, override_indices=None):
        indices = override_indices or self._get_selected_roi_indices()
        if not indices:
            return
        import os
        from PySide6.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(self, "Export Selected ROIs to Directory")
        if not dir_path:
            return
        count = 0
        for idx in indices:
            roi = self.layer_stack.get_roi(idx)
            if roi is None:
                continue
            safe_name = roi.name.replace("/", "_").replace("\\", "_")
            if fmt == "imagej":
                from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi
                roi_dict = mask_to_imagej_roi(roi.mask, roi.name)
                if roi_dict:
                    write_imagej_roi(roi_dict, os.path.join(dir_path, f"{safe_name}.roi"))
                    count += 1
            elif fmt == "png":
                from PIL import Image
                img = Image.fromarray(roi.mask)
                img.save(os.path.join(dir_path, f"{safe_name}.png"))
                count += 1
        # Show toast if app has one
        app = self.window()
        if hasattr(app, 'toast'):
            app.toast.show(f"Exported {count} ROI(s)", "success")

    def _bin_roi(self, roi_index):
        roi = self.layer_stack.get_roi(roi_index)
        if roi is None:
            return
        from PySide6.QtWidgets import (
            QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QComboBox,
            QStackedWidget, QWidget, QLabel,
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Bin ROI — Split into Regions")
        form = QFormLayout(dlg)

        method_combo = QComboBox()
        method_combo.addItems(["Region-aware (Voronoi)", "Grid"])
        form.addRow("Method:", method_combo)

        # Region-aware controls
        region_widget = QWidget()
        region_form = QFormLayout(region_widget)
        region_form.setContentsMargins(0, 0, 0, 0)
        n_bins_spin = QSpinBox()
        n_bins_spin.setRange(2, 500)
        n_bins_spin.setValue(10)
        region_form.addRow("Number of regions:", n_bins_spin)

        # Grid controls
        grid_widget = QWidget()
        grid_form = QFormLayout(grid_widget)
        grid_form.setContentsMargins(0, 0, 0, 0)
        h_spin = QSpinBox()
        h_spin.setRange(2, 512)
        h_spin.setValue(10)
        grid_form.addRow("Columns:", h_spin)
        v_spin = QSpinBox()
        v_spin.setRange(2, 512)
        v_spin.setValue(10)
        grid_form.addRow("Rows:", v_spin)

        stack = QStackedWidget()
        stack.addWidget(region_widget)
        stack.addWidget(grid_widget)
        method_combo.currentIndexChanged.connect(stack.setCurrentIndex)
        form.addRow(stack)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        if dlg.exec() != QDialog.Accepted:
            return

        mask = roi.mask
        h, w = mask.shape
        insert_at = roi_index + 1

        if method_combo.currentIndex() == 0:
            # Region-aware Voronoi binning
            sub_regions = _voronoi_bin(mask, n_bins_spin.value())
        else:
            # Grid binning
            sub_regions = _grid_bin(mask, h_spin.value(), v_spin.value())

        if not sub_regions:
            app = self.window()
            if hasattr(app, 'toast'):
                app.toast.show("No regions created (mask empty?)", "warning")
            return

        count = 0
        for label_idx, (sub_mask, y0, x0) in enumerate(sub_regions):
            cell_h, cell_w = sub_mask.shape
            name = generate_unique_roi_name(
                f"{roi.name}_bin{label_idx + 1}", self.layer_stack.roi_layers
            )
            new_roi = ROILayer(name, cell_w, cell_h)
            new_roi.mask = sub_mask
            new_roi.offset_x = x0
            new_roi.offset_y = y0
            new_roi.opacity = roi.opacity
            new_roi.fill_mode = roi.fill_mode
            self.layer_stack.insert_roi(insert_at + count, new_roi)
            count += 1

        app = self.window()
        if hasattr(app, 'canvas'):
            first_new = self.layer_stack.get_roi(insert_at)
            if first_new:
                app.canvas.set_active_layer(first_new)
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: (
                app.canvas.refresh_overlays(),
                self.refresh(),
            ))
        method_name = "Voronoi" if method_combo.currentIndex() == 0 else "Grid"
        if hasattr(app, 'toast'):
            app.toast.show(f"Split into {count} region(s) ({method_name})", "success")

    def _change_color(self):
        """Open color palette dialog (B.12)."""
        row = self.list_widget.currentRow()
        item = self.list_widget.item(row)
        if item:
            data = item.data(Qt.UserRole)
            if data and data[0] == "roi":
                roi = self.layer_stack.get_roi(data[1])
                if roi:
                    dlg = ColorPaletteDialog(roi.color, self)
                    if dlg.exec() and dlg.selected_color:
                        roi.color = dlg.selected_color
                        self.refresh()
                        self.visibility_changed.emit()

    def _random_color(self):
        """Assign random color to selected ROI (B.11)."""
        row = self.list_widget.currentRow()
        item = self.list_widget.item(row)
        if item:
            data = item.data(Qt.UserRole)
            if data and data[0] == "roi":
                roi = self.layer_stack.get_roi(data[1])
                if roi:
                    roi.color = (
                        random.randint(50, 255),
                        random.randint(50, 255),
                        random.randint(50, 255),
                    )
                    self.refresh()
                    self.visibility_changed.emit()

    def _toggle_all_visibility(self, checked):
        """Toggle all ROI visibility (B.14)."""
        for roi in self.layer_stack.roi_layers:
            roi.visible = checked
        self.refresh()
        self.visibility_changed.emit()

    def _on_global_opacity_changed(self, value):
        """Update global opacity factor (B.30)."""
        self.layer_stack._global_opacity_factor = value / 100.0
        self.visibility_changed.emit()
