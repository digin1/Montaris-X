from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QColorDialog, QInputDialog, QMenu, QAbstractItemView,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QIcon, QPixmap, QAction

from montaris.layers import ROILayer, ROI_COLORS


class LayerPanel(QWidget):
    selection_changed = Signal(object)
    visibility_changed = Signal()
    roi_added = Signal()
    roi_removed = Signal(int)

    def __init__(self, layer_stack, parent=None):
        super().__init__(parent)
        self.layer_stack = layer_stack
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Layers")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self.list_widget)

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

        self.dup_btn = QPushButton("Dup")
        self.dup_btn.setToolTip("Duplicate selected ROI")
        self.dup_btn.clicked.connect(self._duplicate_selected)
        btn_layout.addWidget(self.dup_btn)

        self.merge_btn = QPushButton("Merge")
        self.merge_btn.setToolTip("Merge selected ROIs")
        self.merge_btn.clicked.connect(self._merge_selected)
        btn_layout.addWidget(self.merge_btn)

        self.rename_btn = QPushButton("Rename")
        self.rename_btn.clicked.connect(self._rename_selected)
        btn_layout.addWidget(self.rename_btn)

        self.color_btn = QPushButton("Color")
        self.color_btn.clicked.connect(self._change_color)
        btn_layout.addWidget(self.color_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def refresh(self):
        self._updating = True
        self.list_widget.clear()

        if self.layer_stack.image_layer:
            item = QListWidgetItem(self.layer_stack.image_layer.name)
            item.setData(Qt.UserRole, ("image", 0))
            item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable & ~Qt.ItemIsDragEnabled)
            self.list_widget.addItem(item)

        for i, roi in enumerate(self.layer_stack.roi_layers):
            icon = self._color_icon(roi.color)
            item = QListWidgetItem(roi.name)
            item.setIcon(icon)
            item.setData(Qt.UserRole, ("roi", i))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if roi.visible else Qt.Unchecked)
            self.list_widget.addItem(item)

        self._updating = False

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
                roi.visible = item.checkState() == Qt.Checked
                self.visibility_changed.emit()

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

        insert_above_action = QAction("Insert ROI Above", self)
        insert_above_action.triggered.connect(lambda: self._insert_roi_at(data[1]))
        menu.addAction(insert_above_action)

        insert_below_action = QAction("Insert ROI Below", self)
        insert_below_action.triggered.connect(lambda: self._insert_roi_at(data[1] + 1))
        menu.addAction(insert_below_action)

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
        # Remove in reverse order to preserve indices
        for idx in sorted(indices, reverse=True):
            self.roi_removed.emit(idx)

    def _duplicate_selected(self):
        row = self.list_widget.currentRow()
        item = self.list_widget.item(row)
        if item:
            data = item.data(Qt.UserRole)
            if data and data[0] == "roi":
                self.layer_stack.duplicate_roi(data[1])

    def _merge_selected(self):
        indices = self._get_selected_roi_indices()
        if len(indices) >= 2:
            self.layer_stack.merge_rois(indices)

    def _insert_roi_at(self, index):
        img = self.layer_stack.image_layer
        if img is None:
            return
        h, w = img.shape[:2]
        color_idx = self.layer_stack._color_index
        color = ROI_COLORS[color_idx % len(ROI_COLORS)]
        self.layer_stack._color_index = (color_idx + 1) % len(ROI_COLORS)
        name = f"ROI {len(self.layer_stack.roi_layers) + 1}"
        roi = ROILayer(name, w, h, color)
        self.layer_stack.insert_roi(index, roi)

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
                        roi.name = name
                        self.refresh()

    def _change_color(self):
        row = self.list_widget.currentRow()
        item = self.list_widget.item(row)
        if item:
            data = item.data(Qt.UserRole)
            if data and data[0] == "roi":
                roi = self.layer_stack.get_roi(data[1])
                if roi:
                    color = QColorDialog.getColor(QColor(*roi.color), self)
                    if color.isValid():
                        roi.color = (color.red(), color.green(), color.blue())
                        self.refresh()
                        self.visibility_changed.emit()
