import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSlider, QFormLayout, QComboBox,
    QPushButton, QColorDialog, QSpinBox, QHBoxLayout,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap, QIcon
from montaris import theme as _theme


class PropertiesPanel(QWidget):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self._layer = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        form = QFormLayout()

        self.name_label = QLabel("-")
        form.addRow("Name:", self.name_label)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(128)
        self.opacity_slider.setStyleSheet(_theme.slider_style())
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        form.addRow("Opacity (Selected ROIs):", self.opacity_slider)

        self.fill_mode_combo = QComboBox()
        self.fill_mode_combo.addItems(["Solid", "Boundary", "Solid + Boundary"])
        self.fill_mode_combo.setStyleSheet(_theme.combobox_style())
        self.fill_mode_combo.currentTextChanged.connect(self._on_fill_mode_changed)
        form.addRow("Fill:", self.fill_mode_combo)

        self.pixel_count_label = QLabel("-")
        self.pixel_count_row_label = QLabel("Pixels:")
        form.addRow(self.pixel_count_row_label, self.pixel_count_label)

        # Boundary settings
        self.thickness_spin = QSpinBox()
        self.thickness_spin.setRange(1, 10)
        self.thickness_spin.setValue(1)
        self.thickness_spin.setToolTip("Boundary width (higher = thicker)")
        self.thickness_spin.valueChanged.connect(self._on_thickness_changed)
        form.addRow("Boundary Thickness:", self.thickness_spin)

        color_row = QHBoxLayout()
        self.boundary_color_btn = QPushButton()
        self.boundary_color_btn.setFixedSize(24, 24)
        self.boundary_color_btn.setToolTip("Boundary colour for all ROIs")
        self.boundary_color_btn.clicked.connect(self._pick_boundary_color)
        color_row.addWidget(self.boundary_color_btn)
        color_row.addWidget(QLabel("All"))

        self.active_color_btn = QPushButton()
        self.active_color_btn.setFixedSize(24, 24)
        self.active_color_btn.setToolTip("Boundary colour for selected ROIs")
        self.active_color_btn.clicked.connect(self._pick_active_color)
        color_row.addWidget(self.active_color_btn)
        color_row.addWidget(QLabel("Selected"))
        color_row.addStretch()
        form.addRow("Boundary Colour:", color_row)

        layout.addLayout(form)
        layout.addStretch()

        # Sync initial button colors
        self._update_color_buttons()

    def _update_color_buttons(self):
        ls = self.app.layer_stack if hasattr(self.app, 'layer_stack') else None
        bc = getattr(ls, 'boundary_color', (255, 255, 0))
        ac = getattr(ls, 'active_boundary_color', (0, 255, 255))
        for btn, c in ((self.boundary_color_btn, bc), (self.active_color_btn, ac)):
            px = QPixmap(20, 20)
            px.fill(QColor(*c))
            btn.setIcon(QIcon(px))

    def set_layer(self, layer):
        self._layer = layer
        if layer is None:
            self.name_label.setText("-")
            self.pixel_count_label.setText("-")
            return

        self.name_label.setText(layer.name)

        if getattr(layer, 'is_roi', False):
            self.opacity_slider.setValue(layer.opacity)
            # Sync fill mode from global setting
            fill_mode = getattr(self.app.layer_stack, 'fill_mode', 'solid')
            _display = {'solid': 'Solid', 'boundary': 'Boundary',
                        'both': 'Solid + Boundary',
                        'outline': 'Boundary'}  # backward compat
            self.fill_mode_combo.blockSignals(True)
            self.fill_mode_combo.setCurrentText(_display.get(fill_mode, 'Solid'))
            self.fill_mode_combo.blockSignals(False)
            self.fill_mode_combo.setEnabled(True)
            if getattr(layer, 'is_volume', False):
                # 3D ROI: report whole-volume voxel count, not a single slice.
                self.pixel_count_row_label.setText("Voxels:")
                self.pixel_count_label.setText(f"{layer.voxel_count():,}")
            else:
                self.pixel_count_row_label.setText("Pixels:")
                bbox = layer.get_bbox()
                if bbox is None:
                    px = 0
                else:
                    y1, y2, x1, x2 = bbox
                    px = int(np.count_nonzero(layer.get_mask_crop((y1, y2, x1, x2))))
                self.pixel_count_label.setText(f"{px:,}")
        else:
            self.fill_mode_combo.setEnabled(False)
            self.pixel_count_row_label.setText("Pixels:")
            self.pixel_count_label.setText("-")

    def _on_opacity_changed(self, value):
        if self._layer and hasattr(self._layer, 'opacity'):
            self._layer.opacity = value
            self.app.canvas.refresh_overlays_lut_only()

    def _on_fill_mode_changed(self, text):
        _internal = {'Solid': 'solid', 'Boundary': 'boundary',
                     'Solid + Boundary': 'both'}
        mode = _internal.get(text, 'solid')
        self.app.layer_stack.fill_mode = mode
        self.app.canvas.refresh_overlays()

    def _on_thickness_changed(self, value):
        self.app.layer_stack.boundary_thickness = value
        self.app.canvas.refresh_overlays()

    def _pick_boundary_color(self):
        c = self.app.layer_stack.boundary_color
        color = QColorDialog.getColor(
            QColor(*c), self, "Boundary Colour",
            options=QColorDialog.DontUseNativeDialog,
        )
        if color.isValid():
            self.app.layer_stack.boundary_color = (color.red(), color.green(), color.blue())
            self._update_color_buttons()
            self.app.canvas.refresh_overlays()

    def _pick_active_color(self):
        c = self.app.layer_stack.active_boundary_color
        color = QColorDialog.getColor(
            QColor(*c), self, "Active Boundary Colour",
            options=QColorDialog.DontUseNativeDialog,
        )
        if color.isValid():
            self.app.layer_stack.active_boundary_color = (color.red(), color.green(), color.blue())
            self._update_color_buttons()
            self.app.canvas.refresh_overlays()

    def refresh_theme(self):
        self.opacity_slider.setStyleSheet(_theme.slider_style())
        self.fill_mode_combo.setStyleSheet(_theme.combobox_style())
