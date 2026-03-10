import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSlider, QFormLayout, QComboBox,
)
from PySide6.QtCore import Qt


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

        self.type_label = QLabel("-")
        form.addRow("Type:", self.type_label)

        self.size_label = QLabel("-")
        form.addRow("Size:", self.size_label)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(128)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        form.addRow("Opacity (Selected ROIs):", self.opacity_slider)

        self.fill_mode_combo = QComboBox()
        self.fill_mode_combo.addItems(["Solid", "Outline", "Both"])
        self.fill_mode_combo.currentTextChanged.connect(self._on_fill_mode_changed)
        form.addRow("Fill:", self.fill_mode_combo)

        self.pixel_count_label = QLabel("-")
        form.addRow("Pixels:", self.pixel_count_label)

        layout.addLayout(form)
        layout.addStretch()

    def set_layer(self, layer):
        self._layer = layer
        if layer is None:
            self.name_label.setText("-")
            self.type_label.setText("-")
            self.size_label.setText("-")
            self.pixel_count_label.setText("-")
            return

        self.name_label.setText(layer.name)

        if hasattr(layer, 'mask'):
            self.type_label.setText("ROI")
            h, w = layer.mask.shape
            self.size_label.setText(f"{w} x {h}")
            self.opacity_slider.setValue(layer.opacity)
            fill_mode = getattr(layer, 'fill_mode', 'solid')
            self.fill_mode_combo.blockSignals(True)
            self.fill_mode_combo.setCurrentText(fill_mode.capitalize())
            self.fill_mode_combo.blockSignals(False)
            self.fill_mode_combo.setEnabled(True)
            bbox = layer.get_bbox()
            if bbox is None:
                px = 0
            else:
                y1, y2, x1, x2 = bbox
                px = int(np.count_nonzero(layer.mask[y1:y2, x1:x2]))
            self.pixel_count_label.setText(f"{px:,}")
        else:
            self.type_label.setText("Image")
            shape = layer.data.shape
            if len(shape) == 2:
                self.size_label.setText(f"{shape[1]} x {shape[0]}")
            else:
                self.size_label.setText(f"{shape[1]} x {shape[0]} x {shape[2]}ch")
            self.fill_mode_combo.setEnabled(False)
            self.pixel_count_label.setText("-")

    def _on_opacity_changed(self, value):
        if self._layer and hasattr(self._layer, 'opacity'):
            self._layer.opacity = value
            self.app.canvas.refresh_overlays_lut_only()

    def _on_fill_mode_changed(self, text):
        if self._layer and hasattr(self._layer, 'fill_mode'):
            self._layer.fill_mode = text.lower()
            self.app.canvas.refresh_overlays()
