import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSlider, QFormLayout, QComboBox,
)
from PySide6.QtCore import Qt
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
        self.fill_mode_combo.addItems(["Solid", "Outline", "Solid + Outline"])
        self.fill_mode_combo.setStyleSheet(_theme.combobox_style())
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
            self.pixel_count_label.setText("-")
            return

        self.name_label.setText(layer.name)

        if getattr(layer, 'is_roi', False):
            self.opacity_slider.setValue(layer.opacity)
            fill_mode = getattr(layer, 'fill_mode', 'solid')
            _display = {'solid': 'Solid', 'outline': 'Outline', 'both': 'Solid + Outline'}
            self.fill_mode_combo.blockSignals(True)
            self.fill_mode_combo.setCurrentText(_display.get(fill_mode, 'Solid'))
            self.fill_mode_combo.blockSignals(False)
            self.fill_mode_combo.setEnabled(True)
            bbox = layer.get_bbox()
            if bbox is None:
                px = 0
            else:
                y1, y2, x1, x2 = bbox
                px = int(np.count_nonzero(layer.get_mask_crop((y1, y2, x1, x2))))
            self.pixel_count_label.setText(f"{px:,}")
        else:
            self.fill_mode_combo.setEnabled(False)
            self.pixel_count_label.setText("-")

    def _on_opacity_changed(self, value):
        if self._layer and hasattr(self._layer, 'opacity'):
            self._layer.opacity = value
            self.app.canvas.refresh_overlays_lut_only()

    def _on_fill_mode_changed(self, text):
        if self._layer and hasattr(self._layer, 'fill_mode'):
            _internal = {'Solid': 'solid', 'Outline': 'outline', 'Solid + Outline': 'both'}
            self._layer.fill_mode = _internal.get(text, 'solid')
            self.app.canvas.refresh_overlays()

    def refresh_theme(self):
        self.opacity_slider.setStyleSheet(_theme.slider_style())
        self.fill_mode_combo.setStyleSheet(_theme.combobox_style())
