from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider,
    QPushButton, QFormLayout, QLabel,
)
from PySide6.QtCore import Signal, Qt
from montaris.core.adjustments import ImageAdjustments


class AdjustmentsPanel(QWidget):
    adjustments_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._adjustments = ImageAdjustments()
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        form = QFormLayout()

        # Brightness: -100 to 100 (maps to -1.0 to 1.0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self._on_brightness)
        self._brightness_label = QLabel("0")
        self._brightness_label.setFixedWidth(28)
        row = QHBoxLayout()
        row.addWidget(self.brightness_slider)
        row.addWidget(self._brightness_label)
        form.addRow("Brt:", row)

        # Contrast: -100 to 100 (maps to -1.0 to 1.0, 0 = identity)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self._on_contrast)
        self._contrast_label = QLabel("0")
        self._contrast_label.setFixedWidth(28)
        row = QHBoxLayout()
        row.addWidget(self.contrast_slider)
        row.addWidget(self._contrast_label)
        form.addRow("Con:", row)

        # Exposure: -200 to 200 (maps to -2.0 to 2.0)
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(-200, 200)
        self.exposure_slider.setValue(0)
        self.exposure_slider.valueChanged.connect(self._on_exposure)
        self._exposure_label = QLabel("0.00")
        self._exposure_label.setFixedWidth(28)
        row = QHBoxLayout()
        row.addWidget(self.exposure_slider)
        row.addWidget(self._exposure_label)
        form.addRow("Exp:", row)

        # Gamma: 10 to 500 (maps to 0.1 to 5.0)
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 500)
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self._on_gamma)
        self._gamma_label = QLabel("1.00")
        self._gamma_label.setFixedWidth(28)
        row = QHBoxLayout()
        row.addWidget(self.gamma_slider)
        row.addWidget(self._gamma_label)
        form.addRow("Gam:", row)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        auto_btn = QPushButton("Smart Auto")
        auto_btn.clicked.connect(self._on_auto)
        btn_layout.addWidget(auto_btn)

        boost_btn = QPushButton("Quick Boost")
        boost_btn.clicked.connect(self._on_boost)
        btn_layout.addWidget(boost_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(reset_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

        self._image_data = None

    def set_image_data(self, data):
        self._image_data = data
        self._adjustments.set_pivot(data)

    def _on_brightness(self, value):
        if self._updating:
            return
        self._adjustments.brightness = value / 100.0
        self._brightness_label.setText(str(value))
        self.adjustments_changed.emit(self._adjustments)

    def _on_contrast(self, value):
        if self._updating:
            return
        self._adjustments.contrast = value / 100.0
        self._contrast_label.setText(str(value))
        self.adjustments_changed.emit(self._adjustments)

    def _on_exposure(self, value):
        if self._updating:
            return
        self._adjustments.exposure = value / 100.0
        self._exposure_label.setText(f"{value / 100.0:.2f}")
        self.adjustments_changed.emit(self._adjustments)

    def _on_gamma(self, value):
        if self._updating:
            return
        self._adjustments.gamma = value / 100.0
        self._gamma_label.setText(f"{value / 100.0:.2f}")
        self.adjustments_changed.emit(self._adjustments)

    def _on_auto(self):
        if self._image_data is not None:
            self._adjustments = ImageAdjustments.smart_auto(self._image_data)
        else:
            self._adjustments = ImageAdjustments()
        self._sync_sliders()
        self.adjustments_changed.emit(self._adjustments)

    def _on_boost(self):
        if self._image_data is not None:
            self._adjustments = ImageAdjustments.quick_boost(self._image_data)
        else:
            self._adjustments = ImageAdjustments(contrast=0.15, brightness=0.1)
        self._sync_sliders()
        self.adjustments_changed.emit(self._adjustments)

    def _on_reset(self):
        self._adjustments.reset()
        self._sync_sliders()
        self.adjustments_changed.emit(self._adjustments)

    def _sync_sliders(self):
        self._updating = True
        self.brightness_slider.setValue(int(self._adjustments.brightness * 100))
        self.contrast_slider.setValue(int(self._adjustments.contrast * 100))
        self.exposure_slider.setValue(int(self._adjustments.exposure * 100))
        self.gamma_slider.setValue(int(self._adjustments.gamma * 100))
        self._brightness_label.setText(str(int(self._adjustments.brightness * 100)))
        self._contrast_label.setText(str(int(self._adjustments.contrast * 100)))
        self._exposure_label.setText(f"{self._adjustments.exposure:.2f}")
        self._gamma_label.setText(f"{self._adjustments.gamma:.2f}")
        self._updating = False

    @property
    def adjustments(self):
        return self._adjustments
