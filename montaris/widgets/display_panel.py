from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QGroupBox,
)
from PySide6.QtCore import Signal
from montaris.core.display_modes import DisplayMode


class DisplayPanel(QWidget):
    mode_changed = Signal(object)
    channels_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._channel_checkboxes = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Display")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        for mode in DisplayMode:
            self.mode_combo.addItem(mode.value.replace("_", " ").title(), mode)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # Channel toggles
        self.channels_group = QGroupBox("Channels")
        self.channels_layout = QVBoxLayout(self.channels_group)
        layout.addWidget(self.channels_group)

        layout.addStretch()

    def set_channels(self, channel_names, active_indices=None):
        # Clear existing
        for cb in self._channel_checkboxes:
            self.channels_layout.removeWidget(cb)
            cb.deleteLater()
        self._channel_checkboxes.clear()

        for i, name in enumerate(channel_names):
            cb = QCheckBox(name)
            cb.setChecked(active_indices is None or i in active_indices)
            cb.stateChanged.connect(self._on_channel_toggled)
            self.channels_layout.addWidget(cb)
            self._channel_checkboxes.append(cb)

    def _on_mode_changed(self, index):
        mode = self.mode_combo.itemData(index)
        self.mode_changed.emit(mode)

    def _on_channel_toggled(self, state):
        active = [i for i, cb in enumerate(self._channel_checkboxes) if cb.isChecked()]
        self.channels_changed.emit(active)

    def get_active_channels(self):
        return [i for i, cb in enumerate(self._channel_checkboxes) if cb.isChecked()]

    def get_mode(self):
        return self.mode_combo.currentData()
