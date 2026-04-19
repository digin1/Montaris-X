from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QGroupBox,
)
from PySide6.QtCore import Signal
from montaris.core.display_modes import DisplayMode
from montaris import theme as _theme


# Channel checkboxes with names longer than this get their visible text
# truncated with an ellipsis; the full name lives in the tooltip. Keeps the
# right dock narrow even when TIFFs carry very long filenames.
_CHANNEL_LABEL_MAX_CHARS = 28


def _elide(name: str) -> str:
    if len(name) <= _CHANNEL_LABEL_MAX_CHARS:
        return name
    # Keep head + tail so the channel suffix (e.g. ``_z17``, ``_C0``) stays
    # readable — the discriminator is what users need to tell channels apart.
    head = _CHANNEL_LABEL_MAX_CHARS - 10
    return f"{name[:head]}\u2026{name[-8:]}"


class DisplayPanel(QWidget):
    mode_changed = Signal(object)
    channels_changed = Signal(list)
    composite_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._channel_checkboxes = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Composite toggle
        self.composite_cb = QCheckBox("Composite View")
        self.composite_cb.setToolTip("Merge all channels using their tint colors")
        self.composite_cb.setStyleSheet(_theme.checkbox_style())
        self.composite_cb.toggled.connect(self._on_composite_toggled)
        layout.addWidget(self.composite_cb)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.setStyleSheet(_theme.combobox_style())
        for mode in DisplayMode:
            self.mode_combo.addItem(mode.value.replace("_", " ").title(), mode)
        # Default to False Color (uses tint colors)
        false_color_idx = [m for m in DisplayMode].index(DisplayMode.FALSE_COLOR)
        self.mode_combo.setCurrentIndex(false_color_idx)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # Channel toggles
        self.channels_group = QGroupBox("Channels")
        self.channels_group.setStyleSheet(_theme.groupbox_style())
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
            cb = QCheckBox(_elide(name))
            cb.setToolTip(name)
            cb.setChecked(active_indices is None or i in active_indices)
            cb.setStyleSheet(_theme.checkbox_style())
            cb.stateChanged.connect(self._on_channel_toggled)
            self.channels_layout.addWidget(cb)
            self._channel_checkboxes.append(cb)

    def _on_mode_changed(self, index):
        mode = self.mode_combo.itemData(index)
        self.mode_changed.emit(mode)

    def _on_channel_toggled(self, state):
        active = [i for i, cb in enumerate(self._channel_checkboxes) if cb.isChecked()]
        self.channels_changed.emit(active)

    def _on_composite_toggled(self, checked):
        self.composite_toggled.emit(checked)

    def get_active_channels(self):
        return [i for i, cb in enumerate(self._channel_checkboxes) if cb.isChecked()]

    def get_mode(self):
        return self.mode_combo.currentData()

    def refresh_theme(self):
        self.composite_cb.setStyleSheet(_theme.checkbox_style())
        self.mode_combo.setStyleSheet(_theme.combobox_style())
        self.channels_group.setStyleSheet(_theme.groupbox_style())
        for cb in self._channel_checkboxes:
            cb.setStyleSheet(_theme.checkbox_style())
