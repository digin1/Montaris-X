from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QButtonGroup, QSpinBox, QCheckBox,
)
from PySide6.QtCore import Qt
from montaris import theme as _theme


class ZStackImportDialog(QDialog):
    """Asks the user how to import a z-stack TIFF into Montaris-X.

    Three modes:
      - 'max':    max-intensity projection along Z (recommended for ROI drawing)
      - 'slice':  pick a single Z index
      - 'synced': one channel per file, Z slider scrubs them in lockstep

    The 3D volume is kept in memory in all modes so the "View in 3D" dialog
    can render it regardless of which 2D representation the user chose.
    """

    def __init__(self, parent=None, filename="", n_slices=1, batch=False):
        super().__init__(parent)
        self.setWindowTitle("Z-stack detected")
        self.setMinimumWidth(420)
        self.setStyleSheet(_theme.alert_modal_style())

        self._mode = 'max'
        self._slice_index = n_slices // 2
        self._apply_to_batch = False
        self._accepted = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        header = QLabel(f"<b>{filename}</b> is a z-stack with {n_slices} slices.")
        header.setWordWrap(True)
        layout.addWidget(header)

        sub = QLabel("How would you like to load it?")
        sub.setWordWrap(True)
        layout.addWidget(sub)

        self._group = QButtonGroup(self)

        self._rb_max = QRadioButton("Max-intensity projection (recommended)")
        self._rb_max.setChecked(True)
        self._group.addButton(self._rb_max)
        layout.addWidget(self._rb_max)

        slice_row = QHBoxLayout()
        self._rb_slice = QRadioButton("Single slice at Z =")
        self._group.addButton(self._rb_slice)
        self._slice_spin = QSpinBox()
        self._slice_spin.setRange(0, max(0, n_slices - 1))
        self._slice_spin.setValue(self._slice_index)
        self._slice_spin.setEnabled(False)
        slice_row.addWidget(self._rb_slice)
        slice_row.addWidget(self._slice_spin)
        slice_row.addStretch(1)
        layout.addLayout(slice_row)

        self._rb_synced = QRadioButton("One channel per file, scrub Z in lockstep")
        self._group.addButton(self._rb_synced)
        layout.addWidget(self._rb_synced)

        self._rb_slice.toggled.connect(self._slice_spin.setEnabled)

        if batch:
            self._cb_batch = QCheckBox("Apply this choice to all z-stacks in this batch")
            self._cb_batch.setChecked(True)
            layout.addWidget(self._cb_batch)
        else:
            self._cb_batch = None

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel = QPushButton("Cancel")
        ok = QPushButton("Load")
        ok.setDefault(True)
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self._on_accept)
        btn_row.addWidget(cancel)
        btn_row.addWidget(ok)
        layout.addLayout(btn_row)

    def _on_accept(self):
        if self._rb_max.isChecked():
            self._mode = 'max'
        elif self._rb_slice.isChecked():
            self._mode = 'slice'
            self._slice_index = self._slice_spin.value()
        else:
            self._mode = 'synced'
        if self._cb_batch is not None:
            self._apply_to_batch = self._cb_batch.isChecked()
        self._accepted = True
        self.accept()

    @property
    def result_tuple(self):
        """(mode, slice_index, apply_to_batch)."""
        return (self._mode, self._slice_index, self._apply_to_batch)
