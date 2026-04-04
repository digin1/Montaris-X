"""Grid canvas widget for multi-image workspaces.

Allows the user to define an NxM grid where each cell holds an independent
ImageCanvas + LayerStack.  Clicking a cell makes it the active workspace;
all panels and tools then operate on that cell.
"""
from dataclasses import dataclass, field
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QFrame, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QDialogButtonBox, QSizePolicy,
)
from PySide6.QtCore import Signal, Qt, QEvent
from PySide6.QtGui import QColor

from montaris.canvas import ImageCanvas
from montaris.layers import LayerStack, MontageDocument
from montaris.core.undo import UndoStack
from montaris.core.adjustments import ImageAdjustments


# ---------------------------------------------------------------------------
# Grid cell — one independent workspace
# ---------------------------------------------------------------------------

@dataclass
class GridCell:
    row: int
    col: int
    layer_stack: LayerStack
    canvas: ImageCanvas
    undo_stack: UndoStack
    adjustments: ImageAdjustments
    documents: list = field(default_factory=list)
    active_doc_index: int = -1
    downsample_factor: int = 1
    session_dir: str = None
    roi_import_path: str = None
    frame: QFrame = field(default=None, repr=False)

    def has_content(self):
        """Return True if this cell has an image or ROIs loaded."""
        ls = self.layer_stack
        return (ls.image_layer is not None or len(ls.roi_layers) > 0
                or len(self.documents) > 0)


# ---------------------------------------------------------------------------
# Active-cell border styling
# ---------------------------------------------------------------------------

_ACTIVE_BORDER = "QFrame { border: 2px solid #00b4ff; }"
_INACTIVE_BORDER = "QFrame { border: 2px solid transparent; }"
_INACTIVE_BORDER_MULTI = "QFrame { border: 1px solid #555; }"


# ---------------------------------------------------------------------------
# Grid setup dialog
# ---------------------------------------------------------------------------

class GridSetupDialog(QDialog):
    """Prompt user for grid rows and columns."""

    def __init__(self, current_rows=1, current_cols=1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Grid Layout")
        self.setMinimumWidth(250)

        layout = QVBoxLayout(self)

        form = QHBoxLayout()
        form.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 4)
        self.rows_spin.setValue(current_rows)
        form.addWidget(self.rows_spin)

        form.addWidget(QLabel("Columns:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 4)
        self.cols_spin.setValue(current_cols)
        form.addWidget(self.cols_spin)

        layout.addLayout(form)

        # Preview label
        self._preview = QLabel()
        self._update_preview()
        layout.addWidget(self._preview)
        self.rows_spin.valueChanged.connect(self._update_preview)
        self.cols_spin.valueChanged.connect(self._update_preview)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_preview(self):
        r, c = self.rows_spin.value(), self.cols_spin.value()
        self._preview.setText(f"  {r} x {c} = {r * c} canvas(es)")

    def result_size(self):
        return self.rows_spin.value(), self.cols_spin.value()


# ---------------------------------------------------------------------------
# Canvas grid container
# ---------------------------------------------------------------------------

class CanvasGrid(QWidget):
    """Container that holds a grid of ImageCanvas cells.

    Emits *active_cell_changed* whenever the user clicks a different cell.
    In 1x1 mode this behaves identically to a bare ImageCanvas.
    """

    active_cell_changed = Signal(object)  # emits GridCell

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self._app = app
        self._grid_layout = QGridLayout(self)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(2)
        self._cells: list[list[GridCell]] = []
        self._active_cell: GridCell | None = None
        self._rows = 0
        self._cols = 0
        self._maximized_cell: GridCell | None = None

    # -- Public API --------------------------------------------------------

    @property
    def active_cell(self) -> GridCell | None:
        return self._active_cell

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def is_single(self) -> bool:
        return self._rows == 1 and self._cols == 1

    @property
    def is_maximized(self) -> bool:
        return self._maximized_cell is not None

    def toggle_maximize(self, cell=None):
        """Maximize a cell to fill the entire grid, or restore the grid.

        If *cell* is None, uses the active cell.  If already maximized,
        restores the full grid regardless of which cell is passed.
        """
        if self.is_single():
            return  # nothing to toggle in 1x1

        if self._maximized_cell is not None:
            # Restore: show all cells and reset equal stretch
            self._maximized_cell = None
            for c in self.all_cells():
                if c.frame is not None:
                    c.frame.setVisible(True)
            for r in range(self._rows):
                self._grid_layout.setRowStretch(r, 1)
            for c in range(self._cols):
                self._grid_layout.setColumnStretch(c, 1)
            self._update_borders()
            return

        # Maximize the target cell
        target = cell or self._active_cell
        if target is None:
            return
        self._maximized_cell = target
        if target is not self._active_cell:
            self._set_active(target, emit=True)
        for c in self.all_cells():
            if c.frame is not None:
                c.frame.setVisible(c is target)
        # Give all stretch to the target's row/col so it fills the space
        for r in range(self._rows):
            self._grid_layout.setRowStretch(r, 1 if r == target.row else 0)
        for c in range(self._cols):
            self._grid_layout.setColumnStretch(c, 1 if c == target.col else 0)
        self._update_borders()

    def all_cells(self):
        """Iterate all GridCell objects in row-major order."""
        for row in self._cells:
            yield from row

    def cells_to_be_dropped(self, rows, cols):
        """Return list of cells that would be destroyed by resizing to rows x cols."""
        dropped = []
        for row in self._cells:
            for cell in row:
                if cell.row >= rows or cell.col >= cols:
                    dropped.append(cell)
        return dropped

    def setup_grid(self, rows, cols):
        """Create (or recreate) the grid with *rows* x *cols* cells.

        Existing cells at positions that still exist are preserved.
        New cells get fresh LayerStack / UndoStack / etc.
        """
        old_cells = {(c.row, c.col): c for row in self._cells for c in row}
        old_active = self._active_cell
        self._maximized_cell = None

        # Clear layout
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            # Don't delete preserved widgets — they'll be re-added
            w = item.widget()
            if w:
                w.setParent(None)

        self._cells = []
        self._rows = rows
        self._cols = cols
        is_multi = rows > 1 or cols > 1

        # Clear old stretch factors (handles shrinking from a larger grid)
        for i in range(self._grid_layout.rowCount()):
            self._grid_layout.setRowStretch(i, 0)
        for i in range(self._grid_layout.columnCount()):
            self._grid_layout.setColumnStretch(i, 0)

        for r in range(rows):
            row_cells = []
            for c in range(cols):
                if (r, c) in old_cells:
                    cell = old_cells[(r, c)]
                    cell.row = r
                    cell.col = c
                else:
                    cell = self._create_cell(r, c)
                # Wrap canvas in frame for border
                frame = self._wrap_in_frame(cell.canvas, is_multi)
                cell.frame = frame
                self._grid_layout.addWidget(frame, r, c)
                row_cells.append(cell)
            self._cells.append(row_cells)

        # Equal stretch so cells stay the same size regardless of content
        for r in range(rows):
            self._grid_layout.setRowStretch(r, 1)
        for c in range(cols):
            self._grid_layout.setColumnStretch(c, 1)

        # Destroy cells that no longer have a position
        for key, cell in old_cells.items():
            if key[0] >= rows or key[1] >= cols:
                cell.canvas.setParent(None)
                cell.canvas.deleteLater()

        # Restore or pick active cell
        if old_active and old_active.row < rows and old_active.col < cols:
            self._set_active(self._cells[old_active.row][old_active.col],
                             emit=False)
        else:
            self._set_active(self._cells[0][0], emit=True)

        self._update_borders()

    def cell_at(self, row, col) -> GridCell | None:
        if 0 <= row < self._rows and 0 <= col < self._cols:
            return self._cells[row][col]
        return None

    # -- Internal ----------------------------------------------------------

    def _create_cell(self, row, col) -> GridCell:
        ls = LayerStack()
        canvas = ImageCanvas(ls, self._app)
        adj = ImageAdjustments()
        canvas._adjustments = adj
        undo = UndoStack()
        # Install on the viewport — QGraphicsView delivers mouse events there
        canvas.viewport().installEventFilter(self)
        return GridCell(
            row=row, col=col,
            layer_stack=ls, canvas=canvas,
            undo_stack=undo, adjustments=adj,
        )

    def _wrap_in_frame(self, canvas, is_multi):
        frame = QFrame()
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(canvas)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if is_multi:
            frame.setStyleSheet(_INACTIVE_BORDER_MULTI)
        else:
            frame.setStyleSheet(_INACTIVE_BORDER)
        return frame

    def _set_active(self, cell, emit=True):
        if cell is self._active_cell:
            return
        self._active_cell = cell
        self._update_borders()
        if emit:
            self.active_cell_changed.emit(cell)

    def _update_borders(self):
        is_multi = self._rows > 1 or self._cols > 1
        for cell in self.all_cells():
            if cell.frame is None:
                continue
            if cell is self._active_cell:
                cell.frame.setStyleSheet(_ACTIVE_BORDER)
            else:
                cell.frame.setStyleSheet(
                    _INACTIVE_BORDER_MULTI if is_multi else _INACTIVE_BORDER
                )

    # -- Event filter: click to activate -----------------------------------

    def eventFilter(self, obj, event):
        """Intercept mouse clicks on child canvases to activate their cell."""
        if event.type() == QEvent.MouseButtonPress:
            for cell in self.all_cells():
                if cell.canvas.viewport() is obj:
                    if cell is not self._active_cell:
                        self._set_active(cell, emit=True)
                    break
        elif event.type() == QEvent.MouseButtonDblClick:
            for cell in self.all_cells():
                if cell.canvas.viewport() is obj:
                    self.toggle_maximize(cell)
                    break
        return False  # always propagate
