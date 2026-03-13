from PySide6.QtCore import Qt
from montaris.tools.base import BaseTool
from montaris.core.selection import SelectionModel


class SelectTool(BaseTool):
    """Click to select ROIs. No drawing — just selection."""
    name = "Select ROI"

    def on_press(self, pos, layer, canvas):
        hit = SelectionModel.hit_test(
            pos.x(), pos.y(), canvas.layer_stack.roi_layers
        )
        if hit is not None:
            canvas._selection.set([hit])
        else:
            canvas._selection.clear()

    def cursor(self):
        return Qt.ArrowCursor
