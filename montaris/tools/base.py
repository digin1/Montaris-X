from PySide6.QtCore import Qt


def expand_snapshot(mask, paint_bbox, snapshot_crop, snapshot_bbox):
    """Lazily expand a bbox-cropped pre-stroke snapshot to cover a new paint region."""
    py1, py2, px1, px2 = paint_bbox
    if snapshot_crop is None:
        return mask[py1:py2, px1:px2].copy(), paint_bbox
    sy1, sy2, sx1, sx2 = snapshot_bbox
    if py1 >= sy1 and py2 <= sy2 and px1 >= sx1 and px2 <= sx2:
        return snapshot_crop, snapshot_bbox  # already covered
    # Union bbox
    ey1, ey2 = min(sy1, py1), max(sy2, py2)
    ex1, ex2 = min(sx1, px1), max(sx2, px2)
    # Expansion area has original values (not yet painted by this stroke)
    expanded = mask[ey1:ey2, ex1:ex2].copy()
    # Restore pre-stroke values in old snapshot region (overwritten by earlier paints)
    expanded[sy1 - ey1:sy2 - ey1, sx1 - ex1:sx2 - ex1] = snapshot_crop
    return expanded, (ey1, ey2, ex1, ex2)


class BaseTool:
    name = "Base"

    def __init__(self, app):
        self.app = app

    def on_press(self, pos, layer, canvas):
        pass

    def on_move(self, pos, layer, canvas):
        pass

    def on_release(self, pos, layer, canvas):
        pass

    def on_key_press(self, key, canvas):
        pass

    def cursor(self):
        return Qt.CrossCursor
