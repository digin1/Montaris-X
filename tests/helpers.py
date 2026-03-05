import numpy as np
from PySide6.QtCore import QPointF
from montaris.layers import ImageLayer, ROILayer


def create_test_image(w=200, h=150, channels=None):
    if channels:
        data = np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)
    else:
        data = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    return data


def create_test_roi(w=200, h=150, name="Test ROI"):
    return ROILayer(name, w, h)


def simulate_brush_stroke(tool, layer, canvas, points):
    if not points:
        return
    p0 = QPointF(*points[0])
    tool.on_press(p0, layer, canvas)
    for x, y in points[1:]:
        tool.on_move(QPointF(x, y), layer, canvas)
    tool.on_release(QPointF(*points[-1]), layer, canvas)


def capture_screenshot(widget):
    return widget.grab()
