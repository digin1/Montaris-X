import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QImage, QPixmap, QColor, QPen


MINIMAP_SIZE = 200


class MiniMap(QWidget):
    """Thumbnail minimap showing viewport position. Click to pan."""
    pan_requested = Signal(float, float)  # scene x, y

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(MINIMAP_SIZE, MINIMAP_SIZE)
        self._thumbnail = None
        self._viewport_rect = QRectF()
        self._scene_rect = QRectF()
        self._scale_x = 1.0
        self._scale_y = 1.0

    def set_image(self, image_data):
        """Set the thumbnail from image data (numpy array)."""
        if image_data is None:
            self._thumbnail = None
            self.update()
            return

        h, w = image_data.shape[:2]
        # Compute thumbnail size maintaining aspect ratio
        aspect = w / h
        if aspect > 1:
            tw = MINIMAP_SIZE
            th = int(MINIMAP_SIZE / aspect)
        else:
            th = MINIMAP_SIZE
            tw = int(MINIMAP_SIZE * aspect)

        # Create simple downsampled thumbnail
        step_y = max(1, h // th)
        step_x = max(1, w // tw)
        small = image_data[::step_y, ::step_x]

        if small.ndim == 2:
            if small.dtype != np.uint8:
                mn, mx = float(small.min()), float(small.max())
                if mx > mn:
                    small = ((small.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    small = np.zeros_like(small, dtype=np.uint8)
            sh, sw = small.shape
            qimg = QImage(small.data, sw, sh, sw, QImage.Format_Grayscale8)
        elif small.ndim == 3:
            if small.dtype != np.uint8:
                mn, mx = float(small.min()), float(small.max())
                if mx > mn:
                    small = ((small.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    small = np.zeros_like(small, dtype=np.uint8)
            small = np.ascontiguousarray(small)
            sh, sw, sc = small.shape
            if sc >= 3:
                qimg = QImage(small[:, :, :3].data, sw, sh, sw * 3, QImage.Format_RGB888)
            else:
                qimg = QImage(small[:, :, 0].data, sw, sh, sw, QImage.Format_Grayscale8)
        else:
            return

        self._thumbnail = QPixmap.fromImage(qimg.copy())
        self._thumbnail = self._thumbnail.scaled(
            tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.update()

    def update_viewport(self, viewport_rect, scene_rect):
        """Update the viewport rectangle indicator."""
        self._viewport_rect = viewport_rect
        self._scene_rect = scene_rect
        if scene_rect.width() > 0 and scene_rect.height() > 0:
            self._scale_x = (self._thumbnail.width() if self._thumbnail else MINIMAP_SIZE) / scene_rect.width()
            self._scale_y = (self._thumbnail.height() if self._thumbnail else MINIMAP_SIZE) / scene_rect.height()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._thumbnail:
            # Center thumbnail
            x_off = (self.width() - self._thumbnail.width()) // 2
            y_off = (self.height() - self._thumbnail.height()) // 2
            painter.drawPixmap(x_off, y_off, self._thumbnail)

            # Draw viewport rectangle
            if not self._viewport_rect.isNull() and not self._scene_rect.isNull():
                vr = self._viewport_rect
                sr = self._scene_rect
                rx = x_off + (vr.x() - sr.x()) * self._scale_x
                ry = y_off + (vr.y() - sr.y()) * self._scale_y
                rw = vr.width() * self._scale_x
                rh = vr.height() * self._scale_y

                pen = QPen(QColor(255, 255, 0), 2)
                painter.setPen(pen)
                painter.drawRect(QRectF(rx, ry, rw, rh))

        painter.end()

    def mousePressEvent(self, event):
        self._pan_to(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._pan_to(event.position())

    def _pan_to(self, pos):
        if self._thumbnail is None or self._scene_rect.isNull():
            return
        x_off = (self.width() - self._thumbnail.width()) // 2
        y_off = (self.height() - self._thumbnail.height()) // 2
        # Convert minimap coords to scene coords
        scene_x = self._scene_rect.x() + (pos.x() - x_off) / self._scale_x
        scene_y = self._scene_rect.y() + (pos.y() - y_off) / self._scale_y
        self.pan_requested.emit(scene_x, scene_y)
