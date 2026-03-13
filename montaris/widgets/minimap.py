import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QTimer
from PySide6.QtGui import QPainter, QImage, QPixmap, QColor, QPen
from montaris import theme as _theme


MINIMAP_SIZE = 200  # default / minimum height


class MiniMap(QWidget):
    """Thumbnail minimap showing viewport position. Click to pan."""
    pan_requested = Signal(float, float)  # scene x, y

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(MINIMAP_SIZE)
        self.setMinimumWidth(100)
        self._thumbnail = None
        self._viewport_rect = QRectF()
        self._scene_rect = QRectF()
        self._scale_x = 1.0
        self._scale_y = 1.0

    def set_image(self, image_data):
        """Set the thumbnail from image data (numpy array)."""
        if image_data is None:
            self._thumbnail = None
            self._image_data = None
            self.update()
            return

        self._image_data = image_data  # keep ref for resize rebuilds
        self._rebuild_thumbnail()

    def _rebuild_thumbnail(self):
        """Build thumbnail pixmap sized to current widget dimensions."""
        image_data = getattr(self, '_image_data', None)
        if image_data is None:
            return
        h, w = image_data.shape[:2]
        map_w = max(self.width(), 100)
        map_h = self.height()
        # Compute thumbnail size maintaining aspect ratio
        aspect = w / h
        if aspect > 1:
            tw = map_w
            th = int(map_w / aspect)
        else:
            th = map_h
            tw = int(map_h * aspect)

        # Create simple downsampled thumbnail
        step_y = max(1, h // th)
        step_x = max(1, w // tw)
        small = np.ascontiguousarray(image_data[::step_y, ::step_x])

        if small.ndim == 2:
            if small.dtype != np.uint8:
                mn, mx = float(small.min()), float(small.max())
                if mx > mn:
                    small = ((small.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    small = np.zeros_like(small, dtype=np.uint8)
            small = np.ascontiguousarray(small)
            sh, sw = small.shape
            qimg = QImage(small.data, sw, sh, sw, QImage.Format_Grayscale8)
        elif small.ndim == 3:
            if small.dtype != np.uint8:
                mn, mx = float(small.min()), float(small.max())
                if mx > mn:
                    small = ((small.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    small = np.zeros_like(small, dtype=np.uint8)
            sh, sw, sc = small.shape
            if sc >= 3:
                small = np.ascontiguousarray(small[:, :, :3])
                qimg = QImage(small.data, sw, sh, sw * 3, QImage.Format_RGB888)
            else:
                small = np.ascontiguousarray(small[:, :, 0])
                qimg = QImage(small.data, sw, sh, sw, QImage.Format_Grayscale8)
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
            tw = self._thumbnail.width() if self._thumbnail else self.width()
            th = self._thumbnail.height() if self._thumbnail else self.height()
            self._scale_x = tw / scene_rect.width()
            self._scale_y = th / scene_rect.height()
        self.update()

    def resizeEvent(self, event):
        """Rebuild thumbnail when dock is resized (debounced)."""
        super().resizeEvent(event)
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._rebuild_thumbnail)
        self._resize_timer.start(100)

    def paintEvent(self, event):
        painter = QPainter(self)
        bg, _ = _theme.minimap_colors()
        painter.fillRect(self.rect(), bg)

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

                _, vp_color = _theme.minimap_colors()
                pen = QPen(vp_color, 2)
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

    def refresh_theme(self):
        self.update()
