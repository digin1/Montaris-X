import os
import sys
import pytest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from montaris.app import MontarisApp, apply_dark_theme
from montaris.layers import LayerStack, ImageLayer, ROILayer


@pytest.fixture(scope="session", autouse=True)
def _shutdown_worker_pool():
    yield
    from montaris.core.workers import shutdown_pool
    shutdown_pool()


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setApplicationName("Montaris-X-Test")
        apply_dark_theme(app)
    yield app


@pytest.fixture
def app(qapp):
    window = MontarisApp()
    window.show()
    yield window
    window.close()


@pytest.fixture
def layer_stack():
    return LayerStack()


@pytest.fixture
def image_layer():
    data = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
    return ImageLayer("test_image", data)


@pytest.fixture
def roi_layer():
    return ROILayer("test_roi", 120, 100)


@pytest.fixture
def app_with_image(app, image_layer):
    app.layer_stack.set_image(image_layer)
    app.canvas.refresh_image()
    app.canvas.fit_to_window()
    roi = ROILayer("ROI 1", 120, 100)
    app.layer_stack.add_roi(roi)
    app.canvas.set_active_layer(roi)
    app.canvas.refresh_overlays()
    app.layer_panel.refresh()
    return app
