import os
import sys
import threading
from pathlib import Path

import pytest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from montaris.app import MontarisApp, apply_dark_theme
from montaris.layers import LayerStack, ImageLayer, ROILayer

ROOT = Path(__file__).resolve().parent.parent


# ── Markers ───────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "headed: requires display and real test files")
    config.addinivalue_line("markers", "benchmark: performance benchmark (slow)")


# ── Platform-safe timeout helper ──────────────────────────────────────

def platform_timeout(seconds):
    """Return a daemon threading.Timer that calls os._exit on expiry."""
    def _kill():
        print(f"\n[TIMEOUT] {seconds}s exceeded", file=sys.stderr)
        os._exit(99)
    t = threading.Timer(seconds, _kill)
    t.daemon = True
    return t


# ── Session fixtures ──────────────────────────────────────────────────

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


# ── Real test-data fixtures ───────────────────────────────────────────

@pytest.fixture(scope="session")
def real_image_path():
    p = ROOT / "test.tif"
    if not p.exists():
        pytest.skip("test.tif not found in project root")
    return p


@pytest.fixture(scope="session")
def real_zip_path():
    p = ROOT / "test.zip"
    if not p.exists():
        pytest.skip("test.zip not found in project root")
    return p


@pytest.fixture(scope="session")
def app_with_real_image(qapp, real_image_path, real_zip_path):
    """Load test.tif + test.zip once for the whole session."""
    from unittest.mock import patch
    from montaris.io.image_io import load_image

    window = MontarisApp()
    window.show()
    QApplication.processEvents()

    img_data = load_image(str(real_image_path))
    window.layer_stack.set_image(ImageLayer("test", img_data))
    window.canvas.refresh_image()
    window.canvas.fit_to_window()
    QApplication.processEvents()

    with patch.object(window, '_ask_replace_or_keep', return_value='keep'):
        window.import_roi_zip(str(real_zip_path))
    QApplication.processEvents()

    yield window
    window.close()


# ── Standard fixtures ─────────────────────────────────────────────────

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
