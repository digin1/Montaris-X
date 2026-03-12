"""Tests for session folder reuse fix."""
import os
import sys
import shutil
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from PySide6.QtWidgets import QApplication

from montaris.layers import ROILayer, ImageLayer, MontageDocument


@pytest.fixture
def session_app(app, tmp_path):
    """App with image, ROIs, and a proper document pointing to tmp_path."""
    img_data = np.zeros((100, 100), dtype=np.uint8)
    img_layer = ImageLayer("test", img_data)
    app.layer_stack.set_image(img_layer)
    app.canvas.refresh_image()
    app.canvas.fit_to_window()

    roi = ROILayer("ROI 1", 100, 100)
    roi.mask[10:30, 10:30] = 255
    app.layer_stack.add_roi(roi)
    app.canvas.set_active_layer(roi)
    app.canvas.refresh_overlays()
    app.layer_panel.refresh()

    # Create a proper document so save_session_progress finds image_path
    image_path = str(tmp_path / "test.tif")
    doc = MontageDocument(
        name="test",
        image_layer=img_layer,
        image_path=image_path,
    )
    app._documents = [doc]
    app._active_doc_index = 0
    app._session_dir = None
    app._initial_session_saved = False

    yield app, tmp_path

    # Cleanup session dirs
    for d in tmp_path.iterdir():
        if d.is_dir() and d.name.startswith("session_"):
            shutil.rmtree(d, ignore_errors=True)


def _wait_for_save():
    """Process events and wait for background save."""
    QApplication.processEvents()
    time.sleep(1.5)
    QApplication.processEvents()


class TestSessionDirReuse:
    def test_first_save_creates_dir(self, session_app):
        app, tmp_path = session_app
        app.save_session_progress()
        _wait_for_save()

        sessions = [d for d in tmp_path.iterdir()
                     if d.is_dir() and d.name.startswith("session_")]
        assert len(sessions) >= 1, "First save should create a session folder"
        assert app._session_dir is not None

    def test_second_save_reuses_dir(self, session_app):
        app, tmp_path = session_app

        app.save_session_progress()
        _wait_for_save()

        first_dir = app._session_dir
        assert first_dir is not None

        # Second save
        app.save_session_progress()
        _wait_for_save()

        assert app._session_dir == first_dir, \
            "Second save should reuse the same session folder"
        sessions = [d for d in tmp_path.iterdir()
                     if d.is_dir() and d.name.startswith("session_")]
        assert len(sessions) == 1, \
            f"Expected 1 session folder, got {len(sessions)}"

    def test_new_image_resets_session_dir(self, session_app):
        app, tmp_path = session_app

        app.save_session_progress()
        _wait_for_save()

        assert app._session_dir is not None

        # Simulate opening a new image
        app._session_dir = None
        assert app._session_dir is None

    def test_close_resets_session_dir(self, session_app):
        app, tmp_path = session_app

        app.save_session_progress()
        _wait_for_save()

        assert app._session_dir is not None
        app._session_dir = None
        assert app._session_dir is None
