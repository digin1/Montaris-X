"""Tests for montaris.io.instructions: load_instructions and apply_instructions."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from montaris.io.instructions import load_instructions, apply_instructions


# ---------------------------------------------------------------------------
# load_instructions tests
# ---------------------------------------------------------------------------

class TestLoadInstructions:
    def test_load_valid_json(self, tmp_path):
        """Load a fully-populated instructions file and verify keys."""
        data = {
            "version": 1,
            "image_path": "img.tif",
            "roi_path": "rois.npz",
            "adjustments": {
                "brightness": 0.1,
                "contrast": 1.2,
                "exposure": 0.0,
                "gamma": 1.0,
            },
            "display_mode": "composite_rgb",
            "operations": [
                {"type": "fix_overlaps", "priority": "later_wins"},
                {"type": "export", "format": "npz", "path": "out.npz"},
            ],
        }
        p = tmp_path / "full.json"
        p.write_text(json.dumps(data))

        result = load_instructions(str(p))
        assert isinstance(result, dict)
        assert result["version"] == 1
        assert result["image_path"] == "img.tif"
        assert "adjustments" in result
        assert len(result["operations"]) == 2

    def test_load_minimal_fields(self, tmp_path):
        """A file with only version should load without crashing."""
        data = {"version": 1}
        p = tmp_path / "minimal.json"
        p.write_text(json.dumps(data))

        result = load_instructions(str(p))
        assert result == {"version": 1}

    def test_nonexistent_file_raises(self, tmp_path):
        """Opening a path that does not exist should raise FileNotFoundError."""
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_instructions(str(missing))

    def test_invalid_json_raises(self, tmp_path):
        """Malformed JSON should raise json.JSONDecodeError."""
        p = tmp_path / "bad.json"
        p.write_text("{not valid json!!!")

        with pytest.raises(json.JSONDecodeError):
            load_instructions(str(p))


# ---------------------------------------------------------------------------
# apply_instructions tests
# ---------------------------------------------------------------------------

class TestApplyInstructions:
    @staticmethod
    def _make_app():
        """Build a lightweight mock of MontarisApp."""
        app = MagicMock()
        app.layer_stack = MagicMock()
        app.layer_stack.roi_layers = []
        app.canvas = MagicMock()
        app.adjustments_panel = MagicMock()
        app.adjustments_panel._adjustments = None
        app.adjustments_panel._sync_sliders = MagicMock()
        app.adjustments_panel.adjustments_changed = MagicMock()
        return app

    def test_empty_instructions_returns_empty_log(self):
        """An empty dict should produce no actions and an empty log."""
        app = self._make_app()
        log = apply_instructions(app, {})
        assert log == []

    @patch("montaris.layers.ImageLayer")
    @patch("montaris.io.image_io.load_image")
    def test_image_path_calls_load_image(self, mock_load_image, mock_image_layer):
        """When image_path is present, load_image should be invoked."""
        import numpy as np

        mock_load_image.return_value = np.zeros((64, 64), dtype=np.uint8)
        mock_image_layer.return_value = MagicMock()
        app = self._make_app()
        instructions = {"image_path": "/some/image.tif"}

        log = apply_instructions(app, instructions)

        assert any("Loaded image" in msg for msg in log)
        mock_load_image.assert_called_once_with("/some/image.tif")
        app.layer_stack.set_image.assert_called_once()
        app.canvas.refresh_image.assert_called_once()

    def test_adjustments_updates_panel(self):
        """When adjustments are present, the panel should be updated."""
        app = self._make_app()
        adj_dict = {"brightness": 0.5, "contrast": 1.0, "exposure": 0.0, "gamma": 1.0}
        instructions = {"adjustments": adj_dict}

        with patch("montaris.core.adjustments.ImageAdjustments") as MockAdj:
            MockAdj.return_value = MagicMock()
            log = apply_instructions(app, instructions)

        assert any("Applied adjustments" in msg for msg in log)
        app.adjustments_panel._sync_sliders.assert_called_once()
        app.adjustments_panel.adjustments_changed.emit.assert_called_once()
