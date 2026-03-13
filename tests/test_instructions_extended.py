"""Extended tests for montaris.io.instructions: apply_instructions operations loop."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from montaris.io.instructions import load_instructions, apply_instructions


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


class TestRoiPathLoading:
    """Instructions with roi_path trigger load_roi_set."""

    @patch("montaris.io.roi_io.load_roi_set")
    def test_roi_path_calls_load_roi_set(self, mock_load):
        mock_load.return_value = [MagicMock(), MagicMock()]
        app = _make_app()
        instructions = {"roi_path": "some/rois.npz"}

        log = apply_instructions(app, instructions)

        mock_load.assert_called_once_with("some/rois.npz")
        assert app.layer_stack.add_roi.call_count == 2
        app.canvas.refresh_overlays.assert_called_once()
        assert any("Loaded 2 ROIs" in msg for msg in log)

    @patch("montaris.io.roi_io.load_roi_set")
    def test_roi_path_empty_list(self, mock_load):
        """If load_roi_set returns empty list, log should still record 0 ROIs."""
        mock_load.return_value = []
        app = _make_app()
        log = apply_instructions(app, {"roi_path": "empty.npz"})

        assert any("Loaded 0 ROIs" in msg for msg in log)


class TestAdjustments:
    """Instructions with adjustments dict update the panel."""

    def test_adjustments_updates_panel(self):
        app = _make_app()
        adj_dict = {"brightness": 0.2, "contrast": 1.1, "exposure": 0.0, "gamma": 1.0}
        instructions = {"adjustments": adj_dict}

        with patch("montaris.core.adjustments.ImageAdjustments") as MockAdj:
            MockAdj.return_value = MagicMock()
            log = apply_instructions(app, instructions)

        assert any("Applied adjustments" in msg for msg in log)
        app.adjustments_panel._sync_sliders.assert_called_once()
        app.adjustments_panel.adjustments_changed.emit.assert_called_once()

    def test_adjustments_no_panel_attribute(self):
        """If adjustments_panel is missing, adjustments are still logged."""
        app = _make_app()
        del app.adjustments_panel
        adj_dict = {"brightness": 0.0, "contrast": 1.0, "exposure": 0.0, "gamma": 1.0}
        instructions = {"adjustments": adj_dict}

        with patch("montaris.core.adjustments.ImageAdjustments") as MockAdj:
            MockAdj.return_value = MagicMock()
            log = apply_instructions(app, instructions)

        assert any("Applied adjustments" in msg for msg in log)


class TestFixOverlapsOperation:
    """Operations with type: fix_overlaps."""

    @patch("montaris.core.roi_ops.fix_overlaps")
    def test_fix_overlaps_called_with_priority(self, mock_fix):
        app = _make_app()
        instructions = {
            "operations": [{"type": "fix_overlaps", "priority": "earlier_wins"}]
        }

        log = apply_instructions(app, instructions)

        mock_fix.assert_called_once_with(app.layer_stack.roi_layers, "earlier_wins")
        app.canvas.refresh_overlays.assert_called_once()
        assert any("Fixed overlaps" in msg for msg in log)
        assert any("earlier_wins" in msg for msg in log)

    @patch("montaris.core.roi_ops.fix_overlaps")
    def test_fix_overlaps_default_priority(self, mock_fix):
        app = _make_app()
        instructions = {"operations": [{"type": "fix_overlaps"}]}

        log = apply_instructions(app, instructions)

        mock_fix.assert_called_once_with(app.layer_stack.roi_layers, "later_wins")


class TestExportNpz:
    """Operations with type: export, format: npz."""

    @patch("montaris.io.roi_io.save_roi_set")
    def test_export_npz(self, mock_save):
        app = _make_app()
        instructions = {
            "operations": [{"type": "export", "format": "npz", "path": "out/rois.npz"}]
        }

        log = apply_instructions(app, instructions)

        mock_save.assert_called_once_with("out/rois.npz", app.layer_stack.roi_layers)
        assert any("Exported NPZ" in msg for msg in log)


class TestExportImageJ:
    """Operations with type: export, format: imagej."""

    @patch("montaris.io.imagej_roi.write_imagej_roi")
    @patch("montaris.io.imagej_roi.mask_to_imagej_roi")
    def test_export_imagej(self, mock_m2r, mock_write, tmp_path):
        roi1 = MagicMock()
        roi1.mask = MagicMock()
        roi1.name = "cell_1"
        mock_m2r.return_value = {"type": "polygon"}

        app = _make_app()
        app.layer_stack.roi_layers = [roi1]
        out_dir = str(tmp_path / "imagej_out")
        instructions = {
            "operations": [{"type": "export", "format": "imagej", "path": out_dir}]
        }

        log = apply_instructions(app, instructions)

        mock_m2r.assert_called_once_with(roi1.mask, "cell_1")
        mock_write.assert_called_once()
        assert any("Exported ImageJ" in msg for msg in log)

    @patch("montaris.io.imagej_roi.write_imagej_roi")
    @patch("montaris.io.imagej_roi.mask_to_imagej_roi")
    def test_export_imagej_skip_none_roi(self, mock_m2r, mock_write, tmp_path):
        """If mask_to_imagej_roi returns None, write should not be called."""
        roi1 = MagicMock()
        roi1.mask = MagicMock()
        roi1.name = "empty"
        mock_m2r.return_value = None

        app = _make_app()
        app.layer_stack.roi_layers = [roi1]
        out_dir = str(tmp_path / "imagej_out2")
        instructions = {
            "operations": [{"type": "export", "format": "imagej", "path": out_dir}]
        }

        log = apply_instructions(app, instructions)

        mock_m2r.assert_called_once()
        mock_write.assert_not_called()


class TestExportPng:
    """Operations with type: export, format: png."""

    def test_export_png_calls_app_method(self):
        app = _make_app()
        instructions = {
            "operations": [{"type": "export", "format": "png", "path": "overlay.png"}]
        }

        log = apply_instructions(app, instructions)

        app.export_roi_png_to.assert_called_once_with("overlay.png")
        assert any("Exported PNG" in msg for msg in log)


class TestUnknownOperation:
    """Unknown operation types should be skipped without error."""

    def test_unknown_op_type_skipped(self):
        app = _make_app()
        instructions = {
            "operations": [{"type": "do_magic", "param": 42}]
        }

        log = apply_instructions(app, instructions)

        # No crash; no log entry for unknown operation
        assert all("do_magic" not in msg for msg in log)


class TestMultipleOperations:
    """Multiple operations in sequence."""

    @patch("montaris.io.roi_io.save_roi_set")
    @patch("montaris.core.roi_ops.fix_overlaps")
    def test_fix_then_export(self, mock_fix, mock_save):
        app = _make_app()
        instructions = {
            "operations": [
                {"type": "fix_overlaps", "priority": "later_wins"},
                {"type": "export", "format": "npz", "path": "out.npz"},
            ]
        }

        log = apply_instructions(app, instructions)

        mock_fix.assert_called_once()
        mock_save.assert_called_once()
        assert len(log) == 2

    @patch("montaris.io.roi_io.save_roi_set")
    @patch("montaris.core.roi_ops.fix_overlaps")
    def test_operation_order_preserved(self, mock_fix, mock_save):
        """fix_overlaps should be called before save_roi_set."""
        call_order = []
        mock_fix.side_effect = lambda *a, **k: call_order.append("fix")
        mock_save.side_effect = lambda *a, **k: call_order.append("save")

        app = _make_app()
        instructions = {
            "operations": [
                {"type": "fix_overlaps"},
                {"type": "export", "format": "npz", "path": "o.npz"},
            ]
        }

        apply_instructions(app, instructions)
        assert call_order == ["fix", "save"]

    def test_empty_operations_list(self):
        app = _make_app()
        log = apply_instructions(app, {"operations": []})
        assert log == []
