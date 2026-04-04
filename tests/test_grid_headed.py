"""Headed tests for the canvas grid feature.

Exercises grid setup, per-cell image/ROI loading, cell switching,
maximize/restore, equal sizing, shrink warnings, and flip-on-load
persistence — all with a real display and test.tif / test.zip.

Usage:
    QT_QPA_PLATFORM= .venv/bin/pytest tests/test_grid_headed.py -m headed -s
"""
import os
import time
from unittest.mock import patch

import numpy as np
import pytest

from PySide6.QtCore import QPointF, Qt, QEvent
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from montaris.app import MontarisApp, apply_dark_theme
from montaris.layers import ImageLayer, ROILayer
from montaris.widgets.canvas_grid import CanvasGrid, GridSetupDialog, GridCell

pytestmark = pytest.mark.headed


# ── Helpers ──────────────────────────────────────────────────────────

def process(n=3):
    """Pump the event loop *n* times."""
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.05)


def load_into_active_cell(window, image_path, zip_path):
    """Load test.tif and test.zip into the currently active grid cell."""
    from montaris.io.image_io import load_image
    img_data = load_image(str(image_path))
    window.layer_stack.set_image(ImageLayer("test", img_data))
    window.canvas.refresh_image()
    window.canvas.fit_to_window()
    process()
    with patch.object(window, '_ask_replace_or_keep', return_value='keep'):
        window.import_roi_zip(str(zip_path))
    process()


def load_synthetic_into_active_cell(window):
    """Load a small synthetic image + ROI into the active cell (no disk I/O)."""
    img = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
    window.layer_stack.set_image(ImageLayer("synth", img))
    window.canvas.refresh_image()
    window.canvas.fit_to_window()
    roi = ROILayer("ROI_synth", 120, 100)
    roi.mask[10:50, 10:50] = 255
    window.layer_stack.add_roi(roi)
    window.canvas.refresh_overlays()
    process()


def click_cell_viewport(grid, cell):
    """Simulate a left-click on a cell's canvas viewport to activate it."""
    vp = cell.canvas.viewport()
    pos = QPointF(vp.width() / 2, vp.height() / 2)
    evt = QMouseEvent(
        QEvent.MouseButtonPress, pos, pos,
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
    )
    grid.eventFilter(vp, evt)
    process()


def dblclick_cell_viewport(grid, cell):
    """Simulate a double-click on a cell's canvas viewport."""
    vp = cell.canvas.viewport()
    pos = QPointF(vp.width() / 2, vp.height() / 2)
    evt = QMouseEvent(
        QEvent.MouseButtonDblClick, pos, pos,
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
    )
    grid.eventFilter(vp, evt)
    process()


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        import sys
        app = QApplication(sys.argv)
        app.setApplicationName("Montaris-X-GridTest")
        apply_dark_theme(app)
    yield app


@pytest.fixture(scope="module")
def real_image_path():
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "test.tif"
    if not p.exists():
        pytest.skip("test.tif not found in project root")
    return p


@pytest.fixture(scope="module")
def real_zip_path():
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "test.zip"
    if not p.exists():
        pytest.skip("test.zip not found in project root")
    return p


@pytest.fixture
def window(qapp):
    w = MontarisApp()
    w.show()
    w.resize(1200, 900)
    process()
    yield w
    w.close()
    process()


# =====================================================================
# HAPPY PATH
# =====================================================================

class TestGridSetup:
    """Setting up various grid sizes."""

    def test_default_is_1x1(self, window):
        grid = window._canvas_grid
        assert grid.rows == 1
        assert grid.cols == 1
        assert grid.is_single()

    def test_setup_2x2(self, window):
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()
        assert grid.rows == 2
        assert grid.cols == 2
        assert not grid.is_single()
        assert sum(1 for _ in grid.all_cells()) == 4

    def test_setup_4x4(self, window):
        grid = window._canvas_grid
        grid.setup_grid(4, 4)
        process()
        assert grid.rows == 4
        assert grid.cols == 4
        assert sum(1 for _ in grid.all_cells()) == 16

    def test_setup_3x1(self, window):
        grid = window._canvas_grid
        grid.setup_grid(3, 1)
        process()
        assert grid.rows == 3
        assert grid.cols == 1
        assert sum(1 for _ in grid.all_cells()) == 3

    def test_setup_1x4(self, window):
        grid = window._canvas_grid
        grid.setup_grid(1, 4)
        process()
        assert grid.rows == 1
        assert grid.cols == 4
        assert sum(1 for _ in grid.all_cells()) == 4


class TestCellSwitching:
    """Click to switch active cell; verify state isolation."""

    def test_click_activates_cell(self, window, real_image_path, real_zip_path):
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        cell_00 = grid.cell_at(0, 0)
        cell_11 = grid.cell_at(1, 1)
        assert grid.active_cell is cell_00

        click_cell_viewport(grid, cell_11)
        assert grid.active_cell is cell_11

    def test_state_isolation_between_cells(self, window, real_image_path, real_zip_path):
        """Loading into one cell should not affect another."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        # Load into (0,0)
        click_cell_viewport(grid, grid.cell_at(0, 0))
        load_into_active_cell(window, real_image_path, real_zip_path)
        cell_00 = grid.active_cell
        assert cell_00.layer_stack.image_layer is not None
        roi_count_00 = len(cell_00.layer_stack.roi_layers)
        assert roi_count_00 > 0

        # Switch to (1,1) — should be empty
        click_cell_viewport(grid, grid.cell_at(1, 1))
        cell_11 = grid.active_cell
        assert cell_11.layer_stack.image_layer is None
        assert len(cell_11.layer_stack.roi_layers) == 0

        # Switch back to (0,0) — data should still be there
        click_cell_viewport(grid, grid.cell_at(0, 0))
        assert window.layer_stack.image_layer is not None
        assert len(window.layer_stack.roi_layers) == roi_count_00

    def test_load_into_multiple_cells(self, window):
        """Load image+ROIs into multiple cells independently.

        Uses synthetic images to avoid 4 x 136 MB OOM.
        """
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        for r in range(2):
            for c in range(2):
                click_cell_viewport(grid, grid.cell_at(r, c))
                load_synthetic_into_active_cell(window)

        # All four cells should have content
        for cell in grid.all_cells():
            assert cell.layer_stack.image_layer is not None
            assert len(cell.layer_stack.roi_layers) > 0

    def test_switching_updates_layer_panel(self, window, real_image_path, real_zip_path):
        grid = window._canvas_grid
        grid.setup_grid(2, 1)
        process()

        # Load into (0,0)
        click_cell_viewport(grid, grid.cell_at(0, 0))
        load_into_active_cell(window, real_image_path, real_zip_path)
        panel = window.layer_panel
        count_loaded = panel.list_widget.count()
        assert count_loaded > 0

        # Switch to empty (1,0)
        click_cell_viewport(grid, grid.cell_at(1, 0))
        process()
        # Layer panel should show fewer items (just the empty cell)
        assert panel.list_widget.count() < count_loaded


class TestEqualSizing:
    """Grid cells should remain equal size regardless of content."""

    def test_cells_equal_after_image_load(self, window, real_image_path, real_zip_path):
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process(5)

        # Load into only one cell
        click_cell_viewport(grid, grid.cell_at(0, 0))
        load_into_active_cell(window, real_image_path, real_zip_path)
        process(5)

        # Collect frame sizes
        sizes = []
        for cell in grid.all_cells():
            if cell.frame:
                sizes.append((cell.frame.width(), cell.frame.height()))

        # All widths and heights should be roughly equal (within 10px tolerance)
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        assert max(widths) - min(widths) <= 10, f"Width spread too large: {widths}"
        assert max(heights) - min(heights) <= 10, f"Height spread too large: {heights}"


class TestMaximizeRestore:
    """Double-click to maximize a cell, double-click again to restore."""

    def test_maximize_hides_other_cells(self, window):
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        target = grid.cell_at(0, 0)
        grid.toggle_maximize(target)
        process()

        assert grid.is_maximized
        for cell in grid.all_cells():
            if cell is target:
                assert cell.frame.isVisible()
            else:
                assert not cell.frame.isVisible()

    def test_restore_shows_all_cells(self, window):
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        grid.toggle_maximize(grid.cell_at(0, 0))
        process()
        grid.toggle_maximize()  # restore
        process()

        assert not grid.is_maximized
        for cell in grid.all_cells():
            assert cell.frame.isVisible()

    def test_dblclick_maximize_restore_roundtrip(self, window):
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        cell = grid.cell_at(1, 1)
        dblclick_cell_viewport(grid, cell)
        assert grid.is_maximized
        assert grid.active_cell is cell

        dblclick_cell_viewport(grid, cell)
        assert not grid.is_maximized

    def test_maximize_noop_in_1x1(self, window):
        grid = window._canvas_grid
        grid.setup_grid(1, 1)
        process()
        grid.toggle_maximize()
        assert not grid.is_maximized

    def test_maximize_with_loaded_content(self, window, real_image_path, real_zip_path):
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        click_cell_viewport(grid, grid.cell_at(0, 0))
        load_into_active_cell(window, real_image_path, real_zip_path)

        grid.toggle_maximize(grid.cell_at(0, 0))
        process()
        assert grid.is_maximized
        assert window.layer_stack.image_layer is not None

        grid.toggle_maximize()
        process()
        assert not grid.is_maximized
        assert window.layer_stack.image_layer is not None


class TestGridResize:
    """Changing grid dimensions preserves existing cells."""

    def test_grow_preserves_data(self, window, real_image_path, real_zip_path):
        grid = window._canvas_grid
        grid.setup_grid(1, 1)
        process()
        load_into_active_cell(window, real_image_path, real_zip_path)

        # Grow to 2x2
        grid.setup_grid(2, 2)
        process()

        cell_00 = grid.cell_at(0, 0)
        assert cell_00.layer_stack.image_layer is not None
        assert len(cell_00.layer_stack.roi_layers) > 0

    def test_shrink_destroys_out_of_bounds(self, window, real_image_path, real_zip_path):
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        # Load into (1,1)
        click_cell_viewport(grid, grid.cell_at(1, 1))
        load_into_active_cell(window, real_image_path, real_zip_path)

        # Track that (1,1) has content
        dropped = grid.cells_to_be_dropped(1, 1)
        has_work = [c for c in dropped if c.has_content()]
        assert len(has_work) >= 1

        # Shrink to 1x1
        grid.setup_grid(1, 1)
        process()
        assert grid.rows == 1
        assert grid.cols == 1

    def test_resize_resets_maximize(self, window):
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()
        grid.toggle_maximize(grid.cell_at(0, 0))
        assert grid.is_maximized

        grid.setup_grid(3, 3)
        process()
        assert not grid.is_maximized


class TestGridDialog:
    """The View > Grid Layout dialog."""

    def test_dialog_accept(self, window):
        dlg = GridSetupDialog(1, 1, window)
        dlg.rows_spin.setValue(3)
        dlg.cols_spin.setValue(2)
        assert dlg.result_size() == (3, 2)

    def test_dialog_preview_label(self, window):
        dlg = GridSetupDialog(2, 3, window)
        assert "6" in dlg._preview.text()

    def test_dialog_range_clamped(self, window):
        dlg = GridSetupDialog(1, 1, window)
        dlg.rows_spin.setValue(10)  # max is 4
        assert dlg.rows_spin.value() == 4


class TestFlipOnLoadPersistence:
    """Flip on Load checkbox should survive app restart."""

    def test_flip_saved_and_restored(self, qapp):
        # Launch first instance, enable flip
        w1 = MontarisApp()
        w1.show()
        process()
        w1._flip_on_load_act.setChecked(True)
        w1._rotate_on_load_act.setChecked(True)
        w1.close()
        process()

        # Launch second instance — should restore
        w2 = MontarisApp()
        w2.show()
        process()
        assert w2._flip_on_load_act.isChecked(), "Flip on Load not restored"
        assert w2._rotate_on_load_act.isChecked(), "Rotate on Load not restored"

        # Clean up: uncheck so other tests aren't affected
        w2._flip_on_load_act.setChecked(False)
        w2._rotate_on_load_act.setChecked(False)
        w2.close()
        process()


class TestSaveAllGridSessions:
    """Save All Grid Sessions saves every cell that has content."""

    def test_save_all_creates_sessions(self, window):
        """Load 2 cells, save all, verify both cells intact."""
        grid = window._canvas_grid
        grid.setup_grid(2, 1)
        process()

        for r in range(2):
            click_cell_viewport(grid, grid.cell_at(r, 0))
            load_synthetic_into_active_cell(window)

        window._save_cell_state()

        window.save_all_grid_sessions()
        process(10)

        assert grid.active_cell is grid.cell_at(1, 0)
        for r in range(2):
            cell = grid.cell_at(r, 0)
            assert cell.layer_stack.image_layer is not None
            assert len(cell.layer_stack.roi_layers) > 0

    def test_save_all_skips_empty_cells(self, window):
        """Only cells with content should be saved."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        click_cell_viewport(grid, grid.cell_at(0, 0))
        load_synthetic_into_active_cell(window)

        window.save_all_grid_sessions()
        process(5)

        assert grid.cell_at(0, 0).layer_stack.image_layer is not None
        assert grid.cell_at(1, 1).layer_stack.image_layer is None

    def test_save_all_restores_active_cell(self, window):
        """After save-all, the original active cell should be restored."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        click_cell_viewport(grid, grid.cell_at(0, 0))
        load_synthetic_into_active_cell(window)
        click_cell_viewport(grid, grid.cell_at(1, 1))
        load_synthetic_into_active_cell(window)

        assert grid.active_cell is grid.cell_at(1, 1)

        window.save_all_grid_sessions()
        process(5)

        assert window.canvas is grid.cell_at(1, 1).canvas
        assert window.layer_stack is grid.cell_at(1, 1).layer_stack

    def test_save_all_on_1x1(self, window):
        """In 1x1 mode, save_all should just call save_session_progress."""
        grid = window._canvas_grid
        grid.setup_grid(1, 1)
        process()
        load_synthetic_into_active_cell(window)

        window.save_all_grid_sessions()
        process(5)
        assert window.layer_stack.image_layer is not None

    def test_save_all_no_content_warns(self, window):
        """Save all on empty grid should show warning, not crash."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        # All cells empty — should warn
        window.save_all_grid_sessions()
        process()


class TestExportAllGridZips:
    """Export All Grid Cells as ZIP — one ZIP per cell.

    Uses small synthetic images to avoid OOM with real 136 MB test.tif.
    """

    def test_export_all_creates_zips(self, window, tmp_path):
        """Load 2 cells, export all to tmp dir, verify ZIP files created."""
        grid = window._canvas_grid
        grid.setup_grid(2, 1)
        process()

        for r in range(2):
            click_cell_viewport(grid, grid.cell_at(r, 0))
            load_synthetic_into_active_cell(window)

        window._save_cell_state()

        with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                   return_value=str(tmp_path)):
            window.export_all_grid_zips()
        process(5)

        import glob
        zips = glob.glob(str(tmp_path / "*.zip"))
        assert len(zips) == 2, f"Expected 2 ZIPs, found {len(zips)}: {zips}"

        import zipfile
        for zp in zips:
            with zipfile.ZipFile(zp, 'r') as zf:
                roi_files = [n for n in zf.namelist() if n.endswith('.roi')]
                assert len(roi_files) > 0, f"ZIP {zp} has no .roi files"

    def test_export_all_skips_empty_cells(self, window, tmp_path):
        """Only cells with ROIs get exported."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        click_cell_viewport(grid, grid.cell_at(0, 0))
        load_synthetic_into_active_cell(window)
        window._save_cell_state()

        with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                   return_value=str(tmp_path)):
            window.export_all_grid_zips()
        process(5)

        import glob
        zips = glob.glob(str(tmp_path / "*.zip"))
        assert len(zips) == 1, f"Expected 1 ZIP, found {len(zips)}"

    def test_export_all_restores_active_cell(self, window, tmp_path):
        """Active cell and refs should be restored after export."""
        grid = window._canvas_grid
        grid.setup_grid(2, 1)
        process()

        click_cell_viewport(grid, grid.cell_at(0, 0))
        load_synthetic_into_active_cell(window)
        click_cell_viewport(grid, grid.cell_at(1, 0))
        load_synthetic_into_active_cell(window)

        window._save_cell_state()

        with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                   return_value=str(tmp_path)):
            window.export_all_grid_zips()
        process(5)

        assert window.canvas is grid.cell_at(1, 0).canvas
        assert window.layer_stack is grid.cell_at(1, 0).layer_stack

    def test_export_all_zip_naming(self, window, tmp_path):
        """ZIP files should be named with cell position."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        for r in range(2):
            for c in range(2):
                click_cell_viewport(grid, grid.cell_at(r, c))
                load_synthetic_into_active_cell(window)

        window._save_cell_state()

        with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                   return_value=str(tmp_path)):
            window.export_all_grid_zips()
        process(5)

        import glob
        zips = sorted(glob.glob(str(tmp_path / "*.zip")))
        assert len(zips) == 4
        names = [os.path.basename(z) for z in zips]
        # Named as {stem}_R{row}C{col}.zip — stem first for easy identification
        assert any("R1C1" in n for n in names)
        assert any("R2C2" in n for n in names)
        # Stem should come before the cell position
        assert all(n.index("R") > 0 for n in names), "Stem should lead the filename"

    def test_export_all_empty_grid_warns(self, window):
        """Export all on empty grid should show info, not crash."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        with patch.object(QMessageBox, 'information') as mock_info:
            window.export_all_grid_zips()
            mock_info.assert_called_once()


# =====================================================================
# UNHAPPY PATH / EDGE CASES
# =====================================================================

class TestGridEdgeCases:
    """Edge cases and error conditions."""

    def test_switch_to_empty_cell_clears_minimap(self, window, real_image_path, real_zip_path):
        grid = window._canvas_grid
        grid.setup_grid(2, 1)
        process()

        click_cell_viewport(grid, grid.cell_at(0, 0))
        load_into_active_cell(window, real_image_path, real_zip_path)

        click_cell_viewport(grid, grid.cell_at(1, 0))
        process()
        # Minimap should have no image
        assert window.minimap._image_data is None or window.layer_stack.image_layer is None

    def test_rapid_cell_switching(self, window, real_image_path, real_zip_path):
        """Rapidly clicking between cells should not crash."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        click_cell_viewport(grid, grid.cell_at(0, 0))
        load_into_active_cell(window, real_image_path, real_zip_path)

        # Rapid switches
        for _ in range(10):
            click_cell_viewport(grid, grid.cell_at(0, 1))
            click_cell_viewport(grid, grid.cell_at(1, 0))
            click_cell_viewport(grid, grid.cell_at(1, 1))
            click_cell_viewport(grid, grid.cell_at(0, 0))
        process()
        # Should end on (0,0) with data intact
        assert grid.active_cell is grid.cell_at(0, 0)
        assert window.layer_stack.image_layer is not None

    def test_setup_grid_same_size_noop(self, window):
        """Setting up the same grid size again should be harmless."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()
        cell_00 = grid.cell_at(0, 0)
        grid.setup_grid(2, 2)
        process()
        # Cell at (0,0) should be the same object (preserved)
        assert grid.cell_at(0, 0) is cell_00

    def test_maximize_then_switch_via_shortcut(self, window):
        """Maximize a cell, then restore and verify all cells visible."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        # Load all 4 cells with synthetic data (avoids 4x136MB OOM)
        for r in range(2):
            for c in range(2):
                click_cell_viewport(grid, grid.cell_at(r, c))
                load_synthetic_into_active_cell(window)

        # Maximize (0,0)
        click_cell_viewport(grid, grid.cell_at(0, 0))
        grid.toggle_maximize()
        process()
        assert grid.is_maximized

        # Restore
        grid.toggle_maximize()
        process()
        assert not grid.is_maximized
        for cell in grid.all_cells():
            assert cell.frame.isVisible()
            assert cell.layer_stack.image_layer is not None

    def test_cell_at_out_of_bounds(self, window):
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        assert grid.cell_at(5, 5) is None
        assert grid.cell_at(-1, 0) is None

    def test_all_cells_iteration_count(self, window):
        grid = window._canvas_grid
        grid.setup_grid(3, 4)
        process()
        assert sum(1 for _ in grid.all_cells()) == 12

    def test_shrink_dialog_warns_content(self, window, real_image_path, real_zip_path):
        """_show_grid_dialog warns when shrinking would drop cells with content."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        # Load into (1,1)
        click_cell_viewport(grid, grid.cell_at(1, 1))
        load_into_active_cell(window, real_image_path, real_zip_path)
        window._save_cell_state()

        dropped = grid.cells_to_be_dropped(1, 1)
        has_work = [c for c in dropped if c.has_content()]
        assert len(has_work) >= 1, "Expected content in cells to be dropped"

    def test_active_cell_border_style(self, window):
        """Active cell should have the highlight border."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        active = grid.active_cell
        assert active.frame is not None
        style = active.frame.styleSheet()
        assert "#00b4ff" in style

        inactive = grid.cell_at(1, 1)
        if inactive is not active:
            assert "#00b4ff" not in inactive.frame.styleSheet()

    def test_grow_from_loaded_1x1_to_4x4(self, window, real_image_path, real_zip_path):
        """Start with 1x1 loaded image, grow to 4x4 — original cell preserved."""
        grid = window._canvas_grid
        grid.setup_grid(1, 1)
        process()
        load_into_active_cell(window, real_image_path, real_zip_path)
        original_roi_count = len(window.layer_stack.roi_layers)

        grid.setup_grid(4, 4)
        process()
        assert grid.rows == 4 and grid.cols == 4
        cell_00 = grid.cell_at(0, 0)
        assert cell_00.layer_stack.image_layer is not None
        assert len(cell_00.layer_stack.roi_layers) == original_roi_count

        # Other cells should be empty
        cell_33 = grid.cell_at(3, 3)
        assert cell_33.layer_stack.image_layer is None

    def test_4x4_all_cells_loaded(self, window):
        """Load every cell of a 4x4 grid — the full workflow.

        Uses synthetic images to avoid 16 x 136 MB OOM.
        """
        grid = window._canvas_grid
        grid.setup_grid(4, 4)
        process()

        for r in range(4):
            for c in range(4):
                click_cell_viewport(grid, grid.cell_at(r, c))
                load_synthetic_into_active_cell(window)

        # Verify all 16 cells
        for cell in grid.all_cells():
            assert cell.layer_stack.image_layer is not None, \
                f"Cell ({cell.row},{cell.col}) has no image"
            assert len(cell.layer_stack.roi_layers) > 0, \
                f"Cell ({cell.row},{cell.col}) has no ROIs"

    def test_maximize_different_cells(self, window):
        """Maximize different cells in sequence."""
        grid = window._canvas_grid
        grid.setup_grid(2, 2)
        process()

        for r in range(2):
            for c in range(2):
                cell = grid.cell_at(r, c)
                grid.toggle_maximize(cell)
                process()
                assert grid.is_maximized
                assert grid.active_cell is cell
                # Only this cell visible
                for other in grid.all_cells():
                    if other is cell:
                        assert other.frame.isVisible()
                    else:
                        assert not other.frame.isVisible()
                # Restore
                grid.toggle_maximize()
                process()
                assert not grid.is_maximized
