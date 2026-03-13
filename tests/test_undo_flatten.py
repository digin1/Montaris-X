"""Tests for FlattenUndoCommand in montaris.core.undo."""

import numpy as np
import pytest
from unittest.mock import MagicMock, call, patch

from montaris.core.undo import FlattenUndoCommand


def _make_roi(width=100, height=80):
    """Create a mock ROI layer with a real numpy mask."""
    roi = MagicMock()
    roi.mask = np.zeros((height, width), dtype=np.uint8)
    roi.offset_x = 0
    roi.offset_y = 0
    roi.invalidate_bbox = MagicMock()
    roi.flatten_offset = MagicMock()
    return roi


class TestFlattenUndoCommand:
    """Core undo/redo behavior for FlattenUndoCommand."""

    def test_undo_restores_mask_and_offset(self):
        roi = _make_roi(100, 80)
        old_crop = np.ones((10, 20), dtype=np.uint8) * 255
        old_bbox = (5, 15, 10, 30)
        old_offset = (3, 7)

        cmd = FlattenUndoCommand([(roi, old_crop, old_bbox, old_offset)])
        cmd.undo()

        # Mask should be cleared then crop pasted
        assert roi.mask[5:15, 10:30].sum() == old_crop.sum()
        assert roi.offset_x == 3
        assert roi.offset_y == 7
        roi.invalidate_bbox.assert_called_once()

    def test_undo_zeros_mask_before_restore(self):
        """Undo should zero-fill the entire mask first."""
        roi = _make_roi(50, 50)
        roi.mask[0:50, 0:50] = 128  # Nonzero everywhere

        old_crop = np.full((5, 5), 255, dtype=np.uint8)
        old_bbox = (10, 15, 10, 15)
        old_offset = (0, 0)

        cmd = FlattenUndoCommand([(roi, old_crop, old_bbox, old_offset)])
        cmd.undo()

        # Outside the crop region should be zero
        assert roi.mask[0, 0] == 0
        assert roi.mask[10, 10] == 255

    def test_redo_calls_flatten_offset(self):
        roi = _make_roi()
        old_crop = np.ones((5, 5), dtype=np.uint8)
        old_bbox = (0, 5, 0, 5)
        old_offset = (1, 2)

        cmd = FlattenUndoCommand([(roi, old_crop, old_bbox, old_offset)])
        cmd.redo()

        roi.flatten_offset.assert_called_once()
        roi.invalidate_bbox.assert_called_once()

    def test_byte_size_sum_of_crop_nbytes(self):
        crop1 = np.ones((10, 10), dtype=np.uint8)
        crop2 = np.ones((20, 20), dtype=np.uint8)
        roi1 = _make_roi()
        roi2 = _make_roi()

        cmd = FlattenUndoCommand([
            (roi1, crop1, (0, 10, 0, 10), (0, 0)),
            (roi2, crop2, (0, 20, 0, 20), (0, 0)),
        ])

        assert cmd.byte_size == crop1.nbytes + crop2.nbytes

    def test_byte_size_single_entry(self):
        crop = np.zeros((8, 16), dtype=np.uint8)
        roi = _make_roi()
        cmd = FlattenUndoCommand([(roi, crop, (0, 8, 0, 16), (0, 0))])
        assert cmd.byte_size == 8 * 16


class TestMultipleEntries:
    """Verify all entries are restored/flattened."""

    def test_undo_multiple_rois(self):
        roi1 = _make_roi(50, 50)
        roi2 = _make_roi(60, 60)
        crop1 = np.full((5, 5), 1, dtype=np.uint8)
        crop2 = np.full((8, 8), 2, dtype=np.uint8)

        cmd = FlattenUndoCommand([
            (roi1, crop1, (0, 5, 0, 5), (10, 20)),
            (roi2, crop2, (2, 10, 2, 10), (5, 5)),
        ])
        cmd.undo()

        assert roi1.mask[0:5, 0:5].sum() == 5 * 5
        assert roi1.offset_x == 10
        assert roi1.offset_y == 20
        assert roi2.mask[2:10, 2:10].sum() == 8 * 8 * 2
        assert roi2.offset_x == 5
        assert roi2.offset_y == 5
        assert roi1.invalidate_bbox.call_count == 1
        assert roi2.invalidate_bbox.call_count == 1

    def test_redo_multiple_rois(self):
        roi1 = _make_roi()
        roi2 = _make_roi()
        cmd = FlattenUndoCommand([
            (roi1, np.zeros((1, 1), dtype=np.uint8), (0, 1, 0, 1), (0, 0)),
            (roi2, np.zeros((1, 1), dtype=np.uint8), (0, 1, 0, 1), (0, 0)),
        ])
        cmd.redo()

        roi1.flatten_offset.assert_called_once()
        roi2.flatten_offset.assert_called_once()

    def test_roi_layer_attr_single_entry(self):
        """roi_layer attribute should be set when there's exactly one entry."""
        roi = _make_roi()
        cmd = FlattenUndoCommand([(roi, None, None, (0, 0))])
        assert cmd.roi_layer is roi

    def test_roi_layer_attr_multiple_entries(self):
        """roi_layer should be None when there are multiple entries."""
        roi1 = _make_roi()
        roi2 = _make_roi()
        cmd = FlattenUndoCommand([
            (roi1, None, None, (0, 0)),
            (roi2, None, None, (0, 0)),
        ])
        assert cmd.roi_layer is None


class TestNoneCropEntry:
    """Entry with None crop should be handled gracefully."""

    def test_undo_none_crop_zeros_mask(self):
        roi = _make_roi(30, 30)
        roi.mask[5:10, 5:10] = 200

        cmd = FlattenUndoCommand([(roi, None, None, (2, 3))])
        cmd.undo()

        # Mask should be all zeros (cleared but nothing pasted)
        assert roi.mask.sum() == 0
        assert roi.offset_x == 2
        assert roi.offset_y == 3
        roi.invalidate_bbox.assert_called_once()

    def test_byte_size_none_crop_excluded(self):
        roi = _make_roi()
        crop = np.ones((5, 5), dtype=np.uint8)
        cmd = FlattenUndoCommand([
            (roi, crop, (0, 5, 0, 5), (0, 0)),
            (roi, None, None, (0, 0)),
        ])
        # Only the non-None crop should count
        assert cmd.byte_size == crop.nbytes
