import numpy as np
import pytest
from montaris.layers import ROILayer
from montaris.core.undo import UndoStack, UndoCommand


class TestPerROIUndo:
    def test_per_roi_stack(self):
        roi = ROILayer("test", 100, 100)
        roi.undo_stack = UndoStack(max_size=10)

        # Paint something
        roi.mask[10:20, 10:20] = 255
        old = np.zeros((10, 10), dtype=np.uint8)
        new = np.full((10, 10), 255, dtype=np.uint8)
        cmd = UndoCommand(roi, (10, 20, 10, 20), old, new)
        roi.undo_stack.push(cmd)

        assert roi.mask[15, 15] == 255
        roi.undo_stack.undo()
        assert roi.mask[15, 15] == 0
        roi.undo_stack.redo()
        assert roi.mask[15, 15] == 255

    def test_independent_stacks(self):
        roi1 = ROILayer("A", 100, 100)
        roi2 = ROILayer("B", 100, 100)
        roi1.undo_stack = UndoStack(max_size=10)
        roi2.undo_stack = UndoStack(max_size=10)

        # Paint on roi1
        roi1.mask[10:20, 10:20] = 255
        cmd1 = UndoCommand(
            roi1, (10, 20, 10, 20),
            np.zeros((10, 10), dtype=np.uint8),
            np.full((10, 10), 255, dtype=np.uint8),
        )
        roi1.undo_stack.push(cmd1)

        # Paint on roi2
        roi2.mask[50:60, 50:60] = 128
        cmd2 = UndoCommand(
            roi2, (50, 60, 50, 60),
            np.zeros((10, 10), dtype=np.uint8),
            np.full((10, 10), 128, dtype=np.uint8),
        )
        roi2.undo_stack.push(cmd2)

        # Undo roi1 should not affect roi2
        roi1.undo_stack.undo()
        assert roi1.mask[15, 15] == 0
        assert roi2.mask[55, 55] == 128

    def test_max_size(self):
        roi = ROILayer("test", 100, 100)
        stack = UndoStack(max_size=3)

        for i in range(5):
            old = roi.mask[i:i + 1, 0:1].copy()
            roi.mask[i, 0] = 255
            new = roi.mask[i:i + 1, 0:1].copy()
            cmd = UndoCommand(roi, (i, i + 1, 0, 1), old, new)
            stack.push(cmd)

        # Only last 3 should be undoable
        count = 0
        while stack.undo():
            count += 1
        assert count == 3
