"""Tests for undo memory budget and optimized SnapshotUndoCommand."""
import numpy as np

from montaris.core.undo import UndoCommand, UndoStack, _cmd_byte_size
from montaris.core.multi_undo import CompoundUndoCommand, SnapshotUndoCommand
from montaris.layers import ROILayer


class TestMemoryBudgetEviction:
    def test_memory_budget_eviction(self):
        """Large commands get evicted when memory budget is exceeded."""
        stack = UndoStack(max_size=100, memory_budget=1024)  # 1 KB budget
        layer = ROILayer("test", 100, 100)

        # Each command stores 2 × (20×20) uint8 crops = 800 bytes
        for i in range(5):
            old = np.zeros((20, 20), dtype=np.uint8)
            new = np.ones((20, 20), dtype=np.uint8) * 255
            layer.mask[0:20, 0:20] = new
            stack.push(UndoCommand(layer, (0, 20, 0, 20), old, new))

        # Budget is 1024 bytes. Each cmd is 800 bytes.
        # After 5 pushes, stack should evict until within budget.
        # 1 cmd = 800 <= 1024, 2 cmds = 1600 > 1024 → only 1 should remain
        assert len(stack._stack) == 1
        assert stack._total_bytes <= 1024

    def test_memory_budget_keeps_one(self):
        """Even if a single command exceeds budget, stack keeps at least 1."""
        stack = UndoStack(max_size=100, memory_budget=100)  # tiny budget
        layer = ROILayer("test", 100, 100)

        # This command is 2 × (50×50) = 5000 bytes, far exceeds budget
        old = np.zeros((50, 50), dtype=np.uint8)
        new = np.ones((50, 50), dtype=np.uint8) * 255
        stack.push(UndoCommand(layer, (0, 50, 0, 50), old, new))

        assert len(stack._stack) == 1
        assert stack._total_bytes == 5000

    def test_total_bytes_tracking(self):
        """_total_bytes accurately tracks push/clear operations."""
        stack = UndoStack(max_size=100, memory_budget=1024 * 1024)
        layer = ROILayer("test", 10, 10)

        old = np.zeros((5, 5), dtype=np.uint8)
        new = np.ones((5, 5), dtype=np.uint8) * 255
        cmd_size = old.nbytes + new.nbytes  # 50 bytes

        stack.push(UndoCommand(layer, (0, 5, 0, 5), old, new))
        assert stack._total_bytes == cmd_size

        stack.push(UndoCommand(layer, (0, 5, 0, 5), old, new))
        assert stack._total_bytes == cmd_size * 2

        stack.clear()
        assert stack._total_bytes == 0

    def test_total_bytes_after_redo_discard(self):
        """Discarding redo tail decrements _total_bytes."""
        stack = UndoStack(max_size=100, memory_budget=1024 * 1024)
        layer = ROILayer("test", 10, 10)

        old = np.zeros((5, 5), dtype=np.uint8)
        new = np.ones((5, 5), dtype=np.uint8) * 255
        cmd_size = old.nbytes + new.nbytes

        stack.push(UndoCommand(layer, (0, 5, 0, 5), old, new))
        stack.push(UndoCommand(layer, (0, 5, 0, 5), old, new))
        assert stack._total_bytes == cmd_size * 2

        # Undo one, then push a new one — the redo tail should be discarded
        stack.undo()
        stack.push(UndoCommand(layer, (0, 5, 0, 5), old, new))
        assert stack._total_bytes == cmd_size * 2  # 2 commands, not 3


class TestByteSize:
    def test_undo_command_byte_size(self):
        layer = ROILayer("test", 10, 10)
        old = np.zeros((5, 5), dtype=np.uint8)
        new = np.ones((5, 5), dtype=np.uint8)
        cmd = UndoCommand(layer, (0, 5, 0, 5), old, new)
        assert cmd.byte_size == 50  # 25 + 25

    def test_compound_byte_size(self):
        layer = ROILayer("test", 10, 10)
        cmd1 = UndoCommand(layer, (0, 5, 0, 5),
                           np.zeros((5, 5), np.uint8), np.ones((5, 5), np.uint8))
        cmd2 = UndoCommand(layer, (0, 3, 0, 3),
                           np.zeros((3, 3), np.uint8), np.ones((3, 3), np.uint8))
        compound = CompoundUndoCommand([cmd1, cmd2])
        assert compound.byte_size == cmd1.byte_size + cmd2.byte_size

    def test_cmd_byte_size_helper(self):
        assert _cmd_byte_size("no byte_size attr") == 0


class TestSnapshotUndoCropBased:
    def test_snapshot_stores_only_changed_region(self):
        """SnapshotUndoCommand stores only the bbox of changed pixels."""
        layer = ROILayer("test", 100, 100)
        old_mask = layer.mask.copy()
        layer.mask[40:60, 40:60] = 255  # Change a 20×20 region

        cmd = SnapshotUndoCommand([(layer, old_mask)])

        assert len(cmd._entries) == 1
        _, bbox, old_crop, new_crop = cmd._entries[0]
        assert bbox == (40, 60, 40, 60)
        assert old_crop.shape == (20, 20)
        assert new_crop.shape == (20, 20)
        # Much smaller than full 100×100 mask
        assert cmd.byte_size == 20 * 20 * 2  # old + new crops

    def test_snapshot_undo_redo_correctness(self):
        """Crop-based SnapshotUndoCommand correctly restores on undo/redo."""
        layer = ROILayer("test", 50, 50)
        old_mask = layer.mask.copy()
        layer.mask[10:20, 10:20] = 128

        cmd = SnapshotUndoCommand([(layer, old_mask)])

        # Undo should restore to zeros
        cmd.undo()
        assert layer.mask[15, 15] == 0
        assert np.all(layer.mask == 0)

        # Redo should restore the painted region
        cmd.redo()
        assert layer.mask[15, 15] == 128
        assert np.all(layer.mask[10:20, 10:20] == 128)

    def test_snapshot_no_diff_empty_entries(self):
        """When old == new, entries list is empty."""
        layer = ROILayer("test", 10, 10)
        cmd = SnapshotUndoCommand([(layer, layer.mask.copy())])
        assert len(cmd._entries) == 0
        assert cmd.roi_layer is None
        assert cmd.byte_size == 0

    def test_snapshot_multi_layer(self):
        """SnapshotUndoCommand handles multiple layers correctly."""
        layer1 = ROILayer("a", 20, 20)
        layer2 = ROILayer("b", 20, 20)
        old1 = layer1.mask.copy()
        old2 = layer2.mask.copy()
        layer1.mask[5:10, 5:10] = 200
        layer2.mask[0:3, 0:3] = 100

        cmd = SnapshotUndoCommand([(layer1, old1), (layer2, old2)])
        assert len(cmd._entries) == 2

        cmd.undo()
        assert np.all(layer1.mask == 0)
        assert np.all(layer2.mask == 0)

        cmd.redo()
        assert layer1.mask[7, 7] == 200
        assert layer2.mask[1, 1] == 100
