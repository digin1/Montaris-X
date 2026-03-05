import numpy as np


class CompoundUndoCommand:
    """Groups multiple undo commands into a single undoable operation."""

    def __init__(self, commands):
        self.commands = list(commands)

    @property
    def roi_layer(self):
        """Return the roi_layer from the first sub-command, if any."""
        for cmd in self.commands:
            if hasattr(cmd, 'roi_layer'):
                return cmd.roi_layer
        return None

    def undo(self):
        for cmd in reversed(self.commands):
            cmd.undo()

    def redo(self):
        for cmd in self.commands:
            cmd.redo()


class SnapshotUndoCommand:
    """Stores full mask snapshots for bulk operations across many layers."""

    def __init__(self, layer_snapshots):
        """
        Args:
            layer_snapshots: list of (roi_layer, old_mask_copy) tuples
        """
        self._entries = []
        for layer, old_mask in layer_snapshots:
            self._entries.append((layer, old_mask.copy(), layer.mask.copy()))

    @property
    def roi_layer(self):
        return self._entries[0][0] if self._entries else None

    def undo(self):
        for layer, old_mask, _ in self._entries:
            layer.mask[:] = old_mask

    def redo(self):
        for layer, _, new_mask in self._entries:
            layer.mask[:] = new_mask
