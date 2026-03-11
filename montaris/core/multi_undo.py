import numpy as np
from montaris.core.undo import _cmd_byte_size


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

    @property
    def byte_size(self):
        return sum(_cmd_byte_size(c) for c in self.commands)


class SnapshotUndoCommand:
    """Stores crop-based diffs for bulk operations across many layers."""

    def __init__(self, layer_snapshots):
        """
        Args:
            layer_snapshots: list of (roi_layer, old_mask_copy) tuples
        """
        self._entries = []
        for layer, old_mask in layer_snapshots:
            diff = old_mask != layer.mask
            if not diff.any():
                continue
            ys, xs = np.where(diff)
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            bbox = (y1, y2, x1, x2)
            self._entries.append((layer, bbox,
                                  old_mask[y1:y2, x1:x2].copy(),
                                  layer.mask[y1:y2, x1:x2].copy()))

    @property
    def roi_layer(self):
        return self._entries[0][0] if self._entries else None

    def undo(self):
        for layer, bbox, old_crop, _ in self._entries:
            y1, y2, x1, x2 = bbox
            layer.mask[y1:y2, x1:x2] = old_crop
            layer.invalidate_bbox()

    def redo(self):
        for layer, bbox, _, new_crop in self._entries:
            y1, y2, x1, x2 = bbox
            layer.mask[y1:y2, x1:x2] = new_crop
            layer.invalidate_bbox()

    @property
    def byte_size(self):
        return sum(e[2].nbytes + e[3].nbytes for e in self._entries)
