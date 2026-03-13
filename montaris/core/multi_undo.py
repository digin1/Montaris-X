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
        # Process in batches to limit peak memory while minimizing
        # decompress→compress round-trips.  Each batch holds at most
        # _BATCH decompressed masks (~68 MB each) before compressing.
        _BATCH = 8
        batch = []
        for cmd in reversed(self.commands):
            cmd.undo()
            layer = getattr(cmd, 'roi_layer', None)
            if layer is not None and hasattr(layer, 'compress'):
                batch.append(layer)
            if len(batch) >= _BATCH:
                for l in batch:
                    l.compress()
                batch.clear()
        for l in batch:
            l.compress()

    def redo(self):
        _BATCH = 8
        batch = []
        for cmd in self.commands:
            cmd.redo()
            layer = getattr(cmd, 'roi_layer', None)
            if layer is not None and hasattr(layer, 'compress'):
                batch.append(layer)
            if len(batch) >= _BATCH:
                for l in batch:
                    l.compress()
                batch.clear()
        for l in batch:
            l.compress()

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
        from montaris.core.rle import rle_encode
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
                                  rle_encode(old_mask[y1:y2, x1:x2].copy()),
                                  rle_encode(layer.mask[y1:y2, x1:x2].copy())))

    @property
    def roi_layer(self):
        return self._entries[0][0] if self._entries else None

    def undo(self):
        from montaris.core.rle import rle_decode
        for layer, bbox, old_rle, _ in self._entries:
            y1, y2, x1, x2 = bbox
            layer.mask[y1:y2, x1:x2] = rle_decode(*old_rle)
            layer.invalidate_bbox()
            layer.compress()

    def redo(self):
        from montaris.core.rle import rle_decode
        for layer, bbox, _, new_rle in self._entries:
            y1, y2, x1, x2 = bbox
            layer.mask[y1:y2, x1:x2] = rle_decode(*new_rle)
            layer.invalidate_bbox()
            layer.compress()

    @property
    def byte_size(self):
        return sum(len(e[2][0]) + len(e[3][0]) for e in self._entries)
