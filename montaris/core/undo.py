class UndoCommand:
    def __init__(self, roi_layer, bbox, old_data, new_data):
        self.roi_layer = roi_layer
        self.bbox = bbox  # (y1, y2, x1, x2)
        # RLE-compress binary mask crops for ~10x memory savings
        from montaris.core.rle import rle_encode
        old_copy = old_data.copy()
        new_copy = new_data.copy()
        self._old_rle = rle_encode(old_copy)
        self._new_rle = rle_encode(new_copy)
        # Cache raw size for byte_size reporting
        self._raw_bytes = old_copy.nbytes + new_copy.nbytes

    def _decode_old(self):
        from montaris.core.rle import rle_decode
        return rle_decode(*self._old_rle)

    def _decode_new(self):
        from montaris.core.rle import rle_decode
        return rle_decode(*self._new_rle)

    def undo(self):
        y1, y2, x1, x2 = self.bbox
        self.roi_layer.mask[y1:y2, x1:x2] = self._decode_old()
        self.roi_layer.invalidate_bbox()

    def redo(self):
        y1, y2, x1, x2 = self.bbox
        self.roi_layer.mask[y1:y2, x1:x2] = self._decode_new()
        self.roi_layer.invalidate_bbox()

    @property
    def byte_size(self):
        return len(self._old_rle[0]) + len(self._new_rle[0])


class OffsetUndoCommand:
    """Lightweight undo for layer offset changes (no mask data stored)."""

    def __init__(self, roi_layer, old_offset, new_offset):
        self.roi_layer = roi_layer
        self.old_offset = old_offset  # (offset_x, offset_y)
        self.new_offset = new_offset  # (offset_x, offset_y)

    def undo(self):
        self.roi_layer.offset_x, self.roi_layer.offset_y = self.old_offset
        self.roi_layer.invalidate_bbox()

    def redo(self):
        self.roi_layer.offset_x, self.roi_layer.offset_y = self.new_offset
        self.roi_layer.invalidate_bbox()

    @property
    def byte_size(self):
        return 64


class FlattenUndoCommand:
    """Undo command for offset flatten: restores pre-flatten mask + offset."""

    def __init__(self, entries):
        """entries: list of (roi_layer, old_crop, old_bbox, old_offset)"""
        self._entries = entries
        self.roi_layer = entries[0][0] if len(entries) == 1 else None

    def undo(self):
        for roi, old_crop, old_bbox, old_offset in self._entries:
            roi.mask[:] = 0
            if old_crop is not None and old_bbox is not None:
                y1, y2, x1, x2 = old_bbox
                roi.mask[y1:y2, x1:x2] = old_crop
            roi.offset_x, roi.offset_y = old_offset
            roi.invalidate_bbox()

    def redo(self):
        for roi, _, _, _ in self._entries:
            roi.flatten_offset()
            roi.invalidate_bbox()

    @property
    def byte_size(self):
        return sum(c.nbytes for _, c, _, _ in self._entries if c is not None)


def _cmd_byte_size(cmd):
    """Return the byte size of a command, or 0 if it doesn't report one."""
    if hasattr(cmd, 'byte_size'):
        return cmd.byte_size
    return 0


class UndoStack:
    def __init__(self, max_size=100, memory_budget=256 * 1024 * 1024):
        self._stack = []
        self._index = -1
        self._max_size = max_size
        self._memory_budget = memory_budget
        self._total_bytes = 0

    def push(self, command):
        # Discard redo tail
        discarded = self._stack[self._index + 1:]
        for cmd in discarded:
            self._total_bytes -= _cmd_byte_size(cmd)
        self._stack = self._stack[:self._index + 1]

        self._stack.append(command)
        self._total_bytes += _cmd_byte_size(command)
        self._index = len(self._stack) - 1

        # Evict by count
        while len(self._stack) > self._max_size:
            evicted = self._stack.pop(0)
            self._total_bytes -= _cmd_byte_size(evicted)
            self._index -= 1

        # Evict by memory (keep at least 1)
        while self._total_bytes > self._memory_budget and len(self._stack) > 1:
            evicted = self._stack.pop(0)
            self._total_bytes -= _cmd_byte_size(evicted)
            self._index -= 1

    def undo(self):
        if self._index >= 0:
            cmd = self._stack[self._index]
            cmd.undo()
            self._index -= 1
            return cmd
        return None

    def redo(self):
        if self._index < len(self._stack) - 1:
            self._index += 1
            cmd = self._stack[self._index]
            cmd.redo()
            return cmd
        return None

    def clear(self):
        self._stack.clear()
        self._index = -1
        self._total_bytes = 0

    @property
    def can_undo(self):
        return self._index >= 0

    @property
    def can_redo(self):
        return self._index < len(self._stack) - 1
