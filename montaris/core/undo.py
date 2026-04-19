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

    @staticmethod
    def _safe_decode(rle_pair, expected_shape):
        from montaris.core.rle import rle_decode
        import numpy as np
        data = rle_decode(*rle_pair)
        if data.shape != expected_shape:
            # RLE stored a (0,0) shape for an all-zeros region — rebuild
            return np.zeros(expected_shape, dtype=np.uint8)
        return data

    def undo(self):
        y1, y2, x1, x2 = self.bbox
        self.roi_layer.mask[y1:y2, x1:x2] = self._safe_decode(
            self._old_rle, (y2 - y1, x2 - x1))
        self.roi_layer.invalidate_bbox()

    def redo(self):
        y1, y2, x1, x2 = self.bbox
        self.roi_layer.mask[y1:y2, x1:x2] = self._safe_decode(
            self._new_rle, (y2 - y1, x2 - x1))
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


class AddROIUndoCommand:
    """Undo command for adding an ROI layer."""

    def __init__(self, layer_stack, roi_layer):
        self.layer_stack = layer_stack
        self.roi_layer = roi_layer

    def undo(self):
        try:
            idx = self.layer_stack.roi_layers.index(self.roi_layer)
            self.layer_stack.roi_layers.pop(idx)
        except ValueError:
            pass

    def redo(self):
        if self.roi_layer not in self.layer_stack.roi_layers:
            self.layer_stack.roi_layers.append(self.roi_layer)

    @property
    def byte_size(self):
        return 64


class RemoveROIUndoCommand:
    """Undo command for removing ROI layer(s). Restores them on undo.

    For :class:`VolumeROILayer` entries, captures a bit-packed snapshot of
    the ROI's voxels + its labels_meta so undo can rehydrate the shared
    ``labels_3d`` volume (the wrapper alone is useless without its voxels).
    Packed masks (via :func:`np.packbits`) give ~8× memory savings; a
    10M-voxel ROI costs ~1.25 MB per undo entry.
    """

    def __init__(self, layer_stack, entries):
        """entries: list of (index, roi_layer) tuples, sorted by index."""
        import copy
        import numpy as np
        self.layer_stack = layer_stack
        self._entries = entries  # [(original_index, roi_layer), ...]
        self.roi_layer = entries[0][1] if len(entries) == 1 else None
        # Voxel snapshots keyed by roi_layer id(): (lid, bbox, packed, shape, meta_copy)
        self._volume_snapshots = {}
        for _, roi in entries:
            if not getattr(roi, 'is_volume', False):
                continue
            doc = getattr(roi, '_doc', None)
            lid = getattr(roi, '_label_id', None)
            if doc is None or lid is None or doc.labels_3d is None:
                continue
            mask = (doc.labels_3d == lid)
            if not mask.any():
                # Empty 3D ROI — still record meta so undo restores wrapper cleanly.
                meta = copy.deepcopy(doc.labels_meta.get(lid, {}))
                self._volume_snapshots[id(roi)] = (doc, lid, None, None, None, meta)
                continue
            zs, ys, xs = np.where(mask)
            bbox = (int(zs.min()), int(zs.max()) + 1,
                    int(ys.min()), int(ys.max()) + 1,
                    int(xs.min()), int(xs.max()) + 1)
            z1, z2, y1, y2, x1, x2 = bbox
            crop = mask[z1:z2, y1:y2, x1:x2]
            packed = np.packbits(crop.reshape(-1).astype(np.uint8))
            meta = copy.deepcopy(doc.labels_meta.get(lid, {}))
            self._volume_snapshots[id(roi)] = (doc, lid, bbox, packed, crop.shape, meta)

    def _restore_volume(self, roi):
        import numpy as np
        snap = self._volume_snapshots.get(id(roi))
        if snap is None:
            return
        doc, lid, bbox, packed, shape, meta = snap
        if doc.labels_3d is None:
            return
        # Restore meta first (VolumeROILayer methods read from labels_meta).
        if meta:
            doc.labels_meta[lid] = dict(meta)
        if bbox is None:
            return
        # Promote dtype if the restored lid would overflow current dtype
        if lid > np.iinfo(doc.labels_3d.dtype).max:
            doc.promote_labels_dtype(np.uint16)
        z1, z2, y1, y2, x1, x2 = bbox
        n = shape[0] * shape[1] * shape[2]
        crop = np.unpackbits(packed, count=n).astype(bool).reshape(shape)
        view = doc.labels_3d[z1:z2, y1:y2, x1:x2]
        view[crop] = lid

    def undo(self):
        for idx, roi in self._entries:
            if getattr(roi, 'is_volume', False):
                self._restore_volume(roi)
            pos = min(idx, len(self.layer_stack.roi_layers))
            if roi not in self.layer_stack.roi_layers:
                self.layer_stack.roi_layers.insert(pos, roi)

    def redo(self):
        for _, roi in reversed(self._entries):
            if getattr(roi, 'is_volume', False):
                doc = getattr(roi, '_doc', None)
                lid = getattr(roi, '_label_id', None)
                if doc is not None and lid is not None:
                    doc.release_label_id(int(lid))
            try:
                self.layer_stack.roi_layers.remove(roi)
            except ValueError:
                pass

    @property
    def byte_size(self):
        base = sum(getattr(r, '_raw_bytes', 64) for _, r in self._entries)
        for snap in self._volume_snapshots.values():
            packed = snap[3]
            if packed is not None:
                base += int(packed.nbytes)
        return base


class VolumeStrokeUndoCommand:
    """Undo entry for a single 3D paint or erase stroke.

    Snapshots ``labels_3d[bbox]`` before and after the stroke. The bbox is
    the stroke-wide accumulator (NOT the per-tick ``_dirty_bbox`` that the
    refresh throttle clears every frame). Patches are stored as raw
    labels-dtype crops — for a 100k-voxel stroke at uint16 that's 200 KB.
    """

    def __init__(self, doc, bbox, before_patch, after_patch):
        import numpy as np
        self._doc = doc
        self._bbox = bbox  # (z0, z1, y0, y1, x0, x1)
        self._before = np.ascontiguousarray(before_patch)
        self._after = np.ascontiguousarray(after_patch)
        self.roi_layer = None  # not mergeable into AddROI

    def _apply(self, patch):
        import numpy as np
        lab = self._doc.labels_3d
        if lab is None:
            return
        z0, z1, y0, y1, x0, x1 = self._bbox
        # Promote dtype if the stored patch contains ids that overflow the
        # current labels_3d dtype — strokes can be undone after later
        # ROIs pushed the labels volume to a wider dtype.
        need = np.iinfo(patch.dtype).max
        if need > np.iinfo(lab.dtype).max:
            self._doc.promote_labels_dtype(patch.dtype)
            lab = self._doc.labels_3d
        lab[z0:z1, y0:y1, x0:x1] = patch

    def undo(self):
        self._apply(self._before)

    def redo(self):
        self._apply(self._after)

    @property
    def byte_size(self):
        return int(self._before.nbytes + self._after.nbytes)


class VolumeFillUndoCommand:
    """Undo entry for a napari-style 3D fill (connected-component relabel).

    Stores the bit-packed component mask + its bbox; undo/redo swap between
    ``old_label`` and ``new_label`` on exactly those voxels. Much cheaper
    than snapshotting the bbox's full integer crop because the fill affects
    only one label value.
    """

    def __init__(self, doc, bbox, mask_crop, old_label, new_label):
        import numpy as np
        self._doc = doc
        self._bbox = bbox  # (z0, z1, y0, y1, x0, x1)
        self._shape = tuple(mask_crop.shape)
        self._packed = np.packbits(mask_crop.reshape(-1).astype(np.uint8))
        self._old = int(old_label)
        self._new = int(new_label)
        self.roi_layer = None

    def _unpack(self):
        import numpy as np
        n = self._shape[0] * self._shape[1] * self._shape[2]
        return np.unpackbits(self._packed, count=n).astype(bool).reshape(self._shape)

    def _apply(self, value):
        import numpy as np
        lab = self._doc.labels_3d
        if lab is None:
            return
        if value > np.iinfo(lab.dtype).max:
            self._doc.promote_labels_dtype(np.uint16)
            lab = self._doc.labels_3d
        z0, z1, y0, y1, x0, x1 = self._bbox
        mask = self._unpack()
        view = lab[z0:z1, y0:y1, x0:x1]
        view[mask] = value

    def undo(self):
        self._apply(self._old)

    def redo(self):
        self._apply(self._new)

    @property
    def byte_size(self):
        return int(self._packed.nbytes)


class AddVolumeROIUndoCommand:
    """Undo entry for the creation of a fresh 3D ROI (id reservation + wrapper).

    Unlike :class:`AddROIUndoCommand`, this one also snapshots and restores
    ``labels_meta[lid]`` so undo → redo round-trips the full ROI identity
    (name, color, opacity, fill_mode).
    """

    def __init__(self, layer_stack, doc, lid, wrapper):
        import copy
        self.layer_stack = layer_stack
        self._doc = doc
        self._lid = int(lid)
        self.roi_layer = wrapper
        self._meta = copy.deepcopy(doc.labels_meta.get(int(lid), {}))

    def undo(self):
        try:
            self.layer_stack.roi_layers.remove(self.roi_layer)
        except ValueError:
            pass
        self._doc.labels_meta.pop(self._lid, None)

    def redo(self):
        self._doc.labels_meta[self._lid] = dict(self._meta)
        if self.roi_layer not in self.layer_stack.roi_layers:
            self.layer_stack.roi_layers.append(self.roi_layer)

    @property
    def byte_size(self):
        return 128


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

        # Merge first mask edit into the AddROIUndoCommand that created the ROI,
        # so a single Ctrl+Z removes both the edit and the creation.
        # Only merge mask-editing commands, not structural ones (Remove, Offset, etc.)
        from montaris.core.multi_undo import CompoundUndoCommand
        _MERGEABLE = (UndoCommand, FlattenUndoCommand, CompoundUndoCommand)
        if (self._stack
                and isinstance(self._stack[-1], AddROIUndoCommand)
                and isinstance(command, _MERGEABLE)
                and getattr(command, 'roi_layer', None) is not None
                and command.roi_layer is self._stack[-1].roi_layer):
            add_cmd = self._stack.pop()
            self._total_bytes -= _cmd_byte_size(add_cmd)
            command = CompoundUndoCommand([add_cmd, command])

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

    def purge_for_doc(self, doc):
        """Drop commands that would replay patches into ``doc.labels_3d``.

        Call after the doc's labels volume is replaced wholesale (label
        import, session restore, dtype change into a fresh array). Those
        commands hold a reference to ``doc`` but their patches are sized
        for the *previous* labels array; replaying them would either
        crash on a shape mismatch or silently corrupt the new volume.
        Compound commands are dropped if any sub-command targets ``doc``.
        """
        from montaris.core.multi_undo import CompoundUndoCommand

        def _touches(cmd):
            if isinstance(cmd, CompoundUndoCommand):
                return any(_touches(c) for c in cmd.commands)
            return getattr(cmd, '_doc', None) is doc

        kept = []
        new_bytes = 0
        removed_before_index = 0
        for i, cmd in enumerate(self._stack):
            if _touches(cmd):
                if i <= self._index:
                    removed_before_index += 1
                continue
            kept.append(cmd)
            new_bytes += _cmd_byte_size(cmd)
        self._stack = kept
        self._total_bytes = new_bytes
        self._index -= removed_before_index
        if self._index >= len(self._stack):
            self._index = len(self._stack) - 1
        if self._index < -1:
            self._index = -1

    @property
    def can_undo(self):
        return self._index >= 0

    @property
    def can_redo(self):
        return self._index < len(self._stack) - 1
