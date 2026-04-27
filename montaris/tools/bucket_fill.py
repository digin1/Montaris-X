import collections

import numpy as np
from PySide6.QtCore import Qt
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand
from montaris.core.components import label_connected_components
from montaris.core.roi_transform import get_mask_bbox


class BucketFillTool(BaseTool):
    name = "Bucket Fill"

    def __init__(self, app):
        super().__init__(app)
        self.tolerance = 0  # 0 = exact match, higher = more fill

    def on_press(self, pos, layer, canvas):
        if layer is None or not getattr(layer, 'is_roi', False):
            return

        x, y = int(pos.x()), int(pos.y())
        h, w = layer.shape

        if x < 0 or x >= w or y < 0 or y >= h:
            return

        target_value = layer.mask[y, x]
        # If pixel is painted, erase the connected component; otherwise
        # paint it.
        fill_value = 0 if target_value > 0 else 255

        # Compute the fill region WITHOUT mutating the mask, so we can
        # snapshot the bbox crop with truthful pre-fill values. The
        # earlier shortcut of reconstructing pre-fill values from
        # ``target_value`` corrupted undo on:
        # * the exact-match erase path (target>0 floods every nonzero
        #   pixel — values 5, 7, 12 all became target_value=5 on undo),
        # * the tolerance>0 path (a tolerant flood across mixed
        #   intensities lost the per-pixel original values).
        # Codex review HIGH.
        region = self._compute_fill_region(
            layer.mask, x, y, fill_value, target_value,
        )
        if region is None:
            canvas.refresh_active_overlay(layer)
            return
        bbox = get_mask_bbox(region)
        if bbox is None:
            canvas.refresh_active_overlay(layer)
            return
        y1, y2, x1, x2 = bbox

        # Truthful pre-fill snapshot of only the changed bbox — orders
        # of magnitude smaller than the old full-mask copy.
        old_crop = layer.mask[y1:y2, x1:x2].copy()
        layer.mask[region] = fill_value
        new_crop = layer.mask[y1:y2, x1:x2].copy()

        if not np.array_equal(old_crop, new_crop):
            cmd = UndoCommand(
                layer, (y1, y2, x1, x2), old_crop, new_crop,
            )
            self.app.undo_stack.push(cmd)
            canvas.refresh_dirty_region(layer, (y1, y2, x1, x2))
        else:
            # Tolerance>0 BFS can mark pixels already at ``fill_value``
            # as part of the region (they pass the tolerance test even
            # though writing fill_value is a no-op). Cheap to detect by
            # the array-equal check above; fall back to a regular full
            # refresh which is correct + cheap when nothing changed.
            canvas.refresh_active_overlay(layer)

    def _compute_fill_region(self, mask, start_x, start_y,
                              fill_value, target_value):
        """Compute the connected region the flood fill would change,
        WITHOUT mutating ``mask``. Returns the bool region array (or
        ``None`` when the fill is a no-op).

        Splitting find-region from apply-fill lets the caller snapshot
        the bbox crop with the *real* pre-fill values for undo. The
        earlier in-place fill clobbered original mask values before any
        snapshot could be taken, so undo restored ``target_value``
        instead of the original mixed intensities.
        """
        h, w = mask.shape
        if fill_value == target_value:
            return None

        if self.tolerance == 0:
            # Exact-match flood fill via single-pass connected-component
            # labelling — same algorithm napari uses for its labels-
            # layer bucket fill (`napari/layers/labels/labels.py::fill`).
            # The earlier ``scipy.ndimage.binary_propagation`` did
            # iterative dilation until fixed point, costing O(n ×
            # diameter) — long thin dendrites paid that diameter as
            # iteration count. ``ndi.label`` (via the project helper)
            # is one O(n) pass regardless of region geometry.
            #
            # 8-connectivity (3×3 structure) is intentional and diverges
            # from the helper's 4-conn default: it preserves the prior
            # ``binary_propagation`` behaviour and matches user
            # expectation that a diagonal chain of same-value pixels is
            # one connected fill region. ``core/components.py``'s other
            # callers (move/transform `get_component_at`) keep 4-conn —
            # the discrepancy is pre-existing.
            if target_value > 0:
                matches = mask > 0
            else:
                matches = mask == 0
            labelled, _num = label_connected_components(
                matches, structure=np.ones((3, 3), dtype=bool),
            )
            seed_id = labelled[start_y, start_x]
            if seed_id == 0:
                # Defensive — under the normal call path, ``target_value``
                # was just read from ``mask[start_y, start_x]`` so
                # ``matches[start_y, start_x]`` is always True and
                # ``seed_id`` non-zero. Guard anyway in case a future
                # caller threads a target_value that isn't the literal
                # value at the click point.
                return None
            return labelled == seed_id

        # Tolerance > 0: BFS over original ``mask`` values only. The
        # original code mutated mask in place during BFS but the
        # ``visited`` gate kept it correct; this version is byte-for-
        # byte identical because every pixel checked is at a position
        # we never write (visited gates writes too).
        visited = np.zeros((h, w), dtype=bool)
        queue = collections.deque()
        queue.append((start_x, start_y))
        visited[start_y, start_x] = True

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    if abs(int(mask[ny, nx]) - int(target_value)) <= self.tolerance:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
        return visited

    def cursor(self):
        return Qt.CrossCursor
