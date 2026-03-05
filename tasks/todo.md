# Montaris-X Feature Port - COMPLETE

## Phase 0: Testing Infrastructure - DONE
- [x] tests/__init__.py
- [x] tests/conftest.py
- [x] tests/helpers.py
- [x] tests/test_existing.py (16 tests)
- [x] pyproject.toml test deps
- [x] All 16 tests pass

## Phase 1: Performance Foundation - Tiled Rendering - DONE
- [x] montaris/core/tile_pyramid.py (TilePyramid with 512px tiles, lazy build, multi-level)
- [x] montaris/core/tile_cache.py (LRU cache, 500 tiles max)
- [x] canvas.py: tiled rendering with _update_tiles(), invisible rect for scene bounds
- [x] layers.py: ImageLayer.tile_pyramid (lazy), ROILayer dirty regions
- [x] image_io.py: memory-mapped TIFF for files >100MB
- [x] tests/test_tile_pyramid.py (24 tests) + tests/test_canvas_performance.py (10 tests)

## Phase 2: Multi-Image & Display Modes - DONE
- [x] montaris/core/image_set.py (ImageSet with channel management)
- [x] montaris/core/display_modes.py (5 display modes + compositor)
- [x] montaris/core/adjustments.py (brightness/contrast/exposure/gamma, smart auto, quick boost)
- [x] montaris/core/shaders.py (GLSL shaders for future GPU acceleration)
- [x] montaris/widgets/display_panel.py (mode selector + channel toggles)
- [x] montaris/widgets/adjustments_panel.py (sliders + auto/boost/reset buttons)
- [x] Canvas flip/rotate via QTransform in app.py View menu
- [x] tests/test_display_modes.py (8 tests) + test_adjustments.py (11 tests) + test_image_set.py (10 tests)

## Phase 3: New Drawing Tools - DONE
- [x] montaris/tools/bucket_fill.py (BFS flood fill with tolerance)
- [x] montaris/tools/rectangle.py (drag rectangle with preview)
- [x] montaris/tools/circle.py (drag circle with preview)
- [x] montaris/tools/stamp.py (configurable square stamp)
- [x] Zoom compensation for brush/eraser
- [x] TOOL_REGISTRY in tools/__init__.py + data-driven tool_panel.py
- [x] tests/test_bucket_fill.py (6) + test_rectangle.py (5) + test_circle.py (5) + test_stamp.py (5)

## Phase 4: ROI Transform & Operations - DONE
- [x] montaris/core/roi_transform.py (affine transform, scale/rotate/translate matrices)
- [x] montaris/tools/transform.py (8 handles + rotation, apply_to_all flag)
- [x] montaris/tools/move.py (drag to reposition, apply_to_all flag)
- [x] montaris/core/roi_ops.py (fix_overlaps, compute_overlap_map, find_overlapping_pairs)
- [x] Per-ROI undo (Ctrl+Alt+Z) + auto overlap toggle in Edit menu
- [x] tests/test_transform.py (9) + test_fix_overlaps.py (8) + test_per_roi_undo.py (3)

## Phase 5: ROI Management - DONE
- [x] layers.py: merge_rois, duplicate_roi, reorder_roi, insert_roi, ROILayer.fill_mode
- [x] layer_panel.py: multi-select, drag-drop reorder, context menu, merge/duplicate buttons
- [x] canvas.py: outline-only rendering (mask_to_outline_qimage)
- [x] properties_panel.py: fill mode combo box (Solid/Outline)
- [x] tests/test_roi_management.py (18 tests)

## Phase 6: UI Enhancements - DONE
- [x] montaris/widgets/minimap.py (200x200 thumbnail + viewport rect + click to pan)
- [x] montaris/widgets/perf_monitor.py (FPS, render time, memory, tile cache stats)
- [x] montaris/widgets/debug_console.py (log handler + eval input)
- [x] app.py: dock toggles in View menu, all docks wired up
- [x] tests/test_ui_enhancements.py (11 tests)

## Phase 7: I/O Extensions - DONE
- [x] montaris/io/imagej_roi.py (read/write ImageJ binary ROI format)
- [x] montaris/io/instructions.py (load/apply JSON instructions)
- [x] app.py: Import/Export ImageJ ROI, batch export, load instructions menu items
- [x] tests/test_imagej_roi.py (6 tests) + test_batch_export.py (3 tests)

## Summary
- **171 tests, all passing**
- **24 new source files** created
- **18 test files** created
- **6 key files modified** (canvas.py, layers.py, app.py, tool_panel.py, properties_panel.py, layer_panel.py, image_io.py, tools/__init__.py)
