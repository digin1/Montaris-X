import numpy as np
import pytest
import zipfile
from montaris.io.imagej_roi import (
    read_imagej_roi, write_imagej_roi,
    mask_to_imagej_roi, imagej_roi_to_mask,
    write_imagej_roi_bytes, scale_roi_dict,
    ROI_RECT, ROI_FREEHAND,
)


class TestImageJRect:
    def test_write_read_rect(self, tmp_path):
        roi = {
            'type': ROI_RECT,
            'top': 10,
            'left': 20,
            'bottom': 50,
            'right': 80,
        }
        path = tmp_path / "test.roi"
        write_imagej_roi(roi, str(path))
        loaded = read_imagej_roi(str(path))
        assert loaded['type'] == ROI_RECT
        assert loaded['top'] == 10
        assert loaded['left'] == 20
        assert loaded['bottom'] == 50
        assert loaded['right'] == 80

    def test_rect_to_mask(self):
        roi = {'type': ROI_RECT, 'top': 10, 'left': 20, 'bottom': 50, 'right': 80}
        mask = imagej_roi_to_mask(roi, 100, 100)
        assert mask.shape == (100, 100)
        assert mask[10:50, 20:80].all()
        assert mask[0:10, :].sum() == 0


class TestImageJPolygon:
    def test_write_read_polygon(self, tmp_path):
        roi = {
            'type': ROI_FREEHAND,
            'top': 10,
            'left': 10,
            'bottom': 50,
            'right': 50,
            'x_coords': np.array([10, 50, 50, 10], dtype=np.int32),
            'y_coords': np.array([10, 10, 50, 50], dtype=np.int32),
        }
        path = tmp_path / "test.roi"
        write_imagej_roi(roi, str(path))
        loaded = read_imagej_roi(str(path))
        assert loaded['type'] == ROI_FREEHAND
        assert len(loaded['x_coords']) == 4


class TestMaskConversion:
    def test_mask_to_roi_roundtrip(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:70] = 255
        roi_dict = mask_to_imagej_roi(mask, "test_roi")
        assert roi_dict is not None
        assert roi_dict['type'] == ROI_FREEHAND
        assert roi_dict['name'] == "test_roi"
        assert roi_dict['top'] == 20
        assert roi_dict['left'] == 30

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = mask_to_imagej_roi(mask)
        assert result is None

    def test_polygon_to_mask(self):
        roi = {
            'type': ROI_FREEHAND,
            'top': 10,
            'left': 10,
            'bottom': 50,
            'right': 50,
            'x_coords': np.array([10, 50, 50, 10], dtype=np.int32),
            'y_coords': np.array([10, 10, 50, 50], dtype=np.int32),
        }
        mask = imagej_roi_to_mask(roi, 100, 100)
        assert mask.shape == (100, 100)
        # Interior should be filled
        assert mask[30, 30] == 255

    def test_mask_to_roi_with_bbox(self):
        """bbox parameter produces correct coordinates in full-mask space."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[80:120, 50:90] = 255
        bbox = (80, 120, 50, 90)  # top, bottom, left, right
        roi_dict = mask_to_imagej_roi(mask, "bbox_test", bbox=bbox)
        assert roi_dict is not None
        assert roi_dict['top'] == 80
        assert roi_dict['left'] == 50
        assert roi_dict['bottom'] == 120
        assert roi_dict['right'] == 90
        # Verify coordinates are in full-mask space (not sub-mask local)
        # find_contours traces at 0.5 boundary, so coords may be ±1 of bbox edges
        if roi_dict['x_coords'] is not None:
            assert roi_dict['x_coords'].min() >= 49
            assert roi_dict['y_coords'].min() >= 79

    def test_mask_to_roi_bbox_matches_auto(self):
        """bbox result matches auto-detect result."""
        mask = np.zeros((150, 150), dtype=np.uint8)
        mask[30:70, 40:100] = 255
        auto = mask_to_imagej_roi(mask, "auto")
        bbox_result = mask_to_imagej_roi(mask, "bbox", bbox=(30, 70, 40, 100))
        assert auto is not None and bbox_result is not None
        assert auto['top'] == bbox_result['top']
        assert auto['left'] == bbox_result['left']
        assert auto['bottom'] == bbox_result['bottom']
        assert auto['right'] == bbox_result['right']
        # Coordinates should match
        np.testing.assert_array_equal(auto['x_coords'], bbox_result['x_coords'])
        np.testing.assert_array_equal(auto['y_coords'], bbox_result['y_coords'])

    def test_mask_to_roi_bbox_empty_submask(self):
        """bbox pointing to empty region returns None."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = mask_to_imagej_roi(mask, "empty", bbox=(10, 20, 10, 20))
        assert result is None


class TestScaleRoiDict:
    def test_scale_roi_dict_freehand(self):
        """Halving coordinates scales all values correctly."""
        roi = {
            'type': ROI_FREEHAND,
            'top': 100, 'left': 200, 'bottom': 300, 'right': 400,
            'x_coords': np.array([200, 400, 400, 200], dtype=np.int32),
            'y_coords': np.array([100, 100, 300, 300], dtype=np.int32),
            'paths': None,
            'name': 'test',
        }
        scaled = scale_roi_dict(roi, 0.5, 0.5)
        assert scaled['top'] == 50
        assert scaled['left'] == 100
        assert scaled['bottom'] == 150
        assert scaled['right'] == 200
        np.testing.assert_array_equal(scaled['x_coords'], [100, 200, 200, 100])
        np.testing.assert_array_equal(scaled['y_coords'], [50, 50, 150, 150])

    def test_scale_roi_dict_composite(self):
        """Scaling composite ROI paths works correctly."""
        roi = {
            'type': ROI_RECT,
            'top': 10, 'left': 20, 'bottom': 50, 'right': 80,
            'x_coords': None, 'y_coords': None,
            'paths': [[(20.0, 10.0), (80.0, 10.0), (80.0, 50.0), (20.0, 50.0)]],
            'name': 'composite',
        }
        scaled = scale_roi_dict(roi, 2.0, 2.0)
        assert scaled['top'] == 20
        assert scaled['left'] == 40
        assert scaled['bottom'] == 100
        assert scaled['right'] == 160
        assert scaled['paths'] == [[(40.0, 20.0), (160.0, 20.0), (160.0, 100.0), (40.0, 100.0)]]

    def test_scale_roi_dict_preserves_original(self):
        """Original dict is not mutated by scale_roi_dict."""
        roi = {
            'type': ROI_FREEHAND,
            'top': 100, 'left': 200, 'bottom': 300, 'right': 400,
            'x_coords': np.array([200, 400], dtype=np.int32),
            'y_coords': np.array([100, 300], dtype=np.int32),
            'paths': None,
            'name': 'orig',
        }
        original_x = roi['x_coords'].copy()
        scale_roi_dict(roi, 0.5, 0.5)
        assert roi['top'] == 100
        assert roi['left'] == 200
        np.testing.assert_array_equal(roi['x_coords'], original_x)


class TestPixelLevelCorrectness:
    """Pixel-by-pixel verification of scale-down import and upscale export paths."""

    def test_rect_import_scale_down_pixels(self):
        """Rect ROI at original res → scale 1/2 → rasterize matches manual downsample."""
        # Original: 200x200 image, rect filling [40:120, 60:160]
        orig_roi = {'type': ROI_RECT, 'top': 40, 'left': 60, 'bottom': 120, 'right': 160}
        orig_mask = imagej_roi_to_mask(orig_roi, 200, 200)

        # Manual downsample: take every other pixel
        ds = 2
        expected = orig_mask[::ds, ::ds]  # 100x100

        # Import path: scale coords by 1/ds, rasterize at downsampled size
        scaled_roi = scale_roi_dict(orig_roi, 1.0 / ds, 1.0 / ds)
        got = imagej_roi_to_mask(scaled_roi, 100, 100)

        np.testing.assert_array_equal(got, expected)

    def test_freehand_import_scale_down_pixels(self):
        """Freehand ROI at original res → scale 1/2 → rasterize.

        Compare against reference: rasterize at full res, then downsample with OR
        (any nonzero in 2x2 block → 255). Freehand rasterization isn't pixel-perfect
        after int truncation, so we check the interior region strictly and allow
        ≤2px boundary error.
        """
        # Original: 200x200 image, freehand square [40:160, 40:160]
        orig_roi = {
            'type': ROI_FREEHAND,
            'top': 40, 'left': 40, 'bottom': 160, 'right': 160,
            'x_coords': np.array([40, 160, 160, 40], dtype=np.int32),
            'y_coords': np.array([40, 40, 160, 160], dtype=np.int32),
            'paths': None, 'name': 'fh',
        }
        orig_mask = imagej_roi_to_mask(orig_roi, 200, 200)

        ds = 2
        scaled_roi = scale_roi_dict(orig_roi, 1.0 / ds, 1.0 / ds)
        got = imagej_roi_to_mask(scaled_roi, 100, 100)

        # Interior (well inside boundary) must be identical
        assert got[25:75, 25:75].all(), "Interior pixels must all be 255"
        # Exterior (well outside boundary) must be zero
        assert got[0:15, :].sum() == 0, "Top exterior must be empty"
        assert got[:, 0:15].sum() == 0, "Left exterior must be empty"

        # Overall: compare against block-max downsample, allow ≤2px boundary diff
        ref_ds = np.zeros((100, 100), dtype=np.uint8)
        for dy in range(ds):
            for dx in range(ds):
                ref_ds |= orig_mask[dy::ds, dx::ds]
        diff = np.abs(got.astype(int) - ref_ds.astype(int))
        diff_pixels = np.count_nonzero(diff)
        boundary_pixels = 4 * (160 - 40) // ds  # perimeter at downsampled scale
        assert diff_pixels <= boundary_pixels, (
            f"Too many differing pixels: {diff_pixels} > {boundary_pixels} (boundary)"
        )

    def test_rect_export_upscale_pixel_perfect(self):
        """Downsampled rect mask → np.repeat upscale → matches original exactly."""
        ds = 2
        orig_w, orig_h = 200, 200
        ds_w, ds_h = orig_w // ds, orig_h // ds

        # Original mask
        orig_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        orig_mask[40:120, 60:160] = 255

        # Downsampled mask (take every ds-th pixel)
        ds_mask = orig_mask[::ds, ::ds]

        # Upscale path (what _upscale_mask_if_needed does)
        upscaled = np.repeat(np.repeat(ds_mask, ds, axis=0), ds, axis=1)
        upscaled = upscaled[:orig_h, :orig_w]

        np.testing.assert_array_equal(upscaled, orig_mask)

    def test_full_roundtrip_rect_import_then_export(self):
        """Full roundtrip: original ROI → scale down → rasterize → upscale → compare.

        For axis-aligned rects this should be pixel-perfect.
        """
        ds = 2
        orig_w, orig_h = 200, 200
        ds_w, ds_h = orig_w // ds, orig_h // ds

        # Step 1: Create original-resolution rect ROI
        orig_roi = {'type': ROI_RECT, 'top': 40, 'left': 60, 'bottom': 120, 'right': 160}
        orig_mask = imagej_roi_to_mask(orig_roi, orig_w, orig_h)

        # Step 2: Import path — scale coords by 1/ds, rasterize at ds size
        scaled_roi = scale_roi_dict(orig_roi, 1.0 / ds, 1.0 / ds)
        ds_mask = imagej_roi_to_mask(scaled_roi, ds_w, ds_h)

        # Step 3: Export path — upscale back
        upscaled = np.repeat(np.repeat(ds_mask, ds, axis=0), ds, axis=1)
        upscaled = upscaled[:orig_h, :orig_w]

        # Pixel-perfect for axis-aligned rect
        np.testing.assert_array_equal(upscaled, orig_mask)

    def test_full_roundtrip_freehand_import_then_export(self):
        """Full roundtrip for freehand: original → scale down → rasterize → upscale.

        Freehand contours lose sub-pixel precision on int truncation, so we verify:
        - Interior pixels are preserved
        - Exterior pixels stay zero
        - Total pixel count is within 5% of original
        """
        ds = 2
        orig_w, orig_h = 200, 200
        ds_w, ds_h = orig_w // ds, orig_h // ds

        # Create freehand ROI (a diamond shape for interesting boundary)
        cx, cy = 100, 100
        r = 40
        n_pts = 60
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        x_coords = np.round(cx + r * np.cos(angles)).astype(np.int32)
        y_coords = np.round(cy + r * np.sin(angles)).astype(np.int32)
        orig_roi = {
            'type': ROI_FREEHAND,
            'top': int(y_coords.min()), 'left': int(x_coords.min()),
            'bottom': int(y_coords.max()) + 1, 'right': int(x_coords.max()) + 1,
            'x_coords': x_coords, 'y_coords': y_coords,
            'paths': None, 'name': 'circle',
        }
        orig_mask = imagej_roi_to_mask(orig_roi, orig_w, orig_h)
        orig_count = np.count_nonzero(orig_mask)

        # Import: scale down
        scaled_roi = scale_roi_dict(orig_roi, 1.0 / ds, 1.0 / ds)
        ds_mask = imagej_roi_to_mask(scaled_roi, ds_w, ds_h)

        # Export: upscale
        upscaled = np.repeat(np.repeat(ds_mask, ds, axis=0), ds, axis=1)
        upscaled = upscaled[:orig_h, :orig_w]
        up_count = np.count_nonzero(upscaled)

        # Interior (radius - margin) must be filled in both
        margin = 8  # pixels of boundary tolerance
        for y in range(orig_h):
            for x in range(orig_w):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist < r - margin:
                    assert upscaled[y, x] == 255, f"Interior pixel ({x},{y}) lost"
                elif dist > r + margin:
                    assert upscaled[y, x] == 0, f"Exterior pixel ({x},{y}) gained"

        # Total area within 5%
        assert abs(up_count - orig_count) / orig_count < 0.05, (
            f"Area diverged: orig={orig_count}, upscaled={up_count} "
            f"({abs(up_count - orig_count) / orig_count:.1%})"
        )

    def test_imagej_binary_roundtrip_with_scaling(self, tmp_path):
        """Write ROI to binary → read back → scale down → rasterize → verify pixels.

        Tests the full I/O + scaling pipeline end-to-end at the pixel level.
        """
        ds = 2
        orig_w, orig_h = 200, 200
        ds_w, ds_h = orig_w // ds, orig_h // ds

        # Create and write a freehand ROI at original resolution
        orig_roi = {
            'type': ROI_FREEHAND,
            'top': 30, 'left': 50, 'bottom': 100, 'right': 150,
            'x_coords': np.array([50, 150, 150, 50], dtype=np.int32),
            'y_coords': np.array([30, 30, 100, 100], dtype=np.int32),
            'paths': None, 'name': 'binary_test',
        }
        roi_path = tmp_path / "test.roi"
        write_imagej_roi(orig_roi, str(roi_path))

        # Read back (simulates import_imagej_roi)
        loaded = read_imagej_roi(str(roi_path))

        # Scale (what import_imagej_roi now does)
        scaled = scale_roi_dict(loaded, 1.0 / ds, 1.0 / ds)
        ds_mask = imagej_roi_to_mask(scaled, ds_w, ds_h)

        # Reference: rasterize original, downsample
        orig_mask = imagej_roi_to_mask(orig_roi, orig_w, orig_h)
        ref_ds = orig_mask[::ds, ::ds]

        # For the well-interior region, pixel values must match exactly
        # (boundary can differ by ±1px due to int truncation)
        interior = ds_mask[20:45, 30:70]  # well inside [15:50, 25:75]
        ref_interior = ref_ds[20:45, 30:70]
        np.testing.assert_array_equal(interior, ref_interior)

        # Overall shape: diff should be ≤ boundary pixels
        diff_count = np.count_nonzero(ds_mask != ref_ds)
        perimeter_estimate = 2 * ((100 - 30) + (150 - 50)) // ds
        assert diff_count <= perimeter_estimate, (
            f"Too many differing pixels: {diff_count} > {perimeter_estimate}"
        )

    def test_composite_roi_scale_down_pixels(self):
        """Composite ROI (paths) at original res → scale 1/3 → rasterize → verify pixels."""
        ds = 3
        orig_w, orig_h = 300, 300
        ds_w, ds_h = orig_w // ds, orig_h // ds

        # Composite with one path (a square)
        orig_roi = {
            'type': ROI_RECT,
            'top': 60, 'left': 90, 'bottom': 240, 'right': 210,
            'x_coords': None, 'y_coords': None,
            'paths': [[(90.0, 60.0), (210.0, 60.0), (210.0, 240.0), (90.0, 240.0)]],
            'name': 'comp',
        }
        orig_mask = imagej_roi_to_mask(orig_roi, orig_w, orig_h)

        scaled_roi = scale_roi_dict(orig_roi, 1.0 / ds, 1.0 / ds)
        ds_mask = imagej_roi_to_mask(scaled_roi, ds_w, ds_h)

        # Interior must be filled
        assert ds_mask[25:75, 35:65].all(), "Interior of scaled composite must be filled"
        # Exterior must be empty
        assert ds_mask[0:15, :].sum() == 0, "Top exterior must be empty"

        # Upscale and compare total area
        upscaled = np.repeat(np.repeat(ds_mask, ds, axis=0), ds, axis=1)[:orig_h, :orig_w]
        orig_count = np.count_nonzero(orig_mask)
        up_count = np.count_nonzero(upscaled)
        assert abs(up_count - orig_count) / max(orig_count, 1) < 0.05, (
            f"Area diverged: orig={orig_count}, upscaled={up_count}"
        )


class TestExportReimportZipRoundtrip:
    """Export masks to ZIP at upscaled resolution, reimport, compare pixels to originals."""

    def _make_zip(self, tmp_path, masks, names, ds):
        """Simulate the export_all_rois_zip upscale path:
        upscale each mask → mask_to_imagej_roi → write_imagej_roi_bytes → ZIP.
        """
        zip_path = tmp_path / "export_roundtrip.zip"
        with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
            for mask, name in zip(masks, names):
                # Upscale (what _get_export_mask does)
                upscaled = np.repeat(np.repeat(mask, ds, axis=0), ds, axis=1)
                # Get bbox for upscaled mask
                ys, xs = np.where(upscaled > 0)
                if len(ys) == 0:
                    continue
                bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
                roi_dict = mask_to_imagej_roi(upscaled, name, bbox=bbox)
                if roi_dict:
                    safe = name.replace("/", "_").replace("\\", "_")
                    zf.writestr(f"{safe}.roi", write_imagej_roi_bytes(roi_dict))
        return zip_path

    def _import_zip(self, zip_path, img_w, img_h):
        """Simulate import_roi_zip: read each .roi, rasterize at (img_w, img_h)."""
        masks = {}
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            for name in zf.namelist():
                if name.lower().endswith('.roi'):
                    data = zf.read(name)
                    roi_dict = read_imagej_roi(data)
                    mask = imagej_roi_to_mask(roi_dict, img_w, img_h)
                    base = name.replace('.roi', '')
                    masks[base] = mask
        return masks

    def _assert_boundary_close(self, got, expected, label=""):
        """Assert masks match except for ≤1px boundary ring from contour tracing.

        mask_to_imagej_roi uses find_contours at 0.5 level which traces outside
        the filled region. On re-rasterization, the polygon fill gains ~1 pixel
        ring at the boundary. This is inherent to the ImageJ ROI format.

        We verify:
        - Interior (eroded by 2px) is pixel-perfect
        - Exterior (dilated by 2px out) is pixel-perfect
        - Total diff ≤ perimeter of the shape
        - The reimported mask is a superset (gains only, no losses)
        """
        from scipy.ndimage import binary_erosion, binary_dilation

        assert got.shape == expected.shape, f"{label}: shape mismatch {got.shape} vs {expected.shape}"

        # Interior: erode expected by 2px — must all be filled in got
        struct = np.ones((3, 3), dtype=bool)
        interior = binary_erosion(expected > 0, structure=struct, iterations=2)
        assert got[interior].all(), f"{label}: interior pixels lost after roundtrip"

        # Exterior: dilate expected by 2px — outside must be zero in got
        exterior = ~binary_dilation(expected > 0, structure=struct, iterations=2)
        assert got[exterior].sum() == 0, f"{label}: exterior pixels gained after roundtrip"

        # Only gains, no losses (contour tracing expands, never shrinks)
        lost = np.count_nonzero((got == 0) & (expected > 0))
        assert lost == 0, f"{label}: {lost} pixels lost (expected 0)"

        # Total diff ≤ 2x perimeter (generous bound for contour + rasterization jitter)
        diff_count = np.count_nonzero(got != expected)
        ys, xs = np.where(expected > 0)
        if len(ys) > 0:
            h = ys.max() - ys.min() + 1
            w = xs.max() - xs.min() + 1
            perimeter = 2 * (h + w)
            assert diff_count <= perimeter * 2, (
                f"{label}: {diff_count} diffs > 2*perimeter ({perimeter * 2})"
            )

    def test_rect_zip_roundtrip(self, tmp_path):
        """Rect: export at 2x upscale → reimport at original size → boundary-close match."""
        ds = 2
        orig_w, orig_h = 200, 200

        orig_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        orig_mask[40:120, 60:160] = 255

        ds_mask = orig_mask[::ds, ::ds]

        zip_path = self._make_zip(tmp_path, [ds_mask], ["rect_roi"], ds)
        reimported = self._import_zip(zip_path, orig_w, orig_h)
        assert "rect_roi" in reimported

        self._assert_boundary_close(reimported["rect_roi"], orig_mask, "rect")

    def test_freehand_zip_roundtrip(self, tmp_path):
        """Freehand circle: export at 2x → reimport → interior/exterior correct."""
        ds = 2
        orig_w, orig_h = 200, 200

        cx, cy, r = 100, 100, 40
        yy, xx = np.ogrid[:orig_h, :orig_w]
        orig_mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2).astype(np.uint8) * 255

        ds_mask = orig_mask[::ds, ::ds]

        zip_path = self._make_zip(tmp_path, [ds_mask], ["circle_roi"], ds)
        reimported = self._import_zip(zip_path, orig_w, orig_h)

        self._assert_boundary_close(reimported["circle_roi"], orig_mask, "circle")

        # Total area within 10% — two contour roundtrips (ds mask→roi, reimport roi→mask)
        # each add ~1px boundary ring, so circle area grows by ~2*perimeter
        orig_count = np.count_nonzero(orig_mask)
        got_count = np.count_nonzero(reimported["circle_roi"])
        assert abs(got_count - orig_count) / orig_count < 0.10, (
            f"Area diverged: orig={orig_count}, reimported={got_count} "
            f"({abs(got_count - orig_count) / orig_count:.1%})"
        )

    def test_multiple_rois_zip_roundtrip(self, tmp_path):
        """Multiple ROIs: export at 3x → reimport → each matches within boundary tolerance."""
        ds = 3
        orig_w, orig_h = 300, 300

        mask_a = np.zeros((orig_h, orig_w), dtype=np.uint8)
        mask_a[30:120, 60:210] = 255
        mask_b = np.zeros((orig_h, orig_w), dtype=np.uint8)
        mask_b[150:270, 90:240] = 255

        ds_a = mask_a[::ds, ::ds]
        ds_b = mask_b[::ds, ::ds]

        zip_path = self._make_zip(tmp_path, [ds_a, ds_b], ["roi_a", "roi_b"], ds)
        reimported = self._import_zip(zip_path, orig_w, orig_h)
        assert "roi_a" in reimported and "roi_b" in reimported

        self._assert_boundary_close(reimported["roi_a"], mask_a, "roi_a")
        self._assert_boundary_close(reimported["roi_b"], mask_b, "roi_b")

    def test_full_pipeline_zip_roundtrip(self, tmp_path):
        """Full user workflow: original ROI → import (scale 1/ds) → export (upscale)
        → reimport at original res → compare to original.
        """
        ds = 2
        orig_w, orig_h = 200, 200
        ds_w, ds_h = orig_w // ds, orig_h // ds

        # Step 1: Original ROI dict (as in an ImageJ ZIP)
        orig_roi = {'type': ROI_RECT, 'top': 40, 'left': 60, 'bottom': 120, 'right': 160}
        orig_mask = imagej_roi_to_mask(orig_roi, orig_w, orig_h)

        # Step 2: Import with scaling
        scaled_roi = scale_roi_dict(orig_roi, 1.0 / ds, 1.0 / ds)
        ds_mask = imagej_roi_to_mask(scaled_roi, ds_w, ds_h)

        # Step 3: Export to ZIP at original resolution
        zip_path = self._make_zip(tmp_path, [ds_mask], ["test_roi"], ds)

        # Step 4: Reimport at original resolution
        reimported = self._import_zip(zip_path, orig_w, orig_h)
        got = reimported["test_roi"]

        self._assert_boundary_close(got, orig_mask, "full_pipeline")
