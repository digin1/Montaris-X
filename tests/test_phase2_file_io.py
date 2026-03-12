"""Tests for Phase 2: File I/O Enhancements."""
import os
import tempfile
import zipfile
import pytest
import numpy as np

from montaris.layers import ROILayer
from montaris.io.imagej_roi import (
    mask_to_imagej_roi, write_imagej_roi, write_imagej_roi_bytes,
    read_imagej_roi, imagej_roi_to_mask,
)
from montaris.core.roi_ops import auto_fit_rois


class TestWriteImagejRoiBytes:
    def test_bytes_roundtrip(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        roi_dict = mask_to_imagej_roi(mask, "test")
        data = write_imagej_roi_bytes(roi_dict)
        assert isinstance(data, bytes)
        assert len(data) >= 64

        # Should be readable
        parsed = read_imagej_roi(data)
        assert parsed['type'] == roi_dict['type']

    def test_bytes_equals_file(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        roi_dict = mask_to_imagej_roi(mask, "test")
        data = write_imagej_roi_bytes(roi_dict)

        tmp = tempfile.NamedTemporaryFile(suffix='.roi', delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            write_imagej_roi(roi_dict, tmp_path)
            with open(tmp_path, 'rb') as ff:
                file_data = ff.read()
        finally:
            os.unlink(tmp_path)

        assert data == file_data


class TestAutoFitRois:
    def test_no_change_needed(self):
        roi = ROILayer("test", 100, 80)
        roi.mask[10:20, 10:20] = 255
        count = auto_fit_rois([roi], 100, 80)
        assert count == 0

    def test_resize_different_dimensions(self):
        roi = ROILayer("test", 50, 40)
        roi.mask[5:10, 5:10] = 255
        count = auto_fit_rois([roi], 100, 80)
        assert count == 1
        assert roi.mask.shape == (80, 100)


class TestZipExportImport:
    def test_zip_roundtrip(self):
        """Create ROIs, export to zip, import from zip."""
        mask1 = np.zeros((50, 50), dtype=np.uint8)
        mask1[5:15, 5:15] = 255
        mask2 = np.zeros((50, 50), dtype=np.uint8)
        mask2[20:30, 20:30] = 255

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            zip_path = f.name

        try:
            # Export
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for i, mask in enumerate([mask1, mask2]):
                    roi_dict = mask_to_imagej_roi(mask, f"roi_{i}")
                    if roi_dict:
                        data = write_imagej_roi_bytes(roi_dict)
                        zf.writestr(f"roi_{i}.roi", data)

            # Import
            with zipfile.ZipFile(zip_path, 'r') as zf:
                names = zf.namelist()
                assert len(names) == 2
                for name in names:
                    data = zf.read(name)
                    parsed = read_imagej_roi(data)
                    mask = imagej_roi_to_mask(parsed, 50, 50)
                    assert mask.shape == (50, 50)
                    assert np.any(mask > 0)
        finally:
            os.unlink(zip_path)


class TestPngMaskImport:
    def test_png_mask_creation(self):
        """Verify PNG mask import logic."""
        from PIL import Image
        arr = np.zeros((100, 120), dtype=np.uint8)
        arr[20:40, 30:50] = 200

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            Image.fromarray(arr).save(f.name)
            path = f.name

        try:
            img = Image.open(path).convert('L')
            arr_loaded = np.array(img)
            mask = (arr_loaded > 0).astype(np.uint8) * 255
            assert mask.shape == (100, 120)
            assert np.count_nonzero(mask) > 0
        finally:
            os.unlink(path)
