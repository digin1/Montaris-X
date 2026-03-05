import numpy as np
import pytest
from montaris.io.imagej_roi import (
    read_imagej_roi, write_imagej_roi,
    mask_to_imagej_roi, imagej_roi_to_mask,
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
