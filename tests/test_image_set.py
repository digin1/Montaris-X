import numpy as np
import pytest
from montaris.core.image_set import ImageSet


class TestImageSet:
    def test_add_channel(self):
        iset = ImageSet()
        data = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        iset.add_channel("Red", data)
        assert iset.num_channels == 1
        assert 0 in iset.active_channels

    def test_multiple_channels(self):
        iset = ImageSet()
        for i in range(3):
            data = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
            iset.add_channel(f"Channel {i}", data)
        assert iset.num_channels == 3
        assert len(iset.active_channels) == 3

    def test_toggle_channel(self):
        iset = ImageSet()
        for i in range(3):
            iset.add_channel(f"Ch{i}", np.zeros((10, 10), dtype=np.uint8))
        iset.toggle_channel(1)
        assert 1 not in iset.active_channels
        iset.toggle_channel(1)
        assert 1 in iset.active_channels

    def test_remove_channel(self):
        iset = ImageSet()
        for i in range(3):
            iset.add_channel(f"Ch{i}", np.zeros((10, 10), dtype=np.uint8))
        iset.remove_channel(1)
        assert iset.num_channels == 2

    def test_get_active_data(self):
        iset = ImageSet()
        iset.add_channel("A", np.ones((10, 10), dtype=np.uint8))
        iset.add_channel("B", np.ones((10, 10), dtype=np.uint8) * 2)
        active = iset.get_active_data()
        assert len(active) == 2

    def test_from_multichannel_2d(self):
        data = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        iset = ImageSet.from_multichannel(data)
        assert iset.num_channels == 1

    def test_from_multichannel_3d_channels_last(self):
        data = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
        iset = ImageSet.from_multichannel(data)
        assert iset.num_channels == 3

    def test_from_multichannel_3d_channels_first(self):
        data = np.random.randint(0, 255, (3, 50, 60), dtype=np.uint8)
        iset = ImageSet.from_multichannel(data)
        assert iset.num_channels == 3

    def test_shape(self):
        iset = ImageSet()
        iset.add_channel("A", np.zeros((50, 60), dtype=np.uint8))
        assert iset.shape == (50, 60)

    def test_clear(self):
        iset = ImageSet()
        iset.add_channel("A", np.zeros((10, 10), dtype=np.uint8))
        iset.clear()
        assert iset.num_channels == 0
