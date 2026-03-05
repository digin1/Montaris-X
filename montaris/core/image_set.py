import numpy as np
from PySide6.QtCore import QObject, Signal


class ImageSet(QObject):
    """Holds multiple image channels/images with active channel selection."""
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.channels = []  # list of (name, numpy_array)
        self.active_channels = []  # indices of visible channels

    def add_channel(self, name, data):
        self.channels.append((name, data))
        self.active_channels.append(len(self.channels) - 1)
        self.changed.emit()

    def remove_channel(self, index):
        if 0 <= index < len(self.channels):
            self.channels.pop(index)
            self.active_channels = [
                i if i < index else i - 1
                for i in self.active_channels if i != index
            ]
            self.changed.emit()

    def set_active_channels(self, indices):
        self.active_channels = [i for i in indices if 0 <= i < len(self.channels)]
        self.changed.emit()

    def toggle_channel(self, index):
        if index in self.active_channels:
            self.active_channels.remove(index)
        else:
            self.active_channels.append(index)
            self.active_channels.sort()
        self.changed.emit()

    def get_active_data(self):
        """Return list of (name, data) for active channels."""
        return [self.channels[i] for i in self.active_channels if i < len(self.channels)]

    def clear(self):
        self.channels.clear()
        self.active_channels.clear()
        self.changed.emit()

    @property
    def num_channels(self):
        return len(self.channels)

    @property
    def shape(self):
        if self.channels:
            return self.channels[0][1].shape
        return None

    @classmethod
    def from_multichannel(cls, data, channel_names=None):
        """Create ImageSet from a multi-channel array (H, W, C) or (C, H, W)."""
        image_set = cls()
        if data.ndim == 2:
            name = channel_names[0] if channel_names else "Channel 0"
            image_set.add_channel(name, data)
        elif data.ndim == 3:
            # Determine channel axis
            if data.shape[0] in (1, 2, 3, 4) and data.shape[0] < min(data.shape[1], data.shape[2]):
                # Channels-first
                for i in range(data.shape[0]):
                    name = channel_names[i] if channel_names and i < len(channel_names) else f"Channel {i}"
                    image_set.add_channel(name, data[i])
            else:
                # Channels-last or single-channel
                if data.shape[2] <= 4:
                    for i in range(data.shape[2]):
                        name = channel_names[i] if channel_names and i < len(channel_names) else f"Channel {i}"
                        image_set.add_channel(name, data[:, :, i])
                else:
                    name = channel_names[0] if channel_names else "Channel 0"
                    image_set.add_channel(name, data)
        return image_set
