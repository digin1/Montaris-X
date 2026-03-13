"""Tests for AdjustmentsPanel and DisplayPanel widgets.

Covers slider-to-model mapping, reset, signal emission, channel management,
and mode switching.
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from montaris.widgets.adjustments_panel import AdjustmentsPanel
from montaris.widgets.display_panel import DisplayPanel
from montaris.core.display_modes import DisplayMode


# ---------------------------------------------------------------------------
# AdjustmentsPanel
# ---------------------------------------------------------------------------

class TestAdjustmentsPanel:
    @pytest.fixture
    def panel(self, qapp):
        p = AdjustmentsPanel()
        yield p
        p.close()
        p.deleteLater()
        QApplication.processEvents()

    def test_initial_identity(self, panel):
        adj = panel.adjustments
        assert adj.brightness == 0.0
        assert adj.contrast == 0.0
        assert adj.exposure == 0.0
        assert adj.gamma == 1.0
        assert adj.is_identity()

    def test_on_brightness(self, panel):
        panel._on_brightness(50)
        assert panel.adjustments.brightness == pytest.approx(0.5)

    def test_on_brightness_negative(self, panel):
        panel._on_brightness(-100)
        assert panel.adjustments.brightness == pytest.approx(-1.0)

    def test_on_contrast(self, panel):
        panel._on_contrast(-100)
        assert panel.adjustments.contrast == pytest.approx(-1.0)

    def test_on_contrast_positive(self, panel):
        panel._on_contrast(75)
        assert panel.adjustments.contrast == pytest.approx(0.75)

    def test_on_exposure(self, panel):
        panel._on_exposure(200)
        assert panel.adjustments.exposure == pytest.approx(2.0)

    def test_on_exposure_negative(self, panel):
        panel._on_exposure(-150)
        assert panel.adjustments.exposure == pytest.approx(-1.5)

    def test_on_gamma(self, panel):
        panel._on_gamma(250)
        assert panel.adjustments.gamma == pytest.approx(2.5)

    def test_on_gamma_low(self, panel):
        panel._on_gamma(10)
        assert panel.adjustments.gamma == pytest.approx(0.1)

    def test_on_reset(self, panel):
        # Set some non-identity values
        panel._on_brightness(50)
        panel._on_contrast(30)
        panel._on_exposure(100)
        panel._on_gamma(200)
        assert not panel.adjustments.is_identity()
        # Reset
        panel._on_reset()
        adj = panel.adjustments
        assert adj.brightness == 0.0
        assert adj.contrast == 0.0
        assert adj.exposure == 0.0
        assert adj.gamma == 1.0
        assert adj.is_identity()

    def test_reset_syncs_sliders(self, panel):
        panel._on_brightness(80)
        panel._on_reset()
        assert panel.brightness_slider.value() == 0
        assert panel.contrast_slider.value() == 0
        assert panel.exposure_slider.value() == 0
        assert panel.gamma_slider.value() == 100  # 1.0 * 100

    def test_signal_emitted_on_brightness(self, panel):
        received = []
        panel.adjustments_changed.connect(lambda adj: received.append(adj))
        panel._on_brightness(30)
        assert len(received) == 1
        assert received[0].brightness == pytest.approx(0.3)

    def test_signal_emitted_on_contrast(self, panel):
        received = []
        panel.adjustments_changed.connect(lambda adj: received.append(adj))
        panel._on_contrast(-50)
        assert len(received) == 1
        assert received[0].contrast == pytest.approx(-0.5)

    def test_signal_emitted_on_exposure(self, panel):
        received = []
        panel.adjustments_changed.connect(lambda adj: received.append(adj))
        panel._on_exposure(100)
        assert len(received) == 1

    def test_signal_emitted_on_gamma(self, panel):
        received = []
        panel.adjustments_changed.connect(lambda adj: received.append(adj))
        panel._on_gamma(300)
        assert len(received) == 1
        assert received[0].gamma == pytest.approx(3.0)

    def test_signal_emitted_on_reset(self, panel):
        received = []
        panel._on_brightness(40)  # make non-identity first
        panel.adjustments_changed.connect(lambda adj: received.append(adj))
        panel._on_reset()
        assert len(received) == 1
        assert received[0].is_identity()

    def test_updating_flag_prevents_handler(self, panel):
        """When _updating is True, slider change handlers are no-ops."""
        panel._updating = True
        panel._on_brightness(99)
        # Brightness should NOT have changed
        assert panel.adjustments.brightness == 0.0
        panel._updating = False

    def test_slider_setValue_triggers_handler(self, panel):
        """Programmatic slider setValue fires valueChanged -> handler."""
        received = []
        panel.adjustments_changed.connect(lambda adj: received.append(adj))
        panel.brightness_slider.setValue(60)
        assert panel.adjustments.brightness == pytest.approx(0.6)
        assert len(received) == 1


# ---------------------------------------------------------------------------
# DisplayPanel
# ---------------------------------------------------------------------------

class TestDisplayPanel:
    @pytest.fixture
    def panel(self, qapp):
        p = DisplayPanel()
        yield p
        p.close()
        p.deleteLater()
        QApplication.processEvents()

    def test_initial_mode(self, panel):
        mode = panel.get_mode()
        assert mode == DisplayMode.FALSE_COLOR

    def test_set_channels_creates_checkboxes(self, panel):
        panel.set_channels(["R", "G", "B"])
        assert len(panel._channel_checkboxes) == 3
        assert panel._channel_checkboxes[0].text() == "R"
        assert panel._channel_checkboxes[1].text() == "G"
        assert panel._channel_checkboxes[2].text() == "B"

    def test_get_active_channels_all_active(self, panel):
        panel.set_channels(["R", "G", "B"])
        active = panel.get_active_channels()
        assert active == [0, 1, 2]

    def test_get_active_channels_with_active_indices(self, panel):
        panel.set_channels(["Ch1", "Ch2", "Ch3", "Ch4"], active_indices=[0, 2])
        active = panel.get_active_channels()
        assert active == [0, 2]

    def test_toggle_checkbox_updates_active(self, panel):
        panel.set_channels(["A", "B", "C"])
        # Uncheck the second checkbox
        panel._channel_checkboxes[1].setChecked(False)
        active = panel.get_active_channels()
        assert active == [0, 2]

    def test_channels_changed_signal_on_toggle(self, panel):
        panel.set_channels(["X", "Y"])
        received = []
        panel.channels_changed.connect(lambda ch: received.append(ch))
        panel._channel_checkboxes[0].setChecked(False)
        assert len(received) == 1
        assert received[0] == [1]  # only second channel active

    def test_mode_changed_signal(self, panel):
        received = []
        panel.mode_changed.connect(lambda m: received.append(m))
        # Switch to GRAYSCALE (index 4 per enum order)
        grayscale_idx = None
        for i in range(panel.mode_combo.count()):
            if panel.mode_combo.itemData(i) == DisplayMode.GRAYSCALE:
                grayscale_idx = i
                break
        assert grayscale_idx is not None
        panel.mode_combo.setCurrentIndex(grayscale_idx)
        assert len(received) >= 1
        assert received[-1] == DisplayMode.GRAYSCALE

    def test_on_mode_changed_direct(self, panel):
        received = []
        panel.mode_changed.connect(lambda m: received.append(m))
        # Call _on_mode_changed directly with an index
        # Index 0 = COMPOSITE_RGB per the enum iteration order
        panel._on_mode_changed(0)
        assert len(received) == 1
        assert received[0] == DisplayMode.COMPOSITE_RGB

    def test_set_channels_replaces_old(self, panel):
        panel.set_channels(["A", "B"])
        assert len(panel._channel_checkboxes) == 2
        panel.set_channels(["X", "Y", "Z", "W"])
        assert len(panel._channel_checkboxes) == 4
        assert panel._channel_checkboxes[3].text() == "W"

    def test_composite_toggled_signal(self, panel):
        received = []
        panel.composite_toggled.connect(lambda v: received.append(v))
        panel.composite_cb.setChecked(True)
        assert len(received) == 1
        assert received[0] is True

    def test_empty_channels(self, panel):
        panel.set_channels([])
        assert len(panel._channel_checkboxes) == 0
        assert panel.get_active_channels() == []

    def test_single_channel(self, panel):
        panel.set_channels(["DAPI"])
        assert len(panel._channel_checkboxes) == 1
        assert panel.get_active_channels() == [0]
