"""QSettings persistence for active scale series (crash resume)."""

from __future__ import annotations

from PySide6.QtCore import QSettings

from optitune.ui.main_window import OptiTuneMainWindow


def _clear_scale_settings() -> None:
    s = QSettings()
    s.remove("scale/active_pitch_class")
    s.remove("scale/last_recorded_midi")
    s.remove("scale/armed_midi")
    s.sync()


def test_persist_series_on_arm(qtbot) -> None:
    _clear_scale_settings()
    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    w._record_selected_midi = 24
    w._auto_advance_after_record = True
    w._shown_arm_help = True
    w._toggle_auto_record_arm(True)

    s = QSettings()
    assert s.contains("scale/active_pitch_class")
    assert int(s.value("scale/active_pitch_class")) == 0
    assert s.contains("scale/armed_midi")
    assert int(s.value("scale/armed_midi")) == 24
    _clear_scale_settings()


def test_clear_series_settings_on_disarm(qtbot) -> None:
    _clear_scale_settings()
    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    w._record_selected_midi = 36
    w._auto_advance_after_record = True
    w._shown_arm_help = True
    w._toggle_auto_record_arm(True)
    w._toggle_auto_record_arm(False)

    s = QSettings()
    assert not s.contains("scale/active_pitch_class")
    assert not s.contains("scale/armed_midi")
    _clear_scale_settings()


def test_restore_series_on_startup(qtbot) -> None:
    _clear_scale_settings()
    s = QSettings()
    s.setValue("scale/active_pitch_class", 5)
    s.setValue("scale/last_recorded_midi", 29)
    s.setValue("scale/armed_midi", 41)
    s.sync()

    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()

    assert w._scale_pitch_class == 5
    assert w._last_recorded_midi == 29
    assert w._record_selected_midi == 41
    assert "F" in w._series_label.text()
    _clear_scale_settings()
