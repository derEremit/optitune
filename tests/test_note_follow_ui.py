"""Toolbar note-follow mode selector wires into MainWindow state."""

from __future__ import annotations

from optitune.dsp.note_follow import NoteFollowMode
from optitune.ui.main_window import OptiTuneMainWindow


def test_follow_combo_defaults_to_auto(qtbot) -> None:
    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    assert w._note_follow_mode is NoteFollowMode.AUTO
    # QComboBox may round-trip str Enum userData as plain str
    assert NoteFollowMode(w._follow_combo.currentData()) is NoteFollowMode.AUTO


def test_follow_combo_switches_to_stepwise_and_lock(qtbot) -> None:
    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    w._follow_combo.setCurrentIndex(1)  # Stepwise
    assert w._note_follow_mode is NoteFollowMode.STEPWISE
    w._follow_combo.setCurrentIndex(2)  # Lock
    assert w._note_follow_mode is NoteFollowMode.LOCK


def test_keyboard_click_sets_follow_lock_anchor(qtbot) -> None:
    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    w._on_keyboard_clicked(60)
    assert w._follow_locked_midi == 60
    assert w._record_selected_midi == 60
