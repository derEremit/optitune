"""Unsaved-changes prompt only for dirty open .pfg files."""

from __future__ import annotations

from optitune.model import Key, Piano
from optitune.ui.main_window import OptiTuneMainWindow


def test_dirty_without_pfg_does_not_prompt(qtbot) -> None:
    """Casual record/compute sessions use JSON autosave — no quit nag."""
    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    w._mark_session_dirty()
    assert w.is_session_dirty() is True
    assert w._current_pfg_path is None
    assert w.should_prompt_unsaved() is False


def test_dirty_with_pfg_prompts(qtbot, tmp_path) -> None:
    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    p = Piano()
    p.set_key(Key(midi=60, measured_f0=261.0, measured_b=0.0004))
    w._piano = p
    w._current_pfg_path = str(tmp_path / "studio.pfg")
    w._mark_session_dirty()
    assert w.should_prompt_unsaved() is True
    w._mark_session_clean()
    assert w.should_prompt_unsaved() is False


def test_close_with_clean_session_accepts(qtbot) -> None:
    from PySide6.QtGui import QCloseEvent

    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    w._mark_session_dirty()  # no pfg path → still no prompt
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert ev.isAccepted()
