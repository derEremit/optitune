"""Unsaved-changes prompt when closing with a dirty piano session."""

from __future__ import annotations

from optitune.model import Key, Piano
from optitune.ui.main_window import OptiTuneMainWindow


def test_mark_dirty_on_record_clear_on_pfg_save(qtbot, tmp_path) -> None:
    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    assert w.is_session_dirty() is False

    p = Piano()
    p.set_key(Key(midi=60, measured_f0=261.0, measured_b=0.0004))
    w._piano = p
    w._mark_session_dirty()
    assert w.is_session_dirty() is True

    path = tmp_path / "t.pfg"
    from optitune.persistence.tuning_file import save_pfg

    save_pfg(p, path)
    w._current_pfg_path = str(path)
    w._mark_session_clean()
    assert w.is_session_dirty() is False


def test_close_with_clean_session_accepts(qtbot) -> None:
    from PySide6.QtGui import QCloseEvent

    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    # Clean → closeEvent should accept (no dialog path)
    assert w.is_session_dirty() is False
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert ev.isAccepted()
