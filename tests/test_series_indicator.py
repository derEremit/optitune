"""Series status-bar indicator updates on arm / advance / switch."""

from __future__ import annotations

from optitune.ui.main_window import OptiTuneMainWindow


def test_series_indicator_idle_by_default(qtbot) -> None:
    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    assert "Series:" in w._series_label.text()
    # Idle shows a dash placeholder
    assert "-" in w._series_label.text()


def test_series_indicator_shows_c_on_arm(qtbot) -> None:
    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    w._record_selected_midi = 24
    w._auto_advance_after_record = True
    w._shown_arm_help = True
    w._toggle_auto_record_arm(True)
    w._update_series_status()
    txt = w._series_label.text()
    assert "C" in txt
    assert "24" in txt or "C1" in txt


def test_series_indicator_counts_measured(qtbot) -> None:
    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    from optitune.model import Key, Piano

    # Fresh piano — do not use persisted session keys
    piano = Piano(a4=440.0, name="test")
    piano.set_key(Key(midi=24, measured_f0=32.7, measured_b=0.0001))
    piano.set_key(Key(midi=36, measured_f0=65.4, measured_b=0.0001))
    w._piano = piano
    w._scale_session.enter_scale(48, now=0.0)
    w._record_selected_midi = 48
    w._update_series_status()
    txt = w._series_label.text()
    # 2 of 7 C notes measured in the C1-C7 window
    assert "2/7" in txt or "2 / 7" in txt
