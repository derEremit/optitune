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


def test_series_complete_disarms_and_clears_indicator(qtbot) -> None:
    """When C and F series are fully measured, advance clears series + disarms."""
    from optitune.model import Key, Piano

    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()

    piano = Piano(a4=440.0, name="test")
    # All C1–C7 and F1–F7 measured
    for m in list(range(24, 97, 12)) + list(range(29, 102, 12)):
        piano.set_key(Key(midi=m, measured_f0=440.0 * (2 ** ((m - 69) / 12)), measured_b=0.0001))
    w._piano = piano
    w._scale_session.enter_scale(96, now=0.0)  # last C
    w._last_recorded_midi = 101  # last F just recorded
    w._record_selected_midi = 101
    w._auto_advance_after_record = True

    w._on_record_next()
    assert w._record_selected_midi is None
    # Indicator still shows series until exit; finish path exits scale
    w._scale_session.exit_scale()
    w._update_series_status()
    assert "-" in w._series_label.text()


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
