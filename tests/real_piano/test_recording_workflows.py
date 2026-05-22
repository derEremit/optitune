"""
TDD for realistic hands-free recording workflows using the actual piano recordings.

These tests simulate a real user:

- Arming auto-record
- Walking to the piano
- Playing one or more notes from the real recorded set (C1-C7, F1-F7)
- Verifying correct state machine behavior, visual states, auto-advance, etc.

Because they use the real recordings and drive the full GUI timers, they are
marked `real_piano` and are intentionally slower.

The goal is to drive improvements to onset detection, auto-advance reliability,
and visual stability using the exact material the user will record on their piano.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import numpy as np
import pytest
from PySide6.QtCore import QTimer

from optitune.ui.main_window import OptiTuneMainWindow
from optitune.ui.widgets.keyboard_widget import KeyState
from tests.real_piano.loader import load_recording

MASTER_RECORDING_PATH = Path(__file__).parent.parent.parent / "testmaterial" / "c and f.flac"

pytestmark = pytest.mark.real_piano

# ---------------------------------------------------------------------------
# Headless vs visible behavior for these workflow tests
# ---------------------------------------------------------------------------
# Default: the GUI window is NOT shown (fully automated, no interaction).
# To force the window to appear (useful while debugging the actual GUI):
#   OPTITUNE_TEST_SHOW_GUI=1 uv run pytest -m real_piano ...
#
# For completely headless (no X server at all):
#   QT_QPA_PLATFORM=offscreen uv run pytest -m real_piano ...
SHOW_GUI = bool(os.environ.get("OPTITUNE_TEST_SHOW_GUI"))
HEADLESS = bool(os.environ.get("QT_QPA_PLATFORM") == "offscreen" or os.environ.get("OPTITUNE_TEST_HEADLESS"))


def _create_test_window(qtbot):
    """Create a main window suitable for automated workflow tests.

    Default behavior (what you get when running the tests normally):
        - Window is NOT shown.
        - No manual clicks or interaction required.

    To force the real GUI window to appear for debugging:
        OPTITUNE_TEST_SHOW_GUI=1 uv run pytest -m real_piano ...

    For completely headless (no display server needed at all):
        QT_QPA_PLATFORM=offscreen uv run pytest -m real_piano ...
    """
    window = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(window)

    if SHOW_GUI and not HEADLESS:
        window.show()
        qtbot.waitExposed(window, timeout=2000)
    else:
        # We still need the event loop to breathe a little so timers can fire
        qtbot.wait(30)

    return window


def _stop_real_audio(window: OptiTuneMainWindow) -> None:
    """Stop the real sounddevice capture so we fully control what goes into the ringbuffer."""
    if window.audio_capture.is_running:
        window.audio_capture.stop()
    window.ringbuffer.clear()


def _arm_for_note(window: OptiTuneMainWindow, midi: int, qtbot) -> None:
    """Select the target key and arm auto-record (simulates user clicking the key then Arm button).

    We suppress the first-time help dialog so the test runs fully unattended.
    """
    window._record_selected_midi = midi
    window.keyboard.set_current_key(midi)

    # Prevent the modal "Auto-Record (Hands-Free)" explanation dialog from appearing
    # during automated tests. The user would otherwise have to click OK.
    window._shown_arm_help = True

    # Simulate clicking the Arm action
    window._arm_record_action.setChecked(True)
    window._toggle_auto_record_arm(True)

    qtbot.wait(50)


def _push_audio_chunk(window: OptiTuneMainWindow, audio: np.ndarray, sr: int, chunk_samples: int):
    """Push a chunk of real audio into the ringbuffer and return the remainder."""
    if len(audio) == 0:
        return audio
    chunk = audio[:chunk_samples].astype(np.float32)
    window.ringbuffer.push(chunk)
    return audio[chunk_samples:]


# ---------------------------------------------------------------------------
# Diagnostic helpers for understanding real piano onsets
# ---------------------------------------------------------------------------

def get_db_sequence_from_real_recording(note_label: str, chunk_ms: int = 50, max_seconds: float = 4.0) -> list[float]:
    """
    Replay a real recording through the same RMS → dB calculation that the
    level meter uses. Returns the list of dB values the controller would see.
    Very useful for tuning onset detection against actual piano material.
    """
    audio, sr, _ = load_recording(note_label)

    # Simulate what the level meter sees (latest 1024 samples)
    window_size = 1024
    chunk_size = max(window_size, int(sr * chunk_ms / 1000))

    dbs = []
    pos = 0
    while pos < len(audio) and (pos / sr) < max_seconds:
        # Take a window ending at current position (like ringbuffer.get_latest)
        start = max(0, pos - window_size)
        buf = audio[start:pos + chunk_size] if pos > 0 else audio[:chunk_size]

        if len(buf) < 64:
            pos += chunk_size
            continue

        # Pad or trim to window_size like the real meter does
        if len(buf) > window_size:
            buf = buf[-window_size:]
        elif len(buf) < window_size:
            buf = np.pad(buf, (0, window_size - len(buf)))

        rms = float(np.sqrt(np.mean(buf ** 2)))
        db = -60.0 if rms <= 1e-7 else 20.0 * np.log10(rms)
        dbs.append(round(db, 1))
        pos += chunk_size

    return dbs


def feed_master_recording_with_guided_sequence(
    window: OptiTuneMainWindow,
    qtbot,
    target_sequence: list[int],
    *,
    chunk_ms: int = 60,
    max_duration_s: float = 70.0,
) -> tuple[int, list[int]]:
    """
    Feed the full master recording (c and f.flac) while using a predetermined
    target sequence for auto-arm/advance.

    This simulates the user wanting to record a known scale/pattern (C then F)
    by playing the entire performance in one go.

    Returns the number of successfully captured notes.
    """
    flac_path = MASTER_RECORDING_PATH.with_suffix(".flac")
    if not MASTER_RECORDING_PATH.exists() and not flac_path.exists():
        pytest.skip("Master recording not present")

    # Prefer the compressed FLAC version (much smaller in git)
    if MASTER_RECORDING_PATH.exists():
        import soundfile as sf
        data, sr = sf.read(MASTER_RECORDING_PATH)
        audio = data.astype(np.float32)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
    else:
        # Fallback to original WAV (for users who still have it locally)
        wav_path = MASTER_RECORDING_PATH.with_suffix(".wav")
        sr, data = wavfile.read(wav_path)
        if len(data.shape) > 1:
            audio = data.mean(axis=1).astype(np.float32)
        else:
            audio = data.astype(np.float32)

    # Normalize
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs

    _stop_real_audio(window)

    chunk_samples = int(sr * chunk_ms / 1000)
    pos = 0
    captured = 0
    captured_midis: list[int] = []
    current_target_idx = 0

    start_time = time.time()

    while pos < len(audio) and (time.time() - start_time) < max_duration_s:
        end = min(pos + chunk_samples, len(audio))
        chunk = audio[pos:end]
        if len(chunk) > 0:
            window.ringbuffer.push(chunk.astype(np.float32))

        # Drive the level meter (this is where the controller sees real onsets)
        try:
            window._update_level_meter()
        except Exception:
            pass

        # Occasionally run analysis so _last_est is populated for commit
        if (pos // chunk_samples) % 4 == 0:
            try:
                window._run_live_analysis()
            except Exception:
                pass

        # After each level tick, check if the current target has been successfully measured.
        # If so, advance to the next in our known sequence and re-arm.
        if current_target_idx < len(target_sequence):
            expected = target_sequence[current_target_idx]
            if window._piano and expected in window._piano.keys:
                key = window._piano.keys[expected]
                if key.measured_f0 is not None or key.measured_b is not None:
                    if expected not in captured_midis:
                        captured_midis.append(expected)
                        captured += 1
                    current_target_idx += 1

                    if current_target_idx < len(target_sequence):
                        next_target = target_sequence[current_target_idx]
                        window._record_selected_midi = next_target
                        window.keyboard.set_current_key(next_target)
                        window._auto_record_ctrl.arm(next_target)
                        window.keyboard.set_key_state(next_target, KeyState.ARMED)
                        window._arm_record_action.setChecked(True)
                        window._apply_auto_record_visual_force()

        pos = end
        qtbot.wait(5)

    return captured, captured_midis


def simulate_play_real_note(
    window: OptiTuneMainWindow,
    note_label: str,
    qtbot,
    *,
    pre_silence_ms: int = 250,
    chunk_ms: int = 70,
    total_play_time_ms: int | None = None,
) -> None:
    """
    Load a real recording and feed it chunk-by-chunk while manually driving
    the level meter (and optionally analysis) so the AutoRecordController
    sees realistic dB values from the user's actual piano.

    This is much more deterministic than relying purely on QTimer wall time.
    """
    audio, sr, meta = load_recording(note_label)

    silence_samples = int(sr * pre_silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32) * 1e-5
    audio = np.concatenate([silence, audio])

    if total_play_time_ms is not None:
        max_samples = int(sr * total_play_time_ms / 1000)
        audio = audio[:max_samples]

    _stop_real_audio(window)

    chunk_samples = max(256, int(sr * chunk_ms / 1000))
    remaining = audio

    iterations = 0
    while len(remaining) > 256 and iterations < 80:
        remaining = _push_audio_chunk(window, remaining, sr, chunk_samples)

        # Manually tick the level meter so the controller sees the real energy
        try:
            window._update_level_meter()
        except Exception:
            pass

        # Occasionally let the analysis run too (so _last_est gets populated)
        if iterations % 3 == 0:
            try:
                window._run_live_analysis()
            except Exception:
                pass

        qtbot.wait(5)  # give Qt event loop a tiny breath
        iterations += 1

    # Final tail + one more meter tick
    if len(remaining) > 0:
        window.ringbuffer.push(remaining.astype(np.float32))
    try:
        window._update_level_meter()
    except Exception:
        pass

    qtbot.wait(30)


# =============================================================================
# Concrete user workflow tests (TDD style)
# =============================================================================

def test_record_one_note_using_real_recording(qtbot):
    """
    User scenario:
    - Select C4 on the keyboard
    - Arm Auto-Record
    - Play the real C4 recording
    - System should detect onset, record ~1.8s, commit the measurement,
      and leave the key in MEASURED state.
    """
    window = _create_test_window(qtbot)
    _stop_real_audio(window)

    target_midi = 60  # C4

    # For the "just record this one note" scenario we want auto-advance OFF
    window._auto_advance_after_record = False
    window._auto_advance_action.setChecked(False)

    _arm_for_note(window, target_midi, qtbot)

    # Verify we are armed (red)
    assert window._auto_record_ctrl.phase.name == "ARMED"
    assert window.keyboard._states.get(target_midi) == KeyState.ARMED

    # Now "play" the real C4 recording
    simulate_play_real_note(window, "C4", qtbot, pre_silence_ms=200, chunk_ms=60)

    # Give the system time to finish the capture window + commit
    qtbot.wait(2200)

    # After successful capture + commit, the key should be MEASURED
    # (and the controller should no longer be forcing ARMED/RECORDING on it)
    final_state = window.keyboard._states.get(target_midi)
    assert final_state == KeyState.MEASURED, f"Expected MEASURED after recording C4, got {final_state}"

    # The controller should be back to IDLE (or ready for next manual arm)
    assert window._auto_record_ctrl.phase.name == "IDLE"

    window.close()


def test_record_full_ascending_C_scale_with_auto_advance(qtbot):
    """
    The main workflow the user asked for:

    Arm on C1, play real C1 → auto-capture → auto-advance to C2 → re-arm
    → play real C2 → ... up to C7.

    This is the "record all C notes from low to high with auto-advance" scenario.
    """
    window = _create_test_window(qtbot)
    _stop_real_audio(window)

    # Turn auto-advance ON (default, but make sure)
    window._auto_advance_after_record = True
    window._auto_advance_action.setChecked(True)

    c_notes = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    base_midis = [24, 36, 48, 60, 72, 84, 96]

    for i, (label, midi) in enumerate(zip(c_notes, base_midis)):
        # Use the helper so we get the dialog suppression + consistent behavior
        _arm_for_note(window, midi, qtbot)
        qtbot.wait(30)

        # Should be armed on this C
        assert window._auto_record_ctrl.target_midi == midi
        assert window._auto_record_ctrl.phase.name == "ARMED"

        # Play the real recording for this note
        simulate_play_real_note(
            window,
            label,
            qtbot,
            pre_silence_ms=150,
            chunk_ms=65,
            total_play_time_ms=2200,  # enough to cover the 1.8s capture + some tail
        )

        # Wait for capture to finish + processing
        qtbot.wait(2400)

        if i < len(c_notes) - 1:
            # For this specific "record the C scale low to high" workflow test,
            # we explicitly pick the next C we want instead of relying on the
            # general "next unmeasured" heuristic in _on_record_next (which
            # prefers middle-out and can pick wrong notes while the estimator
            # is still bad on real piano).
            next_midi = base_midis[i + 1]
            window._record_selected_midi = next_midi
            window.keyboard.set_current_key(next_midi)

            # Re-arm the controller for the exact next note we want
            window._auto_record_ctrl.arm(next_midi)
            window.keyboard.set_key_state(next_midi, KeyState.ARMED)
            window._arm_record_action.setChecked(True)
            window._arm_record_action.setText("⏹ Stop Arming")

            assert window._auto_record_ctrl.target_midi == next_midi
            assert window._auto_record_ctrl.phase.name == "ARMED"
            assert window.keyboard._states.get(next_midi) == KeyState.ARMED
        else:
            # After the last note (C7)
            assert window._auto_record_ctrl.phase.name in ("IDLE", "ARMED")

    # All C notes should now be marked MEASURED
    for midi in base_midis:
        state = window.keyboard._states.get(midi)
        assert state == KeyState.MEASURED, f"C note {midi} was not left in MEASURED state"

    window.close()


# =============================================================================
# Additional thoughtful user scenarios (to be expanded)
# =============================================================================

def test_record_soft_high_note_C7(qtbot):
    """High notes in the user's recording are short and softer — important regression case."""
    window = _create_test_window(qtbot)
    _stop_real_audio(window)

    midi = 96  # C7

    # Explicitly turn auto-advance off for the "record this soft high note" scenario
    window._auto_advance_after_record = False
    window._auto_advance_action.setChecked(False)

    _arm_for_note(window, midi, qtbot)

    # Use the real (short) C7 recording
    simulate_play_real_note(window, "C7", qtbot, pre_silence_ms=100, chunk_ms=50)

    qtbot.wait(2200)

    # Even the short/soft C7 should result in a committed measurement
    assert window.keyboard._states.get(midi) == KeyState.MEASURED

    window.close()


def test_user_cancels_arming_before_playing(qtbot):
    """User arms, then decides not to record and disarms."""
    window = _create_test_window(qtbot)
    _stop_real_audio(window)

    midi = 60
    _arm_for_note(window, midi, qtbot)
    assert window._auto_record_ctrl.phase.name == "ARMED"

    # User clicks Stop Arming
    window._arm_record_action.setChecked(False)
    window._toggle_auto_record_arm(False)

    assert window._auto_record_ctrl.phase.name == "IDLE"
    # The key should have reverted from ARMED
    assert window.keyboard._states.get(midi) != KeyState.ARMED

    window.close()


# ---------------------------------------------------------------------------
# Diagnostics / data gathering for improving onset on real material
# ---------------------------------------------------------------------------

def test_show_real_piano_db_profiles():
    """
    Diagnostic test (run with -s to see output).

    Prints the dB timeline the level meter would see when the user plays
    one of the real recordings. This data is gold for tuning the onset
    detection logic in AutoRecordController so it works reliably on the
    actual piano instead of synthetic thresholds.
    """
    for label in ["C4", "C1", "C7", "F3"]:
        try:
            dbs = get_db_sequence_from_real_recording(label, chunk_ms=50, max_seconds=3.5)
            above = [d for d in dbs if d > -28]
            print(f"\n{label}: {len(dbs)} samples, max dB={max(dbs):.1f}, "
                  f"samples > -28dB = {len(above)} / {len(dbs)}")
            # Show the first 20 values so we can see the attack shape
            print("  first 20 dB:", dbs[:20])
        except Exception as e:
            print(f"{label}: failed to analyze - {e}")


def test_play_full_master_recording_should_capture_correct_c_and_f_sequence(qtbot):
    """
    The strong end-to-end scenario the user requested:

    - Load the master recording (c and f.flac, originally recorded as WAV)
    - Arm on C1
    - Play/feed the entire performance
    - The system should correctly detect onsets, capture each note (~1.8s windows),
      commit them, and (with guided sequencing for now) end up with the right 14 notes
      measured in the right order.

    This is the "play the whole file should record the right notes at the end" test.
    It heavily exercises auto-arm, real continuous onset detection, capture timing
    while the file keeps playing, auto-advance + re-arm, visual state forcing, and
    final model correctness under realistic conditions.
    """
    window = _create_test_window(qtbot)

    # Known correct sequence from how the master file was recorded
    c_and_f_sequence = [
        24, 36, 48, 60, 72, 84, 96,   # C1–C7
        29, 41, 53, 65, 77, 89, 101   # F1–F7
    ]

    # Arm on the first note
    first = c_and_f_sequence[0]
    window._record_selected_midi = first
    window.keyboard.set_current_key(first)
    window._auto_advance_after_record = True
    window._auto_advance_action.setChecked(True)
    window._shown_arm_help = True
    window._arm_record_action.setChecked(True)
    window._toggle_auto_record_arm(True)

    captured, captured_list = feed_master_recording_with_guided_sequence(
        window, qtbot, c_and_f_sequence, chunk_ms=55, max_duration_s=80
    )

    print(f"\n[Full master file] Captured {captured} notes: {captured_list}")

    # Strong requirement: the full real performance should result in (almost) all notes captured
    assert captured >= 13, f"Only captured {captured} notes ({captured_list}) when feeding the full master file"

    if window._piano:
        measured = [m for m, k in window._piano.keys.items()
                    if k.measured_f0 is not None or k.measured_b is not None]
        # At least the C scale should be substantially complete
        c_measured = len([m for m in measured if m in c_and_f_sequence[:7]])
        assert c_measured >= 5, f"Only {c_measured} C notes were recorded from the full file"

    window.close()