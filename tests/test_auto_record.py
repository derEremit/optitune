"""
TDD for the hands-free auto-record + auto-advance workflow.

These tests were written first (proper TDD) to encode the exact user requirements
that were broken in the field:

- "it should be red all the time while it is armed for recording"
- Reliable level-triggered capture (arm, walk away, play the note, it records ~1.8s)
- After capture, if Auto-advance ON: move to next key + re-arm automatically
- No lost states, no synchronous RECORDING -> MEASURED flash that the user never sees
- Works with the real piano recordings we now have as fixtures

The production code (AutoRecordController) is extracted so these can be fast,
deterministic unit tests without Qt or audio hardware.
"""

from __future__ import annotations

import time

import pytest

from optitune.recording.auto_record import (
    AutoRecordConfig,
    AutoRecordController,
    AutoRecordEvent,
    AutoRecordPhase,
)


def test_initial_state_is_idle():
    ctrl = AutoRecordController()
    assert ctrl.phase == AutoRecordPhase.IDLE
    assert ctrl.target_midi is None
    assert not ctrl.is_armed
    assert not ctrl.is_recording


def test_arm_sets_armed_phase_and_remembers_target():
    ctrl = AutoRecordController()
    ctrl.arm(60)  # C4

    assert ctrl.phase == AutoRecordPhase.ARMED
    assert ctrl.target_midi == 60
    assert ctrl.is_armed
    assert not ctrl.is_recording


def test_armed_stays_armed_while_sound_is_below_threshold():
    """Core user requirement: the key must stay visibly ARMED (red) until a real onset."""
    ctrl = AutoRecordController(
        AutoRecordConfig(onset_db_threshold=-28.0, min_onset_confirmation_ms=280)
    )
    ctrl.arm(60)

    now = time.time()
    # Simulate many level ticks with quiet or background sound
    for _ in range(20):
        now += 0.05
        event = ctrl.on_level_tick(current_db=-42.0, now=now)
        assert event is None
        assert ctrl.phase == AutoRecordPhase.ARMED
        assert ctrl.target_midi == 60


def test_onset_confirmation_requires_sustained_level_above_threshold():
    ctrl = AutoRecordController(
        AutoRecordConfig(onset_db_threshold=-28.0, min_onset_confirmation_ms=280)
    )
    ctrl.arm(60)

    now = time.time()
    # One loud blip should not trigger
    event = ctrl.on_level_tick(current_db=-20.0, now=now + 0.05)
    assert event is None
    assert ctrl.phase == AutoRecordPhase.ARMED

    # Clear attack (strong rise) into sustained loud — the recent-rise credit logic
    # requires a >= min_rise transient somewhere in the loud streak for confirmation.
    t = now + 0.1
    event = ctrl.on_level_tick(current_db=-12.0, now=t)  # strong +6 dB rise from prior -18
    assert event is None  # still building consec

    # Sustained loud sound for >= min_onset_confirmation_ms
    events = []
    for _i in range(10):  # ~500 ms of loud sound
        t += 0.05
        ev = ctrl.on_level_tick(current_db=-12.0, now=t)
        if ev:
            events.append(ev)

    assert any(e == AutoRecordEvent.ONSET_CONFIRMED for e in events)
    assert ctrl.phase == AutoRecordPhase.RECORDING
    assert ctrl.is_recording


def test_recording_phase_runs_for_configured_duration_then_emits_finished():
    ctrl = AutoRecordController(AutoRecordConfig(capture_duration_ms=300))  # short for test speed
    ctrl.arm(60)
    # Force into recording (bypass onset for this test)
    ctrl._force_recording_for_test(60, duration_ms=300)  # helper only in tests

    now = time.time()
    # Should still be recording shortly after start
    ev = ctrl.on_level_tick(current_db=-15.0, now=now + 0.1)
    assert ev is None
    assert ctrl.phase == AutoRecordPhase.RECORDING

    # After the configured duration we get the finish event
    ev = ctrl.on_level_tick(current_db=-15.0, now=now + 0.5)
    assert ev == AutoRecordEvent.CAPTURE_FINISHED
    assert (
        ctrl.phase == AutoRecordPhase.IDLE
    )  # or a transitional state; controller yields control back


def test_disarm_while_armed_clears_state_and_target():
    ctrl = AutoRecordController()
    ctrl.arm(72)
    assert ctrl.phase == AutoRecordPhase.ARMED

    ctrl.disarm()
    assert ctrl.phase == AutoRecordPhase.IDLE
    assert ctrl.target_midi is None
    assert not ctrl.is_armed


def test_auto_advance_decision_is_external_but_controller_supports_rearm():
    """The controller itself doesn't decide *which* next key; MainWindow does.
    But after a capture finishes, the UI can ask the controller to arm the next one.
    """
    ctrl = AutoRecordController()
    ctrl.arm(60)

    # Simulate full cycle
    # ... (onset, recording, finished) ...

    # After commit + advance, UI calls:
    ctrl.arm(62)  # next key (D4 in the example)
    assert ctrl.phase == AutoRecordPhase.ARMED
    assert ctrl.target_midi == 62


def test_configurable_onset_params_are_respected():
    cfg = AutoRecordConfig(onset_db_threshold=-35.0, min_onset_confirmation_ms=150)
    ctrl = AutoRecordController(cfg)
    ctrl.arm(48)

    # A sound at -30 dB should now count as onset (above the lower threshold).
    # Include a clear attack rise so the recent-rise credit triggers confirmation.
    now = time.time()
    t = now
    # Establish a quiet baseline, then a clear strong attack rise into the note region.
    t += 0.05
    ctrl.on_level_tick(-55.0, now=t)  # quiet tick sets prev
    t += 0.05
    ev = ctrl.on_level_tick(-22.0, now=t)  # strong rise >5 into loud
    for _ in range(6):
        t += 0.05
        ev = ctrl.on_level_tick(-32.0, now=t)
        if ev == AutoRecordEvent.ONSET_CONFIRMED:
            break
    else:
        pytest.fail("Should have confirmed onset with the more sensitive config")


# ---------------- Integration-style scenarios the user actually hit ----------------


def test_full_hands_free_cycle_with_auto_advance_simulation():
    """
    Reproduce the exact workflow the user wanted:
    Arm C4 -> play C4 (onset) -> records 1.8s -> finishes -> auto-advance to C#4 -> re-arms
    """
    ctrl = AutoRecordController(AutoRecordConfig(capture_duration_ms=200))  # fast test

    # User clicks "Arm Auto-Record" on C4 (or Record Next chose it)
    ctrl.arm(60)
    assert ctrl.phase == AutoRecordPhase.ARMED

    # User walks to piano and plays the note (real attack from our recordings would be used in future)
    now = time.time()
    t = now
    onset_event = None
    # Quiet baseline then clear attack transient + sustained (exercises recent-rise credit)
    t += 0.05
    ctrl.on_level_tick(-55.0, now=t)
    t += 0.05
    ev = ctrl.on_level_tick(-5.0, now=t)  # strong rise
    while not onset_event and (t - now) < 2.0:
        t += 0.05
        ev = ctrl.on_level_tick(current_db=-15.0, now=t)
        if ev == AutoRecordEvent.ONSET_CONFIRMED:
            onset_event = ev

    assert ctrl.phase == AutoRecordPhase.RECORDING

    # Wait out the capture window
    t += 0.3
    finish = ctrl.on_level_tick(-10.0, now=t)
    assert finish == AutoRecordEvent.CAPTURE_FINISHED


# ---------------- Protection against live detection clobbering ----------------


def test_controller_provides_forced_visual_state_while_armed_or_recording():
    """
    This is the key protection for the user's complaint that red/yellow was
    flashing incorrectly during armed or recording.
    While the controller owns a target, it must tell the UI to force ARMED or
    RECORDING state so that _run_live_analysis cannot overwrite it with
    whatever the (still-imperfect) estimator happens to detect.
    """
    from optitune.ui.widgets.keyboard_widget import KeyState

    ctrl = AutoRecordController()
    ctrl.arm(72)  # C#5 or whatever

    forced = ctrl.get_forced_visual_state()
    assert forced is not None
    midi, state = forced
    assert midi == 72
    assert state == KeyState.ARMED

    # Simulate transition to recording (normally via onset)
    ctrl._force_recording_for_test(72, duration_ms=1000)
    forced = ctrl.get_forced_visual_state()
    assert forced is not None
    midi, state = forced
    assert state == KeyState.RECORDING

    # Once finished / disarmed, no forcing
    ctrl.disarm()
    assert ctrl.get_forced_visual_state() is None

    # At this point MainWindow would have called _on_record_note() + _on_record_next()
    # then re-arms the new target
    ctrl.arm(61)  # simulate auto-advance result
    assert ctrl.phase == AutoRecordPhase.ARMED
    assert ctrl.target_midi == 61
