"""
TDD for ScaleSession — pure (Qt-free) expectation state machine.

Covers arm → gate → commit → advance → series switch → exhaustion.
MainWindow becomes a thin adapter over this module.
"""

from __future__ import annotations

import pytest

from optitune.recording.scale_session import (
    ONSET_GATE_CENT_TOLERANCE,
    SCALE_MODE_CENT_TOLERANCE,
    ScaleSession,
    pitch_class_matches,
)


def test_constants():
    assert SCALE_MODE_CENT_TOLERANCE == 140.0
    assert ONSET_GATE_CENT_TOLERANCE == 800.0


def test_enter_scale_on_arm():
    s = ScaleSession()
    assert s.scale_pitch_class is None
    s.enter_scale(24, now=1000.0)
    assert s.scale_pitch_class == 0
    assert s.in_grace(1000.1)
    assert not s.in_grace(1001.0)


def test_exit_scale():
    s = ScaleSession()
    s.enter_scale(29, now=0.0)
    assert s.scale_pitch_class == 5
    s.exit_scale()
    assert s.scale_pitch_class is None


def test_pitch_class_matches_c_within_tolerance():
    # Continuous midi near C4 (60)
    assert pitch_class_matches(60.1, expected_pc=0, tolerance=140.0)
    assert pitch_class_matches(59.9, expected_pc=0, tolerance=140.0)
    # Far from any C
    assert not pitch_class_matches(61.5, expected_pc=0, tolerance=140.0)


def test_pitch_class_matches_none_est_is_false():
    assert pitch_class_matches(None, expected_pc=0) is False


def test_onset_gate_allows_no_estimate():
    s = ScaleSession()
    s.enter_scale(24, now=0.0)
    # After grace, no estimate should NOT suppress (energy path allowed)
    assert s.should_suppress_onset(est_midi=None, armed_midi=24, now=10.0) is False


def test_onset_gate_suppresses_wrong_class_after_grace():
    s = ScaleSession()
    s.enter_scale(24, now=0.0)
    # C# far from any C
    assert s.should_suppress_onset(est_midi=61.0, armed_midi=24, now=10.0) is True


def test_onset_gate_allows_close_to_armed_despite_class():
    s = ScaleSession()
    s.enter_scale(24, now=0.0)
    # Within 20 semitones of armed C1 even if class fuzzy
    assert s.should_suppress_onset(est_midi=36.0, armed_midi=24, now=10.0) is False


def test_next_in_series_walks_octaves():
    s = ScaleSession()
    s.enter_scale(24, now=0.0)
    measured: set[int] = set()
    n = s.next_target(last_recorded=24, measured=measured)
    assert n == 36
    measured.add(24)
    measured.add(36)
    n = s.next_target(last_recorded=36, measured=measured)
    assert n == 48


def test_series_exhaustion_switches_c_to_f():
    s = ScaleSession()
    s.enter_scale(24, now=0.0)
    # All C1-C7 measured
    measured = set(range(24, 97, 12))
    n = s.next_target(last_recorded=96, measured=measured)
    assert n == 29  # F1
    assert s.scale_pitch_class == 5


def test_series_exhaustion_switches_f_to_c():
    s = ScaleSession()
    s.enter_scale(29, now=0.0)
    measured = set(range(29, 102, 12))
    n = s.next_target(last_recorded=101, measured=measured)
    assert n == 24
    assert s.scale_pitch_class == 0


def test_any_pitch_class_series_walks_octaves():
    """Arming G (pc=7) walks G1,G2,... without requiring C/F."""
    s = ScaleSession()
    s.enter_scale(31, now=0.0)  # G1
    assert s.scale_pitch_class == 7
    measured: set[int] = set()
    n = s.next_target(last_recorded=31, measured=measured)
    assert n == 43  # G2
    measured |= {31, 43}
    n = s.next_target(last_recorded=43, measured=measured)
    assert n == 55  # G3


def test_non_paired_series_exhausts_without_switch():
    """G series has no pair — exhaustion returns None, pc unchanged."""
    s = ScaleSession()
    s.enter_scale(31, now=0.0)
    # All G in 21..108
    measured = set(m for m in range(21, 109) if m % 12 == 7)
    last = max(measured)
    n = s.next_target(last_recorded=last, measured=measured)
    assert n is None
    assert s.scale_pitch_class == 7


def test_commit_accept_same_class_near_armed():
    s = ScaleSession()
    s.enter_scale(24, now=0.0)
    d = s.decide_commit(
        f_est=32.7,
        captured_midi=24,
        armed_midi=24,
        a4=440.0,
    )
    assert d.accept is True
    assert d.switch_to_pc is None


def test_commit_reject_too_far():
    s = ScaleSession()
    s.enter_scale(24, now=0.0)
    d = s.decide_commit(
        f_est=130.0,  # ~C3 while armed C1
        captured_midi=48,
        armed_midi=24,
        a4=440.0,
    )
    assert d.accept is False


def test_commit_accept_via_tracker_fallback():
    s = ScaleSession()
    s.enter_scale(24, now=0.0)
    d = s.decide_commit(
        f_est=130.0,
        captured_midi=48,
        armed_midi=24,
        a4=440.0,
        tracker_f0=32.8,
    )
    assert d.accept is True
    assert d.f_est_used == pytest.approx(32.8, abs=0.1)


def test_post_capture_guards():
    s = ScaleSession()
    s.set_post_capture_guards(now=100.0, success=True)
    assert s.ignore_onset_until == pytest.approx(100.12)
    assert s.require_strong_attack_until == pytest.approx(100.35)
    s.set_post_capture_guards(now=200.0, success=False)
    assert s.ignore_onset_until == pytest.approx(200.12)
    assert s.require_strong_attack_until == pytest.approx(200.45)
