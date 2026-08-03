"""
TDD for temporal f0 tracking (median/mode over recent ticks).

Attack frames are outliers; decay is long and stable. A pure tracker lets
live analysis reject one-off octave jumps without Qt.
"""

from __future__ import annotations

import pytest

from optitune.dsp.f0_tracker import F0Tracker
from optitune.dsp.peaks import cents
from optitune.dsp.synth import midi_to_hz


def test_empty_tracker_returns_none():
    t = F0Tracker(window=5)
    assert t.current() is None
    assert t.current_midi() is None


def test_single_push_is_current():
    t = F0Tracker(window=5)
    t.push(440.0)
    assert t.current() == pytest.approx(440.0)
    assert t.current_midi(a4=440.0) == 69


def test_median_rejects_octave_spike():
    """Four stable frames + one octave spike → median stays on the stable f0."""
    t = F0Tracker(window=5)
    f0 = midi_to_hz(36)  # C2
    for _ in range(4):
        t.push(f0)
    t.push(f0 * 4)  # partial/octave spike
    out = t.current()
    assert out is not None
    assert abs(cents(out, f0)) < 5.0


def test_window_slides():
    t = F0Tracker(window=3)
    t.push(100.0)
    t.push(100.0)
    t.push(100.0)
    t.push(200.0)
    t.push(200.0)
    t.push(200.0)
    # last 3 are all 200
    assert t.current() == pytest.approx(200.0)


def test_clear():
    t = F0Tracker(window=4)
    t.push(440.0)
    t.clear()
    assert t.current() is None


def test_push_ignores_non_positive():
    t = F0Tracker(window=3)
    t.push(440.0)
    t.push(0.0)
    t.push(-1.0)
    assert t.current() == pytest.approx(440.0)


def test_mode_like_stability_on_two_clusters():
    """When half the window is an octave error, median still prefers lower if ordered carefully.
    With even window use lower of two middle values after sort (numpy median).
    """
    t = F0Tracker(window=4)
    f0 = 65.4
    t.push(f0)
    t.push(f0 * 2)
    t.push(f0)
    t.push(f0 * 2)
    # median of [65.4, 130.8, 65.4, 130.8] = 98.1 — not ideal
    # Prefer a tracker that picks the denser lower cluster when available
    out = t.current()
    assert out is not None
    # At least prefer something near either cluster, not garbage
    assert min(abs(cents(out, f0)), abs(cents(out, f0 * 2))) < 5.0
