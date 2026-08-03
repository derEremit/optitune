"""TDD: Auto / Stepwise / Lock note-follow search windows (spec §3.6)."""

from __future__ import annotations

import pytest

from optitune.dsp.note_follow import NoteFollowMode, apply_follow_to_midi, search_window


def test_auto_is_full_compass():
    lo, hi = search_window(NoteFollowMode.AUTO, locked_midi=60)
    assert lo == 21
    assert hi == 108


def test_stepwise_is_plus_minus_one():
    lo, hi = search_window(NoteFollowMode.STEPWISE, locked_midi=60)
    assert lo == 59
    assert hi == 61


def test_stepwise_clamps_at_edges():
    lo, hi = search_window(NoteFollowMode.STEPWISE, locked_midi=21)
    assert lo == 21
    assert hi == 22
    lo, hi = search_window(NoteFollowMode.STEPWISE, locked_midi=108)
    assert lo == 107
    assert hi == 108


def test_lock_window_is_single_key():
    lo, hi = search_window(NoteFollowMode.LOCK, locked_midi=72)
    assert lo == hi == 72


def test_lock_without_locked_falls_back_to_auto():
    lo, hi = search_window(NoteFollowMode.LOCK, locked_midi=None)
    assert lo == 21 and hi == 108


def test_apply_follow_lock_keeps_locked():
    assert apply_follow_to_midi(NoteFollowMode.LOCK, detected=64, locked=60) == 60


def test_apply_follow_stepwise_accepts_neighbor():
    assert apply_follow_to_midi(NoteFollowMode.STEPWISE, detected=61, locked=60) == 61
    assert apply_follow_to_midi(NoteFollowMode.STEPWISE, detected=72, locked=60) == 60


def test_apply_follow_auto_takes_detected():
    assert apply_follow_to_midi(NoteFollowMode.AUTO, detected=72, locked=60) == 72
