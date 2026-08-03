"""
Note-follow modes for live note identity (spec §3.6).

Auto     — recognizer may pick any key in the piano compass.
Stepwise — only ±1 semitone from the locked/current note (anti-octave jumps).
Lock     — manual only; detected pitch does not change the tracked key.

Scale / series mode is a separate workflow layer (ScaleSession); it uses an
armed prior and does not replace these modes for free listening.
"""

from __future__ import annotations

from enum import Enum

from optitune.dsp.note_recognizer import MIDI_HIGH, MIDI_LOW


class NoteFollowMode(str, Enum):
    AUTO = "auto"
    STEPWISE = "stepwise"
    LOCK = "lock"


def search_window(
    mode: NoteFollowMode,
    locked_midi: int | None,
    *,
    midi_lo: int = MIDI_LOW,
    midi_hi: int = MIDI_HIGH,
) -> tuple[int, int]:
    """
    Candidate MIDI range for the comb recognizer under the given follow mode.

    Returns (lo, hi) inclusive.
    """
    if mode is NoteFollowMode.AUTO or locked_midi is None:
        return int(midi_lo), int(midi_hi)

    locked = int(locked_midi)
    if mode is NoteFollowMode.LOCK:
        return locked, locked

    # STEPWISE
    lo = max(midi_lo, locked - 1)
    hi = min(midi_hi, locked + 1)
    return lo, hi


def apply_follow_to_midi(
    mode: NoteFollowMode,
    *,
    detected: int | None,
    locked: int | None,
) -> int | None:
    """
    Resolve which MIDI should be the live tracked note after recognition.

    Lock: always keep locked (if set).
    Stepwise: accept detected only if within ±1 of locked; else keep locked.
    Auto: take detected when available, else locked.
    """
    if mode is NoteFollowMode.LOCK:
        return locked if locked is not None else detected

    if detected is None:
        return locked

    if mode is NoteFollowMode.AUTO or locked is None:
        return int(detected)

    # STEPWISE
    if abs(int(detected) - int(locked)) <= 1:
        return int(detected)
    return int(locked)
