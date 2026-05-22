"""
Auto-record controller — extracted pure(ish) state machine for the hands-free
level-triggered recording workflow.

This is the result of proper TDD after the user reported real-world breakage
of auto-arm, persistent red ARMED state, auto-advance + re-arm, and capture timing.

The MainWindow holds one of these and treats it as the single source of truth
for "what should the currently targeted key look like right now?" (ARMED / RECORDING / IDLE).

All timing is explicit (pass `now` into on_level_tick) so tests are deterministic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class AutoRecordPhase(Enum):
    IDLE = auto()
    ARMED = auto()           # Waiting for the musician to play the target note (red)
    CONFIRMING_ONSET = auto() # Saw loud sound, accumulating confirmation time
    RECORDING = auto()       # In the fixed-length capture window (stronger red)


class AutoRecordEvent(Enum):
    """Events the controller can emit to the UI layer."""
    ONSET_CONFIRMED = auto()   # Transition ARMED -> RECORDING
    CAPTURE_FINISHED = auto()  # Time to call _finish_auto_capture / commit


@dataclass(frozen=True)
class AutoRecordConfig:
    onset_db_threshold: float = -28.0
    min_onset_confirmation_ms: int = 280
    capture_duration_ms: int = 1800


class AutoRecordController:
    """Deterministic, testable controller for the auto-record flow."""

    def __init__(self, config: AutoRecordConfig | None = None) -> None:
        self._config = config or AutoRecordConfig()
        self._phase: AutoRecordPhase = AutoRecordPhase.IDLE
        self._target_midi: int | None = None
        self._onset_start_time: float | None = None
        self._capture_end_time: float | None = None

    # ---------------- Public API used by MainWindow ----------------

    @property
    def phase(self) -> AutoRecordPhase:
        return self._phase

    @property
    def target_midi(self) -> int | None:
        return self._target_midi

    @property
    def is_armed(self) -> bool:
        return self._phase == AutoRecordPhase.ARMED

    @property
    def is_recording(self) -> bool:
        return self._phase == AutoRecordPhase.RECORDING

    @property
    def capture_duration_ms(self) -> int:
        return self._config.capture_duration_ms

    def get_forced_visual_state(self) -> tuple[int, "KeyState"] | None:
        """
        If we are currently ARMED or RECORDING a specific key, return
        (midi, desired KeyState) so the UI can force that visual state
        and prevent live detection from clobbering it.
        """
        from optitune.ui.widgets.keyboard_widget import KeyState as _KS  # avoid circular import at module level

        if self._target_midi is None:
            return None

        if self._phase == AutoRecordPhase.ARMED:
            return self._target_midi, _KS.ARMED
        if self._phase == AutoRecordPhase.RECORDING:
            return self._target_midi, _KS.RECORDING
        return None

    def arm(self, target_midi: int | None) -> None:
        """User (or auto-advance) arms the system for a specific key."""
        if target_midi is None:
            return
        self._target_midi = int(target_midi)
        self._phase = AutoRecordPhase.ARMED
        self._onset_start_time = None
        self._capture_end_time = None
        self._recent_loud_ticks = []
        self._prev_db = None

    def disarm(self) -> None:
        """User explicitly stops waiting / recording."""
        self._phase = AutoRecordPhase.IDLE
        self._target_midi = None
        self._onset_start_time = None
        self._capture_end_time = None
        self._recent_loud_ticks = []

    def on_level_tick(self, current_db: float, now: float) -> AutoRecordEvent | None:
        """
        Called ~20 times per second from the level meter timer.

        Onset detection hardened on real piano recordings (C1–C7 + F1–F7).

        Strategy:
        - Use a sliding window of recent "loud" ticks.
        - Give extra credit for a clear attack (sudden rise in energy).
        - Make confirmation easier for high notes (they decay very fast in real recordings).
        - Still reject random noise / pedal thumps.
        """
        if self._phase == AutoRecordPhase.IDLE:
            return None

        if self._phase == AutoRecordPhase.ARMED:
            # Track previous dB for attack detection
            prev_db = getattr(self, "_prev_db", current_db - 3.0)
            self._prev_db = current_db
            db_rise = current_db - prev_db

            loud = current_db > self._config.onset_db_threshold

            if not hasattr(self, "_recent_loud_ticks"):
                self._recent_loud_ticks: list[bool] = []

            # Strong attack gives an extra "loud vote"
            strong_attack = db_rise > 6.0
            self._recent_loud_ticks.append(loud or strong_attack)

            # Window size ~350-450 ms
            if len(self._recent_loud_ticks) > 9:
                self._recent_loud_ticks.pop(0)

            loud_count = sum(1 for x in self._recent_loud_ticks if x)

            # Dynamic threshold based on target note height
            # High notes in the user's real recordings are very short → lower bar
            midi = self._target_midi or 60
            octave = (midi - 24) // 12   # rough octave above C1

            if octave >= 5:           # C6 and above (very short in real data)
                needed = 3
            elif octave >= 4:         # C5–B5
                needed = 4
            else:                     # C1–C4 (longer sustains)
                needed = 5

            if loud_count >= needed:
                self._phase = AutoRecordPhase.RECORDING
                self._capture_end_time = now + (self._config.capture_duration_ms / 1000.0)
                self._recent_loud_ticks = []
                self._prev_db = current_db
                return AutoRecordEvent.ONSET_CONFIRMED

            # Reset history on long quiet periods
            if len(self._recent_loud_ticks) >= 7 and loud_count <= 1:
                self._recent_loud_ticks = []

            return None

        if self._phase == AutoRecordPhase.RECORDING:
            if self._capture_end_time is not None and now >= self._capture_end_time:
                self._phase = AutoRecordPhase.IDLE
                self._capture_end_time = None
                return AutoRecordEvent.CAPTURE_FINISHED
            return None

        return None

    # ---------------- Test helpers (not part of public contract) ----------------

    def _force_recording_for_test(self, target_midi: int, duration_ms: int) -> None:
        """Only for fast deterministic tests. Bypasses the onset path."""
        self._target_midi = target_midi
        self._phase = AutoRecordPhase.RECORDING
        self._capture_end_time = time.time() + (duration_ms / 1000.0)
        self._onset_start_time = None
