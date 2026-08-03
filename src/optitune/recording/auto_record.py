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

import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

# Controllable verbosity for heavy per-tick diagnostics (priority 6)
# Set OPTITUNE_DIAG=1 (or verbose/full) to see every [DIAG][Onset] tick when armed.
_DIAG_ONSET_VERBOSE = os.environ.get("OPTITUNE_DIAG", "0").lower() in ("1", "true", "yes", "verbose", "full", "on")


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
        self._consecutive_loud = 0
        self._recent_rises: list[float] = []

    def disarm(self) -> None:
        """User explicitly stops waiting / recording."""
        self._phase = AutoRecordPhase.IDLE
        self._target_midi = None
        self._onset_start_time = None
        self._capture_end_time = None
        self._recent_loud_ticks = []
        self._consecutive_loud = 0
        self._recent_rises = []

    def on_level_tick(self, current_db: float, now: float) -> AutoRecordEvent | None:
        """
        Called ~20 times per second from the level meter timer.

        Onset detection hardened on real piano recordings (C1–C7 + F1–F7).

        Strategy:
        - Use consecutive loud ticks for sustained energy.
        - Credit a clear attack (strong rise) if it occurred recently within the current loud streak.
          This is critical for realistic note envelopes (fast attack transient + slower decay) and
          for chunked simulation feeding where the exact confirming tick may not coincide with peak rise.
        - Make confirmation easier for high notes (they decay very fast in real recordings).
        - Still reject random noise / pedal thumps.
        """
        if self._phase == AutoRecordPhase.IDLE:
            return None

        if self._phase == AutoRecordPhase.ARMED:
            # Track previous dB for attack detection.
            # Handle the explicit None reset from arm() cleanly (first tick after arm
            # or after a capture) so we never crash and the very first post-arm tick
            # can participate in onset logic.
            prev_db = getattr(self, "_prev_db", None)
            if prev_db is None:
                prev_db = current_db - 3.0
            self._prev_db = current_db
            db_rise = current_db - prev_db

            loud = current_db > self._config.onset_db_threshold

            if not hasattr(self, "_recent_loud_ticks"):
                self._recent_loud_ticks: list[bool] = []

            # Track consecutive loud ticks at the end (stricter than sliding majority)
            if loud:
                self._consecutive_loud = getattr(self, "_consecutive_loud", 0) + 1
            else:
                self._consecutive_loud = 0

            # Keep short history for diagnostics only
            self._recent_loud_ticks.append(loud)
            if len(self._recent_loud_ticks) > 12:
                self._recent_loud_ticks.pop(0)

            # Track recent rises during the current loud streak.
            # This lets us credit a strong attack transient even if it happened a few ticks earlier
            # (by the time consec count reaches threshold, the instantaneous rise is often near zero).
            if not hasattr(self, "_recent_rises"):
                self._recent_rises: list[float] = []
            if loud:
                self._recent_rises.append(db_rise)
                if len(self._recent_rises) > 10:  # ~500 ms look-back at 50 ms ticks
                    self._recent_rises.pop(0)
            else:
                self._recent_rises = []

            # Dynamic requirement: higher for low notes, still strict overall
            midi = self._target_midi or 60
            octave = (midi - 24) // 12

            if octave >= 5:      # High notes (C6+): allow slightly faster confirmation
                needed_consecutive = 4
            else:
                needed_consecutive = 6   # Low/mid notes: require solid sustained attack

            # Extra strictness after a recent capture (post-capture attack requirement)
            require_strong_attack = getattr(self, "_require_strong_attack_until", 0) > now
            min_rise = 8.0 if require_strong_attack else 5.0

            if _DIAG_ONSET_VERBOSE:
                print(
                    f"[DIAG][Onset] db={current_db:.1f} rise={db_rise:+.1f} "
                    f"loud={loud} consec={self._consecutive_loud} needed={needed_consecutive} "
                    f"strong_attack_req={require_strong_attack} target={self._target_midi}"
                )

            # Confirm if we have enough consecutive loud ticks *and* there was a sufficiently
            # strong rise at some point in the recent loud streak (not only on this exact tick).
            max_recent_rise = max(self._recent_rises) if self._recent_rises else db_rise
            confirmed = (
                self._consecutive_loud >= needed_consecutive and
                max_recent_rise >= min_rise
            )

            if confirmed:
                print(f"[DIAG][Onset] >>> ONSET CONFIRMED for {self._target_midi} (consec={self._consecutive_loud}, rise={db_rise:.1f}, max_recent_rise={max_recent_rise:.1f})")
                self._phase = AutoRecordPhase.RECORDING
                self._capture_end_time = now + (self._config.capture_duration_ms / 1000.0)
                self._recent_loud_ticks = []
                self._consecutive_loud = 0
                self._recent_rises = []
                self._prev_db = current_db
                # Clear any pending strong-attack requirement
                self._require_strong_attack_until = 0
                return AutoRecordEvent.ONSET_CONFIRMED

            # Occasional reset of history if mostly quiet
            if len(self._recent_loud_ticks) >= 8 and sum(self._recent_loud_ticks) <= 2:
                self._recent_loud_ticks = []
                self._consecutive_loud = 0
                self._recent_rises = []

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
