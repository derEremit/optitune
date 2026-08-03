"""
Auto-record controller - extracted pure(ish) state machine for the hands-free
level-triggered recording workflow.

This is the result of proper TDD after the user reported real-world breakage
of auto-arm, persistent red ARMED state, auto-advance + re-arm, and capture timing.

The MainWindow holds one of these and treats it as the single source of truth
for "what should the currently targeted key look like right now?" (ARMED / RECORDING / IDLE).

All timing is explicit (pass `now` into on_level_tick) so tests are deterministic.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optitune.ui.widgets.keyboard_widget import KeyState

logger = logging.getLogger(__name__)

# Controllable verbosity for heavy per-tick diagnostics (priority 6)
# Set OPTITUNE_DIAG=1 (or verbose/full) to see every [DIAG][Onset] tick when armed.
_DIAG_ONSET_VERBOSE = os.environ.get("OPTITUNE_DIAG", "0").lower() in (
    "1",
    "true",
    "yes",
    "verbose",
    "full",
    "on",
)

if _DIAG_ONSET_VERBOSE and not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


class AutoRecordPhase(Enum):
    IDLE = auto()
    ARMED = auto()  # Waiting for the musician to play the target note (red)
    CONFIRMING_ONSET = auto()  # Saw loud sound, accumulating confirmation time
    RECORDING = auto()  # In the fixed-length capture window (stronger red)


class AutoRecordEvent(Enum):
    """Events the controller can emit to the UI layer."""

    ONSET_CONFIRMED = auto()  # Transition ARMED -> RECORDING
    CAPTURE_FINISHED = auto()  # Time to call _finish_auto_capture / commit


@dataclass(frozen=True)
class AutoRecordConfig:
    onset_db_threshold: float = -28.0
    min_onset_confirmation_ms: int = 280
    # ~1.1 s is enough for f0/B and fits typical scale note spacing (~2-4 s).
    # Longer windows (1.8 s) overlap the next note and desync the series.
    capture_duration_ms: int = 1100


class AutoRecordController:
    """Deterministic, testable controller for the auto-record flow."""

    def __init__(self, config: AutoRecordConfig | None = None) -> None:
        self._config = config or AutoRecordConfig()
        self._phase: AutoRecordPhase = AutoRecordPhase.IDLE
        self._target_midi: int | None = None
        self._onset_start_time: float | None = None
        self._capture_end_time: float | None = None
        self._recent_loud_ticks: list[bool] = []
        self._prev_db: float | None = None
        self._consecutive_loud: int = 0
        self._recent_rises: list[float] = []
        self._require_strong_attack_until: float = 0.0

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

    def get_forced_visual_state(self) -> tuple[int, KeyState] | None:
        """
        If we are currently ARMED or RECORDING a specific key, return
        (midi, desired KeyState) so the UI can force that visual state
        and prevent live detection from clobbering it.
        """
        from optitune.ui.widgets.keyboard_widget import (
            KeyState as _KS,  # avoid circular import at module level
        )

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
        self._recent_rises = []

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

        Onset detection hardened on real piano recordings (C1-C7 + F1-F7).

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
            prev_db = self._prev_db
            if prev_db is None:
                # Seed so the first real attack after arm has a meaningful rise
                # (was -3 dB, which blocked confirmation when the streak started mid-note).
                prev_db = current_db - 12.0
            self._prev_db = current_db
            db_rise = current_db - prev_db

            loud = current_db > self._config.onset_db_threshold

            # Track consecutive loud ticks at the end (stricter than sliding majority)
            if loud:
                self._consecutive_loud += 1
            else:
                self._consecutive_loud = 0

            # Keep short history for diagnostics only
            self._recent_loud_ticks.append(loud)
            if len(self._recent_loud_ticks) > 12:
                self._recent_loud_ticks.pop(0)

            # Track recent rises during the current loud streak.
            # This lets us credit a strong attack transient even if it happened a few ticks earlier
            # (by the time consec count reaches threshold, the instantaneous rise is often near zero).
            if loud:
                self._recent_rises.append(db_rise)
                if len(self._recent_rises) > 10:  # ~500 ms look-back at 50 ms ticks
                    self._recent_rises.pop(0)
            else:
                self._recent_rises = []

            # Dynamic requirement: higher for low notes, still strict overall
            midi = self._target_midi or 60
            octave = (midi - 24) // 12

            # High notes decay in a few 50 ms ticks - require fewer consecutive loud samples.
            # C7 (oct 6): 2; C6 (oct 5): 3; C5 (oct 4): 4; low/mid: 6.
            if octave >= 6:
                needed_consecutive = 2
            elif octave >= 5:
                needed_consecutive = 3
            elif octave >= 4:
                needed_consecutive = 4
            else:
                needed_consecutive = 6

            # Extra strictness after a recent capture (post-capture attack requirement)
            require_strong_attack = self._require_strong_attack_until > now
            # Normal attacks only need a modest rise; strong-attack window is for
            # rejecting pedal noise right after a capture. Keep both low enough that
            # a real note attack after long quiet always clears the bar.
            min_rise = 6.0 if require_strong_attack else 3.0

            if _DIAG_ONSET_VERBOSE:
                logger.debug(
                    "[DIAG][Onset] db=%.1f rise=%+.1f loud=%s consec=%d needed=%d "
                    "strong_attack_req=%s target=%s",
                    current_db,
                    db_rise,
                    loud,
                    self._consecutive_loud,
                    needed_consecutive,
                    require_strong_attack,
                    self._target_midi,
                )

            # Confirm if we have enough consecutive loud ticks *and* there was a sufficiently
            # strong rise at some point in the recent loud streak (not only on this exact tick).
            # Fallback: long sustained loud without a measured rise (attack tick may have been
            # gated out of the controller) still counts after needed+2 ticks.
            max_recent_rise = max(self._recent_rises) if self._recent_rises else db_rise
            sustained_only = self._consecutive_loud >= needed_consecutive + 2
            confirmed = self._consecutive_loud >= needed_consecutive and (
                max_recent_rise >= min_rise or sustained_only
            )

            if confirmed:
                logger.info(
                    "[DIAG][Onset] >>> ONSET CONFIRMED for %s (consec=%d, rise=%.1f, "
                    "max_recent_rise=%.1f)",
                    self._target_midi,
                    self._consecutive_loud,
                    db_rise,
                    max_recent_rise,
                )
                self._phase = AutoRecordPhase.RECORDING
                self._capture_end_time = now + (self._config.capture_duration_ms / 1000.0)
                self._recent_loud_ticks = []
                self._consecutive_loud = 0
                self._recent_rises = []
                self._prev_db = current_db
                # Clear any pending strong-attack requirement
                self._require_strong_attack_until = 0.0
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
