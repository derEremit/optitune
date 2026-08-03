"""
ScaleSession — pure, Qt-free expectation state machine for scale recording.

Holds pitch-class series state, onset gate, commit decision, next-target
selection, and C<->F paired-series switch. OptiTuneMainWindow is a thin adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SCALE_MODE_CENT_TOLERANCE = 140.0
ONSET_GATE_CENT_TOLERANCE = 800.0

# Workflow caps for the master C-then-F series (C7=96, F7=101)
_SERIES_HI = {0: 96, 5: 101}


def _hz_to_midi(f_hz: float, a4: float = 440.0) -> float:
    if f_hz <= 0:
        return 0.0
    return 69.0 + 12.0 * math.log2(f_hz / a4)


def _midi_to_hz(midi: int | float, a4: float = 440.0) -> float:
    return float(a4 * (2.0 ** ((float(midi) - 69.0) / 12.0)))


def pitch_class_matches(
    est_midi_f: float | None,
    expected_pc: int,
    *,
    tolerance: float = SCALE_MODE_CENT_TOLERANCE,
    f_est: float | None = None,
    a4: float = 440.0,
) -> bool:
    """True if est is within tolerance cents of a note with expected pitch class."""
    if est_midi_f is None and f_est is None:
        return False
    try:
        if f_est is not None and f_est > 1:
            midi_f = _hz_to_midi(float(f_est), a4)
        else:
            midi_f = float(est_midi_f)  # type: ignore[arg-type]
    except Exception:
        return False

    base = round(midi_f)
    best_dist = 9999.0
    for oct_off in (-2, -1, 0, 1, 2):
        for delta in (-12, 0, 12):
            cand = base + oct_off * 12 + delta
            if 21 <= cand <= 108 and (cand % 12) == (expected_pc % 12):
                d = abs(cand - midi_f) * 100.0
                if d < best_dist:
                    best_dist = d
    return best_dist <= tolerance


def cents_error_to_target(f_est: float, target_midi: int, a4: float = 440.0) -> float | None:
    if f_est <= 1 or target_midi is None:
        return None
    target_hz = _midi_to_hz(int(target_midi), a4)
    if target_hz <= 1:
        return None
    return float(1200.0 * math.log2(f_est / target_hz))


@dataclass(frozen=True)
class CommitDecision:
    accept: bool
    f_est_used: float | None = None
    captured_midi: int | float | None = None
    switch_to_pc: int | None = None
    reason: str = ""


class ScaleSession:
    """Expectation layer for scale / root-note series recording."""

    def __init__(
        self,
        *,
        scale_cent_tol: float = SCALE_MODE_CENT_TOLERANCE,
        onset_gate_tol: float = ONSET_GATE_CENT_TOLERANCE,
        grace_s: float = 0.65,
    ) -> None:
        self.scale_pitch_class: int | None = None
        self.grace_until: float = 0.0
        self.ignore_onset_until: float = 0.0
        self.require_strong_attack_until: float = 0.0
        self.during_capture_rejection_until: float = 0.0
        self._scale_cent_tol = scale_cent_tol
        self._onset_gate_tol = onset_gate_tol
        self._grace_s = grace_s

    def enter_scale(self, target_midi: int, *, now: float) -> None:
        self.scale_pitch_class = int(target_midi) % 12
        self.grace_until = now + self._grace_s

    def exit_scale(self) -> None:
        self.scale_pitch_class = None
        self.grace_until = 0.0

    def in_grace(self, now: float) -> bool:
        return now < self.grace_until

    def set_post_capture_guards(self, *, now: float, success: bool) -> None:
        if success:
            self.ignore_onset_until = now + 0.12
            self.require_strong_attack_until = now + 0.35
            self.grace_until = now + 0.45
        else:
            self.ignore_onset_until = now + 0.12
            self.require_strong_attack_until = now + 0.45
            self.grace_until = now + 0.4
        self.during_capture_rejection_until = 0.0

    def should_suppress_onset(
        self,
        *,
        est_midi: float | None,
        armed_midi: int | None,
        now: float,
        f_est: float | None = None,
        a4: float = 440.0,
    ) -> bool:
        """
        Layer-1 pre-filter. Returns True if the energy controller must not see this tick.

        No estimate → never suppress (commit gate is authoritative).
        """
        if self.scale_pitch_class is None:
            return False
        if est_midi is None and f_est is None:
            return False  # no-est allow energy path

        pc_ok = pitch_class_matches(
            est_midi,
            self.scale_pitch_class,
            tolerance=self._onset_gate_tol,
            f_est=f_est,
            a4=a4,
        )
        if self.in_grace(now):
            return False
        close_to_armed = (
            armed_midi is not None
            and est_midi is not None
            and abs(float(est_midi) - armed_midi) <= 20
        )
        if close_to_armed:
            return False
        return not pc_ok

    def next_target(
        self,
        *,
        last_recorded: int | None,
        measured: set[int],
        current: int | None = None,
    ) -> int | None:
        """
        Pick next MIDI to arm. Returns None if nothing left in paired series.
        May flip scale_pitch_class C<->F on exhaustion.
        """
        if self.scale_pitch_class is None:
            return None

        def unmeasured(m: int) -> bool:
            return 21 <= m <= 108 and m not in measured

        pc = self.scale_pitch_class
        series_hi = _SERIES_HI.get(pc, 108)
        cur = last_recorded if last_recorded is not None else (current or 24)
        # Next octave of the same class (strictly above last_recorded when set)
        start = cur - (cur % 12) + pc
        if last_recorded is not None and start <= last_recorded:
            start = last_recorded + 12
        for candidate in range(max(21, start), min(109, series_hi + 1), 12):
            if unmeasured(candidate):
                return candidate

        # Paired series (C <-> F)
        other_pc = {0: 5, 5: 0}.get(pc)
        if other_pc is None:
            return None
        other_hi = _SERIES_HI.get(other_pc, 108)
        first_other = next(m for m in range(21, 109) if m % 12 == other_pc)
        for candidate in range(first_other, min(109, other_hi + 1), 12):
            if unmeasured(candidate):
                self.scale_pitch_class = other_pc
                return candidate
        return None

    def decide_commit(
        self,
        *,
        f_est: float,
        captured_midi: int | float,
        armed_midi: int | None,
        a4: float = 440.0,
        tracker_f0: float | None = None,
    ) -> CommitDecision:
        """Authoritative accept/reject at capture finish."""
        if f_est is None or captured_midi is None:
            return CommitDecision(accept=False, reason="incomplete")

        captured_pc = int(captured_midi) % 12
        current_pc = self.scale_pitch_class

        if current_pc is not None and not pitch_class_matches(
            float(captured_midi),
            current_pc,
            tolerance=self._scale_cent_tol,
            f_est=f_est,
            a4=a4,
        ):
            return CommitDecision(
                accept=False,
                f_est_used=f_est,
                captured_midi=captured_midi,
                reason="wrong_class",
            )

        f_used = float(f_est)
        midi_used: int | float = captured_midi

        if armed_midi is not None:
            err = cents_error_to_target(f_used, armed_midi, a4)
            if err is not None and abs(err) > self._scale_cent_tol:
                if tracker_f0 is not None and tracker_f0 > 20:
                    err_tr = cents_error_to_target(tracker_f0, armed_midi, a4)
                    if err_tr is not None and abs(err_tr) <= self._scale_cent_tol:
                        f_used = float(tracker_f0)
                        midi_used = armed_midi
                        err = err_tr
                if err is not None and abs(err) > self._scale_cent_tol:
                    return CommitDecision(
                        accept=False,
                        f_est_used=f_used,
                        captured_midi=midi_used,
                        reason="too_far",
                    )

        switch_to: int | None = None
        if current_pc is not None and captured_pc != current_pc:
            other = 5 if current_pc == 0 else 0
            if captured_pc == other:
                switch_to = captured_pc
                self.scale_pitch_class = captured_pc

        return CommitDecision(
            accept=True,
            f_est_used=f_used,
            captured_midi=midi_used,
            switch_to_pc=switch_to,
            reason="accept",
        )
