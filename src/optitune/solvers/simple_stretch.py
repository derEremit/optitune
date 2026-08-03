"""
Legacy heuristic beat-rate / stretch solver (kept for comparison & fallback).

This was the Phase 4 v0.1 pragmatic solver:
log-linear B-curve fit from measured keys + simple position/B-dependent
heuristic stretch + post-correction Shah-Välimäki 1:2 treble rule.

The preferred production implementation is the proper iterative linearized
weighted beat-rate least-squares solver in `beat_rate.py` (spec §6.2).
It directly minimizes a rich set of interval beat rates β and is the
default exposed as `compute_basic_tuning_curve`.

This module's `compute_heuristic_stretch_curve` is still useful for:
- A/B comparisons in tests
- Emergency fallback on pathological data
- Understanding the "before" behavior

The public API of the solvers package (`compute_basic_tuning_curve`,
`apply_curve_to_piano`) now uses the improved solver; the heuristic is
available explicitly for regression and diagnostics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from optitune.model.piano import Piano

MIDI_LOW = 21
MIDI_HIGH = 108
N_KEYS = 88
A4_MIDI = 69
A4_IDX = A4_MIDI - MIDI_LOW  # 48


def default_railsback_curve() -> list[float]:
    """Public alias for the sensible default ET + Railsback-ish stretch."""
    return _default_railsback_curve()


def _default_railsback_curve() -> list[float]:
    """Sensible default ET + Railsback-ish stretch (no measurements)."""
    curve: list[float] = [0.0] * N_KEYS
    for i in range(N_KEYS):
        m = MIDI_LOW + i
        x = (m - A4_MIDI) / 12.0  # octaves from A4
        if m <= A4_MIDI:
            # Bass: progressively flatter (negative)
            curve[i] = -5.5 * (abs(x) ** 1.85)
        else:
            # Treble: progressively sharper
            curve[i] = 4.2 * (x**1.6)
    curve[A4_IDX] = 0.0
    # Light Shah-like lift in very top
    for i in range(72, N_KEYS):  # from ~C7 upward
        curve[i] = max(curve[i], 3.0 + 0.4 * (i - 72))
    return curve


def _shah_valimaki_adjust(curve: list[float], b_pred: np.ndarray, midis: np.ndarray) -> None:
    """Enforce Shah & Välimäki 1:2 partial matching rule in the top octave.

    For high notes, we bias the target so the upper fundamental approximately
    matches the 2nd partial of the note one octave below (under its own B).
    This is done as a post-correction pass (light touch).  (Legacy heuristic only.)
    """
    # Work from the top downward for a few octaves
    for i in range(N_KEYS - 1, 60, -1):  # top down to ~C6
        m = MIDI_LOW + i
        if m < 84:  # only top ~2 octaves get the strong rule
            break
        # lower = m-12
        lower_idx = i - 12
        if lower_idx < 0:
            continue
        b_low = float(b_pred[lower_idx])
        # 2nd partial of lower under its target offset
        f_low_base = 440.0 * (2.0 ** ((MIDI_LOW + lower_idx - 69) / 12.0))
        # adjust by its own curve offset (approx, small angle)
        f_low_tuned = f_low_base * (2.0 ** (curve[lower_idx] / 1200.0))
        partial2_low = 2.0 * f_low_tuned * math.sqrt(1.0 + 4.0 * b_low)
        # desired f0 for current upper to match that partial2 (pure 2:1)
        desired_f0 = partial2_low / 2.0
        # convert to cent offset from ET
        et_f0 = 440.0 * (2.0 ** ((m - 69) / 12.0))
        if desired_f0 > 1 and et_f0 > 1:
            desired_off = 1200.0 * math.log2(desired_f0 / et_f0)
            # Blend: pull the current heuristic toward the Shah value (stronger near top)
            alpha = 0.6 + 0.35 * min(1.0, (m - 84) / 24.0)
            curve[i] = (1.0 - alpha) * curve[i] + alpha * desired_off


def compute_heuristic_stretch_curve(piano: Piano) -> list[float]:
    """Legacy heuristic stretch curve (B-curve + position heuristic + post Shah).

    Kept for comparison, regression tests, and as an explicit fallback.
    The production solver is the weighted beat-rate LS implementation in
    beat_rate.compute_basic_tuning_curve (the one exposed at package level).

    Uses measured B values if present (≥3 keys recommended). Falls back gracefully.
    Always pins A4 at 0 cents and applies a treble boundary rule.
    """
    # 1. Gather measured (midi, B)
    measured: list[tuple[int, float]] = []
    for k in piano.keys.values():
        if k.measured_b is not None and 1e-6 < k.measured_b < 1.0:
            measured.append((k.midi, float(k.measured_b)))

    if len(measured) < 3:
        return _default_railsback_curve()

    # 2. Log-linear fit on measured B(m)
    ms = np.asarray([m for m, _ in measured], dtype=float)
    bs = np.asarray([b for _, b in measured], dtype=float)
    log_bs = np.log(np.clip(bs, 1e-7, None))

    # Ordinary least squares: logB = slope*m + intercept
    A = np.vstack([ms, np.ones_like(ms)]).T
    sol, *_ = np.linalg.lstsq(A, log_bs, rcond=None)
    slope, intercept = float(sol[0]), float(sol[1])

    # 3. Predict B for entire keyboard
    all_midis = np.arange(MIDI_LOW, MIDI_HIGH + 1, dtype=float)
    logb_pred = slope * all_midis + intercept
    b_pred = np.exp(logb_pred)
    b_pred = np.clip(b_pred, 1e-6, 0.5)

    # 4. Heuristic stretch from B + distance from A4 (captures the spirit of interval-weighted beat-rate)
    curve: list[float] = [0.0] * N_KEYS
    for i in range(N_KEYS):
        m = MIDI_LOW + i
        b = float(b_pred[i])
        x = (m - A4_MIDI) / 12.0  # octaves from A4

        # Magnitude scaled by B (more inharmonic strings want more stretch)
        # Bass negative, treble positive — constants chosen for musically plausible range
        # (~ ±25 ¢ extremes on real pianos with typical B variation)
        if m <= A4_MIDI:
            mag = 520.0 * math.log10(1.0 + 480.0 * b)
            curve[i] = -mag * (abs(x) ** 1.75)
        else:
            mag = 380.0 * math.log10(1.0 + 300.0 * b)
            curve[i] = mag * (x**1.55)

    # 5. Hard pin A4 and smooth a little around center
    curve[A4_IDX] = 0.0
    for j in range(max(0, A4_IDX - 2), min(N_KEYS, A4_IDX + 3)):
        if abs(j - A4_IDX) <= 1:
            curve[j] *= 0.3  # pull neighbors toward zero

    # 6. Apply Shah-Välimäki treble rule (strong in top octave)
    _shah_valimaki_adjust(curve, b_pred, all_midis)

    # Final sanity clip + A4 guarantee
    curve = [float(np.clip(c, -35.0, 35.0)) for c in curve]
    curve[A4_IDX] = 0.0

    # Final treble safeguard: ensure top end stays musically positive (Shah intent)
    for i in range(72, N_KEYS):
        if curve[i] < 0.5:
            curve[i] = 0.5 + 0.6 * (i - 72)
    curve = [float(np.clip(c, -35.0, 35.0)) for c in curve]

    return curve


def apply_curve_to_piano(piano: Piano, curve: Sequence[float] | None = None) -> None:
    """Convenience (legacy path): attach heuristic curve (for comparison use only).

    NOTE: The package-level `apply_curve_to_piano` (re-exported from beat_rate)
    uses the improved solver.  This version forces the old heuristic.
    """
    if curve is None:
        curve = compute_heuristic_stretch_curve(piano)
    piano.tuning_curve = list(curve)
    for k in piano.keys.values():
        k.target_offset_cents = piano.get_target_offset(k.midi)
