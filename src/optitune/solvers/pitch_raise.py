"""
Pitch-raise / overpull profiles (spec M3, Rigaud-style mean octave-type model).

Given measured cents-vs-ET (or vs final target) on a subset of keys and a final
tuning curve, produce temporary overpull targets that sit *above* the final
curve while the piano is flat, tapering toward the treble.
"""

from __future__ import annotations

import numpy as np

from optitune.solvers.base import A4_MIDI, MIDI_LOW, N_KEYS

# Relative overpull strength vs position (bass more, treble less)
# High variant: aggressive raise; low: conservative.
_VARIANT_SCALE = {"high": 0.55, "low": 0.35, "medium": 0.45}


def overpull_profile(
    measured_dev_cents: np.ndarray,
    final_curve_cents: np.ndarray,
    *,
    variant: str = "medium",
) -> np.ndarray:
    """
    measured_dev_cents: (88,) how flat/sharp the piano is vs ET (or vs final).
        Negative = flat (typical pitch-raise starting point).
    final_curve_cents: (88,) desired final stretch curve.
    Returns temporary overpull target curve (cents vs ET).
    """
    meas = np.asarray(measured_dev_cents, dtype=float).reshape(-1)
    final = np.asarray(final_curve_cents, dtype=float).reshape(-1)
    if meas.shape[0] != N_KEYS or final.shape[0] != N_KEYS:
        raise ValueError(f"arrays must have length {N_KEYS}")

    scale = _VARIANT_SCALE.get(str(variant).lower(), 0.45)
    # Amount still flat of the final target
    remaining = final - meas  # positive if measured is flatter than final
    remaining = np.maximum(remaining, 0.0)

    # Taper: more overpull in bass, less in treble (index 0..87)
    x = np.linspace(0.0, 1.0, N_KEYS)
    taper = 1.0 - 0.65 * x  # 1.0 bass → 0.35 treble
    overpull_amt = scale * remaining * taper

    over = final + overpull_amt
    # Pin A4
    over[A4_MIDI - MIDI_LOW] = 0.0
    return over


def pitch_raise_targets(
    measured_dev_cents: np.ndarray,
    final_curve_cents: np.ndarray,
    *,
    variant: str = "medium",
) -> np.ndarray:
    """Alias used by the wizard / tests."""
    return overpull_profile(measured_dev_cents, final_curve_cents, variant=variant)
