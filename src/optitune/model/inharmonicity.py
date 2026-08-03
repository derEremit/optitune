"""
Inharmonicity (B) curve fitting for display and solvers.

Log-linear fit of measured B vs MIDI; evaluates on the full 88-key compass.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

MIDI_LOW = 21
MIDI_HIGH = 108
N_KEYS = 88


def fit_log_linear_b(
    measured: Mapping[int, float] | list[tuple[int, float]],
    *,
    midi_low: int = MIDI_LOW,
    midi_high: int = MIDI_HIGH,
) -> tuple[np.ndarray, float, float]:
    """
    Fit ln(B) = slope * midi + intercept from measured keys.

    Returns (b_pred_88, slope, intercept). With < 2 points, returns a flat
    typical mid-piano B (~3e-4).
    """
    if isinstance(measured, Mapping):
        pairs = [(int(m), float(b)) for m, b in measured.items() if b is not None and float(b) > 0]
    else:
        pairs = [(int(m), float(b)) for m, b in measured if b is not None and float(b) > 0]

    n = midi_high - midi_low + 1
    midis = np.arange(midi_low, midi_high + 1, dtype=float)

    if len(pairs) < 2:
        default_b = 3e-4
        return np.full(n, default_b, dtype=float), 0.0, float(np.log(default_b))

    ms = np.asarray([m for m, _ in pairs], dtype=float)
    bs = np.asarray([b for _, b in pairs], dtype=float)
    bs = np.clip(bs, 1e-8, 0.5)
    log_bs = np.log(bs)
    # Least-squares: log_b ≈ slope * m + intercept
    A = np.column_stack([ms, np.ones_like(ms)])
    coef, _, _, _ = np.linalg.lstsq(A, log_bs, rcond=None)
    slope = float(coef[0])
    intercept = float(coef[1])
    log_pred = slope * midis + intercept
    b_pred = np.exp(log_pred)
    b_pred = np.clip(b_pred, 1e-6, 0.5)
    return b_pred, slope, intercept


def measured_b_from_piano(piano: object) -> dict[int, float]:
    """Extract {midi: B} from a Piano-like object with .keys dict of Key."""
    out: dict[int, float] = {}
    keys = getattr(piano, "keys", {}) or {}
    for m, k in keys.items():
        b = getattr(k, "measured_b", None)
        if b is not None and 1e-8 < float(b) < 1.0:
            out[int(m)] = float(b)
    return out
