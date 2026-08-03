"""TDD: entropy minimizer (spec §5) behind Solver protocol."""

from __future__ import annotations

import numpy as np
import pytest

from optitune.dsp.binning import N_BINS, bin_index
from optitune.solvers import EntropySolver, TuningConstraints
from optitune.solvers.base import A4_MIDI, MIDI_LOW, N_KEYS


def _partial_comb_row(midi: int, *, a4: float = 440.0, b: float = 0.0003, n_max: int = 8) -> np.ndarray:
    """Synthetic A-weighted-like comb row: Gaussians at Fletcher-Young partials."""
    L = np.zeros(N_BINS, dtype=np.float64)
    f0 = a4 * (2.0 ** ((midi - 69) / 12.0))
    for n in range(1, n_max + 1):
        fn = n * f0 * np.sqrt(1.0 + b * n * n)
        if fn >= 9800:
            break
        m0 = int(bin_index(fn))
        amp = 1.0 / (n**1.1)
        for dm in range(-20, 21):
            m = m0 + dm
            if 0 <= m < N_BINS:
                L[m] += amp * np.exp(-(dm * dm) / (2 * 8.0**2))
    return L


def _synthetic_piano_spectra(*, stretch_cents: np.ndarray | None = None) -> np.ndarray:
    """
    88 rows of partial combs. Optional stretch_cents shifts each row so a
    detuned piano needs realignment (entropy should pull them together).
    """
    L = np.zeros((N_KEYS, N_BINS), dtype=np.float64)
    for i in range(N_KEYS):
        midi = MIDI_LOW + i
        row = _partial_comb_row(midi)
        if stretch_cents is not None:
            shift = int(round(float(stretch_cents[i])))
            if shift != 0:
                row = np.roll(row, shift)
        L[i] = row
    # Mild energy normalization
    for i in range(N_KEYS):
        s = L[i].sum()
        if s > 0:
            L[i] /= s
    return L


def test_entropy_solver_name_and_deterministic():
    L = _synthetic_piano_spectra()
    b = np.full(N_KEYS, np.nan)
    c = TuningConstraints(a4=440.0)
    s = EntropySolver(seed=7, max_passes=2, step_cents=1)
    curves1 = list(s.solve(L, b, c))
    s2 = EntropySolver(seed=7, max_passes=2, step_cents=1)
    curves2 = list(s2.solve(L, b, c))
    assert s.name == "entropy"
    assert len(curves1) >= 1
    np.testing.assert_array_equal(curves1[-1].offsets_cents, curves2[-1].offsets_cents)


def test_entropy_solver_pins_a4():
    L = _synthetic_piano_spectra()
    b = np.full(N_KEYS, np.nan)
    tc = next(EntropySolver(seed=1, max_passes=3).solve(L, b, TuningConstraints()))
    assert tc.offset_for_midi(A4_MIDI) == pytest.approx(0.0, abs=0.01)


def test_entropy_aligns_two_detuned_copies():
    """
    Two keys with the same spectrum shape but opposite detune should converge
    toward mutual alignment (relative offset shrinks).
    """
    # Minimal 2-key problem embedded in 88: C4 and G4 with shared comb shape
    # shifted ±15¢
    L = np.zeros((N_KEYS, N_BINS), dtype=np.float64)
    base = _partial_comb_row(60)
    L[60 - MIDI_LOW] = np.roll(base, -15)  # flat
    L[67 - MIDI_LOW] = np.roll(base, +15)  # sharp (reuse shape for pure alignment)
    # Scale
    for i in (60 - MIDI_LOW, 67 - MIDI_LOW):
        s = L[i].sum()
        if s > 0:
            L[i] /= s
    b = np.full(N_KEYS, np.nan)
    # Only optimize keys that have energy
    solver = EntropySolver(seed=3, max_passes=30, step_cents=1, active_only=True)
    tc = next(solver.solve(L, b, TuningConstraints()))
    c1 = tc.offset_for_midi(60)
    c2 = tc.offset_for_midi(67)
    # Relative misalignment should shrink from 30¢ toward 0
    assert abs(c2 - c1) < 20.0, f"relative still large: {c1}, {c2}"


def test_entropy_railsback_shape_on_realistic_combs():
    """
    With a Railsback prior, the final curve stays musically sane (A4 pin,
    no wild extremes). Pure entropy on ET combs is underdetermined; the prior
    supplies the Railsback envelope.
    """
    L = _synthetic_piano_spectra()
    b = np.full(N_KEYS, np.nan)
    solver = EntropySolver(seed=11, max_passes=4, step_cents=1, railsback_prior=0.4)
    # take final curve
    tc = list(solver.solve(L, b, TuningConstraints()))[-1]
    offs = tc.offsets_cents
    assert offs[A4_MIDI - MIDI_LOW] == pytest.approx(0.0, abs=0.01)
    # Prior-dominated: bass not sharp, treble not flat
    assert offs[0] < 0.0
    assert offs[-1] > 0.0
    assert abs(offs[0]) < 40.0
    assert abs(offs[-1]) < 40.0
