"""TDD: Solver protocol, TuningConstraints, TuningCurve, BeatRateSolver adapter."""

from __future__ import annotations

import numpy as np
import pytest

from optitune.model import Key, Piano
from optitune.solvers import (
    BeatRateSolver,
    Solver,
    TuningConstraints,
    TuningCurve,
    compute_basic_tuning_curve,
)
from optitune.solvers.base import MIDI_HIGH, MIDI_LOW, N_KEYS


def test_tuning_curve_shape_and_a4_pin():
    offs = np.zeros(N_KEYS, dtype=float)
    offs[0] = -12.0
    offs[A4_IDX := 69 - MIDI_LOW] = 0.0
    offs[-1] = 8.0
    tc = TuningCurve(offsets_cents=offs, solver_name="test", metadata={"seed": 1})
    assert tc.n_keys == N_KEYS
    assert tc.offset_for_midi(21) == pytest.approx(-12.0)
    assert tc.offset_for_midi(69) == pytest.approx(0.0)
    assert tc.offset_for_midi(108) == pytest.approx(8.0)
    assert tc.solver_name == "test"
    assert tc.metadata["seed"] == 1


def test_tuning_constraints_defaults():
    c = TuningConstraints()
    assert c.a4 == 440.0
    assert c.temperament == "equal"
    assert c.treble_rule == "1:2"
    assert c.locked_notes == {}
    assert isinstance(c.interval_weights, dict)


def test_beat_rate_solver_is_solver_protocol():
    s: Solver = BeatRateSolver()
    assert s.name == "beat-rate"
    assert hasattr(s, "solve")


def test_beat_rate_solver_matches_compute_basic_tuning_curve():
    p = Piano(name="proto", a4=440.0)
    # Enough measured B for the LS path
    for midi, b in ((36, 1.2e-4), (48, 2.0e-4), (60, 4.0e-4), (72, 1.0e-3), (84, 4.0e-3)):
        p.set_key(Key(midi=midi, measured_f0=440.0 * 2 ** ((midi - 69) / 12), measured_b=b))

    legacy = compute_basic_tuning_curve(p)
    solver = BeatRateSolver()
    constraints = TuningConstraints(a4=p.a4)
    # Empty spectra + B from piano — beat-rate uses B only today
    K = N_KEYS
    cent_spectra = np.zeros((K, 8), dtype=float)
    b_est = np.full(K, np.nan, dtype=float)
    for m, k in p.keys.items():
        if k.measured_b is not None:
            b_est[m - MIDI_LOW] = float(k.measured_b)

    curves = list(solver.solve(cent_spectra, b_est, constraints))
    assert len(curves) >= 1
    final = curves[-1]
    assert final.n_keys == 88
    assert final.offset_for_midi(69) == pytest.approx(0.0, abs=0.05)
    np.testing.assert_allclose(final.offsets_cents, np.asarray(legacy), atol=1e-6)


def test_beat_rate_solver_from_piano_helper():
    p = Piano()
    solver = BeatRateSolver()
    tc = solver.solve_piano(p)
    assert len(tc.offsets_cents) == 88
    assert tc.offset_for_midi(69) == pytest.approx(0.0, abs=0.05)
