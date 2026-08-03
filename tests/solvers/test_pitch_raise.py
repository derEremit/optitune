"""Pitch-raise / overpull profile (Rigaud-style mean octave-type model)."""

from __future__ import annotations

import numpy as np
import pytest

from optitune.solvers.base import A4_MIDI, MIDI_LOW, N_KEYS
from optitune.solvers.pitch_raise import overpull_profile, pitch_raise_targets


def test_overpull_above_final_and_tapers_treble():
    final = np.linspace(-12.0, 8.0, N_KEYS)
    # Piano measured ~30¢ flat of final targets
    measured_dev = np.full(N_KEYS, -30.0)
    over = overpull_profile(measured_dev, final, variant="high")
    assert over.shape == (N_KEYS,)
    # Overpull targets sit above final (more sharp) while piano is flat
    assert float(np.mean(over - final)) > 0.0
    # Treble overpull amount less than bass (tapers)
    bass_pull = float(over[10] - final[10])
    treble_pull = float(over[80] - final[80])
    assert bass_pull > treble_pull


def test_pitch_raise_targets_a4_pin():
    final = np.zeros(N_KEYS)
    final[0] = -10
    final[-1] = 5
    measured = np.full(N_KEYS, -25.0)
    targets = pitch_raise_targets(measured, final, variant="low")
    assert targets[A4_MIDI - MIDI_LOW] == pytest.approx(0.0, abs=0.05)
