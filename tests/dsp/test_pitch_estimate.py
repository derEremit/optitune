"""
TDD for the pure live pitch estimator (dsp/pitch_estimate.py).

Qt-free: synthetic tones + optional real_piano fixtures.
Armed MIDI is a soft prior for f0 (not a hard override of a wrong note).
"""

from __future__ import annotations

import numpy as np
import pytest

from optitune.dsp.peaks import cents
from optitune.dsp.pitch_estimate import estimate_pitch
from optitune.dsp.synth import generate_inharmonic_tone, midi_to_hz


def _synth(
    midi: int,
    *,
    detune: float = 0.0,
    b: float = 0.0003,
    duration: float = 1.2,
    fund_atten_db: float = 0.0,
    seed: int = 11,
) -> tuple[np.ndarray, int]:
    fs = 48000
    y = generate_inharmonic_tone(
        midi,
        detune_cents=detune,
        B=b,
        duration=duration,
        fs=fs,
        with_hammer=False,
        seed=seed,
    )
    if fund_atten_db < 0:
        f0 = midi_to_hz(midi) * (2.0 ** (detune / 1200.0))
        t = np.arange(len(y)) / fs
        s = np.sin(2 * np.pi * f0 * t)
        c = np.cos(2 * np.pi * f0 * t)
        a_s = 2.0 * np.dot(y, s) / len(y)
        a_c = 2.0 * np.dot(y, c) / len(y)
        fund = a_s * s + a_c * c
        gain = 10.0 ** (fund_atten_db / 20.0)
        y = y - fund * (1.0 - gain)
    # Use steady middle of the tone
    n = len(y)
    return y[int(0.2 * n) : int(0.85 * n)].astype(np.float64), fs


@pytest.mark.parametrize("midi", [24, 36, 48, 60, 69, 72])
def test_clean_synth_recovers_midi_and_f0(midi: int):
    y, fs = _synth(midi, seed=midi * 5)
    est = estimate_pitch(y, fs, a4=440.0)
    assert est["midi"] == midi, f"got midi={est['midi']} f_est={est['f_est']:.2f}"
    true_f0 = midi_to_hz(midi)
    assert abs(cents(est["f_est"], true_f0)) < 25.0


def test_flat_c1_with_armed_prior_identity_is_c1_not_b0():
    """Detuned flat C1 must not snap identity to B0 when armed on C1."""
    y, fs = _synth(24, detune=-60.0, seed=3)
    est = estimate_pitch(y, fs, a4=440.0, armed_midi=24)
    assert est["midi"] == 24
    # f_est should still reflect the flat pitch (~-60 ¢)
    true_flat = midi_to_hz(24) * (2.0 ** (-60.0 / 1200.0))
    assert abs(cents(est["f_est"], true_flat)) < 40.0


def test_weak_fundamental_bass_with_armed_prior():
    """Armed prior must recover true f0 when fundamental is attenuated 20 dB."""
    y, fs = _synth(24, fund_atten_db=-20.0, duration=1.5, seed=99)
    est = estimate_pitch(y, fs, a4=440.0, armed_midi=24)
    assert est["midi"] == 24
    assert abs(cents(est["f_est"], midi_to_hz(24))) < 80.0


def test_dominant_upper_partial_with_armed_prior_folds_to_fundamental():
    """
    When spectrum energy is dominated by partial ~4 of the armed note
    (classic live bass failure: report C3/C5 while armed on C1/C2), fold to f0.
    """
    midi = 36  # C2
    f0 = midi_to_hz(midi)
    fs = 48000
    t = np.arange(int(fs * 1.0)) / fs
    # Strong 4th partial + weak fundamental (simulates hammer/decay spectrum)
    y = 0.15 * np.sin(2 * np.pi * f0 * t) + 1.0 * np.sin(2 * np.pi * (4 * f0) * t)
    y += 0.5 * np.sin(2 * np.pi * (2 * f0) * t)
    y += 0.35 * np.sin(2 * np.pi * (3 * f0) * t)
    est = estimate_pitch(y, fs, a4=440.0, armed_midi=midi)
    assert est["midi"] == midi
    assert abs(cents(est["f_est"], f0)) < 100.0, f"f_est={est['f_est']:.1f} want ~{f0:.1f}"


def test_armed_prior_does_not_force_wrong_pitch_class():
    """Playing a clearly different note while armed must not hard-force armed midi."""
    y, fs = _synth(60, seed=7)  # C4 sounding
    est = estimate_pitch(y, fs, a4=440.0, armed_midi=24)  # armed on C1
    # Must not claim C1 just because armed — f_est is near 261 Hz
    assert abs(cents(est["f_est"], midi_to_hz(60))) < 40.0
    # Identity may follow f0 (C4) rather than armed C1
    assert est["midi"] == 60


def test_last_f0_guess_continuity_without_armed():
    y, fs = _synth(69, seed=1)
    est = estimate_pitch(y, fs, a4=440.0, last_f0_guess=430.0)
    assert est["midi"] == 69
