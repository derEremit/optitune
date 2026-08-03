"""
TDD for inharmonic comb-filter note recognition (spec §3.6).

Synthetic tones with attenuated fundamentals must still classify to the true key.
"""

from __future__ import annotations

import numpy as np
import pytest

from optitune.dsp.note_recognizer import (
    MIDI_HIGH,
    MIDI_LOW,
    comb_template,
    recognize_from_audio,
    typical_b,
)
from optitune.dsp.synth import generate_inharmonic_tone, midi_to_hz


def _tone(
    midi: int,
    *,
    b: float | None = None,
    detune: float = 0.0,
    duration: float = 1.2,
    snr_db: float | None = None,
    with_hammer: bool = False,
    fund_atten_db: float = 0.0,
    seed: int = 7,
) -> tuple[np.ndarray, int]:
    fs = 48000
    bb = typical_b(midi) if b is None else b
    y = generate_inharmonic_tone(
        midi,
        detune_cents=detune,
        B=bb,
        duration=duration,
        fs=fs,
        snr_db=snr_db,
        with_hammer=with_hammer,
        seed=seed,
    )
    if fund_atten_db < 0:
        # Attenuate the fundamental by subtracting a pure sine at f0 (approx)
        f0 = midi_to_hz(midi) * (2.0 ** (detune / 1200.0))
        t = np.arange(len(y)) / fs
        # Estimate fund amp via projection, then reduce it
        s = np.sin(2 * np.pi * f0 * t)
        c = np.cos(2 * np.pi * f0 * t)
        a_s = 2.0 * np.dot(y, s) / len(y)
        a_c = 2.0 * np.dot(y, c) / len(y)
        fund = a_s * s + a_c * c
        gain = 10.0 ** (fund_atten_db / 20.0)
        y = y - fund * (1.0 - gain)
    return y.astype(np.float64), fs


def test_typical_b_increases_with_midi():
    assert typical_b(21) < typical_b(60) < typical_b(108)
    assert 1e-6 < typical_b(60) < 0.05


def test_comb_template_unit_norm_and_peaks_at_partials():
    tmpl = comb_template(60)
    assert tmpl.shape == (12000,)
    assert abs(np.linalg.norm(tmpl) - 1.0) < 1e-6
    # Template energy should be positive near f0 bin
    from optitune.dsp.binning import bin_index

    m0 = int(bin_index(midi_to_hz(60)))
    assert tmpl[m0] > 0.01


@pytest.mark.parametrize("midi", [24, 29, 36, 48, 60, 72, 84])
def test_recognize_clean_synth_midis(midi: int):
    y, fs = _tone(midi, duration=1.0, seed=midi * 3)
    # Use middle portion (skip any edge)
    seg = y[int(0.15 * len(y)) : int(0.85 * len(y))]
    match = recognize_from_audio(seg, fs)
    assert match is not None, f"no match for MIDI {midi}"
    assert match.midi == midi, f"got {match.midi}, expected {midi} (score={match.score:.3f})"


@pytest.mark.parametrize("midi", [24, 28, 33, 36])
def test_weak_fundamental_still_classifies(midi: int):
    """
    Structural fix for octave errors: fundamental attenuated 20 dB must still
    score highest at the true key (partials 2-4 carry the energy).
    """
    y, fs = _tone(midi, duration=1.5, fund_atten_db=-20.0, seed=100 + midi)
    seg = y[int(0.2 * len(y)) : int(0.9 * len(y))]
    match = recognize_from_audio(seg, fs)
    assert match is not None
    assert match.midi == midi, (
        f"weak-fund MIDI {midi}: got {match.midi} (score={match.score:.3f}, conf={match.confidence:.3f})"
    )


def test_prior_is_soft_tiebreak_not_override():
    """A wrong prior must not force a clearly different note."""
    y, fs = _tone(60, duration=1.0, seed=42)
    seg = y[int(0.2 * len(y)) : int(0.8 * len(y))]
    # Prior on a distant wrong note with small boost - still classify as C4
    match = recognize_from_audio(seg, fs, prior_midi=36)
    assert match is not None
    assert match.midi == 60


def test_search_window_limits_candidates():
    y, fs = _tone(72, duration=0.8, seed=9)
    seg = y[int(0.2 * len(y)) : int(0.8 * len(y))]
    match = recognize_from_audio(seg, fs, midi_lo=70, midi_hi=74)
    assert match is not None
    assert 70 <= match.midi <= 74


def test_silence_returns_none():
    fs = 48000
    y = np.zeros(fs, dtype=np.float64)
    match = recognize_from_audio(y, fs)
    assert match is None


def test_midi_range_constants():
    assert MIDI_LOW == 21
    assert MIDI_HIGH == 108
