"""Per-key cent-binned spectrum storage (entropy-solver input)."""

from __future__ import annotations

import numpy as np
import pytest

from optitune.dsp.binning import N_BINS, bin_index
from optitune.dsp.synth import generate_inharmonic_tone, midi_to_hz
from optitune.model import Key, Piano
from optitune.model.spectrum_codec import pack_spectrum, unpack_spectrum, spectrum_from_audio_a_weighted


def test_pack_unpack_roundtrip():
    # Sparse spectrum compresses well; random does not (expected)
    x = np.zeros(N_BINS, dtype=np.float32)
    x[100:110] = 0.5
    x[2000:2010] = 0.2
    s = pack_spectrum(x)
    assert isinstance(s, str)
    assert len(s) < N_BINS  # much smaller than raw float32
    y = unpack_spectrum(s)
    assert y.shape == (N_BINS,)
    np.testing.assert_allclose(y, x, rtol=1e-5, atol=1e-8)


def test_key_spectrum_json_roundtrip():
    L = np.zeros(N_BINS, dtype=np.float32)
    L[1000:1010] = 1.0
    k = Key(midi=60, measured_f0=261.6, measured_b=0.0004, cent_spectrum=L)
    d = k.to_dict()
    assert "cent_spectrum" in d
    assert isinstance(d["cent_spectrum"], str)
    k2 = Key.from_dict(d)
    assert k2.cent_spectrum is not None
    np.testing.assert_allclose(k2.cent_spectrum, L, atol=1e-6)


def test_piano_spectra_matrix():
    p = Piano()
    L60 = np.zeros(N_BINS, dtype=np.float32)
    L60[500] = 2.0
    p.set_key(Key(midi=60, measured_f0=261.0, measured_b=0.0003, cent_spectrum=L60))
    p.set_key(Key(midi=72, measured_f0=523.0, measured_b=0.0008))  # no spectrum
    mat = p.cent_spectra_matrix()
    assert mat.shape == (88, N_BINS)
    assert mat[60 - 21, 500] == pytest.approx(2.0)
    assert np.all(mat[72 - 21] == 0.0)


def test_spectrum_from_synthetic_tone_peaks_on_partials():
    """Argmax energy bins near Fletcher-Young partials of the tone."""
    midi = 48  # C3
    fs = 48000.0
    y = generate_inharmonic_tone(midi, duration=1.2, fs=int(fs), a4=440.0, B=0.0003, seed=1)
    # skip attack
    seg = y[int(0.2 * fs) : int(1.0 * fs)]
    L = spectrum_from_audio_a_weighted(seg, fs)
    assert L.shape == (N_BINS,)
    assert float(np.max(L)) > 0

    f0 = midi_to_hz(midi, 440.0)
    # First 3 partials should land near strong bins
    for n in (1, 2, 3):
        fn = n * f0 * np.sqrt(1.0 + 0.0003 * n * n)
        m = int(bin_index(float(fn)))
        window = L[max(0, m - 25) : min(N_BINS, m + 26)]
        assert float(np.max(window)) > 0.05 * float(np.max(L)), f"partial {n} weak at bin {m}"
