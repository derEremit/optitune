"""FFT backend matches NumPy on synthetic data."""

from __future__ import annotations

import numpy as np

from optitune.dsp.fft_backend import clear_plan_cache, fft_backend_name, rfft, rfftfreq


def test_backend_name_is_known():
    assert fft_backend_name() in ("numpy", "pyfftw")


def test_rfft_matches_numpy():
    clear_plan_cache()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2048)
    y = rfft(x)
    y_np = np.fft.rfft(x)
    np.testing.assert_allclose(np.abs(y), np.abs(y_np), rtol=1e-6, atol=1e-8)
    # Freqs identical
    np.testing.assert_allclose(rfftfreq(2048, 1 / 48000), np.fft.rfftfreq(2048, 1 / 48000))


def test_rfft_short_buffer():
    x = np.ones(128)
    y = rfft(x)
    assert y.shape == (65,)
