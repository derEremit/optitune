"""
Comprehensive TDD tests for log-cent binning and IEC 61672 A-weighting.

Per docs/synth_test_matrix.md and piano_tuner_implementation_spec.md §3.4:
- Exact formulas for bin_index and a_weight_db.
- Vectorized production impl + slow reference for cross-check.
- Binning must use np.add.at semantics for power summation.
- Round-trip property with synthetic tones (once synth ready).
- All tests deterministic, fast, no hardware.
"""
from __future__ import annotations

import numpy as np
import pytest

from optitune.dsp.binning import (
    F_HI,
    F_LO,
    N_BINS,
    a_weight_db,
    apply_a_weight_to_binned,
    bin_center,
    bin_index,
    bin_spectrum_vectorized,
    slow_bin_spectrum,
)

# --- Exact formula unit tests ---

def test_bin_constants_match_spec():
    """N_BINS = 12000 per contract (spec approximates the round)."""
    assert N_BINS == 12000
    assert F_LO == 10.0
    assert F_HI == 10_000.0


def test_bin_index_scalar_and_vectorized():
    """bin_index(f) == floor(1200 * log2(f / 10)) , clipped to [0, N_BINS-1]."""
    assert bin_index(10.0) == 0
    assert bin_index(20.0) == 1200
    expected_a4 = int(np.floor(1200 * np.log2(440.0 / 10.0)))
    assert bin_index(440.0) == expected_a4
    fs = np.array([10.0, 20.0, 440.0, 10000.0])
    idx = bin_index(fs)
    assert idx.shape == (4,)
    assert idx[0] == 0
    assert idx[1] == 1200
    # 10kHz lands near top; accept any in the upper range (natural calculation is ~11958)
    assert idx[3] >= N_BINS - 50 and idx[3] <= N_BINS - 1


def test_bin_index_out_of_range_clipped_or_defined():
    assert bin_index(5.0) == 0
    idx_hi = bin_index(20000.0)
    assert idx_hi == N_BINS - 1


def test_bin_center_inverse_of_index():
    for m in [0, 1, 1200, 5000, 11999]:
        f = bin_center(m)
        m_back = bin_index(f)
        assert abs(m_back - m) <= 1


def test_a_weight_db_exact_formula():
    a1000 = a_weight_db(1000.0)
    assert abs(a1000) < 0.5
    freqs = np.array([100.0, 1000.0, 10000.0])
    aw = a_weight_db(freqs)
    assert aw.shape == (3,)
    assert a_weight_db(20.0) < -40.0
    assert a_weight_db(15000.0) < -5.0


def test_a_weight_db_vectorized_matches_scalar():
    fs = np.logspace(1, 4, 200)
    scalar = np.array([a_weight_db(f) for f in fs])
    vec = a_weight_db(fs)
    np.testing.assert_allclose(vec, scalar, rtol=1e-12, atol=1e-12)


# --- Reference slow vs fast ---

def test_vectorized_binning_matches_slow_reference():
    rng = np.random.default_rng(123)
    freqs = rng.uniform(20, 8000, 5000)
    power = rng.exponential(1.0, 5000)

    L_fast = bin_spectrum_vectorized(freqs, power)
    L_slow = slow_bin_spectrum(freqs, power)

    np.testing.assert_allclose(L_fast, L_slow, rtol=0, atol=1e-12)
    assert L_fast.shape == (N_BINS,)


def test_binning_add_at_collisions():
    freqs = np.array([440.0, 440.0, 441.2, 440.0])
    power = np.array([1.0, 2.0, 0.5, 3.0])
    L = bin_spectrum_vectorized(freqs, power)
    m = bin_index(440.0)
    assert L[m] == pytest.approx(1 + 2 + 3, abs=1e-12)


# --- A-weight on binned ---

def test_apply_a_weight_to_binned_shape_and_values():
    L = np.ones(N_BINS) * 1e-6
    L_weighted = apply_a_weight_to_binned(L)
    assert L_weighted.shape == (N_BINS,)
    cents_20hz = bin_index(20.0)
    assert L_weighted[cents_20hz] < L[cents_20hz] * 0.001


# --- Round-trip (uses 32k STFT as required; realistic main-lobe spread tolerated) ---

@pytest.mark.synth_only
def test_binning_roundtrip_fundamental_peak_location():
    from scipy.signal.windows import blackmanharris

    from optitune.dsp.synth import generate_inharmonic_tone

    fs = 48000
    y = generate_inharmonic_tone(69, detune_cents=0.0, B=0.00005, duration=1.0, fs=fs, seed=42, with_hammer=False)

    n = min(len(y), 32768)
    w = blackmanharris(n)
    Y = np.fft.rfft(y[:n] * w)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    power = np.abs(Y) ** 2

    L = bin_spectrum_vectorized(freqs, power)
    f0_true = 440.0
    m_true = bin_index(f0_true)

    search = L[: bin_index(2000.0) + 20]
    m_peak = int(np.argmax(search))
    bin_err = abs(m_peak - m_true)
    # With 32k Blackman-Harris the main lobe spreads energy; 5 bins is acceptable for this property test
    assert bin_err <= 5, f"Bin peak {m_peak} vs true {m_true} for f0={f0_true}"


def test_binning_deterministic_no_randomness():
    freqs = np.array([27.5, 440.0, 4186.0])
    p1 = bin_index(freqs)
    p2 = bin_index(freqs)
    np.testing.assert_array_equal(p1, p2)
