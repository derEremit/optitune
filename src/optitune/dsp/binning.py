"""
Log-cent binning (10 Hz - 10 kHz @ 1 cent) + IEC 61672:2003 A-weighting.

Exact implementation of piano_tuner_implementation_spec.md §3.4.
We use N_BINS = 12000 as the contract (spec rounds to this value).
Vectorized production path (np.add.at) + slow reference for verification.
"""

from __future__ import annotations

import numpy as np

F_LO: float = 10.0
F_HI: float = 10_000.0
CENTS_PER_BIN: int = 1
N_BINS: int = 12000  # Contract value per synth_test_matrix.md and spec §3.4 (rounded)


def bin_index(f_hz: float | np.ndarray) -> int | np.ndarray:
    """
    f -> bin m = floor(1200 * log2(f / F_LO))
    Always returns valid index in [0, N_BINS-1].
    """
    f = np.asarray(f_hz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.floor(1200.0 * np.log2(np.maximum(f, F_LO) / F_LO))
    if np.isscalar(f_hz):
        return int(np.clip(m, 0, N_BINS - 1).item())
    return np.asarray(np.clip(m, 0, N_BINS - 1), dtype=int)


def bin_center(m: int | np.ndarray) -> float | np.ndarray:
    """Inverse: bin center frequency f(m) = 10 * 2**(m / 1200) Hz."""
    m_arr = np.asarray(m, dtype=float)
    f = 10.0 * 2.0 ** (m_arr / 1200.0)
    if np.isscalar(m):
        return float(f.item())
    return np.asarray(f)


def a_weight_db(f: float | np.ndarray) -> float | np.ndarray:
    """
    IEC 61672:2003 A-weighting in dB exactly as in spec §3.4.
    """
    f_arr = np.asarray(f, dtype=float)
    f2 = f_arr * f_arr
    num = (12200.0**2) * (f2**2)
    den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12200.0**2)
    Ra = num / np.maximum(den, 1e-300)
    aw = 2.0 + 20.0 * np.log10(np.maximum(Ra, 1e-300))
    if np.isscalar(f):
        return float(aw.item())
    return np.asarray(aw)


def bin_spectrum_vectorized(freqs: np.ndarray, power: np.ndarray) -> np.ndarray:
    """
    Production binning using np.add.at per spec.
    Clips indices safely.
    """
    L = np.zeros(N_BINS, dtype=np.float64)
    idx = np.asarray(bin_index(freqs), dtype=int)
    np.add.at(L, idx, power.astype(np.float64, copy=False))
    return L


def slow_bin_spectrum(freqs: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Pure Python reference for tests."""
    L = np.zeros(N_BINS, dtype=np.float64)
    idx = np.asarray(bin_index(freqs), dtype=int)
    for i, p in zip(idx, power, strict=False):
        if 0 <= int(i) < N_BINS:
            L[int(i)] += float(p)
    return L


def apply_a_weight_to_binned(L: np.ndarray) -> np.ndarray:
    """Apply A-weighting gain to already-binned power."""
    centers = bin_center(np.arange(N_BINS))
    aw_db = a_weight_db(centers)
    gain = 10.0 ** (aw_db / 10.0)
    return L * gain


def bin_and_a_weight(freqs: np.ndarray, power: np.ndarray) -> np.ndarray:
    """End-to-end vectorized bin + A-weight for a power spectrum."""
    L = bin_spectrum_vectorized(freqs, power)
    return apply_a_weight_to_binned(L)
