"""
Lightweight STFT wrapper for synthetic tone analysis and round-trip tests.

Uses Blackman-Harris (4-term) as required by matrix contract and spec §3.3.
Falls back gracefully if pyfftw not present (tests use scipy/numpy).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import get_window
from scipy.signal import stft as scipy_stft


def compute_stft(
    y: np.ndarray,
    fs: float = 48000.0,
    n_fft: int = 32768,
    hop: int = 8192,
    window: str = "blackmanharris",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (freqs, times, power_frames) where power_frames.shape = (n_frames, n_freqs)
    power = |STFT|^2 using the exact window required for matrix roundtrips.
    """
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))

    win = get_window(window, n_fft, fftbins=True)
    f, t, Zxx = scipy_stft(
        y,
        fs=fs,
        window=win,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        return_onesided=True,
        padded=False,
        boundary=None,
    )
    power = np.abs(Zxx) ** 2
    return f, t, power.T  # (frames, bins)


def get_central_frame_power(
    y: np.ndarray, fs: float = 48000.0, n_fft: int = 32768
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: return (freqs, power) of the middle frame for peak analysis."""
    f, t, P = compute_stft(y, fs=fs, n_fft=n_fft)
    if P.shape[0] == 0:
        return f, np.zeros_like(f)
    mid = P.shape[0] // 2
    return f, P[mid]
