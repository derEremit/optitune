"""
Compress / restore cent-binned spectra for Key persistence (spec §4.4 style).

float32 + zlib + base64 keeps JSON autosave manageable (~few KB per key
instead of ~100 KB raw).
"""

from __future__ import annotations

import base64
import zlib

import numpy as np

from optitune.dsp.binning import N_BINS, apply_a_weight_to_binned, bin_spectrum_vectorized


def pack_spectrum(spectrum: np.ndarray) -> str:
    arr = np.asarray(spectrum, dtype=np.float32).reshape(-1)
    if arr.shape[0] != N_BINS:
        # Pad or trim to contract length
        out = np.zeros(N_BINS, dtype=np.float32)
        n = min(N_BINS, arr.shape[0])
        out[:n] = arr[:n]
        arr = out
    compressed = zlib.compress(arr.tobytes(), level=6)
    return base64.b64encode(compressed).decode("ascii")


def unpack_spectrum(blob: str | None) -> np.ndarray | None:
    if not blob:
        return None
    try:
        raw = zlib.decompress(base64.b64decode(blob.encode("ascii")))
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.shape[0] != N_BINS:
            out = np.zeros(N_BINS, dtype=np.float32)
            n = min(N_BINS, arr.shape[0])
            out[:n] = arr[:n]
            return out
        return np.array(arr, dtype=np.float32, copy=True)
    except Exception:
        return None


def spectrum_from_audio_a_weighted(audio: np.ndarray, fs: float) -> np.ndarray:
    """A-weighted cent-binned power spectrum (entropy-solver input row)."""
    x = np.asarray(audio, dtype=np.float64)
    n = len(x)
    if n < 256:
        return np.zeros(N_BINS, dtype=np.float32)
    # Pad toward longer frames for bass resolution
    target = max(n, min(int(fs * 1.5), 65536))
    if n < target:
        pad = np.zeros(target, dtype=np.float64)
        pad[:n] = x
        x = pad
        n = target
    w = np.hanning(n)
    spec = np.fft.rfft(x * w)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    L = bin_spectrum_vectorized(freqs, power)
    L_a = apply_a_weight_to_binned(L)
    return np.asarray(L_a, dtype=np.float32)
