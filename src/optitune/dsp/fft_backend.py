"""
FFT backend: pyfftw when available, NumPy otherwise.

Plan cache keyed by (n, dtype) for repeated live frames of fixed length.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_BACKEND = "numpy"
_pyfftw: Any = None
_plan_cache: dict[tuple[int, str], Any] = {}

try:
    import pyfftw  # type: ignore[import-untyped]

    pyfftw.interfaces.cache.enable()
    _pyfftw = pyfftw
    _BACKEND = "pyfftw"
except Exception:
    _pyfftw = None
    _BACKEND = "numpy"


def fft_backend_name() -> str:
    return _BACKEND


def rfft(x: np.ndarray) -> np.ndarray:
    """Real FFT of 1-D array (complex spectrum)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.ravel()
    n = int(arr.shape[0])
    if _BACKEND == "pyfftw" and _pyfftw is not None and n >= 256:
        key = (n, "float64")
        plan = _plan_cache.get(key)
        if plan is None:
            a = _pyfftw.empty_aligned(n, dtype="float64")
            b = _pyfftw.empty_aligned(n // 2 + 1, dtype="complex128")
            plan = _pyfftw.FFTW(a, b, flags=("FFTW_MEASURE",), planning_timelimit=0.5)
            _plan_cache[key] = plan
        # pyfftw plans write into their input array
        plan.input_array[:] = arr
        return np.asarray(plan(), dtype=np.complex128)
    return np.fft.rfft(arr)


def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    return np.fft.rfftfreq(int(n), d=float(d))


def clear_plan_cache() -> None:
    _plan_cache.clear()
