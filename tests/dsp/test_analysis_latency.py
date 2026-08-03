"""
Latency / CPU budget for live analysis frames (M6).

Asserts estimate_pitch on typical live frame sizes finishes within a generous
CI budget. Marked non-brittle: thresholds are soft upper bounds for CI hardware.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from optitune.dsp.fft_backend import fft_backend_name
from optitune.dsp.pitch_estimate import estimate_pitch
from optitune.dsp.synth import generate_inharmonic_tone


def _median_ms(fn, n: int = 5) -> float:
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


@pytest.mark.parametrize(
    "n_samples,budget_ms",
    [
        (8192, 80.0),  # short frame
        (32768, 120.0),  # typical live window
        (65536, 250.0),  # bass long frame (generous for CI)
    ],
)
def test_estimate_pitch_latency_budget(n_samples: int, budget_ms: float) -> None:
    fs = 48000
    y = generate_inharmonic_tone(48, duration=2.0, fs=fs, B=0.0004, seed=7)
    # Take a contiguous segment of requested length
    seg = y[:n_samples] if len(y) >= n_samples else np.pad(y, (0, n_samples - len(y)))

    def once() -> None:
        est = estimate_pitch(seg, float(fs), a4=440.0, armed_midi=48)
        assert est["f_est"] > 20

    # Warm-up (plan build for pyfftw)
    once()
    med = _median_ms(once, n=4)
    # Soft gate — fail only if egregiously slow
    assert med < budget_ms, (
        f"median {med:.1f} ms exceeds {budget_ms} ms budget for n={n_samples} "
        f"(backend={fft_backend_name()})"
    )


def test_continuous_feed_simulation_stays_under_budget() -> None:
    """Simulate ~1 s of 10 Hz analysis ticks on streaming audio."""
    fs = 48000
    y = generate_inharmonic_tone(60, duration=1.5, fs=fs, B=0.0003, seed=3)
    hop = fs // 10  # 100 ms
    n = 32768
    t0 = time.perf_counter()
    count = 0
    for start in range(0, len(y) - n, hop):
        estimate_pitch(y[start : start + n], float(fs), a4=440.0)
        count += 1
        if count >= 8:
            break
    elapsed = time.perf_counter() - t0
    # 8 frames should finish well under wall-clock of a second
    assert count >= 8
    assert elapsed < 2.0, f"8 analysis frames took {elapsed:.2f}s"
