"""
TDD tests for parabolic interpolation (exact spec formula) and PFD-style (F0, B) estimator.

Covers requirements from synth_test_matrix.md + spec §3.5:
- Parabolic delta formula must match SMS-tools / spec exactly.
- PFD estimator recovers synthetic ground-truth f0 + B within matrix tolerances.
- Works for all 6 matrix conditions (perfect, high-B, noisy, bass, etc.).
- Deterministic, seeded inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from optitune.dsp.peaks import (
    cents,
    parabolic_interpolation,
    pfd_estimate_f0_b,
)


def test_parabolic_interpolation_exact_formula():
    """
    delta = 1/2 * (log|X_{k-1}| - log|X_{k+1}|) / (log|X_{k-1}| - 2*log|X_k| + log|X_{k+1}|)
    Must be bit-accurate to the formula (within fp).
    """
    logm = np.array([0.1, 1.0, 0.4])
    k = 1
    delta, _log_peak = parabolic_interpolation(logm, k)
    y1, y2, y3 = logm
    denom = y1 - 2 * y2 + y3
    expected_delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-12 else 0.0
    assert abs(delta - expected_delta) < 1e-12


def test_parabolic_interpolation_at_edges():
    """Must not crash on first/last bin; returns 0 delta."""
    logm = np.array([10.0, 20.0, 5.0])
    d0, _ = parabolic_interpolation(logm, 0)
    d2, _ = parabolic_interpolation(logm, 2)
    assert d0 == 0.0
    assert d2 == 0.0


def test_parabolic_recovers_ideal_quadratic():
    """
    For a perfectly parabolic log-mag (the assumption of the interpolator),
    recovery must be exact (within fp).
    """
    k = 40
    true_delta = 0.3
    logm = np.zeros(100, dtype=float)
    # Construct exact samples from the parabola that the formula solves for
    # The formula comes from assuming log|X(f)| quadratic in bin index
    # y(x) = c*(x - (k + d))**2 + peak  => the 3-pt formula recovers d exactly
    c = -4.0  # curvature
    for off in (-1, 0, 1):
        x = (k + off) - (k + true_delta)
        logm[k + off] = c * x * x

    delta_rec, _ = parabolic_interpolation(logm, k)
    assert abs(delta_rec - true_delta) < 1e-10


# --- PFD estimator tests on known (f0, B) ---


def make_synthetic_peak_list(
    f0: float, B: float, n_max: int = 12, noise_cents: float = 0.0, rng=None
):
    """Helper: return (peak_freqs, peak_amps) exactly from model + optional tiny freq jitter."""
    if rng is None:
        rng = np.random.default_rng(0)
    ns = np.arange(1, n_max + 1)
    fns = ns * f0 * np.sqrt(1.0 + B * ns**2)
    amps = np.exp(-0.15 * (ns - 1)) * 100.0
    if noise_cents > 0:
        jitter = 2 ** (rng.normal(0, noise_cents / 1200, len(fns)))
        fns = fns * jitter
    return fns, amps


@pytest.mark.parametrize(
    "f0,B",
    [
        (27.5, 0.00005),
        (261.626, 0.0008),
        (4186.01, 0.025),
        (27.5, 0.00012),
    ],
)
def test_pfd_recovers_clean_synthetic(f0, B):
    """On noise-free synthetic partial list, PFD must recover f0 to <<0.01 cent and B to <<1%."""
    peaks_f, peaks_a = make_synthetic_peak_list(f0, B, n_max=10, noise_cents=0.0)
    f0_est, B_est = pfd_estimate_f0_b(peaks_f, peaks_a)
    assert cents(f0_est, f0) < 0.01
    if B > 1e-5:
        rel_err = abs(B_est - B) / B
        assert rel_err < 0.005, f"B rel err {rel_err * 100:.2f}% for B={B}"


def test_pfd_on_matrix_like_conditions():
    """
    Covers multiple rows from the matrix: perfect, slight, high-B, noisy.
    Uses the estimator that the real pipeline will use.
    """
    cases = [
        (440.0, 0.00005, 0.0, None, 0.1, 0.03),
        (440.0, 0.0008, -1.5, None, 0.25, 0.05),
        (261.6, 0.002, 2.7, None, 0.25, 0.05),
        (4186.0, 0.025, 0.0, None, 0.6, 0.08),
        (440.0, 0.0006, 3.0, -18.0, 0.6, 0.08),
    ]
    rng = np.random.default_rng(99)
    for f0, B, _det, snr, tol_c, tol_b in cases:
        peaks_f, peaks_a = make_synthetic_peak_list(f0, B, n_max=8, noise_cents=0.0)
        if snr is not None:
            noise_c = 0.4 if snr < -15 else 0.1
            peaks_f = peaks_f * 2 ** (rng.normal(0, noise_c / 1200, len(peaks_f)))
        f0_est, B_est = pfd_estimate_f0_b(peaks_f, peaks_a, f0_guess=f0)
        err_c = abs(cents(f0_est, f0))
        assert err_c <= tol_c + 0.01, f"Case f0={f0} B={B} cents_err={err_c}"
        if B > 0.0005:
            berr = abs(B_est - B) / B
            assert berr <= tol_b + 0.005


def test_pfd_ignores_phantom_and_outliers():
    """High-B case must not hallucinate; estimator robust (uses f0_guess when helpful)."""
    f0, B = 4186.0, 0.18
    peaks_f, peaks_a = make_synthetic_peak_list(f0, B, n_max=5)
    peaks_f = np.append(peaks_f, [12000.0])
    peaks_a = np.append(peaks_a, [10.0])
    f0_est, B_est = pfd_estimate_f0_b(peaks_f, peaks_a, f0_guess=f0)
    assert cents(f0_est, f0) < 5.0  # realistic for extreme high-B + outlier
    assert 0.0 <= B_est <= 0.30


def test_cents_helper():
    assert abs(cents(440.0, 440.0)) < 1e-9
    assert abs(cents(440.0 * 2 ** (3 / 1200), 440.0) - 3.0) < 0.001
    assert cents(220.0, 440.0) < -1199
