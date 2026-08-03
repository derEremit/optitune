"""
Peak detection, exact parabolic interpolation (§3.5), and robust PFD-style (F0, B) estimator.

Formulas from piano_tuner_implementation_spec.md §3.5 and §2.6.
Candidate search over low-frequency peaks makes it robust for bass, high-B, and noisy cases
while still recovering synthetic ground truth to the required tolerances.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def cents(f_est: float, f_true: float) -> float:
    if f_true <= 0 or f_est <= 0:
        return float(np.inf)
    return float(1200.0 * np.log2(f_est / f_true))


def parabolic_interpolation(log_mag: np.ndarray, peak_idx: int | np.integer) -> tuple[float, float]:
    n = len(log_mag)
    k = int(peak_idx)
    if not (1 <= k < n - 1):
        return 0.0, float(log_mag[k] if 0 <= k < n else 0.0)

    y1 = float(log_mag[k - 1])
    y2 = float(log_mag[k])
    y3 = float(log_mag[k + 1])

    denom = y1 - 2.0 * y2 + y3
    delta = 0.0 if abs(denom) < 1e-14 else 0.5 * (y1 - y3) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    log_peak = y2 - 0.25 * (y1 - y3) * delta
    return delta, log_peak


def find_spectral_peaks(
    freqs: np.ndarray,
    power: np.ndarray,
    min_prominence_db: float = 12.0,
    max_peaks: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    if len(power) < 4:
        return np.array([]), np.array([])

    log_mag = 10.0 * np.log10(np.maximum(power, 1e-30))
    floor = np.median(log_mag) + min_prominence_db

    peaks, _ = find_peaks(
        log_mag,
        height=floor,
        prominence=min_prominence_db * 0.55,
        distance=2,
    )

    if len(peaks) == 0:
        return np.array([]), np.array([])

    refined_freqs = []
    refined_amps = []
    for k in peaks:
        delta, log_p = parabolic_interpolation(log_mag, k)
        df = (freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        f_ref = freqs[k] + delta * df
        amp = 10.0 ** (log_p / 10.0)
        refined_freqs.append(f_ref)
        refined_amps.append(amp)

    order = np.argsort(refined_freqs)
    pf = np.asarray(refined_freqs)[order]
    pa = np.asarray(refined_amps)[order]

    if len(pf) > max_peaks:
        top = np.argsort(pa)[-max_peaks:]
        pf = pf[top]
        pa = pa[top]
        order2 = np.argsort(pf)
        pf = pf[order2]
        pa = pa[order2]

    return pf, pa


def _fit_f0_b_linear(ns: np.ndarray, fms: np.ndarray) -> tuple[float, float]:
    if len(ns) < 2:
        return float(fms[0]) if len(fms) else 440.0, 0.0003
    ys = (fms / ns) ** 2
    xs = ns**2
    X = np.column_stack([np.ones_like(xs), xs])
    sol, *_ = np.linalg.lstsq(X, ys, rcond=None)
    a, b = sol
    if a <= 1e-8:
        return float(fms[0]), 0.0003
    f0 = float(np.sqrt(max(a, 0.0)))
    B = float(max(0.0, min(b / a, 0.5)))
    return f0, B


def pfd_estimate_f0_b(
    peak_freqs: np.ndarray,
    peak_amps: np.ndarray,
    f0_guess: float | None = None,
    max_n: int = 20,
) -> tuple[float, float]:
    """
    Robust PFD with multi-candidate search + strong anchoring to f0_guess (used by matrix tests).
    """
    if len(peak_freqs) < 1:
        return 440.0, 0.0003

    pf = np.sort(peak_freqs)
    pa = np.asarray(peak_amps)
    if len(pa) != len(pf):
        pa = np.ones_like(pf)

    candidates: list[float] = []
    for f in pf[: min(6, len(pf))]:
        if 20 < f < 5000:
            candidates.append(f)
    if f0_guess and 20 < f0_guess < 6000:
        candidates.append(f0_guess)
        candidates.append(f0_guess * 2 if f0_guess < 2500 else f0_guess)
        if f0_guess > 100:
            candidates.append(f0_guess / 2)

    candidates = sorted(set(c for c in candidates if 20 < c < 6000))
    if not candidates:
        candidates = [pf[0] if pf[0] > 20 else 440.0]

    best_f0 = candidates[0]
    best_B = 0.0003
    best_score = -1e9

    for f1 in candidates:
        ns_list = []
        fms_list = []
        for f in pf:
            if f < 18 or f > 14000:
                continue
            n = round(f / f1)
            if 1 <= n <= max_n:
                ns_list.append(n)
                fms_list.append(f)

        if len(ns_list) < 2:
            continue

        ns = np.array(ns_list, dtype=float)
        fms = np.array(fms_list, dtype=float)

        f0, B = _fit_f0_b_linear(ns, fms)

        model_fs = ns * f0 * np.sqrt(1.0 + B * ns * ns)
        residuals_cents = np.abs(
            [cents(mf, mf_model) for mf, mf_model in zip(fms, model_fs, strict=False)]
        )
        inliers = np.sum(np.array(residuals_cents) < 25.0)
        residual_penalty = (
            float(np.mean(residuals_cents[: min(8, len(residuals_cents))]))
            if len(residuals_cents)
            else 100.0
        )

        guess_bonus = 0.0
        if f0_guess is not None:
            guess_cents = abs(cents(f0, f0_guess))
            if guess_cents < 20:
                guess_bonus = 120 - guess_cents * 2
            elif guess_cents < 80:
                guess_bonus = 40 - guess_cents / 3
            else:
                guess_bonus = -80

        score = float(inliers) * 10.0 - residual_penalty + guess_bonus

        if score > best_score:
            best_score = score
            best_f0 = f0
            best_B = B

    # Safeguard against bad octave jumps when guess available
    if f0_guess is not None and abs(cents(best_f0, f0_guess)) > 40:
        dists = np.abs(pf - f0_guess)
        if len(dists) > 0:
            cand = float(pf[np.argmin(dists)])
            # accept only if reasonable
            if abs(cents(cand, f0_guess)) < 80:
                best_f0 = cand
                _, best_B = _fit_f0_b_linear(np.array([1.0, 2.0]), np.array([cand, 2 * cand]))

    # Subharmonic disambiguation: if f0/2 (or f0/3) explains more partials,
    # prefer the lower fundamental (classic weak-fundamental / octave error fix).
    # Skipped when f0_guess already anchors us near the correct octave.
    best_f0, best_B = _prefer_subharmonic_f0(best_f0, best_B, pf, max_n, f0_guess=f0_guess)

    return best_f0, best_B


def _count_partial_inliers(
    f0: float, B: float, peak_freqs: np.ndarray, max_n: int, tol_cents: float = 30.0
) -> int:
    """How many peaks sit within tol_cents of some partial of (f0, B)."""
    if f0 <= 0:
        return 0
    count = 0
    for f in peak_freqs:
        if f < 18 or f > 14000:
            continue
        n = round(f / f0)
        if not (1 <= n <= max_n):
            continue
        model = n * f0 * float(np.sqrt(1.0 + B * n * n))
        if model > 0 and abs(cents(float(f), model)) < tol_cents:
            count += 1
    return count


def _prefer_subharmonic_f0(
    f0: float,
    B: float,
    peak_freqs: np.ndarray,
    max_n: int,
    f0_guess: float | None = None,
) -> tuple[float, float]:
    """
    Prefer f0/2 or f0/3 when they strictly explain more partials.

    Guards:
    - Do not walk away from a good f0_guess (within ~80 ¢).
    - Require a peak near the candidate subharmonic (or very low bass < 90 Hz
      where fundamentals are often weak).
    - Require strictly more inliers than the current f0 (not ≥).
    """
    best_f0, best_B = float(f0), float(B)

    if f0_guess is not None and abs(cents(best_f0, f0_guess)) < 80:
        return best_f0, best_B

    best_inliers = _count_partial_inliers(best_f0, best_B, peak_freqs, max_n)

    for div in (2, 3):
        f_sub = best_f0 / div
        if f_sub < 22.0:
            continue
        # Evidence of the lower fundamental (peak nearby), unless bass-weak-fund
        has_fund_peak = any(f > 18 and abs(cents(float(f), f_sub)) < 50.0 for f in peak_freqs)
        if not has_fund_peak and f_sub > 90.0:
            continue

        ns_list: list[float] = []
        fms_list: list[float] = []
        for f in peak_freqs:
            if f < 18 or f > 14000:
                continue
            n = round(f / f_sub)
            if 1 <= n <= max_n:
                ns_list.append(float(n))
                fms_list.append(float(f))
        if len(ns_list) < 2:
            continue
        f0_fit, B_fit = _fit_f0_b_linear(np.array(ns_list), np.array(fms_list))
        if abs(cents(f0_fit, f_sub)) > 80:
            continue
        # Prefer sub only if closer to guess (when present) or clearly more inliers
        if f0_guess is not None and abs(cents(f0_fit, f0_guess)) >= abs(cents(best_f0, f0_guess)):
            continue
        inliers = _count_partial_inliers(f0_fit, B_fit, peak_freqs, max_n)
        if inliers > best_inliers:
            best_f0, best_B = f0_fit, B_fit
            best_inliers = inliers

    return best_f0, best_B
