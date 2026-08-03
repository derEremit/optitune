"""
Inharmonic comb-filter note recognizer (spec §3.6, Galembo & Askenfelt 1999).

Scores each candidate piano key by how well its Fletcher-Young partial comb
explains the observed spectrum (or peak list). Because the template contains
*all* partials, a note whose energy sits in partials 2-4 still scores highest
at the true fundamental - the structural fix for octave errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optitune.dsp.binning import N_BINS, bin_index, bin_spectrum_vectorized
from optitune.dsp.peaks import cents, find_spectral_peaks
from optitune.dsp.synth import midi_to_hz

MIDI_LOW = 21
MIDI_HIGH = 108
N_KEYS = MIDI_HIGH - MIDI_LOW + 1  # 88


def typical_b(midi: int) -> float:
    """
    Rough log-linear typical B across the piano compass.

    Anchored near published ranges: ~1e-4 bass, ~1e-3 mid, ~0.01-0.05 treble.
    Exact B is not critical for recognition - the comb still peaks at the true key.
    """
    t = (float(midi) - MIDI_LOW) / max(N_KEYS - 1, 1)
    log10_b = -4.2 + 2.6 * t  # ~6e-5 → ~0.025
    return float(10.0**log10_b)


def partial_frequencies(
    midi: int, *, a4: float = 440.0, b: float | None = None, n_max: int = 16
) -> np.ndarray:
    """Fletcher-Young partials for one key, clipped below ~9.8 kHz."""
    f0 = midi_to_hz(midi, a4)
    bb = typical_b(midi) if b is None else float(b)
    ns = np.arange(1, n_max + 1, dtype=float)
    fn = ns * f0 * np.sqrt(1.0 + bb * ns * ns)
    return fn[fn < 9_800.0]


def comb_template(
    midi: int,
    *,
    a4: float = 440.0,
    b: float | None = None,
    n_max: int = 16,
    sigma_cents: float = 12.0,
    n_bins: int = N_BINS,
) -> np.ndarray:
    """
    Build a unit-norm partial-comb template in cent-bin space for one key.

    Exposed for inspection/tests; recognition uses peak-local scoring primarily.
    """
    tmpl = np.zeros(n_bins, dtype=np.float64)
    fns = partial_frequencies(midi, a4=a4, b=b, n_max=n_max)
    half = int(max(4, round(3.0 * sigma_cents)))
    sig2 = 2.0 * sigma_cents * sigma_cents

    for n, fn in enumerate(fns, start=1):
        m0 = int(bin_index(float(fn)))
        amp = 1.0 / (float(n) ** 1.2)
        lo = max(0, m0 - half)
        hi = min(n_bins, m0 + half + 1)
        if lo >= hi:
            continue
        dm = np.arange(lo, hi, dtype=float) - float(m0)
        tmpl[lo:hi] += amp * np.exp(-(dm * dm) / sig2)

    norm = float(np.linalg.norm(tmpl))
    if norm > 1e-12:
        tmpl /= norm
    return tmpl


@dataclass(frozen=True)
class NoteMatch:
    """Result of comb-filter note recognition."""

    midi: int
    score: float
    confidence: float  # best / (best + second)
    scores: np.ndarray  # length 88, aligned to MIDI_LOW..MIDI_HIGH


def spectrum_from_audio(audio: np.ndarray, fs: float) -> np.ndarray:
    """Cent-binned power spectrum (no A-weighting - preserves bass energy)."""
    x = np.asarray(audio, dtype=np.float64)
    n = len(x)
    if n < 256:
        return np.zeros(N_BINS, dtype=np.float64)
    # Pad toward ~1.5 s / 65536 for better low-f0 bin resolution
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
    return bin_spectrum_vectorized(freqs, power)


def peaks_from_audio(
    audio: np.ndarray, fs: float, *, max_peaks: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Parabolic-interpolated spectral peaks from a mono buffer (long-padded)."""
    x = np.asarray(audio, dtype=np.float64)
    n = len(x)
    if n < 256:
        return np.array([]), np.array([])
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
    return find_spectral_peaks(freqs, power, min_prominence_db=8.0, max_peaks=max_peaks)


def _energy_near(L: np.ndarray, f_hz: float, half_bins: int = 18) -> float:
    """Max bin energy in a ±half_bins window around frequency f_hz."""
    m = int(bin_index(float(f_hz)))
    lo = max(0, m - half_bins)
    hi = min(len(L), m + half_bins + 1)
    if lo >= hi:
        return 0.0
    return float(np.max(L[lo:hi]))


def _peak_support(
    peak_fs: np.ndarray, peak_as: np.ndarray, f_hz: float, tol_cents: float = 35.0
) -> float:
    """Amplitude of nearest peak within tol_cents of f_hz, else 0."""
    if len(peak_fs) == 0 or f_hz <= 0:
        return 0.0
    # cents distance to each peak
    best = 0.0
    best_c = 1e9
    for f, a in zip(peak_fs, peak_as, strict=False):
        if f <= 0:
            continue
        c = abs(float(cents(float(f), float(f_hz))))
        if c < best_c and c <= tol_cents:
            best_c = c
            best = float(a)
    return best


def score_keys_from_spectrum(
    spectrum: np.ndarray,
    *,
    a4: float = 440.0,
    midi_lo: int = MIDI_LOW,
    midi_hi: int = MIDI_HIGH,
    n_max: int = 12,
) -> np.ndarray:
    """Score each key by summed (weighted) bin energy at its partial locations."""
    L = np.asarray(spectrum, dtype=np.float64)
    # Mild compression so a few huge bins don't dominate
    L_work = np.sqrt(np.maximum(L, 0.0))
    scores = np.full(N_KEYS, -np.inf, dtype=np.float64)

    for midi in range(max(MIDI_LOW, midi_lo), min(MIDI_HIGH, midi_hi) + 1):
        fns = partial_frequencies(midi, a4=a4, n_max=n_max)
        if len(fns) == 0:
            continue
        s = 0.0
        wsum = 0.0
        for n, fn in enumerate(fns, start=1):
            w = 1.0 / (float(n) ** 1.1)
            s += w * _energy_near(L_work, float(fn))
            wsum += w
        scores[midi - MIDI_LOW] = s / max(wsum, 1e-12)

    return scores


def score_keys_from_peaks(
    peak_fs: np.ndarray,
    peak_as: np.ndarray,
    *,
    a4: float = 440.0,
    midi_lo: int = MIDI_LOW,
    midi_hi: int = MIDI_HIGH,
    n_max: int = 12,
    tol_cents: float = 35.0,
) -> np.ndarray:
    """
    Score each key by how many/strong peaks sit on its Fletcher-Young partials.

    This is the Galembo-style inharmonic comb score and is the primary path.
    """
    scores = np.full(N_KEYS, -np.inf, dtype=np.float64)
    if len(peak_fs) == 0:
        return scores

    # Normalize peak amps for stability
    amps = np.asarray(peak_as, dtype=np.float64)
    peak_fs = np.asarray(peak_fs, dtype=np.float64)
    if np.max(amps) > 0:
        amps = amps / np.max(amps)

    for midi in range(max(MIDI_LOW, midi_lo), min(MIDI_HIGH, midi_hi) + 1):
        fns = partial_frequencies(midi, a4=a4, n_max=n_max)
        if len(fns) == 0:
            continue
        s = 0.0
        hits = 0
        for n, fn in enumerate(fns, start=1):
            w = 1.0 / (float(n) ** 0.9)
            support = _peak_support(peak_fs, amps, float(fn), tol_cents=tol_cents)
            if support > 0:
                hits += 1
            s += w * support
        # Bonus for number of partials explained (disambiguates octaves:
        # the true f0 explains odd partials that 2·f0 cannot)
        s += 0.08 * hits
        scores[midi - MIDI_LOW] = s

    return scores


def _prefer_subharmonic(scores: np.ndarray, best_i: int, margin: float = 0.70) -> int:
    """
    If a lower octave of the best key scores within `margin` of best, prefer it.

    Classical octave error: partials of f0 look like a strong match for 2·f0
    (even partials only). When both score high, the lower key is the correct
    piano note - especially on bass with weak fundamentals.
    """
    best = float(scores[best_i])
    if best <= 0 or not np.isfinite(best):
        return best_i
    chosen = best_i
    for octaves in (1, 2, 3):
        lower_i = best_i - 12 * octaves
        if lower_i < 0:
            continue
        s = float(scores[lower_i])
        if np.isfinite(s) and s >= margin * best:
            # Prefer the lowest such candidate
            chosen = lower_i
    return chosen


def recognize_note(
    spectrum: np.ndarray | None = None,
    *,
    peak_fs: np.ndarray | None = None,
    peak_as: np.ndarray | None = None,
    a4: float = 440.0,
    midi_lo: int = MIDI_LOW,
    midi_hi: int = MIDI_HIGH,
    prior_midi: int | None = None,
    prior_boost: float = 0.06,
    min_score: float = 0.02,
) -> NoteMatch | None:
    """
    Pick the best-matching key.

    Prefer peak-based scoring when peaks are provided; otherwise score the
    cent-binned spectrum. `prior_midi` is a soft tie-break only.
    """
    if peak_fs is not None and peak_as is not None and len(peak_fs) > 0:
        scores = score_keys_from_peaks(peak_fs, peak_as, a4=a4, midi_lo=midi_lo, midi_hi=midi_hi)
    elif spectrum is not None:
        scores = score_keys_from_spectrum(spectrum, a4=a4, midi_lo=midi_lo, midi_hi=midi_hi)
    else:
        return None

    if prior_midi is not None and MIDI_LOW <= prior_midi <= MIDI_HIGH:
        idx = prior_midi - MIDI_LOW
        if np.isfinite(scores[idx]):
            scores = scores.copy()
            scores[idx] += prior_boost * (float(np.nanmax(scores[np.isfinite(scores)])) + 1e-9)

    finite = np.isfinite(scores) & (scores > -1e8)
    if not np.any(finite):
        return None

    order = np.argsort(np.where(finite, scores, -np.inf))[::-1]
    best_i = int(order[0])
    best_i = _prefer_subharmonic(scores, best_i)
    best = float(scores[best_i])
    if best < min_score:
        return None

    second = 0.0
    for j in order:
        if int(j) != best_i:
            second = float(scores[int(j)])
            break
    conf = best / (best + max(second, 0.0) + 1e-12)

    return NoteMatch(
        midi=MIDI_LOW + best_i,
        score=best,
        confidence=float(conf),
        scores=scores,
    )


def recognize_from_audio(
    audio: np.ndarray,
    fs: float,
    *,
    a4: float = 440.0,
    prior_midi: int | None = None,
    midi_lo: int = MIDI_LOW,
    midi_hi: int = MIDI_HIGH,
) -> NoteMatch | None:
    """Convenience: extract peaks (+ spectrum fallback) → recognize_note."""
    peak_fs, peak_as = peaks_from_audio(audio, fs)
    if len(peak_fs) >= 2:
        return recognize_note(
            peak_fs=peak_fs,
            peak_as=peak_as,
            a4=a4,
            prior_midi=prior_midi,
            midi_lo=midi_lo,
            midi_hi=midi_hi,
        )
    L = spectrum_from_audio(audio, fs)
    return recognize_note(
        spectrum=L,
        a4=a4,
        prior_midi=prior_midi,
        midi_lo=midi_lo,
        midi_hi=midi_hi,
    )
