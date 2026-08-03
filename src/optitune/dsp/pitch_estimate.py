"""
Pure live pitch estimator (Qt-free).

Combines:
- spectral peaks + PFD (f0, B)
- inharmonic comb note recognizer (soft identity)
- armed-MIDI prior for scale recording (soft: improves f0 guess / octave fold;
  does not force a wrong pitch class)

Used by OptiTuneMainWindow._estimate_pitch and unit-tested without Qt.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import get_window

from optitune.dsp.note_recognizer import recognize_note
from optitune.dsp.peaks import find_spectral_peaks, pfd_estimate_f0_b
from optitune.dsp.synth import hz_to_midi, midi_to_hz

# Match MainWindow scale-mode commit tolerance
SCALE_MODE_CENT_TOLERANCE = 140.0


def estimate_pitch(
    audio: np.ndarray,
    fs: float,
    *,
    a4: float = 440.0,
    armed_midi: int | None = None,
    last_f0_guess: float = 440.0,
    long_frame_samples: int = 65536,
    scale_cent_tol: float = SCALE_MODE_CENT_TOLERANCE,
) -> dict[str, Any]:
    """
    Estimate f0 / MIDI / cents for a mono float buffer.

    Returns dict with keys: f_est, f0, midi, target_hz, cents, delta_hz, f_dom, b.
    """
    analysis = np.asarray(audio, dtype=np.float64)
    n = len(analysis)
    if n < 256:
        return _fallback(analysis, fs, a4)

    # Longer frames for bass (spec §3.3)
    want_long = n < long_frame_samples and (armed_midi is None or armed_midi < 55 or n < 24000)
    if want_long and n < long_frame_samples:
        pad = np.zeros(long_frame_samples, dtype=np.float64)
        pad[:n] = analysis
        analysis = pad
        n = long_frame_samples

    try:
        w = get_window("blackmanharris", n)
    except Exception:
        w = np.hanning(n)

    spec = np.fft.rfft(analysis * w)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    peak_fs, peak_as = find_spectral_peaks(freqs, power, min_prominence_db=14.0, max_peaks=25)

    f0 = 440.0
    B = 0.0003
    f_dom = 440.0
    recognized_midi: int | None = None

    if len(peak_fs) >= 2:
        match = recognize_note(
            peak_fs=peak_fs,
            peak_as=peak_as,
            a4=a4,
            prior_midi=armed_midi,
        )
        if match is not None and match.confidence >= 0.52:
            recognized_midi = match.midi

    if len(peak_fs) >= 1:
        dom_idx = int(np.argmax(peak_as))
        f_dom = float(peak_fs[dom_idx])
        f_low = float(peak_fs[0])

        # PFD guess: armed is a *soft* prior - only when the spectrum looks like
        # that register (lowest peak near/below ~2.5x armed f0, or recognizer
        # already within +/-2 semitones). Otherwise a high note of the same pitch
        # class (e.g. C4 while armed on C1) would be dragged an octave down.
        if armed_midi is not None:
            armed_hz = midi_to_hz(armed_midi, a4)
            register_ok = f_low < max(armed_hz * 2.8, 100.0)
            near_armed = recognized_midi is not None and abs(recognized_midi - armed_midi) <= 2
            if register_ok or near_armed:
                f0_guess = armed_hz
            elif recognized_midi is not None:
                f0_guess = midi_to_hz(recognized_midi, a4)
            else:
                f0_guess = max(float(last_f0_guess), 40.0)
        elif recognized_midi is not None:
            f0_guess = midi_to_hz(recognized_midi, a4)
        else:
            f0_guess = max(float(last_f0_guess), 40.0)

        f0_pfd, B = pfd_estimate_f0_b(peak_fs, peak_as, f0_guess=f0_guess, max_n=16)

        prior_hz = f0_guess
        pfd_near_prior = (
            prior_hz > 20 and f0_pfd > 20 and abs(1200.0 * np.log2(f0_pfd / prior_hz)) < 80.0
        )
        if f_low > 20 and f0_pfd > 20:
            dc = 1200.0 * np.log2(f0_pfd / f_low)
            f0 = f0_pfd if (abs(dc) < 35.0 or pfd_near_prior) else f_low
        else:
            f0 = f_low

        # Fold octaves toward fold target only when this looks like a partial/octave
        # error of the *same* note - not when a genuinely higher note is playing.
        # Require a peak near the fold-target fundamental, or that PFD's guess
        # *was* that fundamental and stayed there (pfd_near_prior for that target).
        fold_midi = armed_midi if armed_midi is not None else recognized_midi
        if fold_midi is not None and f0 > 20:
            target_f = midi_to_hz(fold_midi, a4)
            if target_f > 20:
                nearest_oct = round(float(np.log2(f0 / target_f)))
                if nearest_oct != 0 and abs(nearest_oct) <= 3:
                    folded = f0 / (2.0**nearest_oct)
                    near_target = abs(1200.0 * np.log2(folded / target_f)) < 120.0
                    has_fund_peak = any(
                        p > 18 and abs(1200.0 * np.log2(float(p) / target_f)) < 60.0
                        for p in peak_fs
                    )
                    # Only trust pfd_near_prior when the PFD guess *was* this target
                    guess_was_target = abs(1200.0 * np.log2(prior_hz / target_f)) < 50.0
                    if near_target and (has_fund_peak or (guess_was_target and pfd_near_prior)):
                        f0 = folded

        if not (25 < f0 < 5500):
            f0 = f_dom if 25 < f_dom < 5500 else float(last_f0_guess)

    f_est = float(np.clip(f0, 25.0, 5500.0))

    midi_from_f = round(hz_to_midi(f_est, a4))
    if armed_midi is not None and f_est > 20:
        armed_hz = midi_to_hz(armed_midi, a4)
        err_to_armed = abs(1200.0 * np.log2(f_est / armed_hz))
        if err_to_armed <= scale_cent_tol:
            midi = armed_midi
        elif recognized_midi is not None and abs(recognized_midi - midi_from_f) <= 4:
            midi = recognized_midi
        else:
            midi = int(midi_from_f)
    elif recognized_midi is not None and abs(recognized_midi - midi_from_f) <= 4:
        midi = recognized_midi
    else:
        midi = int(midi_from_f)
    midi = max(21, min(108, int(midi)))

    target_hz = midi_to_hz(midi, a4)
    if target_hz <= 0:
        target_hz = a4

    if f_est > 1 and target_hz > 1:
        cents_val = 1200.0 * np.log2(f_est / target_hz)
        delta_hz = f_est - target_hz
    else:
        cents_val = 0.0
        delta_hz = 0.0

    return {
        "f_est": f_est,
        "f0": float(f0),
        "midi": midi,
        "target_hz": float(target_hz),
        "cents": float(cents_val),
        "delta_hz": float(delta_hz),
        "f_dom": float(f_dom),
        "b": float(B),
    }


def _fallback(audio: np.ndarray, fs: float, a4: float) -> dict[str, Any]:
    return {
        "f_est": a4,
        "f0": a4,
        "midi": 69,
        "target_hz": a4,
        "cents": 0.0,
        "delta_hz": 0.0,
        "f_dom": a4,
        "b": 0.0003,
    }
