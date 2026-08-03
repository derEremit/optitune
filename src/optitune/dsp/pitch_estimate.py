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

from optitune.dsp.note_follow import NoteFollowMode, apply_follow_to_midi, search_window
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
    follow_mode: NoteFollowMode | str = NoteFollowMode.AUTO,
    locked_midi: int | None = None,
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

    mode = (
        follow_mode
        if isinstance(follow_mode, NoteFollowMode)
        else NoteFollowMode(str(follow_mode))
    )
    # Prefer armed target as the lock anchor when recording; else explicit locked_midi
    lock_anchor = armed_midi if armed_midi is not None else locked_midi
    midi_lo, midi_hi = search_window(mode, lock_anchor)

    if len(peak_fs) >= 2:
        match = recognize_note(
            peak_fs=peak_fs,
            peak_as=peak_as,
            a4=a4,
            prior_midi=armed_midi if armed_midi is not None else lock_anchor,
            midi_lo=midi_lo,
            midi_hi=midi_hi,
        )
        if match is not None and match.confidence >= 0.52:
            recognized_midi = match.midi

    if len(peak_fs) >= 1:
        dom_idx = int(np.argmax(peak_as))
        f_dom = float(peak_fs[dom_idx])
        f_low = float(peak_fs[0])

        free_guess = (
            midi_to_hz(recognized_midi, a4)
            if recognized_midi is not None
            else max(float(last_f0_guess), 40.0)
        )

        def _pick_f0(f0_pfd: float, f_low: float, prior_hz: float) -> float:
            pfd_near = (
                prior_hz > 20 and f0_pfd > 20 and abs(1200.0 * np.log2(f0_pfd / prior_hz)) < 80.0
            )
            if f_low > 20 and f0_pfd > 20:
                dc = 1200.0 * np.log2(f0_pfd / f_low)
                return f0_pfd if (abs(dc) < 35.0 or pfd_near) else f_low
            return f0_pfd if f0_pfd > 20 else f_low

        def _partial_hits(target_hz: float, n_max: int = 6, tol_cents: float = 90.0) -> int:
            """Count how many of the first n_max partials have a nearby peak.

            tol is loose (~90 ¢) so flat/sharp real pianos still match the ladder.
            """
            hits = 0
            for n in range(1, n_max + 1):
                fn = target_hz * float(n)
                for p in peak_fs:
                    if p > 18 and abs(1200.0 * np.log2(float(p) / fn)) < tol_cents:
                        hits += 1
                        break
            return hits

        # Run free PFD always; also armed PFD when recording a target.
        f0_free_pfd, B_free = pfd_estimate_f0_b(peak_fs, peak_as, f0_guess=free_guess, max_n=16)
        f0_free = _pick_f0(f0_free_pfd, f_low, free_guess)
        B = B_free

        if armed_midi is not None:
            armed_hz = midi_to_hz(armed_midi, a4)
            f0_armed_pfd, B_armed = pfd_estimate_f0_b(peak_fs, peak_as, f0_guess=armed_hz, max_n=16)
            f0_armed = _pick_f0(f0_armed_pfd, f_low, armed_hz)
            armed_err = abs(1200.0 * np.log2(f0_armed / armed_hz)) if f0_armed > 20 else 1e9
            hits = _partial_hits(armed_hz)
            # Prefer armed only when spectrum supports its partial ladder.
            # (PFD alone with an armed guess can invent a bass f0 for any harmonic tone.)
            if armed_err < 80.0 and hits >= 2:
                f0, B = f0_armed, B_armed
            else:
                nearest_oct = round(float(np.log2(f0_free / armed_hz))) if f0_free > 20 else 0
                folded = f0_free / (2.0**nearest_oct) if nearest_oct else f0_free
                fold_err = abs(1200.0 * np.log2(folded / armed_hz)) if folded > 20 else 1e9
                if nearest_oct != 0 and abs(nearest_oct) <= 3 and fold_err < 100.0 and hits >= 2:
                    f0, B = folded, B_armed
                else:
                    # Different note (or unsupported armed) — trust free estimate
                    f0 = f0_free
        else:
            f0 = f0_free

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
    # Free-listening follow modes (scale armed path already pinned above)
    if armed_midi is None:
        followed = apply_follow_to_midi(mode, detected=midi, locked=lock_anchor)
        if followed is not None:
            midi = max(21, min(108, int(followed)))

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
