"""
Synthetic inharmonic piano tone generator using the exact Fletcher–Young model.

f_n = n * F0 * sqrt(1 + B * n**2)   — piano_tuner_implementation_spec.md §2.1

Fully deterministic given seed. Supports hammer transient, SNR, detune, all matrix cases.
"""
from __future__ import annotations

import numpy as np


def midi_to_hz(midi: int, a4: float = 440.0) -> float:
    """Standard equal-tempered conversion (A4 = 440 by default for synth)."""
    return a4 * (2.0 ** ((midi - 69) / 12.0))


def fletcher_young_partial_frequencies(
    f0: float, B: float, n_partials: int | None = None, fs: float = 48000.0
) -> np.ndarray:
    """
    Exact model §2.1.
    Returns array of f_n up to just below Nyquist.
    """
    if n_partials is None:
        # Conservative upper bound
        n_partials = 60
    ns = np.arange(1, n_partials + 1, dtype=float)
    fn = ns * f0 * np.sqrt(1.0 + B * ns * ns)
    # Keep only below 0.98 * Nyquist to avoid aliasing artifacts in tests
    mask = fn < (fs / 2.0 * 0.98)
    return fn[mask]


def _piano_decay_envelope(t: np.ndarray, base_tau: float = 2.8) -> np.ndarray:
    """Mild overall decay for sustained test tones (analysis happens on middle)."""
    return np.exp(-t / base_tau)


def _hammer_transient(
    t: np.ndarray, f0: float, strength: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Simple physically-motivated hammer strike transient for bass thump and "with_hammer" cases.
    - Low-frequency thump (slightly inharmonic)
    - Short broadband click
    Decays much faster than string partials.
    """
    hammer_len = min(len(t), int(0.18 * (1 / (f0 / 60 + 0.1))))  # longer for bass
    if hammer_len < 8:
        return np.zeros_like(t)

    ht = t[:hammer_len]
    # Inharmonic thump ~ 0.6-0.9 * f0 , damped fast
    thump_f = f0 * (0.65 + 0.12 * rng.random())
    thump = strength * 0.9 * np.exp(-ht / 0.022) * np.sin(2 * np.pi * thump_f * ht + rng.random() * 0.3)

    # High-freq click / felt noise , very short
    click = strength * rng.normal(0.0, 0.65, hammer_len) * np.exp(-ht / 0.0065)
    # Very light low-pass effect via cumulative sum (cheap)
    click = np.cumsum(click) * 0.08 + click * 0.92

    out = np.zeros_like(t)
    out[:hammer_len] = thump + click * 0.6
    return out


def generate_inharmonic_tone(
    midi_or_f0: int | float,
    detune_cents: float = 0.0,
    B: float = 0.0003,
    duration: float = 3.0,
    fs: int = 48000,
    snr_db: float | None = None,
    with_hammer: bool = True,
    seed: int | None = 42,
    a4: float = 440.0,
    peak_amp: float = 0.92,
) -> np.ndarray:
    """
    Primary generator for the entire Phase 1 matrix.

    Returns float64 array of length round(duration * fs).
    Fully deterministic for given seed (including noise when snr_db given).
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    if isinstance(midi_or_f0, (int, np.integer)):
        f0_nom = midi_to_hz(int(midi_or_f0), a4)
    else:
        f0_nom = float(midi_or_f0)

    f0 = f0_nom * (2.0 ** (detune_cents / 1200.0))

    partial_freqs = fletcher_young_partial_frequencies(f0, B, fs=float(fs))
    n_p = len(partial_freqs)
    if n_p == 0:
        return np.zeros(int(duration * fs), dtype=np.float64)

    t = np.arange(int(duration * fs), dtype=np.float64) / float(fs)

    # Sum partials (vectorized for speed)
    y = np.zeros_like(t)
    for i, fn in enumerate(partial_freqs):
        # amplitude rolloff typical for piano (brighter for low notes)
        amp = peak_amp * (0.6 + 0.4 * np.exp(-i / 8.0)) / (1 + i * 0.15)
        phase = rng.random() * 2 * np.pi if seed is None else (i * 0.7)
        y += amp * np.sin(2 * np.pi * fn * t + phase)

    # Apply mild decay
    env = _piano_decay_envelope(t)
    y *= env

    # Hammer strike transient for realism (bass especially)
    if with_hammer:
        hammer = _hammer_transient(t, f0, strength=0.6, rng=rng)
        y += hammer

    # Add controlled noise (after everything else for accurate SNR)
    if snr_db is not None:
        sig_power = np.mean(y**2) + 1e-12
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = rng.standard_normal(len(y)) * np.sqrt(noise_power)
        y = y + noise

    # Final peak normalization (prevents clipping while preserving relative dynamics)
    peak = np.max(np.abs(y))
    if peak > peak_amp:
        y *= (peak_amp / peak)

    # Very small DC removal
    y -= np.mean(y)

    return y.astype(np.float64)


# --- Convenience wrappers required by plan + matrix ---

def perfect_tone(
    midi: int,
    duration: float = 2.5,
    fs: int = 48000,
    seed: int = 0,
    B: float = 0.00005,
) -> np.ndarray:
    """Zero-detune, very low inharmonicity reference tone."""
    return generate_inharmonic_tone(
        midi, detune_cents=0.0, B=B, duration=duration, fs=fs, snr_db=None, with_hammer=True, seed=seed
    )


def detuned_tone(
    midi: int,
    level: str = "slight",
    duration: float = 2.5,
    fs: int = 48000,
    seed: int = 123,
) -> np.ndarray:
    """
    Quick presets matching matrix "slightly / medium / bad".
    level: "slight" | "medium" | "bad"
    """
    if level == "slight":
        cents, B = -1.5, 0.0002
    elif level == "medium":
        cents, B = -8.0, 0.0008
    elif level == "bad":
        cents, B = -25.0, 0.002
    else:
        raise ValueError(f"Unknown level {level}")
    return generate_inharmonic_tone(
        midi, detune_cents=cents, B=B, duration=duration, fs=fs, snr_db=None, with_hammer=True, seed=seed
    )


# --- Phase 3 pitch utilities (used by live analysis and tests) ---

def hz_to_midi(f_hz: float, a4: float = 440.0) -> float:
    """Continuous MIDI note number (float, 69 = A4) given frequency and A4 reference."""
    if f_hz <= 0.0:
        return 0.0
    return 69.0 + 12.0 * np.log2(f_hz / a4)


def midi_to_note_name(midi: int | float) -> str:
    """Return scientific pitch notation e.g. 'C#4', 'A0' for a MIDI number."""
    m = int(round(float(midi)))
    if m < 0 or m > 127:
        return "?"
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (m // 12) - 1
    name = names[m % 12]
    return f"{name}{octave}"
