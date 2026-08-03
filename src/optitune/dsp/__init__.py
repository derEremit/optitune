"""DSP core: synth (Fletcher-Young), binning (log-cent + A), peaks (parabolic + PFD), stft helpers (Phase 1)."""

from __future__ import annotations

from .binning import (
    F_HI,
    F_LO,
    N_BINS,
    a_weight_db,
    apply_a_weight_to_binned,
    bin_and_a_weight,
    bin_center,
    bin_index,
    bin_spectrum_vectorized,
    slow_bin_spectrum,
)
from .note_recognizer import (
    NoteMatch,
    recognize_from_audio,
    recognize_note,
    spectrum_from_audio,
)
from .peaks import (
    cents,
    find_spectral_peaks,
    parabolic_interpolation,
    pfd_estimate_f0_b,
)
from .stft import compute_stft, get_central_frame_power
from .synth import (
    detuned_tone,
    fletcher_young_partial_frequencies,
    generate_inharmonic_tone,
    hz_to_midi,
    midi_to_hz,
    midi_to_note_name,
    perfect_tone,
)

__all__ = [
    "F_HI",
    "F_LO",
    "N_BINS",
    "NoteMatch",
    "a_weight_db",
    "apply_a_weight_to_binned",
    "bin_and_a_weight",
    "bin_center",
    "bin_index",
    "bin_spectrum_vectorized",
    "cents",
    "compute_stft",
    "detuned_tone",
    "find_spectral_peaks",
    "fletcher_young_partial_frequencies",
    "generate_inharmonic_tone",
    "get_central_frame_power",
    "hz_to_midi",
    "midi_to_hz",
    "midi_to_note_name",
    "parabolic_interpolation",
    "perfect_tone",
    "pfd_estimate_f0_b",
    "recognize_from_audio",
    "recognize_note",
    "slow_bin_spectrum",
    "spectrum_from_audio",
]
