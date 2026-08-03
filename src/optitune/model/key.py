"""
Key dataclass for OptiTune Phase 4 model.

Simple container for per-note measured data from live DSP (f0, B), optional
A-weighted cent spectrum (entropy solver), and the computed target offset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from optitune.model.spectrum_codec import pack_spectrum, unpack_spectrum


@dataclass
class Key:
    """Represents one piano key (MIDI 21=A0 … 108=C8).

    measured_f0: the fundamental frequency (Hz) estimated by PFD when this key was recorded.
    measured_b: the inharmonicity coefficient estimated for this string (dimensionless).
    cent_spectrum: A-weighted cent-binned SPLA (length N_BINS=12000), optional.
    target_offset_cents: the final cent offset from ET that the solver assigned (0 = pure ET for that MIDI).
    """

    midi: int
    measured_f0: float | None = None
    measured_b: float | None = None
    cent_spectrum: np.ndarray | None = field(default=None, repr=False)
    target_offset_cents: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "midi": int(self.midi),
            "measured_f0": self.measured_f0,
            "measured_b": self.measured_b,
            "target_offset_cents": float(self.target_offset_cents),
        }
        if self.cent_spectrum is not None:
            d["cent_spectrum"] = pack_spectrum(self.cent_spectrum)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Key:
        # Allow missing keys for forward compat
        spectrum = None
        raw = d.get("cent_spectrum")
        if raw is not None:
            if isinstance(raw, str):
                spectrum = unpack_spectrum(raw)
            elif isinstance(raw, (list, tuple, np.ndarray)):
                spectrum = np.asarray(raw, dtype=np.float32)
        return cls(
            midi=int(d["midi"]),
            measured_f0=d.get("measured_f0"),
            measured_b=d.get("measured_b"),
            cent_spectrum=spectrum,
            target_offset_cents=float(d.get("target_offset_cents", 0.0)),
        )
