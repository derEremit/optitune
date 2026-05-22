"""
Key dataclass for OptiTune Phase 4 model.

Simple container for per-note measured data from live DSP (f0, B) and the computed target offset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Key:
    """Represents one piano key (MIDI 21=A0 … 108=C8).

    measured_f0: the fundamental frequency (Hz) estimated by PFD when this key was recorded.
    measured_b: the inharmonicity coefficient estimated for this string (dimensionless).
    target_offset_cents: the final cent offset from ET that the solver assigned (0 = pure ET for that MIDI).
    """

    midi: int
    measured_f0: float | None = None
    measured_b: float | None = None
    # cents_spectrum reserved for future entropy solver (not stored in v0.1 JSON)
    target_offset_cents: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Key:
        # Allow missing keys for forward compat
        return cls(
            midi=int(d["midi"]),
            measured_f0=d.get("measured_f0"),
            measured_b=d.get("measured_b"),
            target_offset_cents=float(d.get("target_offset_cents", 0.0)),
        )
