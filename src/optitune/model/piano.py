"""
Piano dataclass and TuningCurve helpers for OptiTune Phase 4.

Holds the instrument state: A4 reference, per-key measurements, and the computed 88-note tuning curve.
Simple JSON persistence for v0.1 (full .pfg/.otf XML later).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .key import Key


@dataclass
class Piano:
    """The user's piano + its tuning session.

    keys: midi (int) -> Key  (only keys that have been measured or touched)
    tuning_curve: list[float] | None  — 88 cent offsets, index 0 == MIDI 21 (A0), index 87 == MIDI 108 (C8).
                     None means "use pure equal temperament".
    """

    name: str = "My Piano"
    a4: float = 440.0
    keys: dict[int, Key] = field(default_factory=dict)
    tuning_curve: list[float] | None = None

    # MIDI range
    MIDI_LOW: int = 21
    MIDI_HIGH: int = 108
    N_KEYS: int = 88

    def __post_init__(self) -> None:
        if self.tuning_curve is not None and len(self.tuning_curve) != self.N_KEYS:
            # Fix bad length defensively
            self.tuning_curve = list(self.tuning_curve)[: self.N_KEYS]
            if len(self.tuning_curve) < self.N_KEYS:
                self.tuning_curve += [0.0] * (self.N_KEYS - len(self.tuning_curve))

    def get_key(self, midi: int) -> Key | None:
        return self.keys.get(midi)

    def set_key(self, key: Key) -> None:
        self.keys[key.midi] = key

    def get_target_offset(self, midi: int) -> float:
        """Return the cent offset the live tuner should use for this MIDI (0 if no curve)."""
        if self.tuning_curve is None:
            return 0.0
        idx = midi - self.MIDI_LOW
        if 0 <= idx < len(self.tuning_curve):
            return float(self.tuning_curve[idx])
        return 0.0

    def has_measurements(self) -> bool:
        return any(k.measured_b is not None or k.measured_f0 is not None for k in self.keys.values())

    def measured_count(self) -> int:
        return sum(1 for k in self.keys.values() if k.measured_b is not None or k.measured_f0 is not None)

    # ---------------- JSON persistence (simple, human-readable) ----------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "a4": float(self.a4),
            "keys": {str(m): k.to_dict() for m, k in sorted(self.keys.items())},
            "tuning_curve": self.tuning_curve,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Piano:
        p = cls(
            name=str(d.get("name", "My Piano")),
            a4=float(d.get("a4", 440.0)),
        )
        for m_str, kd in d.get("keys", {}).items():
            try:
                m = int(m_str)
                p.keys[m] = Key.from_dict(kd)
            except Exception:
                continue
        tc = d.get("tuning_curve")
        if isinstance(tc, list) and len(tc) == p.N_KEYS:
            p.tuning_curve = [float(x) for x in tc]
        else:
            p.tuning_curve = None
        return p

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str | Path) -> Piano | None:
        p = Path(path)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                d = json.load(f)
            return cls.from_dict(d)
        except Exception:
            return None

    # Convenience: default persistence location (~/.config/optitune/current_piano.json)
    @classmethod
    def default_persist_path(cls) -> Path:
        cfg = Path.home() / ".config" / "optitune"
        return cfg / "current_piano.json"
