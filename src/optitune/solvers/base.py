"""
Solver protocol and shared types (spec §4.3).

All production solvers implement `Solver.solve(...) -> Iterator[TuningCurve]`
so the GUI can stream intermediate curves and swap algorithms at runtime.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

MIDI_LOW = 21
MIDI_HIGH = 108
N_KEYS = MIDI_HIGH - MIDI_LOW + 1  # 88
A4_MIDI = 69


@dataclass(frozen=True)
class TuningConstraints:
    """Inputs that shape a solver run (spec §4.3)."""

    a4: float = 440.0
    temperament: str = "equal"  # "equal" | named historical temperaments (later)
    # midi -> forced cent offset (e.g. {69: 0.0}); always pin A4 unless overridden
    locked_notes: Mapping[int, float] = field(default_factory=dict)
    # Interval name -> weight (higher = more important). See beat_rate defaults.
    interval_weights: Mapping[str, float] = field(default_factory=dict)
    # Shah & Välimäki treble rule: "1:2" (default) or "none"
    treble_rule: str = "1:2"
    # Optional temperament target curve (88 cents vs ET) for regularizer / bias
    temperament_offsets: np.ndarray | None = None

    def locked_array(self) -> np.ndarray:
        """Length-88 array: NaN = free, else locked cent offset."""
        out = np.full(N_KEYS, np.nan, dtype=float)
        # Default A4 pin
        out[A4_MIDI - MIDI_LOW] = 0.0
        for midi, cents in self.locked_notes.items():
            if MIDI_LOW <= int(midi) <= MIDI_HIGH:
                out[int(midi) - MIDI_LOW] = float(cents)
        return out


@dataclass(frozen=True)
class TuningCurve:
    """88 cent offsets vs equal temperament + solver metadata."""

    offsets_cents: np.ndarray  # shape (88,), index 0 = MIDI 21
    solver_name: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.offsets_cents, dtype=float).reshape(-1)
        if arr.shape[0] != N_KEYS:
            raise ValueError(f"TuningCurve must have {N_KEYS} offsets, got {arr.shape[0]}")
        object.__setattr__(self, "offsets_cents", arr)

    @property
    def n_keys(self) -> int:
        return N_KEYS

    def offset_for_midi(self, midi: int) -> float:
        idx = int(midi) - MIDI_LOW
        if 0 <= idx < N_KEYS:
            return float(self.offsets_cents[idx])
        return 0.0

    def as_list(self) -> list[float]:
        return [float(x) for x in self.offsets_cents]


@runtime_checkable
class Solver(Protocol):
    """Swappable tuning-curve algorithm (spec §4.3)."""

    name: str

    def solve(
        self,
        cent_spectra: np.ndarray,
        b_estimates: np.ndarray,
        constraints: TuningConstraints,
    ) -> Iterator[TuningCurve]:
        """
        Yield intermediate tuning curves so the GUI can show progress.

        cent_spectra: (K, M) A-weighted cent-binned SPLA per key (may be zeros
            if the solver only needs B, e.g. beat-rate).
        b_estimates: (K,) measured B per key; NaN where unknown.
        """
        ...
