"""
BeatRateSolver — Solver protocol adapter over compute_basic_tuning_curve.

Beat-rate LS uses measured B only (cent spectra ignored until entropy lands).
Yields a single final TuningCurve (no intermediate iterates exposed yet).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from optitune.model import Key, Piano
from optitune.solvers.base import (
    MIDI_LOW,
    N_KEYS,
    TuningConstraints,
    TuningCurve,
)
from optitune.solvers.beat_rate import compute_basic_tuning_curve


class BeatRateSolver:
    """Weighted beat-rate least-squares solver (spec §6.2) behind Solver protocol."""

    name: str = "beat-rate"

    def solve(
        self,
        cent_spectra: np.ndarray,
        b_estimates: np.ndarray,
        constraints: TuningConstraints,
    ) -> Iterator[TuningCurve]:
        # Build a temporary Piano from B estimates so we reuse the proven LS path.
        piano = self._piano_from_b(b_estimates, a4=float(constraints.a4))
        weights = dict(constraints.interval_weights) if constraints.interval_weights else None
        shah_weight = 320.0 if constraints.treble_rule == "1:2" else 0.0
        curve_list = compute_basic_tuning_curve(
            piano,
            interval_weights=weights,
            shah_weight=shah_weight,
        )
        # Apply explicit locked notes (A4 already pinned inside LS)
        offs = np.asarray(curve_list, dtype=float).copy()
        for midi, cents in constraints.locked_notes.items():
            idx = int(midi) - MIDI_LOW
            if 0 <= idx < N_KEYS:
                offs[idx] = float(cents)
        # Ensure A4 pin unless user overrode
        if 69 not in constraints.locked_notes:
            offs[69 - MIDI_LOW] = 0.0

        # Optional temperament table (cents vs ET) layered on stretch
        if constraints.temperament_offsets is not None:
            t_off = np.asarray(constraints.temperament_offsets, dtype=float).reshape(-1)
            if t_off.shape[0] == N_KEYS:
                offs = offs + t_off
                if 69 not in constraints.locked_notes:
                    offs[69 - MIDI_LOW] = 0.0

        yield TuningCurve(
            offsets_cents=offs,
            solver_name=self.name,
            metadata={
                "a4": float(constraints.a4),
                "treble_rule": constraints.treble_rule,
                "temperament": constraints.temperament,
                "measured_b_count": int(np.sum(np.isfinite(b_estimates))),
            },
        )

    def solve_piano(self, piano: Piano, constraints: TuningConstraints | None = None) -> TuningCurve:
        """Convenience: run from a Piano model (GUI path)."""
        c = constraints or TuningConstraints(a4=float(piano.a4))
        if abs(c.a4 - float(piano.a4)) > 1e-9:
            c = TuningConstraints(
                a4=float(piano.a4),
                temperament=c.temperament,
                locked_notes=c.locked_notes,
                interval_weights=c.interval_weights,
                treble_rule=c.treble_rule,
                temperament_offsets=c.temperament_offsets,
            )
        b_est = np.full(N_KEYS, np.nan, dtype=float)
        for m, k in piano.keys.items():
            if k.measured_b is not None and 1e-6 < float(k.measured_b) < 1.0:
                b_est[int(m) - MIDI_LOW] = float(k.measured_b)
        # Spectra unused by beat-rate
        spectra = np.zeros((N_KEYS, 1), dtype=float)
        return next(self.solve(spectra, b_est, c))

    @staticmethod
    def _piano_from_b(b_estimates: np.ndarray, *, a4: float) -> Piano:
        p = Piano(name="solver-scratch", a4=a4)
        b = np.asarray(b_estimates, dtype=float).reshape(-1)
        if b.shape[0] != N_KEYS:
            raise ValueError(f"b_estimates must have length {N_KEYS}, got {b.shape[0]}")
        for i, bi in enumerate(b):
            if not np.isfinite(bi):
                continue
            midi = MIDI_LOW + i
            f0 = a4 * (2.0 ** ((midi - 69) / 12.0))
            p.set_key(Key(midi=midi, measured_f0=float(f0), measured_b=float(bi)))
        return p
