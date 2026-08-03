"""
Octave-local entropy solver (Szwajcowski–Pilch style, spec §6.3).

Outward from A4: for each key, try a small grid of cent shifts and pick the
value that minimizes the entropy of the *local* pair (key + A4 or nearest
already-set neighbor). Deterministic, ~50 trials per key.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from optitune.solvers.base import A4_MIDI, MIDI_LOW, N_KEYS, TuningConstraints, TuningCurve
from optitune.solvers.entropy import _entropy


class OctaveEntropySolver:
    """Coordinate-wise entropy on expanding octaves from A4."""

    name: str = "octave-entropy"

    def __init__(
        self,
        *,
        step_cents: int = 1,
        max_shift: int = 25,
        n_trials: int = 51,
    ) -> None:
        self.step_cents = int(step_cents)
        self.max_shift = int(max_shift)
        self.n_trials = int(n_trials)

    def solve(
        self,
        cent_spectra: np.ndarray,
        b_estimates: np.ndarray,
        constraints: TuningConstraints,
    ) -> Iterator[TuningCurve]:
        L = np.asarray(cent_spectra, dtype=np.float64)
        if L.ndim != 2 or L.shape[0] != N_KEYS:
            raise ValueError(f"cent_spectra must be ({N_KEYS}, M), got {L.shape}")
        shifts = np.zeros(N_KEYS, dtype=int)
        a4_idx = A4_MIDI - MIDI_LOW
        shifts[a4_idx] = 0

        # Order: A4, then ±1, ±2, … outward
        order: list[int] = [a4_idx]
        for d in range(1, N_KEYS):
            for idx in (a4_idx - d, a4_idx + d):
                if 0 <= idx < N_KEYS:
                    order.append(idx)

        row_e = np.sum(L, axis=1)
        active = row_e > 1e-12 * max(float(np.max(row_e)), 1.0)

        grid = np.linspace(-self.max_shift, self.max_shift, self.n_trials)
        grid = np.unique(np.round(grid / self.step_cents).astype(int) * self.step_cents)

        for k in order:
            if k == a4_idx or not active[k]:
                continue
            # Neighbor already set: prefer lower index if both, else A4
            neighbors = [j for j in order[: order.index(k)] if active[j] or j == a4_idx]
            ref = neighbors[-1] if neighbors else a4_idx
            best_s = 0
            best_H = float("inf")
            base_ref = np.roll(L[ref], -int(shifts[ref]))
            for s in grid:
                p = base_ref + np.roll(L[k], -int(s))
                H = _entropy(p)
                if H < best_H:
                    best_H = H
                    best_s = int(s)
            shifts[k] = best_s

        offs = shifts.astype(float)
        offs -= offs[a4_idx]
        offs[a4_idx] = 0.0
        for midi, cents in constraints.locked_notes.items():
            idx = int(midi) - MIDI_LOW
            if 0 <= idx < N_KEYS:
                offs[idx] = float(cents)

        yield TuningCurve(
            offsets_cents=offs,
            solver_name=self.name,
            metadata={"n_active": int(np.sum(active))},
        )
