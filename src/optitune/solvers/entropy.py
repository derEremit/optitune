"""
Entropy-minimizing tuning-curve solver (spec §5 / Hinrichsen).

Zero-temperature Monte Carlo: randomly propose ±step_cents shifts per key,
accept only when Shannon entropy of the summed rolled spectra decreases.
Deterministic given seed. Yields intermediate TuningCurves so the GUI can stream.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from optitune.solvers.base import (
    A4_MIDI,
    MIDI_LOW,
    N_KEYS,
    TuningConstraints,
    TuningCurve,
)
from optitune.solvers.simple_stretch import _default_railsback_curve


def _entropy(p: np.ndarray) -> float:
    # Clip numerical undershoot from incremental roll updates
    p = np.maximum(p, 0.0)
    s = float(np.sum(p))
    if s <= 0:
        return 0.0
    pn = p / s
    # Stable Shannon entropy (nats)
    return float(-np.sum(pn * np.log(pn + 1e-30)))


def _shifted_sum(L: np.ndarray, shifts: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Sum rolled rows for active keys."""
    M = L.shape[1]
    p = np.zeros(M, dtype=np.float64)
    for k in np.nonzero(active)[0]:
        p += np.roll(L[k], -int(shifts[k]))
    return p


class EntropySolver:
    """Hinrichsen-style entropy minimizer behind the Solver protocol."""

    name: str = "entropy"

    def __init__(
        self,
        *,
        seed: int = 0,
        max_passes: int = 20,
        step_cents: int = 1,
        eps: float = 1e-12,
        active_only: bool = True,
        railsback_prior: float = 0.0,
        yield_every: int = 500,
        max_shift: int = 40,
    ) -> None:
        self.seed = int(seed)
        self.max_passes = int(max_passes)
        self.step_cents = int(step_cents)
        self.eps = float(eps)
        self.active_only = bool(active_only)
        self.railsback_prior = float(railsback_prior)
        self.yield_every = max(1, int(yield_every))
        self.max_shift = int(max_shift)

    def solve(
        self,
        cent_spectra: np.ndarray,
        b_estimates: np.ndarray,
        constraints: TuningConstraints,
    ) -> Iterator[TuningCurve]:
        L = np.asarray(cent_spectra, dtype=np.float64)
        if L.ndim != 2 or L.shape[0] != N_KEYS:
            raise ValueError(f"cent_spectra must be ({N_KEYS}, M), got {L.shape}")
        K, M = L.shape
        # Row energy for active set
        row_e = np.sum(L, axis=1)
        if self.active_only:
            active = row_e > (1e-12 * max(float(np.max(row_e)), 1.0))
        else:
            active = np.ones(K, dtype=bool)
        # Always keep A4 free to pin later unless no energy anywhere
        if not np.any(active):
            # Nothing to optimize — pure ET / prior
            offs = self._final_offsets(np.zeros(K, dtype=int), constraints)
            yield TuningCurve(offsets_cents=offs, solver_name=self.name, metadata={"empty": True})
            return

        active_idx = np.nonzero(active)[0]
        n_active = len(active_idx)

        # Initial shifts: optional mild Railsback prior (integer cents)
        shifts = np.zeros(K, dtype=int)
        if self.railsback_prior > 0:
            prior = np.asarray(_default_railsback_curve(), dtype=float)
            shifts = np.round(self.railsback_prior * prior).astype(int)
            shifts = np.clip(shifts, -self.max_shift, self.max_shift)

        # Locked notes (including default A4 pin handled at the end)
        locked = constraints.locked_array()  # NaN free, else forced
        for i in range(K):
            if np.isfinite(locked[i]):
                shifts[i] = int(round(float(locked[i])))
                active[i] = False  # don't move locked keys
        active_idx = np.nonzero(active)[0]
        n_active = max(len(active_idx), 1)

        rng = np.random.default_rng(self.seed)
        p = _shifted_sum(L, shifts, active if self.active_only else np.ones(K, dtype=bool))
        # Include locked keys that still have spectra
        locked_mask = np.isfinite(locked)
        if np.any(locked_mask):
            for k in np.nonzero(locked_mask)[0]:
                if row_e[k] > 0:
                    p += np.roll(L[k], -int(shifts[k]))
        H = _entropy(p)

        no_accepts = 0
        trials = 0
        accepts = 0
        # Cap total trials: max_passes * K * factor
        max_trials = max(self.max_passes * K * 4, K * 8)
        stop_after = max(n_active, 2)

        # Keys we may pick from
        pick_from = active_idx if len(active_idx) else np.arange(K)

        while no_accepts < stop_after and trials < max_trials:
            k = int(rng.choice(pick_from))
            delta = int(rng.choice([-self.step_cents, self.step_cents]))
            new_shift = int(shifts[k]) + delta
            if abs(new_shift) > self.max_shift:
                no_accepts += 1
                trials += 1
                continue

            # Incremental p update (clip after to kill float dust)
            p_new = p - np.roll(L[k], -int(shifts[k])) + np.roll(L[k], -new_shift)
            p_new = np.maximum(p_new, 0.0)
            H_new = _entropy(p_new)
            trials += 1

            if H_new < H - self.eps:
                shifts[k] = new_shift
                p = p_new
                H = H_new
                no_accepts = 0
                accepts += 1
                if accepts % self.yield_every == 0:
                    offs = self._final_offsets(shifts, constraints)
                    yield TuningCurve(
                        offsets_cents=offs,
                        solver_name=self.name,
                        metadata={"H": H, "accepts": accepts, "trials": trials, "partial": True},
                    )
            else:
                no_accepts += 1

        offs = self._final_offsets(shifts, constraints)
        yield TuningCurve(
            offsets_cents=offs,
            solver_name=self.name,
            metadata={
                "H": H,
                "accepts": accepts,
                "trials": trials,
                "seed": self.seed,
                "n_active": int(np.sum(active)),
            },
        )

    def _final_offsets(self, shifts: np.ndarray, constraints: TuningConstraints) -> np.ndarray:
        offs = shifts.astype(float).copy()
        # Pin A4 unless user locked otherwise
        a4_idx = A4_MIDI - MIDI_LOW
        if A4_MIDI not in constraints.locked_notes:
            offs -= offs[a4_idx]
            offs[a4_idx] = 0.0
        for midi, cents in constraints.locked_notes.items():
            idx = int(midi) - MIDI_LOW
            if 0 <= idx < N_KEYS:
                offs[idx] = float(cents)
        return offs
