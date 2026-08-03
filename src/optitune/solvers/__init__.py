"""Tuning solvers: beat-rate LS (primary), legacy heuristic, protocol types.

`compute_basic_tuning_curve` remains the low-level beat-rate API.
`BeatRateSolver` implements the swappable `Solver` protocol (spec §4.3).
"""

from __future__ import annotations

from .base import Solver, TuningConstraints, TuningCurve
from .beat_rate import (
    apply_curve_to_piano,
    compute_basic_tuning_curve,
    compute_beat_rate_for_interval,
)
from .beat_rate_solver import BeatRateSolver
from .simple_stretch import compute_heuristic_stretch_curve

__all__ = [
    "BeatRateSolver",
    "Solver",
    "TuningConstraints",
    "TuningCurve",
    "apply_curve_to_piano",
    "compute_basic_tuning_curve",
    "compute_beat_rate_for_interval",
    "compute_heuristic_stretch_curve",
]
