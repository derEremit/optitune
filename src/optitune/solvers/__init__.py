"""Tuning solvers: beat-rate LS (primary), legacy heuristic, etc.

`compute_basic_tuning_curve` is now the production-grade iterative
weighted beat-rate least-squares solver (spec §6.2).

The original fast heuristic remains available as
`compute_heuristic_stretch_curve` for comparisons and diagnostics.
"""

from __future__ import annotations

from .beat_rate import (
    apply_curve_to_piano,
    compute_basic_tuning_curve,
    compute_beat_rate_for_interval,
)
from .simple_stretch import compute_heuristic_stretch_curve

__all__ = [
    "apply_curve_to_piano",
    "compute_basic_tuning_curve",
    "compute_beat_rate_for_interval",
    "compute_heuristic_stretch_curve",
]
