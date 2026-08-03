"""Named solver registry for GUI / CLI selection."""

from __future__ import annotations

from optitune.solvers.base import Solver
from optitune.solvers.beat_rate_solver import BeatRateSolver
from optitune.solvers.entropy import EntropySolver
from optitune.solvers.entropy_octave import OctaveEntropySolver

# Factory map: name -> zero-arg constructor
SOLVER_FACTORIES: dict[str, type] = {
    "beat-rate": BeatRateSolver,
    "entropy": EntropySolver,
    "octave-entropy": OctaveEntropySolver,
}


def available_solvers() -> list[str]:
    return list(SOLVER_FACTORIES.keys())


def get_solver(name: str, **kwargs) -> Solver:
    key = str(name).strip().lower()
    if key not in SOLVER_FACTORIES:
        raise KeyError(f"Unknown solver {name!r}; choose from {available_solvers()}")
    cls = SOLVER_FACTORIES[key]
    try:
        return cls(**kwargs)  # type: ignore[call-arg]
    except TypeError:
        return cls()  # type: ignore[call-arg]
