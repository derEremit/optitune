"""
Background solver worker (spec §4.1): runs Solver.solve on a QThread and
streams intermediate TuningCurves to the GUI via signals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from optitune.solvers.base import TuningConstraints, TuningCurve
from optitune.solvers.registry import get_solver


class SolverWorker(QObject):
    """
    Lives on a QThread. Call `start_solve` via queued connection from the GUI.
    Emits progress for each intermediate curve and finished with the last one.
    """

    progress = Signal(object)  # TuningCurve
    finished = Signal(object)  # TuningCurve | None
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cancel = False

    @Slot()
    def request_cancel(self) -> None:
        self._cancel = True

    @Slot(str, object, object, object, object)
    def start_solve(
        self,
        solver_name: str,
        cent_spectra: object,
        b_estimates: object,
        constraints: object,
        solver_kwargs: object = None,
    ) -> None:
        self._cancel = False
        try:
            kwargs = dict(solver_kwargs) if isinstance(solver_kwargs, dict) else {}
            solver = get_solver(str(solver_name), **kwargs)
            L = np.asarray(cent_spectra, dtype=float)
            b = np.asarray(b_estimates, dtype=float)
            if not isinstance(constraints, TuningConstraints):
                constraints = TuningConstraints()
            self.status.emit(f"Running {solver.name}…")
            last: TuningCurve | None = None
            for tc in solver.solve(L, b, constraints):
                if self._cancel:
                    self.status.emit("Solver cancelled")
                    self.finished.emit(last)
                    return
                last = tc
                self.progress.emit(tc)
            self.status.emit(f"{solver.name} done")
            self.finished.emit(last)
        except Exception as exc:
            self.failed.emit(str(exc))
            self.finished.emit(None)


def run_solver_in_thread(
    solver_name: str,
    cent_spectra: np.ndarray,
    b_estimates: np.ndarray,
    constraints: TuningConstraints,
    *,
    solver_kwargs: dict[str, Any] | None = None,
    parent: QObject | None = None,
) -> tuple[QThread, SolverWorker]:
    """
    Create thread+worker, start the thread, and queue start_solve.
    Caller must connect finished/failed and keep refs until done.
    """
    thread = QThread(parent)
    worker = SolverWorker()
    worker.moveToThread(thread)
    thread.started.connect(
        lambda: worker.start_solve(
            solver_name,
            cent_spectra,
            b_estimates,
            constraints,
            solver_kwargs or {},
        )
    )
    worker.finished.connect(thread.quit)
    worker.failed.connect(lambda _msg: thread.quit())
    return thread, worker
