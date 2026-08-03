"""pytest-qt: SolverWorker streams curves off the GUI thread."""

from __future__ import annotations

import numpy as np

from optitune.solvers.base import N_KEYS, TuningConstraints
from optitune.solvers.worker import SolverWorker


def test_worker_beat_rate_finishes(qtbot) -> None:
    worker = SolverWorker()
    L = np.zeros((N_KEYS, 8), dtype=float)
    b = np.full(N_KEYS, np.nan)
    # a few B values
    b[15] = 1e-4
    b[30] = 3e-4
    b[45] = 8e-4
    c = TuningConstraints()

    results: list = []
    worker.finished.connect(lambda tc: results.append(tc))

    with qtbot.waitSignal(worker.finished, timeout=5000):
        worker.start_solve("beat-rate", L, b, c, {})

    assert len(results) == 1
    assert results[0] is not None
    assert results[0].n_keys == 88
    assert results[0].offset_for_midi(69) == 0.0 or abs(results[0].offset_for_midi(69)) < 0.05
