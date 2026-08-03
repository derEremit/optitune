"""AnalysisWorker runs estimate_pitch off the calling path via direct slot."""

from __future__ import annotations

import numpy as np

from optitune.dsp.analysis_worker import AnalysisWorker
from optitune.dsp.synth import generate_inharmonic_tone


def test_worker_emits_frame_for_tone(qtbot) -> None:
    w = AnalysisWorker()
    results: list = []
    w.frame_ready.connect(lambda d: results.append(d))

    y = generate_inharmonic_tone(60, duration=0.8, fs=48000, B=0.0003, seed=1)
    seg = y[int(0.15 * 48000) : int(0.7 * 48000)]

    with qtbot.waitSignal(w.frame_ready, timeout=10000):
        w.process_buffer(seg, 48000.0)

    assert len(results) == 1
    assert "midi" in results[0]
    assert "f_est" in results[0]
    # Near C4
    assert abs(int(results[0]["midi"]) - 60) <= 2


def test_worker_drops_when_busy(qtbot) -> None:
    w = AnalysisWorker()
    w._busy = True
    results: list = []
    w.frame_ready.connect(lambda d: results.append(d))
    y = np.random.randn(4096).astype(np.float64) * 0.01
    w.process_buffer(y, 48000.0)
    assert results == []
