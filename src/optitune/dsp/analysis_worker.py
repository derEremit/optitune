"""
Background analysis worker (M6 / spec §4.1).

Runs pure estimate_pitch on a QThread; GUI only consumes signals.
Not yet the sole live path — MainWindow can gradually migrate from timers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from optitune.dsp.note_follow import NoteFollowMode
from optitune.dsp.pitch_estimate import estimate_pitch


class AnalysisWorker(QObject):
    """
    Process mono float buffers and emit result dicts (same shape as estimate_pitch).
    """

    frame_ready = Signal(object)  # dict from estimate_pitch
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._a4 = 440.0
        self._armed_midi: int | None = None
        self._last_f0 = 440.0
        self._follow_mode: NoteFollowMode | str = NoteFollowMode.AUTO
        self._locked_midi: int | None = None
        self._busy = False

    @Slot(float)
    def set_a4(self, a4: float) -> None:
        self._a4 = float(a4)

    @Slot(object)
    def set_armed_midi(self, midi: object) -> None:
        self._armed_midi = int(midi) if midi is not None else None

    @Slot(object)
    def set_follow_mode(self, mode: object) -> None:
        if isinstance(mode, NoteFollowMode):
            self._follow_mode = mode
        elif mode is not None:
            self._follow_mode = NoteFollowMode(str(mode))

    @Slot(object)
    def set_locked_midi(self, midi: object) -> None:
        self._locked_midi = int(midi) if midi is not None else None

    @Slot(object, float)
    def process_buffer(self, audio: object, fs: float) -> None:
        """Analyze one mono buffer; skip if already busy (drop frame)."""
        if self._busy:
            return
        self._busy = True
        try:
            arr = np.asarray(audio, dtype=np.float64)
            if arr.size < 256:
                return
            est = estimate_pitch(
                arr,
                float(fs),
                a4=self._a4,
                armed_midi=self._armed_midi,
                last_f0_guess=self._last_f0,
                follow_mode=self._follow_mode,
                locked_midi=self._locked_midi,
            )
            f0 = est.get("f0") or est.get("f_est")
            if f0 is not None and 30 < float(f0) < 6000:
                self._last_f0 = float(f0)
            self.frame_ready.emit(est)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self._busy = False


def start_analysis_thread(
    parent: QObject | None = None,
) -> tuple[QThread, AnalysisWorker]:
    """Create thread + worker; caller must connect signals and call thread.start()."""
    thread = QThread(parent)
    worker = AnalysisWorker()
    worker.moveToThread(thread)
    thread.finished.connect(worker.deleteLater)
    return thread, worker
