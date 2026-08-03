"""
BCurveWidget — log-B vs MIDI scatter of measured keys + fitted curve.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

from optitune.model.inharmonicity import fit_log_linear_b

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover
    pg = None

MIDI_LOW = 21
N_KEYS = 88


class BCurveWidget(QWidget):
    """
    Public API:
        set_measured_b({midi: B})
        fitted_b() -> ndarray | None
        clear()
        measured_count()
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(140)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)

        self._measured: dict[int, float] = {}
        self._fit: np.ndarray | None = None

        if pg is not None:
            self._plot = pg.PlotWidget()
            self._plot.setBackground((20, 20, 25))
            self._plot.showGrid(x=True, y=True, alpha=0.3)
            self._plot.setLabel("left", "B (log)", color="#aaa")
            self._plot.setLabel("bottom", "MIDI", color="#aaa")
            self._plot.setLogMode(y=True)
            self._plot.setXRange(MIDI_LOW, MIDI_LOW + N_KEYS - 1, padding=0.02)
            self._plot.addLegend(offset=(8, 8))
            self._fit_item = self._plot.plot(
                pen=pg.mkPen("#c39bd3", width=2), name="fit"
            )
            self._meas_item = self._plot.plot(
                pen=None,
                symbol="t",
                symbolSize=8,
                symbolBrush="#5dade2",
                name="measured B",
            )
            self._layout.addWidget(self._plot)
        else:
            self._plot = None
            self._fit_item = None
            self._meas_item = None

    def set_measured_b(self, measured: Mapping[int, float] | None) -> None:
        self._measured = {
            int(m): float(b)
            for m, b in (measured or {}).items()
            if b is not None and float(b) > 0
        }
        midis = np.arange(MIDI_LOW, MIDI_LOW + N_KEYS)
        if len(self._measured) >= 1:
            self._fit, _, _ = fit_log_linear_b(self._measured)
        else:
            self._fit = None

        if self._meas_item is not None:
            if self._measured:
                xs = np.array(sorted(self._measured.keys()), dtype=float)
                ys = np.array([self._measured[int(m)] for m in xs], dtype=float)
                self._meas_item.setData(xs, ys)
            else:
                self._meas_item.setData([], [])
        if self._fit_item is not None:
            if self._fit is not None:
                self._fit_item.setData(midis, self._fit)
            else:
                self._fit_item.setData([], [])

    def fitted_b(self) -> np.ndarray | None:
        return None if self._fit is None else self._fit.copy()

    def clear(self) -> None:
        self._measured.clear()
        self._fit = None
        if self._meas_item is not None:
            self._meas_item.setData([], [])
        if self._fit_item is not None:
            self._fit_item.setData([], [])

    def measured_count(self) -> int:
        return len(self._measured)
