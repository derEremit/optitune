"""
RailsbackWidget — tuning-curve (cent offsets vs MIDI) display.

Shows the computed stretch curve, optional per-key measured deviations,
and an A4 marker. Dark theme consistent with SpectrumWidget.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover
    pg = None

MIDI_LOW = 21
N_KEYS = 88


class RailsbackWidget(QWidget):
    """
    Public API:
        set_tuning_curve(offsets: array-like length 88)
        set_measured_deviations({midi: cents})
        set_a4_marker(bool)
        clear()
        curve_points() / measured_count()  # for tests
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(140)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)

        self._curve: np.ndarray | None = None
        self._measured: dict[int, float] = {}
        self._show_a4 = True

        if pg is not None:
            self._plot = pg.PlotWidget()
            self._plot.setBackground((20, 20, 25))
            self._plot.showGrid(x=True, y=True, alpha=0.3)
            self._plot.setLabel("left", "cents vs ET", color="#aaa")
            self._plot.setLabel("bottom", "MIDI", color="#aaa")
            self._plot.setXRange(MIDI_LOW, MIDI_LOW + N_KEYS - 1, padding=0.02)
            self._plot.setYRange(-25, 25, padding=0.05)
            self._plot.addLegend(offset=(8, 8))
            self._curve_item = self._plot.plot(
                pen=pg.mkPen("#5fa8ff", width=2), name="curve"
            )
            self._meas_item = self._plot.plot(
                pen=None,
                symbol="o",
                symbolSize=7,
                symbolBrush="#f0c040",
                name="measured",
            )
            self._a4_line = pg.InfiniteLine(
                pos=69, angle=90, pen=pg.mkPen("#7dcea0", style=Qt.PenStyle.DashLine)
            )
            self._plot.addItem(self._a4_line)
            self._zero = pg.InfiniteLine(
                pos=0, angle=0, pen=pg.mkPen("#444", width=1)
            )
            self._plot.addItem(self._zero)
            self._layout.addWidget(self._plot)
        else:
            self._plot = None
            self._curve_item = None
            self._meas_item = None
            self._a4_line = None

    def set_tuning_curve(self, offsets: np.ndarray | list[float] | None) -> None:
        if offsets is None:
            self._curve = None
            if self._curve_item is not None:
                self._curve_item.setData([], [])
            return
        arr = np.asarray(offsets, dtype=float).reshape(-1)
        if arr.shape[0] != N_KEYS:
            # pad/trim
            out = np.zeros(N_KEYS, dtype=float)
            n = min(N_KEYS, arr.shape[0])
            out[:n] = arr[:n]
            arr = out
        self._curve = arr
        midis = np.arange(MIDI_LOW, MIDI_LOW + N_KEYS)
        if self._curve_item is not None:
            self._curve_item.setData(midis, arr)
            # Autoscale Y with padding if extremes are large
            lo, hi = float(np.min(arr)), float(np.max(arr))
            pad = max(3.0, 0.15 * (hi - lo + 1e-6))
            if self._plot is not None:
                self._plot.setYRange(lo - pad, hi + pad, padding=0)

    def set_measured_deviations(self, measured: Mapping[int, float] | None) -> None:
        self._measured = {int(m): float(c) for m, c in (measured or {}).items()}
        if self._meas_item is None:
            return
        if not self._measured:
            self._meas_item.setData([], [])
            return
        xs = np.array(sorted(self._measured.keys()), dtype=float)
        ys = np.array([self._measured[int(m)] for m in xs], dtype=float)
        self._meas_item.setData(xs, ys)

    def set_a4_marker(self, show: bool = True) -> None:
        self._show_a4 = bool(show)
        if self._a4_line is not None:
            self._a4_line.setVisible(self._show_a4)

    def clear(self) -> None:
        self._curve = None
        self._measured.clear()
        if self._curve_item is not None:
            self._curve_item.setData([], [])
        if self._meas_item is not None:
            self._meas_item.setData([], [])

    def curve_points(self) -> np.ndarray | None:
        return None if self._curve is None else self._curve.copy()

    def measured_count(self) -> int:
        return len(self._measured)
