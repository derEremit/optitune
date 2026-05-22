"""
SpectrumWidget — real-time spectrum / cent-binned power display using PyQtGraph.

Phase 3: Now functional — plots live magnitude spectrum (dB) from analysis,
with a vertical marker line at the currently detected dominant / f0 frequency.
Dark professional theme, fast curve updates.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover
    pg = None


class SpectrumWidget(QWidget):
    """
    Real-time spectrum display (PyQtGraph).

    Public API (kept stable):
        update_frame(x: np.ndarray, y: np.ndarray)   # x=freqs_Hz or cent_bins, y=power or db
        set_partial_markers(frequencies: list[float])
        set_note_label(name: str)
        clear()
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(160)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(2)

        self._detected_f: float = 0.0
        self._note_label_text = ""

        if pg is not None:
            # Create plot
            self._plot = pg.PlotWidget()
            self._plot.setBackground((20, 20, 25))
            self._plot.showGrid(x=True, y=True, alpha=0.3)
            self._plot.setLabel("left", "Level (dB)", color="#aaa")
            self._plot.setLabel("bottom", "Frequency (Hz)", color="#aaa")
            self._plot.setXRange(40, 4500, padding=0)
            self._plot.setYRange(-80, 5, padding=0)

            # Theme the axis
            axis_pen = pg.mkPen(color=(120, 120, 130))
            self._plot.getAxis("left").setPen(axis_pen)
            self._plot.getAxis("bottom").setPen(axis_pen)
            self._plot.getAxis("left").setTextPen("#ccc")
            self._plot.getAxis("bottom").setTextPen("#ccc")

            # Main spectrum curve (yellow-green for visibility)
            self._curve = self._plot.plot(
                pen=pg.mkPen(color=(120, 220, 160), width=1.2),
                name="Spectrum",
            )

            # Detected pitch marker (bright vertical line + label)
            self._marker = pg.InfiniteLine(
                pos=440.0,
                angle=90,
                pen=pg.mkPen(color=(255, 80, 80), width=2.0, style=Qt.PenStyle.DashLine),
                label="f0",
                labelOpts={"color": (255, 100, 100), "position": 0.95, "movable": False},
            )
            self._plot.addItem(self._marker)

            # Optional partial markers (small ticks)
            self._partial_items: list = []

            self._layout.addWidget(self._plot)
        else:
            # Fallback if pyqtgraph missing (should not happen)
            from PySide6.QtWidgets import QLabel

            self._plot = None
            self._fallback = QLabel("Spectrum (pyqtgraph unavailable)")
            self._fallback.setStyleSheet("background:#111; color:#666; font-size:12px;")
            self._layout.addWidget(self._fallback)

    def update_frame(self, x: np.ndarray | list, y: np.ndarray | list) -> None:
        """Update the spectrum plot. x may be frequencies in Hz; y linear power or dB-ish."""
        if self._plot is None or pg is None:
            return
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            if len(x_arr) < 3 or len(y_arr) < 3:
                return

            # Convert y to dB if it looks like linear power
            if np.max(y_arr) > 1e-3 and np.min(y_arr) >= 0:
                y_db = 10.0 * np.log10(np.maximum(y_arr, 1e-12))
            else:
                y_db = y_arr

            # Downsample lightly for speed if huge
            if len(x_arr) > 1200:
                step = len(x_arr) // 800
                x_arr = x_arr[::step]
                y_db = y_db[::step]

            # Plot only the musical range we care about
            mask = (x_arr >= 30) & (x_arr <= 6000)
            if np.any(mask):
                x_plot = x_arr[mask]
                y_plot = y_db[mask]
            else:
                x_plot, y_plot = x_arr, y_db

            self._curve.setData(x_plot, y_plot)

            # Update marker if we have a detected f
            if self._detected_f > 20:
                self._marker.setPos(self._detected_f)
                pass  # label set at construction

            if self._note_label_text:
                self._plot.setTitle(self._note_label_text, color="#aaa")
        except Exception:
            pass  # never crash GUI from spectrum update

    def set_detected_pitch(self, f_hz: float) -> None:
        """Move the red marker to the tracked pitch."""
        self._detected_f = float(f_hz)
        if self._plot is not None and pg is not None and f_hz > 20:
            self._marker.setPos(f_hz)
            pass  # label via opts

    def set_partial_markers(self, frequencies_hz: list[float]) -> None:
        """Draw small vertical ticks for expected partials of locked note."""
        if self._plot is None or pg is None:
            return
        # Remove old
        for item in self._partial_items:
            self._plot.removeItem(item)
        self._partial_items.clear()

        for f in frequencies_hz[:12]:  # limit
            if 30 < f < 5500:
                ln = pg.InfiniteLine(
                    pos=float(f),
                    angle=90,
                    pen=pg.mkPen(color=(80, 160, 255), width=1, style=Qt.PenStyle.DotLine),
                )
                self._plot.addItem(ln)
                self._partial_items.append(ln)

    def set_note_label(self, name: str) -> None:
        self._note_label_text = str(name)
        if self._plot is not None and pg is not None:
            self._plot.setTitle(name or "", color="#aaa")

    def clear(self) -> None:
        if self._plot is not None and pg is not None:
            self._curve.setData([], [])
            self._marker.setPos(0)
            self._marker.setLabel("")
            for item in self._partial_items:
                self._plot.removeItem(item)
            self._partial_items.clear()
            self._plot.setTitle("")
        self._detected_f = 0.0
        self._note_label_text = ""
