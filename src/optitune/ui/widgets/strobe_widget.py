"""
StrobeWidget — rotating strobe for visual tuning (Peterson-style).

Primary ring tracks fundamental error (Hz). Optional multi-partial mode draws
concentric rings for partials 1..N with independent phase rates.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget


class StrobeWidget(QWidget):
    """
    Rotating strobe display for piano tuning.

    API:
        set_phase_delta_hz(delta_hz)          # primary / fundamental
        set_target_frequency(f_hz)
        set_partial_number(n)
        set_multi_partial_enabled(bool)
        set_partial_deltas([(n, delta_hz), ...])
        set_running(enabled)
    """

    strobe_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.setMaximumSize(420, 420)

        self._phase_delta_hz: float = 0.0
        self._target_hz: float = 440.0
        self._partial: int = 1
        self._running: bool = True
        self._angle_deg: float = 0.0
        self._multi_partial: bool = False
        # list of (partial_number, delta_hz, angle_deg)
        self._rings: list[tuple[int, float, float]] = [(1, 0.0, 0.0)]

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        self.setToolTip(
            "Strobe — stationary = in tune. Clockwise = sharp. "
            "Multi-partial mode: outer ring = fundamental, inner = higher partials."
        )

    def set_phase_delta_hz(self, delta_hz: float) -> None:
        """How many Hz the measured tone is away from target (positive = sharp)."""
        self._phase_delta_hz = float(delta_hz)
        if not self._multi_partial or len(self._rings) <= 1:
            # Keep single-ring mode in sync
            ang = self._rings[0][2] if self._rings else 0.0
            self._rings = [(self._partial, self._phase_delta_hz, ang)]

    def set_target_frequency(self, f_hz: float) -> None:
        self._target_hz = float(f_hz)

    def set_partial_number(self, n: int) -> None:
        self._partial = max(1, int(n))

    def set_multi_partial_enabled(self, enabled: bool) -> None:
        self._multi_partial = bool(enabled)
        if not self._multi_partial:
            ang = self._rings[0][2] if self._rings else self._angle_deg
            self._rings = [(1, self._phase_delta_hz, ang)]
            self.update()

    def set_partial_deltas(self, deltas: list[tuple[int, float]]) -> None:
        """
        deltas: list of (partial_number, delta_hz). Used when multi-partial is on.
        Preserves angles for matching partial numbers.
        """
        if not deltas:
            self._rings = [(1, self._phase_delta_hz, self._angle_deg)]
            return
        old = {n: a for n, _d, a in self._rings}
        rings: list[tuple[int, float, float]] = []
        for n, d in deltas:
            nn = max(1, int(n))
            rings.append((nn, float(d), float(old.get(nn, 0.0))))
        rings.sort(key=lambda t: t[0])
        self._rings = rings
        # Primary stays partial 1 if present
        for n, d, _a in self._rings:
            if n == 1:
                self._phase_delta_hz = d
                break
        self.update()

    def partial_count(self) -> int:
        if self._multi_partial and self._rings:
            return len(self._rings)
        return 1

    def set_running(self, enabled: bool) -> None:
        self._running = bool(enabled)
        if not enabled:
            self._angle_deg = 0.0
            self._rings = [(n, d, 0.0) for n, d, _a in self._rings]
            self.update()

    def reset(self) -> None:
        self._phase_delta_hz = 0.0
        self._angle_deg = 0.0
        self._rings = [(1, 0.0, 0.0)]
        self.update()

    def _on_tick(self) -> None:
        if not self._running:
            return
        new_rings: list[tuple[int, float, float]] = []
        for n, d, ang in self._rings:
            # Higher partials: same cents → higher Hz error, slightly faster visual
            speed = float(d) * 18.0
            ang = (ang + speed) % 360.0
            new_rings.append((n, d, ang))
        self._rings = new_rings
        if self._rings:
            self._angle_deg = self._rings[0][2]
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(8, 8, -8, -8)
        center = rect.center()
        radius = min(rect.width(), rect.height()) / 2 - 4

        painter.setPen(QPen(QColor(45, 45, 52), 3))
        painter.setBrush(QBrush(QColor(22, 22, 28)))
        painter.drawEllipse(center, radius, radius)

        rings = self._rings if (self._multi_partial and self._rings) else [
            (self._partial, self._phase_delta_hz, self._angle_deg)
        ]
        n_rings = max(1, len(rings))
        colors = [
            QColor(70, 160, 255),
            QColor(120, 200, 140),
            QColor(240, 180, 80),
            QColor(200, 120, 220),
        ]

        for i, (pn, _d, ang0) in enumerate(rings):
            # Outer ring = fundamental (first); inners shrink
            r_outer = radius * (1.0 - 0.12 * i)
            r_inner = radius * (0.35 + 0.08 * i)
            if r_outer <= r_inner + 4:
                continue
            col = colors[i % len(colors)]
            painter.setPen(QPen(col, 2.2 if i == 0 else 1.6))
            segs = 12
            for s in range(segs):
                angle = (ang0 + s * (360.0 / segs)) * (math.pi / 180.0)
                x1 = center.x() + r_inner * math.cos(angle)
                y1 = center.y() + r_inner * math.sin(angle)
                x2 = center.x() + r_outer * math.cos(angle)
                y2 = center.y() + r_outer * math.sin(angle)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        painter.setPen(QPen(QColor(200, 200, 210), 1))
        painter.setBrush(QBrush(QColor(70, 160, 255)))
        painter.drawEllipse(center, 6, 6)

        painter.setPen(QColor(180, 180, 190))
        mode = f"  ×{n_rings}" if self._multi_partial and n_rings > 1 else f"  (p{self._partial})"
        painter.drawText(
            rect.adjusted(0, int(radius * 0.55), 0, 0),
            Qt.AlignmentFlag.AlignHCenter,
            f"{self._target_hz:.1f} Hz{mode}",
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.strobe_clicked.emit()
        super().mousePressEvent(event)
