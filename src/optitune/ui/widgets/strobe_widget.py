"""
StrobeWidget — Phase 3+ implementation target.

A buttery-smooth rotating strobe display (60 fps target) for visual tuning.
Classic Peterson-style or modern vector look.

When the measured frequency exactly matches the target, the pattern should appear stationary.
Sharp → clockwise rotation, flat → counter-clockwise.

This is currently a high-quality stub. Real implementation will use QPainter + QTimer
(or a high-priority QThread + QElapsedTimer) driven by phase delta from the AnalysisWorker.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget


class StrobeWidget(QWidget):
    """
    Rotating strobe display for piano tuning.

    API the rest of the app will use:
        set_phase_delta_hz(delta_hz: float)   # + = sharp, - = flat, 0 = in tune
        set_target_frequency(f_hz: float)
        set_partial_number(n: int)            # 1 = fundamental, 2 = octave, etc.
        set_running(enabled: bool)
    """

    # Emitted when user interacts (future: tap-to-zero, sensitivity)
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

        # 60 fps update timer (real implementation may drive this from a dedicated thread)
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        self.setToolTip("Strobe — stationary = in tune. Clockwise = sharp.")

    # ---------------- Public API ----------------

    def set_phase_delta_hz(self, delta_hz: float) -> None:
        """How many Hz the measured tone is away from target (positive = sharp)."""
        self._phase_delta_hz = float(delta_hz)

    def set_target_frequency(self, f_hz: float) -> None:
        self._target_hz = float(f_hz)

    def set_partial_number(self, n: int) -> None:
        self._partial = max(1, int(n))

    def set_running(self, enabled: bool) -> None:
        self._running = bool(enabled)
        if not enabled:
            self._angle_deg = 0.0
            self.update()

    def reset(self) -> None:
        self._phase_delta_hz = 0.0
        self._angle_deg = 0.0
        self.update()

    # ---------------- Internal ----------------

    def _on_tick(self) -> None:
        if not self._running:
            return

        # Simple visual model: rotation speed proportional to delta
        # In real code this will come from analytic signal / Goertzel phase accumulation
        rotation_speed = self._phase_delta_hz * 18.0  # degrees per frame (tuned visually)
        self._angle_deg = (self._angle_deg + rotation_speed) % 360.0
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(8, 8, -8, -8)
        center = rect.center()
        radius = min(rect.width(), rect.height()) / 2 - 4

        # Background disk
        painter.setPen(QPen(QColor(45, 45, 52), 3))
        painter.setBrush(QBrush(QColor(22, 22, 28)))
        painter.drawEllipse(center, radius, radius)

        # Strobe pattern (simple 12-segment wheel for the stub)
        painter.setPen(QPen(QColor(70, 160, 255), 2.5))
        for i in range(12):
            angle = (self._angle_deg + i * 30.0) * (3.14159 / 180.0)
            x1 = center.x() + (radius * 0.35) * __import__("math").cos(angle)
            y1 = center.y() + (radius * 0.35) * __import__("math").sin(angle)
            x2 = center.x() + (radius * 0.92) * __import__("math").cos(angle)
            y2 = center.y() + (radius * 0.92) * __import__("math").sin(angle)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Center dot + frequency label (stub)
        painter.setPen(QPen(QColor(200, 200, 210), 1))
        painter.setBrush(QBrush(QColor(70, 160, 255)))
        painter.drawEllipse(center, 6, 6)

        painter.setPen(QColor(180, 180, 190))
        painter.drawText(
            rect.adjusted(0, int(radius * 0.55), 0, 0),
            Qt.AlignmentFlag.AlignHCenter,
            f"{self._target_hz:.1f} Hz  (p{self._partial})",
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.strobe_clicked.emit()
        super().mousePressEvent(event)
