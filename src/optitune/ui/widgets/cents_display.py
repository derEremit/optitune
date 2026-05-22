"""
CentsDisplay — large, color-coded cents deviation readout.

Shows ± cents from target with big, readable typography suitable for viewing
from the piano bench (Phase 3+ target: 90–140 pt font).

Color logic (per plan):
    |error| < 3¢   → bright green
    3–10¢          → yellow/amber
    > 10¢          → red

This is currently a high-quality stub with the public API the tuner will drive.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QLabel, QSizePolicy


class CentsDisplay(QLabel):
    """
    Big numeric cents display.

    Public API:
        set_cents(cents: float)
        set_in_tune_threshold(threshold: float = 3.0)
        reset()
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._cents: float = 0.0
        self._threshold = 3.0

        # Big, readable font (will be tuned further in Phase 3)
        font = QFont("DejaVu Sans Mono", 92, QFont.Weight.Bold)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        self.setFont(font)

        self.setMinimumHeight(140)
        self.reset()

    def set_cents(self, cents: float) -> None:
        self._cents = float(cents)
        self._update_display()

    def set_in_tune_threshold(self, threshold: float) -> None:
        self._threshold = max(0.5, float(threshold))
        self._update_display()

    def reset(self) -> None:
        self._cents = 0.0
        self._update_display()

    def _update_display(self) -> None:
        sign = "+" if self._cents > 0 else ""
        text = f"{sign}{self._cents:.1f} ¢"

        if abs(self._cents) < self._threshold:
            color = QColor(80, 220, 120)  # bright green
        elif abs(self._cents) < 10.0:
            color = QColor(255, 200, 80)  # amber
        else:
            color = QColor(255, 90, 90)   # red

        self.setText(text)
        self.setStyleSheet(f"color: {color.name()};")
