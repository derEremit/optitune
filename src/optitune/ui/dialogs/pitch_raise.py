"""Pitch-raise wizard: overpull targets from measured flatness + final curve."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from optitune.solvers.base import N_KEYS
from optitune.solvers.pitch_raise import pitch_raise_targets


class PitchRaiseDialog(QDialog):
    """
    Given a final stretch curve and measured cents-vs-ET (flat piano),
    preview/apply temporary overpull targets.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        final_curve: list[float] | np.ndarray | None = None,
        measured_dev: list[float] | np.ndarray | None = None,
        mean_flat_cents: float | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pitch Raise / Overpull")
        self.setMinimumWidth(420)

        self._final = np.asarray(
            final_curve if final_curve is not None else np.zeros(N_KEYS), dtype=float
        )
        if self._final.shape[0] != N_KEYS:
            pad = np.zeros(N_KEYS)
            n = min(N_KEYS, self._final.shape[0])
            pad[:n] = self._final[:n]
            self._final = pad

        if measured_dev is not None:
            self._measured = np.asarray(measured_dev, dtype=float)
        elif mean_flat_cents is not None:
            self._measured = np.full(N_KEYS, float(mean_flat_cents))
        else:
            self._measured = np.full(N_KEYS, -30.0)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Temporary overpull targets sit above the final curve while the\n"
                "piano is flat, tapering toward the treble (Rigaud-style)."
            )
        )

        form = QFormLayout()
        self._variant = QComboBox()
        self._variant.addItem("Medium", "medium")
        self._variant.addItem("High (aggressive)", "high")
        self._variant.addItem("Low (conservative)", "low")
        self._variant.currentIndexChanged.connect(self._update_preview)
        form.addRow("Variant:", self._variant)
        layout.addLayout(form)

        self._preview = QLabel()
        self._preview.setWordWrap(True)
        self._preview.setStyleSheet("color:#a0a0a8; padding:6px;")
        layout.addWidget(self._preview)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply overpull")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._targets: np.ndarray = np.zeros(N_KEYS)
        self._update_preview()

    def _update_preview(self) -> None:
        variant = str(self._variant.currentData() or "medium")
        self._targets = pitch_raise_targets(self._measured, self._final, variant=variant)
        pull = self._targets - self._final
        bass = float(np.mean(pull[:24]))
        mid = float(np.mean(pull[32:56]))
        treble = float(np.mean(pull[64:]))
        self._preview.setText(
            f"Mean overpull — bass: {bass:+.1f}¢  mid: {mid:+.1f}¢  treble: {treble:+.1f}¢\n"
            f"A4 pinned at 0. Apply replaces the live tuning curve with these temporary targets."
        )

    def targets(self) -> list[float]:
        return [float(x) for x in self._targets]

    def variant(self) -> str:
        return str(self._variant.currentData() or "medium")
