"""Dialog: edit beat-rate interval weights with named presets."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from optitune.solvers.interval_weights import (
    DEFAULT_INTERVAL_WEIGHTS,
    PRESET_LABELS,
    WEIGHT_LABELS,
    get_preset,
    list_presets,
)


class IntervalWeightsDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Interval Weights")
        self.setMinimumWidth(400)

        self._weights = deepcopy(weights or DEFAULT_INTERVAL_WEIGHTS)
        self._spins: dict[str, QDoubleSpinBox] = {}

        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Preset:"))
        self._preset = QComboBox()
        for key in list_presets():
            self._preset.addItem(PRESET_LABELS.get(key, key), key)
        self._preset.currentIndexChanged.connect(self._on_preset)
        row.addWidget(self._preset, 1)
        apply_btn = QPushButton("Apply preset")
        apply_btn.clicked.connect(self._apply_preset)
        row.addWidget(apply_btn)
        layout.addLayout(row)

        form = QFormLayout()
        for key in DEFAULT_INTERVAL_WEIGHTS:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setValue(float(self._weights.get(key, DEFAULT_INTERVAL_WEIGHTS[key])))
            self._spins[key] = spin
            form.addRow(WEIGHT_LABELS.get(key, key) + ":", spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_preset(self, _idx: int) -> None:
        pass  # user clicks Apply

    def _apply_preset(self) -> None:
        key = self._preset.currentData()
        if not key:
            return
        for k, v in get_preset(str(key)).items():
            if k in self._spins:
                self._spins[k].setValue(float(v))

    def weights(self) -> dict[str, float]:
        return {k: float(spin.value()) for k, spin in self._spins.items()}
