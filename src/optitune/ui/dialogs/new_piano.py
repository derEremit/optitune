"""New Piano dialog: name, A4, temperament."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from optitune.model.temperaments import TEMPERAMENT_LABELS, list_temperaments


class NewPianoDialog(QDialog):
    """Collect piano session metadata before starting a fresh recording pass."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        name: str = "My Piano",
        a4: float = 440.0,
        temperament: str = "equal",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Piano")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._name = QLineEdit(name)
        form.addRow("Name:", self._name)

        self._a4 = QDoubleSpinBox()
        self._a4.setRange(415.0, 466.0)
        self._a4.setDecimals(1)
        self._a4.setSingleStep(0.1)
        self._a4.setValue(float(a4))
        self._a4.setSuffix(" Hz")
        form.addRow("A4:", self._a4)

        self._temp = QComboBox()
        for key in list_temperaments():
            label = TEMPERAMENT_LABELS.get(key, key)
            self._temp.addItem(label, key)
        # select temperament
        idx = self._temp.findData(temperament)
        if idx >= 0:
            self._temp.setCurrentIndex(idx)
        form.addRow("Temperament:", self._temp)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def piano_name(self) -> str:
        t = self._name.text().strip()
        return t or "My Piano"

    def a4(self) -> float:
        return float(self._a4.value())

    def temperament(self) -> str:
        data = self._temp.currentData()
        return str(data) if data else "equal"
