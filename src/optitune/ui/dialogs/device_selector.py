"""
DeviceSelectorDialog — polished, searchable input device selector with loopback self-test.

Lists all input-capable devices (sounddevice), shows hostapi, rate, latency.
Remembers selection via QSettings.
"Test" button: plays short synthetic tone (dsp/synth) on default output while
recording from the chosen input and reports measured cents offset (crude FFT peak).
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QSettings, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from optitune.audio.devices import list_input_devices
from optitune.dsp.synth import generate_inharmonic_tone


class DeviceSelectorDialog(QDialog):
    """Modal dialog for choosing and testing audio input device."""

    device_selected = Signal(int)  # emits the chosen device index when user accepts

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Audio Input Device")
        self.setMinimumSize(620, 420)
        self.resize(680, 480)

        self._settings = QSettings()
        self._devices: list[dict[str, Any]] = []
        self._current_selection: int | None = None

        self._setup_ui()
        self._populate_devices()
        self._load_last_selection()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header / instructions
        header = QLabel(
            "Choose your microphone, interface, or loopback device for piano capture.\n"
            "Use the Test button for a quick loopback check (plays a tone through speakers, listens on the input)."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(header)

        # Search
        search_layout = QHBoxLayout()
        search_label = QLabel("Filter:")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Type to filter devices (name, hostapi...)")
        self.search_edit.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_edit, 1)
        layout.addLayout(search_layout)

        # Device list
        self.device_list = QListWidget()
        self.device_list.setAlternatingRowColors(True)
        self.device_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.device_list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.device_list, 1)

        # Status / test result area
        self.test_result_label = QLabel("Test result: (click Test on a device)")
        self.test_result_label.setStyleSheet(
            "background-color: #1a1a1f; color: #ccc; padding: 6px; border: 1px solid #3a3a40; font-family: monospace;"
        )
        self.test_result_label.setWordWrap(True)
        self.test_result_label.setMinimumHeight(48)
        layout.addWidget(self.test_result_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._populate_devices)
        self.test_btn = QPushButton("Test (play tone + measure)")
        self.test_btn.clicked.connect(self._on_test_clicked)
        self.test_btn.setEnabled(False)

        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.test_btn)

        # Dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        btn_layout.addWidget(self.button_box)

        layout.addLayout(btn_layout)

        # Footer hint
        hint = QLabel(
            "Tip: For best results use a device with low latency. PipeWire/Pulse devices usually work great."
        )
        hint.setStyleSheet("color:#666; font-size:10px;")
        layout.addWidget(hint)

    def _populate_devices(self) -> None:
        self.device_list.clear()
        try:
            self._devices = list_input_devices()
        except Exception as exc:
            QMessageBox.warning(self, "Audio Error", f"Could not query audio devices:\n{exc}")
            self._devices = []
            return

        if not self._devices:
            item = QListWidgetItem(
                "No input devices found. Check your sound system (PipeWire/Pulse/ALSA)."
            )
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.device_list.addItem(item)
            return

        for dev in self._devices:
            text = (
                f"{dev['index']:2d}: {dev['name']}\n"
                f"     [{dev['hostapi']}]  {dev['default_samplerate']:.0f} Hz   "
                f"lat≈{dev['input_latency'] * 1000:.1f} ms   "
                f"{'(default)' if dev['is_default'] else ''}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, dev["index"])
            # Nice default selection styling
            if dev["is_default"]:
                item.setBackground(Qt.GlobalColor.darkGreen)  # subtle, overridden by palette mostly
            self.device_list.addItem(item)

        # Re-apply filter if any
        self._on_search_changed(self.search_edit.text())

    def _on_search_changed(self, text: str) -> None:
        text = text.lower().strip()
        for i in range(self.device_list.count()):
            item = self.device_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsSelectable:
                dev_text = item.text().lower()
                item.setHidden(bool(text) and text not in dev_text)

    def _on_selection_changed(self) -> None:
        items = self.device_list.selectedItems()
        if items:
            self._current_selection = items[0].data(Qt.ItemDataRole.UserRole)
            self.test_btn.setEnabled(True)
        else:
            self._current_selection = None
            self.test_btn.setEnabled(False)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        if item.flags() & Qt.ItemFlag.ItemIsSelectable:
            self._current_selection = item.data(Qt.ItemDataRole.UserRole)
            self._on_accept()

    def _load_last_selection(self) -> None:
        last_idx = self._settings.value("audio/last_input_device_index", None, type=int)
        if last_idx is None:
            # auto-select default if present
            for i in range(self.device_list.count()):
                item = self.device_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) is not None:
                    # find the default one
                    for dev in self._devices:
                        if dev["is_default"]:
                            self._select_index(dev["index"])
                            return
                    # else first
                    self.device_list.setCurrentRow(0)
                    return
            return

        if isinstance(last_idx, int):
            self._select_index(last_idx)

    def _select_index(self, idx: int) -> None:
        for i in range(self.device_list.count()):
            item = self.device_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == idx:
                self.device_list.setCurrentItem(item)
                self.device_list.scrollToItem(item)
                return

    @Slot()
    def _on_test_clicked(self) -> None:
        if self._current_selection is None:
            return

        dev = next((d for d in self._devices if d["index"] == self._current_selection), None)
        if not dev:
            return

        self.test_btn.setEnabled(False)
        self.test_result_label.setText(
            "Testing... playing tone on output while recording from input (≈2s)"
        )

        # Run test off the GUI thread so the dialog stays responsive
        def _test_worker() -> None:
            try:
                fs = int(dev["default_samplerate"])
                # Short, clean tone (middle C-ish, low inharmonicity for clear pitch)
                tone = generate_inharmonic_tone(
                    midi_or_f0=60,
                    detune_cents=0.0,
                    B=0.00005,
                    duration=1.4,
                    fs=fs,
                    snr_db=None,
                    with_hammer=False,
                    seed=7,
                    peak_amp=0.75,
                ).astype(np.float32)

                # Play on default output, record on chosen input (simultaneously)
                out_dev = sd.default.device[1] if sd.default.device is not None else None
                sd.play(tone, samplerate=fs, device=out_dev, blocking=False)

                rec_frames = int(1.6 * fs)
                rec = sd.rec(
                    rec_frames,
                    samplerate=fs,
                    device=dev["index"],
                    channels=1,
                    dtype="float32",
                    blocking=False,
                )
                sd.wait()  # wait for both

                # Measure dominant frequency -> cents from nominal C4 (261.63 Hz)
                nominal_hz = 261.63
                cents = self._estimate_cents(rec[:, 0] if rec.ndim > 1 else rec, fs, nominal_hz)

                peak = float(np.max(np.abs(rec))) if len(rec) > 0 else 0.0
                rms = float(np.sqrt(np.mean(rec**2))) if len(rec) > 0 else 0.0

                if cents is None or peak < 0.001:
                    result = f"Test on {dev['name']}: No significant signal detected (peak={peak:.4f}). Check wiring/mic levels."
                else:
                    sign = "+" if cents > 0 else ""
                    quality = (
                        "excellent"
                        if abs(cents) < 3
                        else ("good" if abs(cents) < 12 else "weak/offset")
                    )
                    result = (
                        f"Test on {dev['name']}: {sign}{cents:.1f} ¢ from C4  |  "
                        f"peak={peak:.3f} rms={rms:.4f}  ({quality} capture)"
                    )

                # Update UI from main thread
                self._update_test_result(result)
            except Exception as exc:
                self._update_test_result(f"Test failed: {exc}")
            finally:
                # Re-enable from main thread
                self._enable_test_btn()

        threading.Thread(target=_test_worker, daemon=True).start()

    def _estimate_cents(self, audio: np.ndarray, fs: int, nominal_hz: float) -> float | None:
        """Very crude but effective dominant-frequency estimator for the self-test."""
        if len(audio) < 256 or np.max(np.abs(audio)) < 1e-4:
            return None
        # Windowed FFT
        win = np.hanning(len(audio))
        spec = np.abs(np.fft.rfft(audio * win))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / fs)

        # Focus on musically relevant band for C4 test tone + harmonics
        mask = (freqs > 50.0) & (freqs < 2200.0)
        if not np.any(mask):
            return None
        idxs = np.where(mask)[0]
        sub_spec = spec[idxs]
        if np.max(sub_spec) < 1e-8:
            return None

        peak_local = int(np.argmax(sub_spec))
        f_peak = float(freqs[idxs[peak_local]])

        if f_peak <= 20.0:
            return None

        cents = 1200.0 * np.log2(f_peak / nominal_hz)
        # Clamp to sane range for display
        return float(np.clip(cents, -120.0, 120.0))

    def _update_test_result(self, text: str) -> None:
        # Called from worker thread — use queued connection
        self.test_result_label.setText(text)

    def _enable_test_btn(self) -> None:
        self.test_btn.setEnabled(self._current_selection is not None)

    @Slot()
    def _on_accept(self) -> None:
        if self._current_selection is None:
            QMessageBox.information(
                self, "No Selection", "Please select an input device from the list."
            )
            return

        # Persist
        self._settings.setValue("audio/last_input_device_index", int(self._current_selection))
        self._settings.sync()

        self.device_selected.emit(int(self._current_selection))
        self.accept()

    def get_selected_device(self) -> int | None:
        """Return the selected device index after dialog has been accepted."""
        return self._current_selection
