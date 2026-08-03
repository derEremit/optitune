"""
Professional dark-themed main window for OptiTune.

Phase 3: Real DSP live analysis integration.
Phase 4: Recording workflow + basic tuning-curve solver (model + simple stretch).
- Captures measured f0/B per key via "Record Note / Record Next" (guided or free).
- Computes minimal Railsback-style stretch curve from measured B values (B-curve fit + heuristic + Shah-Välimäki treble rule).
- Live tuner now uses curve targets (when present) instead of pure ET: cents/strobe show deviation from the piano-specific stretch.
- Simple JSON persistence for the current Piano session.
- Keyboard paints MEASURED keys in blue; solver results immediately affect live targeting.
"""

from __future__ import annotations

import contextlib
import logging
import os

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer, Slot
from PySide6.QtGui import QAction, QCloseEvent, QColor, QPalette
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from optitune.audio import (
    AudioCapture,
    RingBuffer,
    get_device_display_name,
    list_input_devices,
    resolve_device_index,
)
from optitune.dsp import (
    F0Tracker,
    estimate_pitch,
    hz_to_midi,
    midi_to_hz,
    midi_to_note_name,
)
from optitune.model import Key, Piano
from optitune.recording.auto_record import (
    AutoRecordConfig,
    AutoRecordController,
    AutoRecordEvent,
)
from optitune.solvers import compute_basic_tuning_curve
from optitune.ui.dialogs import DeviceSelectorDialog
from optitune.ui.widgets import (
    CentsDisplay,
    KeyboardWidget,
    SpectrumWidget,
    StrobeWidget,
)
from optitune.ui.widgets.keyboard_widget import KeyState

logger = logging.getLogger(__name__)

# Controllable verbosity for heavy per-tick diagnostics (priority 6)
# The per-tick [DIAG][ScaleGate] spam (when armed in scale mode) and controller Onset spam
# are off by default to keep normal runs quiet. Set OPTITUNE_DIAG=1 (or verbose/full)
# to enable everything. The full-master diagnostic test forces it on.
# Maps onto the module logger: DIAG on → DEBUG; DIAG off → INFO for key events only.
_DIAG_VERBOSE = os.environ.get("OPTITUNE_DIAG", "0").lower() in (
    "1",
    "true",
    "yes",
    "verbose",
    "full",
    "on",
)


def _diag(msg: str, *args: object, verbose_only: bool = False) -> None:
    """Log a [DIAG] line. Key decisions use INFO; per-tick noise is DEBUG when DIAG is on."""
    if verbose_only and not _DIAG_VERBOSE:
        return
    level = logging.DEBUG if verbose_only else logging.INFO
    logger.log(level, msg, *args)


if _DIAG_VERBOSE and not logger.handlers:
    # OPTITUNE_DIAG maps to visible console diagnostics for interactive / real_piano runs
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


class OptiTuneMainWindow(QMainWindow):
    """Live tuner main window with real audio input, level meter, accurate DSP analysis, and Phase-4 recording + curve."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        a4: float = 440.0,
        device: str | None = None,  # CLI override (takes precedence for this launch)
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("OptiTune")
        self.resize(1100, 720)
        self.setMinimumSize(800, 600)

        self._initial_a4 = float(a4)
        self._cli_device = device  # may be name/index from CLI

        # Audio core (Phase 2)
        self.ringbuffer = RingBuffer(max_samples=192_000)  # ~4 s @ 48 kHz
        self.audio_capture = AudioCapture(self.ringbuffer, samplerate=48000, blocksize=1024)
        self._current_device_index: int | None = None
        self._settings = QSettings()

        # Analysis state (Phase 3)
        self._analysis_timer: QTimer | None = None
        self._level_timer: QTimer | None = None
        self._analysis_tick = 0
        self._last_f0_guess: float = 440.0  # for PFD anchoring across frames
        self._f0_tracker = F0Tracker(window=7)

        # Phase 4: model + recording + curve
        self._last_est: dict | None = None
        self._piano: Piano | None = None
        self._record_selected_midi: int | None = None

        # Auto-recording state machine (extracted + TDD'd for reliability)
        self._auto_record_ctrl = AutoRecordController(
            AutoRecordConfig(
                onset_db_threshold=-28.0,
                min_onset_confirmation_ms=280,
                capture_duration_ms=1800,
            )
        )
        self._auto_advance_after_record: bool = True  # very useful for walking the piano
        self._ignore_onset_until: float = 0.0
        self._require_strong_attack_until: float = 0.0
        self._scale_pitch_class: int | None = None  # for "scale recording" mode
        self._scale_gate_grace_until: float = (
            0.0  # short window after (re)arm where we trust armed intent over noisy early est
        )
        self._during_capture_rejection_until: float = (
            0.0  # for during-capture validation + subtle rejection feedback (step 5)
        )
        self._prev_level_db: float = -60.0
        self._last_during_cap_check: float = 0.0
        self._curve_status_label: QLabel | None = None

        self._setup_theme()
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_status_bar()

        # Try to restore previous piano session (measurements + curve)
        self._load_persisted_piano()

        # Wire audio + start capture if we have a device (CLI wins, else last saved)
        self._initialize_audio()

        # Start live meters / real analysis
        self._start_live_updates()

    # ---------------- Setup ----------------

    def _setup_theme(self) -> None:
        """Apply a clean, high-contrast dark professional theme via palette + QSS."""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 35))  # type: ignore[attr-defined]
        palette.setColor(QPalette.WindowText, QColor(240, 240, 245))  # type: ignore[attr-defined]
        palette.setColor(QPalette.Base, QColor(22, 22, 26))  # type: ignore[attr-defined]
        palette.setColor(QPalette.AlternateBase, QColor(35, 35, 40))  # type: ignore[attr-defined]
        palette.setColor(QPalette.ToolTipBase, QColor(50, 50, 55))  # type: ignore[attr-defined]
        palette.setColor(QPalette.ToolTipText, QColor(240, 240, 245))  # type: ignore[attr-defined]
        palette.setColor(QPalette.Text, QColor(235, 235, 240))  # type: ignore[attr-defined]
        palette.setColor(QPalette.Button, QColor(45, 45, 50))  # type: ignore[attr-defined]
        palette.setColor(QPalette.ButtonText, QColor(240, 240, 245))  # type: ignore[attr-defined]
        palette.setColor(QPalette.BrightText, QColor(255, 80, 80))  # type: ignore[attr-defined]
        palette.setColor(QPalette.Link, QColor(100, 170, 255))  # type: ignore[attr-defined]
        palette.setColor(QPalette.Highlight, QColor(70, 130, 200))  # type: ignore[attr-defined]
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))  # type: ignore[attr-defined]
        self.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e23; }
            QMenuBar {
                background-color: #25252a; color: #f0f0f5;
                border-bottom: 1px solid #3a3a40; padding: 2px;
            }
            QMenuBar::item { background-color: transparent; padding: 6px 12px; }
            QMenuBar::item:selected { background-color: #3a3a40; border-radius: 4px; }
            QMenu { background-color: #25252a; color: #f0f0f5; border: 1px solid #3a3a40; }
            QMenu::item:selected { background-color: #3a3a40; }
            QStatusBar {
                background-color: #25252a; color: #a0a0a5;
                border-top: 1px solid #3a3a40; padding: 2px 8px;
            }
            QLabel#centralLabel { color: #a0a0a5; font-size: 16px; padding: 40px; }
            QProgressBar#levelBar {
                border: 1px solid #3a3a40; border-radius: 2px;
                background: #1a1a1f; text-align: center; font-size: 9px;
            }
            QProgressBar#levelBar::chunk { background-color: #5fa8ff; }
        """)

    def _setup_ui(self) -> None:
        """Main tuner layout with live widgets driven by real DSP analysis."""
        central = QWidget()
        central.setObjectName("centralWidget")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Top row: cents + strobe
        top_row = QSplitter(Qt.Orientation.Horizontal)
        self.cents_display = CentsDisplay()
        self.strobe = StrobeWidget()
        top_row.addWidget(self.cents_display)
        top_row.addWidget(self.strobe)
        top_row.setSizes([380, 320])
        layout.addWidget(top_row, 3)

        # Spectrum (now real pyqtgraph)
        self.spectrum = SpectrumWidget()
        layout.addWidget(self.spectrum, 2)

        # Keyboard (now highlights detected note + supports measured states for recording)
        self.keyboard = KeyboardWidget()
        layout.addWidget(self.keyboard, 1)

        # Subtle live hint (updated in Phase 4)
        self.hint = QLabel(
            "Live DSP active — play notes on your piano. Use toolbar to record keys & compute a stretch curve.  "
            "Select device via Audio → Input Device..."
        )
        self.hint.setStyleSheet("color:#555; font-size:11px; padding:4px;")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.hint)

        self.setCentralWidget(central)

        # Connect widget signals
        self.strobe.strobe_clicked.connect(self._on_strobe_clicked)
        self.keyboard.key_clicked.connect(self._on_keyboard_clicked)

    def _setup_menus(self) -> None:
        menubar: QMenuBar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")
        new_action = QAction("New Piano...", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._on_new_piano)
        file_menu.addAction(new_action)

        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Audio — now fully wired
        audio_menu = menubar.addMenu("&Audio")
        device_action = QAction("Input Device...", self)
        device_action.setShortcut("Ctrl+D")
        device_action.triggered.connect(self._on_input_device)
        audio_menu.addAction(device_action)

        test_signal_action = QAction("Test Signal", self)
        test_signal_action.triggered.connect(self._on_test_signal)
        audio_menu.addAction(test_signal_action)

        # Phase 4: Tuning menu for recording / solver
        tune_menu = menubar.addMenu("&Tuning")
        rec_action = QAction("Start Recording Session", self)
        rec_action.triggered.connect(self._on_start_recording)
        tune_menu.addAction(rec_action)

        compute_action = QAction("Compute Basic Curve", self)
        compute_action.triggered.connect(self._on_compute_curve)
        tune_menu.addAction(compute_action)

        clear_action = QAction("Clear Measurements", self)
        clear_action.triggered.connect(self._on_clear_measurements)
        tune_menu.addAction(clear_action)

        # Help
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About OptiTune", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        play_action = QAction("▶ Play Test Tone", self)
        play_action.setShortcut("Ctrl+T")
        play_action.setToolTip(
            "Play synthetic inharmonic tone and watch live cents/strobe (Phase 3+)"
        )
        play_action.triggered.connect(self._on_play_test_tone)
        tb.addAction(play_action)

        tb.addSeparator()

        # Phase 4 recording controls (user's final-test capability)
        self._record_action = QAction("⏺ Record Note", self)
        self._record_action.setShortcut("Ctrl+R")
        self._record_action.setToolTip(
            "Record the current live analysis (f0 + B) for the selected or detected key"
        )
        self._record_action.triggered.connect(self._on_record_note)
        tb.addAction(self._record_action)

        self._record_next_action = QAction("➡️ Record Next", self)
        self._record_next_action.setToolTip(
            "Auto-advance to next unmeasured key and prepare for recording"
        )
        self._record_next_action.triggered.connect(self._on_record_next)
        tb.addAction(self._record_next_action)

        tb.addSeparator()

        # New hands-free auto-record (user request)
        self._arm_record_action = QAction("🎙️ Arm Auto-Record", self)
        self._arm_record_action.setShortcut("Ctrl+Shift+R")
        self._arm_record_action.setCheckable(True)
        self._arm_record_action.setToolTip(
            "HANDS-FREE MODE: Click this, then go to the piano.\n"
            "When you play the target note loudly, it will automatically detect the onset, record 1.8s, analyze, and store it.\n"
            "If Auto-advance is ON it will move to the next key and re-arm."
        )
        self._arm_record_action.triggered.connect(self._toggle_auto_record_arm)
        tb.addAction(self._arm_record_action)

        # Quick toggle for the user's most common workflow (walk to piano, play, it auto-captures and advances)
        self._auto_advance_action = QAction("➡️ Auto-advance", self)
        self._auto_advance_action.setCheckable(True)
        self._auto_advance_action.setChecked(True)
        self._auto_advance_action.setToolTip(
            "When ON: after auto-capture, jump to next unmeasured key and re-arm automatically."
        )
        self._auto_advance_action.toggled.connect(self._update_auto_advance_ui)
        self._update_auto_advance_ui(True)
        tb.addAction(self._auto_advance_action)

        tb.addSeparator()

        self._compute_action = QAction("📈 Compute Curve", self)
        self._compute_action.setShortcut("Ctrl+K")
        self._compute_action.setToolTip(
            "Run minimal beat-rate stretch solver on recorded B values → live targets update immediately"
        )
        self._compute_action.triggered.connect(self._on_compute_curve)
        tb.addAction(self._compute_action)

        tb.addSeparator()

        reset_action = QAction("Reset Displays", self)
        reset_action.triggered.connect(self._on_reset_displays)
        tb.addAction(reset_action)

    def _setup_status_bar(self) -> None:
        """Professional status bar with live device + level meter + A4 + curve indicator (Phase 4)."""
        status = QStatusBar()
        self.setStatusBar(status)

        self._device_label = QLabel("Input: (none)  ")
        self._device_label.setMinimumWidth(280)

        # Live level meter as a compact progress bar (0-100 representing -60dB..0dB)
        self._level_bar = QProgressBar()
        self._level_bar.setObjectName("levelBar")
        self._level_bar.setMaximum(100)
        self._level_bar.setTextVisible(True)
        self._level_bar.setFormat("Level: %p%")
        self._level_bar.setMaximumWidth(140)
        self._level_bar.setMinimumWidth(90)

        self._a4_label = QLabel(f"  A4 = {self._initial_a4:.1f} Hz")

        self._curve_status_label = QLabel("  Curve: ET only")
        self._curve_status_label.setMinimumWidth(160)

        status.addPermanentWidget(self._device_label)
        status.addPermanentWidget(self._level_bar)
        status.addPermanentWidget(self._a4_label)
        status.addPermanentWidget(self._curve_status_label)

        # Initial message (will be overwritten by audio init)
        status.showMessage("Initializing audio...", 1500)

    def _update_curve_status(self) -> None:
        """Refresh the curve indicator in the status bar."""
        if self._curve_status_label is None:
            return
        if self._piano is None or self._piano.tuning_curve is None:
            self._curve_status_label.setText("  Curve: ET only")
            self._curve_status_label.setStyleSheet("color: #888;")
        else:
            n = self._piano.measured_count()
            self._curve_status_label.setText(f"  Curve: active ({n} measured)")
            self._curve_status_label.setStyleSheet("color: #5fa8ff; font-weight: 500;")

    # ---------------- Audio initialization & device handling ----------------

    def _initialize_audio(self) -> None:
        """Choose device (CLI > saved > first available) and start capture."""
        dev_to_use: int | None = None

        # 1. CLI override (string or we treat as name/index hint)
        if self._cli_device is not None:
            try:
                if str(self._cli_device).strip().isdigit():
                    dev_to_use = int(self._cli_device)
                else:
                    dev_to_use = resolve_device_index(self._cli_device)
            except Exception:
                dev_to_use = None

        # 2. Saved preference
        if dev_to_use is None:
            saved = self._settings.value("audio/last_input_device_index", None, type=int)
            if isinstance(saved, int):
                # Validate it still exists as input
                try:
                    inputs = list_input_devices()
                    if any(d["index"] == saved for d in inputs):
                        dev_to_use = saved
                except Exception:
                    pass

        # 3. First available default
        if dev_to_use is None:
            try:
                inputs = list_input_devices()
                for d in inputs:
                    if d["is_default"]:
                        dev_to_use = d["index"]
                        break
                if dev_to_use is None and inputs:
                    dev_to_use = inputs[0]["index"]
            except Exception:
                pass

        if dev_to_use is not None:
            self._apply_audio_device(dev_to_use, startup=True)
        else:
            self._device_label.setText("Input: (no input devices found)  ")
            self.statusBar().showMessage(
                "No audio input devices detected. Use Audio → Input Device... to retry.", 5000
            )

    def _apply_audio_device(self, device_index: int, startup: bool = False) -> None:
        """Stop any running stream, start new one, persist choice, update UI."""
        name = get_device_display_name(device_index)

        try:
            if self.audio_capture.is_running:
                self.audio_capture.stop()

            self.audio_capture.start(device=device_index)
            self._current_device_index = device_index

            # Persist (dialog also does this, but ensure)
            self._settings.setValue("audio/last_input_device_index", int(device_index))
            self._settings.sync()

            self._device_label.setText(f"Input: {name}  ")
            self.statusBar().showMessage(f"Capture started on {name}", 2500)

            # Give the strobe something to do
            self.strobe.set_running(True)
            self.strobe.set_target_frequency(440.0)

            if not startup:
                # User just chose it via dialog
                QMessageBox.information(
                    self,
                    "Audio Input",
                    f"Now capturing from:\n{name}\n\n"
                    "Play notes on your piano — the cents, strobe, spectrum and keyboard will track using real DSP (peaks + PFD).",
                )
        except Exception as exc:
            self._current_device_index = None
            self._device_label.setText(f"Input: ERROR {name}  ")
            QMessageBox.warning(
                self,
                "Audio Capture Failed",
                f"Could not start capture on device {device_index} ({name}):\n\n{exc}\n\n"
                "Common causes: device in exclusive use, sample rate mismatch, or permissions.\n"
                "Try another device via Audio → Input Device...",
            )
            self.statusBar().showMessage(f"Capture error on {name}", 4000)

    # ---------------- Live updates (level + real DSP analysis) ----------------

    def _start_live_updates(self) -> None:
        """Start the two Qt timers. Level fast/cheap; analysis ~8-12 Hz with real Phase-1 DSP."""
        # Level meter — fast, cheap (every 50 ms)
        self._level_timer = QTimer(self)
        self._level_timer.timeout.connect(self._update_level_meter)
        self._level_timer.start(50)

        # Real analysis — 100 ms gives good overlap on 32k windows while staying light
        self._analysis_timer = QTimer(self)
        self._analysis_timer.timeout.connect(self._run_live_analysis)
        self._analysis_timer.start(100)

    def _update_level_meter(self) -> None:
        """Poll ringbuffer for short window and update the level bar (0-100)."""
        try:
            buf = self.ringbuffer.get_latest(1024)
            if len(buf) == 0:
                self._level_bar.setValue(0)
                return
            rms = float(np.sqrt(np.mean(buf * buf)))
            # Map -60 dB ... 0 dB  →  0 ... 100
            db = -60.0 if rms <= 1e-7 else 20.0 * np.log10(rms)
            level = int(np.clip((db + 60.0) * (100.0 / 60.0), 0, 100))
            self._level_bar.setValue(level)

            # New TDD'd auto-record controller (replaces the old ad-hoc flags + logic)
            import time as _time

            now = _time.time()

            # Compute dB rise for attack detection
            prev_db = getattr(self, "_prev_level_db", db - 2.0)
            db_rise = db - prev_db
            self._prev_level_db = db

            # Always compute this here so it is in scope for the diagnostic prints below
            # (it used to be defined only inside the !ignore branch, causing NameError / UnboundLocal
            # on every tick while post-capture suppression was active — exactly the "hanging" symptom
            # during scale-mode series with _ignore_onset_until).
            require_strong = getattr(self, "_require_strong_attack_until", 0) > now

            # Post-capture protection
            if getattr(self, "_ignore_onset_until", 0) > now:
                event = None
            else:
                # Extra strict attack requirement after recent capture (especially in scale mode)
                min_rise = 8.0 if require_strong else 4.5

                if require_strong and db_rise < min_rise:
                    event = None
                else:
                    # === Layer 1: Expectation-driven pitch-class gate (scale recording mode) ===
                    # If we are in a known root-note series (_scale_pitch_class set), only feed the
                    # controller when the most recent high-quality analysis (_last_est) says the
                    # current sound is within ~140 cents of the expected pitch class.
                    # This prevents the energy detector from latching onto wrong notes (C# etc.)
                    # while the user is playing a clean C-then-F series.
                    scale_pc = getattr(self, "_scale_pitch_class", None)
                    if scale_pc is not None:
                        # Pass whatever we have; the helper will compute the best precise midi_f
                        # from f_est if available, and decide acceptance.
                        est_midi_for_gate = None
                        if self._last_est:
                            m = self._last_est.get("midi")
                            if m is not None:
                                est_midi_for_gate = float(m)

                        in_grace = getattr(self, "_scale_gate_grace_until", 0) > now
                        pc_ok = self._pitch_class_matches_expectation(
                            est_midi_for_gate, scale_pc, tolerance=self.ONSET_GATE_CENT_TOLERANCE
                        )

                        # Additional robustness for real piano low-note material (seen in long diagnostics):
                        # The live estimator often reports harmonics or octaves (e.g. 27 or 48 when armed on C1=24).
                        # If we have a specific armed target in the current scale class, allow the energy
                        # controller to see loud ticks if the est is not insanely far from the armed note.
                        # This is still pure expectation logic (the armed target *is* the expectation).
                        # The strict commit-time gate with fresh analysis remains the final filter.
                        armed = self._record_selected_midi
                        # ~2 octaves tolerance for estimator weaknesses on real low notes
                        close_to_armed = (
                            armed is not None
                            and est_midi_for_gate is not None
                            and abs(est_midi_for_gate - armed) <= 20
                        )

                        if not pc_ok and not in_grace and not close_to_armed:
                            # Hard rejection for this tick — do not let the energy-based controller see it.
                            _diag(
                                "[DIAG][ScaleGate] SUPPRESSED (wrong pitch class) | "
                                "est_midi_approx=%s expected_pc=%s dB=%.1f",
                                est_midi_for_gate,
                                scale_pc,
                                db,
                                verbose_only=True,
                            )
                            event = None
                        else:
                            if (in_grace or close_to_armed) and not pc_ok:
                                reason = "GRACE" if in_grace else "close_to_armed"
                                _diag(
                                    "[DIAG][ScaleGate] %s allowed (armed intent) | "
                                    "est_midi_approx=%s armed=%s dB=%.1f",
                                    reason,
                                    est_midi_for_gate,
                                    armed,
                                    db,
                                    verbose_only=True,
                                )
                            event = self._auto_record_ctrl.on_level_tick(db, now)
                    else:
                        event = self._auto_record_ctrl.on_level_tick(db, now)

            if event == AutoRecordEvent.ONSET_CONFIRMED:
                _diag(
                    f"[DIAG] ONSET CONFIRMED | dB={db:.1f} rise={db_rise:.1f} strong_req={require_strong}"
                )
                self._on_auto_onset_confirmed()

            elif event == AutoRecordEvent.CAPTURE_FINISHED:
                _diag(f"[DIAG] CAPTURE FINISHED | target={self._record_selected_midi}")
                self._finish_auto_capture(commit=True)

            # --- During-capture validation + subtle rejection feedback (priority 5) ---
            # While recording in scale mode we continuously validate the live estimate
            # against the expected pitch class (using the same tolerance as commit time)
            # and against the armed target. This is the "continuous mid-capture" path
            # from the design. On mismatch we:
            #   - Log a clear rejection (for diagnostics)
            #   - Set a short rejection window (can drive subtle visual feedback later,
            #     e.g. brief flash on the armed key or status message)
            # We do *not* abort the capture here (keeps controller pure); the strong
            # commit-time gate in _decide_commit_and_maybe_switch remains the final authority.
            if (
                self._auto_record_ctrl.is_recording
                and self._scale_pitch_class is not None
                and self._record_selected_midi is not None
                and self._last_est is not None
                # Throttle to keep logs reasonable (~3x per second max)
                and (now - self._last_during_cap_check) > 0.32
            ):
                self._last_during_cap_check = now
                est_m = self._last_est.get("midi")
                if est_m is not None:
                    armed = self._record_selected_midi
                    scale_pc = self._scale_pitch_class

                    # Class check (same fuzzy logic as onset gate)
                    class_ok = self._pitch_class_matches_expectation(float(est_m), scale_pc)

                    # Target tolerance check (same 140¢ used at commit)
                    target_err = self._cents_error_to_target(
                        self._last_est.get("f_est", est_m), armed
                    )
                    target_ok = (
                        target_err is None or abs(target_err) <= self.SCALE_MODE_CENT_TOLERANCE
                    )

                    # During capture we are more tolerant of the estimator's known weaknesses
                    # on real low piano notes (octave jumps, strong partials). We still log
                    # when it's clearly wrong so we have data for future improvements, but we
                    # avoid spamming the rejection window on every decay frame.
                    close_to_armed = armed is not None and abs(est_m - armed) <= 15
                    if (not class_ok or not target_ok) and not close_to_armed:
                        octave_err = self._is_probable_octave_or_partial_error(armed, est_m)
                        tag = " (probable octave/partial error)" if octave_err else ""
                        err_str = f"{target_err:.1f}" if target_err is not None else "n/a"
                        _diag(
                            "[DIAG][DuringCapture] REJECT during window%s | armed=%s "
                            "scale_pc=%s live_midi≈%s class_ok=%s target_err=%s¢",
                            tag,
                            armed,
                            scale_pc,
                            est_m,
                            class_ok,
                            err_str,
                        )
                        # Set short rejection window for subtle feedback (UI can poll this)
                        self._during_capture_rejection_until = now + 0.4
                    else:
                        # Good — optionally log stable periods at lower frequency (debug only)
                        if abs(est_m - armed) > 1.5 and (int(now * 10) % 8 == 0):
                            _diag(
                                f"[DIAG][DuringCapture] OK but drifting | armed={armed} live≈{est_m}"
                            )

            # Keep the forced red state fresh at the fast level-meter rate (50 ms)
            self._apply_auto_record_visual_force()
        except Exception:
            self._level_bar.setValue(0)

    def _run_live_analysis(self) -> None:
        """
        Phase 3/4 live analysis.
        ... (same as before) ...
        After computing ET-based est, we re-target using any active piano curve.
        """
        try:
            fs = self.audio_capture.samplerate or 48000
            a4 = self._initial_a4

            # ~0.68 s window — excellent resolution (~1.5 Hz bins), still responsive
            n = 32768
            audio = self.ringbuffer.get_latest(n)
            if len(audio) < 2048:
                return

            rms = float(np.sqrt(np.mean(audio * audio)))
            energy_high = rms > 0.0045  # audible but forgiving threshold for real piano/mic

            if not energy_high:
                # Gentle decay toward zero when idle
                cur = getattr(self.cents_display, "_cents", 0.0)
                if abs(cur) > 0.2:
                    self.cents_display.set_cents(cur * 0.55)
                self.strobe.set_phase_delta_hz(0.0)
                # fade spectrum a little by sending near-zero frame occasionally
                if self._analysis_tick % 4 == 0:
                    self.spectrum.update_frame(np.array([100, 200]), np.array([1e-6, 1e-6]))
                return

            # --- Real DSP pitch estimation (still ET inside) ---
            est = self._estimate_pitch(audio, fs, a4)
            f_est = float(est["f_est"])
            midi = int(est["midi"])
            f0_used = float(est.get("f0", f_est))

            # Temporal tracking: reject one-off octave/partial spikes
            tracked = self._f0_tracker.push(f0_used)
            if (
                tracked is not None
                and tracked > 20
                and abs(1200.0 * np.log2(f_est / tracked)) > 500.0
            ):
                f_est = tracked
                f0_used = tracked
                midi = round(hz_to_midi(f_est, a4))
                armed = self._record_selected_midi
                if armed is not None:
                    armed_hz = midi_to_hz(armed, a4)
                    if abs(1200.0 * np.log2(f_est / armed_hz)) <= self.SCALE_MODE_CENT_TOLERANCE:
                        midi = armed

            # Phase 4: re-target using curve if present (this is what makes the final test work)
            target_hz = self._get_target_hz(midi, a4)
            if f_est > 1 and target_hz > 1:
                cents = 1200.0 * np.log2(f_est / target_hz)
                delta_hz = f_est - target_hz
            else:
                cents = 0.0
                delta_hz = 0.0

            # Cache for recording workflow
            self._last_est = {
                **est,
                "f_est": float(f_est),
                "f0": float(f0_used),
                "midi": int(midi),
                "target_hz": float(target_hz),
                "cents": float(cents),
                "delta_hz": float(delta_hz),
            }

            # Update last guess for next-frame PFD stability
            if 30 < f0_used < 6000:
                self._last_f0_guess = f0_used

            # Drive widgets (clip cents for display sanity)
            clipped_cents = float(np.clip(cents, -55.0, 55.0))
            self.cents_display.set_cents(clipped_cents)

            # Strobe gets the true frequency error in Hz (positive = sharp → clockwise)
            self.strobe.set_phase_delta_hz(float(delta_hz))
            self.strobe.set_target_frequency(target_hz)

            # Real spectrum data from the same FFT we already computed inside estimator
            self._update_spectrum_from_audio(audio, fs, f_est)

            # Keyboard highlight (detected/locked note)
            forced = self._auto_record_ctrl.get_forced_visual_state()
            if forced:
                forced_midi, forced_state = forced
                self.keyboard.set_key_state(forced_midi, forced_state)
                self.keyboard.set_current_key(forced_midi)
            else:
                self.keyboard.set_current_key(midi)
                self.keyboard.highlight_detected(midi)

            # Extra safety: if we are in an auto-record phase, make absolutely sure
            # the target key keeps the correct color even if other code touched the widget.
            self._apply_auto_record_visual_force()

            # Occasional status / note name
            self._analysis_tick += 1
            if self._analysis_tick % 12 == 0:
                note_name = midi_to_note_name(midi)
                curve_note = ""
                if self._piano and self._piano.tuning_curve:
                    off = self._piano.get_target_offset(midi)
                    if abs(off) > 0.05:
                        curve_note = f"  [curve {off:+.1f}¢]"
                self.statusBar().showMessage(
                    f"Tracking {note_name} ({midi})  {cents:+.1f} ¢   Δ{delta_hz:.2f} Hz{curve_note}",
                    1800,
                )

        except Exception:
            # Analysis must never bring down the GUI or audio pipeline
            pass

    def _get_target_hz(self, midi: int, a4: float) -> float:
        """Phase 4: return ET frequency or ET + curve offset (the key change for solver integration)."""
        base = midi_to_hz(midi, a4)
        if self._piano is not None:
            off = self._piano.get_target_offset(midi)
            if abs(off) > 0.01:
                return float(base * (2.0 ** (off / 1200.0)))
        return base

    def _estimate_pitch(self, audio: np.ndarray, fs: float, a4: float) -> dict:
        """
        Core live estimator — thin adapter over pure dsp.estimate_pitch.

        Armed target and last-frame f0 are soft priors only.
        """
        n = len(audio)
        if n < 256:
            return self._fallback_estimate(audio, fs, a4)
        return estimate_pitch(
            audio,
            fs,
            a4=a4,
            armed_midi=self._record_selected_midi,
            last_f0_guess=self._last_f0_guess,
            scale_cent_tol=self.SCALE_MODE_CENT_TOLERANCE,
        )

    def _fallback_estimate(self, audio: np.ndarray, fs: float, a4: float) -> dict:
        """Very cheap fallback (used only on tiny buffers)."""
        if len(audio) < 256:
            return {
                "f_est": 440.0,
                "midi": 69,
                "target_hz": a4,
                "cents": 0.0,
                "delta_hz": 0.0,
                "f0": 440.0,
                "b": 0.0003,
            }
        win = np.hanning(len(audio))
        spec = np.abs(np.fft.rfft(audio * win))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / fs)
        mask = (freqs > 50) & (freqs < 4000)
        if not np.any(mask):
            return {
                "f_est": 440.0,
                "midi": 69,
                "target_hz": a4,
                "cents": 0.0,
                "delta_hz": 0.0,
                "f0": 440.0,
                "b": 0.0003,
            }
        idxs = np.where(mask)[0]
        peak = int(np.argmax(spec[idxs]))
        f = float(freqs[idxs[peak]])
        midi_f = hz_to_midi(f, a4)
        midi = round(midi_f)
        target = midi_to_hz(midi, a4)
        cents = 1200.0 * np.log2(max(f, 1) / max(target, 1)) if target > 1 else 0.0
        delta = f - target
        return {
            "f_est": f,
            "midi": midi,
            "target_hz": target,
            "cents": cents,
            "delta_hz": delta,
            "f0": f,
            "b": 0.0003,
        }

    def _update_spectrum_from_audio(self, audio: np.ndarray, fs: float, detected_f: float) -> None:
        """Feed the (now real) SpectrumWidget a usable view + marker."""
        try:
            n = min(4096, len(audio))
            if n < 128:
                return
            w = np.hanning(n)
            spec = np.abs(np.fft.rfft(audio[-n:] * w))
            freqs = np.fft.rfftfreq(n, 1.0 / fs)
            power = spec**2
            step = max(1, len(freqs) // 600)
            self.spectrum.update_frame(freqs[::step], power[::step])
            self.spectrum.set_detected_pitch(float(detected_f))
        except Exception:
            pass

    # ---------------- Phase 4: Recording workflow + solver integration ----------------

    def _ensure_piano(self) -> Piano:
        if self._piano is None:
            self._piano = Piano(a4=self._initial_a4, name="My Piano")
        return self._piano

    def _on_keyboard_clicked(self, midi: int) -> None:
        """Clicking a key now also selects it as the target for the next Record operation."""
        self.keyboard.set_current_key(midi)
        self._record_selected_midi = midi
        self.statusBar().showMessage(
            f"Key {midi} ({midi_to_note_name(midi)}) selected for recording. Play the note then click 'Record Note'.",
            3000,
        )

    def _on_record_note(self, visual_feedback: bool = True) -> None:
        """
        Capture current live DSP result and store it.

        When called from the auto-record path we often pass visual_feedback=False
        because the AutoRecordController + _finish_auto_capture now own the correct
        ARMED / RECORDING / MEASURED transitions and we don't want the old
        "set RECORDING then immediately MEASURED" synchronous flash.
        """
        if self._last_est is None:
            QMessageBox.information(
                self, "Record", "No live analysis yet. Play a note on the piano first."
            )
            return

        piano = self._ensure_piano()

        midi = self._record_selected_midi or self._last_est.get("midi", 69)
        midi = max(21, min(108, int(midi)))

        if visual_feedback:
            self.keyboard.set_key_state(midi, KeyState.RECORDING)
            self._record_action.setText("⏺ Recording...")

        f0 = float(self._last_est.get("f0", self._last_est.get("f_est", 440.0)))
        b = float(self._last_est.get("b", 0.0003))

        k = Key(midi=midi, measured_f0=f0, measured_b=b)
        piano.set_key(k)

        if visual_feedback:
            self.keyboard.set_key_state(midi, KeyState.MEASURED)
            self.keyboard.set_current_key(midi)
            self._record_action.setText("⏺ Record Note")

        self._record_selected_midi = midi
        self._update_curve_status()
        self.statusBar().showMessage(
            f"Recorded MIDI {midi} ({midi_to_note_name(midi)})  f0≈{f0:.1f} Hz  B={b:.6f}",
            4000,
        )

        # Persist immediately (cheap)
        self._save_persisted_piano()

    def _on_record_next(self) -> None:
        """Auto-advance helper with strong support for scale recording.

        When the user is recording a scale (C1→C2→C3... or similar), this now
        does the right thing: it strongly prefers the next octave of the same
        note class before falling back to generic ascending or the old heuristic.
        """
        _diag(
            f"[DIAG][AutoAdvance] _on_record_next called | last_recorded={getattr(self, '_last_recorded_midi', None)} | scale_class={self._scale_pitch_class}"
        )
        piano = self._ensure_piano()
        last = getattr(self, "_last_recorded_midi", None)
        current = (
            self._record_selected_midi
            or last
            or (self._last_est.get("midi", 60) if self._last_est else 60)
        )

        def is_unmeasured(m: int) -> bool:
            if not (21 <= m <= 108):
                return False
            if m not in piano.keys:
                return True
            k = piano.keys[m]
            return k.measured_b is None and k.measured_f0 is None

        # === Dedicated scale recording mode ===
        if self._scale_pitch_class is not None:
            # Find the lowest unmeasured note that matches the current pitch class,
            # starting from the last recorded one and going upward.
            start = (last or current) - ((last or current) % 12) + self._scale_pitch_class
            for candidate in range(max(21, start), 109, 12):
                if is_unmeasured(candidate):
                    self._record_selected_midi = candidate
                    self.keyboard.set_current_key(candidate)
                    self.statusBar().showMessage(
                        f"Next to record: {candidate} ({midi_to_note_name(candidate)}).",
                        3500,
                    )
                    return

            # If no more in this class, we have exhausted the current root-note series.
            # The commit-time switch logic in _decide_commit_and_maybe_switch will have
            # already flipped _scale_pitch_class if the musician just played the first
            # note of the other known series (C↔F). If we are still here with the old class,
            # either the user stopped or they are starting something outside the known pair.
            _diag(
                f"[DIAG][AutoAdvance] Scale series for pc={self._scale_pitch_class} exhausted "
                f"(last={last}). Will use fallback or wait for commit-time switch on next capture."
            )
            # We deliberately do *not* clear _scale_pitch_class here — the next successful
            # capture of a note in the other known class will perform the eager switch.
            # Manual disarm or arming a different target is the explicit way to leave scale mode.

        # === Fallbacks ===

        # Simple ascending from current (still useful)
        for candidate in range(current + 1, 109):
            if is_unmeasured(candidate):
                _diag(f"[DIAG][AutoAdvance] Chose via fallback ascending: {candidate}")
                self._record_selected_midi = candidate
                self.keyboard.set_current_key(candidate)
                self.statusBar().showMessage(
                    f"Next to record: {candidate} ({midi_to_note_name(candidate)}).",
                    3500,
                )
                return

        # Old broad heuristic as last resort
        candidates = (
            list(range(48, 85)) + list(range(36, 48)) + list(range(85, 109)) + list(range(21, 36))
        )
        for m in candidates:
            if is_unmeasured(m):
                self._record_selected_midi = m
                self.keyboard.set_current_key(m)
                self.statusBar().showMessage(
                    f"Next to record: {m} ({midi_to_note_name(m)}).",
                    3500,
                )
                return

        # Everything measured
        nxt = min(108, current + 1)
        self._record_selected_midi = nxt
        self.keyboard.set_current_key(nxt)
        self.statusBar().showMessage(f"All keys measured. Advanced to {nxt}.", 2000)

    # ---------------- Auto-record (level triggered, hands-free) ----------------

    def _toggle_auto_record_arm(self, checked: bool) -> None:
        """User clicks the big Arm button. Delegates to the TDD'd controller."""
        target = self._record_selected_midi or (
            self._last_est.get("midi") if self._last_est else None
        )

        if checked:
            self._auto_record_ctrl.arm(target)
            self._f0_tracker.clear()  # fresh temporal history for this arm

            # Layer 1 bootstrap: when the user starts arming a series with auto-advance,
            # immediately enter scale mode for that pitch class. This gives the very first
            # note of the series (C1 or F1, etc.) the benefit of the pitch-class gate.
            if getattr(self, "_auto_advance_after_record", True) and target is not None:
                self._scale_pitch_class = int(target) % 12
                import time as _t

                self._scale_gate_grace_until = (
                    _t.time() + 0.65
                )  # grace for transient est on real piano attacks
                _diag(
                    f"[DIAG][ScaleGate] Entered scale mode for pitch class {self._scale_pitch_class} (armed on {target})"
                )

            target_str = f"{target} ({midi_to_note_name(target)})" if target else "any note"
            self._arm_record_action.setText("⏹ Stop Arming")

            # The controller owns the phase; we just paint what it says
            if target:
                self.keyboard.set_key_state(target, KeyState.ARMED)
                self.keyboard.set_current_key(target)
                self._apply_auto_record_visual_force()

            self.statusBar().showMessage(
                f"AUTO-RECORD ARMED — Play {target_str} on the piano. It will auto-capture after onset.",
                0,
            )

            # One-time helpful explanation (unchanged UX)
            if not getattr(self, "_shown_arm_help", False):
                QMessageBox.information(
                    self,
                    "Auto-Record (Hands-Free)",
                    "Walk to the piano.\n\n"
                    "1. Click the target key on screen (or use Record Next).\n"
                    "2. Play the physical note on your piano.\n"
                    "3. The key will be red while armed. The software will auto-detect the sound and record ~1.8s.\n\n"
                    "If Auto-advance is ON, it will move to the next key and re-arm after capture.",
                )
                self._shown_arm_help = True

        else:
            self._auto_record_ctrl.disarm()
            self._arm_record_action.setText("🎙️ Arm Auto-Record")

            # Layer 1: leaving scale mode when the user explicitly stops arming
            self._scale_pitch_class = None
            _diag("[DIAG][ScaleGate] Exited scale mode (manual disarm)")

            # Revert visual state for the (ex-)target
            if target:
                piano = self._piano
                if piano and target in piano.keys and piano.keys[target].measured_b is not None:
                    self.keyboard.set_key_state(target, KeyState.MEASURED)
                else:
                    self.keyboard.set_key_state(target, KeyState.UNMEASURED)

            self.statusBar().showMessage("Auto-record disarmed.", 2000)

    def _on_auto_onset_confirmed(self) -> None:
        """Reaction to the controller saying 'we have a real sustained onset'."""
        # Make sure we have a target
        if self._record_selected_midi is None and self._last_est:
            self._record_selected_midi = int(self._last_est.get("midi", 60))

        target = self._record_selected_midi or 60

        self._arm_record_action.setChecked(False)
        self._arm_record_action.setText("🎙️ Arm Auto-Record")

        self.statusBar().showMessage(
            f"RECORDING... (auto) — {target} ({midi_to_note_name(target)}) — keep playing for ~{self._auto_record_ctrl.capture_duration_ms} ms",
            0,
        )

        self.keyboard.set_current_key(target)
        self.keyboard.set_key_state(
            target, KeyState.RECORDING
        )  # strong red for the whole capture window
        self._apply_auto_record_visual_force()

        # Reset any previous during-capture rejection state for the new capture window (step 5)
        self._during_capture_rejection_until = 0.0

    def _apply_auto_record_visual_force(self) -> None:
        """If the controller currently owns a target in ARMED or RECORDING, force that painting."""
        forced = self._auto_record_ctrl.get_forced_visual_state()
        if forced:
            m, s = forced
            self.keyboard.set_key_state(m, s)
            self.keyboard.set_current_key(m)

    def _finish_auto_capture(self, commit: bool = True) -> None:
        """End of auto-capture window.

        With the better architecture we no longer blindly commit whatever the energy
        detector happened to record. We ask _decide_commit_and_maybe_switch first.
        Only good captures (correct pitch class + within tolerance of armed target)
        are stored and allowed to advance the series.
        """
        import time as _time

        if not commit:
            self.statusBar().showMessage("Auto-record cancelled.", 1500)
            self._auto_record_ctrl.disarm()
            return

        if self._last_est is None:
            self.statusBar().showMessage(
                "Auto-record finished but no analysis data. Try again.", 3000
            )
            self._auto_record_ctrl.disarm()
            return

        # === The key architectural change: authoritative decision at commit time ===
        if not self._decide_commit_and_maybe_switch():
            # Bad capture (wrong class or too far from what was armed).
            # Reject: do NOT store, do NOT advance. Stay armed on the same target
            # so the user can immediately try again.
            target = self._record_selected_midi
            _diag(f"[DIAG][AutoCapture] Capture REJECTED — staying armed on {target}")
            self.statusBar().showMessage(
                "Capture rejected (wrong note for current series/target). Try again.", 2500
            )

            # Give the user a moment and require a clean new attack
            now = _time.time()
            self._require_strong_attack_until = now + 1.0
            self._ignore_onset_until = now + 0.35

            # Re-arm the exact same target so the red ARMED state + controller stay alive
            if target is not None:
                self._auto_record_ctrl.arm(target)
                self._f0_tracker.clear()
                self.keyboard.set_key_state(target, KeyState.ARMED)
                self.keyboard.set_current_key(target)
                self._apply_auto_record_visual_force()
                import time as _t

                self._scale_gate_grace_until = _t.time() + 0.5  # allow quick retry
            self._during_capture_rejection_until = 0.0
            return

        # === Good capture — proceed with the original commit + advance logic ===
        just_recorded = self._record_selected_midi
        self._f0_tracker.clear()  # new note after advance will build its own history
        self._on_record_note(visual_feedback=False)

        if just_recorded:
            self.keyboard.set_key_state(just_recorded, KeyState.MEASURED)
            self._last_recorded_midi = just_recorded

            # Note: we no longer blindly set _scale_pitch_class here from just_recorded;
            # the decision helper already handled any series switch.

        # Optional auto-advance + immediate re-arm (the main user workflow)
        if getattr(self, "_auto_advance_after_record", True):
            self._on_record_next()
            next_midi = self._record_selected_midi
            if next_midi:
                _diag(f"[DIAG][AutoAdvance] Re-armed via _on_record_next → {next_midi}")
                self._auto_record_ctrl.arm(next_midi)
                self.keyboard.set_key_state(next_midi, KeyState.ARMED)
                self.keyboard.set_current_key(next_midi)
                self._apply_auto_record_visual_force()
                import time as _t

                self._scale_gate_grace_until = _t.time() + 0.65

            self._arm_record_action.setChecked(True)
            self._arm_record_action.setText("⏹ Stop Arming")
            self.statusBar().showMessage(
                f"Auto-captured. Re-armed for next key ({next_midi}). Go play it.",
                0,
            )

            # Require a stronger attack for the next onset (especially useful in scale mode)
            now = _time.time()
            self._require_strong_attack_until = now + 0.8
            self._ignore_onset_until = now + 0.3
        else:
            self._auto_record_ctrl.disarm()
            self.statusBar().showMessage(
                "Auto-captured. Arm again when ready for the next note.", 4000
            )
            now = _time.time()
            self._ignore_onset_until = now + 0.3
            self._require_strong_attack_until = now + 0.5
            self._during_capture_rejection_until = 0.0

    # ---------------- Expectation-driven scale mode helpers (Layer 1+) ----------------

    SCALE_MODE_CENT_TOLERANCE = 140.0
    # Looser tolerance used *only* for the Layer-1 onset pre-filter gate.
    # Real piano low-note attacks + chunked feeding often produce transient ests
    # that are 1 octave or partial-biased (e.g. 500-800 cents off the true fund).
    # The authoritative commit-time decision (_decide_commit_and_maybe_switch +
    # fresh estimator) still uses the strict SCALE tolerance + class check.
    ONSET_GATE_CENT_TOLERANCE = 800.0

    def _pitch_class_matches_expectation(
        self, est_midi_f: float | None, expected_pc: int, *, tolerance: float | None = None
    ) -> bool:
        """
        Return True if the estimated continuous MIDI value is within
        the given tolerance (default: SCALE_MODE_CENT_TOLERANCE) cents of a note
        whose pitch class matches expected_pc.

        The onset gate (Layer 1 pre-filter) calls with the looser ONSET_GATE_CENT_TOLERANCE
        to tolerate transient octave/partial estimation errors on real low piano notes.
        Commit-time validation always uses the strict default.
        """
        if est_midi_f is None:
            return (
                False  # conservative: if we have no estimate, don't trust the onset in scale mode
            )

        try:
            f_est = self._last_est.get("f_est") if self._last_est else None
            if f_est is None or f_est <= 1:
                # fall back to the (rounded) midi we were given
                midi_f = float(est_midi_f)
            else:
                midi_f = hz_to_midi(float(f_est), self._initial_a4)
        except Exception:
            midi_f = float(est_midi_f) if est_midi_f is not None else 60.0

        # Find the closest note (in cents) whose MIDI % 12 == expected_pc
        # Search a couple of octaves around the observation
        base = round(midi_f)
        best_dist = 9999.0
        for oct_off in (-2, -1, 0, 1, 2):
            for delta in (-12, 0, 12):
                cand = base + oct_off * 12 + delta
                if 21 <= cand <= 108 and (cand % 12) == (expected_pc % 12):
                    # cents distance = |cand - midi_f| * 100
                    d = abs(cand - midi_f) * 100.0
                    if d < best_dist:
                        best_dist = d

        tol = tolerance if tolerance is not None else self.SCALE_MODE_CENT_TOLERANCE
        matches = best_dist <= tol
        return matches

    def _decide_commit_and_maybe_switch(self) -> bool:
        """
        Called at CAPTURE_FINISHED.

        This is the authoritative correctness gate (better architecture):
        even if a wrong note slipped the onset gate, we can still refuse to
        store the measurement and refuse to auto-advance.

        Rules (v1):
        - If we are in scale mode (_scale_pitch_class is set), the captured note's
          pitch class must match.
        - The captured pitch must be within SCALE_MODE_CENT_TOLERANCE of the
          currently armed target (if any).
        - If both pass: accept, update last_recorded, possibly switch series,
          return True.
        - If anything fails: reject (caller must not commit), stay on the same
          armed target, return False.
        """
        armed = self._record_selected_midi

        # Prefer a fresh analysis of the audio that was just captured.
        # This is the key fix for continuous playing: _last_est can easily be
        # from the previous note because the full PFD analysis is slower than
        # the note rate.
        fresh = self._get_fresh_estimate_for_commit()
        live = self._last_est

        if fresh and fresh.get("f_est"):
            f_est = fresh.get("f_est")
            captured_midi = fresh.get("midi")
            est_source = "fresh"
        elif live and (live.get("f_est") or live.get("f0")):
            f_est = live.get("f_est") or live.get("f0")
            captured_midi = live.get("midi")
            est_source = "live (stale fallback)"
            _diag(
                "[DIAG][CommitDecision] WARNING: no fresh estimate, falling back to live _last_est"
            )
        else:
            _diag(
                "[DIAG][CommitDecision] REJECT: no usable pitch estimate (fresh or live) at capture end"
            )
            return False

        if f_est is None or captured_midi is None:
            _diag("[DIAG][CommitDecision] REJECT: incomplete estimate at capture end")
            return False

        captured_pc = int(captured_midi) % 12
        current_scale_pc = getattr(self, "_scale_pitch_class", None)

        # 1. Pitch-class check when in scale mode
        if current_scale_pc is not None and not self._pitch_class_matches_expectation(
            float(captured_midi), current_scale_pc
        ):
            err_cents = self._cents_error_to_target(f_est, armed) if armed else 999.0
            octave_err = self._is_probable_octave_or_partial_error(armed, captured_midi)
            tag = " (probable octave/partial error)" if octave_err else ""
            _diag(
                f"[DIAG][CommitDecision] REJECT (wrong class at commit){tag} | "
                f"source={est_source} captured_pc={captured_pc} expected_pc={current_scale_pc} "
                f"armed={armed} err_to_target={err_cents:.1f}¢ f_est={f_est:.1f}"
            )
            return False

        # 2. Tolerance check against the armed target (the thing the user intended to record)
        if armed is not None:
            err = self._cents_error_to_target(f_est, armed)
            if err is not None and abs(err) > self.SCALE_MODE_CENT_TOLERANCE:
                octave_err = self._is_probable_octave_or_partial_error(armed, f_est)
                tag = " (probable octave/partial error)" if octave_err else ""
                _diag(
                    f"[DIAG][CommitDecision] REJECT (too far from armed target){tag} | "
                    f"source={est_source} armed={armed} ({midi_to_note_name(armed)}) "
                    f"captured≈{captured_midi} error={err:.1f}¢ > {self.SCALE_MODE_CENT_TOLERANCE}¢ f_est={f_est:.1f}"
                )
                return False

        # Accept
        class_ok = current_scale_pc is None or self._pitch_class_matches_expectation(
            float(captured_midi), current_scale_pc
        )
        _diag(
            f"[DIAG][CommitDecision] ACCEPT | source={est_source} armed={armed} "
            f"captured≈{captured_midi} class_ok={class_ok}"
        )

        # Extra diagnostic: show how much the fresh estimate differed from the live one
        if fresh and live and est_source.startswith("fresh"):
            live_f = live.get("f_est") or live.get("f0")
            if live_f:
                diff_cents = 1200.0 * np.log2(f_est / max(live_f, 1))
                _diag(f"[DIAG][CommitDecision] fresh vs live diff = {diff_cents:+.1f}¢")

        # 3. Series switch at commit time (eager, single-note trigger, only for the known C/F pair)
        self._maybe_switch_series(captured_pc, current_scale_pc, captured_midi)

        return True

    def _maybe_switch_series(
        self, captured_pc: int, current_scale_pc: int | None, captured_midi: int | float
    ) -> None:
        """
        Eager series switch at commit time.

        Only switches between the two known root-note series we support today (C=0 and F=5).
        Triggered by a single captured note whose pitch class is the "other" one.
        No multi-note hysteresis (per design).

        Called only after a successful accept decision so we never switch on rejected captures.
        """
        if current_scale_pc is None:
            return

        if captured_pc == current_scale_pc:
            return

        other = 5 if current_scale_pc == 0 else 0
        if captured_pc == other:
            self._scale_pitch_class = captured_pc
            _diag(
                f"[DIAG][CommitDecision] Series SWITCHED to pitch class {captured_pc} "
                f"based on captured note {captured_midi}"
            )
            # Give the new series the same short grace as a fresh arm (helps first note of the new series)
            import time as _t

            self._scale_gate_grace_until = _t.time() + 0.65
        else:
            # Future-proof: we saw activity from a third class while in a known series.
            # Do not auto-switch; just log so we can decide later behavior.
            _diag(
                f"[DIAG][CommitDecision] Series switch opportunity ignored (unexpected class {captured_pc}, "
                f"current series {current_scale_pc}) for note {captured_midi}"
            )

    def _is_probable_octave_or_partial_error(
        self, armed_midi: int | None, est_midi_f: float | None
    ) -> bool:
        """
        Heuristic to detect the common failure mode seen in long diagnostics on real
        low piano notes: the estimator locking onto a strong upper partial or octave
        (e.g. reporting 48 or 72 when the actual note is C1=24).

        Returns True when the distance is close to a multiple of 12 semitones and
        the error is large enough that it's almost certainly not a real adjacent note.
        Used only for clearer diagnostic logging.
        """
        if armed_midi is None or est_midi_f is None:
            return False
        try:
            dist_semi = abs(est_midi_f - armed_midi)
            # Close to an exact number of octaves (or 2 octaves, etc.)
            nearest_octave = round(dist_semi / 12) * 12
            error_from_octave = abs(dist_semi - nearest_octave)
            # If it's within ~1.5 semitones of an octave multiple and the total distance
            # is > ~1.5 octaves, it's very likely an octave/partial error rather than
            # a musically adjacent wrong note.
            if error_from_octave < 1.5 and dist_semi > 18:
                return True
        except Exception:
            pass
        return False

    def _cents_error_to_target(self, f_est: float, target_midi: int) -> float | None:
        """Return signed cents error of f_est relative to the equal-tempered target_midi."""
        if f_est <= 1 or target_midi is None:
            return None
        try:
            target_hz = midi_to_hz(int(target_midi), self._initial_a4)
            if target_hz <= 1:
                return None
            return float(1200.0 * np.log2(f_est / target_hz))
        except Exception:
            return None

    def _get_fresh_estimate_for_commit(self) -> dict | None:
        """
        Run a fresh pitch analysis on the most recent audio in the ring buffer.
        Used at CAPTURE_FINISHED so the commit/reject decision is based on what
        was actually sounding during/near the end of this capture, not a stale
        live _last_est from earlier.

        Returns a dict in the same shape as _estimate_pitch (f_est, midi, f0, ...).
        Returns None if there isn't enough good audio right now.
        """
        try:
            fs = self.audio_capture.samplerate or 48000
            # Use a good analysis window (same size as live analysis for consistency)
            n = 32768
            audio = self.ringbuffer.get_latest(n)
            if len(audio) < 4096:
                return None

            # Quick energy check — if it's basically silence we shouldn't trust it
            rms = float(np.sqrt(np.mean(audio * audio)))
            if rms < 0.0008:  # very quiet, probably decay or noise
                return None

            est = self._estimate_pitch(audio, fs, self._initial_a4)
            return est
        except Exception as e:
            _diag(f"[DIAG][FreshEst] commit-time fresh analysis failed: {e}")
            return None

    def _on_compute_curve(self) -> None:
        """Run the minimal solver and wire the resulting curve into the live tuner immediately."""
        piano = self._ensure_piano()
        if not piano.has_measurements():
            # Still allow default curve for demo / "ET + simple stretch"
            QMessageBox.information(
                self,
                "Compute Curve",
                "No measurements yet — computing a default Railsback-style stretch curve (usable ET + stretch).\n\n"
                "For best results on your real piano, record 8-12 keys first (B values drive the fit).",
            )

        try:
            curve = compute_basic_tuning_curve(piano)
            piano.tuning_curve = curve
            # Also push offsets back into any recorded Key objects
            for k in piano.keys.values():
                k.target_offset_cents = piano.get_target_offset(k.midi)

            self._update_curve_status()
            self.statusBar().showMessage(
                "Curve computed. Colors: Green = close to your target, Orange = needs attention (far from target), Blue = measured. "
                "Tune until the strobe stops and cents are near 0.",
                8000,
            )

            # Give immediate visual confirmation
            self.hint.setText(
                "Curve active — play a note. The tuner now shows deviation from the computed per-key targets."
            )

            self._save_persisted_piano()

            # Optional: mark a few keys visually as "needs attention" if they deviate a lot (demo)
            for m in [21, 33, 45, 57, 69, 81, 93, 105]:
                if abs(piano.get_target_offset(m)) > 4.0:
                    self.keyboard.set_key_state(m, KeyState.NEEDS_ATTENTION)

        except Exception as exc:
            QMessageBox.warning(self, "Solver Error", f"Curve computation failed:\n{exc}")

    def _on_clear_measurements(self) -> None:
        piano = self._piano
        if piano is None:
            return
        piano.keys.clear()
        piano.tuning_curve = None
        self.keyboard.clear_all()
        self._record_selected_midi = None
        self._update_curve_status()
        self.statusBar().showMessage("Measurements and curve cleared.", 2000)
        self._save_persisted_piano()

    def _on_start_recording(self) -> None:
        """Convenience: reset selection and give guidance."""
        self._record_selected_midi = 60  # middle C area
        self.keyboard.set_current_key(60)
        self.statusBar().showMessage(
            "Recording mode ready. Click keys on the virtual keyboard or use 'Record Next'. Play real piano key then Record.",
            5000,
        )

    # ---------------- Persistence (Phase 4 simple JSON) ----------------

    def _load_persisted_piano(self) -> None:
        path = Piano.default_persist_path()
        loaded = Piano.load_json(path)
        if loaded is not None:
            self._piano = loaded
            # Restore visual measured states on keyboard
            for m, k in self._piano.keys.items():
                if k.measured_b is not None or k.measured_f0 is not None:
                    self.keyboard.set_key_state(m, KeyState.MEASURED)
            if self._piano.tuning_curve:
                self._update_curve_status()
            else:
                self._update_curve_status()
            self.statusBar().showMessage("Restored previous piano session.", 1800)

    def _save_persisted_piano(self) -> None:
        if self._piano is None:
            return
        # never let persistence kill the app
        with contextlib.suppress(Exception):
            self._piano.save_json(Piano.default_persist_path())

    # ---------------- Menu / toolbar handlers (updated for Phase 4) ----------------

    @Slot()
    def _on_input_device(self) -> None:
        """Open the polished DeviceSelectorDialog and apply choice."""
        dlg = DeviceSelectorDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            dev = dlg.get_selected_device()
            if dev is not None:
                self._apply_audio_device(dev)

    @Slot()
    def _on_test_signal(self) -> None:
        QMessageBox.information(
            self,
            "Test Signal",
            "Test Signal / reference tone playback is available inside the DeviceSelector 'Test' button.\n\n"
            "Phase 3+ will add toolbar-driven synth playback that you can detune live while watching the strobe.",
        )

    @Slot()
    def _on_new_piano(self) -> None:
        """Phase 4: start fresh recording session."""
        self._piano = None
        self._record_selected_midi = None
        self.keyboard.clear_all()
        self._update_curve_status()
        self.statusBar().showMessage("New piano session started. Begin recording keys.", 3000)
        self._save_persisted_piano()

    @Slot()
    def _on_open(self) -> None:
        # For v0.1 the auto-load on launch + explicit New/Clear cover the need.
        # Full file dialog can be added later.
        path = Piano.default_persist_path()
        loaded = Piano.load_json(path)
        if loaded:
            self._piano = loaded
            self.keyboard.clear_all()
            for m, k in self._piano.keys.items():
                if k.measured_b is not None or k.measured_f0 is not None:
                    self.keyboard.set_key_state(m, KeyState.MEASURED)
            self._update_curve_status()
            QMessageBox.information(self, "Open", f"Loaded session from {path}")
        else:
            QMessageBox.information(
                self, "Open", "No saved session found (auto-saved on every record)."
            )

    @Slot()
    def _on_save(self) -> None:
        if self._piano:
            self._save_persisted_piano()
            QMessageBox.information(
                self,
                "Save",
                f"Session saved to {Piano.default_persist_path()} (also auto-saved after each record).",
            )
        else:
            QMessageBox.information(self, "Save", "Nothing to save yet — record a few keys first.")

    @Slot()
    def _on_about(self) -> None:
        ver = "0.4"
        with contextlib.suppress(Exception):
            ver = __import__("optitune").__version__
        QMessageBox.about(
            self,
            "About OptiTune",
            f"""<b>OptiTune</b> v{ver}<br><br>
            Professional one-click Linux piano tuning workstation.<br><br>
            Built with PySide6, pyqtgraph, sounddevice, NumPy/SciPy/Numba.<br>
            100% test-driven with synthetic inharmonic piano tones (Fletcher-Young + PFD).<br><br>
            <b>Phase 4:</b> Record notes from your real detuned piano → minimal B-curve + stretch solver → live tuner targets the computed curve.<br>
            Final user test: capture 8-12 keys, Compute Curve, tune to the resulting per-key targets.<br><br>
            © 2026 OptiTune Contributors — Licensed under the GNU GPL v3.<br>
            <a href="https://github.com/z3n/optitune">github.com/z3n/optitune</a>
            """,
        )

    @Slot()
    def _on_strobe_clicked(self) -> None:
        self.cents_display.reset()
        self.strobe.reset()

    @Slot()
    def _on_play_test_tone(self) -> None:
        QMessageBox.information(
            self,
            "Play Test Tone",
            "This will become a powerful training tool:\n\n"
            "• Choose a note + exact detune in cents\n"
            "• Play a perfectly synthesized inharmonic piano tone\n"
            "• Watch the strobe + cents track in real time as you 'tune'\n\n"
            "Full wiring coming soon (uses the same dsp.synth engine already validated in Phase 1).",
        )

    @Slot()
    def _on_reset_displays(self) -> None:
        self.cents_display.reset()
        self.strobe.reset()
        self.spectrum.clear()
        self.keyboard.clear_all()
        self.statusBar().showMessage("Displays reset", 1200)

    def _update_auto_advance_ui(self, checked: bool) -> None:
        """Make the auto-advance toggle state extremely obvious."""
        self._auto_advance_after_record = checked
        if checked:
            self._auto_advance_action.setText("➡️ Auto-advance ON")
            self._auto_advance_action.setToolTip(
                "ON: After auto-capture, jump to next unmeasured key and re-arm automatically (great for walking the piano)."
            )
        else:
            self._auto_advance_action.setText("Auto-advance OFF")
            self._auto_advance_action.setToolTip("OFF: You control which key to target next.")

    # Graceful shutdown
    def closeEvent(self, event: QCloseEvent) -> None:
        with contextlib.suppress(Exception):
            if self._level_timer:
                self._level_timer.stop()
            if self._analysis_timer:
                self._analysis_timer.stop()
            self.audio_capture.stop()
            self._save_persisted_piano()
        super().closeEvent(event)
