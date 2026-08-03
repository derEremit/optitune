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
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QCloseEvent, QColor, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
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
from optitune.dsp.note_follow import NoteFollowMode
from optitune.model import Key, Piano
from optitune.recording.auto_record import (
    AutoRecordConfig,
    AutoRecordController,
    AutoRecordEvent,
)
from optitune.recording.scale_session import (
    ONSET_GATE_CENT_TOLERANCE as _ONSET_GATE_CENT_TOLERANCE,
)
from optitune.recording.scale_session import (
    SCALE_MODE_CENT_TOLERANCE as _SCALE_MODE_CENT_TOLERANCE,
)
from optitune.recording.scale_session import (
    ScaleSession,
    pitch_class_matches,
)
from optitune.solvers import (
    BeatRateSolver,
    TuningConstraints,
    available_solvers,
    compute_basic_tuning_curve,
    get_solver,
)
from optitune.solvers.interval_weights import DEFAULT_INTERVAL_WEIGHTS
from optitune.ui.dialogs import (
    DeviceSelectorDialog,
    IntervalWeightsDialog,
    NewPianoDialog,
    PitchRaiseDialog,
)
from optitune.model.inharmonicity import measured_b_from_piano
from optitune.persistence.settings import AppSettings
from optitune.persistence.ept_import import load_ept
from optitune.persistence.tuning_file import load_pfg, save_pfg
from optitune.ui.widgets import (
    BCurveWidget,
    CentsDisplay,
    KeyboardWidget,
    RailsbackWidget,
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
        self._app_settings = AppSettings()
        self._settings = self._app_settings.raw  # compat for any remaining direct use
        self._current_pfg_path: str | None = None
        self._session_dirty: bool = False

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
                capture_duration_ms=1100,  # fit scale note spacing; see AutoRecordConfig
            )
        )
        self._auto_advance_after_record: bool = True  # very useful for walking the piano
        # Expectation layer (pure SM) — properties below mirror fields for compat
        self._scale_session = ScaleSession()
        self._last_recorded_midi: int | None = None
        self._note_follow_mode: NoteFollowMode = NoteFollowMode.AUTO
        self._follow_locked_midi: int | None = None  # Lock/Stepwise anchor
        self._temperament: str = "equal"
        self._interval_weights: dict[str, float] = dict(DEFAULT_INTERVAL_WEIGHTS)
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
        self._refresh_curve_widgets()
        # Resume scale series if the previous session was interrupted mid-arm
        self._restore_scale_session_settings()

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

        # Spectrum + curve graphs (Railsback + B)
        mid_row = QSplitter(Qt.Orientation.Horizontal)
        self.spectrum = SpectrumWidget()
        curve_col = QWidget()
        curve_layout = QVBoxLayout(curve_col)
        curve_layout.setContentsMargins(0, 0, 0, 0)
        curve_layout.setSpacing(4)
        self.railsback = RailsbackWidget()
        self.b_curve = BCurveWidget()
        curve_layout.addWidget(self.railsback, 1)
        curve_layout.addWidget(self.b_curve, 1)
        mid_row.addWidget(self.spectrum)
        mid_row.addWidget(curve_col)
        mid_row.setSizes([480, 420])
        layout.addWidget(mid_row, 3)

        # Keyboard (highlights detected note + measured states for recording)
        self.keyboard = KeyboardWidget()
        layout.addWidget(self.keyboard, 1)

        # Subtle live hint (updated in Phase 4)
        self.hint = QLabel(
            "Live DSP active - play notes on your piano. Use toolbar to record keys & compute a stretch curve.  "
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

        save_as_action = QAction("Save &As…", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._on_save_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Audio - now fully wired
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

        weights_action = QAction("Interval Weights…", self)
        weights_action.triggered.connect(self._on_interval_weights)
        tune_menu.addAction(weights_action)

        pitch_raise_action = QAction("Pitch Raise / Overpull…", self)
        pitch_raise_action.triggered.connect(self._on_pitch_raise)
        tune_menu.addAction(pitch_raise_action)

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

        # Note-follow modes (spec §3.6): Auto / Stepwise / Lock
        tb.addWidget(QLabel(" Follow:"))
        self._follow_combo = QComboBox()
        self._follow_combo.addItem("Auto", NoteFollowMode.AUTO)
        self._follow_combo.addItem("Stepwise", NoteFollowMode.STEPWISE)
        self._follow_combo.addItem("Lock", NoteFollowMode.LOCK)
        self._follow_combo.setToolTip(
            "Auto: jump to any detected note.\n"
            "Stepwise: only ±1 semitone from the locked key (anti-octave jumps).\n"
            "Lock: keep the selected key; detection does not switch."
        )
        self._follow_combo.setMinimumWidth(100)
        self._follow_combo.currentIndexChanged.connect(self._on_follow_mode_changed)
        tb.addWidget(self._follow_combo)

        tb.addSeparator()

        tb.addWidget(QLabel(" Solver:"))
        self._solver_combo = QComboBox()
        for name in available_solvers():
            self._solver_combo.addItem(name, name)
        self._solver_combo.setToolTip(
            "beat-rate: weighted interval LS (needs B measurements).\n"
            "entropy: Hinrichsen spectrum entropy (needs cent spectra from Record)."
        )
        self._solver_combo.setMinimumWidth(110)
        tb.addWidget(self._solver_combo)

        self._compute_action = QAction("📈 Compute Curve", self)
        self._compute_action.setShortcut("Ctrl+K")
        self._compute_action.setToolTip(
            "Run selected solver on recorded measurements → live targets update immediately"
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

        self._series_label = QLabel("  Series: -")
        self._series_label.setMinimumWidth(180)
        self._series_label.setStyleSheet("color: #a0a0a8;")

        status.addPermanentWidget(self._device_label)
        status.addPermanentWidget(self._level_bar)
        status.addPermanentWidget(self._a4_label)
        status.addPermanentWidget(self._series_label)
        status.addPermanentWidget(self._curve_status_label)

        # Initial message (will be overwritten by audio init)
        status.showMessage("Initializing audio...", 1500)
        self._update_series_status()

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

    def _update_series_status(self) -> None:
        """Refresh 'Series: C (2/7) -> C3' style indicator."""
        if not hasattr(self, "_series_label") or self._series_label is None:
            return
        pc = self._scale_pitch_class
        if pc is None:
            self._series_label.setText("  Series: -")
            self._series_label.setStyleSheet("color: #a0a0a8;")
            return
        from optitune.recording.scale_session import pitch_class_name, series_hi as _series_hi

        name = pitch_class_name(pc)
        hi = _series_hi(pc)
        first = next(m for m in range(21, 109) if m % 12 == pc)
        total = len(list(range(first, hi + 1, 12)))
        measured = 0
        if self._piano is not None:
            for m in range(first, hi + 1, 12):
                k = self._piano.keys.get(m)
                if k is not None and (k.measured_f0 is not None or k.measured_b is not None):
                    measured += 1
        armed = self._record_selected_midi
        armed_txt = ""
        if armed is not None:
            armed_txt = f" -> {midi_to_note_name(armed)}"
        self._series_label.setText(f"  Series: {name} ({measured}/{total}){armed_txt}")
        self._series_label.setStyleSheet("color: #7dcea0; font-weight: 500;")

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
            saved = self._app_settings.get_last_input_device_index()
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
            self._app_settings.set_last_input_device_index(int(device_index))
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
                    "Play notes on your piano - the cents, strobe, spectrum and keyboard will track using real DSP (peaks + PFD).",
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
        # Level meter - fast, cheap (every 50 ms)
        self._level_timer = QTimer(self)
        self._level_timer.timeout.connect(self._update_level_meter)
        self._level_timer.start(50)

        # Real analysis - 100 ms gives good overlap on 32k windows while staying light
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
            # on every tick while post-capture suppression was active - exactly the "hanging" symptom
            # during scale-mode series with _ignore_onset_until).
            require_strong = getattr(self, "_require_strong_attack_until", 0) > now

            # Post-capture protection
            if getattr(self, "_ignore_onset_until", 0) > now:
                event = None
            else:
                # Extra strict attack requirement after recent capture (especially in scale mode)
                min_rise = 6.0 if require_strong else 3.0

                if require_strong and db_rise < min_rise:
                    event = None
                else:
                    # === Layer 1: Expectation-driven pitch-class gate (scale recording mode) ===
                    # Pure logic in ScaleSession.should_suppress_onset; UI only logs + feeds.
                    scale_pc = self._scale_pitch_class
                    if scale_pc is not None:
                        est_midi_for_gate = None
                        f_est_gate = None
                        if self._last_est:
                            m = self._last_est.get("midi")
                            if m is not None:
                                est_midi_for_gate = float(m)
                            f_est_gate = self._last_est.get("f_est")
                        armed = self._record_selected_midi
                        suppress = self._scale_session.should_suppress_onset(
                            est_midi=est_midi_for_gate,
                            armed_midi=armed,
                            now=now,
                            f_est=float(f_est_gate) if f_est_gate else None,
                            a4=float(self._initial_a4),
                        )
                        if suppress:
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
                            if est_midi_for_gate is None and not self._scale_session.in_grace(now):
                                _diag(
                                    "[DIAG][ScaleGate] no-est allow energy path | armed=%s dB=%.1f",
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

            # --- During-capture validation + subtle rejection feedback ---
            # Live estimate vs expected class/target while recording. Does not
            # abort capture (controller stays energy-only); commit gate is final.
            if (
                self._auto_record_ctrl.is_recording
                and self._scale_pitch_class is not None
                and self._record_selected_midi is not None
                and self._last_est is not None
                and (now - self._last_during_cap_check) > 0.32
            ):
                self._last_during_cap_check = now
                est_m = self._last_est.get("midi")
                if est_m is not None:
                    armed = self._record_selected_midi
                    scale_pc = self._scale_pitch_class
                    class_ok = self._pitch_class_matches_expectation(float(est_m), scale_pc)
                    target_err = self._cents_error_to_target(
                        self._last_est.get("f_est", est_m), armed
                    )
                    target_ok = (
                        target_err is None or abs(target_err) <= self.SCALE_MODE_CENT_TOLERANCE
                    )
                    # Tolerate estimator octave/partial jumps on low piano notes
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
                        self._during_capture_rejection_until = now + 0.4
                        # Only re-trigger flash if not already flashing this key
                        if self.keyboard.rejection_flash_midi != armed:
                            self.keyboard.flash_rejection(armed, duration_ms=350)
                    else:
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

            # ~0.68 s window - excellent resolution (~1.5 Hz bins), still responsive
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
        Core live estimator - thin adapter over pure dsp.estimate_pitch.

        Armed target and last-frame f0 are soft priors only.
        """
        n = len(audio)
        if n < 256:
            return self._fallback_estimate(audio, fs, a4)
        # Armed soft prior while in scale/auto-record; free listening uses follow mode.
        in_record_workflow = (
            self._scale_pitch_class is not None
            or self._auto_record_ctrl.is_armed
            or self._auto_record_ctrl.is_recording
        )
        armed = self._record_selected_midi if in_record_workflow else None
        return estimate_pitch(
            audio,
            fs,
            a4=a4,
            armed_midi=armed,
            last_f0_guess=self._last_f0_guess,
            scale_cent_tol=self.SCALE_MODE_CENT_TOLERANCE,
            follow_mode=self._note_follow_mode,
            locked_midi=self._follow_locked_midi,
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
        self._follow_locked_midi = midi  # anchor for Stepwise / Lock follow modes
        self.statusBar().showMessage(
            f"Key {midi} ({midi_to_note_name(midi)}) selected for recording. Play the note then click 'Record Note'.",
            3000,
        )

    def _on_follow_mode_changed(self, _index: int) -> None:
        data = self._follow_combo.currentData()
        if isinstance(data, NoteFollowMode):
            self._note_follow_mode = data
        elif data is not None:
            self._note_follow_mode = NoteFollowMode(str(data))
        # Seed lock anchor from current selection if missing
        if self._follow_locked_midi is None and self._record_selected_midi is not None:
            self._follow_locked_midi = self._record_selected_midi
        mode = self._note_follow_mode.value
        self.statusBar().showMessage(f"Note follow: {mode}", 2000)

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

        # Capture A-weighted cent spectrum for entropy solver (compressed in JSON)
        spectrum = None
        try:
            from optitune.model.spectrum_codec import spectrum_from_audio_a_weighted

            fs = float(self.audio_capture.samplerate or 48000)
            n = min(int(fs * 1.2), 65536)
            audio = self.ringbuffer.get_latest(n)
            if audio is not None and len(audio) >= 256:
                spectrum = spectrum_from_audio_a_weighted(audio, fs)
        except Exception:
            spectrum = None

        k = Key(midi=midi, measured_f0=f0, measured_b=b, cent_spectrum=spectrum)
        piano.set_key(k)

        if visual_feedback:
            self.keyboard.set_key_state(midi, KeyState.MEASURED)
            self.keyboard.set_current_key(midi)
            self._record_action.setText("⏺ Record Note")

        self._record_selected_midi = midi
        self._update_curve_status()
        self._refresh_curve_widgets()
        self._mark_session_dirty()
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

        # === Dedicated scale recording mode (pure ScaleSession) ===
        if self._scale_pitch_class is not None:
            measured = {
                m
                for m, k in piano.keys.items()
                if k.measured_b is not None or k.measured_f0 is not None
            }
            prev_pc = self._scale_pitch_class
            candidate = self._scale_session.next_target(
                last_recorded=last, measured=measured, current=current
            )
            if candidate is not None:
                if self._scale_pitch_class != prev_pc:
                    _diag(
                        f"[DIAG][AutoAdvance] Scale series pc={prev_pc} exhausted "
                        f"(last={last}); switching to paired series pc={self._scale_pitch_class} "
                        f"-> {candidate}"
                    )
                    msg = (
                        f"Series complete. Next series: {candidate} "
                        f"({midi_to_note_name(candidate)})."
                    )
                else:
                    msg = f"Next to record: {candidate} ({midi_to_note_name(candidate)})."
                self._record_selected_midi = candidate
                self.keyboard.set_current_key(candidate)
                self.statusBar().showMessage(msg, 4000)
                return
            # Paired C↔F series fully measured — leave selection cleared for
            # completion UX (caller disarms + exit_scale). Do not fall through
            # to ascending non-series keys while still in scale mode.
            _diag(
                f"[DIAG][AutoAdvance] Scale series for pc={self._scale_pitch_class} exhausted "
                f"(last={last}). No paired series notes left."
            )
            self._record_selected_midi = None
            return

        # === Fallbacks (non-scale / free auto-advance) ===

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
                import time as _t

                self._scale_session.enter_scale(int(target), now=_t.time())
                self._update_series_status()
                self._persist_scale_session_settings()
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
                f"AUTO-RECORD ARMED - Play {target_str} on the piano. It will auto-capture after onset.",
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
            self._scale_session.exit_scale()
            self._clear_scale_session_settings()
            self._update_series_status()
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
            f"RECORDING... (auto) - {target} ({midi_to_note_name(target)}) - keep playing for ~{self._auto_record_ctrl.capture_duration_ms} ms",
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

        # Prefer a fresh ring-buffer estimate even when live _last_est is missing
        # (analysis tick may have been skipped during the capture window).
        if self._last_est is None:
            fresh = self._get_fresh_estimate_for_commit()
            if fresh is not None:
                self._last_est = fresh
                _diag("[DIAG][AutoCapture] seeded _last_est from fresh commit estimate")
            else:
                _diag("[DIAG][AutoCapture] REJECT: no live or fresh estimate at capture end")
                self.statusBar().showMessage(
                    "Auto-record finished but no analysis data. Try again.", 3000
                )
                # Stay armed on the same target for a retry (scale workflow)
                target = self._record_selected_midi
                if target is not None:
                    self._auto_record_ctrl.arm(target)
                    self._f0_tracker.clear()
                    self.keyboard.set_key_state(target, KeyState.ARMED)
                    self.keyboard.set_current_key(target)
                    self._apply_auto_record_visual_force()
                return

        # === The key architectural change: authoritative decision at commit time ===
        if not self._decide_commit_and_maybe_switch():
            # Bad capture (wrong class or too far from what was armed).
            # Reject: do NOT store, do NOT advance. Stay armed on the same target
            # so the user can immediately try again.
            target = self._record_selected_midi
            _diag(f"[DIAG][AutoCapture] Capture REJECTED - staying armed on {target}")
            self.statusBar().showMessage(
                "Capture rejected (wrong note for current series/target). Try again.", 2500
            )

            # Brief post-reject guard (must stay short vs scale note spacing)
            now = _time.time()
            self._scale_session.set_post_capture_guards(now=now, success=False)

            # Re-arm the exact same target so the red ARMED state + controller stay alive
            if target is not None:
                self._auto_record_ctrl.arm(target)
                self._f0_tracker.clear()
                self.keyboard.set_key_state(target, KeyState.ARMED)
                self.keyboard.set_current_key(target)
                self.keyboard.flash_rejection(target, duration_ms=450)
                self._apply_auto_record_visual_force()
            self._during_capture_rejection_until = now + 0.4
            self._update_series_status()
            return

        # === Good capture - proceed with the original commit + advance logic ===
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
            now = _time.time()
            self._scale_session.set_post_capture_guards(now=now, success=True)
            if next_midi:
                _diag(f"[DIAG][AutoAdvance] Re-armed via _on_record_next → {next_midi}")
                self._auto_record_ctrl.arm(next_midi)
                self.keyboard.set_key_state(next_midi, KeyState.ARMED)
                self.keyboard.set_current_key(next_midi)
                self._apply_auto_record_visual_force()
                self._arm_record_action.setChecked(True)
                self._arm_record_action.setText("⏹ Stop Arming")
                self.statusBar().showMessage(
                    f"Auto-captured. Re-armed for next key ({next_midi}). Go play it.",
                    0,
                )
            else:
                # Paired series exhausted (C1–C7 and F1–F7 all measured)
                self._auto_record_ctrl.disarm()
                self._arm_record_action.setChecked(False)
                self._arm_record_action.setText("🎯 Arm for Auto-Record")
                self._scale_session.exit_scale()
                self._clear_scale_session_settings()
                self.statusBar().showMessage(
                    "Series complete — no more notes in this (paired) series. Disarmed.",
                    6000,
                )
            if next_midi:
                self._persist_scale_session_settings()
            self._update_series_status()
        else:
            self._auto_record_ctrl.disarm()
            self.statusBar().showMessage(
                "Auto-captured. Arm again when ready for the next note.", 4000
            )
            self._scale_session.set_post_capture_guards(now=_time.time(), success=True)

    # ---------------- Expectation-driven scale mode helpers (Layer 1+) ----------------

    SCALE_MODE_CENT_TOLERANCE = _SCALE_MODE_CENT_TOLERANCE
    ONSET_GATE_CENT_TOLERANCE = _ONSET_GATE_CENT_TOLERANCE

    @property
    def _scale_pitch_class(self) -> int | None:
        return self._scale_session.scale_pitch_class

    @_scale_pitch_class.setter
    def _scale_pitch_class(self, value: int | None) -> None:
        self._scale_session.scale_pitch_class = value

    @property
    def _scale_gate_grace_until(self) -> float:
        return self._scale_session.grace_until

    @_scale_gate_grace_until.setter
    def _scale_gate_grace_until(self, value: float) -> None:
        self._scale_session.grace_until = float(value)

    @property
    def _during_capture_rejection_until(self) -> float:
        return self._scale_session.during_capture_rejection_until

    @_during_capture_rejection_until.setter
    def _during_capture_rejection_until(self, value: float) -> None:
        self._scale_session.during_capture_rejection_until = float(value)

    @property
    def _ignore_onset_until(self) -> float:
        return self._scale_session.ignore_onset_until

    @_ignore_onset_until.setter
    def _ignore_onset_until(self, value: float) -> None:
        self._scale_session.ignore_onset_until = float(value)

    @property
    def _require_strong_attack_until(self) -> float:
        return self._scale_session.require_strong_attack_until

    @_require_strong_attack_until.setter
    def _require_strong_attack_until(self, value: float) -> None:
        self._scale_session.require_strong_attack_until = float(value)

    def _pitch_class_matches_expectation(
        self, est_midi_f: float | None, expected_pc: int, *, tolerance: float | None = None
    ) -> bool:
        """Delegate to pure scale_session.pitch_class_matches."""
        f_est = None
        try:
            if self._last_est:
                f_est = self._last_est.get("f_est")
        except Exception:
            f_est = None
        return pitch_class_matches(
            est_midi_f,
            expected_pc,
            tolerance=tolerance if tolerance is not None else self.SCALE_MODE_CENT_TOLERANCE,
            f_est=float(f_est) if f_est else None,
            a4=self._initial_a4,
        )

    def _decide_commit_and_maybe_switch(self) -> bool:
        """
        Called at CAPTURE_FINISHED.

        Thin adapter: fetch estimates here, pure accept/reject + C↔F switch
        live in ScaleSession.decide_commit.
        """
        armed = self._record_selected_midi

        # Prefer a fresh analysis of the audio that was just captured.
        # _last_est can easily be from the previous note (full PFD is slower
        # than the note rate on continuous playing).
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

        tracked = self._f0_tracker.current()
        tracker_f0 = float(tracked) if tracked is not None and tracked > 20 else None
        prev_pc = self._scale_pitch_class

        decision = self._scale_session.decide_commit(
            f_est=float(f_est),
            captured_midi=captured_midi,
            armed_midi=armed,
            a4=float(self._initial_a4),
            tracker_f0=tracker_f0,
        )

        f_used = decision.f_est_used if decision.f_est_used is not None else float(f_est)
        midi_used = (
            decision.captured_midi if decision.captured_midi is not None else captured_midi
        )

        if not decision.accept:
            octave_err = self._is_probable_octave_or_partial_error(armed, midi_used)
            tag = " (probable octave/partial error)" if octave_err else ""
            err_cents = (
                self._cents_error_to_target(f_used, armed) if armed is not None else 999.0
            )
            reason = decision.reason or "reject"
            if reason == "wrong_class":
                _diag(
                    f"[DIAG][CommitDecision] REJECT (wrong class at commit){tag} | "
                    f"source={est_source} captured_pc={int(midi_used) % 12} "
                    f"expected_pc={prev_pc} armed={armed} "
                    f"err_to_target={err_cents if err_cents is not None else 999:.1f}¢ "
                    f"f_est={f_used:.1f}"
                )
            else:
                _diag(
                    f"[DIAG][CommitDecision] REJECT ({reason}){tag} | "
                    f"source={est_source} armed={armed} "
                    f"({midi_to_note_name(armed) if armed is not None else '?'}) "
                    f"captured≈{midi_used} "
                    f"error={err_cents if err_cents is not None else 999:.1f}¢ "
                    f"> {self.SCALE_MODE_CENT_TOLERANCE}¢ f_est={f_used:.1f}"
                )
            return False

        if tracker_f0 is not None and abs(f_used - float(f_est)) > 0.5:
            est_source = f"{est_source}+tracker"
            _diag(
                "[DIAG][CommitDecision] fresh far but tracker OK (%.1f Hz) - accepting via tracker",
                f_used,
            )

        _diag(
            f"[DIAG][CommitDecision] ACCEPT | source={est_source} armed={armed} "
            f"captured≈{midi_used} reason={decision.reason}"
        )

        if fresh and live and est_source.startswith("fresh"):
            live_f = live.get("f_est") or live.get("f0")
            if live_f:
                diff_cents = 1200.0 * np.log2(f_used / max(float(live_f), 1))
                _diag(f"[DIAG][CommitDecision] fresh vs live diff = {diff_cents:+.1f}¢")

        if decision.switch_to_pc is not None:
            _diag(
                f"[DIAG][CommitDecision] Series SWITCHED to pitch class {decision.switch_to_pc} "
                f"based on captured note {midi_used} (was pc={prev_pc})"
            )
            import time as _t

            self._scale_gate_grace_until = _t.time() + 0.65

        return True

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

            # Quick energy check - if it's basically silence we shouldn't trust it
            rms = float(np.sqrt(np.mean(audio * audio)))
            if rms < 0.0008:  # very quiet, probably decay or noise
                return None

            est = self._estimate_pitch(audio, fs, self._initial_a4)
            return est
        except Exception as e:
            _diag(f"[DIAG][FreshEst] commit-time fresh analysis failed: {e}")
            return None

    def _on_compute_curve(self) -> None:
        """Run the selected solver (optionally on a worker thread) and apply the curve."""
        piano = self._ensure_piano()
        if not piano.has_measurements():
            QMessageBox.information(
                self,
                "Compute Curve",
                "No measurements yet - computing a default Railsback-style stretch curve (usable ET + stretch).\n\n"
                "For best results on your real piano, record 8-12 keys first (B values drive the fit).",
            )

        solver_name = "beat-rate"
        if hasattr(self, "_solver_combo") and self._solver_combo is not None:
            data = self._solver_combo.currentData()
            if data:
                solver_name = str(data)

        try:
            import numpy as np

            from optitune.solvers.base import MIDI_LOW, N_KEYS
            from optitune.solvers.worker import SolverWorker

            spectra = piano.cent_spectra_matrix().astype(float)
            b_est = np.full(N_KEYS, np.nan, dtype=float)
            for m, k in piano.keys.items():
                if k.measured_b is not None:
                    b_est[int(m) - MIDI_LOW] = float(k.measured_b)

            needs_spectra = solver_name in ("entropy", "octave-entropy")
            if needs_spectra and float(spectra.sum()) <= 0:
                QMessageBox.information(
                    self,
                    "Spectrum solver",
                    "No cent spectra stored yet. Record notes first "
                    "(spectra are captured automatically on Record), "
                    "or switch to beat-rate.",
                )
                return

            from optitune.model.temperaments import temperament_offsets_88

            temp_offs = None
            if getattr(self, "_temperament", "equal") not in (None, "equal", "et"):
                try:
                    temp_offs = temperament_offsets_88(self._temperament)
                except KeyError:
                    temp_offs = None
            constraints = TuningConstraints(
                a4=float(piano.a4),
                temperament=getattr(self, "_temperament", "equal"),
                temperament_offsets=temp_offs,
                interval_weights=getattr(self, "_interval_weights", {}) or {},
            )
            kwargs: dict = {}
            if solver_name == "entropy":
                kwargs = {"seed": 0, "max_passes": 12, "railsback_prior": 0.2}

            # Run worker synchronously via direct call for reliability in tests;
            # still uses SolverWorker API so intermediate progress can stream later.
            worker = SolverWorker()
            result: list = []

            def _on_done(tc: object) -> None:
                result.append(tc)

            worker.finished.connect(_on_done)
            worker.failed.connect(lambda msg: result.append(RuntimeError(msg)))
            worker.start_solve(solver_name, spectra, b_est, constraints, kwargs)

            if not result:
                # Fallback sync path
                if solver_name == "beat-rate":
                    curve = BeatRateSolver().solve_piano(piano).as_list()
                else:
                    solver = get_solver(solver_name, **kwargs)
                    curve = list(solver.solve(spectra, b_est, constraints))[-1].as_list()
            elif isinstance(result[0], Exception):
                raise result[0]
            elif result[0] is None:
                curve = compute_basic_tuning_curve(piano)
            else:
                curve = result[0].as_list()

            self._apply_tuning_curve(piano, curve, solver_name=solver_name)
        except Exception as exc:
            QMessageBox.warning(self, "Solver Error", f"Curve computation failed:\n{exc}")

    def _apply_tuning_curve(
        self, piano: Piano, curve: list[float], *, solver_name: str = "beat-rate"
    ) -> None:
        piano.tuning_curve = curve
        for k in piano.keys.values():
            k.target_offset_cents = piano.get_target_offset(k.midi)
        self._update_curve_status()
        self._refresh_curve_widgets()
        self.statusBar().showMessage(
            f"Curve computed ({solver_name}). Green ≈ on target, orange needs attention. "
            "Tune until the strobe stops and cents near 0.",
            8000,
        )
        self.hint.setText(
            "Curve active - play a note. The tuner now shows deviation from the computed per-key targets."
        )
        self._save_persisted_piano()
        self._mark_session_dirty()
        for m in [21, 33, 45, 57, 69, 81, 93, 105]:
            if abs(piano.get_target_offset(m)) > 4.0:
                self.keyboard.set_key_state(m, KeyState.NEEDS_ATTENTION)

    def is_session_dirty(self) -> bool:
        return bool(getattr(self, "_session_dirty", False))

    def _mark_session_dirty(self) -> None:
        self._session_dirty = True

    def _mark_session_clean(self) -> None:
        self._session_dirty = False

    def _refresh_curve_widgets(self) -> None:
        """Push piano curve + B measurements into Railsback / B-curve plots."""
        if not hasattr(self, "railsback") or not hasattr(self, "b_curve"):
            return
        piano = self._piano
        if piano is None:
            self.railsback.clear()
            self.b_curve.clear()
            return
        if piano.tuning_curve is not None:
            self.railsback.set_tuning_curve(piano.tuning_curve)
        # Measured cents vs ET: if we have f0, convert; else use stored target offset as proxy
        measured_dev: dict[int, float] = {}
        for m, k in piano.keys.items():
            if k.measured_f0 is not None and k.measured_f0 > 1:
                et = float(self._initial_a4) * (2.0 ** ((m - 69) / 12.0))
                measured_dev[m] = float(1200.0 * np.log2(float(k.measured_f0) / et))
            elif k.target_offset_cents:
                measured_dev[m] = float(k.target_offset_cents)
        self.railsback.set_measured_deviations(measured_dev)
        self.railsback.set_a4_marker(True)
        self.b_curve.set_measured_b(measured_b_from_piano(piano))

    @Slot()
    def _on_interval_weights(self) -> None:
        dlg = IntervalWeightsDialog(self, weights=self._interval_weights)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._interval_weights = dlg.weights()
            self.statusBar().showMessage(
                f"Interval weights updated (4:2={self._interval_weights.get('octave_4_2', 0):.1f}).",
                3000,
            )

    @Slot()
    def _on_pitch_raise(self) -> None:
        piano = self._ensure_piano()
        final = piano.tuning_curve
        if final is None:
            # Need a final target curve first
            try:
                final = BeatRateSolver().solve_piano(piano).as_list()
            except Exception:
                final = compute_basic_tuning_curve(piano)
        # Measured cents vs ET from stored f0
        measured = np.zeros(88)
        n_meas = 0
        for m, k in piano.keys.items():
            if k.measured_f0 is not None and k.measured_f0 > 1:
                et = float(piano.a4) * (2.0 ** ((m - 69) / 12.0))
                measured[m - 21] = float(1200.0 * np.log2(float(k.measured_f0) / et))
                n_meas += 1
        mean_flat = None
        if n_meas == 0:
            mean_flat = -30.0  # default assumption: ~30¢ flat
            measured_arg = None
        else:
            measured_arg = measured
        dlg = PitchRaiseDialog(
            self,
            final_curve=final,
            measured_dev=measured_arg,
            mean_flat_cents=mean_flat,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._apply_tuning_curve(piano, dlg.targets(), solver_name=f"pitch-raise/{dlg.variant()}")

    def _on_clear_measurements(self) -> None:
        piano = self._piano
        if piano is None:
            return
        piano.keys.clear()
        piano.tuning_curve = None
        self.keyboard.clear_all()
        self._record_selected_midi = None
        self._update_curve_status()
        self._refresh_curve_widgets()
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

    def _persist_scale_session_settings(self) -> None:
        """Write active series + arm target so a crash mid-session can resume."""
        self._app_settings.set_scale_session(
            active_pitch_class=self._scale_pitch_class,
            last_recorded_midi=self._last_recorded_midi,
            armed_midi=self._record_selected_midi,
        )

    def _clear_scale_session_settings(self) -> None:
        self._app_settings.clear_scale_session()

    def _restore_scale_session_settings(self) -> None:
        """If QSettings has an interrupted series, re-enter scale mode (not auto-armed)."""
        d = self._app_settings.get_scale_session()
        pc = d.get("active_pitch_class")
        if pc is None:
            return
        seed = next(m for m in range(21, 109) if m % 12 == int(pc))
        import time as _t

        self._scale_session.enter_scale(seed, now=_t.time())
        self._last_recorded_midi = d.get("last_recorded_midi")
        armed = d.get("armed_midi")
        if armed is not None:
            self._record_selected_midi = int(armed)
            self.keyboard.set_current_key(int(armed))
            self.keyboard.set_key_state(int(armed), KeyState.ARMED)
        self._update_series_status()
        armed_s = (
            f"{self._record_selected_midi} ({midi_to_note_name(self._record_selected_midi)})"
            if self._record_selected_midi is not None
            else "—"
        )
        self.statusBar().showMessage(
            f"Resumed series pc={pc}; target {armed_s}. Click Arm to continue.",
            4000,
        )

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
        """Start a fresh piano session (name, A4, temperament)."""
        dlg = NewPianoDialog(
            self,
            name="My Piano",
            a4=self._initial_a4,
            temperament=self._temperament,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._initial_a4 = dlg.a4()
        self._temperament = dlg.temperament()
        self._piano = Piano(a4=self._initial_a4, name=dlg.piano_name())
        self._record_selected_midi = None
        self._last_recorded_midi = None
        self._scale_session.exit_scale()
        self._clear_scale_session_settings()
        self.keyboard.clear_all()
        if hasattr(self, "_a4_label") and self._a4_label is not None:
            self._a4_label.setText(f"  A4 = {self._initial_a4:.1f} Hz")
        self._update_curve_status()
        self._refresh_curve_widgets()
        self.statusBar().showMessage(
            f"New piano “{dlg.piano_name()}” A4={self._initial_a4:.1f} "
            f"({self._temperament}). Begin recording keys.",
            4000,
        )
        self._current_pfg_path = None
        self._mark_session_clean()  # empty new session
        self._save_persisted_piano()

    @Slot()
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open tuning file",
            "",
            "OptiTune tuning (*.pfg);;EPT (*.ept);;JSON session (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            low = path.lower()
            if low.endswith(".pfg"):
                piano, meta = load_pfg(path)
                self._piano = piano
                self._initial_a4 = float(piano.a4)
                self._temperament = str(meta.get("temperament") or "equal")
                self._current_pfg_path = path
                self._app_settings.add_recent_file(path)
            elif low.endswith(".ept"):
                self._piano = load_ept(path)
                self._initial_a4 = float(self._piano.a4)
                self._current_pfg_path = None
                self._app_settings.add_recent_file(path)
            else:
                loaded = Piano.load_json(path)
                if loaded is None:
                    QMessageBox.warning(self, "Open", f"Could not load {path}")
                    return
                self._piano = loaded
                self._current_pfg_path = None
            self.keyboard.clear_all()
            assert self._piano is not None
            for m, k in self._piano.keys.items():
                if k.measured_b is not None or k.measured_f0 is not None:
                    self.keyboard.set_key_state(m, KeyState.MEASURED)
            if hasattr(self, "_a4_label") and self._a4_label is not None:
                self._a4_label.setText(f"  A4 = {self._initial_a4:.1f} Hz")
            self._update_curve_status()
            self._refresh_curve_widgets()
            self._mark_session_clean()
            self.statusBar().showMessage(f"Opened {path}", 4000)
        except Exception as exc:
            QMessageBox.warning(self, "Open", f"Failed to open:\n{exc}")

    @Slot()
    def _on_save(self) -> None:
        if self._piano is None:
            QMessageBox.information(self, "Save", "Nothing to save yet.")
            return
        if self._current_pfg_path:
            try:
                save_pfg(self._piano, self._current_pfg_path, temperament=self._temperament)
                self._save_persisted_piano()
                self._app_settings.add_recent_file(self._current_pfg_path)
                self._mark_session_clean()
                self.statusBar().showMessage(f"Saved {self._current_pfg_path}", 3000)
            except Exception as exc:
                QMessageBox.warning(self, "Save", f"Failed:\n{exc}")
            return
        self._on_save_as()

    @Slot()
    def _on_save_as(self) -> None:
        if self._piano is None:
            QMessageBox.information(self, "Save As", "Nothing to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save tuning file",
            self._current_pfg_path or "piano.pfg",
            "OptiTune tuning (*.pfg)",
        )
        if not path:
            return
        if not path.lower().endswith(".pfg"):
            path = path + ".pfg"
        try:
            save_pfg(self._piano, path, temperament=self._temperament)
            self._current_pfg_path = path
            self._save_persisted_piano()
            self._app_settings.add_recent_file(path)
            self._mark_session_clean()
            self.statusBar().showMessage(f"Saved {path}", 3000)
        except Exception as exc:
            QMessageBox.warning(self, "Save As", f"Failed:\n{exc}")

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
            © 2026 OptiTune Contributors - Licensed under the GNU GPL v3.<br>
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
        if self.is_session_dirty():
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Unsaved changes")
            box.setText("This piano session has unsaved changes.")
            box.setInformativeText(
                "Save a .pfg tuning file, discard changes, or cancel and keep working."
            )
            save_btn = box.addButton("Save…", QMessageBox.ButtonRole.AcceptRole)
            discard_btn = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
            cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(save_btn)
            box.exec()
            clicked = box.clickedButton()
            if clicked is cancel_btn:
                event.ignore()
                return
            if clicked is save_btn:
                self._on_save()
                if self.is_session_dirty():
                    # User cancelled Save As or save failed
                    event.ignore()
                    return
            # Discard: fall through and close
        with contextlib.suppress(Exception):
            if self._level_timer:
                self._level_timer.stop()
            if self._analysis_timer:
                self._analysis_timer.stop()
            self.audio_capture.stop()
            self._save_persisted_piano()
        event.accept()
        super().closeEvent(event)
