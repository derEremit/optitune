"""
KeyboardWidget — 88-key interactive piano keyboard (Phase 3: functional highlight).

Renders a compact visual of the middle keyboard range with proper white/black key graphics.
Highlights the currently detected / locked note (bright outline + fill).
Clicking a key emits the MIDI number (for future locking).
"""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class KeyState(Enum):
    UNMEASURED = "unmeasured"
    MEASURED = "measured"  # blue (white), bright cyan accent (black)
    IN_TUNE = "in_tune"
    NEEDS_ATTENTION = "needs_attention"
    ARMED = "armed"  # bright red while waiting for auto-record
    RECORDING = "recording"  # strong red while actually capturing


# MIDI range we actually draw (compact 3+ octaves for visibility on screen)
KEYBOARD_LOW = 48  # C3
KEYBOARD_HIGH = 84  # C6 (inclusive)


def midi_to_key_color(midi: int) -> tuple[bool, int]:
    """Return (is_black, position_in_white_keys) for layout calc."""
    pc = midi % 12
    is_black = pc in (1, 3, 6, 8, 10)
    # white key index within octave (0-6)
    white_pos = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}[pc] if not is_black else -1
    return is_black, white_pos


class KeyboardWidget(QWidget):
    """
    Interactive 88-key keyboard (MIDI 21-108). We render a usable central span.

    Public API (stable):
        set_key_state(midi: int, state: KeyState)
        set_current_key(midi: int | None)
        highlight_detected(midi: int)
        clear_all()
    """

    key_clicked = Signal(int)
    key_right_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(92)
        self.setMaximumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._states: dict[int, KeyState] = {}
        self._current: int | None = None
        self._detected: int | None = None
        self._rejection_flash_midi: int | None = None
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_rejection_flash)

        # Precompute which MIDI we draw
        self._drawn_midis: list[int] = list(range(KEYBOARD_LOW, KEYBOARD_HIGH + 1))

        self.setToolTip(
            "Piano keyboard - highlighted key shows live detected note (yellow). Click to select (future lock)."
        )

        # For hit testing
        self._key_rects: dict[int, QRectF] = {}  # midi -> rect (white or black)

    @property
    def rejection_flash_midi(self) -> int | None:
        return self._rejection_flash_midi

    def flash_rejection(self, midi: int, duration_ms: int = 400) -> None:
        """Brief bright flash on a key (does not replace ARMED/MEASURED state)."""
        self._rejection_flash_midi = int(midi)
        self._flash_timer.stop()
        self._flash_timer.start(max(50, int(duration_ms)))
        self.update()

    def _clear_rejection_flash(self) -> None:
        self._rejection_flash_midi = None
        self.update()

    def set_key_state(self, midi: int, state: KeyState) -> None:
        self._states[midi] = state
        self.update()

    def set_current_key(self, midi: int | None) -> None:
        self._current = midi
        self.update()

    def highlight_detected(self, midi: int) -> None:
        """Brief visual emphasis for auto-detected note (we just set as current)."""
        self._detected = int(midi)
        self.set_current_key(midi)
        self.update()

    def clear_all(self) -> None:
        self._states.clear()
        self._current = None
        self._detected = None
        self._rejection_flash_midi = None
        self._flash_timer.stop()
        self.update()

    # ---------------- Painting (real keyboard look) ----------------

    def _is_black(self, midi: int) -> bool:
        pc = midi % 12
        return pc in (1, 3, 6, 8, 10)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(4, 4, -4, -4)
        w, h = rect.width(), rect.height()

        # Background
        painter.fillRect(rect, QColor(25, 25, 30))

        # How many white keys in our range?
        white_keys = [m for m in self._drawn_midis if not self._is_black(m)]
        n_white = len(white_keys)
        if n_white == 0:
            return

        white_w = w / n_white
        black_w = white_w * 0.58
        black_h = h * 0.62

        self._key_rects.clear()

        # First draw all white keys
        white_idx = 0
        for midi in self._drawn_midis:
            if self._is_black(midi):
                continue
            x = rect.x() + white_idx * white_w
            key_rect = QRectF(x, rect.y(), white_w - 1, h)
            self._key_rects[midi] = key_rect

            # State color or base
            state = self._states.get(midi)
            flashing = midi == self._rejection_flash_midi
            if flashing:
                fill = QColor(255, 40, 90)  # hot pink-red rejection flash
                border = QColor(255, 200, 220)
            elif midi == self._current:
                fill = QColor(255, 220, 80)  # bright yellow/gold for current/detected
                border = QColor(255, 240, 150)
            elif state == KeyState.IN_TUNE:
                fill = QColor(80, 180, 110)
                border = QColor(110, 210, 140)
            elif state == KeyState.MEASURED:
                fill = QColor(90, 140, 200)
                border = QColor(110, 160, 220)
            elif state == KeyState.NEEDS_ATTENTION:
                fill = QColor(220, 140, 70)
                border = QColor(240, 160, 90)
            elif state == KeyState.RECORDING:
                fill = QColor(220, 50, 50)
                border = QColor(255, 80, 80)
            elif state == KeyState.ARMED:
                fill = QColor(200, 60, 60)
                border = QColor(255, 100, 100)
            else:
                fill = QColor(250, 250, 255)
                border = QColor(60, 60, 70)

            painter.setPen(QPen(border, 1))
            painter.setBrush(QBrush(fill))
            painter.drawRect(key_rect)

            # Small C label every octave
            if midi % 12 == 0:
                painter.setPen(QColor(40, 40, 50))
                painter.setFont(QFont("DejaVu Sans", 7))
                painter.drawText(int(x + 2), int(rect.y() + h - 4), f"C{midi // 12 - 1}")

            white_idx += 1

        # Then black keys (overlay)
        for midi in self._drawn_midis:
            if not self._is_black(midi):
                continue

            # Find the white key to the left for positioning
            # Simple: use the previous white's x
            # Compute position: blacks sit between whites
            # We approximate by scanning drawn whites
            left_white = midi - 1 if (midi - 1) in self._key_rects else midi - 2
            if left_white not in self._key_rects:
                # fallback compute
                continue

            base_rect = self._key_rects[left_white]
            # black x is ~ 0.65 of white width from left of its left white? standard layout
            # Better heuristic: blacks are centered between two whites
            x = base_rect.x() + base_rect.width() * 0.72
            # for the 'group of 2' vs '3' the offset varies slightly, but good enough
            key_rect = QRectF(x, rect.y(), black_w, black_h)
            self._key_rects[midi] = key_rect

            state = self._states.get(midi)
            flashing = midi == self._rejection_flash_midi
            if flashing:
                fill = QColor(220, 30, 70)
                border = QColor(255, 180, 200)
            elif midi == self._current:
                fill = QColor(255, 200, 50)
                border = QColor(255, 230, 120)
            elif state == KeyState.IN_TUNE:
                fill = QColor(60, 150, 90)
                border = QColor(90, 180, 120)
            elif state == KeyState.MEASURED:
                # Make recorded state visible on black keys
                fill = QColor(45, 55, 75)
                border = QColor(80, 200, 255)  # bright cyan
            elif state == KeyState.RECORDING:
                fill = QColor(180, 40, 40)
                border = QColor(255, 90, 90)
            elif state == KeyState.ARMED:
                fill = QColor(140, 45, 45)
                border = QColor(255, 110, 110)  # strong red border for visibility on black
            else:
                fill = QColor(35, 35, 42)
                border = QColor(20, 20, 25)

            painter.setPen(QPen(border, 1))
            painter.setBrush(QBrush(fill))
            painter.drawRect(key_rect)

            # Extra bright indicator at bottom of black keys for recorded / armed states
            if state == KeyState.MEASURED:
                ind = QRectF(key_rect.x() + 2, key_rect.bottom() - 6, key_rect.width() - 4, 4)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(90, 230, 255)))
                painter.drawRect(ind)
            elif state == KeyState.ARMED:
                ind = QRectF(key_rect.x() + 2, key_rect.bottom() - 6, key_rect.width() - 4, 4)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(255, 100, 100)))
                painter.drawRect(ind)
            elif state == KeyState.RECORDING:
                ind = QRectF(key_rect.x() + 2, key_rect.bottom() - 6, key_rect.width() - 4, 4)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(255, 80, 80)))
                painter.drawRect(ind)

        # Bottom label bar
        painter.setPen(QColor(140, 140, 150))
        painter.setFont(QFont("DejaVu Sans Mono", 8))
        cur_txt = ""
        if self._current is not None:
            from optitune.dsp.synth import midi_to_note_name  # local import ok

            cur_txt = f"  • {midi_to_note_name(self._current)} ({self._current})"
        painter.drawText(
            rect.adjusted(2, int(h - 14), 0, 0),
            Qt.AlignmentFlag.AlignLeft,
            f"Keys: Blue=measured, Green=good, Orange=needs attention, Red=recording{cur_txt}",
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            midi = self._midi_at_pos(event.position().x(), event.position().y())
            if midi is not None:
                self.key_clicked.emit(midi)
                self.set_current_key(midi)
        elif event.button() == Qt.MouseButton.RightButton:
            midi = self._midi_at_pos(event.position().x(), event.position().y())
            if midi is not None:
                self.key_right_clicked.emit(midi)
        super().mousePressEvent(event)

    def _midi_at_pos(self, px: float, py: float) -> int | None:
        """Hit test — prefer black keys (they are drawn on top)."""
        # Check blacks first
        for midi in self._drawn_midis:
            if midi not in self._key_rects:
                continue
            if self._is_black(midi) and self._key_rects[midi].contains(px, py):
                return midi
        # Then whites
        for midi in self._drawn_midis:
            if midi not in self._key_rects:
                continue
            if not self._is_black(midi) and self._key_rects[midi].contains(px, py):
                return midi
        return None
