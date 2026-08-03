"""pytest-qt: rejection flash on keyboard decays after a short window."""

from __future__ import annotations

from optitune.ui.widgets.keyboard_widget import KeyboardWidget, KeyState


def test_flash_rejection_sets_state_then_clears(qtbot) -> None:
    kb = KeyboardWidget()
    qtbot.addWidget(kb)
    kb.set_key_state(60, KeyState.ARMED)

    assert kb.rejection_flash_midi is None
    kb.flash_rejection(60, duration_ms=80)
    assert kb.rejection_flash_midi == 60
    # ARMED base state preserved
    assert kb._states.get(60) == KeyState.ARMED

    # After the flash duration, singleShot clears it
    qtbot.wait(150)
    assert kb.rejection_flash_midi is None


def test_flash_rejection_other_key_does_not_clear_armed(qtbot) -> None:
    kb = KeyboardWidget()
    qtbot.addWidget(kb)
    kb.set_key_state(24, KeyState.ARMED)
    kb.flash_rejection(24, duration_ms=50)
    assert kb.rejection_flash_midi == 24
    kb.set_key_state(24, KeyState.ARMED)
    qtbot.wait(80)
    assert kb._states.get(24) == KeyState.ARMED
