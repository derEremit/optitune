"""New Piano dialog collects name, A4, temperament."""

from __future__ import annotations

from optitune.ui.dialogs.new_piano import NewPianoDialog


def test_new_piano_dialog_defaults(qtbot) -> None:
    d = NewPianoDialog()
    qtbot.addWidget(d)
    assert d.piano_name() == "My Piano"
    assert d.a4() == 440.0
    assert d.temperament() == "equal"


def test_new_piano_dialog_values(qtbot) -> None:
    d = NewPianoDialog(name="Steinway", a4=442.0, temperament="werckmeister_iii")
    qtbot.addWidget(d)
    assert d.piano_name() == "Steinway"
    assert d.a4() == 442.0
    assert d.temperament() == "werckmeister_iii"
