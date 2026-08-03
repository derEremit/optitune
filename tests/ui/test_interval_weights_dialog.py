"""Interval weights dialog presets and export."""

from __future__ import annotations

from optitune.solvers.interval_weights import DEFAULT_INTERVAL_WEIGHTS, get_preset
from optitune.ui.dialogs.interval_weights import IntervalWeightsDialog


def test_dialog_returns_default(qtbot) -> None:
    d = IntervalWeightsDialog()
    qtbot.addWidget(d)
    w = d.weights()
    assert w["octave_4_2"] == DEFAULT_INTERVAL_WEIGHTS["octave_4_2"]


def test_dialog_apply_preset(qtbot) -> None:
    d = IntervalWeightsDialog()
    qtbot.addWidget(d)
    idx = d._preset.findData("clean_octaves")
    d._preset.setCurrentIndex(idx)
    d._apply_preset()
    assert d.weights()["octave_4_2"] == get_preset("clean_octaves")["octave_4_2"]
