"""Pitch-raise dialog produces overpull targets."""

from __future__ import annotations

import numpy as np

from optitune.solvers.base import N_KEYS
from optitune.ui.dialogs.pitch_raise import PitchRaiseDialog


def test_pitch_raise_dialog_targets_above_final(qtbot) -> None:
    final = np.linspace(-10, 8, N_KEYS)
    d = PitchRaiseDialog(final_curve=final, mean_flat_cents=-30.0)
    qtbot.addWidget(d)
    d._variant.setCurrentIndex(d._variant.findData("high"))
    d._update_preview()
    t = np.asarray(d.targets())
    assert t.shape == (N_KEYS,)
    assert float(np.mean(t - final)) > 0
    assert abs(t[69 - 21]) < 0.05
