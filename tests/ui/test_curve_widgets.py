"""pytest-qt: Railsback + B-curve widgets accept data and update plots."""

from __future__ import annotations

import numpy as np

from optitune.ui.widgets.b_curve_widget import BCurveWidget
from optitune.ui.widgets.railsback_widget import RailsbackWidget


def test_railsback_set_curve_and_measured(qtbot) -> None:
    w = RailsbackWidget()
    qtbot.addWidget(w)
    curve = np.linspace(-12.0, 10.0, 88)
    curve[48] = 0.0  # A4
    w.set_tuning_curve(curve)
    w.set_measured_deviations({36: -8.0, 60: 1.5, 84: 6.0})
    w.set_a4_marker(True)
    assert w.curve_points() is not None
    assert len(w.curve_points()) == 88
    assert w.measured_count() == 3


def test_railsback_clear(qtbot) -> None:
    w = RailsbackWidget()
    qtbot.addWidget(w)
    w.set_tuning_curve(np.zeros(88))
    w.clear()
    assert w.curve_points() is None
    assert w.measured_count() == 0


def test_b_curve_set_measured_and_fit(qtbot) -> None:
    w = BCurveWidget()
    qtbot.addWidget(w)
    measured = {36: 1.2e-4, 48: 2.5e-4, 60: 5e-4, 72: 1.2e-3, 84: 4e-3}
    w.set_measured_b(measured)
    assert w.measured_count() == 5
    fit = w.fitted_b()
    assert fit is not None
    assert len(fit) == 88
    # Fitted B should rise toward treble
    assert fit[80] > fit[10]
