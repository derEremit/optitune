"""Multi-partial strobe rings."""

from __future__ import annotations

from optitune.ui.widgets.strobe_widget import StrobeWidget


def test_strobe_partial_deltas_default_one_ring(qtbot) -> None:
    w = StrobeWidget()
    qtbot.addWidget(w)
    w.set_phase_delta_hz(0.5)
    assert w.partial_count() == 1
    w.set_multi_partial_enabled(True)
    w.set_partial_deltas([(1, 0.1), (2, -0.2), (3, 0.05)])
    assert w.partial_count() == 3
    w.set_multi_partial_enabled(False)
    assert w.partial_count() == 1  # falls back to primary delta only


def test_strobe_reset_clears_partials(qtbot) -> None:
    w = StrobeWidget()
    qtbot.addWidget(w)
    w.set_multi_partial_enabled(True)
    w.set_partial_deltas([(1, 1.0), (2, 2.0)])
    w.reset()
    assert w.partial_count() == 1
    assert w._phase_delta_hz == 0.0
