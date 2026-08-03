"""Log-linear B-curve fit (model/inharmonicity)."""

from __future__ import annotations

import numpy as np
import pytest

from optitune.model.inharmonicity import fit_log_linear_b, measured_b_from_piano
from optitune.model import Key, Piano


def test_fit_log_linear_rises_with_midi():
    measured = {36: 1e-4, 48: 2e-4, 60: 5e-4, 72: 1.5e-3, 84: 5e-3}
    b_pred, slope, intercept = fit_log_linear_b(measured)
    assert len(b_pred) == 88
    assert slope > 0
    assert b_pred[80] > b_pred[10]


def test_fit_sparse_defaults():
    b_pred, slope, _ = fit_log_linear_b({60: 4e-4})
    assert len(b_pred) == 88
    assert slope == 0.0
    assert np.all(b_pred > 0)


def test_measured_b_from_piano():
    p = Piano()
    p.set_key(Key(midi=48, measured_b=2e-4, measured_f0=130.0))
    p.set_key(Key(midi=60, measured_b=None, measured_f0=261.0))
    d = measured_b_from_piano(p)
    assert d == {48: pytest.approx(2e-4)}
