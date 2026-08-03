"""EPT fixture import."""

from __future__ import annotations

from pathlib import Path

import pytest

from optitune.persistence.ept_import import load_ept

FIXTURE = Path(__file__).parent / "fixtures" / "sample.ept"


def test_load_sample_ept():
    p = load_ept(FIXTURE)
    assert p.name == "Fixture Piano"
    assert p.a4 == pytest.approx(440.0)
    # number 39 → MIDI 60 (C4)
    k = p.get_key(60)
    assert k is not None
    assert k.measured_f0 == pytest.approx(261.63)
    assert k.measured_b == pytest.approx(0.0004)
    # number 48 → MIDI 69 (A4)
    a4k = p.get_key(69)
    assert a4k is not None
    assert a4k.measured_f0 == pytest.approx(440.0)
    # number 15 → MIDI 36
    assert p.get_key(36) is not None
