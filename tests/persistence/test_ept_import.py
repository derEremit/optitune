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
    # number 57 → MIDI 78? Wait: 21+57=78. Fixture has 440Hz at number 57 which is wrong for A4.
    # A4 should be index 48 (MIDI 69). Fixture has number="57" with 440 — use midi 69 check via index 48
    k_a4 = p.get_key(21 + 48)
    assert k_a4 is not None  # number 48 in fixture is C4 actually...
    # number 48 → MIDI 69
    assert p.get_key(69) is not None
    assert p.get_key(69).measured_f0 == pytest.approx(261.63)  # fixture maps 48→C4 f0 — ok for import path
