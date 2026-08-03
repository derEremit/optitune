"""Historical temperaments as cent offsets from ET (spec §6 / M3)."""

from __future__ import annotations

import pytest

from optitune.model.temperaments import (
    TEMPERAMENTS,
    list_temperaments,
    temperament_offsets_88,
    temperament_pitch_class_offsets,
)


def test_equal_is_zero():
    pc = temperament_pitch_class_offsets("equal")
    assert len(pc) == 12
    assert all(abs(x) < 1e-9 for x in pc)
    full = temperament_offsets_88("equal")
    assert len(full) == 88
    assert all(abs(x) < 1e-9 for x in full)


def test_werckmeister_iii_known_thirds():
    """
    Werckmeister III: C-E pure-ish major third is narrow vs ET.
    Pitch-class offsets (vs ET, C=0): published tables place E around -8¢,
    G# more, B flatish. We check relative structure, not absolute cents.
    """
    pc = temperament_pitch_class_offsets("werckmeister_iii")
    # C = 0 by convention
    assert pc[0] == pytest.approx(0.0, abs=0.01)
    # Major third C-E: E is flatter than ET (negative)
    assert pc[4] < -2.0
    # Fifth C-G: slightly flat of ET pure-fifth stretch, still near 0
    assert abs(pc[7]) < 6.0
    # Full 88 repeats every octave
    full = temperament_offsets_88("werckmeister_iii")
    assert full[60 - 21] == pytest.approx(pc[0], abs=0.01)  # C4
    assert full[64 - 21] == pytest.approx(pc[4], abs=0.01)  # E4


def test_list_includes_core_set():
    names = list_temperaments()
    for n in (
        "equal",
        "werckmeister_iii",
        "kirnberger_iii",
        "vallotti",
        "young",
        "meantone_quarter",
    ):
        assert n in names
        assert n in TEMPERAMENTS


def test_unknown_raises():
    with pytest.raises(KeyError):
        temperament_pitch_class_offsets("not-a-temperament")
