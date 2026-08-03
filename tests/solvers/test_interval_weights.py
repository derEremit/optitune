"""Interval weight presets for beat-rate solver."""

from __future__ import annotations

import pytest

from optitune.solvers.interval_weights import (
    DEFAULT_INTERVAL_WEIGHTS,
    get_preset,
    list_presets,
    merge_weights,
)


def test_default_has_primary_octave():
    assert DEFAULT_INTERVAL_WEIGHTS["octave_4_2"] > DEFAULT_INTERVAL_WEIGHTS["octave_2_1"]


def test_clean_octaves_boosts_octaves():
    p = get_preset("clean_octaves")
    assert p["octave_4_2"] > DEFAULT_INTERVAL_WEIGHTS["octave_4_2"]
    assert p["fifth_3_2"] < DEFAULT_INTERVAL_WEIGHTS["fifth_3_2"]


def test_singing_twelfths_boosts_twelfth():
    p = get_preset("singing_twelfths")
    assert p["twelfth_3_1"] > DEFAULT_INTERVAL_WEIGHTS["twelfth_3_1"]


def test_merge_overrides():
    w = merge_weights(get_preset("default"), octave_4_2=50.0)
    assert w["octave_4_2"] == 50.0
    assert w["octave_2_1"] == DEFAULT_INTERVAL_WEIGHTS["octave_2_1"]


def test_unknown_preset():
    with pytest.raises(KeyError):
        get_preset("nope")
    assert "default" in list_presets()
