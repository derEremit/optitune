"""
Default interval weights and named presets for the beat-rate solver.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

# Production defaults (must match beat_rate.compute_basic_tuning_curve)
DEFAULT_INTERVAL_WEIGHTS: dict[str, float] = {
    "octave_2_1": 5.5,
    "octave_4_2": 28.0,
    "octave_6_3": 8.5,
    "octave_8_4": 2.2,
    "fifth_3_2": 0.6,
    "fourth_4_3": 0.2,
    "twelfth_3_1": 0.9,
    "double_oct_4_1": 0.4,
}

WEIGHT_LABELS: dict[str, str] = {
    "octave_2_1": "Octave 2:1",
    "octave_4_2": "Octave 4:2 (primary)",
    "octave_6_3": "Octave 6:3",
    "octave_8_4": "Octave 8:4",
    "fifth_3_2": "Fifth 3:2",
    "fourth_4_3": "Fourth 4:3",
    "twelfth_3_1": "Twelfth 3:1",
    "double_oct_4_1": "Double octave 4:1",
}

PRESETS: dict[str, dict[str, float]] = {
    "default": deepcopy(DEFAULT_INTERVAL_WEIGHTS),
    "clean_octaves": {
        "octave_2_1": 12.0,
        "octave_4_2": 40.0,
        "octave_6_3": 14.0,
        "octave_8_4": 4.0,
        "fifth_3_2": 0.2,
        "fourth_4_3": 0.05,
        "twelfth_3_1": 0.4,
        "double_oct_4_1": 0.8,
    },
    "singing_twelfths": {
        "octave_2_1": 4.0,
        "octave_4_2": 18.0,
        "octave_6_3": 6.0,
        "octave_8_4": 1.5,
        "fifth_3_2": 1.2,
        "fourth_4_3": 0.3,
        "twelfth_3_1": 8.0,
        "double_oct_4_1": 0.5,
    },
}

PRESET_LABELS: dict[str, str] = {
    "default": "Default (balanced)",
    "clean_octaves": "Clean octaves",
    "singing_twelfths": "Singing twelfths",
}


def list_presets() -> list[str]:
    return list(PRESETS.keys())


def get_preset(name: str) -> dict[str, float]:
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "balanced": "default",
        "octaves": "clean_octaves",
        "clean": "clean_octaves",
        "twelfths": "singing_twelfths",
        "singing": "singing_twelfths",
    }
    key = aliases.get(key, key)
    if key not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; choose from {list_presets()}")
    return deepcopy(PRESETS[key])


def merge_weights(base: Mapping[str, float] | None = None, **overrides: float) -> dict[str, float]:
    out = deepcopy(DEFAULT_INTERVAL_WEIGHTS)
    if base:
        out.update({k: float(v) for k, v in base.items()})
    out.update({k: float(v) for k, v in overrides.items()})
    return out
