"""
Historical temperaments as pitch-class cent offsets from equal temperament.

Values are conventional table approximations used by piano-tuning software
(not bit-exact historical manuscripts). C (pc 0) is always 0. Applied to all
octaves via `temperament_offsets_88`.
"""

from __future__ import annotations

from typing import Mapping

# Pitch-class offsets (C, C#, D, … B) in cents vs ET.
# Sources: common tuner-table approximations (EPT / piano literature).
TEMPERAMENTS: dict[str, tuple[float, ...]] = {
    "equal": (0.0,) * 12,
    # Werckmeister III (well temperament)
    "werckmeister_iii": (
        0.0,
        -10.0,
        -8.0,
        -6.0,
        -10.0,
        -2.0,
        -12.0,
        -4.0,
        -8.0,
        -12.0,
        -4.0,
        -8.0,
    ),
    # Kirnberger III
    "kirnberger_iii": (
        0.0,
        -10.0,
        -7.0,
        -6.0,
        -14.0,
        -2.0,
        -10.0,
        -3.0,
        -8.0,
        -11.0,
        -4.0,
        -8.0,
    ),
    # Vallotti (circulating)
    "vallotti": (
        0.0,
        -6.0,
        -4.0,
        -2.0,
        -8.0,
        -2.0,
        -8.0,
        -2.0,
        -4.0,
        -6.0,
        0.0,
        -4.0,
    ),
    # Thomas Young (1799 style table)
    "young": (
        0.0,
        -6.0,
        -4.0,
        -2.0,
        -8.0,
        -2.0,
        -10.0,
        -2.0,
        -4.0,
        -8.0,
        0.0,
        -4.0,
    ),
    # 1/4-comma meantone (rough; wolf near G#–E♭)
    "meantone_quarter": (
        0.0,
        -24.0,
        -7.0,
        -31.0,
        -14.0,
        +3.0,
        -20.0,
        -3.0,
        -27.0,
        -10.0,
        -34.0,
        -17.0,
    ),
}

# Friendly display labels
TEMPERAMENT_LABELS: Mapping[str, str] = {
    "equal": "Equal temperament",
    "werckmeister_iii": "Werckmeister III",
    "kirnberger_iii": "Kirnberger III",
    "vallotti": "Vallotti",
    "young": "Young",
    "meantone_quarter": "¼-comma meantone",
}


def list_temperaments() -> list[str]:
    return list(TEMPERAMENTS.keys())


def temperament_pitch_class_offsets(name: str) -> tuple[float, ...]:
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    # aliases
    aliases = {
        "et": "equal",
        "equal_temperament": "equal",
        "werckmeister": "werckmeister_iii",
        "wm3": "werckmeister_iii",
        "kirnberger": "kirnberger_iii",
        "meantone": "meantone_quarter",
        "quarter_comma_meantone": "meantone_quarter",
    }
    key = aliases.get(key, key)
    if key not in TEMPERAMENTS:
        raise KeyError(f"Unknown temperament {name!r}; choose from {list_temperaments()}")
    return TEMPERAMENTS[key]


def temperament_offsets_88(name: str) -> list[float]:
    """88-length cent offsets for MIDI 21..108 (same class each octave)."""
    pc = temperament_pitch_class_offsets(name)
    return [float(pc[m % 12]) for m in range(21, 109)]
