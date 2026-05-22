"""
Real piano recording test data.

This package contains recordings from an actual detuned piano and utilities
to load + analyze them. These are used to develop and test a robust live
pitch + inharmonicity estimator (Option B direction).

Current recordings:
- 7 C notes (lowest C to highest C played)
- 7 F notes (lowest F to highest F played)

See loader.py for usage.
"""

from .loader import (
    get_ground_truth_midi,
    list_recordings,
    load_recording,
)

__all__ = [
    "get_ground_truth_midi",
    "list_recordings",
    "load_recording",
]
