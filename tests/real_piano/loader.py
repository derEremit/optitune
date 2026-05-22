"""
Loader for real piano recordings.

Usage example:

    from tests.real_piano.loader import load_recording, list_recordings

    recordings = list_recordings()
    audio, sr, meta = load_recording("C4")

    print(meta)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

_RECORDINGS_DIR = Path(__file__).parent / "recordings"
_METADATA_PATH = Path(__file__).parent / "segments.json"


def list_recordings() -> list[str]:
    """Return list of available recording names (e.g. ['C1', 'C2', ..., 'F7'])."""
    if not _METADATA_PATH.exists():
        return []
    with open(_METADATA_PATH) as f:
        meta = json.load(f)
    return [item["filename"].replace(".wav", "") for item in meta]


def load_recording(name: str) -> tuple[np.ndarray, int, dict[str, Any]]:
    """
    Load a real piano recording.

    Args:
        name: e.g. "C4", "F2", "C1"

    Returns:
        audio (float32, mono, normalized -1..1),
        sample_rate,
        metadata dict
    """
    name = name.upper()
    wav_path = _RECORDINGS_DIR / f"{name}.wav"

    if not wav_path.exists():
        available = list_recordings()
        raise FileNotFoundError(
            f"Recording '{name}' not found. Available: {available}"
        )

    sr, data = wavfile.read(wav_path)

    # Convert to mono float32, normalized
    if len(data.shape) > 1:
        audio = data.mean(axis=1).astype(np.float32)
    else:
        audio = data.astype(np.float32)

    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs

    # Load corresponding metadata
    with open(_METADATA_PATH) as f:
        all_meta = json.load(f)

    meta = next(
        (m for m in all_meta if m["filename"] == f"{name}.wav"),
        {"filename": f"{name}.wav", "note_label": name}
    )

    return audio, sr, meta


def get_ground_truth_midi(name: str) -> int | None:
    """Return the expected MIDI note if we have manually corrected it."""
    # For now we return the approximate value from segments.json.
    # Later we can maintain a corrected version.
    with open(_METADATA_PATH) as f:
        all_meta = json.load(f)

    for m in all_meta:
        if m["filename"] == f"{name.upper()}.wav":
            return m.get("approx_midi")
    return None
