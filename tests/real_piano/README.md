# Real Piano Recordings

This directory contains **trusted** real recordings from the target piano.

## Current Data (as of 2026-05)

- **Source**: `testmaterial/c and f.wav` (user's detuned acoustic piano)
- First pass: C1–C7 ascending
- Second pass: F1–F7 ascending
- 14 segments extracted with a simple noise gate + the known recording order
- Ground-truth MIDI labels are now reliable (derived from experimental design, not from the estimator)

## Regenerating / Updating the Fixtures

If you improve the segmentation logic, run:

```bash
python tools/segment_real_recording.py --export-to-tests
```

This copies the new clean WAVs into `recordings/` and writes a correct `segments.json`.

## Running the Automated Tests

```bash
# Normal test run (these are skipped)
pytest -q

# Run the real-piano regression baseline (local only)
pytest -m real_piano -q tests/real_piano/
```

The test `test_estimator_on_real.py` runs the current production estimator on all 14 notes and prints a clear error table (semitone error vs. the known C/F sequence). This is the living benchmark for Option B work.

## Usage from Python

```python
from tests.real_piano.loader import load_recording, list_recordings

print(list_recordings())
# ['C1', 'C2', ..., 'F7']

audio, sr, meta = load_recording("C4")
print(meta)
# {'filename': 'C4.wav', 'note_label': 'C4', 'approx_midi': 60, 'pass': 'C', ...}
```

## Purpose

These recordings are the primary real-world material for improving the live pitch + inharmonicity estimator. The goal is to drive the median semitone error on this exact set down over time.
