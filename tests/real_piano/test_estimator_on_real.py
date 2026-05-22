"""
Regression / diagnostic tests using real piano recordings.

These are the first automated tests that run against the user's actual detuned piano.

They are intentionally marked `real_piano` so they are skipped during normal CI
(`pytest -m "not real_piano"`). Run them locally when working on the pitch estimator:

    pytest -m real_piano -q tests/real_piano/

The data in `recordings/` + `segments.json` is the trusted output of
`tools/segment_real_recording.py --export-to-tests` (noise gate + known C-then-F sequence).

Current purpose:
- Establish a baseline of how badly (or well) the production estimator performs
  on real hammer-strike material from the target instrument.
- Make future improvements to the estimator produce measurable progress on this
  exact material.
"""

from __future__ import annotations

import numpy as np
import pytest

from optitune.dsp import (
    find_spectral_peaks,
    hz_to_midi,
    midi_to_note_name,
    pfd_estimate_f0_b,
)

from tests.real_piano.loader import list_recordings, load_recording


pytestmark = pytest.mark.real_piano


def _analyze_segment(audio: "np.ndarray", sr: int) -> tuple[float, float]:
    """Run the same core estimator the live app uses on the middle 60% of the note."""
    n = len(audio)
    start = int(n * 0.20)
    end = int(n * 0.80)
    seg = audio[start:end]

    if len(seg) < 2048:
        return 0.0, 0.0

    w = np.hanning(len(seg))
    spec = np.fft.rfft(seg * w)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)

    peak_fs, peak_as = find_spectral_peaks(
        freqs, power, min_prominence_db=12.0, max_peaks=20
    )

    if len(peak_fs) == 0:
        return 0.0, 0.0

    f0, B = pfd_estimate_f0_b(peak_fs, peak_as, f0_guess=200.0, max_n=16)
    return float(f0), float(B)


def test_real_piano_estimator_baseline():
    """
    Run the current production pitch + B estimator on every real recording
    and report error vs the known ground-truth MIDI (from the C/F sequence).

    This test always passes today; its value is the printed table.
    When you improve the estimator you should see the error numbers drop.
    """
    import numpy as np

    names = list_recordings()
    assert len(names) == 14, "Expected the 14 trusted C1-C7 + F1-F7 recordings"

    results = []

    for name in names:
        audio, sr, meta = load_recording(name)
        expected_midi = meta.get("approx_midi")
        if expected_midi is None:
            continue

        f0, B = _analyze_segment(audio, sr)
        if f0 < 20:
            detected_midi = None
            semitone_err = 99
        else:
            detected_midi = round(hz_to_midi(f0))
            semitone_err = abs(detected_midi - expected_midi)

        results.append(
            {
                "name": name,
                "expected": expected_midi,
                "detected": detected_midi,
                "err": semitone_err,
                "f0": round(f0, 1),
                "B": round(B, 6),
            }
        )

    # Pretty table
    print("\n" + "=" * 78)
    print("REAL PIANO ESTIMATOR BASELINE (current production code)")
    print("=" * 78)
    print(f"{'Note':<6} | {'Expected':>8} | {'Detected':>8} | {'Err':>4} | {'f0 (Hz)':>9} | {'B':>10}")
    print("-" * 78)

    for r in results:
        det = r["detected"] if r["detected"] is not None else "???"
        print(
            f"{r['name']:<6} | {r['expected']:>8} | {det:>8} | {r['err']:>4} | "
            f"{r['f0']:>9.1f} | {r['B']:>10.6f}"
        )

    errors = [r["err"] for r in results if r["err"] < 50]
    if errors:
        median_err = float(np.median(errors))
        mean_err = float(np.mean(errors))
        max_err = max(errors)
        print("-" * 78)
        print(f"Median semitone error : {median_err:.1f}")
        print(f"Mean   semitone error : {mean_err:.1f}")
        print(f"Max    semitone error : {max_err}")
    print("=" * 78 + "\n")

    # Soft informational "test" — we always pass so the suite stays green
    # while we work on making these numbers actually good.
    assert len(results) == 14
    # Future improvement target example (uncomment when the estimator is better):
    # assert median_err <= 2.0, f"Estimator still has {median_err:.1f} semitone median error on real piano"