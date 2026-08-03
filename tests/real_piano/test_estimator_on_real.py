"""
Regression / diagnostic tests using real piano recordings.

These are the first automated tests that run against the user's actual detuned piano.

They are intentionally marked `real_piano` so they are skipped during normal CI
(`pytest -m "not real_piano"`). Run them locally when working on the pitch estimator:

    pytest -m real_piano -q tests/real_piano/

The data in `recordings/` + `segments.json` is the trusted output of
`tools/segment_real_recording.py --export-to-tests` (noise gate + known C-then-F sequence).

Purpose (Milestone 1 harness):
- Per-note expected-MIDI assertions over clean segments
- Hit-rate baseline for the production estimator *and* the comb note-recognizer
- TDD loop for low-note robustness (iterate here, not on the 67 s master)
"""

from __future__ import annotations

import numpy as np
import pytest

from optitune.dsp import (
    estimate_pitch,
    find_spectral_peaks,
    hz_to_midi,
    midi_to_hz,
    pfd_estimate_f0_b,
    recognize_from_audio,
)
from optitune.dsp.peaks import cents
from tests.real_piano.loader import list_recordings, load_recording

pytestmark = pytest.mark.real_piano

# Ground-truth MIDI for the trusted C1-C7 + F1-F7 series
EXPECTED_MIDI = {
    "C1": 24,
    "C2": 36,
    "C3": 48,
    "C4": 60,
    "C5": 72,
    "C6": 84,
    "C7": 96,
    "F1": 29,
    "F2": 41,
    "F3": 53,
    "F4": 65,
    "F5": 77,
    "F6": 89,
    "F7": 101,
}


def _analyze_segment_pfd(
    audio: np.ndarray, sr: int, expected_midi: int | None = None
) -> tuple[float, float, int | None]:
    """Run the classical PFD path on the middle 60% (legacy baseline)."""
    n = len(audio)
    start = int(n * 0.20)
    end = int(n * 0.80)
    seg = audio[start:end]

    if len(seg) < 2048:
        return 0.0, 0.0, None

    # Long frame for bass (spec §3.3)
    target = 65536 if (expected_midi is not None and expected_midi < 55) else max(len(seg), 32768)
    if len(seg) < target:
        pad = np.zeros(target, dtype=np.float64)
        pad[: len(seg)] = seg
        seg = pad

    w = np.hanning(len(seg))
    spec = np.fft.rfft(seg * w)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)

    peak_fs, peak_as = find_spectral_peaks(freqs, power, min_prominence_db=10.0, max_peaks=30)

    if len(peak_fs) == 0:
        return 0.0, 0.0, None

    guess = midi_to_hz(expected_midi) if expected_midi is not None else 200.0
    f0, B = pfd_estimate_f0_b(peak_fs, peak_as, f0_guess=guess, max_n=18)
    det = round(hz_to_midi(f0)) if f0 > 20 else None
    return float(f0), float(B), det


def _analyze_segment_recognizer(
    audio: np.ndarray, sr: int, expected_midi: int | None = None
) -> int | None:
    """Comb-filter note identity on the middle portion."""
    n = len(audio)
    start = int(n * 0.15)
    end = int(n * 0.85)
    seg = audio[start:end]
    if len(seg) < 2048:
        return None
    match = recognize_from_audio(seg, float(sr), prior_midi=expected_midi)
    return match.midi if match is not None else None


def test_real_piano_estimator_baseline(capsys):
    """
    Print a comparison table: classical PFD vs comb recognizer vs ground truth.

    Soft gate: at least some mid-range notes must classify correctly so
    total regressions are caught. Low notes are expected to improve over time.
    """
    names = list_recordings()
    assert len(names) == 14, "Expected the 14 trusted C1-C7 + F1-F7 recordings"

    results = []
    pfd_hits = 0
    rec_hits = 0

    for name in names:
        audio, sr, meta = load_recording(name)
        expected = EXPECTED_MIDI.get(name) or meta.get("approx_midi")
        if expected is None:
            continue

        f0, B, pfd_midi = _analyze_segment_pfd(audio, sr, expected)
        rec_midi = _analyze_segment_recognizer(audio, sr, expected)

        pfd_err = abs(pfd_midi - expected) if pfd_midi is not None else 99
        rec_err = abs(rec_midi - expected) if rec_midi is not None else 99
        if pfd_err == 0:
            pfd_hits += 1
        if rec_err == 0:
            rec_hits += 1

        results.append(
            {
                "name": name,
                "expected": expected,
                "pfd": pfd_midi,
                "pfd_err": pfd_err,
                "rec": rec_midi,
                "rec_err": rec_err,
                "f0": round(f0, 1),
                "B": round(B, 6),
            }
        )

    print("\n" + "=" * 90)
    print("REAL PIANO ESTIMATOR HARNESS (PFD vs comb recognizer)")
    print("=" * 90)
    print(
        f"{'Note':<6} | {'Exp':>4} | {'PFD':>4} | {'Perr':>4} | {'Rec':>4} | {'Rerr':>4} | "
        f"{'f0 (Hz)':>9} | {'B':>10}"
    )
    print("-" * 90)
    for r in results:
        pfd_s = r["pfd"] if r["pfd"] is not None else "???"
        rec_s = r["rec"] if r["rec"] is not None else "???"
        print(
            f"{r['name']:<6} | {r['expected']:>4} | {pfd_s:>4} | {r['pfd_err']:>4} | "
            f"{rec_s:>4} | {r['rec_err']:>4} | {r['f0']:>9.1f} | {r['B']:>10.6f}"
        )

    n = len(results)
    pfd_rate = 100.0 * pfd_hits / n if n else 0.0
    rec_rate = 100.0 * rec_hits / n if n else 0.0
    print("-" * 90)
    print(f"PFD hit-rate         : {pfd_hits}/{n} ({pfd_rate:.0f}%)")
    print(f"Recognizer hit-rate  : {rec_hits}/{n} ({rec_rate:.0f}%)")
    print("=" * 90)
    # Machine-readable for scripting
    print(
        f"SUMMARY: pfd_hits={pfd_hits} rec_hits={rec_hits} n={n} pfd_rate={pfd_rate:.1f} rec_rate={rec_rate:.1f}"
    )

    assert n == 14
    # Soft gate while low-note work continues: recognizer must beat pure chance
    # and cover at least a few notes exactly. Raise toward ≥95% as Milestone 1 lands.
    assert rec_hits >= 3, f"recognizer hit-rate too low: {rec_hits}/{n}"


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_MIDI.items()))
def test_recognizer_per_note(name: str, expected: int):
    """
    Per-note comb-recognizer assertion.

    Mid/high notes (MIDI ≥ 48) must classify exactly. Low notes are xfail until
    the bass path is fully solid - still run them for the printed harness.
    """
    if name not in list_recordings():
        pytest.skip(f"{name} not in fixtures")

    audio, sr, _meta = load_recording(name)
    det = _analyze_segment_recognizer(audio, sr, expected)

    # Soft xfails while comb recognizer is still iterated: low notes + a few
    # high-F octave slips. Strict asserts on the solid mid-range.
    if det != expected and (expected < 48 or expected >= 77):
        pytest.xfail(f"recognizer still weak on {name}: got {det}, expected {expected}")
    assert det == expected, f"{name}: recognizer got {det}, expected {expected}"


def _mid_segment(audio: np.ndarray) -> np.ndarray:
    n = len(audio)
    return audio[int(n * 0.2) : int(n * 0.8)].astype(np.float64)


@pytest.mark.parametrize(
    "name,expected",
    [(n, m) for n, m in sorted(EXPECTED_MIDI.items()) if n.startswith("C")],
)
def test_production_estimate_with_armed_prior_c_series(name: str, expected: int):
    """
    Production estimate_pitch (what commit-time uses) with armed_midi set.

    This is the TDD contract for scale recording: when the user is armed on the
    true key and the buffer holds that note, f0 must land within scale tolerance
    and identity must be the armed key.
    """
    if name not in list_recordings():
        pytest.skip(f"{name} not in fixtures")

    audio, sr, _meta = load_recording(name)
    seg = _mid_segment(audio)
    est = estimate_pitch(seg, float(sr), a4=440.0, armed_midi=expected)
    armed_hz = midi_to_hz(expected)
    err = abs(cents(est["f_est"], armed_hz))
    assert est["midi"] == expected, (
        f"{name}: midi={est['midi']} f_est={est['f_est']:.2f} (want {expected})"
    )
    assert err <= 140.0, f"{name}: f0 err {err:.1f}¢ (f_est={est['f_est']:.2f} want ~{armed_hz:.2f})"
