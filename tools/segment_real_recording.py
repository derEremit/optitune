#!/usr/bin/env python3
"""
Standalone script to segment the real piano test recording (C's then F's).

This is the canonical way to turn the master recording into trustworthy per-note WAVs.

Usage:
    # Just analyze + produce recordings_clean/ (for inspection)
    python tools/segment_real_recording.py

    # Export the good segments directly into the automated test fixtures
    python tools/segment_real_recording.py --export-to-tests

The export path is what you run after any improvement to the noise-gate logic.
It produces exactly what the pytest real_piano tests and analyze.py expect.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# ---------------- Configuration ----------------
INPUT_FILE = Path("testmaterial/c and f.wav")
OUTPUT_DIR = Path("testmaterial/recordings_clean")
NOISE_FLOOR_DB = -50.0  # Noise gate threshold (a bit lower to catch soft high notes)
MIN_NOTE_DURATION = 0.35  # seconds - relaxed for short high notes
MIN_GAP = 0.18  # minimum silence between notes
HOP_SIZE = 512
FRAME_SIZE = 1024

# Expected structure: first N notes are C's, next M are F's
EXPECTED_C_COUNT = 7
EXPECTED_F_COUNT = 7


def db_to_linear(db):
    return 10 ** (db / 20.0)


def linear_to_db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def compute_rms_envelope(signal, sr, hop_size, frame_size):
    """Compute RMS energy envelope."""
    rms = []
    for i in range(0, len(signal) - frame_size, hop_size):
        frame = signal[i : i + frame_size]
        rms.append(np.sqrt(np.mean(frame**2)))
    return np.array(rms)


def find_note_boundaries(rms, sr, hop_size, noise_floor, min_duration, min_gap):
    """Find start and end indices of notes based on energy above noise floor."""
    above = rms > noise_floor
    changes = np.diff(above.astype(int))

    onsets = np.where(changes == 1)[0]
    offsets = np.where(changes == -1)[0]

    # Handle edge cases
    if len(onsets) == 0:
        return []

    if len(offsets) < len(onsets):
        offsets = np.append(offsets, len(rms) - 1)

    notes = []
    for on, off in zip(onsets, offsets, strict=False):
        duration = (off - on) * hop_size / sr
        if duration < min_duration:
            continue

        start_sample = int(on * hop_size)
        end_sample = int(off * hop_size)
        notes.append((start_sample, end_sample))

    # Merge notes that are too close (less than min_gap apart)
    if not notes:
        return notes

    merged = [notes[0]]
    for start, end in notes[1:]:
        prev_start, prev_end = merged[-1]
        gap = (start - prev_end) / sr
        if gap < min_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def estimate_dominant_peak(signal, sr, min_freq=40, max_freq=4000):
    """
    Very simple but honest "what was the loudest frequency content" in the note.
    Returns the single strongest spectral peak (not trying to be f0).
    Useful for sanity checking even when true fundamental is weak.
    """
    n = len(signal)
    start = int(n * 0.2)
    end = int(n * 0.8)
    seg = signal[start:end]

    if len(seg) < 1024:
        return None

    window = np.hanning(len(seg))
    spectrum = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)

    mask = (freqs >= min_freq) & (freqs <= max_freq)
    spectrum = spectrum[mask]
    freqs = freqs[mask]

    if len(spectrum) == 0:
        return None

    peak_idx = np.argmax(spectrum)
    return float(freqs[peak_idx])


def classify_note(f0):
    """Classify as C or F family based on pitch class."""
    if f0 is None:
        return None, None

    midi = hz_to_midi(f0)
    pc = round(midi) % 12

    # C family: C, C#, Db (pitch class 0 or 1)
    # F family: F, F#, Gb (pitch class 5 or 6)
    if pc in (0, 1):
        note_class = "C"
    elif pc in (5, 6):
        note_class = "F"
    else:
        # Fall back to closest musically plausible
        c_dist = min(abs(pc - 0), abs(pc - 12))
        f_dist = min(abs(pc - 5), abs(pc - 17))
        note_class = "C" if c_dist < f_dist else "F"

    return note_class, midi


def hz_to_midi(f):
    return 69 + 12 * np.log2(f / 440.0)


def midi_to_hz(m):
    return 440 * (2 ** ((m - 69) / 12))


def segment_master_recording() -> tuple[list[dict], np.ndarray, int]:
    """
    Core segmentation logic. Returns (metadata_list, mono_audio, sample_rate).
    Used both for the clean/ inspection output and for --export-to-tests.
    """
    print("Loading recording...")
    sr, data = wavfile.read(INPUT_FILE)

    mono = data.mean(axis=1).astype(np.float32) if len(data.shape) > 1 else data.astype(np.float32)

    mono = mono / np.max(np.abs(mono))
    print(f"Loaded {len(mono) / sr:.2f} seconds @ {sr} Hz")

    noise_floor = db_to_linear(NOISE_FLOOR_DB)
    rms = compute_rms_envelope(mono, sr, HOP_SIZE, FRAME_SIZE)
    gated_rms = np.where(rms > noise_floor, rms, 0.0)

    boundaries = find_note_boundaries(
        gated_rms, sr, HOP_SIZE, noise_floor, MIN_NOTE_DURATION, MIN_GAP
    )
    print(f"Found {len(boundaries)} note regions after noise gating.")

    # Fresh run for the clean inspection dir
    if OUTPUT_DIR.exists():
        for old in OUTPUT_DIR.glob("*.wav"):
            old.unlink()
        for old in OUTPUT_DIR.glob("*.json"):
            old.unlink()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata: list[dict] = []
    note_regions = boundaries[:14]

    print(f"Using first {len(note_regions)} regions (known sequence: 7 C then 7 F).")

    for i, (start, end) in enumerate(note_regions):
        segment = mono[start:end]
        dom = estimate_dominant_peak(segment, sr)

        if i < 7:
            octave = i + 1
            label = f"C{octave}"
            expected_midi = 24 + i * 12
        else:
            octave = (i - 7) + 1
            label = f"F{octave}"
            expected_midi = 29 + (i - 7) * 12

        out_path = OUTPUT_DIR / f"{label}.wav"
        wavfile.write(out_path, sr, (segment * 32767).astype(np.int16))

        meta = {
            "label": label,
            "dominant_peak_hz": round(dom, 1) if dom else None,
            "expected_midi": expected_midi,
            "start_time": round(start / sr, 3),
            "end_time": round(end / sr, 3),
            "duration": round((end - start) / sr, 3),
            "pass": "C" if i < 7 else "F",
        }
        metadata.append(meta)

        print(
            f"{label:6} | dominant={dom:7.1f} Hz | {meta['start_time']:6.2f}s - {meta['end_time']:.2f}s  ({meta['duration']:.2f}s)"
        )

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote inspection copy to {OUTPUT_DIR}")
    return metadata, mono, sr


def export_to_test_fixtures(metadata: list[dict]) -> None:
    """Copy the good labeled WAVs + write the exact segments.json the test suite expects."""
    test_recordings_dir = Path("tests/real_piano/recordings")
    test_meta_path = Path("tests/real_piano/segments.json")

    test_recordings_dir.mkdir(parents=True, exist_ok=True)

    # Remove old WAVs so we don't leave stale bad files
    for old in test_recordings_dir.glob("*.wav"):
        old.unlink()

    segments_json: list[dict] = []

    for m in metadata:
        label = m["label"]
        src = OUTPUT_DIR / f"{label}.wav"
        dst = test_recordings_dir / f"{label}.wav"
        shutil.copy2(src, dst)

        segments_json.append(
            {
                "filename": f"{label}.wav",
                "note_label": label,
                "approx_midi": m["expected_midi"],  # now trusted (order-based)
                "start_time": m["start_time"],
                "end_time": m["end_time"],
                "duration": m["duration"],
                "pass": m["pass"],
            }
        )

    with open(test_meta_path, "w") as f:
        json.dump(segments_json, f, indent=2)

    print("\n=== Exported to test fixtures ===")
    print(f"  WAVs copied to : {test_recordings_dir}")
    print(f"  segments.json  : {test_meta_path}")
    print(f"  {len(segments_json)} recordings with correct ground-truth MIDI labels.")


def main():
    parser = argparse.ArgumentParser(
        description="Segment the master 'c and f.wav' recording using noise gate + known sequence."
    )
    parser.add_argument(
        "--export-to-tests",
        action="store_true",
        help="Copy the resulting clean segments into tests/real_piano/ and write a correct segments.json",
    )
    args = parser.parse_args()

    metadata, _, _ = segment_master_recording()

    if args.export_to_tests:
        export_to_test_fixtures(metadata)


if __name__ == "__main__":
    main()
