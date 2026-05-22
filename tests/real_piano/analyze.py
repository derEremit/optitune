"""
Quick analysis tool for real piano recordings.

Usage examples:

    # Analyze one note
    python -m tests.real_piano.analyze C4

    # With waveform + spectrum plot
    python -m tests.real_piano.analyze C4 --plot

    # Analyze all recordings
    python -m tests.real_piano.analyze --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make sure we can import from the project
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from optitune.dsp import find_spectral_peaks, pfd_estimate_f0_b, hz_to_midi, midi_to_note_name
from tests.real_piano.loader import load_recording, list_recordings


def analyze_one(name: str, plot: bool = False) -> None:
    """Analyze a single recording."""
    try:
        audio, sr, meta = load_recording(name)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"\n=== Analyzing {name} ===")
    print(f"File       : {meta.get('filename')}")
    print(f"Label      : {meta.get('note_label')}")
    print(f"Approx MIDI: {meta.get('approx_midi')}")
    print(f"Duration   : {meta.get('duration', len(audio)/sr):.2f} s")
    print(f"Sample rate: {sr} Hz")

    # Use middle portion to avoid hammer attack and late decay
    n = len(audio)
    start = int(n * 0.2)
    end = int(n * 0.8)
    segment = audio[start:end]

    if len(segment) < 2048:
        print("Segment too short for reliable analysis.")
        return

    # Run the same peak finder + PFD as the live app
    try:
        w = np.hanning(len(segment))
    except Exception:
        w = np.hanning(len(segment))

    spec = np.fft.rfft(segment * w)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(len(segment), 1.0 / sr)

    peak_fs, peak_as = find_spectral_peaks(
        freqs, power, min_prominence_db=12.0, max_peaks=20
    )

    if len(peak_fs) == 0:
        print("No significant peaks found.")
        return

    f0, B = pfd_estimate_f0_b(
        peak_fs, peak_as, f0_guess=200.0, max_n=16
    )

    detected_midi = round(hz_to_midi(f0)) if f0 > 20 else None
    detected_note = midi_to_note_name(detected_midi) if detected_midi else "?"

    print(f"\n--- Current Estimator Result ---")
    print(f"Detected f0 : {f0:.1f} Hz")
    print(f"Detected    : {detected_note} (MIDI {detected_midi})")
    print(f"Inharmonicity B : {B:.6f}")

    label_midi = meta.get("approx_midi")
    if label_midi and detected_midi:
        error = abs(detected_midi - label_midi)
        print(f"Error vs label : {error} semitones")

    if plot:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))

            # Waveform
            plt.subplot(2, 1, 1)
            t = np.linspace(0, len(audio) / sr, len(audio))
            plt.plot(t, audio, alpha=0.7)
            plt.axvline(start / sr, color="red", linestyle="--", alpha=0.6)
            plt.axvline(end / sr, color="red", linestyle="--", alpha=0.6)
            plt.title(f"{name} - Waveform (analysis window highlighted)")
            plt.xlabel("Time (s)")

            # Spectrum
            plt.subplot(2, 1, 2)
            plt.plot(freqs, 10 * np.log10(power + 1e-12))
            if len(peak_fs) > 0:
                plt.scatter(peak_fs, 10 * np.log10(peak_as + 1e-12), color="red", label="Peaks")
            if f0:
                plt.axvline(f0, color="green", linestyle="--", label=f"Detected f0 = {f0:.1f} Hz")
            plt.title("Spectrum (dB)")
            plt.xlabel("Frequency (Hz)")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("\nmatplotlib not installed — skipping plot. Install with: pip install matplotlib")


def analyze_all(plot: bool = False) -> None:
    """Analyze every available recording."""
    names = list_recordings()
    print(f"Found {len(names)} recordings.\n")
    for name in names:
        analyze_one(name, plot=plot)


def main():
    parser = argparse.ArgumentParser(description="Analyze real piano recordings with current DSP.")
    parser.add_argument("name", nargs="?", help="Recording name, e.g. C4 or F2")
    parser.add_argument("--all", action="store_true", help="Analyze all recordings")
    parser.add_argument("--plot", action="store_true", help="Show waveform + spectrum plot")

    args = parser.parse_args()

    if args.all:
        analyze_all(plot=args.plot)
    elif args.name:
        analyze_one(args.name, plot=args.plot)
    else:
        parser.print_help()
        print("\nExample: python -m tests.real_piano.analyze C4 --plot")


if __name__ == "__main__":
    main()
