"""
OptiTune command-line entry point and GUI launcher.

`uv run optitune --help` works immediately.
`uv run optitune` (or `optitune` after install) launches the responsive dark GUI shell.
Phase 1 adds the `generate-tone` subcommand for the synthetic matrix contract.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from optitune.ui.main_window import OptiTuneMainWindow


def _add_generate_tone_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the Phase 1 CLI helper for synthetic tone generation."""
    p = subparsers.add_parser(
        "generate-tone",
        help="Generate a deterministic Fletcher-Young inharmonic piano tone (WAV).",
        description="Produce a reproducible synthetic tone for testing and fixtures. Uses the exact model from the implementation spec.",
    )
    p.add_argument("--midi", type=int, default=69, help="MIDI note number (0-127)")
    p.add_argument("--cents", type=float, default=0.0, help="Detune in cents relative to ET")
    p.add_argument("--b", type=float, default=0.0003, help="Inharmonicity coefficient B")
    p.add_argument("--duration", type=float, default=2.0, help="Duration in seconds")
    p.add_argument("--fs", type=int, default=48000, help="Sample rate")
    p.add_argument("--snr-db", type=float, default=None, help="Add white noise for this SNR (None = clean)")
    p.add_argument("--no-hammer", action="store_true", help="Disable hammer transient model")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--out", type=Path, required=True, help="Output .wav path")
    p.add_argument("--a4", type=float, default=440.0, help="A4 reference for MIDI conversion")
    p.set_defaults(cmd="generate_tone")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Supports GUI flags + Phase 1 subcommands."""
    parser = argparse.ArgumentParser(
        prog="optitune",
        description="OptiTune — Professional Linux piano tuning workstation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        "-d",
        metavar="NAME",
        default=None,
        help="Preferred audio input device name or index (restored from settings in later phases)",
    )
    parser.add_argument(
        "--a4",
        type=float,
        default=440.0,
        metavar="HZ",
        help="Reference A4 frequency in Hz (affects all pitch calculations)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__import__('optitune').__version__}",
    )

    subparsers = parser.add_subparsers(dest="cmd")
    _add_generate_tone_subparser(subparsers)

    return parser.parse_args(argv)


def _cmd_generate_tone(args: argparse.Namespace) -> int:
    """Execute the generate-tone subcommand (no Qt needed)."""
    from scipy.io import wavfile

    from optitune.dsp.synth import generate_inharmonic_tone

    y = generate_inharmonic_tone(
        args.midi,
        detune_cents=args.cents,
        B=args.b,
        duration=args.duration,
        fs=args.fs,
        snr_db=args.snr_db,
        with_hammer=not args.no_hammer,
        seed=args.seed,
        a4=args.a4,
    )

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write as float32 WAV (standard for analysis tools)
    wavfile.write(out_path, args.fs, y.astype("float32"))
    print(f"Wrote {len(y)/args.fs:.3f}s tone to {out_path} (midi={args.midi}, cents={args.cents}, B={args.b}, seed={args.seed})")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: handle subcommands (generate-tone) or launch the GUI."""
    args = parse_args(argv)

    if getattr(args, "cmd", None) == "generate_tone":
        return _cmd_generate_tone(args)

    # Default: launch GUI (Phase 0+ behavior)
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("OptiTune")
    app.setApplicationVersion(__import__("optitune").__version__)
    app.setOrganizationName("OptiTune")

    window = OptiTuneMainWindow(a4=args.a4, device=args.device)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
