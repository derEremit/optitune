# OptiTune

**One-click launchable, test-driven Linux piano tuning workstation.**

Professional GUI (PySide6) with selectable audio input, live strobe + cents display,
spectrum view, interactive keyboard, beat-rate tuning-curve solver, and an active
hands-free auto-recording workflow (expectation-driven scale mode).

> **Goal**: Match or exceed commercial tools (pianoscope, PianoMeter) on Linux while
> remaining 100% open source under GPL-3 and fully testable via synthetic inharmonic
> tones — no piano required for 95 % of development.

---

## One-Click Quickstart

```bash
# After cloning
cd optitune
uv sync --extra dev          # installs everything, including PySide6 + test tooling
optitune                     # or: ./launch.sh
```

- The GUI opens in < 2 seconds with a polished dark professional theme.
- Full menus: **File** (New Piano, Open/Save `.pfg`, EPT import) • **Audio** • **Tuning** (weights, pitch raise, compute) • **Help**
- CLI also supports `optitune --help`, `optitune --a4 442.0 --device "Focusrite"`

### Development Installation

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src/optitune
uv run pytest -q -m "not real_piano"
```

`uv run optitune --help` and the GUI launch both work cleanly.

---

## Current Status

**Version**: `0.5.0` — live tuner + hands-free C-then-F scale workflow + multi-solver
suite + Railsback/B graphs + `.pfg` / EPT files. Roadmap to `v1.0.0` in
[`TODO.md`](TODO.md); history in [`CHANGELOG.md`](CHANGELOG.md).

| Area | Status |
|------|--------|
| Synthetic Fletcher–Young generator + cent binning + PFD | ✅ (3 extreme matrix rows still soft-xfail until NMF) |
| Live audio + device selector + AnalysisWorker | ✅ |
| Strobe (optional multi-partial rings), cents, spectrum, keyboard | ✅ |
| Hands-free auto-record + ScaleSession (C-then-F on real master) | ✅ |
| Solvers: beat-rate, entropy, octave-entropy + temperaments | ✅ |
| Railsback + B-curve graphs, interval weights, pitch-raise | ✅ |
| `.pfg` save/load, EPT import, crash autosave JSON | ✅ |
| Free comb recognizer on some high/low F notes | 🚧 soft xfails (armed scale path is solid) |
| User guide + Flatpak scaffold + `uv build` | ✅ scaffold |
| NMF deep B, published Flatpak/PyPI, 1.0 tag | ⏳ later |

### How to check live status

```bash
uv run pytest -q -m "not real_piano"          # CI-safe suite
uv run pytest tests/dsp/test_synth.py -q      # 8 matrix pass + 3 xfail, 0 xpass
cat TODO.md                                   # 1.x roadmap (source of truth)
```

Real-piano acceptance tests (need `testmaterial/c and f.flac`):

```bash
uv run pytest tests/real_piano/ -q            # full real-master workflows
OPTITUNE_FAST_C=1 OPTITUNE_DIAG=full \
  uv run pytest tests/real_piano/test_recording_workflows.py -k c_series -s
```

---

## Development is 100% Test-Driven with Synthetic Inharmonic Piano Tones

**This is the non-negotiable heart of the project.**

No DSP, peak picker, strobe, or solver code is allowed to land without first passing
a comprehensive battery of tests that use **synthetic inharmonic piano tones**
generated from the Fletcher–Young model (exactly as specified in
`piano_tuner_implementation_spec.md` §2.1).

### The Mandatory Synthetic Test Matrix

| Condition              | Cents detune     | B range            | MIDI notes     | Purpose                                      |
|------------------------|------------------|--------------------|----------------|----------------------------------------------|
| Perfect                | 0.0              | 0.00005 – 0.05     | 21, 60, 108    | Baseline recovery, binning round-trip        |
| Slightly off           | ±1.5, ±2.7       | 0.0002 – 0.02      | 45, 69, 88     | Human JND resolution of strobe & cents       |
| Clearly off            | ±12, ±25, –40    | realistic          | all registers  | Note recognizer robustness                   |
| High inharmonicity     | 0 + small        | 0.01 – 0.4         | 100–108        | Treble stress (Shah-Välimäki rule)           |
| Bass with hammer thump | 0                | 0.0001–0.001       | 21–33          | Transient + low-frequency detection          |
| Noisy (–18 dB SNR)     | ±3.0             | mid                | 3 per octave   | Real-world robustness                        |

See `docs/synth_test_matrix.md` and `tests/dsp/test_synth.py` for the full contract.
Three extreme classical-PFD cases (P2, H1, H2) remain strict xfails until an NMF path.

---

## Architecture notes

- **`AutoRecordController`** stays strictly energy/attack-based — no musical knowledge.
  Expectation logic lives in pure `ScaleSession` (`recording/scale_session.py`); the
  main window is a thin Qt adapter.
- Live pitch analysis can run on an **`AnalysisWorker`** QThread (production default).
  Tests set `OPTITUNE_SYNC_ANALYSIS=1` for deterministic feed harnesses.
- Diagnostics: set `OPTITUNE_DIAG=1` / `full` for verbose `[DIAG]` logs.
  Real-piano runs emit a machine-readable `SUMMARY:` line on stdout.

---

## Roadmap (high level)

See **[`TODO.md`](TODO.md)** for the full milestone checklist. Summary:

0–2. Baseline, low-note estimator, hands-free productization → **shipped through `0.4.0`**
3–5. Solvers, visualization, `.pfg`/EPT → **largely shipped in `0.5.0`** (NMF optional)
6. Architecture & performance (pyfftw, audio crash-safety) → `0.8.0`
7. Packaging, docs, `v1.0.0` release gate

---

## Contributing

All contributions must keep the synthetic test matrix green (no new XPASSes on
strict xfails). New DSP or solver work **must** be preceded by failing tests on
synthetic tones.

Pull requests are reviewed against the spec and `TODO.md`.

---

## License

Copyright © 2026 OptiTune Contributors.
Released under the **GNU General Public License v3.0 or later**.

See `LICENSE` for the full text.

---

*OptiTune — Because every piano deserves a perfect, reproducible, Linux-native tuning.*
