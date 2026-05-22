# OptiTune

**One-click launchable, test-driven Linux piano tuning workstation.**

Professional GUI (PySide6) with selectable audio input, live strobe + cents display, spectrum view, interactive keyboard, and full tuning-curve solvers (beat-rate first, entropy per the original Entropy Piano Tuner research).

> **Goal**: Match or exceed commercial tools (pianoscope, PianoMeter) on Linux while remaining 100% open source under GPL-3 and fully testable via synthetic inharmonic tones — no piano required for 95 % of development.

---

## One-Click Quickstart

```bash
# After cloning
cd optitune
uv sync --extra dev          # installs everything, including PySide6 + test tooling
optitune                     # or: ./launch.sh
```

- The GUI opens in < 2 seconds with a polished dark professional theme.
- Full menus: **File** (New Piano, Open, Save, Quit) • **Audio** (Input Device..., Test Signal) • **Help** (About)
- CLI also supports `optitune --help`, `optitune --a4 442.0 --device "Focusrite"`

Alternative launch (works from any directory after clone):

```bash
./launch.sh
```

### Development Installation

```bash
uv sync --extra dev
uv run ruff check
uv run mypy src/optitune
uv run pytest
```

`uv run optitune --help` and the GUI launch both work cleanly.

---

## Current Development Status (auto-updated)

**Latest (2026-05-21)**: Phase 0 complete + repaired. GUI now shows a realistic tuner layout with live (stub) cents display, rotating strobe, spectrum area, and 88-key keyboard.

**Active work**: Phase 1 — strict TDD implementation of the synthetic inharmonic tone generator + full 6-condition test matrix (the foundation everything else depends on).

A dedicated DSP + Test Engineer agent is currently building:
- `dsp/synth.py` (Fletcher-Young exact model)
- `dsp/binning.py` + `dsp/peaks.py` (PFD B estimator)
- Comprehensive pytest matrix that must all be green

Once Phase 1 lands you will be able to:
- Use `optitune generate-tone ...` to create perfect test signals
- Click "Test Signal" in the Audio menu and see the (still stub) strobe + cents react
- Start the real audio capture + device selection work (Phase 2)

**How to check live status**
```bash
uv run pytest -q -k "synth or matrix"   # will be the key command after Phase 1
cat TODO.md                              # always the source of truth
```

The project will not move to Phase 2 until the entire synthetic matrix is 100% green and the CLI tone generator works.

---

## Development is 100% Test-Driven with Synthetic Inharmonic Piano Tones

**This is the non-negotiable heart of the project.**

No DSP, peak picker, strobe, or solver code is allowed to land without first passing a comprehensive battery of tests that use **synthetic inharmonic piano tones** generated from the Fletcher–Young model (exactly as specified in `piano_tuner_implementation_spec.md` §2.1).

### The Mandatory 6-Condition Synthetic Test Matrix (Phase 1 Gate)

| Condition              | Cents detune     | B range            | MIDI notes     | Purpose                                      |
|------------------------|------------------|--------------------|----------------|----------------------------------------------|
| Perfect                | 0.0              | 0.00005 – 0.05     | 21, 60, 108    | Baseline recovery, binning round-trip        |
| Slightly off           | ±1.5, ±2.7       | 0.0002 – 0.02      | 45, 69, 88     | Human JND resolution of strobe & cents       |
| Clearly off            | ±12, ±25, –40    | realistic          | all registers  | Note recognizer robustness                   |
| High inharmonicity     | 0 + small        | 0.01 – 0.4         | 100–108        | Treble stress (Shah-Välimäki rule)           |
| Bass with hammer thump | 0                | 0.0001–0.001       | 21–33          | Transient + low-frequency detection          |
| Noisy (–18 dB SNR)     | ±3.0             | mid                | 3 per octave   | Real-world robustness                        |

Every cell asserts:
- Recovered f₀ within **0.25 cent**
- First 6 partials within **0.5 cent** (parabolic interpolation)
- Estimated B within **8 %** relative (when B > 0.0005)

The generator (`src/optitune/dsp/synth.py`) + tests (`tests/test_synth.py`) are written **before** any analysis code.

This means the next engineer (DSP + Test pair) can immediately start writing the test matrix in `tests/test_synth.py` and the implementation in `dsp/synth.py`. The Phase 0 skeleton leaves the project in exactly that state: ready for TDD on the synthetic tone contract.

---

## Current Status (Phase 0 — Scaffolding Complete)

- ✅ Clean `uv`-managed `src/` layout
- ✅ Full `pyproject.toml` with pinned professional dependencies + GPL-3
- ✅ GPL-3 `LICENSE`
- ✅ Responsive dark-themed `QMainWindow` with exact required menus
- ✅ `argparse` in `__main__.py` for `--help`, `--device`, `--a4`
- ✅ `pytest` + `pytest-qt` skeleton (window instantiation test passes)
- ✅ `ruff` + `mypy` configs (baseline clean)
- ✅ `.gitignore`, `launch.sh`, `assets/icon.svg`, `assets/theme.qss`
- ✅ `README` + living `TODO.md`

**Verification commands** (all must succeed):

```bash
uv run optitune --help
uv run pytest
uv run ruff check
uv run mypy src/optitune
```

---

## Roadmap (High Level)

- **Phase 1**: Synthetic tone generator + cent binning + PFD (TDD gate)
- **Phase 2**: Live audio pipeline + searchable device dialog + loopback self-test
- **Phase 3**: Live strobe, cents display, spectrum, keyboard — usable daily driver on real piano
- **Phase 4**: Recording workflow + beat-rate solver
- **Phase 5**: Persistence, polish, entropy solver skeleton

See `TODO.md` and `piano_tuner_implementation_spec.md` for the complete contract.

---

## Contributing

All contributions must keep the synthetic test matrix green. New DSP or solver work **must** be preceded by failing tests on synthetic tones.

Pull requests are reviewed against the spec and plan.

---

## License

Copyright © 2026 OptiTune Contributors.  
Released under the **GNU General Public License v3.0 or later**.

See `LICENSE` for the full text.

---

*OptiTune — Because every piano deserves a perfect, reproducible, Linux-native tuning.*
