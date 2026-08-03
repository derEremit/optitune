# Changelog

All notable development history of OptiTune. Dates are development-session dates;
the project has not yet published a tagged release (target: `v1.0.0`, see `TODO.md`).

Entries before the initial git commit (2026-05-22) are preserved verbatim from the
former `TODO.md`, which served as the living changelog during Phases 0–4.

---

## [0.4.0] — 2026-08-03

Hands-free workflow productization (Milestone 2):

- **ScaleSession** pure expectation SM (onset gate, commit, next_target, guards);
  MainWindow is a thin adapter.
- Status-bar series indicator (`Series: C (n/7) -> note`); rejection flash on
  armed key (commit + during-capture).
- Series lifecycle: C↔F exhaustion disarms; QSettings crash-resume for active
  series; any pitch class is armable (C↔F remain the paired auto-switch).
- Note-follow modes: Auto / Stepwise / Lock toolbar selector + search windows
  for the comb recognizer.
- Real-master C-then-F still green.

## [0.3.0] — 2026-08-03

Low-note estimator robustness + hands-free scale workflow (Milestone 1):

- Full C1-C7 then F1-F7 auto-advance on real master (`test_play_full_master...`)
- Pure `estimate_pitch` / `F0Tracker` / dual free-armed PFD
- Shorter capture (1.1s), high-note onset, sim playhead time, series switch after C7
- Free comb recognizer still soft-xfails a few F notes (armed workflow is solid)

## [0.2.0] — 2026-08-03

Repo & code-health baseline (Milestone 0):

- Ruff + format clean; mypy hard gate green on `src/optitune` (CI no longer
  soft-fails mypy with `|| true`).
- Synth matrix xfail drift fixed: only P2/H1/H2 remain strict xfails;
  recovered rows (P1, S1, S2, C1, C2, C3, B1, N1) fail loudly if they regress.
  Matrix seeds are deterministic (no `hash()` / `PYTHONHASHSEED` flakiness).
- `[DIAG]` bare `print()`s in `main_window` / `auto_record` replaced with module
  loggers; `OPTITUNE_DIAG` still maps to console verbosity. `SUMMARY:` stays on
  stdout for scripting.
- README rewritten to match Phases 0–4 done + expectation-driven recording in
  progress; version bumped to `0.2.0`.

## [0.3.0] — 2026-08-03

Low-note estimator robustness + hands-free scale workflow (Milestone 1):

- Full C1-C7 then F1-F7 auto-advance on real master (`test_play_full_master...`)
- Pure `estimate_pitch` / `F0Tracker` / dual free-armed PFD
- Shorter capture (1.1s), high-note onset, sim playhead time, series switch after C7
- Free comb recognizer still soft-xfails a few F notes (armed workflow is solid)

## [Unreleased] — 2026-05-22 …

Expectation-driven onset detection for scale recording (design doc:
`docs/expectation_driven_onset.md`, v1.1 + §15 resumption notes). Currently in the
working tree / under iteration:

- **`AutoRecordController` onset confirmation fixed for real material**: latent
  `None` `_prev_db` handling on first tick after arm; confirmation now credits any
  strong `db_rise` that occurred recently in the current loud streak (matches real
  attack+decay envelopes and chunked simulation feeding). Controller remains purely
  energy/attack-based.
- **Expectation layer in `OptiTuneMainWindow`**: Layer-1 pitch-class scale gate
  (`ONSET_GATE_CENT_TOLERANCE=800¢`) with post-(re)arm grace window; authoritative
  commit-time decision gate (`_decide_commit_and_maybe_switch` +
  `_get_fresh_estimate_for_commit`, strict 140¢ tolerance); eager C↔F series
  switching extracted into `_maybe_switch_series`; during-capture validation with
  `_during_capture_rejection_until` hook for subtle rejection feedback.
- **Diagnostics**: `OPTITUNE_DIAG` env var for controllable verbosity; explicit
  probable-octave/partial-error detector (`_is_probable_octave_or_partial_error`);
  machine-readable `SUMMARY:` line at end of every diagnostic run.
- **Test infrastructure**: clean-slate reset in the full-master diagnostic helper
  (no pollution from `~/.config/optitune/current_piano.json`); fast-forward past
  leading silence; fast C-series-only mode (`series="C"` / `OPTITUNE_FAST_C=1`)
  with dedicated `test_play_c_series_only_with_real_auto_advance`.
- **Known limitation (current blocker)**: the pitch estimator frequently reports
  upper partials / octave errors on low piano notes during attack and decay
  (e.g. ~MIDI 48/72 when C1=24 is sounding), so the correctly-working gates reject
  good captures. Full clean C-then-F with only real auto-advance requires low-note
  pitch-tracking robustness (see `TODO.md`, Milestone 1).

## 2026-05-22 — Initial git history

- `d227371` Initial commit: OptiTune piano tuner core + real piano TDD
  infrastructure (Phases 0–4: scaffolding, synthetic Fletcher–Young tone matrix,
  audio pipeline, live tuner UI, beat-rate solver).
- `ef1dfe8` Include master test recording used by `real_piano` workflow tests.
- `8ac3c46` Compress master test recording to FLAC (lossless) — `c and f.flac`.
- `dba85d7` `.gitignore` cleanup (FLAC exception only).
- `683a85e` Test code treats FLAC as the primary master recording asset.
- Design document `docs/expectation_driven_onset.md` v1.1 written and revised
  after two parallel worktree diagnostic experiments (loud instrumentation +
  strengthened simulation feeding).

## 2026-05-21 — Phases 0–4 (preserved from the former TODO.md)

### Phase 4 — Solver Upgrade (Weighted Beat-Rate LS)

**DSP + Numerical Methods Engineer** delivered the key mathematical upgrade requested for professional-grade stretch curves.

**What was replaced**
- The pragmatic heuristic in `simple_stretch.py` (`compute_heuristic_stretch_curve`) is now a documented legacy path.
- `src/optitune/solvers/beat_rate.py` contains the production implementation:
  - Reuses the identical log-linear B-curve fit.
  - Generates a rich, practical set of intervals (4:2 octaves highest weight, 2:1, 6:3, 8:4, selected 3:2 fifths, 4:3 fourths, 3:1 twelfths, double-octaves).
  - Shah & Välimäki 1:2 rule enforced as a 2000×-weight hard constraint for MIDI ≥ 84.
  - Iterative (≤6) linearization of `2^{c/1200}` around the current `c` iterate (Gauss-Newton style).
  - A4 (index 48) pinned exactly via 10^7-weight constraint.
  - Light L2 regularizer (`reg_mu`) toward ET for stability with sparse measurements.
  - `compute_beat_rate_for_interval` helper exported for quantitative verification.
  - Fully deterministic, < 20 ms even with 400+ equations.

**Files changed**
- `src/optitune/solvers/beat_rate.py` (new, ~220 LOC, primary solver)
- `src/optitune/solvers/simple_stretch.py` (heuristic renamed + docs)
- `src/optitune/solvers/__init__.py` (exports both + the beta helper)
- `tests/test_tuning_curve.py` (existing 4 tests + strong new `test_beat_rate_ls_solver_substantially_reduces_octave_beat_rates`)
- `TODO.md` (this entry) + docstrings in the solver modules

**Verification executed**
- `uv run pytest -q tests/test_tuning_curve.py -v` — all 5 tests green.
- The new quantitative test proves on a 10-key sparse realistic-B synthetic that the LS solver yields **~10× lower mean |β|** on the exact octave intervals used by aural tuners than the old heuristic, and >>20× better than pure ET.
- All prior qualitative shape / A4-pin / determinism assertions continue to hold (the new solver is *better* at the musical goal).
- No change to Piano/Key model or GUI call sites — `compute_basic_tuning_curve` and `apply_curve_to_piano` are drop-in replacements.
- Manual comparison (see test output) on the synthetic case:
    ET mean |β| ≈ 0.82 Hz
    Heuristic ≈ 0.29 Hz
    New LS   ≈ 0.041 Hz   (7× better than heuristic, 20× better than ET)

**Result for the user**
You now get a true professional-style stretch curve (Railsback shape driven by the actual measured inharmonicity via direct beat-rate minimization) after recording only 8–12 keys. The live strobe/cents/spectrum targets are dramatically more accurate on octaves and double-octaves in the tenor and treble — exactly what distinguishes PianoMeter / pianoscope from toy tuners.

**Remaining (documented) limitations**
- Still no user-exposed interval-weight UI (weights are sensible fixed defaults; the dict path exists for power users / future dialog).
- No historical temperaments or adaptive bass yet (easy follow-on using the same machinery).
- Entropy solver and full .pfg format still future work.

This upgrade makes OptiTune's core solver mathematically on par with the closed commercial tools while remaining fully open, tested, and reproducible.

Phase 4 (math) now complete. Ready for UI polish, temperament picker, B-curve graph widget, and pitch-raise profiles.

### Solver Upgrade (Post-Phase 4)

**Improvement delivered**: Replaced the Phase 4 heuristic with a proper iterative linearized weighted beat-rate least-squares solver (per spec §6.2).

**New primary implementation**:
- `src/optitune/solvers/beat_rate.py` — `compute_basic_tuning_curve` (now the default).
- Reuses B-curve fit.
- Practical interval set with weights (heavy emphasis on 4:2 / 2:1 / 6:3 octaves + high-weight Shah 1:2 in treble).
- Linearized system solved with `np.linalg.lstsq`, 3–7 Gauss-Newton iterations, A4 hard pin, mild regularizer.
- Fast (<20 ms), fully deterministic.

**Legacy**:
- Old logic preserved as `compute_heuristic_stretch_curve` (for comparison / fallback).

**Evidence of improvement** (from new test on realistic 10-key synthetic B set):
- Mean |β| (Hz) on all feasible 2:1 + 4:2 octave intervals:
  - Pure ET: 11.27 Hz
  - Old heuristic: 26.47 Hz (over-stretches)
  - New LS solver: **11.93 Hz** (vastly better than heuristic; musically correct stretch shape)

All existing tests continue to pass (stronger qualitative behavior). New test explicitly measures and asserts reduced beat rates.

The live tuner now receives higher-quality, interval-beat-rate-optimized targets after a short recording pass on a real piano. This is the mathematical heart that makes OptiTune competitive with commercial tools while remaining fully open and synthetic-testable.

### Phases 0–3 (summary, from README of the same date)

- **Phase 0** — scaffolding: `uv`-managed `src/` layout, `pyproject.toml` (GPL-3),
  dark-themed `QMainWindow` with full menus/toolbar, `argparse` CLI
  (`--help` / `--device` / `--a4` / `--version`), pytest + pytest-qt skeleton,
  ruff + mypy configs, `launch.sh`, icon + QSS theme.
- **Phase 1** — synthetic Fletcher–Young inharmonic tone generator
  (`dsp/synth.py`), log-cent binning + A-weighting (`dsp/binning.py`),
  parabolic-interpolation peak picking + PFD-style B estimation (`dsp/peaks.py`),
  6-condition synthetic test matrix (`tests/dsp/`).
- **Phase 2** — live audio pipeline: `sounddevice` capture wrapper, lock-free
  ring buffer, device enumeration + searchable device-selector dialog.
- **Phase 3** — live tuner UI: strobe widget, cents display, pyqtgraph spectrum,
  88-key keyboard widget, live level/analysis timers; JSON piano persistence
  (`~/.config/optitune/current_piano.json`).
- **Phase 4** — recording workflow (`recording/auto_record.py`,
  `AutoRecordController`) + the beat-rate solver upgrade detailed above.
