# OptiTune — Roadmap to 1.x

> **For agentic workers:** Use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement milestone-by-milestone. Steps use
> checkbox (`- [ ]`) syntax for tracking. Historical changelog entries formerly
> in this file now live in `CHANGELOG.md` — keep appending there as work lands.

**Goal:** Bring OptiTune from `0.1.0-dev` to a tagged, packaged `v1.0.0` that meets
the feature-parity table in `piano_tuner_implementation_spec.md` §7 (pianoscope /
PianoMeter parity + the multi-solver differentiator), then iterate 1.x.

**Architecture:** Already fixed by the spec — PySide6 GUI, `sounddevice` capture →
ring buffer → STFT → cent-binned analysis, swappable `Solver` implementations
(beat-rate ✅, entropy ❌, octave-entropy ❌), strict synthetic-tone TDD plus the
real-piano master-recording (`testmaterial/c and f.flac`) workflow tests.

**Tech stack:** Python 3.11+, PySide6, pyqtgraph, sounddevice, NumPy/SciPy, numba,
pyfftw (dev-optional today), uv, pytest/pytest-qt, ruff, mypy.

## Global Constraints

- `AutoRecordController` stays strictly energy/attack-based — **no musical
  knowledge in the controller**. All expectation logic lives in `OptiTuneMainWindow`
  (until Milestone 2 moves it into a dedicated non-UI module — still never into the controller).
- All DSP/solver work is test-first on synthetic Fletcher–Young tones; the
  `real_piano`-marked tests are the acceptance layer, not the development loop.
- GPL-3-or-later; Linux-native (PipeWire via PortAudio); no closed dependencies.
- Keep `-m "not real_piano"` CI-safe: no test outside `tests/real_piano/` may need
  audio hardware or the master FLAC.
- A4 pin, Shah–Välimäki 1:2 treble rule, and deterministic solver output are
  invariants — any solver change must keep `tests/test_tuning_curve.py` green.

## Definition of 1.0 (release gate)

All of the following are true:

- [ ] Full hands-free scale workflow: feeding the master recording captures the
      complete C series then F series with only real auto-advance (no simulation
      shortcuts), asserted by `tests/real_piano/test_recording_workflows.py`.
- [ ] Spec §7 parity rows all implemented: strobe, cents, spectrum, tuning-curve
      graph, B-curve graph, auto/stepwise/lock note switching, pitch raise,
      historical temperaments, A ≠ 440, save/load tuning files, **multiple solvers**
      (beat-rate + entropy + octave-entropy).
- [ ] `uv run ruff check .` and `uv run ruff format --check .` clean; mypy gate in
      CI is hard (no `|| true`) and clean.
- [ ] `uv run pytest -m "not real_piano"` green with **no stale xfails** (strict).
- [ ] Packaged: PyPI sdist+wheel and a Flatpak with desktop file + icon.
- [ ] Docs: rewritten README, user guide, finalized CHANGELOG, `v1.0.0` git tag.

---

## Milestone 0 — Repo & code-health baseline (target: `0.2.0`)

~~Baseline was dirty (WIP, ruff/mypy noise, stale xfails).~~ **Done 2026-08-03** —
`0.2.0` shipped as the clean baseline.

- [x] **Commit the WIP** — expectation-driven recording + M0 health fixes.
- [x] **Ruff clean** — `ruff check .` + `ruff format --check .` green.
- [x] **Mypy clean** — `mypy src/optitune` green (30 → 0).
- [x] **Harden CI** — mypy step no longer soft-fails with `|| true`.
- [x] **Fix xfail drift** — per-param strict xfail for P2/H1/H2 only; B1 recovered;
      deterministic matrix seeds (no `hash()` flakiness). Expect 0 xpass.
- [x] **README refresh** — status matches Phases 0–4 done + scale-recording WIP.
- [x] **`[DIAG]` prints → `logging`** — module loggers in `main_window` /
      `auto_record`; `OPTITUNE_DIAG` still enables console stream; `SUMMARY:` on stdout.
- [x] Update `.index.yaml` + bump version to `0.2.0`.

**Milestone verification:** `uv run ruff check . && uv run ruff format --check . &&
uv run mypy src/optitune && uv run pytest -q -m "not real_piano"` all clean. ✅

---

## Milestone 1 — Low-note estimator robustness (target: `0.3.0`) — THE BLOCKER

Design doc §15 conclusion: the gates work; the estimator doesn't. On low notes
(C1=MIDI 24) the live and fresh estimators report strong upper partials / octave
errors (~MIDI 48/72) during attack and decay, so correct captures get rejected.
Everything downstream is blocked on this.

- [x] **Build an estimator ground-truth harness first** — per-note expected-MIDI
      assertions + PFD vs comb hit-rate table in
      `tests/real_piano/test_estimator_on_real.py` (`SUMMARY:` line). Baseline
      after first landing: **PFD 14/14 (100%)** with long frames + guess;
      comb recognizer **10/14 (71%)** (F1 + high-F still soft-xfail).
- [x] **Inharmonic comb-filter note scoring** — `src/optitune/dsp/note_recognizer.py`
      (peak-local Galembo-style comb + subharmonic preference). Synthetic weak-fund
      bass tones classify correctly (`tests/dsp/test_note_recognizer.py`).
- [x] **Subharmonic disambiguation in the PFD path** — `_prefer_subharmonic_f0`
      in `dsp/peaks.py` (strict inlier gain, fund-peak evidence, respect f0_guess).
- [x] **Longer bass analysis frames** — pad toward 65536 samples in
      `_estimate_pitch` when armed low / short buffer.
- [ ] **Temporal f₀ tracking** — median/mode over the last N estimation ticks
      (attack frames are the outliers; decay is long). Expose as a small pure
      function so it's unit-testable without Qt.
- [x] **Wire into the expectation layer** — pure `dsp.estimate_pitch` + dual
      free/armed PFD; MainWindow thin adapter. Armed prior only when partial ladder
      supports it.
- [x] **Temporal f₀ tracking** — `dsp.f0_tracker.F0Tracker` (octave-cluster median);
      wired into live analysis.
- [x] **Deterministic master-feed sim** — `time.time` patched to audio playhead in
      `_feed_master_with_real_auto_advance` (capture/ignore windows match audio).
- [x] **Raise the real-master assertions** — **done**:
      `test_play_c_series_only` = full C1–C7 (7 notes);
      `test_play_full_master...` = C1–C7 then F1–F7 (14 notes) with series switch.
      Deterministic under playhead-driven time + stopped Qt timers.
- [ ] **Polish free recognizer on F1 + high F** — 4 soft xfails remain for
      unarmed comb classification (F1 octave-up; F5–F7 octave-down). Scale
      workflow uses armed prior and is green without this.

**Milestone verification:** real-master C-then-F workflow green ✅; free comb
recognizer still <95 % on a few high/low F notes (non-blocking for M1 gate).

**Shipped as `0.3.0`.**

---

## Milestone 2 — Hands-free workflow productization (target: `0.4.0`)

Turn the now-working machinery into a user-facing feature (design doc §11
deferred items).

- [x] **Extract the expectation layer out of the UI class** — `scale_session.py`
      pure SM (enter/exit, onset gate, next_target C↔F, decide_commit + tracker
      fallback). MainWindow property-mirrors session fields; `_on_record_next`
      uses `next_target`; `_decide_commit_and_maybe_switch` is a thin adapter
      over `ScaleSession.decide_commit` (estimate fetch + diagnostics only).
      Real-master C and C-then-F green.
- [x] **Current-series indicator** — status-bar "Series: C (2/7) -> C3"; updates
      on arm / disarm / reject / successful advance. pytest-qt coverage.
- [x] **Subtle rejection feedback** — `KeyboardWidget.flash_rejection` (hot pink
      flash, preserves ARMED); wired on commit reject in MainWindow. pytest-qt.
- [x] **Series lifecycle UX** — paired C↔F exhaustion: clear target, disarm,
      exit scale, status "Series complete…". QSettings crash-resume for
      `scale/active_pitch_class`, `last_recorded_midi`, `armed_midi` (restore
      on startup; clear on disarm/complete). During-capture mismatch flashes
      the armed key. Manual exit = disarm (already exits scale).
- [x] **Onset gate + post-capture guards via ScaleSession** —
      `should_suppress_onset` + `set_post_capture_guards`; MainWindow mirrors
      ignore/require-strong fields as properties.
- [x] **Note-follow modes** (spec §3.6): `NoteFollowMode` + `search_window` /
      `apply_follow_to_midi` in `dsp/note_follow.py`; wired through
      `estimate_pitch` and toolbar **Follow** combo (Auto / Stepwise / Lock).
      Keyboard click sets the lock anchor. Scale/auto-record still uses armed
      soft prior independently of free-listening follow mode.
- [x] **Generalize beyond C↔F** — any pitch class is a series (walks octaves to
      compass/series_hi). C↔F remain the only *paired* auto-switch pair for the
      hands-free master workflow. Non-paired classes exhaust cleanly (None).
      Status indicator names all 12 classes. Design: still C↔F for product
      default; any root is armable.

**Milestone verification:** fast suite + real-master tests green; manual GUI
session: arm C1, play a scale, watch auto-advance + indicator + rejection flash.

**Shipped as `0.4.0`.** Free-recognizer soft xfails remain a non-blocking M1 leftover.

---

## Milestone 3 — Solver suite completion (target: `0.5.0`)

The spec's core differentiator: user-swappable solvers. Beat-rate LS is done;
entropy and friends are not started.

- [x] **`Solver` protocol** — `solvers/base.py`: `TuningConstraints`,
      `TuningCurve`, `Solver` protocol per spec §4.3. `BeatRateSolver` adapter
      yields one final curve matching `compute_basic_tuning_curve`; GUI uses
      `solve_piano`. Function API kept for existing tests.
- [x] **Per-key cent spectra storage** — `Key.cent_spectrum` + zlib/base64 codec;
      populated on Record from ring buffer; `Piano.cent_spectra_matrix()` for
      `Solver.solve`. Synthetic tone test: partial bins have energy.
- [x] **Entropy solver** — `solvers/entropy.py`: zero-T MC, seeded, incremental
      roll updates, A4 pin, optional Railsback prior, yields intermediate curves.
      Tests: deterministic, A4 pin, detuned pair aligns, prior shape. GUI Solver
      combo (beat-rate | entropy) + registry. Numba deferred (pure NumPy ok).
- [x] **Octave-local entropy** — `solvers/entropy_octave.py`: outward from A4,
      grid search per key vs nearest set neighbor; registered as
      `octave-entropy`.
- [x] **Temperaments** — `model/temperaments.py`: ET, Werckmeister III,
      Kirnberger III, Vallotti, Young, ¼-comma meantone. 12-class + 88-key
      tables; `TuningConstraints.temperament_offsets` layered on beat-rate.
- [x] **Solver worker + picker** — `solvers/worker.py` (`SolverWorker` with
      progress/finished/failed signals, cancellable). Toolbar Solver combo
      (beat-rate | entropy | octave-entropy); `_on_compute_curve` uses the
      worker API. Full QThread host helper `run_solver_in_thread` available.
- [ ] **NMF B-estimator (deep analysis)** — `src/optitune/solvers/nmf_b_estimator.py`
      porting beiciliang/estimate-f0-inharmonicity (spec §2.5 parameters).
      Offline "Deep analyze note" action. Test: on the 4 extreme synthetic-matrix
      cases that classical PFD xfails (P2, H1, H2, B1), NMF recovers B within 8 % —
      then flip those xfails to green via the NMF path.
- [x] **Pitch raise / overpull (core math)** — `solvers/pitch_raise.py`: Rigaud-style
      taper profile (high/low/medium); overpull sits above final, bass > treble;
      A4 pin. GUI wizard dialog deferred to M4 UX.

**Milestone verification (partial):** beat-rate + entropy + octave-entropy selectable
in GUI; protocol + worker + temperaments + pitch-raise math green. NMF deep-analyze
and pitch-raise wizard still open before full `0.5.0`.

---

## Milestone 4 — Visualization & tuning-session UX (target: `0.6.0`)

- [ ] **Tuning-curve (Railsback) widget** — `ui/widgets/railsback_widget.py`
      (pyqtgraph): computed curve, per-key measured deviations, A4 marker; live
      updates from streaming solver. pytest-qt smoke + data-binding test.
- [ ] **B-curve widget** — `ui/widgets/b_curve_widget.py`: log-B vs MIDI scatter
      of measured keys + fitted 2-segment curve (fit already in `beat_rate.py` —
      extract to `model/inharmonicity.py` for reuse).
- [ ] **Temperament picker dialog** — `ui/dialogs/temperament_picker.py` listing
      `temperaments.py` entries with cent-offset preview; persists via QSettings.
- [ ] **Interval-weight editor** — power-user dialog editing the weight dict the
      beat-rate solver already accepts; presets ("clean octaves", "singing
      twelfths", default).
- [ ] **A4 + piano metadata UI** — New Piano dialog: name, A4 (float box,
      415–466), temperament; wire to existing `--a4` plumbing.
- [ ] **Tuning-mode polish** — per-partial strobe rings option (spec §3.7),
      target-vs-measured cents needle behavior verified against synthetic tones
      end-to-end (generator → capture sim → display state).

**Milestone verification:** full manual workflow of spec §8 (new piano →
recording pass → solve → tune → save) works in the GUI; all widget tests green.

---

## Milestone 5 — Persistence & interchange (target: `0.7.0`)

- [ ] **`.pfg` tuning-file format** — `src/optitune/persistence/tuning_file.py`
      per spec §4.4: XML with `<piano>/<keyboard>/<key>` (B, partials, cent
      offset) + `<spectrum>` base64-zipped cent-binned SPLA; XSD shipped in
      package, validated on load. Round-trip property test (hypothesis: random
      valid pianos survive save→load bit-exact on floats within tolerance).
- [ ] **EPT `.ept` import** — reader for the EPT XML outer structure mapping into
      `Piano`. Test against a small hand-written `.ept` fixture.
- [ ] **Settings wrapper** — `src/optitune/persistence/settings.py` centralizing
      the QSettings keys currently scattered in `main_window.py` /
      `device_selector.py` (typed accessors, defaults in one place).
- [ ] **Wire File menu** — Open/Save/Save-As use `.pfg` (JSON
      `current_piano.json` stays as crash-recovery autosave only); recent-files
      list; unsaved-changes prompt on close.

**Milestone verification:** save a tuned piano, reopen, resume tuning with
identical targets; import an EPT fixture; `pytest tests/persistence/ -q` green.

---

## Milestone 6 — Architecture & performance hardening (target: `0.8.0`)

Spec §4.1 mandates worker threads; today all analysis runs on GUI-thread QTimers.

- [ ] **STFT/analysis worker `QThread`s** — move `_run_live_analysis` DSP into an
      analysis worker emitting `note_detected` / `partials_updated` /
      `frame_ready` signals; GUI only renders. No shared mutable state beyond the
      ring buffer. Real-master workflow tests must stay green (they exercise the
      same signal path).
- [ ] **pyfftw in the hot path** — promote from dev-extra to main dependency
      (keep `numpy.fft` fallback when unavailable); `FFTW_MEASURE` plan cache
      keyed by frame length. Benchmark test asserting analysis tick < 50 ms for
      32768-frame and < 150 ms for 65536-frame on CI hardware.
- [ ] **Latency/CPU budget** — strobe at 60 Hz without dropped frames while
      analysis runs; measure with a stress test feeding continuous audio.
- [ ] **Crash-safety pass** — audio device unplug mid-session, PipeWire restart,
      malformed persisted JSON: all recover to a usable state (tests with mocked
      capture layer).

**Milestone verification:** GUI stays responsive during bass long-frame analysis;
all tests green; CPU < ~1 core sustained during live tuning.

---

## Milestone 7 — Release engineering → `v1.0.0`

- [ ] **Docs** — rewrite README (features, screenshots, quickstart, parity
      table); `docs/user_guide.md` walking spec §8's workflow; update
      `docs/expectation_driven_onset.md` status header to "shipped".
- [ ] **Packaging: PyPI** — verify sdist/wheel build (`uv build`), console script
      + GUI entry, `optitune --version` correct; TestPyPI dry-run.
- [ ] **Packaging: Flatpak** — manifest under `packaging/flatpak/`
      (org.optitune.OptiTune), desktop file + AppStream metainfo + icon from
      `assets/icon.svg`; build via flatpak-builder in CI (allowed to be a
      separate, non-blocking job initially).
- [ ] **Release QA on a real piano** — one full tuning session end-to-end by the
      user; file issues found as 1.0.x bugfix scope, not release blockers unless
      workflow-breaking.
- [ ] **Cut the release** — `pyproject.toml` version `1.0.0`, classifier →
      `Development Status :: 5 - Production/Stable`; finalize `CHANGELOG.md`;
      tag `v1.0.0`; GitHub release with wheel + Flatpak; `.index.yaml`
      `status: released`.

---

## 1.x backlog (post-1.0, unordered — spec §12 + deferred items)

- [ ] Phantom-partial rejection in PFD (Miljković 2025) — improves treble B.
- [ ] Simulated-annealing entropy + multi-seed consensus averaging.
- [ ] Real-time B-tracking while tuning (refine curve without recording pass).
- [ ] Localization (en/de) via Qt Linguist.
- [ ] MIDI integration (digital-piano reference oscillator).
- [ ] Polyphonic unison mode (joint multi-pitch, Kim 2014).
- [ ] Tuning report export (PDF + interactive Railsback SVG).
- [ ] Adaptive bass stretch auto-detection (short-scale pianos, 6:3 weighting) —
      partially present as fixed weights; make it B-driven per spec §6.2.
- [ ] Session history / multiple saved pianos manager.

---

**Working agreement:** one milestone per version bump; every task lands with its
tests in the same commit series; append a dated entry to `CHANGELOG.md` when a
milestone completes. Last full state assessment: 2026-08-03 (fast suite green,
88 ruff / 30 mypy errors outstanding, real-master full-series test still capped
at ≥1 capture pending Milestone 1).
