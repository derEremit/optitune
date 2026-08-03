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

- [ ] **Build an estimator ground-truth harness first** — extend
      `tests/real_piano/test_estimator_on_real.py` with per-note expected-MIDI
      assertions over the clean segments in `testmaterial/recordings_clean/`
      (segmented via `tools/segment_real_recording.py`). Record the current
      hit-rate per note as the baseline (expect low notes to fail). This is the
      TDD loop for everything below — iterate here, not on the 67 s master.
- [ ] **Inharmonic comb-filter note scoring** (spec §3.6, Galembo & Askenfelt 1999)
      — new `src/optitune/dsp/note_recognizer.py`: cross-correlate the cent-binned
      spectrum of the last ~0.5 s against a partial-comb template
      `Σₙ exp(−(m − m(fₙ))²/2σ²)` per candidate key (88 dot products on ~12k bins).
      Because the template contains *all* partials, a note whose energy sits in
      partials 2–4 still scores highest at the true fundamental — this is the
      structural fix for octave errors. Test: synthetic low-B bass tones with the
      fundamental attenuated 20 dB must still classify to the true key.
- [ ] **Subharmonic disambiguation in the PFD path** — in `dsp/peaks.py` /
      `_estimate_pitch`: before accepting f₀, check whether f₀/2 (and f₀/3) has
      comb-score/partial support; prefer the subharmonic when it explains ≥ the
      same partial set. Test on synthetic tones with weak fundamentals.
- [ ] **Longer bass analysis frames** — spec §3.3: switch to 65536-sample frames
      (~1.37 s) when the armed/candidate note is below ~A2 (MIDI 45), both in live
      analysis and `_get_fresh_estimate_for_commit`. Test: C1/A0 synthetic tones
      resolve f₀ within 0.25 cent with the long frame where the short frame fails.
- [ ] **Temporal f₀ tracking** — median/mode over the last N estimation ticks
      (attack frames are the outliers; decay is long). Expose as a small pure
      function so it's unit-testable without Qt.
- [ ] **Wire into the expectation layer** — replace the raw `_estimate_pitch`
      result in the scale gate, during-capture validation, and commit decision
      with the note-recognizer + tracking output. The armed target is a *prior*
      (tie-break), never a hard override — the commit gate must still reject a
      genuinely wrong note.
- [ ] **Raise the real-master assertions** — step by step:
      `test_play_c_series_only_with_real_auto_advance` from `>= 1` captured to the
      full C series; then the full test to complete C-then-F with the series
      switch. Run with `OPTITUNE_DIAG=full`, check the `SUMMARY:` line; iterate
      with `OPTITUNE_FAST_C=1` until green.

**Milestone verification:** `uv run pytest tests/real_piano/ -q` green including
the full C-then-F workflow test; estimator harness reports ≥95 % correct note
classification on `recordings_clean`.

---

## Milestone 2 — Hands-free workflow productization (target: `0.4.0`)

Turn the now-working machinery into a user-facing feature (design doc §11
deferred items).

- [ ] **Extract the expectation layer out of the UI class** — new
      `src/optitune/recording/scale_session.py` holding `_scale_pitch_class`,
      grace timers, gate/commit/switch decisions as a pure, Qt-free state machine
      (`OptiTuneMainWindow` becomes a thin adapter). Port existing behavior 1:1;
      unit-test the state machine directly (arm → gate → capture → commit →
      switch → exhaustion), then re-run the real-master tests unchanged.
- [ ] **Current-series indicator** — status-bar widget showing "Series: C
      (5/7 captured)" with the armed target; updates on switch/exhaustion.
- [ ] **Subtle rejection feedback** — consume the existing
      `_during_capture_rejection_until` / rejection-flash hooks: brief red flash
      on the armed key in `keyboard_widget.py` + transient status-bar message.
      pytest-qt test: simulated rejection sets the flash state and it decays.
- [ ] **Series lifecycle UX** — completion feedback when a series is exhausted;
      manual series exit; persist active series + last-recorded note in QSettings
      so a crash mid-session resumes cleanly.
- [ ] **Note-follow modes** (spec §3.6, parity row "Stepwise / Lock"): **Auto**
      (recognizer picks any note), **Stepwise** (±1 semitone from locked note),
      **Lock** (manual only) — toolbar selector; scale mode is a fourth,
      workflow-driven mode. Recognizer logic lives in `dsp/note_recognizer.py`
      from Milestone 1; UI mode just constrains its search window.
- [ ] **Generalize beyond C↔F** — series = any root pitch-class set the user
      arms; `_maybe_switch_series` already sees "third class" events (currently
      logged as ignored) — allow switching to any armed-workflow class, keep
      eagerness. Update the design doc when behavior changes.

**Milestone verification:** fast suite + real-master tests green; manual GUI
session: arm C1, play a scale, watch auto-advance + indicator + rejection flash.

---

## Milestone 3 — Solver suite completion (target: `0.5.0`)

The spec's core differentiator: user-swappable solvers. Beat-rate LS is done;
entropy and friends are not started.

- [ ] **`Solver` protocol** — `src/optitune/solvers/base.py` exactly per spec
      §4.3: `solve(cent_spectra (K,M), b_estimates (K,), constraints) →
      Iterator[TuningCurve]`; `TuningConstraints` (A4, temperament, locked notes,
      interval weights, treble rule), `TuningCurve` ((88,) cent offsets + metadata).
      Retro-fit `beat_rate.py` behind it (thin adapter; keep the existing
      function API for current call sites/tests).
- [ ] **Per-key cent spectra storage** — the entropy solver needs the A-weighted
      cent-binned SPLA per key (spec §2.3 step 3–4). Add `cent_spectrum` to
      `model/key.py`, populate it during capture commit, include in JSON
      persistence (compressed). Test: capture on synthetic tone stores a spectrum
      whose argmax bins sit on the partials.
- [ ] **Entropy solver** — `src/optitune/solvers/entropy.py` per spec §5
      pseudocode: cent-shift = index shift, incremental `p` update (diff regions,
      not full `np.roll`), zero-T Monte Carlo, seeded `default_rng`, stop on K
      consecutive rejections + `H_new < H − eps` + pass cap. Numba only if the
      pure-NumPy sweep exceeds ~100 ms. Tests: (a) deterministic given seed;
      (b) on a synthetic realistic-B 88-key piano the result is Railsback-shaped
      (bass negative, treble positive, ~monotone envelope) and within ~2 cents of
      the beat-rate solver at the extremes; (c) two detuned copies of the same
      spectrum converge to alignment.
- [ ] **Octave-local entropy** — `src/optitune/solvers/entropy_octave.py`
      (Szwajcowski–Pilch, spec §6.3): one variable per key, ~50 trial cents,
      outward from A4, deterministic. ~100 LOC reusing the entropy machinery.
- [ ] **Temperaments** — `src/optitune/model/temperaments.py`: ET, Werckmeister
      III, Kirnberger III, Vallotti, Young, Meantone as cent-offset tables from
      ET; feed into solvers via `TuningConstraints` (regularizer target in
      beat-rate). Test: known third/fifth deviations for Werckmeister III.
- [ ] **Solver worker thread** — `QThread` (spec §4.1) consuming the
      `Iterator[TuningCurve]`, streaming intermediate curves to the GUI,
      cancellable. `_on_compute_curve` grows a solver picker (beat-rate default).
- [ ] **NMF B-estimator (deep analysis)** — `src/optitune/solvers/nmf_b_estimator.py`
      porting beiciliang/estimate-f0-inharmonicity (spec §2.5 parameters).
      Offline "Deep analyze note" action. Test: on the 4 extreme synthetic-matrix
      cases that classical PFD xfails (P2, H1, H2, B1), NMF recovers B within 8 % —
      then flip those xfails to green via the NMF path.
- [ ] **Pitch raise / overpull** — measure-pass on every ~4th key → B-curve fit →
      overpull profile (Rigaud mean octave-type model, high/low variants) →
      `ui/dialogs/pitch_raise.py` wizard. Test: 30-cent-flat synthetic piano
      yields overpull targets above final targets, tapering treble-ward.

**Milestone verification:** all solvers selectable in GUI and green in
`tests/test_tuning_curve.py` + new solver tests; synthetic-matrix xfails resolved
via NMF; CI time still acceptable (mark NMF tests `slow` if needed).

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
