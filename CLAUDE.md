# OptiTune — Claude Code Instructions

## Current Focus (May 2026)

We are actively improving the **hands-free auto-recording workflow** (auto-arm + auto-advance + capture) using real piano recordings as the primary TDD driver, with a strong focus on expectation-driven (scale-mode) behavior.

### Primary Artifact

**Design Document**: [docs/expectation_driven_onset.md](docs/expectation_driven_onset.md)

This document describes the "Expectation-Driven Onset Detection" system (scale / root-note series mode) and contains the latest diagnostic findings.

### Latest Status (after parallel diagnostic experiments)

Two isolated worktree experiments (loud instrumentation + aggressively frequent fresh analysis in the simulation helper) revealed the real blocker:

- The new commit-time decision gate (`_decide_commit_and_maybe_switch` + `_get_fresh_estimate_for_commit`) and the Layer-1 scale gate are correctly implemented.
- However, `AutoRecordController` almost never reaches `CAPTURE_FINISHED` on the real master recording under the current simulation feeding. It stays in `ARMED` on C1 the entire time.
- Onset confirmation (loud ticks + `db_rise`) fails because of the combination of chunked audio + strict rise thresholds + post-capture protection timers.
- Reported “captures” in the test were polluted by persisted piano state (`~/.config/optitune/current_piano.json`).

The fresh-estimator + commit-time architecture is ready — it just needs the controller to actually deliver completed captures.

### Next Steps (in priority order)

**Immediate (unblock the gates we built):**

1. Make the `AutoRecordController` reach `ONSET_CONFIRMED` → `CAPTURE_FINISHED` on the real master file in the simulation (tune rise/loud-tick logic or simulation feeding).
2. Make the full-master diagnostic test start with a **clean slate** (no persisted piano measurements, fresh `_scale_pitch_class = None`) so the captured count only reflects new activity.
3. Re-run the strict full-master test and confirm the commit-time decision + fresh estimator are now exercised and start rejecting wrong notes.

**Once the above unblocks the flow:**

4. Polish series switching at commit time.
5. Add during-capture validation and subtle rejection feedback.
6. Improve diagnostics (controllable verbosity) and harden the test assertions.
7. Iterate until the full C-then-F series is captured cleanly with only real auto-advance.

### Important Constraints

- Keep `AutoRecordController` strictly energy/attack-based (no musical knowledge).
- All expectation logic lives in `OptiTuneMainWindow`.
- We are deliberately taking a slow, deliberate approach on the detection workflow (as requested).

### How to Resume This Work

After a context clear, the recommended way to resume is:

```bash
# From the project root
cat docs/expectation_driven_onset.md
cat CLAUDE.md
```

Then start a new session and paste the contents of the two files above as context, or simply say:

> "Resume work on expectation-driven onset detection. Latest design is in docs/expectation_driven_onset.md. Current focus and next steps are in CLAUDE.md."

---

**Last updated**: 2026-05 (resumption, continued polishing):
- 1-7 core work complete (as above).
- Iteration speed & diagnostics polished:
  - Fast C-series-only mode (`series="C"` or `OPTITUNE_FAST_C=1`) + dedicated test (`test_play_c_series_only...`). Makes rapid TDD on the C series much cheaper.
  - Machine-readable `SUMMARY:` line at end of every diagnostic run.
  - Explicit "probable octave/partial error" detector (`_is_probable_octave_or_partial_error`) wired into gate / during-capture / commit logs for much clearer diagnostics.
- See design doc §15 for latest status.

Current practical state: Excellent tooling for fast iteration on the C series. Full clean C-then-F still gated by estimator quality on real low notes.