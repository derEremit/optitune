# OptiTune Synthetic Test Tone Matrix — Phase 1 Contract

This document is the non-negotiable TDD contract for the DSP team.

All tests in `tests/dsp/test_synth.py`, `test_binning.py`, `test_peaks.py` etc. must be written against (or include) this matrix **before** the corresponding implementation code is considered complete.

## Ground Rules

- Use the exact Fletcher–Young inharmonicity model from the spec §2.1.
- fs = 48000 Hz (native PipeWire rate).
- Every generated tone must be deterministic given the same seed (for reproducible tests).
- "Recovered" values come from the PFD estimator + parabolic peak picker (the same code path the live GUI will use).
- Tolerances are deliberately tighter than the literature (Rigaud 0.33–0.76 %) because we control the signal.

## The Matrix (must all be green)

| ID | Condition          | detune_cents | B          | MIDI notes          | duration | snr_db | with_hammer | Assertions (per tone) |
|----|--------------------|--------------|------------|---------------------|----------|--------|-------------|-----------------------|
| P1 | Perfect (baseline) | 0.0          | 0.00005    | 21 (A0), 60 (C4), 108 (C8) | 3.0 s   | None   | Yes         | f0 err ≤ 0.1 ¢, B err ≤ 3 %, partials 1-6 ≤ 0.2 ¢ |
| P2 | Perfect mid        | 0.0          | 0.0008     | 45, 69, 88          | 2.5 s   | None   | Yes         | same as P1 |
| S1 | Slightly flat      | -1.5         | 0.0002     | 60, 69              | 3.0 s   | None   | Yes         | recovered detune within 0.25 ¢ of injected |
| S2 | Slightly sharp     | +2.7         | 0.002      | 45, 88              | 2.5 s   | None   | Yes         | recovered detune within 0.25 ¢ |
| C1 | Clearly flat       | -12.0        | 0.0003     | 21, 60, 108         | 4.0 s   | -20    | Yes         | recovered within 0.4 ¢; note recognizer still locks |
| C2 | Clearly sharp      | +25.0        | 0.01       | 72, 96              | 2.0 s   | None   | No          | recovered within 0.5 ¢ |
| C3 | Badly detuned      | -40.0        | 0.0005     | 55                  | 3.0 s   | -15    | Yes         | recovered within 1.0 ¢ (stress case) |
| H1 | High-B treble      | 0.0 + small jitter | 0.025 | 100, 104, 107     | 1.8 s   | None   | Yes         | PFD must not hallucinate phantom partials; B recovered within 8 % |
| H2 | Extreme treble     | +1.8         | 0.18       | 108                 | 1.5 s   | -25    | Yes         | Shah-Välimäki 1:2 rule context (later solver) |
| B1 | Bass with thump    | 0.0          | 0.00012    | 21, 28, 33          | 4.5 s   | -22    | Yes (strong) | note recognizer + peak picker must ignore hammer transient for f0 |
| N1 | Noisy real-world   | +3.0         | 0.0006     | 3 notes per octave (21-108 step 12) | 3.0 s | -18 | Yes | f0 still within 0.6 ¢ despite noise; tests robustness of A-weight + binning |

## Additional Required Test Behaviors

- Energy conservation: total RMS of generated tone within 1 % of requested amplitude across all partials.
- Phase continuity: consecutive calls with same parameters (different random seed) produce different but statistically identical spectra.
- Round-trip via binning: after generating tone → STFT (Blackman-Harris 32768) → log-cent binning (10 Hz–10 kHz, 1 cent) → the energy peak for the fundamental must be within 0.3 bin of the analytically computed bin.
- Reproducibility: `generate_inharmonic_tone(..., seed=42)` must produce bit-identical output on every run and every machine (for golden WAV fixtures).

## Golden Fixture Generation (one-time, committed)

After the matrix is green, run once:

```bash
optitune generate-piano-fixture \
  --name "moderately-detuned-practice-piano" \
  --b-curve "realistic" \
  --detune-profile "railsback-plus-random-0-18-cents" \
  --out tests/fixtures/golden/moderately_detuned_practice_piano/
```

This produces:
- 88 short WAVs (or a single multi-channel for convenience)
- `metadata.json` with exact B, applied detune, and target cent offset per key
- A synthetic "ground truth" tuning curve

These fixtures are used for:
- Solver regression tests (Phase 4+)
- User's "compare my real detuned piano against synthetic equivalent" workflow

## How to Add a New Row to the Matrix

1. Add a row above with a new ID.
2. Add the corresponding `def test_matrix_<id>(...)` in the test file(s).
3. The DSP Engineer must make it pass without relaxing tolerances.
4. Update this document and the plan.

**This matrix is the law for Phase 1.** No shortcuts.

*Maintained by the Architect + DSP/Test Engineer pair.*
