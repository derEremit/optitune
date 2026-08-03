# OptiTune User Guide

Walkthrough of a full tuning session (spec §8 style). Version: **0.5.0**.

## Install and launch

```bash
cd optitune
uv sync --extra dev
optitune
# or: uv run optitune
```

Optional: `optitune --a4 442 --device "Focusrite"` to set A4 and pick an input by name.

## 1. New piano session

1. **File → New Piano…**
2. Set **name**, **A4** (415–466 Hz), and **temperament** (ET or historical).
3. **Audio → Input Device…** and select your microphone / interface.

## 2. Hands-free scale recording (recommended)

1. Click a starting key on the keyboard (e.g. **C1**).
2. Ensure **➡️ Auto-advance** is ON.
3. Click **🎙️ Arm Auto-Record**.
4. Walk to the piano and play the C series (C1…C7), then F (F1…F7).  
   OptiTune arms the next octave automatically and switches C→F after the C series.
5. Status bar shows `Series: C (n/7) → note`. Wrong notes flash briefly and do not advance.
6. Stop with **⏹ Stop Arming** when finished.

Tips:

- Play with clear attacks and leave a short gap between notes.
- If a capture is rejected, stay on the same key and try again.
- Crash recovery autosaves measurements to JSON; use **File → Save As…** for a `.pfg` tuning file.

## 3. Compute a stretch curve

1. Toolbar **Solver:** choose `beat-rate` (default), `entropy`, or `octave-entropy`.  
   Entropy modes need notes recorded with spectra (normal Record path).
2. **📈 Compute Curve** (or Tuning → Compute).
3. Railsback and B-curve plots update; live cents/strobe target the curve.

Optional:

- **Tuning → Interval Weights…** — presets (clean octaves, singing twelfths).
- **Tuning → Pitch Raise / Overpull…** — temporary overpull targets while the piano is flat.
- **◎ Multi-partial strobe** — concentric rings for partials 1–3.

## 4. Tune

1. Play each string; wait for the **strobe to stand still** and **cents near 0**.
2. Use keyboard colors: blue = measured, green-ish = near target (after curve).
3. Follow your usual temperament / unisons workflow; OptiTune guides pitch to the curve.

## 5. Save and reopen

- **File → Save As…** → `mypiano.pfg` (XML with B, f0, spectra, curve).
- Later: **File → Open…** and continue.
- **File → Open…** also accepts basic **`.ept`** imports and JSON autosaves.

Unsaved-changes prompt only appears if you opened a `.pfg` and edited it without saving. Casual sessions rely on JSON crash recovery and quit quietly.

## Note follow modes

Toolbar **Follow:**

| Mode | Behavior |
|------|----------|
| **Auto** | Jump to any detected note |
| **Stepwise** | Only ±1 semitone from the locked key (anti-octave jumps) |
| **Lock** | Keep the selected key; detection does not switch |

Click a key to set the lock anchor. Scale auto-record uses its own armed prior.

## Diagnostics

```bash
OPTITUNE_DIAG=1 optitune          # verbose scale/commit logs
OPTITUNE_DIAG=full optitune       # more noise
OPTITUNE_SYNC_ANALYSIS=1 optitune # force GUI-thread analysis (default in tests)
```

## Tests (developers)

```bash
uv run pytest -q -m "not real_piano"
uv run pytest tests/real_piano/ -q    # needs testmaterial FLAC / WAV segments
```

## What next?

Roadmap and open milestones: [`TODO.md`](../TODO.md). Design of scale mode: [`expectation_driven_onset.md`](expectation_driven_onset.md).
