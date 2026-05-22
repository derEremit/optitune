# A Linux Piano Tuner — Implementation Specification

**Target reader:** Senior software engineer with Python/C++, DSP, and Qt experience
**Target stack:** Python 3.11+, PyQt6, PyQtGraph, sounddevice (PortAudio over PipeWire), NumPy/SciPy, FFTW (via `pyfftw`)
**Goal:** A Linux-native, GPL-3-compatible piano-tuning workstation that meets or exceeds [pianoscope](https://www.pianoscope.app/en) and [PianoMeter](https://pianometer.com) in functionality, with the entropy method of [Entropy Piano Tuner](https://gitlab.com/tp3/Entropy-Piano-Tuner) available as a first-class algorithm alongside conventional beat-rate methods.

Document version: 1.0 — 2026-05-21

---

## 1. Why this document exists

Three of the four serious piano-tuning apps (pianoscope, PianoMeter, TuneLab desktop) are closed source and unavailable on Linux. The only open-source option, [Entropy Piano Tuner (EPT)](https://gitlab.com/tp3/Entropy-Piano-Tuner) by Hinrichsen & Wick at Universität Würzburg, has not had a release since v1.2.0 in March 2017, the project website [piano-tuner.org](http://piano-tuner.org) is offline, and the AUR package is broken against modern qwt. The math, however, is sound and well-published.

This document specifies a clean reimplementation in Python/PyQt6 that:

1. Reuses the **mathematical model** of EPT (entropy minimization) and pianoscope/PianoMeter (interval-weighted beat-rate minimization) — both as user-selectable solvers.
2. Replaces EPT's C++/Qt5/qmake architecture with a modern Python/PyQt6 design.
3. Adds what EPT lacks: historical temperaments, robust pitch raise, modern UX, deterministic reproducible tunings.
4. Stays fully Linux-native via PipeWire/PortAudio.

The remainder of this document is organized so that an engineer can implement it module-by-module without further research.

---

## 2. Mathematical foundation

### 2.1 The Fletcher–Young inharmonicity model

For a stiff piano string of length \(L\), tension \(T\), linear mass \(\mu\), Young's modulus \(E\) and diameter \(d\), the \(n\)th partial frequency is ([Rigaud, David & Daudet, 2013](https://www.institut-langevin.espci.fr/biblio/2020/3/5/916/files/2013_a_parametric_model_and_estimation_techniques_for_the_inharmonicity_and_tuning_of_the_piano.pdf), eqs. 1–3):

\[
f_n \;=\; n\, F_0\, \sqrt{1 + B\, n^2}, \qquad n = 1, 2, 3, \ldots
\]

\[
F_0 \;=\; \frac{1}{2L}\sqrt{\frac{T}{\mu}}, \qquad B \;=\; \frac{\pi^3 E\, d^4}{64\, T\, L^2}
\]

\(B\) is the dimensionless **inharmonicity coefficient**. Typical range across the keyboard: \(B \in [10^{-5},\, 10^{-2}]\) in [Rigaud et al.](https://www.institut-langevin.espci.fr/biblio/2020/3/5/916/files/2013_a_parametric_model_and_estimation_techniques_for_the_inharmonicity_and_tuning_of_the_piano.pdf); Hinrichsen quotes 0.0002 (bass) to 0.4 (high treble) in [arXiv:1203.5101](https://arxiv.org/pdf/1203.5101). Both ranges are used in practice — the difference is whether one fits \(B\) directly or \(\log B\).

Across the keyboard \(B(m)\) (with MIDI \(m\)) is well modeled in log-space as two linear segments (bass/treble bridges) joined by a smooth transition. Rigaud et al. parameterize this with slope/intercept pairs \((s_B, y_B)\) per bridge and report initialization values \(s_B = 8.9 \times 10^{-2}\), \(y_B = -7\) for the treble bridge.

> **Engineering implication.** The full B-curve has only ~4 free parameters. Once you measure \(B\) for a dozen keys you can interpolate the rest reliably. This is what makes "pitch raise" possible — a fast pass on every 4th key gives enough data.

### 2.2 The Railsback stretch

A perfectly equal-tempered tuning on a piano sounds flat in the bass and sharp in the treble because of inharmonicity. The empirically-observed deviation from ET is called the **Railsback curve**. [Giordano (2015)](https://asa.scitation.org/doi/pdf/10.1121/1.4931439) shows that minimizing pairwise sensory dissonance (Plomp–Levelt model) on real piano spectra quantitatively reproduces the Railsback curve. [Jaatinen & Pätynen (2022)](https://pubs.aip.org/jasa/article/152/2/1146/2838401) further establishes that the inharmonic Railsback curve is also the curve human listeners *prefer*, even compared to subjective octave-stretch experiments with harmonic tones.

> **Engineering implication.** The "correct" tuning is not a free choice — any solver should produce a Railsback-like curve on real pianos. Use this for unit tests: a synthetic dataset with realistic B-values should produce a stretch curve matching published Railsback data within ~2 cents at the extremes.

### 2.3 Hinrichsen's entropy-minimization principle

From [Hinrichsen, Rev. Bras. Ens. Fís. 34(2) 2301 (2012)](https://arxiv.org/pdf/1203.5101), Appendix A. Quoting directly:

> "Add the A-weighted power spectra of all 88 tones and compute the entropy. Randomly change one of the pitches and compute the entropy again. If the entropy is lower accept the pitch change, otherwise restore the previous value. This simple procedure is iterated until no further improvement is obtained, meaning that the algorithm has found a local minimum of the entropy."

Formally, for each key \(k = 1\ldots K\) (with \(K = 88\)):

1. Record \(T \approx 20\) s of audio at \(S = 44100\) Hz; let \(y^{(k)}_j\) for \(j = 0\ldots ST-1\) be the samples.
2. Compute \(\tilde y^{(k)}_q = \text{FFT}(y^{(k)})\) for \(q = 0\ldots Q\) with \(Q = ST/2\). Frequency of bin \(q\) is \(f(q) = q/T\).
3. Coarse-grain by **logarithmic 1-cent binning** \(m \in [0, 12000]\) covering 10 Hz to 10 kHz:

\[
I^{(k)}_m \;=\; \sum_{q=0}^{Q} \delta_{m,\,\lfloor 1200 + \log_2(q/(10T)) \rfloor}\; |\tilde y^{(k)}_q|^2
\]

Bin centers: \(f(m) = 10 \cdot 2^{m/1200}\) Hz, i.e. one cent per bin.

4. Apply [IEC 61672:2003](https://en.wikipedia.org/wiki/A-weighting) A-weighting to convert intensity to SPLA \(L^{(k)}_m\) (formula in §3.4 below).

5. **Tuning is then a pure index shift.** Changing the tuning of key \(k\) by \(c\) cents is the array operation \(L^{(k)}_m \to L^{(k)}_{m-c}\). No re-FFT, no re-record. This is what makes the entropy optimizer fast.

6. The cumulative spectrum and entropy are:

\[
p_m = \sum_{k=1}^{K} L^{(k)}_m, \qquad p_m \leftarrow p_m / \sum_m p_m, \qquad H = -\sum_m p_m \ln p_m
\]

7. **Zero-temperature Monte Carlo optimization**: perturb one key's pitch shift by \(\pm 1\) cent, recompute \(H\), accept iff \(\Delta H < 0\). Iterate until no acceptance over a full sweep.

**Physical intuition.** When partials of different keys overlap (well-tuned), they pile up in fewer bins → \(p_m\) becomes more concentrated → \(H\) drops. The minimum of \(H\) corresponds to maximal partial coincidence — the same notion of consonance Giordano formalizes via sensory dissonance.

**Key strength**: needs no explicit partial picking, no B-fitting, no interval-weight choices.
**Key weakness**: non-deterministic (local minima), bass-noise-sensitive, slow (minutes), no temperament freedom.

### 2.4 Szwajcowski–Pilch octave-local entropy

[Szwajcowski & Pilch (2020)](https://www.sciencedirect.com/science/article/pii/S0003682X20300050) modify the method to apply spectral entropy minimization **octave-by-octave** rather than globally. This restores determinism, accelerates convergence, and reportedly performs competitively with aural tuning in blind tests. The math is the same; the optimization scope shrinks. **Implement this as an option** — see §6.3.

### 2.5 Rigaud–David–Daudet NMF estimator for \((F_0, B)\)

The state-of-the-art \((F_0, B)\) joint estimator ([Rigaud et al., JASA 2013](https://www.institut-langevin.espci.fr/biblio/2020/3/5/916/files/2013_a_parametric_model_and_estimation_techniques_for_the_inharmonicity_and_tuning_of_the_piano.pdf)). Reference Python implementation: [beiciliang/estimate-f0-inharmonicity](https://github.com/beiciliang/estimate-f0-inharmonicity).

Build observation matrix \(V\) from short-time spectra. Approximate \(V \approx WH\) where \(H\) is fixed-binary (which note is played at which frame), and the dictionary \(W\) is a parametric column model:

\[
W^{\theta_r}_{kr} = \sum_{n=1}^{N_r} a_{nr} \cdot g_\tau(f_k - f_{nr})
\]

with \(g_\tau\) a Gaussian kernel and \(f_{nr} = n F_{0r}\sqrt{1 + B_r n^2}\). Minimize a Kullback–Leibler divergence plus an inharmonicity regularizer:

\[
C(\theta,\gamma,H) \;=\; \sum_{k,t} d_{\beta=1}\!\left(V_{kt}\,\Big|\, \sum_r W^{\theta_r}_{kr} H_{rt}\right) \;+\; \lambda \sum_{r,n} \left(f_{nr} - n F_{0r}\sqrt{1 + B_r n^2}\right)^2
\]

Multiplicative updates (eqs. 17–20 in the paper) with the closed-form

\[
F_{0r} \;=\; \frac{\sum_n f_{nr}\, n\sqrt{1 + B_r n^2}}{\sum_n n^2 (1 + B_r n^2)}.
\]

Reported parameter settings: \(F_s = 22050\) Hz, 500 ms Hanning window (1 s for extreme bass), \(N_r = \min(30,\, f_{N_r,r} < F_s/2)\), \(\beta = 1\), initial \(\lambda = 1.25\times 10^{-1}\) decaying to \(5\times 10^{-3}\) after iteration 100, 150 outer iterations × 30 inner. Average relative deviation from ground truth: 0.33 % synthetic, 0.76 % real piano — vs 0.78 %/3.3 % for the classical Partial-Frequency-Deviation algorithm.

> **Engineering implication.** For high-accuracy single-note B estimation, use NMF. For real-time per-note display, a faster PFD-style peak picker (next section) is enough.

### 2.6 Fast PFD-style B estimation

The classical alternative is the Partial-Frequency-Deviation method ([Rauhala, Lehtonen & Välimäki, JASA 121(5) EL184 (2007)](https://pubs.aip.org/jasa/article/121/5/EL184/538552)): detect partial peaks via parabolic interpolation on the log-magnitude spectrum, then fit \((F_0, B)\) by minimizing

\[
\sum_{n} w_n \left( f^{\text{meas}}_n - n F_0\sqrt{1 + B n^2} \right)^2
\]

with weights \(w_n\) inversely proportional to noise. Fast (< 10 ms per note), good enough for the inharmonicity-display loop. Use NMF only when the user explicitly requests a "deep analysis" pass on a note.

A modern refinement excluding phantom partials is given in [Miljković et al., JASA 158(4) 3187 (2025)](https://pubs.aip.org/jasa/article/158/4/3187/3368787) — implement as a v2 polish.

### 2.7 The high-treble rule of Shah & Välimäki

For the top octave, where inharmonicity dominates and aural tuners cannot reliably hear beats, [Shah & Välimäki (Applied Sciences 10(6) 1983, 2020)](https://www.mdpi.com/2076-3417/10/6/1983) found via listening tests that the best rule is **1:2 partial matching**: tune the upper note's fundamental to the lower note's 2nd partial. Apply this as the boundary condition in the treble end of any stretch solver.

---

## 3. Signal-processing pipeline

### 3.1 Pipeline overview

```
microphone ─▶ PipeWire ─▶ sounddevice InputStream (callback) ─▶ ring buffer
                                                                    │
                                                                    ▼
                                                  STFT worker (QThread): pyFFTW
                                                                    │
                              ┌────────────────────────────┬────────┴───────────────┐
                              ▼                            ▼                        ▼
                       cent-binned spectrum         note recognizer         partial peak picker
                              │                            │                        │
                              ▼                            ▼                        ▼
                       entropy aggregator            current key                B estimator
                              │                            │                        │
                              └────────────────────────────┴────────────┬───────────┘
                                                                        ▼
                                                                 tuning solver
                                                                        │
                                                                        ▼
                                                                       GUI
```

### 3.2 Audio I/O

Use [`python-sounddevice`](https://python-sounddevice.readthedocs.io) (PortAudio wrapper). Under PipeWire on modern distros this transparently talks to `pipewire-pulse` or the native PortAudio backend; no extra config needed.

```python
import sounddevice as sd
stream = sd.InputStream(
    samplerate=48000,
    blocksize=1024,             # ~21 ms per callback
    channels=1,
    dtype='float32',
    callback=audio_callback,
    latency='low',              # PortAudio chooses best PipeWire setting
)
```

Avoid the [`pipewire_python`](https://pypi.org/project/pipewire_python/) PyPI package — it shells out to `pw-cat` and is unsuitable. If sample-accurate behavior or appearing as a proper PipeWire node is required (rare for a tuner), use [`JACK-Client`](https://jackclient-python.readthedocs.io) with the `pipewire-jack` shim.

The audio callback runs in PortAudio's real-time thread. Do nothing in it except write into a lock-free ring buffer; all DSP runs on a worker `QThread`.

### 3.3 STFT parameters

| Parameter | Value | Rationale |
|---|---|---|
| Sample rate | 48 kHz | Native PipeWire rate; covers all piano partials. |
| Frame length \(N\) | 32768 samples (~683 ms) | 1.47 Hz bin width → ~5.6 cents at C4, ~22 cents at A0. Long frame for bass resolution. |
| Hop | 8192 samples (~170 ms) | 75% overlap; smooth GUI updates. |
| Window | Blackman–Harris (4-term) | −92 dB sidelobes, essential to resolve dense bass partials. |
| FFT | `pyfftw` with `FFTW_MEASURE` plan, cached | ~3× faster than `numpy.fft` and matches the EPT FFTW3 implementation. |

For the bass register (A0–A2) optionally switch to a longer frame (65536 / ~1.37 s). EPT's [`modules/core/analyzers/fftanalyzer.cpp`](https://gitlab.com/tp3/Entropy-Piano-Tuner/-/tree/master/modules/core/analyzers) and [`signalanalyzer.cpp`](https://gitlab.com/tp3/Entropy-Piano-Tuner/-/tree/master/modules/core/analyzers) take the same approach.

### 3.4 Log-cent binning and A-weighting

Implement Hinrichsen §A exactly, but at 48 kHz:

```python
import numpy as np

CENTS_PER_BIN = 1
F_LO = 10.0                            # Hz
F_HI = 10_000.0                        # Hz
M = int(round(1200 * np.log2(F_HI / F_LO)))   # = 12000 bins

def bin_index(f_hz):                   # f -> log-cent bin
    return np.floor(1200 * np.log2(f_hz / F_LO)).astype(int)

def a_weight_db(f):                    # IEC 61672:2003
    f2 = f * f
    Ra = (12200**2 * f2**2) / (
        (f2 + 20.6**2) *
        np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) *
        (f2 + 12200**2)
    )
    return 2.0 + 20.0 * np.log10(Ra)
```

Bin the magnitude-squared FFT into the cent grid by `np.add.at(L_k, bin_idx, mag2)` (handles collisions correctly), then add the A-weight to convert to SPLA. Cache the bin-index lookup and the A-weighting vector — they're constants across the session.

### 3.5 Peak detection and partial picking

For real-time per-note feedback (cents-off display, strobe) and PFD-style B estimation:

1. On each STFT frame, find local maxima above an adaptive noise floor (median of log-spectrum + ~12 dB).
2. **Parabolic interpolation** on the three log-magnitude samples around each peak gives sub-bin frequency accuracy ([standard SMS-tools approach](https://github.com/MTG/sms-tools)):

\[
\delta = \tfrac{1}{2}\,\frac{\log|X_{k-1}| - \log|X_{k+1}|}{\log|X_{k-1}| - 2\log|X_k| + \log|X_{k+1}|}, \qquad f_{\text{peak}} = (k + \delta)\, f_s / N
\]

3. Associate peaks to expected \(f_n = n F_0 \sqrt{1 + B n^2}\) by nearest-frequency match within a tolerance of, say, ±50 cents.

For initial note detection at startup, use `librosa.piptrack` or `aubio` YIN — both are good enough and one-liners.

### 3.6 Note recognition (auto note switching)

Modes (match PianoMeter conventions):

- **Auto**: detect any played note in a configurable cent window around expected ET frequencies.
- **Stepwise**: only allow ±1 semitone jumps from the locked note; prevents accidental octave jumps under sustain.
- **Lock**: manual only.

Algorithm: compute the **summed cent-binned spectrum** of the last ~0.5 s, cross-correlate against a templated partial-comb \(\sum_n e^{-(m - m(f_n))^2/2\sigma^2}\) for each candidate key in the search window. Pick the argmax above a confidence threshold. Cost: 88 dot products on ~12k-element arrays; negligible.

### 3.7 Strobe display

Same idea as a Peterson Strobe Tuner. For the currently-tracked note's fundamental (and optionally each partial as a separate ring), bandpass-filter the audio around \(f_n\), then compute the phase \(\phi(t)\) via a single-bin DFT or via the analytic signal (Hilbert transform on a narrow band). Beat against the target frequency:

\[
\theta(t) = 2\pi (f_n^{\text{meas}} - f_n^{\text{target}})\, t
\]

Render as a rotating disk: when in tune the disk stands still; clockwise = sharp, counterclockwise = flat. Use a `QQuickItem` or a custom `QWidget` with `QPainter`; update at 60 Hz from a `QTimer`.

---

## 4. Architecture (PyQt6)

### 4.1 Process & thread model

| Thread | Responsibility |
|---|---|
| **Main / GUI** | PyQt6 event loop, all widget rendering, user input. |
| **Audio (PortAudio RT)** | Owned by sounddevice; copies samples into ring buffer only. |
| **STFT worker** (`QThread`) | Reads ring buffer, windows + FFTs, computes cent-binned spectrum, emits `frame_ready(spectrum, time)` signal. |
| **Analysis worker** (`QThread`) | Note recognition, peak picking, B estimation. Emits `note_detected(key, cents)`, `partials_updated(...)`. |
| **Solver worker** (`QThread`) | Long-running tuning-curve computation (entropy MC or beat-rate LS). Cancellable. Emits `solver_progress`, `solver_done`. |

Cross-thread communication: PyQt6 signals only. No shared mutable state. The ring buffer is the one exception and uses `multiprocessing.shared_memory` or a `numpy` view with a `threading.Lock` for the head pointer.

### 4.2 Package layout

```
pianoforge/                              # top-level package
├── audio/
│   ├── capture.py                       # sounddevice InputStream wrapper
│   ├── ringbuffer.py
│   └── playback.py                      # for reference tones / strobe target
├── dsp/
│   ├── stft.py                          # pyfftw, windows
│   ├── binning.py                       # log-cent binning + A-weighting
│   ├── peaks.py                         # parabolic interpolation
│   ├── note_recognizer.py
│   └── strobe.py                        # phase extraction
├── model/
│   ├── piano.py                         # Piano, Key, Keyboard classes
│   ├── temperaments.py                  # ET, Werckmeister, Kirnberger, ...
│   ├── inharmonicity.py                 # B-curve fit & interpolation
│   └── tuning_curve.py                  # final cent offsets per key
├── solvers/
│   ├── base.py                          # abstract Solver protocol
│   ├── entropy.py                       # Hinrichsen full-keyboard MC
│   ├── entropy_octave.py                # Szwajcowski–Pilch per-octave
│   ├── beat_rate.py                     # weighted-interval least-squares
│   └── nmf_b_estimator.py               # Rigaud (offline, high accuracy)
├── ui/
│   ├── main_window.py
│   ├── widgets/
│   │   ├── strobe_widget.py
│   │   ├── spectrum_widget.py           # PyQtGraph
│   │   ├── railsback_widget.py
│   │   ├── b_curve_widget.py
│   │   ├── keyboard_widget.py
│   │   └── cents_dial.py
│   └── dialogs/
│       ├── pitch_raise.py
│       └── temperament_picker.py
├── persistence/
│   ├── tuning_file.py                   # .pfg / .ept-compatible XML
│   └── settings.py                      # QSettings wrapper
└── __main__.py
```

Solvers implement a `Solver` protocol so the user can swap them at runtime — this is the architectural advantage over both pianoscope and EPT, which each ship one algorithm.

### 4.3 Solver protocol

```python
from typing import Protocol, Iterator
import numpy as np

class Solver(Protocol):
    name: str

    def solve(
        self,
        cent_spectra: np.ndarray,       # shape (K, M) — A-weighted SPLA per key
        b_estimates: np.ndarray,        # shape (K,) — measured B per key (NaN if missing)
        constraints: 'TuningConstraints',
    ) -> Iterator['TuningCurve']:
        """Yield intermediate tuning curves so the GUI can show progress."""
```

`TuningConstraints` carries: A4 frequency, temperament, locked notes, interval weights, treble rule choice (1:2 partial match by default, per Shah & Välimäki). `TuningCurve` is a `(88,)` array of cent offsets relative to ET, plus metadata.

### 4.4 File format

Use XML with the same outer structure as EPT's `.ept` files so existing EPT recordings can be imported. The schema is straightforward — `<piano>`, `<keyboard>`, `<key index="">` with attributes for measured \(B\), measured frequencies of the first ~8 partials, and computed cent offset. Add a `<spectrum>` element holding the base64-zipped cent-binned SPLA so the entropy solver can be re-run later without re-recording. Validate against an XSD shipped in the package.

---

## 5. The entropy solver — concrete implementation

This is the algorithm with the most non-obvious engineering. Pseudocode that follows the paper exactly:

```python
def entropy_solve(L, max_passes=20, step_cents=1, rng=None):
    """
    L  : ndarray (K=88, M=12000) — A-weighted SPLA per key, cent-binned.
    Returns: shifts (K,) — integer cent shifts to apply to each key.
    """
    K, M = L.shape
    shifts = initial_shifts_equal_temperament(K)   # array of ints, all 0 initially

    def shifted_sum():
        # roll each row by -shifts[k] and sum; this is the hot path
        p = np.zeros(M, dtype=np.float64)
        for k in range(K):
            p += np.roll(L[k], -shifts[k])
        return p

    p = shifted_sum()
    p_norm = p / p.sum()
    H = -np.sum(p_norm * np.log(p_norm + 1e-30))

    no_accepts_in_a_row = 0
    while no_accepts_in_a_row < K:
        k = rng.integers(K)
        delta = rng.choice([-step_cents, +step_cents])

        # incremental update: subtract old row, add new shifted row
        p_new = p - np.roll(L[k], -shifts[k]) + np.roll(L[k], -(shifts[k] + delta))
        p_norm = p_new / p_new.sum()
        H_new = -np.sum(p_norm * np.log(p_norm + 1e-30))

        if H_new < H:
            shifts[k] += delta
            p = p_new
            H = H_new
            no_accepts_in_a_row = 0
        else:
            no_accepts_in_a_row += 1

    return shifts
```

**Performance.** Naïve `np.roll` on a 12000-element array is cheap; with K=88 we do at most ~milliseconds per attempt. Hinrichsen's reference implementation in EPT is fast enough on a 2010 laptop; on modern hardware this loop is bound by Python overhead. Three optimizations:

1. **Pre-shift via index arithmetic.** Instead of `np.roll`, maintain `p` and update only the two diff regions: subtract `L[k]` at positions `[s..s+M]` and add at `[s+δ..s+δ+M]` (with wraparound). Reduces the per-step cost to `O(1)` in effect — entire sweep < 100 ms.
2. **Vectorize candidate shifts.** Try all 88 × 2 perturbations in one pass and accept the best — converts the MC to a coordinate-descent that may converge faster but loses some of the stochasticity that avoids local minima.
3. **Numba JIT** the inner loop if pure NumPy isn't fast enough.

**Determinism.** Seed `rng = np.random.default_rng(seed)` from user setting. Allow the user to re-run with different seeds and average the resulting tuning curves — this is the practical workaround for the local-minimum problem Hinrichsen acknowledges in §6.

**Stop criterion.** "No improvement after K consecutive trials" is the paper's. In practice, augment with `H_new < H - eps` to avoid numerical jitter, and a hard cap on passes.

**Simulated annealing extension.** Hinrichsen states "more advanced Monte Carlo techniques such as simulated annealing have not yet been tested" — implement classical SA with a logarithmic cooling schedule \(T_n = T_0 / \log(1 + n)\) as a v2 option. Should improve convergence to the global minimum, particularly in the bass.

---

## 6. The beat-rate solver — concrete implementation

This is the math behind pianoscope, PianoMeter, TuneLab. It is faster, deterministic, and respects user-chosen interval weights, at the cost of needing reliable \(B\) estimates first.

### 6.1 B-curve fitting

1. Estimate per-key \(B_k\) via PFD (§2.6) on the recording.
2. Fit a 2-segment log-linear model in MIDI \(m\):

\[
\log B(m) = \begin{cases}
s_B^{\text{bass}}\, m + y_B^{\text{bass}}, & m < m_{\text{break}} \\
s_B^{\text{treble}}\, m + y_B^{\text{treble}}, & m \ge m_{\text{break}}
\end{cases}
\]

with a smooth join. Use Rigaud's L1 regression on \(\log B(m)\) for outlier resistance. Initial values: \(s_B = 8.9 \times 10^{-2}\), \(y_B = -7\). Missing keys are filled by evaluating the fit.

### 6.2 Tuning-curve optimization

Given the per-key partial frequencies \(f^{(k)}_n = n F^{(k)}_0 \sqrt{1 + B_k n^2}\), define an interval as a pair of partials \((k_1, n_1, k_2, n_2)\) that should beat slowly (e.g. 2:1 octave = \(k_2 = k_1+12, n_2 = 1, n_1 = 2\); 4:2 octave = same keys, \(n_1=4, n_2=2\); 3:1 twelfth, 6:3 octave, etc.).

The beat rate is

\[
\beta_{(k_1,n_1,k_2,n_2)} \;=\; n_2\, F^{(k_2)}_0\sqrt{1+B_{k_2}n_2^2} \cdot 2^{c_{k_2}/1200} \;-\; n_1\, F^{(k_1)}_0\sqrt{1+B_{k_1}n_1^2} \cdot 2^{c_{k_1}/1200}
\]

where \(c_k\) is the unknown cent offset of key \(k\). Solve

\[
\min_{\mathbf{c}} \;\; \sum_{(\text{intervals})} w_i\, \beta_i^2 \;+\; \mu \sum_k (c_k - c_k^{\text{temperament}})^2
\]

via weighted least squares. Weights \(w_i\) are user-exposed and define the "tuning style" — exactly as in [PianoMeter's interval weight UI](https://pianometer.com/support/). The regularizer pins offsets to the chosen temperament base values (ET, Werckmeister III, Kirnberger III, Vallotti, Young, Valotti, Meantone — all loadable from `temperaments.py` as cent offsets from ET).

Boundary conditions:
- **A4 = 440 Hz** (or user-set) — locks \(c_{69}\) to 0.
- **Top octave**: apply Shah & Välimäki's 1:2 rule by giving \(n_1=2, n_2=1\) interval enormous weight for \(k \ge 81\).
- **Bottom octave**: Rigaud's "adaptive bass stretch" — increase weight on 6:3 vs 4:2 for short-scale (high-B) bass strings; detect short scale by mean bass \(B\).

Solution: linearize \(2^{c/1200} \approx 1 + c \ln 2 / 1200\) around current iterate; the resulting linear system is sparse, solve with `scipy.sparse.linalg.lsmr`. Iterate 3–5 times. Converges in well under a second.

### 6.3 Octave-local entropy (Szwajcowski-Pilch)

Hybrid worth implementing: within each octave the cent shifts of the upper key are chosen by entropy minimization on the local two-key spectrum (one variable, ~50 trial cents), going either upward or downward from A4. Deterministic, fast, and good results — particularly for users who distrust interval-weight choices. ~100 lines of code given the entropy machinery from §5.

---

## 7. User-facing features (parity table)

| Feature | pianoscope | PianoMeter | EPT | **This spec** |
|---|---|---|---|---|
| Strobe display | ✓ | ✓ | ✓ | ✓ |
| Cents needle | ✓ | ✓ | ✓ | ✓ |
| Spectrum view | – | ✓ | ✓ | ✓ |
| Tuning curve graph | ✓ | ✓ | ✓ | ✓ |
| B-curve graph | – | ✓ | ✓ | ✓ |
| Auto note switching | ✓ | ✓ | ✓ | ✓ |
| Stepwise / Lock modes | ✓ | ✓ | – | ✓ |
| Pitch raise / overpull | ✓ | ✓ pro | – | ✓ |
| Historical temperaments | ✓ pro | ✓ | – | ✓ |
| A ≠ 440 | ✓ | ✓ | ✓ | ✓ |
| Save / load tuning files | ✓ pro | ✓ pro | ✓ | ✓ |
| Multiple solver algorithms | – | – | – | **✓** (entropy / beat-rate / hybrid) |
| Open source | – | – | ✓ | **✓ (GPL-3)** |
| Linux native | – | – | (broken) | **✓** |

The bolded rows are the differentiators that make this project worth building rather than waiting for someone to fix EPT.

---

## 8. Workflow

1. **New piano**: user names it, picks A4 and temperament. Defaults: 440 Hz, ET.
2. **Recording pass**: user plays each note across the keyboard once. For each key:
   - STFT + peak picker computes initial \(F_0, B_k\) within ~500 ms of sustained sound.
   - The cent-binned A-weighted spectrum is stored in `Key.cent_spectrum`.
   - Visual progress: keyboard widget paints recorded keys in green.
3. **Solve**: user picks the solver. Computation runs in the solver thread; intermediate `TuningCurve`s stream into the Railsback widget.
4. **Optional**: lock the curve, do a pitch-raise pre-pass if the piano is >10 cents flat (use Rigaud's mean octave-type model with high/low variants as the overpull profile).
5. **Tune**: in tuning mode, the strobe + cents widget guides the user note-by-note. Manual switching via keyboard click; stepwise mode prevents octave jumps.
6. **Save**: `.pfg` file containing piano metadata, B-curve, spectra (base64-zipped), and final tuning curve.

---

## 9. Reference implementations to study

| Project | What to take | What to avoid |
|---|---|---|
| [Entropy Piano Tuner](https://gitlab.com/tp3/Entropy-Piano-Tuner) (C++/Qt5) | Architecture: `modules/core/{analyzers,audio,calculation,math,piano}` mirrors the layout in §4.2. Especially [`fftanalyzer.cpp`](https://gitlab.com/tp3/Entropy-Piano-Tuner/-/tree/master/modules/core/analyzers), [`keyrecognizer.cpp`](https://gitlab.com/tp3/Entropy-Piano-Tuner/-/tree/master/modules/core/analyzers), [`signalanalyzer.cpp`](https://gitlab.com/tp3/Entropy-Piano-Tuner/-/tree/master/modules/core/analyzers), [`stroboscope.cpp`](https://gitlab.com/tp3/Entropy-Piano-Tuner/-/tree/master/modules/core/audio/recorder), and [`modules/algorithms/entropyminimizer/`](https://gitlab.com/tp3/Entropy-Piano-Tuner/-/tree/master/modules/algorithms). | Qt5/qmake, the qwt dependency (use PyQtGraph instead), the broken plugin loader. |
| [beiciliang/estimate-f0-inharmonicity](https://github.com/beiciliang/estimate-f0-inharmonicity) | Direct port of Rigaud NMF algorithm to Python. Drop into `solvers/nmf_b_estimator.py` almost as-is. | – |
| [RobertBoganKang/piano_tuning](https://github.com/RobertBoganKang/piano_tuning) | Mathematica notebook implementing both a traditional (TuneLab-style) and entropy method, including a "pure-sound tuner" that compensates inharmonicity in playback. Useful as a worked example of both algorithms side by side. | Mathematica-only; not directly reusable code. |
| [Friture](https://friture.org) (Python/Qt, GPL3) | Real-time audio analyzer with scope, spectrum, octave bands, rolling spectrogram. Best public reference for the audio→DSP→GUI pipeline in Python. | Friture is a measurement tool, not a tuner — no note recognition, no tuning math. |
| [aubio](https://aubio.org) | YIN / YINFFT pitch detector, onset detection. Use for fast initial note recognition. | – |
| [librosa](https://librosa.org) | `piptrack`, CQT, salient peak finding. | Don't use for hot path — it's reference-grade but not optimized for streaming. |
| [pyfftw](https://pyfftw.readthedocs.io) | FFTW3 wrapper; same library EPT uses. | – |
| [PySDR PyQt chapter](https://pysdr.org/content/pyqt.html) | Idiomatic PyQt6 + PyQtGraph real-time spectrum analyzer. | – |
| [SMS-tools](https://github.com/MTG/sms-tools) | Reference parabolic-peak interpolation. | – |

---

## 10. Bibliography (in citation order)

1. H. Hinrichsen, "[Entropy-based tuning of musical instruments](https://arxiv.org/pdf/1203.5101)", *Rev. Bras. Ens. Fís.* 34(2) (2012). **Source for the entropy algorithm in full.**
2. F. Rigaud, B. David, L. Daudet, "[A parametric model and estimation techniques for the inharmonicity and tuning of the piano](https://www.institut-langevin.espci.fr/biblio/2020/3/5/916/files/2013_a_parametric_model_and_estimation_techniques_for_the_inharmonicity_and_tuning_of_the_piano.pdf)", *JASA* 133(5) 3107–3118 (2013). **Source for the inharmonicity model, NMF estimator, whole-compass tuning model.**
3. N. Giordano, "[Explaining the Railsback stretch in terms of the inharmonicity of piano tones and sensory dissonance](https://asa.scitation.org/doi/pdf/10.1121/1.4931439)", *JASA* 138(4) 2359–2366 (2015). **Why minimizing dissonance reproduces the Railsback curve — theoretical validation of any solver.**
4. J. Jaatinen, J. Pätynen, "[Effect of inharmonicity on pitch perception and subjective tuning of piano tones](https://pubs.aip.org/jasa/article/152/2/1146/2838401)", *JASA* 152(2) 1146 (2022). **Listener-preference validation.**
5. A. Szwajcowski, A. Pilch, "[Optimization of piano tuning by means of spectral entropy minimization](https://linkinghub.elsevier.com/retrieve/pii/S0003682X20300050)", *Applied Acoustics* 166 (2020). **Octave-local entropy variant.**
6. S. Shah, V. Välimäki, "[Automatic tuning of high piano tones](https://www.mdpi.com/2076-3417/10/6/1983)", *Applied Sciences* 10(6) 1983 (2020). **The 1:2 partial-matching rule for the top octave.**
7. J. Rauhala, H.-M. Lehtonen, V. Välimäki, "[Fast automatic inharmonicity estimation algorithm](https://pubs.aip.org/jasa/article/121/5/EL184/538552)", *JASA* 121(5) EL184 (2007). **PFD-style fast B estimator.**
8. T. Miljković et al., "[Estimation of harp string inharmonicity influenced by phantom partials](https://pubs.aip.org/jasa/article/158/4/3187/3368787)", *JASA* 158(4) 3187 (2025). **Modern refinement to PFD excluding phantom partials.**
9. F. Rigaud, A. Drémeau, B. David, L. Daudet, "[A probabilistic line spectrum model for musical instrument sounds and its application to piano tuning estimation](http://ieeexplore.ieee.org/document/6701879/)", WASPAA 2013. **Probabilistic alternative to NMF.**
10. C.-H. Kim et al., "[Joint estimation of multiple notes and inharmonicity coefficient based on f0-triplet for automatic piano transcription](http://ieeexplore.ieee.org/document/6868968/)", *IEEE SP Lett.* 21(12) (2014). **Useful for polyphonic note-detection mode.**
11. A. Galembo, A. Askenfelt, "[Signal representation and estimation of spectral parameters by inharmonic comb filters with application to the piano](http://ieeexplore.ieee.org/document/748124/)", *IEEE TSAP* 7(2) (1999). **Classical reference for inharmonic comb-filter peak picking.**
12. J. Jaatinen, J. Pätynen, K. Alho, "[Octave stretching phenomenon with complex tones of orchestral instruments](https://pubs.aip.org/jasa/article/146/5/3203/993425)", *JASA* 146(5) 3203 (2019). **Background on perceptual stretch.**

---

## 11. Milestones (suggested order)

1. **Scaffolding & audio I/O.** `sounddevice` capture → ring buffer → STFT worker → PyQtGraph spectrum widget. Verify on PipeWire. (≈ 2 days)
2. **Cent-binning + A-weighting + display.** Implement §3.4 exactly. Verify against a sine sweep. (≈ 1 day)
3. **Peak picker + parabolic interpolation + B estimator (PFD).** Verify against `beiciliang/estimate-f0-inharmonicity` outputs on identical WAVs. (≈ 2 days)
4. **Note recognizer + strobe widget.** Tune a single note end-to-end. (≈ 2 days)
5. **Piano model, recording workflow, .pfg persistence.** (≈ 2 days)
6. **Beat-rate solver** with ET. Reproduces Railsback curve on synthetic test piano. (≈ 3 days)
7. **Entropy solver (§5).** Cross-check against EPT on the same recordings. (≈ 2 days)
8. **Temperaments + pitch raise + adaptive bass stretch + Shah-Välimäki treble rule.** (≈ 3 days)
9. **NMF B-estimator (offline, port beiciliang code).** (≈ 1 day)
10. **Octave-local entropy (Szwajcowski-Pilch).** (≈ 1 day)
11. **Localization (en/de), packaging (Flatpak), docs.** (≈ 3 days)

Total: ~3 weeks of focused work for a senior engineer to reach pianoscope feature parity, plus another week for polish and the differentiating multi-solver story.

---

## 12. Open questions & v2 ideas

- **MIDI integration** for connecting to a digital piano as a reference oscillator or for driving a player-piano-actuated tuning robot ([Zhou, Wu & Wu 2021](https://www.mdpi.com/1996-1073/14/20/6627)).
- **Polyphonic mode** for tuning unisons audibly (multiple strings of one key) using a joint multi-pitch model (Kim 2014).
- **Phantom-partial rejection** (Miljković 2025) in the PFD pipeline.
- **Simulated annealing** in the entropy solver, plus consensus across multiple seeds.
- **Real-time-mode B-tracking** while you tune, so the B-curve refines itself without a dedicated recording pass.
- **Web export** of the tuning report (PDF + interactive Railsback SVG) for professional tuners.

---

*End of specification.*
