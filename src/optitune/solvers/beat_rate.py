"""
Weighted beat-rate least-squares tuning curve solver (Phase 4+ improvement).

Implements the iterative linearized weighted interval beat-rate minimization
from piano_tuner_implementation_spec.md §6.2:

- log-linear B-curve fit from measured keys (reuses the same approach as the
  legacy heuristic for compatibility).
- Practical set of intervals with tunable weights: 4:2/2:1/6:3/8:4 octaves
  (4:2 highest), selected 3:2 fifths, 4:3 fourths, 3:1 twelfths, etc.
- Shah & Välimäki 1:2 rule enforced as high-weight hard constraint for
  the top octave (MIDI ≳ 86).
- Linearization of the stretch factor 2^(c/1200) around the current iterate.
- Very light L2 regularizer pulling c toward a gentle default Railsback curve
  (prevents wild excursions with sparse data while still letting measurements dominate).
- Exact A4 (MIDI 69, index 48) lock at 0 cents via high-weight pin.
- 3–7 Gauss-Newton-style iterations; converges in < 20 ms.

The public `compute_basic_tuning_curve` replaces the old heuristic and is
what the GUI and `apply_curve_to_piano` use. The legacy heuristic remains
available as `compute_heuristic_stretch_curve` (from simple_stretch) for
A/B comparison, fallback, and regression tests.

Output: 88-element list of cent offsets vs. equal temperament, A4 pinned at
exactly 0, bass generally negative, treble positive, musically plausible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from optitune.model.piano import Piano
from optitune.solvers.simple_stretch import (
    _default_railsback_curve,  # for <3-measurement fallback + mild prior
)

MIDI_LOW = 21
MIDI_HIGH = 108
N_KEYS = 88
A4_MIDI = 69
A4_IDX = A4_MIDI - MIDI_LOW  # 48


def _et_f0(midi: int, a4: float = 440.0) -> float:
    """Equal-tempered fundamental for the given MIDI (no stretch)."""
    return a4 * (2.0 ** ((midi - 69) / 12.0))


def compute_beat_rate_for_interval(
    f01: float,
    b1: float,
    n1: int,
    c1: float,
    f02: float,
    b2: float,
    n2: int,
    c2: float,
) -> float:
    """
    Exact (non-linearized) beat rate β in Hz for one partial pair.

    β = n2 * F02 * sqrt(1 + B2 n2²) * 2^(c2/1200)
        - n1 * F01 * sqrt(1 + B1 n1²) * 2^(c1/1200)

    Positive ⇒ upper partial is sharp relative to lower.
    Used for quantitative evaluation of a candidate curve (see tests).
    """
    r1 = 2.0 ** (c1 / 1200.0)
    r2 = 2.0 ** (c2 / 1200.0)
    s1 = math.sqrt(1.0 + b1 * n1 * n1)
    s2 = math.sqrt(1.0 + b2 * n2 * n2)
    return n2 * f02 * s2 * r2 - n1 * f01 * s1 * r1


def compute_basic_tuning_curve(
    piano: Piano,
    *,
    interval_weights: dict[str, float] | None = None,
    reg_mu: float = 0.0008,
    max_iter: int = 7,
    shah_weight: float = 320.0,
    shah_from_midi: int = 86,
) -> list[float]:
    """
    Return 88 cent offsets (MIDI 21..108) for the given piano.

    This is the improved solver: iterative linearized weighted least-squares
    minimization of interval beat rates β (with B-driven partial frequencies)
    subject to A4 lock, Shah–Välimäki treble rule, and mild Railsback prior
    regularizer.

    When fewer than 3 measured B values are present, falls back to the
    sensible default Railsback-style curve (identical to the legacy heuristic
    fallback) so that existing behavior and tests are preserved.

    interval_weights: optional override of the default practical weighting
        (higher = more important to satisfy). Keys include:
        "octave_2_1", "octave_4_2" (highest), "octave_6_3", "octave_8_4",
        "fifth_3_2", "fourth_4_3", "twelfth_3_1", ...
    reg_mu: strength of L2 pull toward a mild default Railsback. 
    """
    # 1. Gather measured (midi, B) — identical collection logic to heuristic
    measured: list[tuple[int, float]] = []
    for k in piano.keys.values():
        if k.measured_b is not None and 1e-6 < float(k.measured_b) < 1.0:
            measured.append((k.midi, float(k.measured_b)))

    if len(measured) < 3:
        # Graceful fallback — produces exactly the same default curve the
        # old heuristic produced, preserving all prior tests & behavior.
        return _default_railsback_curve()

    # 2. Log-linear B-curve fit (exactly as before; slope/intercept on ln B)
    ms = np.asarray([m for m, _ in measured], dtype=float)
    bs = np.asarray([b for _, b in measured], dtype=float)
    log_bs = np.log(np.clip(bs, 1e-8, None))

    A_fit = np.vstack([ms, np.ones_like(ms)]).T
    sol, *_ = np.linalg.lstsq(A_fit, log_bs, rcond=None)
    slope, intercept = float(sol[0]), float(sol[1])

    # 3. Predict B for entire keyboard (smoothed, robust)
    all_midis = np.arange(MIDI_LOW, MIDI_HIGH + 1, dtype=float)
    logb_pred = slope * all_midis + intercept
    b_pred = np.exp(logb_pred)
    b_pred = np.clip(b_pred, 1e-6, 0.5)

    # ET fundamentals and weights
    a4 = float(piano.a4)
    f0_et = np.array([_et_f0(int(m), a4) for m in all_midis], dtype=float)

    if interval_weights is None:
        interval_weights = {
            "octave_2_1": 5.5,
            "octave_4_2": 28.0,   # primary audible octave control (highest weight)
            "octave_6_3": 8.5,
            "octave_8_4": 2.2,
            "fifth_3_2": 0.6,     # light — helps overall coherence without fighting stretch
            "fourth_4_3": 0.2,
            "twelfth_3_1": 0.9,
            "double_oct_4_1": 0.4,
        }

    # 4. Build the list of (i1, i2, a1, a2, w) interval equations
    inter_list: list[dict[str, Any]] = []

    # --- Octaves (every possible lower key) ---
    ow = interval_weights
    oct_variants = [
        (2, 1, ow.get("octave_2_1", 5.5)),
        (4, 2, ow.get("octave_4_2", 28.0)),
        (6, 3, ow.get("octave_6_3", 8.5)),
        (8, 4, ow.get("octave_8_4", 2.2)),
    ]
    for lm in range(MIDI_LOW, 97):
        i1 = lm - MIDI_LOW
        i2 = i1 + 12
        if i2 >= N_KEYS:
            continue
        f1, f2 = f0_et[i1], f0_et[i2]
        b1, b2 = float(b_pred[i1]), float(b_pred[i2])
        for n1, n2, ww in oct_variants:
            if ww <= 0.0:
                continue
            s1 = math.sqrt(1.0 + b1 * n1 * n1)
            s2 = math.sqrt(1.0 + b2 * n2 * n2)
            a1 = n1 * f1 * s1
            a2 = n2 * f2 * s2
            inter_list.append({"i1": i1, "i2": i2, "a1": a1, "a2": a2, "w": float(ww)})

    # --- Selected fifths (every ~2nd possible) ---
    fw = ow.get("fifth_3_2", 0.6)
    if fw > 0.0:
        for lm in range(MIDI_LOW, 102, 2):
            i1 = lm - MIDI_LOW
            i2 = i1 + 7
            if i2 >= N_KEYS:
                continue
            f1, f2 = f0_et[i1], f0_et[i2]
            b1, b2 = float(b_pred[i1]), float(b_pred[i2])
            n1, n2 = 3, 2
            s1 = math.sqrt(1.0 + b1 * n1 * n1)
            s2 = math.sqrt(1.0 + b2 * n2 * n2)
            a1 = n1 * f1 * s1
            a2 = n2 * f2 * s2
            inter_list.append({"i1": i1, "i2": i2, "a1": a1, "a2": a2, "w": float(fw)})

    # --- Selected fourths ---
    frw = ow.get("fourth_4_3", 0.2)
    if frw > 0.0:
        for lm in range(MIDI_LOW, 104, 3):
            i1 = lm - MIDI_LOW
            i2 = i1 + 5
            if i2 >= N_KEYS:
                continue
            f1, f2 = f0_et[i1], f0_et[i2]
            b1, b2 = float(b_pred[i1]), float(b_pred[i2])
            n1, n2 = 4, 3
            s1 = math.sqrt(1.0 + b1 * n1 * n1)
            s2 = math.sqrt(1.0 + b2 * n2 * n2)
            a1 = n1 * f1 * s1
            a2 = n2 * f2 * s2
            inter_list.append({"i1": i1, "i2": i2, "a1": a1, "a2": a2, "w": float(frw)})

    # --- Selected twelfths (P8+P5, 3:1) ---
    tw = ow.get("twelfth_3_1", 0.9)
    if tw > 0.0:
        for lm in range(MIDI_LOW, 90, 3):
            i1 = lm - MIDI_LOW
            i2 = i1 + 19
            if i2 >= N_KEYS:
                continue
            f1, f2 = f0_et[i1], f0_et[i2]
            b1, b2 = float(b_pred[i1]), float(b_pred[i2])
            n1, n2 = 3, 1
            s1 = math.sqrt(1.0 + b1 * n1 * n1)
            s2 = math.sqrt(1.0 + b2 * n2 * n2)
            a1 = n1 * f1 * s1
            a2 = n2 * f2 * s2
            inter_list.append({"i1": i1, "i2": i2, "a1": a1, "a2": a2, "w": float(tw)})

    # --- Occasional double-octave constraints ---
    do_w = ow.get("double_oct_4_1", 0.4)
    if do_w > 0.0:
        for lm in range(MIDI_LOW, 85, 4):
            i1 = lm - MIDI_LOW
            i2 = i1 + 24
            if i2 >= N_KEYS:
                continue
            f1, f2 = f0_et[i1], f0_et[i2]
            b1, b2 = float(b_pred[i1]), float(b_pred[i2])
            n1, n2 = 4, 1
            s1 = math.sqrt(1.0 + b1 * n1 * n1)
            s2 = math.sqrt(1.0 + b2 * n2 * n2)
            a1 = n1 * f1 * s1
            a2 = n2 * f2 * s2
            inter_list.append({"i1": i1, "i2": i2, "a1": a1, "a2": a2, "w": float(do_w)})

    # --- Shah & Välimäki 1:2 rule (high weight, top octave) ---
    for um in range(shah_from_midi, MIDI_HIGH + 1):
        i2 = um - MIDI_LOW
        i1 = i2 - 12
        if i1 < 0:
            continue
        f1, f2 = f0_et[i1], f0_et[i2]
        b1, b2 = float(b_pred[i1]), float(b_pred[i2])
        n1, n2 = 2, 1  # lower 2nd partial ≈ upper fundamental
        s1 = math.sqrt(1.0 + b1 * n1 * n1)
        s2 = math.sqrt(1.0 + b2 * n2 * n2)
        a1 = n1 * f1 * s1
        a2 = n2 * f2 * s2
        inter_list.append({"i1": i1, "i2": i2, "a1": a1, "a2": a2, "w": float(shah_weight)})

    # 5. Iterative linearized weighted LS
    N = N_KEYS
    alpha = math.log(2.0) / 1200.0

    # Mild Railsback prior (35 % of the legacy default) — excellent stabilizer
    default_prior = _default_railsback_curve()
    mild_target = np.asarray([0.35 * x for x in default_prior], dtype=float)

    c = mild_target.copy()   # start from a musically sane point (data will refine)

    for it in range(max_iter):
        r = np.power(2.0, c / 1200.0)  # current stretch factors for linearization point

        A_rows: list[np.ndarray] = []
        b_rows: list[float] = []

        # Interval equations (scaled by sqrt(w) for weighted LS)
        for inter in inter_list:
            i1 = inter["i1"]
            i2 = inter["i2"]
            a1 = inter["a1"]
            a2 = inter["a2"]
            ww = float(inter["w"])
            if ww <= 0.0:
                continue
            sw = math.sqrt(ww)
            r1 = r[i1]
            r2 = r[i2]

            # β_lin(d) ≈ (a2*r2 - a1*r1) + α*(a2*r2 * d2 - a1*r1 * d1)
            # We want β_lin ≈ 0  ⇒  coef1*d1 + coef2*d2 ≈ -(a2*r2 - a1*r1)
            coef1 = -alpha * a1 * r1
            coef2 = alpha * a2 * r2
            rhs = -(a2 * r2 - a1 * r1)

            row = np.zeros(N, dtype=float)
            row[i1] = coef1
            row[i2] = coef2
            A_rows.append(row * sw)
            b_rows.append(rhs * sw)

        # Regularizer toward *mild_target* (not zero) on total c = c_cur + d
        #   sqrt(mu) * (c_k + d_k - t_k) ≈ 0
        sqrt_mu = math.sqrt(reg_mu)
        for k in range(N):
            if k == A4_IDX:
                continue  # pinned exactly below
            row = np.zeros(N, dtype=float)
            row[k] = sqrt_mu
            A_rows.append(row)
            b_rows.append(-sqrt_mu * (c[k] - mild_target[k]))

        # Hard A4 pin: (c_A4 + d_A4) ≈ 0 with enormous weight
        pin_w = 1.0e7
        row = np.zeros(N, dtype=float)
        row[A4_IDX] = pin_w
        A_rows.append(row)
        b_rows.append(-pin_w * c[A4_IDX])

        if not A_rows:
            break

        Amat = np.vstack(A_rows)
        bvec = np.asarray(b_rows, dtype=float)

        d, *_ = np.linalg.lstsq(Amat, bvec, rcond=None)
        c = c + d

        if np.max(np.abs(d)) < 0.005:  # sub-cent convergence
            break

    # 6. Final polishing & guarantees (identical spirit to legacy)
    c = np.clip(c, -32.0, 32.0)
    c[A4_IDX] = 0.0

    # Treble safeguard (Shah intent) — only lift if solver left it too low
    for i in range(72, N_KEYS):
        if c[i] < 0.3:
            c[i] = 0.3 + 0.45 * (i - 72)

    c = np.clip(c, -32.0, 32.0)
    c[A4_IDX] = 0.0

    return [float(x) for x in c]


def apply_curve_to_piano(piano: Piano, curve: Sequence[float] | None = None) -> None:
    """Attach (or recompute via the improved solver) the curve and per-key targets."""
    if curve is None:
        curve = compute_basic_tuning_curve(piano)
    piano.tuning_curve = list(curve)
    for k in piano.keys.values():
        k.target_offset_cents = piano.get_target_offset(k.midi)
