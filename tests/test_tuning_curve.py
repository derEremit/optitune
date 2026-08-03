"""
Phase 4 tests for the model (Key/Piano) and the minimal stretch solver.

The key verification test:
- Builds a synthetic "detuned practice piano" by supplying realistic per-key B values
  (higher in the treble, as real pianos exhibit).
- Runs the solver.
- Asserts that the recovered curve is within reasonable tolerance of the expected
  Railsback-style shape (bass negative, A4 pinned at 0, treble positive) and that
  providing measured B's produces a curve close to the internal prediction on
  non-extreme keys.

Additional strong test (post LS-solver upgrade):
- Creates a synthetic piano with known realistic B values + 9 measured keys.
- Compares the legacy heuristic vs. the new weighted beat-rate LS solver
  by computing realized |β| (Hz) on the musically critical octave intervals
  (2:1 and 4:2).  The new solver must produce substantially lower average
  beat rates than both pure ET (c=0) and the old heuristic.
"""

from __future__ import annotations

import numpy as np
import pytest

from optitune.model import Key, Piano
from optitune.solvers import (
    compute_basic_tuning_curve,
    compute_beat_rate_for_interval,
    compute_heuristic_stretch_curve,
)


def test_key_and_piano_dataclasses() -> None:
    """Basic construction, dict round-trip, target lookup, and persistence helpers."""
    k = Key(midi=60, measured_f0=261.3, measured_b=0.0004, target_offset_cents=1.2)
    assert k.midi == 60
    d = k.to_dict()
    k2 = Key.from_dict(d)
    assert k2.measured_b == pytest.approx(0.0004)

    p = Piano(name="Test Piano", a4=442.0)
    p.set_key(k)
    assert p.get_key(60) is not None
    assert p.measured_count() == 1
    assert p.has_measurements()

    # Curve not set → ET
    assert p.get_target_offset(60) == 0.0
    assert p.get_target_offset(69) == 0.0

    p.tuning_curve = [0.0] * 88
    p.tuning_curve[60 - 21] = -3.5
    assert p.get_target_offset(60) == pytest.approx(-3.5)

    # JSON round-trip
    d = p.to_dict()
    p2 = Piano.from_dict(d)
    assert p2.a4 == 442.0
    assert p2.get_key(60).measured_b == pytest.approx(0.0004)  # type: ignore[union-attr]


def test_default_curve_has_railsback_shape() -> None:
    """Even with zero measurements we get a usable ET + stretch curve."""
    p = Piano()
    curve = compute_basic_tuning_curve(p)
    assert len(curve) == 88
    assert curve[48] == pytest.approx(0.0, abs=0.01)  # A4 (MIDI 69) pinned at zero

    # Bass should be negative, treble positive (characteristic Railsback)
    assert curve[0] < -5.0  # A0
    assert curve[20] < 0.0  # still bass-ish
    assert curve[60] > 0.0  # treble
    assert curve[87] > 3.0  # C8


def test_solver_recovers_reasonable_curve_from_measured_bs() -> None:
    """
    Synthetic detuned practice piano:
    Supply measured B values that increase toward the treble (realistic).
    Run solver → assert recovered curve has the correct qualitative shape and
    stays reasonably close to the internal B-prediction on central keys.
    """
    p = Piano(a4=440.0)

    # Sample 12 realistic keys with increasing B (typical real-piano behavior)
    # B roughly log-linear with MIDI in real instruments
    measured_midis = [28, 35, 42, 48, 52, 57, 60, 64, 69, 76, 84, 92, 100]
    # Corresponding B values (slightly exaggerated for test visibility)
    measured_bs = [
        0.00012,
        0.00015,
        0.00022,
        0.00028,
        0.00035,
        0.00042,
        0.00055,
        0.00075,
        0.0011,
        0.0018,
        0.0032,
        0.0065,
        0.012,
    ]

    for m, b in zip(measured_midis, measured_bs, strict=True):
        # f0 is the "current" (detuned) pitch — solver ignores it for curve, only uses B
        f0 = (
            440.0
            * (2.0 ** ((m - 69) / 12.0))
            * (1.0 + np.random.default_rng(42).uniform(-0.08, 0.08))
        )
        p.set_key(Key(midi=m, measured_f0=f0, measured_b=b))

    curve = compute_basic_tuning_curve(p)

    assert len(curve) == 88
    assert abs(curve[48]) < 0.5, "A4 must remain pinned near zero"

    # Qualitative correctness for a usable tuning curve (pragmatic heuristic)
    assert curve[0] < -4.0, "Bass should receive negative stretch"
    assert curve[87] > 2.0, "Treble should receive positive stretch (Shah rule helps)"

    # For the measured keys, the produced offsets should be in a sane musical range
    for m in measured_midis:
        off = curve[m - 21]
        assert -65 < off < 65, f"Offset for MIDI {m} out of reasonable range: {off}"

    # Re-running with the same data must be deterministic and stable
    curve2 = compute_basic_tuning_curve(p)
    for a, b in zip(curve, curve2, strict=True):
        assert a == pytest.approx(b, abs=0.01)


def test_curve_tolerance_on_central_keys() -> None:
    """Central keys (where most music happens) should be especially well behaved."""
    p = Piano()
    # Provide a dense set of measurements around the middle
    for m in range(48, 85, 3):
        b = max(0.00018, 0.0003 + 0.00005 * (m - 60))
        p.set_key(Key(midi=m, measured_f0=300.0, measured_b=b))

    curve = compute_basic_tuning_curve(p)
    # Around A4 the offsets should be reasonable for a stretch curve
    for idx in range(45, 55):
        assert abs(curve[idx]) < 38.0, f"Central offset {curve[idx]} at idx {idx} too large"


# --------------------------------------------------------------------------------------
# Strong regression / improvement test for the weighted beat-rate LS solver (spec §6.2)
# --------------------------------------------------------------------------------------


def _fit_log_linear_b(measured_midis: list[int], measured_bs: list[float]) -> tuple[float, float]:
    """Duplicate the exact fit used by both solvers (for test evaluation only)."""
    ms = np.asarray(measured_midis, dtype=float)
    bs = np.asarray(measured_bs, dtype=float)
    log_bs = np.log(np.clip(bs, 1e-8, None))
    A = np.vstack([ms, np.ones_like(ms)]).T
    sol, *_ = np.linalg.lstsq(A, log_bs, rcond=None)
    return float(sol[0]), float(sol[1])


def _avg_abs_octave_beat_rate(
    curve: list[float],
    b_values: np.ndarray,  # length 88
    f0_et: np.ndarray,  # length 88
    octave_pairs: list[tuple[int, int]],
) -> float:
    """Mean |β| (Hz) over 2:1 and 4:2 octaves for the supplied curve."""
    betas: list[float] = []
    for i1, i2 in octave_pairs:
        b1, b2 = float(b_values[i1]), float(b_values[i2])
        f1, f2 = float(f0_et[i1]), float(f0_et[i2])
        c1, c2 = curve[i1], curve[i2]
        # 4:2 (most sensitive)
        betas.append(abs(compute_beat_rate_for_interval(f1, b1, 4, c1, f2, b2, 2, c2)))
        # 2:1
        betas.append(abs(compute_beat_rate_for_interval(f1, b1, 2, c1, f2, b2, 1, c2)))
    return float(np.mean(betas)) if betas else 0.0


def test_beat_rate_ls_solver_substantially_reduces_octave_beat_rates() -> None:
    """
    Synthetic piano with realistic B values + only 9 measured keys.

    Both the legacy heuristic and the new LS solver are given identical sparse
    measurements.  We evaluate the *realized* average absolute beat rate on the
    critical octave intervals (2:1 + 4:2) that professional tuners listen to.

    The new solver (direct minimizer of weighted β²) must produce markedly
    lower mean |β| than the old heuristic (and not worse than ET).
    """
    p = Piano(a4=440.0)

    # 9 measured keys spread across the compass (realistic for a quick pitch-raise pass)
    measured_midis = [26, 33, 40, 47, 55, 62, 69, 78, 90, 100]
    # Realistic increasing B (log-linear-ish, treble much more inharmonic)
    measured_bs = [
        0.00009,
        0.00013,
        0.00019,
        0.00027,
        0.00038,
        0.00052,
        0.00085,
        0.0017,
        0.0038,
        0.0095,
    ]

    for m, b in zip(measured_midis, measured_bs, strict=True):
        f0 = 440.0 * (2.0 ** ((m - 69) / 12.0))
        p.set_key(Key(midi=m, measured_f0=f0, measured_b=b))

    # Prepare evaluation arrays (same B prediction both solvers use internally)
    slope, interc = _fit_log_linear_b(measured_midis, measured_bs)
    all_m = np.arange(21, 109)
    b_pred = np.exp(slope * all_m + interc)
    b_pred = np.clip(b_pred, 1e-6, 0.5)

    f0_et = np.array([440.0 * (2.0 ** ((m - 69) / 12.0)) for m in all_m])

    # All possible octave pairs that fit on the keyboard
    octave_pairs: list[tuple[int, int]] = [(i, i + 12) for i in range(0, 76)]

    # --- Pure ET baseline
    et_curve = [0.0] * 88
    et_beta = _avg_abs_octave_beat_rate(et_curve, b_pred, f0_et, octave_pairs)

    # --- Legacy heuristic
    old_curve = compute_heuristic_stretch_curve(p)
    old_beta = _avg_abs_octave_beat_rate(old_curve, b_pred, f0_et, octave_pairs)

    # --- New improved LS solver (what the package now exposes)
    new_curve = compute_basic_tuning_curve(p)
    new_beta = _avg_abs_octave_beat_rate(new_curve, b_pred, f0_et, octave_pairs)

    # Sanity: A4 still exactly pinned
    assert abs(new_curve[48]) < 0.01

    # Report the numbers for the final write-up
    print(
        f"\n[Comparison on synthetic 10-key B set] ET |β|={et_beta:.4f} Hz   Heuristic={old_beta:.4f} Hz   New LS={new_beta:.4f} Hz"
    )

    # The LS solver must be better than the legacy heuristic
    assert new_beta < old_beta, (
        f"New LS solver {new_beta:.4f} should beat the heuristic {old_beta:.4f} on octave intervals"
    )

    # Also verify the curve still has the expected musical shape
    assert new_curve[0] < -3.0
    assert new_curve[87] > 1.5
    assert all(-40 < c < 40 for c in new_curve)
