"""Round-trip .pfg tuning files."""

from __future__ import annotations

import numpy as np
import pytest

from optitune.dsp.binning import N_BINS
from optitune.model import Key, Piano
from optitune.persistence.tuning_file import load_pfg, save_pfg


def test_pfg_roundtrip(tmp_path) -> None:
    p = Piano(name="Studio", a4=442.0)
    L = np.zeros(N_BINS, dtype=np.float32)
    L[1000:1005] = 0.5
    p.set_key(Key(midi=60, measured_f0=261.6, measured_b=0.0004, cent_spectrum=L, target_offset_cents=-1.5))
    p.set_key(Key(midi=69, measured_f0=442.0, measured_b=0.0003, target_offset_cents=0.0))
    p.tuning_curve = [0.0] * 88
    p.tuning_curve[60 - 21] = -1.5

    path = tmp_path / "studio.pfg"
    save_pfg(p, path, temperament="werckmeister_iii")
    assert path.exists()
    assert "<piano" in path.read_text()

    p2, meta = load_pfg(path)
    assert p2.name == "Studio"
    assert p2.a4 == pytest.approx(442.0)
    assert meta.get("temperament") == "werckmeister_iii"
    k60 = p2.get_key(60)
    assert k60 is not None
    assert k60.measured_b == pytest.approx(0.0004)
    assert k60.measured_f0 == pytest.approx(261.6)
    assert k60.cent_spectrum is not None
    assert k60.cent_spectrum[1000] == pytest.approx(0.5, abs=1e-5)
    assert p2.tuning_curve is not None
    assert p2.tuning_curve[60 - 21] == pytest.approx(-1.5)
