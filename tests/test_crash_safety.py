"""Crash-safety: corrupt persistence and recover to a usable state."""

from __future__ import annotations

import json
from pathlib import Path

from optitune.model import Key, Piano
from optitune.ui.main_window import OptiTuneMainWindow


def test_load_json_corrupt_returns_none(tmp_path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("{ not json at all", encoding="utf-8")
    assert Piano.load_json(p) is None


def test_load_json_partial_keys_skips_bad_entries(tmp_path) -> None:
    p = tmp_path / "partial.json"
    p.write_text(
        json.dumps(
            {
                "name": "X",
                "a4": 440,
                "keys": {
                    "60": {"midi": 60, "measured_f0": 261.6, "measured_b": 0.0003},
                    "bad": {"no_midi": True},
                    "61": "not-an-object",
                },
                "tuning_curve": None,
            }
        ),
        encoding="utf-8",
    )
    piano = Piano.load_json(p)
    assert piano is not None
    assert piano.get_key(60) is not None
    assert 61 not in piano.keys


def test_main_window_survives_corrupt_persist_path(qtbot, tmp_path, monkeypatch) -> None:
    broken = tmp_path / "current_piano.json"
    broken.write_text("[[[", encoding="utf-8")
    monkeypatch.setattr(Piano, "default_persist_path", classmethod(lambda cls: broken))
    w = OptiTuneMainWindow(device="NonExistentDummyForTest")
    qtbot.addWidget(w)
    if w.audio_capture.is_running:
        w.audio_capture.stop()
    # Usable empty session
    assert w._piano is None or isinstance(w._piano, Piano)
    w._ensure_piano()
    assert w._piano is not None
    w._piano.set_key(Key(midi=69, measured_f0=440.0, measured_b=0.0003))
    assert w._piano.measured_count() == 1
