"""AppSettings typed QSettings facade."""

from __future__ import annotations

from PySide6.QtCore import QSettings

from optitune.persistence.settings import AppSettings


def test_audio_device_roundtrip(tmp_path, monkeypatch) -> None:
    # Isolate QSettings from user config
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    s = AppSettings(QSettings(str(tmp_path / "test.ini"), QSettings.Format.IniFormat))
    assert s.get_last_input_device_index() is None
    s.set_last_input_device_index(3)
    assert s.get_last_input_device_index() == 3


def test_scale_session_clear(tmp_path) -> None:
    s = AppSettings(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    s.set_scale_session(active_pitch_class=0, last_recorded_midi=36, armed_midi=48)
    d = s.get_scale_session()
    assert d["active_pitch_class"] == 0
    assert d["armed_midi"] == 48
    s.clear_scale_session()
    d2 = s.get_scale_session()
    assert d2["active_pitch_class"] is None


def test_recent_files(tmp_path) -> None:
    s = AppSettings(QSettings(str(tmp_path / "r.ini"), QSettings.Format.IniFormat))
    s.add_recent_file("/a.pfg")
    s.add_recent_file("/b.pfg")
    s.add_recent_file("/a.pfg")  # moves to front
    files = s.get_recent_files()
    assert files[0] == "/a.pfg"
    assert files[1] == "/b.pfg"
