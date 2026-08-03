"""
Centralized QSettings accessors (typed defaults in one place).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSettings


class AppSettings:
    """Thin typed facade over QSettings for OptiTune."""

    ORG = "OptiTune"
    APP = "OptiTune"

    def __init__(self, settings: QSettings | None = None) -> None:
        # Default QSettings() respects QCoreApplication org/app when set
        # (matches bare QSettings() used by tests and older call sites).
        self._s = settings or QSettings()

    @property
    def raw(self) -> QSettings:
        return self._s

    # ---- audio ----
    def get_last_input_device_index(self) -> int | None:
        if not self._s.contains("audio/last_input_device_index"):
            return None
        v = self._s.value("audio/last_input_device_index")
        try:
            return int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def set_last_input_device_index(self, index: int) -> None:
        self._s.setValue("audio/last_input_device_index", int(index))
        self._s.sync()

    # ---- scale session (crash resume) ----
    def get_scale_session(self) -> dict[str, int | None]:
        out: dict[str, int | None] = {
            "active_pitch_class": None,
            "last_recorded_midi": None,
            "armed_midi": None,
        }
        if self._s.contains("scale/active_pitch_class"):
            try:
                out["active_pitch_class"] = int(self._s.value("scale/active_pitch_class"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
        if self._s.contains("scale/last_recorded_midi"):
            try:
                out["last_recorded_midi"] = int(self._s.value("scale/last_recorded_midi"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
        if self._s.contains("scale/armed_midi"):
            try:
                out["armed_midi"] = int(self._s.value("scale/armed_midi"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
        return out

    def set_scale_session(
        self,
        *,
        active_pitch_class: int | None,
        last_recorded_midi: int | None,
        armed_midi: int | None,
    ) -> None:
        if active_pitch_class is None:
            self.clear_scale_session()
            return
        self._s.setValue("scale/active_pitch_class", int(active_pitch_class))
        if last_recorded_midi is not None:
            self._s.setValue("scale/last_recorded_midi", int(last_recorded_midi))
        else:
            self._s.remove("scale/last_recorded_midi")
        if armed_midi is not None:
            self._s.setValue("scale/armed_midi", int(armed_midi))
        else:
            self._s.remove("scale/armed_midi")
        self._s.sync()

    def clear_scale_session(self) -> None:
        self._s.remove("scale/active_pitch_class")
        self._s.remove("scale/last_recorded_midi")
        self._s.remove("scale/armed_midi")
        self._s.sync()

    # ---- tuning preferences ----
    def get_temperament(self, default: str = "equal") -> str:
        v = self._s.value("tuning/temperament", default)
        return str(v) if v else default

    def set_temperament(self, name: str) -> None:
        self._s.setValue("tuning/temperament", str(name))
        self._s.sync()

    def get_a4(self, default: float = 440.0) -> float:
        v = self._s.value("tuning/a4", default)
        try:
            return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    def set_a4(self, a4: float) -> None:
        self._s.setValue("tuning/a4", float(a4))
        self._s.sync()

    def get_recent_files(self, limit: int = 10) -> list[str]:
        raw = self._s.value("files/recent", [])
        if not isinstance(raw, list):
            return []
        return [str(x) for x in raw][:limit]

    def add_recent_file(self, path: str, *, limit: int = 10) -> None:
        files = self.get_recent_files(limit=limit)
        path = str(path)
        if path in files:
            files.remove(path)
        files.insert(0, path)
        self._s.setValue("files/recent", files[:limit])
        self._s.sync()

    def value(self, key: str, default: Any = None) -> Any:
        return self._s.value(key, default)

    def set_value(self, key: str, value: Any) -> None:
        self._s.setValue(key, value)
        self._s.sync()
