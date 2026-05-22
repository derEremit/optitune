"""
Audio device enumeration and helpers for OptiTune (Phase 2).

Wraps sounddevice.query_devices() / query_hostapis() with friendly formatting
and input-device filtering. Used by DeviceSelectorDialog and main capture wiring.
"""

from __future__ import annotations

from typing import Any

import sounddevice as sd


def list_input_devices() -> list[dict[str, Any]]:
    """
    Return list of input-capable devices with enriched info.

    Each dict has:
      index: int (original device index)
      name: str
      hostapi: str (friendly, e.g. "PipeWire", "ALSA")
      default_samplerate: float
      input_latency: float (default_low_input_latency)
      max_channels: int
      is_default: bool
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    default_in = sd.default.device[0] if sd.default.device is not None else None

    inputs: list[dict[str, Any]] = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        hostapi_idx = d.get("hostapi", 0)
        hostapi_name = hostapis[hostapi_idx]["name"] if 0 <= hostapi_idx < len(hostapis) else "Unknown"
        # Clean common hostapi names
        if "PipeWire" in hostapi_name:
            hostapi_name = "PipeWire"
        elif "Pulse" in hostapi_name:
            hostapi_name = "PulseAudio"
        elif "ALSA" in hostapi_name:
            hostapi_name = "ALSA"
        elif "JACK" in hostapi_name:
            hostapi_name = "JACK"

        inputs.append(
            {
                "index": idx,
                "name": str(d.get("name", f"Device {idx}")),
                "hostapi": hostapi_name,
                "default_samplerate": float(d.get("default_samplerate", 48000.0)),
                "input_latency": float(d.get("default_low_input_latency", 0.01)),
                "max_channels": int(d.get("max_input_channels", 1)),
                "is_default": idx == default_in,
            }
        )
    return inputs


def get_device_display_name(device: int | str | None) -> str:
    """Return a short friendly label for a device index or name, or '(none)'."""
    if device is None:
        return "(none selected)"
    try:
        if isinstance(device, str):
            # Try to resolve by name substring
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                if device.lower() in str(d.get("name", "")).lower():
                    device = i
                    break
            else:
                return str(device)
        info = sd.query_devices(device)
        name = str(info.get("name", device))
        # truncate long names
        if len(name) > 42:
            name = name[:39] + "…"
        return f"{device}: {name}" if isinstance(device, int) else name
    except Exception:
        return str(device)


def resolve_device_index(device: int | str | None) -> int | None:
    """Best-effort convert device spec (index or name) to a valid input index, or None."""
    if device is None:
        return None
    try:
        inputs = list_input_devices()
        if isinstance(device, int):
            if any(d["index"] == device for d in inputs):
                return device
            return None
        # string: match name contains
        dev_lower = str(device).lower()
        for d in inputs:
            if dev_lower in d["name"].lower():
                return d["index"]  # type: ignore[no-any-return]
        return None
    except Exception:
        return None
