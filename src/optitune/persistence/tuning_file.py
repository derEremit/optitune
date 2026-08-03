"""
.pfg tuning file — XML compatible with EPT-style outer structure (spec §4.4).

Stores piano name, A4, per-key B/f0/cent offset, optional compressed spectrum,
and the 88-point tuning curve.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.dom import minidom

import numpy as np

from optitune.model.key import Key
from optitune.model.piano import Piano
from optitune.model.spectrum_codec import pack_spectrum, unpack_spectrum

PFG_VERSION = "1.0"
MIDI_LOW = 21
N_KEYS = 88


def save_pfg(piano: Piano, path: str | Path, *, temperament: str = "equal") -> None:
    root = ET.Element("piano", version=PFG_VERSION)
    meta = ET.SubElement(root, "meta")
    ET.SubElement(meta, "name").text = piano.name
    ET.SubElement(meta, "a4").text = f"{float(piano.a4):.6f}"
    ET.SubElement(meta, "temperament").text = str(temperament)

    keyboard = ET.SubElement(root, "keyboard", low=str(MIDI_LOW), high=str(MIDI_LOW + N_KEYS - 1))
    for midi in range(MIDI_LOW, MIDI_LOW + N_KEYS):
        k = piano.keys.get(midi)
        if k is None:
            continue
        attrs: dict[str, str] = {"index": str(midi - MIDI_LOW), "midi": str(midi)}
        if k.measured_b is not None:
            attrs["B"] = f"{float(k.measured_b):.8g}"
        if k.measured_f0 is not None:
            attrs["f0"] = f"{float(k.measured_f0):.6f}"
        attrs["cents"] = f"{float(k.target_offset_cents):.4f}"
        key_el = ET.SubElement(keyboard, "key", **attrs)
        if k.cent_spectrum is not None:
            spec = ET.SubElement(key_el, "spectrum", encoding="zlib-base64-f32")
            spec.text = pack_spectrum(k.cent_spectrum)

    if piano.tuning_curve is not None:
        curve_el = ET.SubElement(root, "tuning_curve", n=str(N_KEYS))
        curve_el.text = ",".join(f"{float(c):.4f}" for c in piano.tuning_curve)

    xml_str = _prettify(root)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(xml_str, encoding="utf-8")


def load_pfg(path: str | Path) -> tuple[Piano, dict[str, Any]]:
    """
    Load a .pfg file. Returns (Piano, metadata dict with temperament etc.).
    """
    tree = ET.parse(Path(path))
    root = tree.getroot()
    if root.tag != "piano":
        raise ValueError(f"Expected <piano> root, got <{root.tag}>")

    meta: dict[str, Any] = {}
    meta_el = root.find("meta")
    name = "My Piano"
    a4 = 440.0
    if meta_el is not None:
        name = (meta_el.findtext("name") or name).strip()
        try:
            a4 = float(meta_el.findtext("a4") or a4)
        except ValueError:
            a4 = 440.0
        meta["temperament"] = (meta_el.findtext("temperament") or "equal").strip()

    piano = Piano(name=name, a4=a4)
    keyboard = root.find("keyboard")
    if keyboard is not None:
        for key_el in keyboard.findall("key"):
            try:
                midi = int(key_el.get("midi") or (MIDI_LOW + int(key_el.get("index", "0"))))
            except ValueError:
                continue
            b = _opt_float(key_el.get("B"))
            f0 = _opt_float(key_el.get("f0"))
            cents = _opt_float(key_el.get("cents")) or 0.0
            spectrum = None
            spec_el = key_el.find("spectrum")
            if spec_el is not None and spec_el.text:
                spectrum = unpack_spectrum(spec_el.text.strip())
            piano.set_key(
                Key(
                    midi=midi,
                    measured_f0=f0,
                    measured_b=b,
                    cent_spectrum=spectrum,
                    target_offset_cents=float(cents),
                )
            )

    curve_el = root.find("tuning_curve")
    if curve_el is not None and curve_el.text:
        parts = [p.strip() for p in curve_el.text.split(",") if p.strip()]
        if len(parts) == N_KEYS:
            piano.tuning_curve = [float(x) for x in parts]

    return piano, meta


def _opt_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _prettify(elem: ET.Element) -> str:
    rough = ET.tostring(elem, encoding="utf-8")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
