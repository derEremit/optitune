"""
Minimal Entropy Piano Tuner (.ept) XML import → Piano.

EPT layouts vary by version; this reader accepts a small common subset:
  <piano>
    <name>…</name>
    <concertPitch>440</concertPitch>
    <keys>
      <key number="0..87">  <!-- index from A0, or midi="21..108" -->
        <recordedFrequency>…</recordedFrequency>
        <inharmonicity>…</inharmonicity>
      </key>
    </keys>
  </piano>
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from optitune.model.key import Key
from optitune.model.piano import Piano

MIDI_LOW = 21


def load_ept(path: str | Path) -> Piano:
    tree = ET.parse(Path(path))
    root = tree.getroot()
    # Some EPT files wrap in a document element; accept piano at root or child
    if root.tag != "piano":
        found = root.find(".//piano")
        if found is None:
            raise ValueError(f"No <piano> element in {path}")
        root = found

    name = (root.findtext("name") or root.findtext("Name") or "Imported EPT").strip()
    a4_txt = root.findtext("concertPitch") or root.findtext("a4") or "440"
    try:
        a4 = float(a4_txt)
    except ValueError:
        a4 = 440.0

    piano = Piano(name=name, a4=a4)
    keys_el = root.find("keys")
    if keys_el is None:
        keys_el = root.find("keyboard")
    if keys_el is None:
        return piano

    for key_el in keys_el.findall("key"):
        midi = _key_midi(key_el)
        if midi is None:
            continue
        f0 = _child_float(key_el, "recordedFrequency", "f0", "frequency")
        b = _child_float(key_el, "inharmonicity", "B", "b")
        cents = _child_float(key_el, "tuning", "cents") or 0.0
        if f0 is None and b is None:
            continue
        piano.set_key(
            Key(midi=midi, measured_f0=f0, measured_b=b, target_offset_cents=float(cents))
        )
    return piano


def _key_midi(key_el: ET.Element) -> int | None:
    if key_el.get("midi") is not None:
        try:
            return int(key_el.get("midi"))  # type: ignore[arg-type]
        except ValueError:
            return None
    num = key_el.get("number") or key_el.get("index")
    if num is None:
        return None
    try:
        idx = int(num)
    except ValueError:
        return None
    # Treat 0..87 as A0-based index; 21..108 as MIDI
    if 0 <= idx <= 87:
        return MIDI_LOW + idx
    if 21 <= idx <= 108:
        return idx
    return None


def _child_float(el: ET.Element, *tags: str) -> float | None:
    for t in tags:
        txt = el.findtext(t)
        if txt is None or txt.strip() == "":
            continue
        try:
            return float(txt)
        except ValueError:
            continue
    return None
