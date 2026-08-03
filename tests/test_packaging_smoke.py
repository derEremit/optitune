"""Packaging / entry-point smoke checks (M7)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import optitune


def test_version_string():
    assert optitune.__version__
    parts = optitune.__version__.split(".")
    assert len(parts) >= 2


def test_cli_version():
    r = subprocess.run(
        [sys.executable, "-m", "optitune", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert optitune.__version__ in (r.stdout + r.stderr)


def test_flatpak_scaffold_files_exist():
    root = Path(__file__).resolve().parents[1]
    base = root / "packaging" / "flatpak"
    assert (base / "org.optitune.OptiTune.yml").is_file()
    assert (base / "org.optitune.OptiTune.desktop").is_file()
    assert (base / "org.optitune.OptiTune.metainfo.xml").is_file()
    assert (root / "assets" / "icon.svg").is_file()
    assert (root / "docs" / "user_guide.md").is_file()
