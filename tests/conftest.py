"""
Pytest configuration and shared fixtures for OptiTune.

pytest-qt is auto-discovered via its entry point when installed in the environment.
No manual registration needed (and declaring the string name can cause import issues
with the actual top-level module 'pytestqt').
"""

from __future__ import annotations

# Intentionally empty — all Qt test configuration is handled by pytest-qt plugin
# and the [tool.pytest.ini_options] section in pyproject.toml (qt_api = "pyside6").

# Future fixtures (synthetic audio, qtbot helpers, etc.) go here in Phase 1+.
