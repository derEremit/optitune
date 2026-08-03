"""Pytest defaults for OptiTune.

Force synchronous live analysis in tests so QThread teardown does not race
widget destruction. Production GUI still uses AnalysisWorker by default.
"""

from __future__ import annotations

import os

# Must be set before MainWindow is constructed in any test
os.environ.setdefault("OPTITUNE_SYNC_ANALYSIS", "1")
