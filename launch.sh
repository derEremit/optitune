#!/usr/bin/env bash
# OptiTune one-click launcher (convenience wrapper)
# Usage: ./launch.sh [args...]
# After `uv sync`, this is equivalent to `optitune` once the entry point is on PATH.

set -euo pipefail

# Resolve directory of this script (works even if called via symlink)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run via uv from the project root so editable src/ layout and scripts work
exec uv run --project "${SCRIPT_DIR}" optitune "$@"
