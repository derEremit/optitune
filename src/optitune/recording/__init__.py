"""Recording workflow controllers (auto-record, onset, guided sessions)."""

from .auto_record import (
    AutoRecordConfig,
    AutoRecordController,
    AutoRecordEvent,
    AutoRecordPhase,
)

__all__ = [
    "AutoRecordConfig",
    "AutoRecordController",
    "AutoRecordEvent",
    "AutoRecordPhase",
]
