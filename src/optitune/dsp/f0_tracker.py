"""
Temporal f0 tracking: median / mode over the last N estimation ticks.

Attack frames and partial lock-ons are outliers; decay is long and stable.
Pure and Qt-free so it can be unit-tested without the GUI.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from optitune.dsp.synth import hz_to_midi


class F0Tracker:
    """Sliding-window robust f0 estimate."""

    def __init__(self, window: int = 7) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = int(window)
        self._buf: deque[float] = deque(maxlen=self._window)

    def push(self, f0_hz: float) -> float | None:
        """Append a new instantaneous f0; ignore non-positive. Returns current()."""
        if f0_hz is None or not np.isfinite(f0_hz) or f0_hz <= 0:
            return self.current()
        self._buf.append(float(f0_hz))
        return self.current()

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def current(self) -> float | None:
        """
        Robust current f0.

        1. Prefer the denser octave-cluster among values (handles 50/50 octave splits
           by choosing the lower cluster when sizes tie — bass fundamentals are the
           intended target under partial lock-on).
        2. Within the chosen cluster, return the median.
        """
        if not self._buf:
            return None
        vals = np.asarray(self._buf, dtype=float)
        if len(vals) == 1:
            return float(vals[0])

        # Cluster by octave relative to the minimum (log2 space)
        lo = float(np.min(vals))
        if lo <= 0:
            return float(np.median(vals))

        octaves = np.rint(np.log2(vals / lo)).astype(int)
        # Count members per octave bin
        best_oct = 0
        best_count = -1
        for o in sorted(set(octaves.tolist())):
            c = int(np.sum(octaves == o))
            # Prefer larger cluster; on tie prefer lower octave
            if c > best_count or (c == best_count and o < best_oct):
                best_count = c
                best_oct = o

        cluster = vals[octaves == best_oct]
        return float(np.median(cluster))

    def current_midi(self, a4: float = 440.0) -> int | None:
        f = self.current()
        if f is None or f <= 0:
            return None
        return round(hz_to_midi(f, a4))
