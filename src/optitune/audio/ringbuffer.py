"""
Thread-safe bounded ring buffer for real-time audio (improved implementation).

Uses a single contiguous float32 array + head pointer for O(1) amortized operations
and perfect "most recent N samples" semantics.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """
    High-performance bounded ring buffer for mono float32 audio.

    Writer (audio callback) calls push().
    Reader (DSP thread) calls get_latest(n).

    Guarantees that get_latest(n) always returns the most recent min(n, available) samples,
    left-padded with zeros if necessary.
    """

    def __init__(self, max_samples: int = 192_000) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0")
        self.max_samples = int(max_samples)
        self._data = np.zeros(self.max_samples, dtype=np.float32)
        self._head = 0  # next write position
        self._count = 0  # how many valid samples are currently stored
        self._lock = threading.RLock()
        self._dropped = 0

    def push(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        samples = np.asarray(samples, dtype=np.float32).ravel()

        with self._lock:
            n = samples.size
            if n >= self.max_samples:
                # Incoming block is bigger than entire buffer — keep only the tail
                self._data[:] = samples[-self.max_samples :]
                self._head = 0
                self._count = self.max_samples
                self._dropped += n - self.max_samples
                return

            # Normal case
            end = (self._head + n) % self.max_samples
            if end > self._head or n == 0:
                self._data[self._head : self._head + n] = samples
            else:
                # wraps
                first = self.max_samples - self._head
                self._data[self._head :] = samples[:first]
                self._data[:end] = samples[first:]

            # Update count and head
            if self._count + n > self.max_samples:
                self._dropped += (self._count + n) - self.max_samples
                self._count = self.max_samples
            else:
                self._count += n

            self._head = end

    def get_latest(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        with self._lock:
            available = min(n, self._count)
            result = np.zeros(n, dtype=np.float32)
            if available == 0:
                return result

            # Read the most recent `available` samples into the end of the result array
            start = (self._head - available) % self.max_samples
            if start + available <= self.max_samples:
                result[-available:] = self._data[start : start + available]
            else:
                first = self.max_samples - start
                result[-available : -available + first] = self._data[start:]
                result[-available + first :] = self._data[: available - first]
            return result

    def clear(self) -> None:
        with self._lock:
            self._head = 0
            self._count = 0
            self._data.fill(0.0)

    def available_samples(self) -> int:
        with self._lock:
            return self._count

    @property
    def dropped_samples(self) -> int:
        with self._lock:
            return self._dropped
