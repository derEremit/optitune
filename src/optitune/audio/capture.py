"""
AudioCapture — real implementation wrapping sounddevice.InputStream.

Feeds a RingBuffer from the real-time audio callback. Supports device selection
by index or name, graceful start/stop, and samplerate discovery per device.
Thread-safe; callback is non-blocking.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import sounddevice as sd

from optitune.audio.ringbuffer import RingBuffer


class AudioCapture:
    """
    Manages a sounddevice input stream and feeds a RingBuffer.

    Public API:
        start(device: int | str | None = None)
        stop()
        is_running -> bool
        current_device -> str | None
        samplerate -> int
    """

    def __init__(
        self,
        ringbuffer: RingBuffer,
        samplerate: int = 48000,
        blocksize: int = 1024,
    ) -> None:
        self.ringbuffer = ringbuffer
        self._requested_samplerate = int(samplerate)
        self.blocksize = int(blocksize)
        self._stream: sd.InputStream | None = None
        self._device: int | str | None = None
        self._actual_samplerate: int = self._requested_samplerate
        self._lock = threading.Lock()
        self._last_error: str | None = None

    def start(self, device: int | str | None = None) -> None:
        """Start (or restart) capture on the given device (index or name). Idempotent."""
        with self._lock:
            if self._stream is not None:
                self._stop_locked()

            self._device = device
            self._last_error = None

            try:
                # Discover actual device params
                if device is not None:
                    try:
                        dev_info: dict[str, Any] = sd.query_devices(device)
                        self._actual_samplerate = int(
                            dev_info.get("default_samplerate", self._requested_samplerate)
                        )
                    except Exception:
                        self._actual_samplerate = self._requested_samplerate
                else:
                    self._actual_samplerate = self._requested_samplerate

                # Create and start the input stream (mono, float32, low latency)
                self._stream = sd.InputStream(
                    device=device,
                    channels=1,
                    samplerate=self._actual_samplerate,
                    blocksize=self.blocksize,
                    dtype="float32",
                    callback=self._audio_callback,
                    latency="low",
                )
                self._stream.start()
            except Exception as exc:
                self._last_error = str(exc)
                self._stream = None
                raise

    def stop(self) -> None:
        """Stop capture and release the stream. Safe to call anytime."""
        with self._lock:
            self._stop_locked()

    def restart(self) -> bool:
        """
        Stop and start again on the last device. Returns True if running.
        Used after device errors / PipeWire blips.
        """
        dev = self._device
        try:
            self.stop()
            self.start(dev)
            return self.is_running
        except Exception as exc:
            self._last_error = str(exc)
            return False

    def health_ok(self) -> bool:
        """True if stream exists and claims active (best-effort)."""
        with self._lock:
            if self._stream is None:
                return False
            try:
                return bool(self._stream.active)
            except Exception:
                return False

    def _stop_locked(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """PortAudio real-time callback — keep it extremely fast."""
        try:
            if status:
                # Non-fatal (xruns etc.) — do not allocate/print in hot path
                pass
            mono = indata[:, 0] if indata.ndim > 1 else indata
            self.ringbuffer.push(mono)
        except Exception as exc:
            # Device yank mid-callback — record and let outer layer restart
            self._last_error = str(exc)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._stream is not None and self._stream.active

    @property
    def current_device(self) -> str | None:
        if self._device is None:
            return None
        return str(self._device)

    @property
    def samplerate(self) -> int:
        return self._actual_samplerate

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def __del__(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self.stop()
