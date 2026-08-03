"""AudioCapture stop/restart safety without real hardware."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from optitune.audio.capture import AudioCapture
from optitune.audio.ringbuffer import RingBuffer


def test_stop_is_idempotent():
    rb = RingBuffer(max_samples=4096)
    cap = AudioCapture(rb)
    cap.stop()
    cap.stop()
    assert cap.is_running is False


def test_restart_after_failed_start_returns_false():
    rb = RingBuffer(max_samples=4096)
    cap = AudioCapture(rb)
    with patch("optitune.audio.capture.sd.InputStream", side_effect=OSError("no device")):
        with pytest.raises(OSError):
            cap.start(device=99)
    assert cap.restart() is False
    assert cap.last_error


def test_restart_success_with_mock_stream():
    rb = RingBuffer(max_samples=4096)
    cap = AudioCapture(rb)
    stream = MagicMock()
    stream.active = True
    with patch("optitune.audio.capture.sd.InputStream", return_value=stream):
        with patch("optitune.audio.capture.sd.query_devices", return_value={"default_samplerate": 48000}):
            ok = cap.restart()
    assert ok is True
    stream.start.assert_called()


def test_callback_does_not_raise_on_push_failure():
    rb = RingBuffer(max_samples=4096)
    cap = AudioCapture(rb)
    with patch.object(rb, "push", side_effect=RuntimeError("yanked")):
        # Should not raise out of callback
        cap._audio_callback(np.zeros((64, 1), dtype=np.float32), 64, None, 0)
    assert cap.last_error is not None
