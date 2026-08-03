"""Tests for the real-time audio ring buffer (can run with no hardware)."""

import numpy as np

from optitune.audio.ringbuffer import RingBuffer


def test_basic_push_and_get():
    buf = RingBuffer(max_samples=100)
    buf.push(np.arange(10, dtype=np.float32))
    latest = buf.get_latest(5)
    assert latest.shape == (5,)
    assert np.allclose(latest, [5, 6, 7, 8, 9])


def test_get_more_than_available_returns_padded():
    buf = RingBuffer(max_samples=100)
    buf.push(np.ones(7, dtype=np.float32))
    latest = buf.get_latest(20)
    assert latest.shape == (20,)
    assert np.sum(latest[:13]) == 0.0  # leading zeros
    assert np.allclose(latest[13:], 1.0)


def test_bounded_drops_old_data():
    buf = RingBuffer(max_samples=20)
    buf.push(np.arange(15, dtype=np.float32))
    buf.push(np.arange(15, 30, dtype=np.float32))  # should drop some old data

    latest = buf.get_latest(20)
    # We should have the most recent 20 samples: 10..29
    assert np.allclose(latest, np.arange(10, 30, dtype=np.float32))


def test_thread_safety_smoke():
    import threading

    buf = RingBuffer(max_samples=1000)
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            block = np.full(64, i, dtype=np.float32)
            buf.push(block)
            i += 1
            if i % 100 == 0:
                # occasional bigger block
                buf.push(np.full(300, i, dtype=np.float32))

    t = threading.Thread(target=writer, daemon=True)
    t.start()

    # Reader for a short time
    for _ in range(200):
        data = buf.get_latest(256)
        assert data.shape == (256,)

    stop.set()
    t.join(timeout=1.0)
    assert not t.is_alive()
