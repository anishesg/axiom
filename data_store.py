"""Thread-safe ring buffers for real-time signal storage.

Each buffer is a fixed-size numpy array that overwrites oldest data.
Consumers (dashboard, RL agent) read snapshots without blocking the producer.
"""

import threading
import numpy as np


class RingBuffer:
    """Lock-free-ish ring buffer backed by a numpy array.

    Writers append chunks; readers get a contiguous snapshot (oldest-first).
    """

    def __init__(self, channels: int, max_samples: int):
        self._buf = np.zeros((channels, max_samples), dtype=np.float64)
        self._ts = np.zeros(max_samples, dtype=np.float64)
        self._channels = channels
        self._capacity = max_samples
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()

    @property
    def total_written(self) -> int:
        return self._total_written

    def append(self, data: np.ndarray, timestamps: np.ndarray):
        """Append data shaped (channels, n_samples) with timestamps (n_samples,)."""
        n = data.shape[1]
        if n == 0:
            return
        with self._lock:
            if n >= self._capacity:
                self._buf[:] = data[:, -self._capacity:]
                self._ts[:] = timestamps[-self._capacity:]
                self._write_pos = 0
                self._total_written += n
                return

            end = self._write_pos + n
            if end <= self._capacity:
                self._buf[:, self._write_pos:end] = data
                self._ts[self._write_pos:end] = timestamps
            else:
                first = self._capacity - self._write_pos
                self._buf[:, self._write_pos:] = data[:, :first]
                self._ts[self._write_pos:] = timestamps[:first]
                rest = n - first
                self._buf[:, :rest] = data[:, first:]
                self._ts[:rest] = timestamps[first:]

            self._write_pos = end % self._capacity
            self._total_written += n

    def get_last_n(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the last n samples as (channels, n) and timestamps (n,)."""
        with self._lock:
            available = min(n, self._total_written, self._capacity)
            if available == 0:
                return np.zeros((self._channels, 0)), np.zeros(0)

            end = self._write_pos
            start = end - available
            if start >= 0:
                return self._buf[:, start:end].copy(), self._ts[start:end].copy()
            else:
                part1 = self._buf[:, start % self._capacity:]
                part2 = self._buf[:, :end]
                ts1 = self._ts[start % self._capacity:]
                ts2 = self._ts[:end]
                return (
                    np.concatenate([part1, part2], axis=1),
                    np.concatenate([ts1, ts2]),
                )

    def get_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all available data, oldest first."""
        return self.get_last_n(self._capacity)


class MuseDataStore:
    """Central store holding ring buffers for all Muse S signal types."""

    def __init__(self):
        from config import (
            EEG_SAMPLE_RATE, EEG_BUFFER_SECONDS,
            PPG_SAMPLE_RATE, PPG_BUFFER_SECONDS,
            IMU_SAMPLE_RATE, IMU_BUFFER_SECONDS,
        )
        self.eeg = RingBuffer(4, EEG_SAMPLE_RATE * EEG_BUFFER_SECONDS)
        self.ppg = RingBuffer(3, PPG_SAMPLE_RATE * PPG_BUFFER_SECONDS)
        self.acc = RingBuffer(3, IMU_SAMPLE_RATE * IMU_BUFFER_SECONDS)
        self.gyro = RingBuffer(3, IMU_SAMPLE_RATE * IMU_BUFFER_SECONDS)

        self.connected = False
        self.device_name = ""
        self.signal_quality = np.zeros(4)  # per EEG channel, 0-1
