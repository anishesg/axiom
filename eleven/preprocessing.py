"""
Real-time EEG signal preprocessing.

This module handles:
- Bandpass filtering (1-40 Hz for general EEG)
- Notch filtering (50/60 Hz power line noise)
- Windowing and buffering for feature extraction
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Optional
from collections import deque


@dataclass
class FilterConfig:
    """Configuration for EEG filtering."""

    sample_rate: int = 256
    lowcut: float = 1.0      # Hz - removes DC drift
    highcut: float = 40.0    # Hz - removes high-frequency noise
    notch_freq: float = 60.0  # Hz - power line (use 50.0 for Europe)
    notch_q: float = 30.0    # Quality factor for notch filter
    filter_order: int = 4    # Butterworth filter order


class RealTimeFilter:
    """
    Real-time bandpass and notch filter for EEG signals.

    Uses scipy's sosfilt with zi state for continuous filtering
    without edge artifacts between chunks.
    """

    def __init__(self, config: Optional[FilterConfig] = None, n_channels: int = 4):
        self.config = config or FilterConfig()
        self.n_channels = n_channels

        # Design bandpass filter (second-order sections for stability)
        nyquist = self.config.sample_rate / 2
        low = self.config.lowcut / nyquist
        high = self.config.highcut / nyquist

        # Clamp to valid range
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.99))

        self._bandpass_sos = signal.butter(
            self.config.filter_order, [low, high], btype="band", output="sos"
        )

        # Design notch filter
        notch_low = (self.config.notch_freq - 1) / nyquist
        notch_high = (self.config.notch_freq + 1) / nyquist
        notch_low = max(0.001, min(notch_low, 0.99))
        notch_high = max(notch_low + 0.001, min(notch_high, 0.99))

        self._notch_sos = signal.butter(2, [notch_low, notch_high], btype="bandstop", output="sos")

        # Initialize filter states for each channel
        self._bandpass_zi = [
            signal.sosfilt_zi(self._bandpass_sos) for _ in range(n_channels)
        ]
        self._notch_zi = [
            signal.sosfilt_zi(self._notch_sos) for _ in range(n_channels)
        ]

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass and notch filters to EEG data.

        Args:
            data: Shape (n_channels, n_samples) raw EEG data.

        Returns:
            Filtered data with same shape.
        """
        if data.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {data.shape[0]}")

        filtered = np.zeros_like(data)

        for ch in range(self.n_channels):
            # Bandpass filter
            bp_out, self._bandpass_zi[ch] = signal.sosfilt(
                self._bandpass_sos, data[ch], zi=self._bandpass_zi[ch] * data[ch, 0]
            )

            # Notch filter
            notch_out, self._notch_zi[ch] = signal.sosfilt(
                self._notch_sos, bp_out, zi=self._notch_zi[ch] * bp_out[0]
            )

            filtered[ch] = notch_out

        return filtered

    def reset(self):
        """Reset filter states (call when starting new session)."""
        self._bandpass_zi = [
            signal.sosfilt_zi(self._bandpass_sos) for _ in range(self.n_channels)
        ]
        self._notch_zi = [
            signal.sosfilt_zi(self._notch_sos) for _ in range(self.n_channels)
        ]


class SlidingWindowBuffer:
    """
    Sliding window buffer for continuous EEG processing.

    Maintains a rolling buffer of EEG data and provides
    overlapping windows for feature extraction.
    """

    def __init__(
        self,
        window_size: int = 256,    # 1 second at 256 Hz
        step_size: int = 64,       # 250ms step (75% overlap)
        n_channels: int = 4,
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.n_channels = n_channels

        # Internal buffer
        self._buffer = np.zeros((n_channels, window_size))
        self._samples_in_buffer = 0
        self._samples_since_output = 0

    def add(self, data: np.ndarray) -> list[np.ndarray]:
        """
        Add new data to buffer and return complete windows.

        Args:
            data: Shape (n_channels, n_samples) new EEG data.

        Returns:
            List of complete windows, each shape (n_channels, window_size).
            May return empty list if not enough data yet.
        """
        n_samples = data.shape[1]
        windows = []

        for i in range(n_samples):
            # Shift buffer left and add new sample
            self._buffer = np.roll(self._buffer, -1, axis=1)
            self._buffer[:, -1] = data[:, i]

            self._samples_in_buffer = min(self._samples_in_buffer + 1, self.window_size)
            self._samples_since_output += 1

            # Check if we have a complete window and enough samples since last output
            if (self._samples_in_buffer >= self.window_size and
                self._samples_since_output >= self.step_size):
                windows.append(self._buffer.copy())
                self._samples_since_output = 0

        return windows

    def reset(self):
        """Clear the buffer."""
        self._buffer = np.zeros((self.n_channels, self.window_size))
        self._samples_in_buffer = 0
        self._samples_since_output = 0


class EEGPreprocessor:
    """
    Complete preprocessing pipeline for real-time EEG.

    Combines filtering and windowing into a single interface.
    """

    def __init__(
        self,
        sample_rate: int = 256,
        window_duration: float = 1.0,    # seconds
        step_duration: float = 0.25,     # seconds
        notch_freq: float = 60.0,        # Hz
    ):
        self.sample_rate = sample_rate
        self.n_channels = 4  # Muse S has 4 EEG channels

        # Calculate sizes
        window_size = int(window_duration * sample_rate)
        step_size = int(step_duration * sample_rate)

        # Initialize components
        self.filter = RealTimeFilter(
            FilterConfig(sample_rate=sample_rate, notch_freq=notch_freq),
            n_channels=self.n_channels,
        )
        self.buffer = SlidingWindowBuffer(
            window_size=window_size,
            step_size=step_size,
            n_channels=self.n_channels,
        )

        self.window_size = window_size
        self.step_size = step_size

    def process(self, raw_data: np.ndarray) -> list[np.ndarray]:
        """
        Process raw EEG data through the full pipeline.

        Args:
            raw_data: Shape (5, n_samples) where rows are [TP9, AF7, AF8, TP10, timestamp].

        Returns:
            List of processed windows, each shape (4, window_size).
        """
        # Extract EEG channels (exclude timestamp)
        eeg_data = raw_data[:4, :]

        # Apply filters
        filtered = self.filter.filter(eeg_data)

        # Buffer and extract windows
        windows = self.buffer.add(filtered)

        return windows

    def reset(self):
        """Reset all internal state."""
        self.filter.reset()
        self.buffer.reset()


# Utility functions for offline processing

def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    sample_rate: int,
    order: int = 4,
) -> np.ndarray:
    """
    Apply bandpass filter to data (offline version).

    Args:
        data: Shape (n_channels, n_samples) or (n_samples,)
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        sample_rate: Sampling rate in Hz
        order: Filter order

    Returns:
        Filtered data with same shape.
    """
    nyquist = sample_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    sos = signal.butter(order, [low, high], btype="band", output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1)


def notch_filter(
    data: np.ndarray,
    freq: float,
    sample_rate: int,
    q: float = 30.0,
) -> np.ndarray:
    """
    Apply notch filter to remove power line noise (offline version).

    Args:
        data: Shape (n_channels, n_samples) or (n_samples,)
        freq: Frequency to remove in Hz (50 or 60)
        sample_rate: Sampling rate in Hz
        q: Quality factor

    Returns:
        Filtered data with same shape.
    """
    b, a = signal.iirnotch(freq, q, sample_rate)
    return signal.filtfilt(b, a, data, axis=-1)


# CLI for testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate test signal: 10 Hz alpha + 60 Hz noise + random noise
    sample_rate = 256
    duration = 5  # seconds
    t = np.linspace(0, duration, sample_rate * duration)

    # Simulate 4 channels
    alpha = 20 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    noise_60hz = 10 * np.sin(2 * np.pi * 60 * t)  # Power line noise
    random_noise = np.random.randn(len(t)) * 5

    raw = alpha + noise_60hz + random_noise
    raw_4ch = np.vstack([raw, raw * 0.8, raw * 0.8, raw])  # Simulate 4 channels

    # Apply offline filtering
    filtered = bandpass_filter(raw_4ch, 1, 40, sample_rate)
    filtered = notch_filter(filtered, 60, sample_rate)

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(t[:500], raw_4ch[0, :500], label="Raw (CH1)")
    axes[0].set_title("Raw EEG (first 2 seconds)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (μV)")
    axes[0].legend()

    axes[1].plot(t[:500], filtered[0, :500], label="Filtered (CH1)")
    axes[1].set_title("Filtered EEG (1-40 Hz bandpass + 60 Hz notch)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (μV)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("preprocessing_test.png")
    print("Saved plot to preprocessing_test.png")
