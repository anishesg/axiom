"""Real-time signal processing for Muse S EEG/PPG data.

Uses BrainFlow's built-in DSP where possible (C-optimized),
falls back to scipy for anything BrainFlow doesn't cover.
"""

import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations, NoiseTypes
from scipy.signal import find_peaks

from config import (
    EEG_SAMPLE_RATE, PPG_SAMPLE_RATE,
    NOTCH_FREQ, BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER,
    FREQ_BANDS,
)


def filter_eeg_channel(data: np.ndarray, sample_rate: int = EEG_SAMPLE_RATE) -> np.ndarray:
    """Apply standard EEG preprocessing to a single channel (in-place copy)."""
    filtered = data.copy()
    if len(filtered) < FILTER_ORDER * 3:
        return filtered

    DataFilter.detrend(filtered, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(
        filtered, sample_rate,
        BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER,
        FilterTypes.BUTTERWORTH.value, 0.0,
    )
    DataFilter.remove_environmental_noise(
        filtered, sample_rate, NoiseTypes.SIXTY.value,
    )
    return filtered


def compute_band_powers(data: np.ndarray, sample_rate: int = EEG_SAMPLE_RATE) -> dict[str, float]:
    """Compute power in each frequency band for a single channel.

    Returns dict like {"Delta": 12.3, "Theta": 8.1, ...} in uV^2/Hz.
    """
    if len(data) < sample_rate:
        return {band: 0.0 for band in FREQ_BANDS}

    nfft = DataFilter.get_nearest_power_of_two(sample_rate)
    psd = DataFilter.get_psd_welch(
        data, nfft, nfft // 2, sample_rate,
        WindowOperations.HANNING.value,
    )
    powers = {}
    for band_name, (low, high) in FREQ_BANDS.items():
        powers[band_name] = DataFilter.get_band_power(psd, low, high)
    return powers


def compute_all_band_powers(eeg_data: np.ndarray, sample_rate: int = EEG_SAMPLE_RATE) -> dict[str, np.ndarray]:
    """Compute band powers for all 4 EEG channels.

    Returns {"Delta": [ch0, ch1, ch2, ch3], "Theta": [...], ...}
    """
    result = {band: np.zeros(4) for band in FREQ_BANDS}
    for ch_idx in range(min(4, eeg_data.shape[0])):
        channel = eeg_data[ch_idx]
        if len(channel) < sample_rate:
            continue
        filtered = filter_eeg_channel(channel, sample_rate)
        powers = compute_band_powers(filtered, sample_rate)
        for band, val in powers.items():
            result[band][ch_idx] = val
    return result


def compute_signal_quality(eeg_data: np.ndarray, sample_rate: int = EEG_SAMPLE_RATE) -> np.ndarray:
    """Estimate signal quality per channel (0=bad, 1=good).

    Heuristic: good signal has low high-frequency noise and reasonable amplitude.
    """
    quality = np.zeros(eeg_data.shape[0])
    for ch_idx in range(eeg_data.shape[0]):
        ch = eeg_data[ch_idx]
        n = len(ch)
        if n < sample_rate:
            continue

        last_sec = ch[-sample_rate:]
        std = np.std(last_sec)
        if std < 1.0:
            quality[ch_idx] = 0.1
            continue
        if std > 200.0:
            quality[ch_idx] = 0.1
            continue

        filtered = filter_eeg_channel(last_sec.copy(), sample_rate)
        signal_power = np.mean(filtered ** 2)
        noise_power = np.mean((last_sec - filtered) ** 2)
        if noise_power < 1e-10:
            quality[ch_idx] = 1.0
        else:
            snr = signal_power / noise_power
            quality[ch_idx] = min(1.0, snr / 5.0)

    return np.clip(quality, 0.0, 1.0)


def compute_heart_rate(ppg_data: np.ndarray, sample_rate: int = PPG_SAMPLE_RATE) -> float:
    """Estimate heart rate from PPG IR channel using peak detection.

    Returns BPM or 0.0 if insufficient data.
    """
    if ppg_data.shape[1] < sample_rate * 5:
        return 0.0

    ir_channel = ppg_data[1]  # IR channel
    ir = ir_channel[-sample_rate * 10:].copy()

    if np.std(ir) < 1e-6:
        return 0.0

    DataFilter.detrend(ir, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(
        ir, sample_rate,
        0.5, 4.0, 2,
        FilterTypes.BUTTERWORTH.value, 0.0,
    )

    min_distance = int(sample_rate * 0.4)  # at least 0.4s between beats (150 bpm max)
    peaks, _ = find_peaks(ir, distance=min_distance, prominence=np.std(ir) * 0.3)

    if len(peaks) < 2:
        return 0.0

    intervals = np.diff(peaks) / sample_rate
    intervals = intervals[(intervals > 0.4) & (intervals < 1.5)]
    if len(intervals) == 0:
        return 0.0

    return 60.0 / np.median(intervals)


def compute_psd(data: np.ndarray, sample_rate: int = EEG_SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD for a single channel. Returns (power, freqs)."""
    if len(data) < sample_rate:
        return np.array([]), np.array([])

    nfft = DataFilter.get_nearest_power_of_two(sample_rate)
    psd = DataFilter.get_psd_welch(
        data, nfft, nfft // 2, sample_rate,
        WindowOperations.HANNING.value,
    )
    return psd[0], psd[1]  # amplitudes, frequencies
