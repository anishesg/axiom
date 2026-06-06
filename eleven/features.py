"""
Feature extraction for EEG signals.

Extracts features used for:
- Attention/focus detection
- VQ encoding
- State classification
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from dataclasses import dataclass
from typing import Optional


# EEG frequency bands
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


@dataclass
class BandPowers:
    """EEG band powers for all channels."""

    delta: np.ndarray  # Shape: (4,) for each channel
    theta: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray

    def to_array(self) -> np.ndarray:
        """Convert to flat feature array. Shape: (20,)"""
        return np.concatenate([
            self.delta, self.theta, self.alpha, self.beta, self.gamma
        ])

    def relative_powers(self) -> "BandPowers":
        """Convert to relative (normalized) band powers."""
        total = self.delta + self.theta + self.alpha + self.beta + self.gamma
        total = np.maximum(total, 1e-10)  # Avoid division by zero

        return BandPowers(
            delta=self.delta / total,
            theta=self.theta / total,
            alpha=self.alpha / total,
            beta=self.beta / total,
            gamma=self.gamma / total,
        )


def compute_band_powers(
    data: np.ndarray,
    sample_rate: int = 256,
    method: str = "welch",
) -> BandPowers:
    """
    Compute absolute band powers for each channel.

    Args:
        data: Shape (n_channels, n_samples) EEG data
        sample_rate: Sampling rate in Hz
        method: "welch" (default) or "fft"

    Returns:
        BandPowers object with power in each band per channel.
    """
    n_channels = data.shape[0]
    powers = {band: np.zeros(n_channels) for band in BANDS}

    for ch in range(n_channels):
        if method == "welch":
            freqs, psd = signal.welch(
                data[ch], fs=sample_rate, nperseg=min(256, data.shape[1])
            )
        else:
            # Simple FFT method
            n = data.shape[1]
            freqs = rfftfreq(n, 1 / sample_rate)
            fft_vals = rfft(data[ch])
            psd = np.abs(fft_vals) ** 2 / n

        # Extract power in each band
        for band_name, (low, high) in BANDS.items():
            idx = np.logical_and(freqs >= low, freqs < high)
            powers[band_name][ch] = np.mean(psd[idx]) if np.any(idx) else 0

    return BandPowers(**powers)


def compute_spectral_features(
    data: np.ndarray,
    sample_rate: int = 256,
) -> np.ndarray:
    """
    Compute spectral features for VQ encoding.

    Returns a feature vector including:
    - Band powers (absolute and relative)
    - Spectral entropy
    - Peak frequency
    - Band ratios (theta/beta, alpha/beta)

    Args:
        data: Shape (n_channels, n_samples) EEG data
        sample_rate: Sampling rate in Hz

    Returns:
        Feature vector of shape (n_features,)
    """
    n_channels = data.shape[0]
    features = []

    # Band powers
    bp = compute_band_powers(data, sample_rate)
    features.extend(bp.to_array())  # 20 features

    # Relative powers
    rel_bp = bp.relative_powers()
    features.extend(rel_bp.to_array())  # 20 features

    # Per-channel spectral features
    for ch in range(n_channels):
        freqs, psd = signal.welch(
            data[ch], fs=sample_rate, nperseg=min(256, data.shape[1])
        )

        # Spectral entropy
        psd_norm = psd / (np.sum(psd) + 1e-10)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        features.append(entropy)

        # Peak frequency
        peak_freq = freqs[np.argmax(psd)]
        features.append(peak_freq)

    # Band ratios (useful for attention/relaxation)
    theta_mean = np.mean(bp.theta)
    alpha_mean = np.mean(bp.alpha)
    beta_mean = np.mean(bp.beta)

    features.append(theta_mean / (beta_mean + 1e-10))  # Theta/beta ratio
    features.append(alpha_mean / (beta_mean + 1e-10))  # Alpha/beta ratio

    # Frontal asymmetry (AF7 vs AF8)
    # Positive = more left activation
    alpha_asymmetry = np.log(bp.alpha[2] + 1e-10) - np.log(bp.alpha[1] + 1e-10)
    features.append(alpha_asymmetry)

    return np.array(features)


def compute_temporal_features(data: np.ndarray) -> np.ndarray:
    """
    Compute time-domain features.

    Args:
        data: Shape (n_channels, n_samples) EEG data

    Returns:
        Feature vector of shape (n_features,)
    """
    features = []

    for ch in range(data.shape[0]):
        x = data[ch]

        # Statistical features
        features.append(np.mean(x))
        features.append(np.std(x))
        features.append(np.min(x))
        features.append(np.max(x))
        features.append(np.max(x) - np.min(x))  # Peak-to-peak

        # Zero crossings
        zero_crossings = np.sum(np.abs(np.diff(np.sign(x))) > 0)
        features.append(zero_crossings)

        # Hjorth parameters
        diff1 = np.diff(x)
        diff2 = np.diff(diff1)

        activity = np.var(x)
        mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
        complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)

        features.append(activity)
        features.append(mobility)
        features.append(complexity)

    return np.array(features)


class FeatureExtractor:
    """
    Extracts features from EEG windows for classification/VQ.

    Combines spectral and temporal features into a single vector.
    """

    def __init__(self, sample_rate: int = 256):
        self.sample_rate = sample_rate
        self._feature_dim: Optional[int] = None

    def extract(self, window: np.ndarray) -> np.ndarray:
        """
        Extract all features from an EEG window.

        Args:
            window: Shape (4, n_samples) filtered EEG data

        Returns:
            Feature vector of shape (n_features,)
        """
        spectral = compute_spectral_features(window, self.sample_rate)
        temporal = compute_temporal_features(window)

        features = np.concatenate([spectral, temporal])

        if self._feature_dim is None:
            self._feature_dim = len(features)

        return features

    @property
    def feature_dim(self) -> int:
        """Get feature dimension (call extract once first)."""
        if self._feature_dim is None:
            # Compute on dummy data
            dummy = np.random.randn(4, 256)
            self.extract(dummy)
        return self._feature_dim


class AttentionEstimator:
    """
    Estimates attention/focus level from EEG.

    Uses frontal theta/beta ratio and other markers.
    Higher theta/beta = less focused (mind wandering)
    Lower theta/beta = more focused
    """

    def __init__(self, sample_rate: int = 256):
        self.sample_rate = sample_rate

        # Running statistics for normalization
        self._theta_beta_history: list[float] = []
        self._alpha_history: list[float] = []
        self._max_history = 100  # About 25 seconds at 4 Hz

    def estimate(self, window: np.ndarray) -> dict:
        """
        Estimate attention metrics from EEG window.

        Args:
            window: Shape (4, n_samples) filtered EEG data

        Returns:
            Dict with attention metrics (0-1 scale):
            - focus: Higher = more focused
            - relaxation: Higher = more relaxed (alpha)
            - engagement: Combined focus + low alpha
        """
        bp = compute_band_powers(window, self.sample_rate)

        # Frontal channels (AF7=1, AF8=2)
        frontal_theta = (bp.theta[1] + bp.theta[2]) / 2
        frontal_beta = (bp.beta[1] + bp.beta[2]) / 2
        theta_beta_ratio = frontal_theta / (frontal_beta + 1e-10)

        # Temporal alpha (TP9=0, TP10=3) - strongest alpha location
        temporal_alpha = (bp.alpha[0] + bp.alpha[3]) / 2

        # Update history
        self._theta_beta_history.append(theta_beta_ratio)
        self._alpha_history.append(temporal_alpha)

        if len(self._theta_beta_history) > self._max_history:
            self._theta_beta_history.pop(0)
            self._alpha_history.pop(0)

        # Normalize to 0-1 based on recent history
        def normalize(value, history):
            if len(history) < 2:
                return 0.5
            min_val = np.percentile(history, 10)
            max_val = np.percentile(history, 90)
            if max_val <= min_val:
                return 0.5
            return np.clip((value - min_val) / (max_val - min_val), 0, 1)

        # Focus: inverse of theta/beta ratio
        focus = 1 - normalize(theta_beta_ratio, self._theta_beta_history)

        # Relaxation: proportional to alpha
        relaxation = normalize(temporal_alpha, self._alpha_history)

        # Engagement: focus when not too relaxed
        engagement = focus * (1 - relaxation * 0.5)

        return {
            "focus": float(focus),
            "relaxation": float(relaxation),
            "engagement": float(engagement),
            "theta_beta_ratio": float(theta_beta_ratio),
            "alpha_power": float(temporal_alpha),
        }


# CLI for testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate test signal with alpha oscillation
    sample_rate = 256
    duration = 2  # seconds
    t = np.linspace(0, duration, sample_rate * duration)

    # Simulate 4 channels with alpha + noise
    alpha = 15 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    theta = 8 * np.sin(2 * np.pi * 6 * t)    # 6 Hz theta
    noise = np.random.randn(len(t)) * 3

    test_data = np.array([
        alpha + noise,           # TP9 - strong alpha
        theta + noise * 2,       # AF7 - more theta (frontal)
        theta + noise * 2,       # AF8 - more theta (frontal)
        alpha + noise,           # TP10 - strong alpha
    ])

    # Compute features
    extractor = FeatureExtractor(sample_rate)
    features = extractor.extract(test_data)
    print(f"Feature dimension: {len(features)}")

    # Compute band powers
    bp = compute_band_powers(test_data, sample_rate)
    print("\nBand powers (absolute):")
    print(f"  Delta: {bp.delta}")
    print(f"  Theta: {bp.theta}")
    print(f"  Alpha: {bp.alpha}")
    print(f"  Beta:  {bp.beta}")
    print(f"  Gamma: {bp.gamma}")

    rel_bp = bp.relative_powers()
    print("\nBand powers (relative):")
    print(f"  Alpha: {rel_bp.alpha}")

    # Attention estimation
    estimator = AttentionEstimator(sample_rate)
    attention = estimator.estimate(test_data)
    print("\nAttention metrics:")
    for k, v in attention.items():
        print(f"  {k}: {v:.3f}")
