"""Feature extraction pipeline for RL state vectors.

Takes raw EEG/PPG/IMU data from BrainFlow and produces a 29-dim
state vector suitable for reinforcement learning agents.
"""

import numpy as np
from brainflow.data_filter import (
    DataFilter, FilterTypes, DetrendOperations,
    NoiseTypes, WindowOperations,
)

EEG_SR = 256
BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 50.0),
]
CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]


def preprocess_channel(data: np.ndarray, sr: int = EEG_SR) -> np.ndarray:
    out = data.copy()
    if len(out) < 12:
        return out
    DataFilter.detrend(out, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(out, sr, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
    DataFilter.remove_environmental_noise(out, sr, NoiseTypes.SIXTY.value)
    return out


def band_powers(data: np.ndarray, sr: int = EEG_SR) -> dict[str, float]:
    if len(data) < sr:
        return {name: 0.0 for name, _, _ in BANDS}
    nfft = DataFilter.get_nearest_power_of_two(sr)
    psd = DataFilter.get_psd_welch(data, nfft, nfft // 2, sr, WindowOperations.HANNING.value)
    return {name: float(DataFilter.get_band_power(psd, lo, hi)) for name, lo, hi in BANDS}


def extract_state(eeg_4ch: np.ndarray, prev_bands: dict | None = None) -> dict:
    """Extract full state vector from 4-channel EEG array (4, N).

    Returns dict with all features + the raw band powers for next call's delta.
    """
    n_samples = eeg_4ch.shape[1] if eeg_4ch.ndim == 2 else 0
    if n_samples < EEG_SR:
        return {"state": np.zeros(29).tolist(), "bands": None}

    # Filter all channels
    filtered = np.zeros_like(eeg_4ch, dtype=np.float64)
    for ch in range(4):
        filtered[ch] = preprocess_channel(eeg_4ch[ch])

    # Band powers per channel
    ch_bands = []
    for ch in range(4):
        ch_bands.append(band_powers(filtered[ch]))

    # Average band powers across channels
    avg_bands = {}
    for name, _, _ in BANDS:
        avg_bands[name] = float(np.mean([cb[name] for cb in ch_bands]))

    # Band ratios
    alpha = avg_bands["alpha"] + 1e-6
    theta = avg_bands["theta"] + 1e-6
    beta = avg_bands["beta"] + 1e-6
    gamma = avg_bands["gamma"] + 1e-6

    alpha_theta = alpha / theta
    beta_alpha = beta / alpha
    gamma_beta = gamma / beta

    # Frontal asymmetry (AF7 vs AF8 alpha power)
    af7_alpha = ch_bands[1]["alpha"]
    af8_alpha = ch_bands[2]["alpha"]
    asymmetry = float(np.log(af8_alpha + 1e-6) - np.log(af7_alpha + 1e-6))

    # Cross-channel coherence (simplified: correlation of filtered signals)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    coherence = []
    for i, j in pairs:
        c = float(np.abs(np.corrcoef(filtered[i, -EEG_SR:], filtered[j, -EEG_SR:])[0, 1]))
        coherence.append(c if np.isfinite(c) else 0.0)

    # Signal quality per channel
    quality = []
    for ch in range(4):
        raw_std = float(np.std(eeg_4ch[ch, -EEG_SR:]))
        if raw_std < 1.0 or raw_std > 500.0:
            quality.append(0.0)
        else:
            filt_power = float(np.mean(filtered[ch, -EEG_SR:] ** 2))
            noise_power = float(np.mean((eeg_4ch[ch, -EEG_SR:] - filtered[ch, -EEG_SR:]) ** 2))
            snr = filt_power / (noise_power + 1e-10)
            quality.append(min(1.0, snr / 5.0))

    # Temporal delta (change from previous extraction)
    temporal = [0.0] * 5
    if prev_bands is not None:
        for idx, (name, _, _) in enumerate(BANDS):
            temporal[idx] = avg_bands[name] - prev_bands.get(name, 0.0)

    # Assemble state vector (29 dims)
    state = (
        [avg_bands[name] for name, _, _ in BANDS]  # 5: band powers
        + [alpha_theta, beta_alpha, gamma_beta]      # 3: ratios
        + [asymmetry]                                # 1: frontal asymmetry
        + coherence                                  # 6: pairwise coherence
        + [0.0]                                      # 1: heart rate (placeholder)
        + [0.0]                                      # 1: HRV (placeholder)
        + [0.0, 0.0, 0.0]                           # 3: motion (placeholder)
        + quality                                    # 4: signal quality
        + temporal                                   # 5: temporal deltas
    )

    return {
        "state": [round(v, 4) for v in state],
        "bands": avg_bands,
        "band_ratios": {
            "alpha_theta": round(alpha_theta, 4),
            "beta_alpha": round(beta_alpha, 4),
            "gamma_beta": round(gamma_beta, 4),
        },
        "asymmetry": round(asymmetry, 4),
        "coherence": {f"{CHANNEL_NAMES[i]}-{CHANNEL_NAMES[j]}": round(c, 4) for (i, j), c in zip(pairs, coherence)},
        "quality": {CHANNEL_NAMES[i]: round(q, 4) for i, q in enumerate(quality)},
    }
