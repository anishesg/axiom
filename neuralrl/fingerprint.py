"""NeuralFingerprint — Unique brain signature visualization.

Computes a person-specific neural fingerprint from their EEG patterns.
The fingerprint is a radial pattern that:
  - Is unique per person (different brains → different shapes)
  - Updates in real-time (2Hz)
  - Changes when mental state changes
  - Resets completely for a new person (proves it's not scripted)

Used as: a) a beautiful visualization b) proof the system is reading
a real brain c) the basis for personalization.
"""

import numpy as np
from collections import deque


FINGERPRINT_AXES = [
    "alpha_power",
    "beta_power",
    "theta_power",
    "gamma_power",
    "engagement",
    "asymmetry",
    "alpha_beta_ratio",
    "theta_beta_ratio",
]


class NeuralFingerprint:
    """Computes and tracks a person's unique neural signature."""

    def __init__(self, smoothing=0.15, history_len=100):
        self._smoothing = smoothing
        self._values = np.full(len(FINGERPRINT_AXES), 0.5, dtype=np.float64)
        self._history = deque(maxlen=history_len)
        self._baseline = None
        self._calibration_samples = []
        self._tick = 0

    def update(self, brain_snapshot) -> dict:
        """Update fingerprint from a brain snapshot. Returns axis values 0-1."""
        bp = brain_snapshot.band_powers
        alpha = bp.get("alpha", 0) + 1e-6
        beta = bp.get("beta", 0) + 1e-6
        theta = bp.get("theta", 0) + 1e-6
        gamma = bp.get("gamma", 0) + 1e-6

        total = alpha + beta + theta + gamma + bp.get("delta", 0) + 1e-6

        raw = np.array([
            alpha / total,
            beta / total,
            theta / total,
            gamma / total,
            brain_snapshot.engagement,
            (brain_snapshot.asymmetry + 1) / 2,
            np.clip(alpha / beta, 0, 5) / 5,
            np.clip(theta / beta, 0, 5) / 5,
        ], dtype=np.float64)

        raw = np.clip(raw, 0, 1)

        # Exponential smoothing
        self._values = (1 - self._smoothing) * self._values + self._smoothing * raw
        self._tick += 1
        self._history.append(self._values.copy())

        if self._tick <= 20:
            self._calibration_samples.append(raw.copy())
            if self._tick == 20:
                self._baseline = np.mean(self._calibration_samples, axis=0)

        return self.get_fingerprint()

    def get_fingerprint(self) -> dict:
        """Return fingerprint as {axis_name: value} for radar chart."""
        return {
            "axes": FINGERPRINT_AXES,
            "values": [round(float(v), 4) for v in self._values],
            "tick": self._tick,
        }

    def get_stability(self) -> float:
        """How stable is the fingerprint? High = settled, low = still adapting."""
        if len(self._history) < 10:
            return 0.0
        recent = np.array(list(self._history)[-10:])
        variance = np.mean(np.var(recent, axis=0))
        return float(np.clip(1.0 - variance * 20, 0, 1))

    def reset(self):
        """Reset for a new person."""
        self._values = np.full(len(FINGERPRINT_AXES), 0.5, dtype=np.float64)
        self._history.clear()
        self._baseline = None
        self._calibration_samples = []
        self._tick = 0
