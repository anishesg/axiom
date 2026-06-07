"""Axiom Brain State Engine.

Classifies continuous EEG into cognitive states the agent needs:
  - engagement (am I locked in on something?)
  - focus (deep work vs scattered?)
  - relaxation (winding down?)
  - cognitive_load (processing something hard?)
  - valence (positive vs negative affect)
  - jaw_clench (user's explicit "click" signal)
  - context_switch (about to switch tasks?)
  - error_response (just saw something wrong?)

Each state is a float in [0, 1] computed from BrainFlow features.
States update at 10Hz from the EEG ring buffer.
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from brainflow.data_filter import (
    DataFilter, FilterTypes, DetrendOperations,
    NoiseTypes, WindowOperations,
)

EEG_SR = 256
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 50.0)]
CH = ["TP9", "AF7", "AF8", "TP10"]


@dataclass
class BrainState:
    engagement: float = 0.0
    focus: float = 0.0
    relaxation: float = 0.0
    cognitive_load: float = 0.0
    valence: float = 0.5       # 0=negative, 0.5=neutral, 1=positive
    jaw_clench: bool = False
    double_clench: bool = False
    context_switch: float = 0.0
    error_response: float = 0.0

    # Raw features for the agent
    band_powers: dict = field(default_factory=dict)
    band_ratios: dict = field(default_factory=dict)
    asymmetry: float = 0.0
    signal_quality: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class BrainStateEngine:
    """Stateful engine that maintains baselines and detects transitions."""

    def __init__(self):
        # Running baselines (exponential moving average)
        self._alpha_baseline = None
        self._beta_baseline = None
        self._theta_baseline = None
        self._ema_alpha = 0.05  # slow adaptation

        # Jaw clench detection
        self._clench_cooldown_until = 0.0
        self._last_clench_time = 0.0
        self._clench_count = 0

        # Context switch detection (tracks rapid alpha/theta transitions)
        self._prev_alpha = 0.0
        self._prev_theta = 0.0
        self._transition_score = 0.0

        # Error detection (tracks sudden frontal asymmetry shifts)
        self._prev_asymmetry = 0.0
        self._asymmetry_history = []

        self._tick = 0

    def process(self, eeg_4ch: np.ndarray, timestamp: float = 0.0) -> BrainState:
        """Process a window of EEG data (4, N) and return the current brain state."""
        state = BrainState()
        n = eeg_4ch.shape[1] if eeg_4ch.ndim == 2 else 0
        if n < EEG_SR:
            return state

        self._tick += 1

        # Filter
        filtered = np.zeros_like(eeg_4ch, dtype=np.float64)
        for ch in range(4):
            filtered[ch] = self._filter(eeg_4ch[ch])

        # Band powers per channel
        ch_powers = [self._band_powers(filtered[ch]) for ch in range(4)]
        avg = {name: float(np.mean([cp[name] for cp in ch_powers])) for name, _, _ in BANDS}
        state.band_powers = avg

        alpha = avg["alpha"] + 1e-6
        theta = avg["theta"] + 1e-6
        beta = avg["beta"] + 1e-6
        gamma = avg["gamma"] + 1e-6
        delta = avg["delta"] + 1e-6

        # Update baselines
        if self._alpha_baseline is None:
            self._alpha_baseline = alpha
            self._beta_baseline = beta
            self._theta_baseline = theta
        else:
            ema = self._ema_alpha
            self._alpha_baseline = (1 - ema) * self._alpha_baseline + ema * alpha
            self._beta_baseline = (1 - ema) * self._beta_baseline + ema * beta
            self._theta_baseline = (1 - ema) * self._theta_baseline + ema * theta

        # Band ratios
        alpha_theta = alpha / theta
        beta_alpha = beta / alpha
        gamma_beta = gamma / beta
        theta_alpha = theta / alpha
        state.band_ratios = {
            "alpha_theta": round(alpha_theta, 4),
            "beta_alpha": round(beta_alpha, 4),
            "gamma_beta": round(gamma_beta, 4),
            "theta_alpha": round(theta_alpha, 4),
        }

        # --- ENGAGEMENT ---
        # High beta + gamma relative to baseline = engaged
        beta_rel = beta / (self._beta_baseline + 1e-6)
        engagement_raw = (beta_rel - 0.5) / 1.5  # normalize: 0.5x baseline=0, 2x=1
        state.engagement = float(np.clip(engagement_raw, 0, 1))

        # --- FOCUS ---
        # Beta/alpha ratio. High = focused, low = diffuse
        focus_raw = (beta_alpha - 1.0) / 4.0  # ratio of 1=0, ratio of 5=1
        state.focus = float(np.clip(focus_raw, 0, 1))

        # --- RELAXATION ---
        # Alpha dominance relative to baseline
        alpha_rel = alpha / (self._alpha_baseline + 1e-6)
        relax_raw = (alpha_rel - 0.5) / 1.5
        state.relaxation = float(np.clip(relax_raw, 0, 1))

        # --- COGNITIVE LOAD ---
        # Theta/alpha ratio + gamma. High theta with gamma = working hard
        load_raw = (theta_alpha - 0.5) / 2.0
        state.cognitive_load = float(np.clip(load_raw, 0, 1))

        # --- VALENCE (frontal asymmetry) ---
        # log(AF8_alpha) - log(AF7_alpha). Positive = approach/positive affect
        af7_alpha = ch_powers[1]["alpha"] + 1e-6
        af8_alpha = ch_powers[2]["alpha"] + 1e-6
        raw_asym = float(np.log(af8_alpha) - np.log(af7_alpha))
        state.asymmetry = raw_asym
        # Map to 0-1: negative asymmetry = negative affect, positive = positive
        state.valence = float(np.clip(0.5 + raw_asym * 0.3, 0, 1))

        # --- JAW CLENCH ---
        state.jaw_clench, state.double_clench = self._detect_clench(eeg_4ch, timestamp)

        # --- CONTEXT SWITCH ---
        # Rapid alpha rise + theta drop then theta spike = switching
        alpha_delta = alpha - self._prev_alpha
        theta_delta = theta - self._prev_theta
        if alpha_delta > 0 and theta_delta < 0:
            self._transition_score = min(1.0, self._transition_score + 0.3)
        elif theta_delta > 0 and alpha_delta < 0 and self._transition_score > 0.2:
            state.context_switch = self._transition_score
            self._transition_score = 0.0
        else:
            self._transition_score *= 0.85  # decay
        self._prev_alpha = alpha
        self._prev_theta = theta

        # --- ERROR RESPONSE ---
        # Sudden rightward frontal asymmetry shift = negative surprise
        asym_shift = raw_asym - self._prev_asymmetry
        if asym_shift < -0.3:  # sudden negative shift
            state.error_response = min(1.0, abs(asym_shift))
        self._prev_asymmetry = raw_asym

        # Signal quality
        state.signal_quality = self._signal_quality(eeg_4ch, filtered)

        return state

    def _filter(self, data: np.ndarray) -> np.ndarray:
        out = data.copy()
        if len(out) < 12:
            return out
        DataFilter.detrend(out, DetrendOperations.LINEAR.value)
        DataFilter.perform_bandpass(out, EEG_SR, 1.0, 50.0, 4,
                                    FilterTypes.BUTTERWORTH.value, 0.0)
        DataFilter.remove_environmental_noise(out, EEG_SR, NoiseTypes.SIXTY.value)
        return out

    def _band_powers(self, data: np.ndarray) -> dict:
        if len(data) < EEG_SR:
            return {name: 0.0 for name, _, _ in BANDS}
        nfft = DataFilter.get_nearest_power_of_two(EEG_SR)
        psd = DataFilter.get_psd_welch(data, nfft, nfft // 2, EEG_SR,
                                        WindowOperations.HANNING.value)
        return {name: float(DataFilter.get_band_power(psd, lo, hi))
                for name, lo, hi in BANDS}

    def _detect_clench(self, raw_eeg: np.ndarray, timestamp: float) -> tuple[bool, bool]:
        """Detect jaw clench from EMG artifacts in temporal channels (TP9, TP10).

        Jaw clenches produce high-amplitude, high-frequency bursts in ear channels.
        """
        if timestamp < self._clench_cooldown_until:
            return False, False

        window = min(25, raw_eeg.shape[1])
        tp9 = raw_eeg[0, -window:]
        tp10 = raw_eeg[3, -window:]

        combined = (np.abs(np.diff(tp9)) + np.abs(np.diff(tp10))) / 2.0
        hf_energy = float(np.mean(combined ** 2))

        threshold = 200000
        is_clench = hf_energy > threshold

        double = False
        if is_clench:
            self._clench_cooldown_until = timestamp + 0.5
            if timestamp - self._last_clench_time < 0.8:
                double = True
                self._clench_count = 0
            else:
                self._clench_count = 1
            self._last_clench_time = timestamp

        return is_clench, double

    def _signal_quality(self, raw: np.ndarray, filtered: np.ndarray) -> list[float]:
        quality = []
        for ch in range(4):
            std = float(np.std(raw[ch, -EEG_SR:]))
            if std < 1.0 or std > 500.0:
                quality.append(0.0)
            else:
                sig = float(np.mean(filtered[ch, -EEG_SR:] ** 2))
                noise = float(np.mean((raw[ch, -EEG_SR:] - filtered[ch, -EEG_SR:]) ** 2))
                snr = sig / (noise + 1e-10)
                quality.append(min(1.0, snr / 5.0))
        return quality
